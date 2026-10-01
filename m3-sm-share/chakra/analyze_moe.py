#!/usr/bin/env python3
"""M3 (trace part, extension): per-MoE-layer breakdown and NCCL "waiting vs. moving" analysis for the
Mixtral-8x7B Kineto device traces (nemo_raw/device_<r>.json, any subset of ranks 0..7; TP2 x EP4, 1 node).

What it does (per rank, within the GPU-side ProfilerStep#0 window):
  A. Attributes every GPU kernel to the CPU op stack that launched it (Kineto correlation id ->
     cuda_runtime launch -> enclosing cpu_op / user_annotation ranges on the same thread).
  B. Segments the step into transformer-layer instances (micro-batch x layer x {fwd,bwd}) with
     TransformerEngine RMSNorm kernels as anchors:
       fwd: layer starts at the rmsnorm_fwd launched by _LayerNormLinear (input norm fused into QKV);
            it ends at the next such kernel, or at the final-norm rmsnorm_fwd (an _RMSNorm rmsnorm_fwd
            that directly follows another _RMSNorm one, i.e. pre-MLP norm of layer 32 -> final norm).
       bwd: layer ends at the end of the rmsnorm_bwd_finalize launched by _LayerNormLinearBackward; it
            starts at the end of the previous such kernel, or of the final-norm backward (an
            _RMSNormBackward finalize directly followed by another _RMSNormBackward finalize).
     A kernel belongs to the layer instance in which it starts.
  C. Classifies kernels inside a layer instance:
       EP all-to-all (SendRecv on the EP process group): a2a_1 / a2a_2 by call order in the layer.
         fwd: a2a_1 = dispatch (tokens -> experts), a2a_2 = combine (expert outputs -> token owners).
         bwd: a2a_1 = backward of combine (output grads -> experts),
              a2a_2 = backward of dispatch (input grads -> token owners).
       TP collectives (AllGather/ReduceScatter on the 2-rank TP group): "TP-MoE" if they start between
         a2a_1 and a2a_2 (sequence-parallel gather/scatter of the dispatched tokens), else "TP-attn".
       small ALL_REDUCE (router aux-loss etc.), DP collectives (4-rank data-parallel group, non-a2a).
       expert GEMM: GEMM kernels launched under Megatron's LinearWithGradAccumulationAndAsyncCommunication
         (the SequentialMLP expert linears; attention uses TE _LayerNormLinear/_Linear), inside a layer.
       MoE-other: other non-NCCL work between a2a_1 and a2a_2 (SwiGLU, cat/split, memsets).
       dense/other: everything else (attention, router, permute/unpermute, norms, elementwise).
       idle: layer span minus the union of all GPU activity in it.
  D. For every NCCL kernel in the step: bytes from Kineto msg sizes, cross-checked against the Chakra
     ET comm_size of the same rank (matched by per-process-group call order); bus bytes (NCCL
     convention), achieved bus bandwidth, per held SM, and the share of kernel time not explained by
     data movement at the NVLink peak.

Usage: python3 analyze_moe.py <nvlink_GBps_per_dir> <et_dir> device_*.json
Writes moe_summary_Mixtral-8x7B.json next to this script (inputs for project_offsm.py).
The Kineto JSONs are stream-parsed (jstream.py); only the event categories used below are kept.
classify() (steps A-C, per-kernel labels) is reused by analyze_skew.py.
"""
import sys, os, json, re, collections, statistics
from multiprocessing import Pool
import chakra_et, jstream

CATS = {"kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "user_annotation", "cuda_runtime", "cuda_driver",
        "gpu_user_annotation"}

ES = {"BFloat16": 2, "Half": 2, "Float": 4, "Long": 8, "Int": 4, "Byte": 1, "Double": 8}
LAT_THR = 64 * 1024  # per-peer bytes below which a message is treated as latency-bound


def union(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]: out[-1][1] = e
        else:
            out.append([s, e])
    return out


def ulen(iv):
    return sum(e - s for s, e in union(iv))


def coll(e):
    n = e["name"]
    if "SendRecv" in n: return "ALL_TO_ALL"
    for k, v in (("AllGather", "ALL_GATHER"), ("ReduceScatter", "REDUCE_SCATTER"),
                 ("AllReduce", "ALL_REDUCE"), ("Broadcast", "BROADCAST")):
        if k in n: return v
    return "OTHER"


def bus_and_peer(ct, sin, sout, n):
    """NCCL bus-bytes convention and per-peer message size (bytes)."""
    if n <= 1: return 0.0, sin
    f = (n - 1) / n
    if ct == "ALL_GATHER": return sout * f, sin
    if ct == "REDUCE_SCATTER": return sin * f, sin / n
    if ct == "ALL_REDUCE": return 2 * sin * f, sin / n
    if ct == "ALL_TO_ALL": return sin * f, sin / n      # equal splits (In split size == [] in trace)
    return sin, sin                                      # broadcast


def build_stacks(ev):
    """correlation id -> ' > '-joined names of cpu ops enclosing the launching runtime call."""
    per_tid = collections.defaultdict(list)
    for e in ev:
        c = e.get("cat")
        if c in ("cpu_op", "user_annotation"):
            per_tid[e["tid"]].append((e["ts"], -e["dur"], 0, e["ts"] + e["dur"], e["name"]))
        elif c in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            per_tid[e["tid"]].append((e["ts"], -e["dur"], 1, e["ts"] + e["dur"], e["args"]["correlation"]))
    out = {}
    for tid, L in per_tid.items():
        L.sort()
        st = []
        for ts, nd, kind, end, x in L:
            while st and st[-1][0] < ts: st.pop()
            if kind == 0: st.append((end, x))
            else: out[x] = " > ".join(n for _, n in st)
    return out


def et_nccl_sizes(path):
    """per pg_name: list of comm_size in ET node order (GPU NCCL nodes only)."""
    _, nodes = chakra_et.read_et(path)
    d = collections.defaultdict(list)
    for n in nodes:
        a = n["attr"]
        if not a.get("is_cpu_op", True) and "comm_type" in a:
            d[str(a.get("pg_name"))].append((a.get("comm_size", 0), n["dur"]))
    return d


def med(x):
    return statistics.median(x) if x else float("nan")


def classify(path):
    """Steps A-C for one rank. Every NCCL/compute kernel dict in gk gets k["cls"] (class name, see CLS) and
    k["lay"] = (pass, micro-batch, layer index) if it starts inside a layer instance with exactly 2 EP a2a."""
    d = jstream.load_kineto(path, CATS)
    rank = d["distributedInfo"]["rank"]; nsm = d["deviceProperties"][0]["numSms"]
    pgcfg = {p["pg_name"]: p["ranks"] for p in d["distributedInfo"]["pg_config"]}
    ev = d["traceEvents"]
    stacks = build_stacks(ev)
    step = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")][0]
    T0, T1 = step["ts"], step["ts"] + step["dur"]
    gk = []   # all GPU activities in step: dict(s, e, name, cat, grid, args, stack)
    allnccl = []
    for e in ev:
        c = e.get("cat")
        if c not in ("kernel", "gpu_memcpy", "gpu_memset"): continue
        if c == "kernel" and e["name"].startswith("nccl"): allnccl.append(e)
        if e["ts"] < T0 or e["ts"] >= T1: continue
        a = e.get("args", {})
        g = a.get("grid", [0, 0, 0]) if c == "kernel" else [0, 0, 0]
        gk.append(dict(s=e["ts"], e=min(e["ts"] + e["dur"], T1), name=e["name"], cat=c,
                       ctas=g[0] * g[1] * g[2], a=a, st=stacks.get(a.get("correlation"), "")))
    gk.sort(key=lambda k: k["s"])
    # --- process-group roles
    ep_pg = {k["a"]["Process Group Name"] for k in gk if k["name"].startswith("ncclDevKernel_SendRecv")}
    assert len(ep_pg) == 1, ep_pg
    ep_pg = ep_pg.pop()

    def role(k):
        pg = k["a"].get("Process Group Name"); r = pgcfg.get(pg, [])
        ct = coll(k)
        if pg == ep_pg and ct == "ALL_TO_ALL": return "EP"
        if ct == "ALL_REDUCE" or ct == "BROADCAST": return "AR"
        if len(r) == 2: return "TP"
        if len(r) == 4: return "DP"
        return "OTHER"
    for k in gk:
        k["nccl"] = k["cat"] == "kernel" and k["name"].startswith("nccl")
        k["role"] = role(k) if k["nccl"] else None
    # --- anchors
    fa = [k for k in gk if "rmsnorm_fwd" in k["name"]]
    ba = [k for k in gk if "rmsnorm_bwd_finalize" in k["name"]]
    fkind = ["LNL" if "_LayerNormLinear" in k["st"] else "RMS" for k in fa]
    bkind = ["LNL" if "_LayerNormLinearBackward" in k["st"] else "RMS" for k in ba]
    spans = []  # (pass, s, e)
    for i, k in enumerate(fa):
        if fkind[i] != "LNL": continue
        j = i + 1
        while j < len(fa):
            if fkind[j] == "LNL" or (fkind[j] == "RMS" and j > 0 and fkind[j - 1] == "RMS"): break
            j += 1
        if j < len(fa): spans.append(("fwd", k["s"], fa[j]["s"]))
    for i, k in enumerate(ba):
        if bkind[i] != "LNL": continue
        j = i - 1
        while j >= 0:
            if bkind[j] == "LNL" or (bkind[j] == "RMS" and j + 1 < len(ba) and bkind[j + 1] == "RMS"): break
            j -= 1
        if j >= 0: spans.append(("bwd", ba[j]["e"], k["e"]))
    spans.sort(key=lambda x: x[1])
    # micro-batch / layer index: count fwd layer starts; a new micro-batch begins after a bwd span
    lay = []; mb = -1; nf = nb = 0; prev = None
    for p, s, e in spans:
        if p == "fwd" and prev != "fwd": mb += 1; nf = nb = 0
        if p == "fwd": nf += 1; li = nf
        else: li = nf - nb; nb += 1
        lay.append((p, mb, li, s, e)); prev = p
    # --- per-layer classification
    CLS = ["a2a_1", "a2a_2", "TP-MoE", "TP-attn", "AR", "DP", "expert_GEMM", "MoE_other", "dense_other"]
    rows = []
    idx = 0
    for p, mbi, li, s, e in lay:
        while idx < len(gk) and gk[idx]["s"] < s: idx += 1
        j = idx; ks = []
        while j < len(gk) and gk[j]["s"] < e: ks.append(gk[j]); j += 1
        ep = [k for k in ks if k["nccl"] and k["role"] == "EP"]
        t = dict.fromkeys(CLS, 0.0)
        if len(ep) != 2:
            rows.append(dict(p=p, mb=mbi, l=li, span=e - s, bad=len(ep))); continue
        m0, m1 = ep[0]["s"], ep[1]["s"]
        for k in ks:
            dur = min(k["e"], e) - k["s"]
            if k["nccl"]:
                r = k["role"]
                if r == "EP": c = "a2a_1" if k is ep[0] else "a2a_2"
                elif r == "TP": c = "TP-MoE" if m0 <= k["s"] < m1 else "TP-attn"
                else: c = r if r in t else "AR"
            elif k["cat"] == "kernel" and "gemm" in k["name"].lower() and "LinearWithGradAccumulationAndAsyncCommun" in k["st"]:
                c = "expert_GEMM"
            elif m0 <= k["s"] < m1:
                c = "MoE_other"
            else:
                c = "dense_other"
            t[c] += dur
            k["cls"] = c; k["lay"] = (p, mbi, li)
        busy = ulen([(k["s"], min(k["e"], e)) for k in ks])
        comm_u = ulen([(k["s"], min(k["e"], e)) for k in ks if k["nccl"]])
        comp_u = ulen([(k["s"], min(k["e"], e)) for k in ks if not k["nccl"]])
        a2a_bytes = [int(k["a"]["In msg nelems"]) * ES[k["a"]["dtype"]] for k in ep]
        rows.append(dict(p=p, mb=mbi, l=li, span=e - s, bad=0, idle=e - s - busy, comm_u=comm_u,
                         comp_u=comp_u, a2a_bytes=a2a_bytes, **t))
    return dict(rank=rank, nsm=nsm, pgcfg=pgcfg, ev=ev, d=d, gk=gk, T0=T0, T1=T1, allnccl=allnccl, ep_pg=ep_pg,
                lay=lay, rows=rows)


def analyze(args):
    path, peak, et_dir = args
    c = classify(path)
    rank, nsm, gk, rows, allnccl, ep_pg = c["rank"], c["nsm"], c["gk"], c["rows"], c["allnccl"], c["ep_pg"]
    T0, T1 = c["T0"], c["T1"]
    del c
    # --- NCCL kernel bandwidth (whole step)
    nk = []
    for k in gk:
        if not k["nccl"]: continue
        a = k["a"]; es = ES.get(a.get("dtype"), 1)
        sin = int(a.get("In msg nelems", 0)) * es; sout = int(a.get("Out msg nelems", 0)) * es
        n = int(a.get("Group size", 1)); ct = coll(k)
        bus, peer = bus_and_peer(ct, sin, sout, n)
        dur = k["e"] - k["s"]
        nk.append(dict(ct=ct, role=k["role"], pg=a.get("Process Group Name"), n=n, sin=sin, bus=bus,
                       peer=peer, dur=dur, ctas=k["ctas"], tpk=bus / (peak * 1e3) if bus else 0.0))
    # --- cross-check sizes vs. ET comm_size (all NCCL kernels in the Kineto file, per pg, in order)
    etp = os.path.join(et_dir, f"chakra_trace.{rank}.et")
    xc = {}
    if os.path.exists(etp):
        et = et_nccl_sizes(etp)
        kin = collections.defaultdict(list)
        for e in sorted(allnccl, key=lambda e: e["ts"]):
            a = e["args"]
            kin[a.get("Process Group Name")].append((int(a.get("In msg nelems", 0)) * ES.get(a.get("dtype"), 1), e["dur"]))
        for pg in sorted(set(et) | set(kin), key=lambda x: int(x) if x.isdigit() else 1e9):
            A, B = et.get(pg, []), kin.get(pg, [])
            m = min(len(A), len(B))
            same = sum(1 for i in range(m) if A[i][0] == B[i][0])
            dd = [abs(A[i][1] - B[i][1]) for i in range(m)]
            xc[pg] = dict(et=len(A), kineto=len(B), size_equal=same, max_abs_dur_diff_us=max(dd) if dd else None)
    # whole-step overlap and NCCL SM-time (same definitions as analyze_kineto.py)
    uc = union([(k["s"], k["e"]) for k in gk if k["nccl"]]); uk = union([(k["s"], k["e"]) for k in gk if not k["nccl"]])
    i = j = 0; ov = 0.0
    while i < len(uc) and j < len(uk):
        a0 = max(uc[i][0], uk[j][0]); b0 = min(uc[i][1], uk[j][1])
        if b0 > a0: ov += b0 - a0
        if uc[i][1] < uk[j][1]: i += 1
        else: j += 1
    sm_nccl = sum(min(k["ctas"], nsm) * (k["e"] - k["s"]) for k in gk if k["nccl"])
    whole = dict(comm_wall_ms=sum(e - s for s, e in uc) / 1e3, overlap_ms=ov / 1e3, nccl_sm_ms=sm_nccl / 1e3,
                 nccl_sum_ms=sum(k["e"] - k["s"] for k in gk if k["nccl"]) / 1e3)
    # "moving" time of each NCCL kernel: filled in report() from the pooled per-size minimum
    return dict(rank=rank, nsm=nsm, step_us=T1 - T0, ep_pg=ep_pg, layers=rows, nccl=nk, xcheck=xc, whole=whole,
                nlayers_fwd=sum(1 for r in rows if r["p"] == "fwd"), nlayers_bwd=sum(1 for r in rows if r["p"] == "bwd"))


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def report(R, peak):
    CLS = ["a2a_1", "a2a_2", "TP-MoE", "TP-attn", "AR", "DP", "expert_GEMM", "MoE_other", "dense_other", "idle"]
    print(f"# Mixtral-8x7B Kineto per-layer MoE breakdown and NCCL bandwidth; NVLink peak = {peak} GB/s/direction")
    summ = {}
    for r in R:
        rk = r["rank"]; L = [x for x in r["layers"] if not x["bad"]]
        bad = [x for x in r["layers"] if x["bad"]]
        print(f"\n## rank {rk}: step {r['step_us']/1e3:.1f} ms; EP pg {r['ep_pg']}; layer instances fwd {r['nlayers_fwd']} bwd {r['nlayers_bwd']}; "
              f"instances without exactly 2 EP a2a: {len(bad)}")
        print("size check: every fwd/bwd layer's two a2a have In bytes =",
              sorted({b for x in L for b in x['a2a_bytes']}))
        summ[rk] = {}
        for p in ("fwd", "bwd"):
            X = [x for x in L if x["p"] == p]
            for tag, Y in (("all micro-batches", X), ("micro-batches 1..7 (excl. first)", [x for x in X if x["mb"] > 0])):
                print(f"\n### rank {rk} {p} layers, {tag}: n={len(Y)}; median layer span {med([x['span'] for x in Y])/1e3:.3f} ms")
                print("class | median us | p10 us | p90 us | median % of layer span | sum over instances ms")
                for c in CLS:
                    v = sorted(x[c] for x in Y); f = [pct(x[c], x["span"]) for x in Y]
                    print(f"{c} | {med(v):.1f} | {v[len(v)//10]:.1f} | {v[(9*len(v))//10]:.1f} | {med(f):.1f} | {sum(v)/1e3:.1f}")
                comm = [pct(x["comm_u"], x["span"]) for x in Y]
                comp = [pct(x["comp_u"], x["span"]) for x in Y]
                print(f"NCCL wall (union) median % of span {med(comm):.1f}; compute wall median % of span {med(comp):.1f}")
                if tag.startswith("all"):
                    summ[rk][p] = {c: dict(med_us=med([x[c] for x in Y]), med_pct=med([pct(x[c], x["span"]) for x in Y]),
                                           sum_ms=sum(x[c] for x in Y) / 1e3) for c in CLS}
                    summ[rk][p]["span"] = dict(med_us=med([x["span"] for x in Y]), sum_ms=sum(x["span"] for x in Y) / 1e3)
        # per layer index medians (over micro-batches), fwd+bwd
        print(f"\n### rank {rk}: per layer index (median over 8 micro-batches), ms: fwd span | fwd a2a_1+a2a_2 | fwd expert GEMM || bwd span | bwd a2a | bwd expert GEMM")
        for li in range(1, 33):
            f = [x for x in L if x["p"] == "fwd" and x["l"] == li]; b = [x for x in L if x["p"] == "bwd" and x["l"] == li]
            print(f"L{li:02d} | {med([x['span'] for x in f])/1e3:.2f} | {med([x['a2a_1']+x['a2a_2'] for x in f])/1e3:.2f} | "
                  f"{med([x['expert_GEMM'] for x in f])/1e3:.2f} || {med([x['span'] for x in b])/1e3:.2f} | "
                  f"{med([x['a2a_1']+x['a2a_2'] for x in b])/1e3:.2f} | {med([x['expert_GEMM'] for x in b])/1e3:.2f}")
        # whole-step expert GEMM / MoE comm totals
        tot = {c: sum(x[c] for x in L) / 1e3 for c in CLS}
        tot["span"] = sum(x["span"] for x in L) / 1e3
        summ[rk]["step_totals_ms"] = tot; summ[rk]["step_ms"] = r["step_us"] / 1e3
        print(f"\nrank {rk} totals over all layer instances (ms): " + ", ".join(f"{c} {v:.1f}" for c, v in tot.items()))
        print(f"layer instances cover {pct(tot['span'], r['step_us']/1e3):.1f}% of the step")
        # xcheck
        print(f"\n### rank {rk}: Kineto In-bytes vs. Chakra ET comm_size, per pg matched by call order")
        for pg, v in r["xcheck"].items(): print(f"pg {pg}: {v}")
    # ---- NCCL bandwidth
    print("\n## NCCL kernels: achieved bus bandwidth, per held SM, and time not explained by data movement at peak")
    print("bus bytes: AG (n-1)/n*out, RS (n-1)/n*in, AR 2(n-1)/n*in, a2a (n-1)/n*in; t_peak = bus/peak; "
          "unexplained = 1 - sum(t_peak)/sum(dur)  (upper bound on wait/sync/latency share)")
    print("rank | type | role | n | kernels | CTAs | median per-peer KiB | sum dur ms | median busBW GB/s | p95 busBW | max busBW | median GB/s per held SM | unexplained % (time-weighted)")
    agg = collections.defaultdict(list)
    for r in R:
        g = collections.defaultdict(list)
        for k in r["nccl"]: g[(k["ct"], k["role"], k["n"])].append(k); agg[(k["ct"], k["role"], k["n"])].append(k)
        for key, K in sorted(g.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
            bw = sorted(k["bus"] / k["dur"] / 1e3 for k in K if k["dur"] > 0)
            ps = [k["bus"] / k["dur"] / 1e3 / max(k["ctas"], 1) for k in K if k["dur"] > 0]
            cs = collections.Counter(k["ctas"] for k in K).most_common(2)
            un = 1 - sum(k["tpk"] for k in K) / sum(k["dur"] for k in K)
            print(f"{r['rank']} | {key[0]} | {key[1]} | {key[2]} | {len(K)} | {cs} | {med([k['peer'] for k in K])/1024:.1f} | "
                  f"{sum(k['dur'] for k in K)/1e3:.1f} | {med(bw):.1f} | {bw[int(0.95*(len(bw)-1))]:.1f} | {bw[-1]:.1f} | {med(ps):.2f} | {100*un:.1f}")
    print(f"\n### all {len(R)} ranks pooled: excess over the fastest same-size kernel (empirical wait estimate)")
    print(f"for each (type, role, group size, in-bytes) the minimum duration over all {len(R)} ranks is taken as the 'moving' time;")
    print("excess = sum(dur - min_dur) / sum(dur). This is a tighter, empirical estimate of time spent waiting/skewed.")
    print("type | role | n | in MiB | kernels | min dur us | busBW at min GB/s | median dur us | sum dur ms | excess over min %")
    ex = collections.defaultdict(list)
    for K in agg.values():
        for k in K: ex[(k["ct"], k["role"], k["n"], k["sin"])].append(k)
    extot = [0.0, 0.0]
    for key, K in sorted(ex.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
        mn = min(k["dur"] for k in K); sd = sum(k["dur"] for k in K); xs = sum(k["dur"] - mn for k in K)
        extot[0] += xs; extot[1] += sd
        if sd < 5e3: continue
        print(f"{key[0]} | {key[1]} | {key[2]} | {key[3]/2**20:.2f} | {len(K)} | {mn:.1f} | {K[0]['bus']/mn/1e3:.1f} | "
              f"{med([k['dur'] for k in K]):.1f} | {sd/1e3:.1f} | {100*xs/sd:.1f}")
    print(f"(groups with < 5 ms total omitted from the table but included here) all NCCL: excess over min = {100*extot[0]/extot[1]:.1f}% of {extot[1]/1e3:.1f} ms")
    mins = {key: min(k["dur"] for k in K) for key, K in ex.items()}
    for r in R:
        mv = collections.defaultdict(float)
        for k in r["nccl"]:
            mv[k["role"]] += mins[(k["ct"], k["role"], k["n"], k["sin"])] / 1e3
        r["whole"]["moving_ms_by_role"] = dict(mv)
        summ[r["rank"]]["whole"] = r["whole"]
        print(f"rank {r['rank']}: NCCL sum {r['whole']['nccl_sum_ms']:.1f} ms, of which 'moving' (sum of per-size minima) by role: "
              + ", ".join(f"{a} {b:.1f}" for a, b in sorted(mv.items())) + f"; comm wall {r['whole']['comm_wall_ms']:.1f} ms; "
              f"comm/compute overlap {r['whole']['overlap_ms']:.1f} ms; NCCL SM-time {r['whole']['nccl_sm_ms']:.0f} SM*ms")
    print(f"\n### all {len(R)} ranks pooled, by type and per-peer size regime")
    print("type | role | n | regime | kernels | sum dur ms | median busBW GB/s | median GB/s per held SM | unexplained %")
    bwsum = {}
    for key, K in sorted(agg.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
        for reg, Kr in (("all", K), ("<64KiB/peer", [k for k in K if k["peer"] < LAT_THR]), (">=64KiB/peer", [k for k in K if k["peer"] >= LAT_THR])):
            if not Kr: continue
            un = 1 - sum(k["tpk"] for k in Kr) / sum(k["dur"] for k in Kr)
            bw = [k["bus"] / k["dur"] / 1e3 for k in Kr if k["dur"] > 0]
            ps = [k["bus"] / k["dur"] / 1e3 / max(k["ctas"], 1) for k in Kr if k["dur"] > 0]
            print(f"{key[0]} | {key[1]} | {key[2]} | {reg} | {len(Kr)} | {sum(k['dur'] for k in Kr)/1e3:.1f} | {med(bw):.1f} | {med(ps):.2f} | {100*un:.1f}")
            if reg == "all":
                bwsum[f"{key[0]}|{key[1]}|{key[2]}"] = dict(n=len(Kr), sum_ms=sum(k["dur"] for k in Kr) / 1e3, med_bw=med(bw),
                                                           med_per_sm=med(ps), unexplained_pct=100 * un,
                                                           ctas=collections.Counter(k["ctas"] for k in Kr).most_common(1)[0][0])
    Kall = [k for K in agg.values() for k in K]
    un = 1 - sum(k["tpk"] for k in Kall) / sum(k["dur"] for k in Kall)
    print(f"\nall NCCL kernels, {len(R)} ranks pooled: {len(Kall)} kernels, {sum(k['dur'] for k in Kall)/1e3:.1f} ms; "
          f"time explained by data movement at {peak} GB/s = {100*(1-un):.1f}%, unexplained = {100*un:.1f}%")
    print(f"per-SM rate needed to reach the NVLink peak: 32 CTAs -> {peak/32:.1f} GB/s/SM, 24 CTAs -> {peak/24:.1f} GB/s/SM")
    if len(R) > 4:   # cross-rank summary of the per-rank medians above (E2 table); printed only for > 4 ranks
        print(f"\n## across ranks {[r['rank'] for r in R]}: median of the per-rank medians [min - max over ranks], "
              "us (% of layer span); and the rank with the min / max median")
        for p in ("fwd", "bwd"):
            print(f"\n### {p}")
            print("class | median of per-rank median us | min us | max us | median of per-rank median % | min % | max % | rank at min | rank at max")
            for c in ["span"] + CLS:
                v = {rk: summ[rk][p][c]["med_us"] for rk in summ}
                f = {rk: summ[rk][p][c].get("med_pct", 100.0) for rk in summ}
                lo, hi = min(v, key=v.get), max(v, key=v.get)
                print(f"{c} | {med(list(v.values())):.1f} | {v[lo]:.1f} | {v[hi]:.1f} | {med(list(f.values())):.1f} | "
                      f"{min(f.values()):.1f} | {max(f.values()):.1f} | {lo} | {hi}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_summary_Mixtral-8x7B.json")
    json.dump(dict(peak_GBps=peak, per_rank=summ, nccl_bw=bwsum, unexplained_all_pct=100 * un, excess_over_min_pct=100 * extot[0] / extot[1]), open(out, "w"), indent=1)
    print(f"\n(summary written to {out})")


if __name__ == "__main__":
    peak = float(sys.argv[1]); et_dir = sys.argv[2]; files = sys.argv[3:]
    with Pool(len(files)) as p:
        R = sorted(p.map(analyze, [(f, peak, et_dir) for f in files]), key=lambda r: r["rank"])
    report(R, peak)
