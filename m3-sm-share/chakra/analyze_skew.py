#!/usr/bin/env python3
"""M3 (trace part, E8): arrival skew vs. the cost of communication itself, Mixtral-8x7B, all 8 ranks.

Inputs: nemo_raw/device_<r>.json (Kineto) and nemo_raw/host_<r>.json (PyTorch host ET) for every rank.
Both are stream-parsed (jstream.py). Per-kernel classes (EP a2a_1/a2a_2, TP-MoE, TP-attn, AR, DP) come from
analyze_moe.classify(), i.e. exactly the E1 layer segmentation and class rules.

1. Clock alignment. Kineto ts are host-clock based (us since baseTimeNanoseconds). Checks:
   baseTimeNanoseconds identical on all ranks; ProfilerStep#0 (CPU and GPU) boundaries; per rank,
   min(kernel start - launching runtime-call start) >= 0 (CPU->GPU causality); across ranks, causality of
   every matched collective: for AllGather/ReduceScatter/AllReduce/all-to-all no member can finish before
   every member has started (each member needs data that depends on every other member), so
   end_i >= start_j for all i, j. With per-rank clock offsets x_r (measured = true + x_r) this gives
   x_i - x_j <= min(end_i - start_j) =: U_ij; the feasible offsets relative to rank 0 follow from shortest
   paths over U (Floyd-Warshall). Broadcast is excluded (a ring broadcast's upstream ranks may finish before
   downstream ranks start). If 0 lies inside every feasible interval, no offset is applied.
2. Matching. Instance = (process group, k), k = index of the kernel among this rank's NCCL kernels on that
   pg in start order (one stream per pg is checked). Checked: every member has the same count; type,
   In/Out nelems and dtype are equal on all members; and the ProcessGroupNCCL sequence number, taken from
   the host ET (Kineto kernel External id -> record_param_comms cpu_op 'Record function id' -> host-ET
   record_param_comms rf_id -> (seq, isP2P) input), is equal on all members.
3. Decomposition per matched instance (all members' kernels start inside their rank's GPU ProfilerStep#0;
   broadcasts excluded): last_arrival LA = max start, first_end FE = min end; per rank
   wait_i = LA - start_i (arrival skew), post_i = end_i - LA. movement_bound mb = bus bytes / peak (NCCL
   bus-bytes convention of E1, per rank). Time-weighted shares over sum(dur):
   skew = sum(wait); moving@peak = sum(min(post, mb)); post-arrival excess = sum(max(0, post - mb)).
   The three add up to 100%. (Some data can move before LA, e.g. a2a pairs whose peers are already
   present, so post < mb is possible; such cases are counted.)
4. SM-time: SMs held = min(grid CTAs, 132) (as in analyze_kineto.py) x each component, as % of
   132 x GPU ProfilerStep#0 of that rank.
5. Stragglers: the last arriver of every instance; how much wait it imposes on the others; its launch lead
   (kernel start - end of the launching runtime call) and the GPU idle gap before the kernel, vs. the
   other members (small lead + idle GPU before = the rank's host thread issued the collective late).
Usage: python3 analyze_skew.py <nvlink_GBps_per_dir> <nemo_raw_dir> [ranks, default 0..7]
"""
import sys, os, json, collections, statistics
from multiprocessing import Pool
import jstream, analyze_moe
from analyze_moe import ES, coll, bus_and_peer


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def med(v):
    return statistics.median(v) if v else float("nan")


def host_seq(path):
    """host ET record_param_comms rf_id -> (seq, isP2P, pg_name, collective_name, in_msg_nelems)."""
    out = {}
    for n in jstream.stream_array(path, "nodes"):
        if n.get("name") != "record_param_comms": continue
        at = {a["name"]: a["value"] for a in n.get("attrs", [])}
        ty, v = n["inputs"]["types"], n["inputs"]["values"]
        seq = next((tuple(v[i]) for i, t in enumerate(ty) if t == "Tuple[Int,Bool]"), (None, None))
        out[at["rf_id"]] = (seq[0], seq[1], at.get("pg_name"), at.get("collective_name"), at.get("in_msg_nelems"))
    return out


def per_rank(args):
    dev, host, peak = args
    c = analyze_moe.classify(dev)
    d, ev, gk, nsm = c["d"], c["ev"], c["gk"], c["nsm"]
    di = d["distributedInfo"]
    H = host_seq(host)
    cpu_step = [(e["ts"], e["dur"]) for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep")]
    rpc = {e["args"].get("External id"): e["args"].get("Record function id") for e in ev
           if e.get("cat") == "cpu_op" and e["name"] == "record_param_comms"}
    launch = {e["args"]["correlation"]: (e["ts"], e["ts"] + e["dur"]) for e in ev
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    # GPU idle gap right before each activity in the step (any stream), and min launch->start (all kernels)
    gap, mx, lead_min = {}, -1e30, 1e30
    for k in gk:                                  # gk is sorted by start
        corr = k["a"].get("correlation")
        if k["nccl"]: gap[corr] = max(0.0, k["s"] - mx)
        mx = max(mx, k["e"])
        if k["cat"] == "kernel" and corr in launch: lead_min = min(lead_min, k["s"] - launch[corr][0])
    lab = {k["a"].get("correlation"): k for k in gk if k["nccl"]}
    # host-boundness indicators: GPU busy share, launch lead of compute kernels, compute totals
    comp = [k for k in gk if k["cat"] == "kernel" and not k["nccl"]]
    cl = [k["s"] - launch[k["a"]["correlation"]][1] for k in comp if k["a"].get("correlation") in launch]
    hb = dict(busy_pct=100 * analyze_moe.ulen([(k["s"], k["e"]) for k in gk]) / (c["T1"] - c["T0"]),
              comp_ms=sum(k["e"] - k["s"] for k in comp) / 1e3,
              gemm_ms=sum(k["e"] - k["s"] for k in comp if k.get("cls") == "expert_GEMM") / 1e3,
              lead_med=med(cl), lead_lt20=100 * sum(1 for v in cl if v < 20) / max(1, len(cl)), ncomp=len(cl))
    rec, nolink, stream_of = [], 0, collections.defaultdict(set)
    for e in sorted(c["allnccl"], key=lambda e: e["ts"]):
        a = e["args"]; es = ES.get(a.get("dtype"), 1); corr = a.get("correlation")
        ct = coll(e); n = int(a.get("Group size", 1))
        sin, sout = int(a.get("In msg nelems", 0)) * es, int(a.get("Out msg nelems", 0)) * es
        bus, _ = bus_and_peer(ct, sin, sout, n)
        h = H.get(rpc.get(a.get("External id")))
        if h is None: nolink += 1
        g = lab.get(corr)
        pg = a.get("Process Group Name"); stream_of[pg].add(a.get("stream"))
        g3 = a.get("grid", [1, 1, 1])
        rec.append(dict(s=e["ts"], e=e["ts"] + e["dur"], dur=e["dur"], pg=pg,
                        ranks=tuple(json.loads(a.get("Process Group Ranks", "[]"))), ct=ct, sin=sin, sout=sout,
                        dtype=a.get("dtype"), n=n, bus=bus, mb=bus / (peak * 1e3) if bus else 0.0,
                        sms=min(g3[0] * g3[1] * g3[2], nsm), seq=h[0] if h else None,
                        hseq_ok=bool(h) and h[2] == pg and h[4] == int(a.get("In msg nelems", 0)),
                        in_step=c["T0"] <= e["ts"] < c["T1"],
                        cls=g.get("cls") if g else None, lay=g.get("lay") if g else None, role=g["role"] if g else None,
                        lead=e["ts"] - launch[corr][1] if corr in launch else None, gap=gap.get(corr)))
    return dict(rank=c["rank"], nsm=nsm, T0=c["T0"], T1=c["T1"], cpu_step=cpu_step, base=d.get("baseTimeNanoseconds"),
                trace_id=d.get("trace_id"), nccl=di.get("nccl_version"), world=di.get("world_size"), rec=rec,
                nolink=nolink, lead_min=lead_min, multi_stream={p: sorted(s) for p, s in stream_of.items() if len(s) > 1},
                host_n=len(H), hb=hb)


def label(k):
    if k["cls"] is None: return "outside layers: " + (k["role"] or k["ct"])
    p, c = k["lay"][0], k["cls"]
    if c == "a2a_1": return "EP a2a fwd dispatch" if p == "fwd" else "EP a2a bwd a2a_1 (combine-grad)"
    if c == "a2a_2": return "EP a2a fwd combine" if p == "fwd" else "EP a2a bwd a2a_2 (dispatch-grad)"
    return f"{c} {p}"


def floyd(R, U):
    INF = float("inf"); idx = {r: i for i, r in enumerate(R)}; n = len(R)
    D = [[0.0 if i == j else INF for j in range(n)] for i in range(n)]
    for (i, j), u in U.items():   # x_i - x_j <= u  -> edge j -> i, weight u
        D[idx[j]][idx[i]] = min(D[idx[j]][idx[i]], u)
    for m in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][m] + D[m][j] < D[i][j]: D[i][j] = D[i][m] + D[m][j]
    neg = any(D[i][i] < 0 for i in range(n))
    return {r: (-D[idx[r]][0], D[0][idx[r]]) for r in R}, neg


def main():
    peak = float(sys.argv[1]); dr = sys.argv[2]
    ranks = [int(x) for x in sys.argv[3:]] or list(range(8))
    with Pool(len(ranks)) as p:
        R = p.map(per_rank, [(os.path.join(dr, f"device_{r}.json"), os.path.join(dr, f"host_{r}.json"), peak) for r in ranks])
    R = sorted(R, key=lambda r: r["rank"]); RK = {r["rank"]: r for r in R}
    print(f"# E8: arrival skew vs. communication cost, Mixtral-8x7B, ranks {ranks}; NVLink peak {peak} GB/s/dir")
    print(f"world_size {R[0]['world']}, NCCL {R[0]['nccl']}; trace_ids distinct per process: "
          f"{len({r['trace_id'] for r in R}) == len(R)}")

    # ---------------- 1. clock alignment (rank-local evidence)
    print("\n## 1. Clock alignment")
    print("baseTimeNanoseconds per rank:", {r["rank"]: r["base"] for r in R})
    print("rank | CPU ProfilerStep#0 start (us, rel. to rank 0) | CPU dur ms | GPU ProfilerStep#0 start (rel.) | GPU end (rel.) | GPU dur ms | min(kernel start - launch call start) us")
    c0 = RK[ranks[0]]["cpu_step"][0][0]; g0 = RK[ranks[0]]["T0"]
    for r in R:
        cs = r["cpu_step"][0]
        print(f"{r['rank']} | {cs[0]-c0:+.1f} | {cs[1]/1e3:.1f} | {r['T0']-g0:+.1f} | {r['T1']-RK[ranks[0]]['T1']:+.1f} | "
              f"{(r['T1']-r['T0'])/1e3:.1f} | {r['lead_min']:.1f}")
    cps = [r["cpu_step"][0][0] for r in R]; gps = [r["T0"] for r in R]; ges = [r["T1"] for r in R]
    print(f"spread (max-min): CPU step start {max(cps)-min(cps):.1f} us; GPU step start {max(gps)-min(gps):.1f} us; "
          f"GPU step end {max(ges)-min(ges):.1f} us")

    # ---------------- 2. matching
    print("\n## 2. Matching collective instances across ranks: (pg, k-th kernel on that pg)")
    by_pg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        for k in r["rec"]: by_pg[k["pg"]][r["rank"]].append(k)
    print("ranks with a pg on >1 stream:", {r["rank"]: r["multi_stream"] for r in R if r["multi_stream"]} or "none")
    print("NCCL kernels not linked to a host-ET record_param_comms:", {r["rank"]: r["nolink"] for r in R},
          "; host-ET pg/In-nelems disagreeing with Kineto:", sum(1 for r in R for k in r["rec"] if k["seq"] is not None and not k["hseq_ok"]))
    print("pg | members | kernels per member | instances | type/nelems/dtype mismatches | ET-seq mismatches | ET seq range | types")
    inst = []
    for pg in sorted(by_pg, key=int):
        d = by_pg[pg]
        mem = sorted({m for L in d.values() for k in L for m in k["ranks"]})
        traced = [m for m in mem if m in RK]
        cnt = {m: len(d.get(m, [])) for m in traced}
        ni = min(cnt.values()); mm = sm = 0; seqs = []
        for i in range(ni):
            ks = {m: d[m][i] for m in traced}
            v = list(ks.values())
            if len({(k["ct"], k["sin"], k["sout"], k["dtype"]) for k in v}) != 1: mm += 1
            if len({k["seq"] for k in v}) != 1 or v[0]["seq"] is None: sm += 1
            seqs.append(v[0]["seq"])
            if len(traced) == len(mem): inst.append(dict(pg=pg, i=i, ks=ks))
        types = collections.Counter(d[traced[0]][i]["ct"] for i in range(ni))
        contig = all(seqs[i + 1] == seqs[i] + 1 for i in range(len(seqs) - 1)) if None not in seqs else False
        print(f"{pg} | {mem} | {sorted(set(cnt.values()))} | {ni} | {mm} | {sm} | {min(s for s in seqs if s is not None)}..{max(s for s in seqs if s is not None)} "
              f"(consecutive in kernel order: {contig}) | {dict(types)}")
    print(f"matched instances (all members traced): {len(inst)}")
    RK_mem = {x["pg"]: tuple(x["ks"]) for x in inst}

    # ---------------- 1b. causality across ranks
    print("\n## 1b. Cross-rank causality of matched instances (clock-offset bounds)")
    tmid = statistics.median([k["s"] for x in inst for k in x["ks"].values()])
    U = {"all": {}, "first half": {}, "second half": {}}
    viol = collections.Counter(); slack = collections.defaultdict(list); bc = []
    for x in inst:
        v = x["ks"]; la = max(k["s"] for k in v.values()); fe = min(k["e"] for k in v.values())
        ct = next(iter(v.values()))["ct"]
        if ct == "BROADCAST": bc.append((x["pg"], x["i"], fe - la)); continue
        slack[ct].append(fe - la); viol[ct] += fe < la
        half = "first half" if la < tmid else "second half"
        for i, ki in v.items():
            for j, kj in v.items():
                if i == j: continue
                u = ki["e"] - kj["s"]
                for h in ("all", half):
                    if u < U[h].get((i, j), float("inf")): U[h][(i, j)] = u
    print("type | instances | first_end < last_arrival | min (first_end - last_arrival) us | p1 us | median us")
    for ct, L in sorted(slack.items()):
        print(f"{ct} | {len(L)} | {viol[ct]} | {min(L):.1f} | {q(L, .01):.1f} | {med(L):.1f}")
    print("excluded broadcasts (pg, k, first_end - last_arrival us):", [(p, i, round(s, 1)) for p, i, s in bc])
    for x in inst:
        v = x["ks"]
        if next(iter(v.values()))["ct"] != "BROADCAST": continue
        s0 = v[min(v)]["s"]
        print(f"  broadcast pg {x['pg']} k={x['i']}: per rank (start, end) us rel. to rank {min(v)} start: " +
              ", ".join(f"r{m} ({v[m]['s']-s0:+.0f}, {v[m]['e']-s0:+.0f})" for m in sorted(v)) +
              "; GPU ProfilerStep#0 end rel.: " + ", ".join(f"r{m} {RK[m]['T1']-s0:+.0f}" for m in sorted(v)))
    for h in ("all", "first half", "second half"):
        b, neg = floyd(ranks, U[h])
        print(f"feasible clock offset of rank r relative to rank {ranks[0]} [{h}] (us): " +
              ", ".join(f"r{r} [{lo:+.1f}, {hi:+.1f}]" for r, (lo, hi) in b.items()) + f"; negative cycle: {neg}")
    b, neg = floyd(ranks, U["all"])
    zero_ok = all(lo <= 0 <= hi for lo, hi in b.values()) and not neg
    print(f"zero offset feasible for every rank: {zero_ok} -> offset applied: {'none' if zero_ok else 'MIDPOINT (see above)'}")
    pairs = sorted({tuple(sorted(p)) for p in U["all"]})
    print("direct pairs, min(end_i - start_j) both ways (us):",
          "; ".join(f"({i},{j}) {U['all'][(i, j)]:.1f}/{U['all'][(j, i)]:.1f}" for i, j in pairs if j == i + 1 and i % 2 == 0))

    # ---------------- 3. decomposition
    print("\n## 3. Decomposition per matched instance (in-step, non-broadcast)")
    rows = []  # one per (instance, rank)
    excl = collections.Counter(); excl_ms = 0.0; mixed = collections.Counter(); IN = []
    for x in inst:
        v = x["ks"]; k0 = next(iter(v.values()))
        if k0["ct"] == "BROADCAST" or not all(k["in_step"] for k in v.values()):
            excl["broadcast" if k0["ct"] == "BROADCAST" else "not all members in step"] += 1
            excl_ms += sum(k["dur"] for k in v.values() if k["in_step"]); continue
        labs = collections.Counter(label(k) for k in v.values())
        lab = labs.most_common(1)[0][0]
        if len(labs) > 1: mixed[tuple(sorted(labs))] += 1
        la = max(k["s"] for k in v.values()); fa = min(k["s"] for k in v.values())
        fe = min(k["e"] for k in v.values()); le = max(k["e"] for k in v.values())
        last = max(v, key=lambda m: v[m]["s"])
        second = sorted((k["s"] for k in v.values()), reverse=True)[1]
        xi = dict(pg=x["pg"], lab=lab, ct=k0["ct"], sin=k0["sin"], n=k0["n"], la=la, spread=la - fa, fe=fe - la, le=le - la,
                  last=last, margin=la - second, members=tuple(sorted(v)), wait_sum=sum(la - k["s"] for k in v.values()))
        IN.append(xi)
        for m, k in v.items():
            w, po = la - k["s"], k["e"] - la
            rows.append(dict(rank=m, lab=lab, ct=k["ct"], sin=k["sin"], n=k["n"], pg=x["pg"], dur=k["dur"], wait=w, post=po,
                             mb=k["mb"], mv=min(po, k["mb"]), ex=max(0.0, po - k["mb"]), sms=k["sms"], last=(m == last),
                             lead=k["lead"], gap=k["gap"], bus=k["bus"]))
    print(f"instances used: {len(IN)}; excluded: {dict(excl)} ({excl_ms/1e3:.1f} ms of in-step kernel time); "
          f"instances whose members disagree on the class label: {sum(mixed.values())} {dict(mixed) if mixed else ''}")
    in_step_total = sum(k["dur"] for r in R for k in r["rec"] if k["in_step"]) / 1e3
    print(f"in-step NCCL kernel time, all ranks: {in_step_total:.1f} ms; covered by the decomposition: "
          f"{sum(x['dur'] for x in rows)/1e3:.1f} ms")
    order = ["EP a2a fwd dispatch", "EP a2a fwd combine", "EP a2a bwd a2a_1 (combine-grad)", "EP a2a bwd a2a_2 (dispatch-grad)",
             "TP-MoE fwd", "TP-MoE bwd", "TP-attn fwd", "TP-attn bwd", "AR fwd", "AR bwd", "DP fwd", "DP bwd"]
    labs = order + sorted({x["lab"] for x in rows} - set(order))
    preds = [(l, (lambda l: lambda x: x["lab"] == l)(l)) for l in labs]
    preds += [("== EP a2a, all", lambda x: x["lab"].startswith("EP a2a")),
              ("== TP-MoE, all", lambda x: x["lab"].startswith("TP-MoE")),
              ("== TP-attn, all", lambda x: x["lab"].startswith("TP-attn")),
              ("== AR (in layers + outside)", lambda x: x["ct"] == "ALL_REDUCE"),
              ("== DP (in layers + outside)", lambda x: x["lab"].startswith("DP") or x["lab"] == "outside layers: DP"),
              ("== all NCCL", lambda x: True)]
    groups = [(l, [x for x in rows if f(x)], [y for y in IN if f(y)]) for l, f in preds]
    print("class | instances | rank-kernels | sum dur ms | skew % | moving@peak % | post-arrival excess % | "
          "median dur us | median wait us | median post us | median mb us | median spread (LA-first start) us | "
          "median FE-LA us | post<mb (rank-kernels) | mean wait us | mean post us")
    S = {}
    for l, X, I in groups:
        if not X: continue
        sd = sum(x["dur"] for x in X)
        ni = len(I); sp = [y["spread"] for y in I]; fe = [y["fe"] for y in I]
        S[l] = dict(skew=100 * sum(x["wait"] for x in X) / sd, mv=100 * sum(x["mv"] for x in X) / sd,
                    ex=100 * sum(x["ex"] for x in X) / sd, sd=sd)
        print(f"{l} | {ni} | {len(X)} | {sd/1e3:.1f} | {S[l]['skew']:.1f} | {S[l]['mv']:.1f} | {S[l]['ex']:.1f} | "
              f"{med([x['dur'] for x in X]):.1f} | {med([x['wait'] for x in X]):.1f} | {med([x['post'] for x in X]):.1f} | "
              f"{med([x['mb'] for x in X]):.1f} | {med(sp):.1f} | {med(fe):.1f} | {sum(1 for x in X if x['post'] < x['mb'])} | "
              f"{sum(x['wait'] for x in X)/len(X):.1f} | {sum(x['post'] for x in X)/len(X):.1f}")
    print("(skew = sum(LA - start_i); moving@peak = sum(min(post_i, mb)); excess = sum(max(0, post_i - mb)); all / sum(dur_i))")
    # raw (unclipped) post-arrival excess
    sd = sum(x["dur"] for x in rows)
    print(f"all NCCL, unclipped: sum(post - mb) = {sum(x['post']-x['mb'] for x in rows)/1e3:.1f} ms = "
          f"{100*sum(x['post']-x['mb'] for x in rows)/sd:.1f}% of {sd/1e3:.1f} ms; sum(post) = {100*sum(x['post'] for x in rows)/sd:.1f}%")

    print("\n### per collective type and size (all classes pooled)")
    print("type | n | in MiB | rank-kernels | sum dur ms | skew % | moving@peak % | excess % | mb us | post-arrival us: min | p1 | p10 | median | p90 | "
          "median dur us | min dur us | busBW at median post GB/s")
    tg = collections.defaultdict(list)
    for x in rows: tg[(x["ct"], x["n"], x["sin"])].append(x)
    for key, X in sorted(tg.items(), key=lambda kv: -sum(x["dur"] for x in kv[1])):
        sd = sum(x["dur"] for x in X)
        if sd < 5e3: continue
        P = [x["post"] for x in X]; mp = med(P)
        print(f"{key[0]} | {key[1]} | {key[2]/2**20:.2f} | {len(X)} | {sd/1e3:.1f} | {100*sum(x['wait'] for x in X)/sd:.1f} | "
              f"{100*sum(x['mv'] for x in X)/sd:.1f} | {100*sum(x['ex'] for x in X)/sd:.1f} | {X[0]['mb']:.1f} | "
              f"{min(P):.1f} | {q(P, .01):.1f} | {q(P, .1):.1f} | {mp:.1f} | {q(P, .9):.1f} | {med([x['dur'] for x in X]):.1f} | "
              f"{min(x['dur'] for x in X):.1f} | {X[0]['bus']/mp/1e3 if mp > 0 else float('nan'):.1f}")
    A = [x for x in rows if x["ct"] == "ALL_TO_ALL" and x["sin"] == 32 * 2**20]
    AI = [y for y in IN if y["ct"] == "ALL_TO_ALL" and y["sin"] == 32 * 2**20]
    if A:
        P = [x["post"] for x in A]
        print(f"\n32 MiB EP all-to-all: {len(AI)} instances, {len(A)} rank-kernels; movement bound {A[0]['mb']:.1f} us "
              f"(bus {A[0]['bus']/2**20:.2f} MiB at {peak} GB/s)")
        print(f"  post-arrival per rank-kernel: min {min(P):.1f} us, p10 {q(P,.1):.1f}, median {med(P):.1f}, p90 {q(P,.9):.1f}; "
              f"median/min = {med(P)/min(P):.2f}")
        print(f"  per instance, first_end - LA: min {min(y['fe'] for y in AI):.1f}, median {med([y['fe'] for y in AI]):.1f} us; "
              f"last_end - LA: min {min(y['le'] for y in AI):.1f}, median {med([y['le'] for y in AI]):.1f} us")
        print(f"  arrival spread (LA - first start): median {med([y['spread'] for y in AI]):.1f} us, p90 {q([y['spread'] for y in AI],.9):.1f} us; "
              f"kernel duration median {med([x['dur'] for x in A]):.1f} us, min {min(x['dur'] for x in A):.1f} us")
        print(f"  post-arrival on the last arriver itself (= its full duration): median {med([x['post'] for x in A if x['last']]):.1f} us; "
              f"on the others: median {med([x['post'] for x in A if not x['last']]):.1f} us")

    # ---------------- 4. SM-time
    print("\n## 4. SM-time held by NCCL kernels: waiting for peers vs. post-arrival (SMs = min(grid CTAs, 132))")
    print("rank | GPU step ms | SM capacity SM*ms | NCCL SM-time in step % cap | decomposed: wait % cap | post-arrival % cap "
          "(= moving@peak + excess) | moving@peak % cap | excess % cap | wait share of decomposed NCCL SM-time %")
    tot = collections.Counter()
    for r in R:
        cap = r["nsm"] * (r["T1"] - r["T0"])
        alln = sum(k["sms"] * k["dur"] for k in r["rec"] if k["in_step"])
        X = [x for x in rows if x["rank"] == r["rank"]]
        w = sum(x["sms"] * x["wait"] for x in X); po = sum(x["sms"] * x["post"] for x in X)
        mv = sum(x["sms"] * x["mv"] for x in X); ex = sum(x["sms"] * x["ex"] for x in X)
        for kk, vv in (("cap", cap), ("all", alln), ("w", w), ("po", po), ("mv", mv), ("ex", ex)): tot[kk] += vv
        print(f"{r['rank']} | {(r['T1']-r['T0'])/1e3:.1f} | {cap/1e3:.0f} | {100*alln/cap:.2f} | {100*w/cap:.2f} | {100*po/cap:.2f} | "
              f"{100*mv/cap:.2f} | {100*ex/cap:.2f} | {100*w/(w+po):.1f}")
    print(f"all | - | {tot['cap']/1e3:.0f} | {100*tot['all']/tot['cap']:.2f} | {100*tot['w']/tot['cap']:.2f} | {100*tot['po']/tot['cap']:.2f} | "
          f"{100*tot['mv']/tot['cap']:.2f} | {100*tot['ex']/tot['cap']:.2f} | {100*tot['w']/(tot['w']+tot['po']):.1f}")
    print("by class, all ranks pooled, % of total SM capacity (8 x 132 x step): wait | moving@peak | excess")
    for l, X, _ in groups:
        if not X: continue
        print(f"  {l}: {100*sum(x['sms']*x['wait'] for x in X)/tot['cap']:.2f} | {100*sum(x['sms']*x['mv'] for x in X)/tot['cap']:.2f} | "
              f"{100*sum(x['sms']*x['ex'] for x in X)/tot['cap']:.2f}")

    # ---------------- 5. stragglers
    print("\n## 5. Stragglers: who arrives last")
    def coarse(lab):   # "EP a2a" for all four a2a sub-classes; fwd/bwd merged; outside-layer labels kept
        if lab.startswith("EP a2a"): return "EP a2a"
        return lab if lab.startswith("outside") else lab.split(" ")[0]
    gk_ = collections.defaultdict(list)
    for y in IN: gk_[(y["members"], coarse(y["lab"]))].append(y)
    print("group members | class | instances | last-arriver counts {rank: n} | share of the group's wait imposed by each last arriver % | median margin over 2nd-last us")
    for (mem, cl), Y in sorted(gk_.items(), key=lambda kv: (-len(kv[0][0]), kv[0])):
        if sum(y["wait_sum"] for y in Y) < 1e3: continue
        c = collections.Counter(y["last"] for y in Y)
        ws = collections.Counter()
        for y in Y: ws[y["last"]] += y["wait_sum"]
        W = sum(ws.values())
        print(f"{list(mem)} | {cl} | {len(Y)} | {dict(sorted(c.items()))} | "
              f"{ {m: round(100*v/W, 1) for m, v in sorted(ws.items())} } | {med([y['margin'] for y in Y]):.1f}")
    print("\nskew share by group (all classes of a family pooled): members | family | rank-kernels | sum dur ms | skew % | moving@peak % | excess %")
    fam = lambda x: "EP a2a" if x["lab"].startswith("EP a2a") else ("TP" if x["lab"].startswith(("TP-", "outside layers: TP")) else
                                                                    ("AR" if x["ct"] == "ALL_REDUCE" else "DP"))
    gg = collections.defaultdict(list)
    for x in rows: gg[(tuple(sorted(RK_mem[x["pg"]])), fam(x))].append(x)
    for (mem, f), X in sorted(gg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        sd = sum(x["dur"] for x in X)
        print(f"{list(mem)} | {f} | {len(X)} | {sd/1e3:.1f} | {100*sum(x['wait'] for x in X)/sd:.1f} | "
              f"{100*sum(x['mv'] for x in X)/sd:.1f} | {100*sum(x['ex'] for x in X)/sd:.1f}")
    print("\nper rank, time spent waiting for peers (sum of wait_i, ms) by family: rank | EP a2a | TP | AR | DP | total")
    for r in ranks:
        X = [x for x in rows if x["rank"] == r]; wf = collections.Counter()
        for x in X: wf[fam(x)] += x["wait"]
        print(f"{r} | {wf['EP a2a']/1e3:.1f} | {wf['TP']/1e3:.1f} | {wf['AR']/1e3:.1f} | {wf['DP']/1e3:.1f} | {sum(wf.values())/1e3:.1f}")
    print("\nEP a2a by sub-class: last-arriver counts per EP group")
    for l in order[:4]:
        for mem in sorted({y["members"] for y in IN if y["lab"] == l}):
            Y = [y for y in IN if y["lab"] == l and y["members"] == mem]
            print(f"  {l} {list(mem)}: {dict(sorted(collections.Counter(y['last'] for y in Y).items()))}")
    tw = sum(y["wait_sum"] for y in IN); ws = collections.Counter()
    for y in IN: ws[y["last"]] += y["wait_sum"]
    print(f"\nall classes: total wait (sum over ranks) {tw/1e3:.1f} ms; share imposed by each rank as last arriver: "
          + ", ".join(f"r{m} {100*v/tw:.1f}%" for m, v in sorted(ws.items())))
    print("\nlast arriver vs. other members, EP a2a and TP (rank-kernels):")
    print("subset | rank-kernels | median launch lead us (kernel start - launch call end) | p10 lead | median GPU idle gap before kernel us | "
          "share with lead < 50 us")
    for nm, sel in (("EP a2a", lambda x: x["lab"].startswith("EP a2a")), ("TP (MoE+attn)", lambda x: x["lab"].startswith("TP-"))):
        for tag, f in (("last arriver", lambda x: x["last"]), ("others", lambda x: not x["last"])):
            X = [x for x in rows if sel(x) and f(x) and x["lead"] is not None]
            L = [x["lead"] for x in X]
            print(f"{nm} {tag} | {len(X)} | {med(L):.1f} | {q(L,.1):.1f} | {med([x['gap'] for x in X if x['gap'] is not None]):.1f} | "
                  f"{100*sum(1 for v in L if v < 50)/max(1,len(L)):.1f}%")
    print("\nper rank, host-boundness indicators (in-step): GPU busy % of step (any kernel/memcpy/memset) | non-NCCL kernel sum ms | "
          "expert GEMM ms (E1 class) | median launch lead of non-NCCL kernels us | % non-NCCL kernels starting < 20 us after their launch call returns")
    for r in R:
        h = r["hb"]
        print(f"{r['rank']} | {h['busy_pct']:.1f} | {h['comp_ms']:.1f} | {h['gemm_ms']:.1f} | {h['lead_med']:.1f} | {h['lead_lt20']:.1f}")
    print("\nper rank, EP a2a: times last | median launch lead us | median GPU idle gap before us | median wait us | sum wait ms | sum dur ms")
    for r in ranks:
        X = [x for x in rows if x["rank"] == r and x["lab"].startswith("EP a2a")]
        print(f"{r} | {sum(1 for x in X if x['last'])}/{len(X)} | {med([x['lead'] for x in X]):.1f} | {med([x['gap'] for x in X]):.1f} | "
              f"{med([x['wait'] for x in X]):.1f} | {sum(x['wait'] for x in X)/1e3:.1f} | {sum(x['dur'] for x in X)/1e3:.1f}")


if __name__ == "__main__":
    main()
