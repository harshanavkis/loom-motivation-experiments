#!/usr/bin/env python3
"""M3 (trace part): Kineto device traces (Mixtral-8x7B, nemo_raw/device_<r>.json) -> communication
time, compute/communication overlap, and SM-time share of NCCL kernels within ProfilerStep#0.

SM-time model: a kernel with G = gridX*gridY*gridZ CTAs occupies min(G, numSms) SMs for its whole
duration (upper bound on residency; NCCL launches 1 CTA per channel, one CTA per SM).
Usage: python3 analyze_kineto.py <gpus_per_node> device_*.json
Each file is ~155 MB; it is stream-parsed with jstream.py (stdlib json.raw_decode), keeping only the
event categories used here.
"""
import sys, json, re, collections
import jstream


def union(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]: out[-1][1] = e
        else:
            out.append([s, e])
    return out


def length(u):
    return sum(e - s for s, e in u)


def intersect(a, b):
    i = j = 0; tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: tot += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot


def coll_type(e):
    n = e["name"]; c = e["args"].get("Collective name", "")
    if "SendRecv" in n:
        return "ALL_TO_ALL" if "all_to_all" in c or c == "" else "SEND_RECV(" + c + ")"
    for k, v in (("AllGather", "ALL_GATHER"), ("ReduceScatter", "REDUCE_SCATTER"),
                 ("AllReduce", "ALL_REDUCE"), ("Broadcast", "BROADCAST")):
        if k in n: return v
    return "OTHER_NCCL"


def analyze(path, G):
    d = jstream.load_kineto(path, {"gpu_user_annotation", "kernel", "gpu_memcpy", "gpu_memset"})
    rank = d["distributedInfo"]["rank"]; world = d["distributedInfo"]["world_size"]
    nsm = d["deviceProperties"][0]["numSms"]; gpu = d["deviceProperties"][0]["name"]
    ev = d["traceEvents"]
    step = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")]
    assert len(step) == 1, step
    T0 = step[0]["ts"]; T1 = T0 + step[0]["dur"]
    comm, comp = [], []
    by = collections.defaultdict(lambda: [0, 0.0, 0.0, collections.Counter()])  # (type,cls)->n,dur,smtime,grids
    sm_comm = sm_comp = 0.0; nccl_grid = collections.Counter(); pgs = {}
    memx = 0.0
    for e in ev:
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"): continue
        s = max(e["ts"], T0); t = min(e["ts"] + e["dur"], T1)
        if t <= s: continue
        dur = t - s
        if cat != "kernel":
            comp.append((s, t)); memx += dur; continue
        g = e["args"].get("grid", [1, 1, 1]); ctas = g[0] * g[1] * g[2]
        sms = min(ctas, nsm)
        if e["name"].startswith("nccl"):
            ranks = json.loads(e["args"].get("Process Group Ranks", "[]"))
            nodes = {r // G for r in ranks}
            cls = "intra" if len(nodes) <= 1 else "inter"
            pgs[e["args"].get("Process Group Name")] = ranks
            ct = coll_type(e)
            b = by[(ct, cls)]; b[0] += 1; b[1] += dur; b[2] += sms * dur; b[3][ctas] += 1
            sm_comm += sms * dur; comm.append((s, t))
        else:
            sm_comp += sms * dur; comp.append((s, t))
    step_us = T1 - T0
    uc, uk = union(comm), union(comp)
    lc, lk = length(uc), length(uk)
    ov = intersect(uc, uk)
    busy = length(union(comm + comp))
    print(f"\n## rank {rank}/{world} ({gpu}, {nsm} SMs) file {path}")
    print(f"ProfilerStep#0 on GPU: {step_us/1e3:.1f} ms")
    print(f"GPU busy (any kernel/memcpy/memset): {busy/1e3:.1f} ms = {100*busy/step_us:.1f}% of step")
    print(f"comm-kernel wall time (union of NCCL kernel intervals): {lc/1e3:.1f} ms = {100*lc/step_us:.1f}% of step")
    print(f"compute wall time (union of non-NCCL kernels+memcpy/memset): {lk/1e3:.1f} ms = {100*lk/step_us:.1f}% of step")
    print(f"comm overlapped with compute: {ov/1e3:.1f} ms = {100*ov/max(lc,1):.1f}% of comm wall time")
    print(f"exposed comm (comm running, no compute running): {(lc-ov)/1e3:.1f} ms = {100*(lc-ov)/step_us:.1f}% of step")
    print("type | class | kernels | sum dur ms | % of step (sum dur) | SM-time (SM*ms) | CTA counts (grid) histogram")
    for (ct, cls), b in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"{ct} | {cls} | {b[0]} | {b[1]/1e3:.1f} | {100*b[1]/step_us:.1f} | {b[2]/1e3:.0f} | {dict(b[3].most_common(4))}")
    tot = sm_comm + sm_comp
    cap = nsm * step_us
    print(f"SM-time NCCL = {sm_comm/1e3:.0f} SM*ms; non-NCCL kernels = {sm_comp/1e3:.0f} SM*ms")
    print(f"NCCL share of kernel SM-time = {100*sm_comm/tot:.1f}%")
    print(f"NCCL SM-time / SM capacity of step ({nsm} SMs x step) = {100*sm_comm/cap:.2f}%; "
          f"non-NCCL / capacity = {100*sm_comp/cap:.1f}%")
    print(f"avg SMs held by NCCL while a NCCL kernel runs = {sm_comm/max(lc,1):.1f}")
    print("process groups seen on NCCL kernels:", {k: v for k, v in sorted(pgs.items(), key=lambda x: int(x[0]))})
    return dict(rank=rank, step_ms=step_us / 1e3, comm_wall_pct=100 * lc / step_us,
                overlap_pct_of_comm=100 * ov / max(lc, 1), exposed_pct=100 * (lc - ov) / step_us,
                nccl_sm_share_pct=100 * sm_comm / tot, nccl_sm_cap_pct=100 * sm_comm / cap)


if __name__ == "__main__":
    G = int(sys.argv[1]); rows = [analyze(p, G) for p in sys.argv[2:]]
    print("\n## summary")
    print("rank | step ms | comm wall % step | % comm overlapped | exposed comm % step | NCCL % of kernel SM-time | NCCL % of SM capacity")
    for r in rows:
        print(f"{r['rank']} | {r['step_ms']:.1f} | {r['comm_wall_pct']:.1f} | {r['overlap_pct_of_comm']:.1f} | "
              f"{r['exposed_pct']:.1f} | {r['nccl_sm_share_pct']:.1f} | {r['nccl_sm_cap_pct']:.2f}")
