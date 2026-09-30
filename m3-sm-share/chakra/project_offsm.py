#!/usr/bin/env python3
"""M3 (trace part, extension): PROJECTION of compute time lost because communication kernels hold SMs,
and what an off-SM communication engine would recover. Every number printed here is a projection that
combines trace-derived inputs (moe_summary_Mixtral-8x7B.json from analyze_moe.py,
nccl_et_Mixtral-8x22B.json from analyze_nccl_et.py) with costs measured on our H200 testbed:

  BF16 expert GEMM throughput while k SMs are held by another kernel
      measured co-location:            k=8 90.5%, k=16 72.7%, k=20 48.9%
      perfect partitioning (lower bnd): k=8 90.3%, k=16 88.8%, k=20 79.1%
  IBGDA post: 6-8 us of SM time per put; CPU-proxy post: 0.075 us CPU time per post.

Model (per rank, per training step):
  E   = expert GEMM time (trace).
  C   = time the MoE communication kernels hold SMs (EP all-to-all + TP collectives), either
        "held" = kernel durations as traced (includes waiting), or
        "moving" = sum over kernels of the fastest same-size kernel's duration (analyze_moe.py /
        analyze_nccl_et.py per-size minima; an empirical estimate of time actually moving data).
  theta(k) = GEMM throughput while comm holds k SMs.
  S1 "as traced": only the traced comm/compute overlap O co-runs; lost = O * (1 - theta).
  S2 "overlapped MoE": expert GEMMs are scheduled concurrently with the MoE comm (what an overlapped
     MoE implementation does). GEMM work that can co-run is W = min(E, theta*C); it is slowed by
     1/theta, so lost = W * (1/theta - 1). An off-SM engine (theta = 1) recovers all of it.
  k: 8x7B NCCL kernels hold 32 (a2a) / 24 (AG/RS) CTAs = SMs (Kineto grids). The testbed curve stops at
     k=20, so k=20 is used as a LOWER bound on the penalty (assumes throughput does not rise with k).
     8x22B has no grids; a k sweep is shown.
Usage: python3 project_offsm.py
"""
import json, os, statistics

H = os.path.dirname(os.path.abspath(__file__))
COLOC = {8: 0.905, 16: 0.727, 20: 0.489}
PART = {8: 0.903, 16: 0.888, 20: 0.791}
IBGDA_US = (6.0, 8.0); PROXY_CPU_US = 0.075


def loss_S2(E, C, th):
    W = min(E, th * C)
    return W * (1 / th - 1)


def rng(f, k):
    a, b = f(PART[k]), f(COLOC[k])
    return min(a, b), max(a, b)


def main():
    m = json.load(open(os.path.join(H, "moe_summary_Mixtral-8x7B.json")))
    print("# PROJECTION (not measured): SM-holding cost of communication kernels and off-SM recovery")
    print("# theta ranges use testbed [perfect-partitioning, measured co-location] throughput; all ms per training step\n")
    print("## Mixtral-8x7B (Kineto ranks 0/2/3/6; comm kernels hold 24-32 SMs -> k=20 used as a lower bound)")
    print("rank | step ms | E expert GEMM ms | C_held MoE comm ms | C_moving ms | overlap O ms | "
          "S1 lost ms | S2-held lost ms (% step) | S2-moving lost ms (% step) | NCCL SM-time as full-GPU ms (% step)")
    rows = []
    for rk, v in sorted(m["per_rank"].items(), key=lambda x: int(x[0])):
        t = v["step_totals_ms"]; w = v["whole"]; step = v["step_ms"]
        E = t["expert_GEMM"]; Ch = t["a2a_1"] + t["a2a_2"] + t["TP-MoE"]
        Cm = w["moving_ms_by_role"].get("EP", 0) + w["moving_ms_by_role"].get("TP", 0)
        O = w["overlap_ms"]
        s1 = rng(lambda th: O * (1 - th), 20)
        s2h = rng(lambda th: loss_S2(E, Ch, th), 20)
        s2m = rng(lambda th: loss_S2(E, Cm, th), 20)
        gpu_eq = w["nccl_sm_ms"] / 132
        rows.append((E, Ch, Cm, s2h, s2m, step))
        print(f"{rk} | {step:.0f} | {E:.0f} | {Ch:.0f} | {Cm:.0f} | {O:.1f} | {s1[0]:.0f}-{s1[1]:.0f} | "
              f"{s2h[0]:.0f}-{s2h[1]:.0f} ({100*s2h[0]/step:.1f}-{100*s2h[1]/step:.1f}%) | "
              f"{s2m[0]:.0f}-{s2m[1]:.0f} ({100*s2m[0]/step:.1f}-{100*s2m[1]/step:.1f}%) | {gpu_eq:.0f} ({100*gpu_eq/step:.1f}%)")
    print("\nk sweep for rank 0 (what the loss would be if the comm kernels held only k SMs), S2-held / S2-moving, ms:")
    E, Ch, Cm, _, _, step = rows[0]
    for k in (8, 16, 20):
        a = rng(lambda th: loss_S2(E, Ch, th), k); b = rng(lambda th: loss_S2(E, Cm, th), k)
        print(f"  k={k}: S2-held {a[0]:.0f}-{a[1]:.0f} ms; S2-moving {b[0]:.0f}-{b[1]:.0f} ms  "
              f"(E x (1/theta - 1) upper limit {E*(1/PART[k]-1):.0f}-{E*(1/COLOC[k]-1):.0f} ms)")

    # ---- Mixtral-8x22B
    x = json.load(open(os.path.join(H, "nccl_et_Mixtral-8x22B.json")))
    ranks = sorted(x["per_rank_held_moving_ms"], key=int)
    Es, Chs, Cms = [], [], []
    for rk in ranks:
        p = x["per_rank_held_moving_ms"][rk]
        Es.append(x["expert_gemm_ms"][rk])
        Chs.append(p["ALL_TO_ALL|inter"][0] + p.get("ALL_GATHER|intra", [0, 0])[0] + p.get("REDUCE_SCATTER|intra", [0, 0])[0])
        Cms.append(p["ALL_TO_ALL|inter"][1] + p.get("ALL_GATHER|intra", [0, 0])[1] + p.get("REDUCE_SCATTER|intra", [0, 0])[1])
    E, Ch, Cm = statistics.median(Es), statistics.median(Chs), statistics.median(Cms)
    print("\n## Mixtral-8x22B (ET only; no grids, so k is swept; medians over 32 ranks)")
    print(f"E (GEMMs under LinearWithGradAccumulation..., includes LM head) = {E:.0f} ms; "
          f"C_held (EP a2a inter + TP AG/RS intra) = {Ch:.0f} ms; C_moving = {Cm:.0f} ms")
    for k in (8, 16, 20):
        a = rng(lambda th: loss_S2(E, Ch, th), k); b = rng(lambda th: loss_S2(E, Cm, th), k)
        print(f"  k={k}: S2-held lost {a[0]:.0f}-{a[1]:.0f} ms ({100*a[0]/E:.0f}-{100*a[1]/E:.0f}% of E); "
              f"S2-moving lost {b[0]:.0f}-{b[1]:.0f} ms ({100*b[0]/E:.0f}-{100*b[1]/E:.0f}% of E)")
    # GPU-initiated vs proxy posting cost for the inter-node a2a
    peers = 6
    g = x["groups"]; a2a = g["ALL_TO_ALL|inter|8|2"]
    na2a = round(a2a["kernels"] / len(ranks))
    held_per = a2a["sum_ms"] * 1e3 / a2a["kernels"]
    print("\n## Posting cost of the 8x22B inter-node a2a if issued one RDMA put per remote peer (PROJECTION)")
    print(f"remote peers per rank per a2a = {peers} (EP8 group has 2 members per node); a2a per rank per step = {na2a}")
    print(f"mean traced kernel duration per a2a = {held_per:.0f} us (held by every CTA of the NCCL kernel)")
    print(f"IBGDA: {peers}x{IBGDA_US[0]:.0f}-{IBGDA_US[1]:.0f} us = {peers*IBGDA_US[0]:.0f}-{peers*IBGDA_US[1]:.0f} us of SM time per a2a; "
          f"per step {na2a*peers*IBGDA_US[0]/1e3:.1f}-{na2a*peers*IBGDA_US[1]/1e3:.1f} SM-ms")
    print(f"CPU proxy: {peers}x{PROXY_CPU_US} us = {peers*PROXY_CPU_US:.2f} us CPU per a2a; per step {na2a*peers*PROXY_CPU_US/1e3:.2f} ms CPU")
    print("(per-put costs scale linearly with the number of chunks per peer; 1 put/peer is the minimum)")


if __name__ == "__main__":
    main()
