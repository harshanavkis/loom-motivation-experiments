#!/usr/bin/env python3
"""Summarize interfere.cu output into markdown tables (medians over reps)."""
import csv, statistics, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results_steve.csv"
header, runs, sizes, checks = None, [], [], []
for line in open(path):
    if line.startswith("#"):
        header = line.strip()
        continue
    f = line.strip().split(",")
    if f[0] == "run" and f[1] != "rep":
        runs.append(dict(zip(["rep", "workload", "mode", "dir", "k", "target", "sm_target", "ms", "metric",
                              "unit", "comm", "distinct"], f[1:])))
    elif f[0] == "ce_size":
        sizes.append(f[1:])
    elif f[0] == "ce_check":
        checks.append(f[1:])

med = defaultdict(list)
for r in runs:
    med[(r["workload"], r["mode"], r["dir"], int(r["k"]), float(r["target"]))].append(
        (float(r["metric"]), float(r["comm"]), int(r["distinct"])))
def get(*key):
    v = med.get(key)
    if not v:
        return None
    return (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v), min(x[2] for x in v), len(v))

print(header)
print()
print("Copy-engine check (copy completes while all SMs are held => it runs on a copy engine):")
for c in checks:
    print(f"- {c[0]} {c[1]}: {c[2]}")
nsm = int(header.split("sms=")[1].split()[0])
ks = sorted({int(r["k"]) for r in runs if r["mode"] == "idle"})
rates = sorted({float(r["target"]) for r in runs if r["mode"] == "ce" and r["dir"] == "d2d"}, key=lambda x: (x == 0, x))

for w in ["gemm", "gemm_down", "triad"]:
    base = get(w, "none", "-", 0, 0.0)
    if not base:
        continue
    unit = "TFLOP/s" if w != "triad" else "GB/s"
    print(f"\n### {w}: baseline {base[0]:.1f} {unit} (median of {base[3]})\n")
    print("Throughput as % of the baseline; `ideal` = (SMs-k)/SMs; smcopy/ce columns give achieved comm GB/s in brackets.\n")
    cols = ["k", "ideal"] + (["target"] if w != "triad" else []) + ["idle"]
    cols += [f"smcopy d2d {int(r) if r else 'max'}" for r in rates] + ["smcopy d2h 50"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for k in ks:
        row = [str(k), f"{100 * (nsm - k) / nsm:.1f}"]
        for mode in (["target"] if w != "triad" else []) + ["idle"]:
            v = get(w, mode, "-", k, 0.0)
            row.append(f"{100 * v[0] / base[0]:.1f}" if v else "-")
        for d, rr in [("d2d", r) for r in rates] + [("d2h", 50.0)]:
            v = get(w, "smcopy", d, k, rr)
            row.append(f"{100 * v[0] / base[0]:.1f} [{v[1]:.0f}]" if v else "-")
        print("| " + " | ".join(row) + " |")
    print(f"\nCopy engine moving the same bytes (no SMs held):\n")
    print("| dir | target GB/s | % of baseline | achieved GB/s |")
    print("|---|---|---|---|")
    for d in ["d2d", "d2h"]:
        for rr in rates:
            v = get(w, "ce", d, 0, rr)
            if v:
                print(f"| {d} | {int(rr) if rr else 'max'} | {100 * v[0] / base[0]:.1f} | {v[1]:.0f} |")

if sizes:
    agg = defaultdict(list)
    for api, d, s, count, rep, ms, gbps, mcps in sizes:
        agg[(api, d, int(s))].append((float(gbps), float(mcps), int(count)))
    print("\n### Copy-engine throughput vs request size (scattered destinations)\n")
    print("| size (B) | copies | batch d2d GB/s | batch d2h GB/s | loop d2d GB/s | loop d2h GB/s | batch d2d Mcopies/s |")
    print("|---|---|---|---|---|---|---|")
    for s in sorted({k[2] for k in agg}):
        def m(api, d, i=0):
            v = agg.get((api, d, s))
            return statistics.median(x[i] for x in v) if v else float("nan")
        cnt = agg[("cudaMemcpyBatchAsync", "d2d", s)][0][2]
        print(f"| {s} | {cnt} | {m('cudaMemcpyBatchAsync', 'd2d'):.1f} | {m('cudaMemcpyBatchAsync', 'd2h'):.1f} | "
              f"{m('cudaMemcpyAsync_loop', 'd2d'):.1f} | {m('cudaMemcpyAsync_loop', 'd2h'):.1f} | "
              f"{m('cudaMemcpyBatchAsync', 'd2d', 1):.2f} |")
