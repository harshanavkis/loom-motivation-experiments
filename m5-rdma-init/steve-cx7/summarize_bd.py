#!/usr/bin/env python3
"""Summarize dispatch_bd.cu runs: medians per (path, H, load, tokens, ctas), then the breakdown.

Usage: summarize_bd.py dispatch_bd_*.csv [--csv out.csv]

Critical path (D1), us from the sender's first CTA start until the receiver sees the signal:
  send     the send loop: every token routed, loaded, (staged,) posted or stored (loop_end)
  drain    every CTA's own writes complete: store completion + fence (local), completion
           wait (flush), nothing (ordered)                                (drain_end - loop_end)
  signal   last CTA found, signal issued: release stores (local), RDMA atomic posts (remote)
                                                                          (signal - drain_end)
  flight   signal issued -> seen by the receiver                          (seen - signal)
SM time (D2), summed over CTAs (CTA-us = how long CTAs stay resident on their SMs):
  cta_loop / cta_drain / cta_tail, and the loop split by warp activity (warp-us):
  route, load, smem, stage, post, store.
"""
import csv
import statistics
import sys
from collections import defaultdict

NUM = ["msgs", "event_us", "loop_end_us", "post_last_us", "arr_first_us", "arr_last_us", "drain_end_us",
       "signal_us", "seen_us", "oneway_med_us", "oneway_p90_us", "missing", "cta_loop_us", "cta_drain_us",
       "cta_tail_us", "w_route_us", "w_load_us", "w_smem_us", "w_stage_us", "w_post_us", "w_store_us"]
PATHS = ["local", "ordered", "flush"]   # plus QP variants, e.g. "flush-warp24", "ordered-dest24"


def load(files):
    runs = defaultdict(list)
    for f in files:
        with open(f) as fh:
            # NVSHMEM prints its WARN lines to stdout too
            for row in csv.DictReader(l for l in fh if l.strip() and not l.startswith(("#", "WARN"))):
                path = row["path"] if row.get("qp", "none") == "none" else f'{row["path"]}-{row["qp"]}{row["nqp"]}'
                key = (path, int(row["H"]), int(row["load"]), int(row["tokens"]), int(row["ctas"]))
                runs[key].append({k: float(row[k]) for k in NUM})
    med = {}
    for key, rs in runs.items():
        m = {k: statistics.median(r[k] for r in rs) for k in NUM}
        m["runs"] = len(rs)
        m["missing_max"] = max(r["missing"] for r in rs)
        m["seen_p90_us"] = sorted(r["seen_us"] for r in rs)[int(len(rs) * 0.9)]
        # segments from per-run differences, then medians (robust to run-to-run shifts)
        for name, a, b in [("send", None, "loop_end_us"), ("drain", "loop_end_us", "drain_end_us"),
                           ("signal", "drain_end_us", "signal_us"), ("flight", "signal_us", "seen_us")]:
            m[name] = statistics.median(r[b] - (r[a] if a else 0) for r in rs)
        m["cta_total_us"] = statistics.median(r["cta_loop_us"] + r["cta_drain_us"] + r["cta_tail_us"] for r in rs)
        med[key] = m
    return med


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    if out in args:
        args.remove(out)
    med = load(args)
    if out:
        with open(out, "w", newline="") as fh:
            cols = ["path", "H", "load", "tokens", "ctas", "runs", "send", "drain", "signal", "flight", "seen_us",
                    "seen_p90_us", "arr_last_us", "oneway_med_us", "cta_loop_us", "cta_drain_us", "cta_tail_us",
                    "cta_total_us", "w_route_us", "w_load_us", "w_smem_us", "w_stage_us", "w_post_us", "w_store_us",
                    "missing_max"]
            w = csv.writer(fh)
            w.writerow(cols)
            for key in sorted(med):
                m = med[key]
                w.writerow(list(key) + [m["runs"]] + [round(m[c], 2) for c in cols[6:]])
    groups = sorted({(k[1], k[2], k[3], k[4]) for k in med})
    for H, ld, T, ctas in groups:
        have = sorted({k[0] for k in med if k[1:] == (H, ld, T, ctas)}, key=lambda p: (p != "local", p))
        if not have:
            continue
        ms = {p: med[(p, H, ld, T, ctas)] for p in have}
        print(f"\n=== H={H} B, tokens={T} ({ms[have[0]]['msgs']:.0f} messages), {ctas} CTAs, "
              f"{'GEMM running' if ld else 'idle'}; medians of {min(m['runs'] for m in ms.values())} runs")
        hdr = f"{'':34s}" + "".join(f"{p:>10s}" for p in have)
        if "local" in ms:
            hdr += "".join(f"{p + '-local':>15s}" for p in have if p != "local")
        print(hdr)

        def line(label, f, fmt="{:10.1f}"):
            s = f"{label:34s}" + "".join(fmt.format(f(ms[p])) for p in have)
            if "local" in ms:
                s += "".join(f"{f(ms[p]) - f(ms['local']):15.1f}" for p in have if p != "local")
            print(s)

        print("critical path (us)")
        line("  send loop", lambda m: m["send"])
        line("  own writes complete", lambda m: m["drain"])
        line("  signal issued", lambda m: m["signal"])
        line("  signal flight", lambda m: m["flight"])
        line("  = receiver sees signal", lambda m: m["seen_us"])
        line("  (last data seen)", lambda m: m["arr_last_us"])
        line("  (one-way per message, median)", lambda m: m["oneway_med_us"])
        print("SM time (CTA-us, summed over CTAs)")
        line("  send loop", lambda m: m["cta_loop_us"])
        line("  own writes complete", lambda m: m["cta_drain_us"])
        line("  signal + exit", lambda m: m["cta_tail_us"])
        line("  = total", lambda m: m["cta_total_us"])
        print("send loop by warp activity (warp-us, summed over warps)")
        for a in ["route", "load", "smem", "stage", "post", "store"]:
            line(f"  {a}", lambda m, a=a: m[f"w_{a}_us"])
        bad = [p for p in have if ms[p]["missing_max"] > 0]
        if bad:
            print(f"  WARNING: data missing when the signal was seen: {bad}")


if __name__ == "__main__":
    main()
