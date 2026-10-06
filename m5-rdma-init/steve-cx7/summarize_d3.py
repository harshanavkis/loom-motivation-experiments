#!/usr/bin/env python3
"""Summarize dispatch_bd.cu --d3 runs: compute lost per dispatch (median over reps).

Usage: summarize_d3.py d3_*.csv [--csv out.csv]

lost    filler work lost per dispatch, SM-us: (1 - filler rate with dispatches / without) x SMs x
        window / dispatches
resid   the dispatch's own CTA residency per dispatch, CTA-us (D2's SM time, same run)
span    first dispatch CTA start -> signal issued, us
With --d3-filler gemm-*: the filler's TFLOP/s alone and with dispatches, the throughput lost, and
the dispatch's launch -> done time (CUDA events, including the wait for SMs).
"""
import csv
import statistics
import sys
from collections import defaultdict


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    if out in files:
        files.remove(out)
    runs = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(l for l in fh if l.startswith(("d3,path", "d3,"))):
                if row["path"] == "path":
                    continue
                path = row["path"] if row["qp"] == "none" else f'{row["path"]}-{row["qp"]}{row["nqp"]}'
                filler = row.get("filler", "fma")
                key = (filler, path, int(row["H"]), int(row["tokens"]), int(row["ctas"]), int(row["period_us"]))
                cols = ["lost_sm_us_per_dispatch", "resid_cta_us_per_dispatch", "span_us_per_dispatch", "lost_frac", "filler_cta_us"]
                if "tflops_alone" in row:
                    cols += ["tflops_alone", "tflops_with", "dispatch_event_med_us"]
                runs[key].append({k: float(row[k]) for k in cols})
    med = {k: {c: statistics.median(r[c] for r in v) for c in v[0]} | {"reps": len(v)} for k, v in runs.items()}
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filler", "path", "H", "tokens", "ctas", "period_us", "reps", "lost_sm_us", "resid_cta_us", "span_us",
                        "lost_frac", "tflops_alone", "tflops_with", "dispatch_event_us"])
            for k in sorted(med):
                m = med[k]
                w.writerow(list(k) + [m["reps"], round(m["lost_sm_us_per_dispatch"], 1), round(m["resid_cta_us_per_dispatch"], 1),
                                      round(m["span_us_per_dispatch"], 2), round(m["lost_frac"], 5), round(m.get("tflops_alone", 0), 1),
                                      round(m.get("tflops_with", 0), 1), round(m.get("dispatch_event_med_us", 0), 1)])
    paths = sorted({k[1] for k in med}, key=lambda p: (p != "local", p))
    for F, H, T, ctas, P in sorted({(k[0],) + k[2:] for k in med}):
        have = [p for p in paths if (F, p, H, T, ctas, P) in med]
        ms = {p: med[(F, p, H, T, ctas, P)] for p in have}
        print(f"\n=== filler {F}: H={H} B, tokens={T}, {ctas} CTAs, a dispatch every {P} us")
        print(f"{'':36s}" + "".join(f"{p:>16s}" for p in have))
        rows = [("lost_sm_us_per_dispatch", "compute lost per dispatch (SM-us)"),
                ("resid_cta_us_per_dispatch", "dispatch residency (CTA-us)"),
                ("span_us_per_dispatch", "dispatch span (us)")]
        if F != "fma":
            rows += [("tflops_alone", "GEMM TFLOP/s alone"), ("tflops_with", "GEMM TFLOP/s with dispatches"),
                     ("lost_pct", "GEMM throughput lost (%)"), ("dispatch_event_med_us", "dispatch launch -> done (us)")]
            for m in ms.values():
                m["lost_pct"] = 100 * m["lost_frac"]
        for c, lab in rows:
            print(f"  {lab:34s}" + "".join(f"{ms[p][c]:16.1f}" for p in have))


if __name__ == "__main__":
    main()
