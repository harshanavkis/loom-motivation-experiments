#!/usr/bin/env python3
"""M3 (trace part): communication share in converted Chakra ET traces (MLCommons Chakra Open Trace
Library: Llama3-70B, Mixtral-8x22B, Mixtral-8x7B).

The converted .et files carry, per GPU node: kernel name, stream, duration_micros, and for NCCL
kernels comm_type / comm_size (bytes) / pg_name. They carry NO start timestamps (all 0), NO grid/block
dims and NO process-group rank lists. So this script:
  1. scans every rank's ET (in parallel) and records which pg_names each rank issues collectives on;
     membership(pg) := set of ranks that issue collectives on that pg_name (inference, see README);
  2. classifies each pg as intra-node (all members on one node, node = rank // GPUS_PER_NODE) or
     inter-node;
  3. aggregates per rank: comm kernel time by collective type x {intra, inter}, total GPU kernel time,
     per-stream busy time, and the ET wall window (finish_ts - start_ts, ms).

Usage: python3 analyze_et.py <label> <gpus_per_node> <et files...>
"""
import sys, os, re, json, collections
from multiprocessing import Pool
import chakra_et

GPU_TYPES = (4, 5, 6, 7)  # COMP, SEND, RECV, COLL


def rank_of(path):
    return int(re.search(r"\.(\d+)\.et$", path).group(1))


def scan(path):
    meta, nodes = chakra_et.read_et(path)
    r = {"rank": rank_of(path), "path": path,
         "window_ms": int(meta["finish_ts"]) - int(meta["start_ts"]),
         "comm": collections.defaultdict(lambda: [0, 0, 0]),  # (ctype, pg) -> [n, dur_us, bytes]
         "stream_busy": collections.Counter(), "gpu_total_us": 0, "comm_total_us": 0,
         "n_gpu": 0, "kernel_names": collections.Counter()}
    for n in nodes:
        a = n["attr"]
        if a.get("is_cpu_op", True) or n["type"] not in GPU_TYPES:
            continue
        r["n_gpu"] += 1
        d = n["dur"]
        r["gpu_total_us"] += d
        r["stream_busy"][a.get("stream")] += d
        if "comm_type" in a:
            ct = chakra_et.COLL_TYPES.get(a["comm_type"], str(a["comm_type"]))
            if n["type"] in (5, 6):
                ct = chakra_et.NODE_TYPES[n["type"]]
            # NCCL SendRecv kernels issued by all_to_all are tagged comm_type ALL_TO_ALL
            e = r["comm"][(ct, str(a.get("pg_name")))]
            e[0] += 1; e[1] += d; e[2] += a.get("comm_size", 0)
            r["comm_total_us"] += d
            r["kernel_names"][(ct, n["name"].split("(")[0])] += 1
    r["comm"] = {f"{k[0]}|{k[1]}": v for k, v in r["comm"].items()}
    r["stream_busy"] = dict(r["stream_busy"])
    r["kernel_names"] = {f"{k[0]}|{k[1]}": v for k, v in r["kernel_names"].items()}
    return r


def main():
    label, G, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    with Pool(min(len(files), 32)) as p:
        res = sorted(p.map(scan, files), key=lambda r: r["rank"])
    ranks = [r["rank"] for r in res]
    world = len(res)
    members = collections.defaultdict(set)
    for r in res:
        for k in r["comm"]:
            members[k.split("|")[1]].add(r["rank"])
    pgclass = {}
    print(f"# {label}: {world} rank ETs, ranks {min(ranks)}..{max(ranks)}; ASSUMED gpus/node = {G}")
    print(f"# nodes implied by rank count = {world / G:g}")
    print("\n## Process groups (membership inferred = ranks issuing collectives on the pg_name)")
    print("pg | size | members | nodes | class | collective types")
    for pg in sorted(members, key=lambda x: int(x) if x.isdigit() else 1e9):
        m = sorted(members[pg]); nodes = sorted({x // G for x in m})
        cls = "intra" if len(nodes) == 1 else "inter"
        pgclass[pg] = cls
        types = sorted({k.split("|")[0] for r in res for k in r["comm"] if k.split("|")[1] == pg})
        ms = str(m) if len(m) <= 16 else f"{m[:8]}...({len(m)})"
        print(f"{pg} | {len(m)} | {ms} | {nodes} | {cls} | {','.join(types)}")
    # distinct groups per class (e.g. how many different TP groups)
    print("\n## NCCL kernel names seen (all ranks, counts)")
    kn = collections.Counter()
    for r in res:
        kn.update(r["kernel_names"])
    for k, v in sorted(kn.items()):
        print(f"{k} : {v}")

    print("\n## Per-rank summary (times in ms; comm = sum of NCCL kernel durations)")
    print("rank | ET window | GPU kernel sum | max stream busy (stream) | stream-7 busy | comm sum | comm intra | comm inter | comm/GPU-kernel-sum | comm/ET-window")
    summ = []
    for r in res:
        intra = sum(v[1] for k, v in r["comm"].items() if pgclass[k.split("|")[1]] == "intra")
        inter = sum(v[1] for k, v in r["comm"].items() if pgclass[k.split("|")[1]] == "inter")
        sb = r["stream_busy"]; ms_ = max(sb, key=sb.get)
        summ.append((r, intra, inter))
        print(f"{r['rank']} | {r['window_ms']} | {r['gpu_total_us']/1e3:.1f} | {sb[ms_]/1e3:.1f} ({ms_}) | "
              f"{sb.get(7, 0)/1e3:.1f} | {r['comm_total_us']/1e3:.1f} | {intra/1e3:.1f} | {inter/1e3:.1f} | "
              f"{r['comm_total_us']/max(r['gpu_total_us'],1):.3f} | {r['comm_total_us']/1e3/r['window_ms']:.3f}")
    fr = sorted(r["comm_total_us"] / r["gpu_total_us"] for r in res)
    print(f"\ncomm/GPU-kernel-sum across ranks: min {fr[0]:.3f} median {fr[len(fr)//2]:.3f} max {fr[-1]:.3f}")
    fi = sorted(inter / r["comm_total_us"] for r, intra, inter in summ)
    print(f"inter-node share of comm time across ranks: min {fi[0]:.3f} median {fi[len(fi)//2]:.3f} max {fi[-1]:.3f}")

    reps = [res[0]["rank"], res[len(res) // 2 + (1 if len(res) > 8 else 0)]["rank"]]
    reps = sorted(set(reps))
    for rr in reps:
        r = next(x for x in res if x["rank"] == rr)
        print(f"\n## Rank {rr}: comm by collective type x class (ms, % of this rank's comm, % of GPU kernel sum)")
        print("type | class | pg(s) | count | time ms | % comm | % GPU-kernel-sum | MB issued")
        agg = collections.defaultdict(lambda: [0, 0, 0, set()])
        for k, v in r["comm"].items():
            ct, pg = k.split("|")
            a = agg[(ct, pgclass[pg])]; a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3].add(pg)
        for (ct, cls), a in sorted(agg.items(), key=lambda x: -x[1][1]):
            print(f"{ct} | {cls} | {','.join(sorted(a[3]))} | {a[0]} | {a[1]/1e3:.1f} | "
                  f"{100*a[1]/r['comm_total_us']:.1f} | {100*a[1]/r['gpu_total_us']:.1f} | {a[2]/1e6:.1f}")
        print("stream busy (ms):", {s: round(v / 1e3, 1) for s, v in sorted(r["stream_busy"].items(), key=lambda x: -x[1])})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"et_summary_{label}.json")
    json.dump({"label": label, "gpus_per_node_assumed": G, "pg_members": {k: sorted(v) for k, v in members.items()},
               "pg_class": pgclass, "ranks": res}, open(out, "w"), indent=1, default=str)
    print(f"\n(raw per-rank aggregates written to {out})")


if __name__ == "__main__":
    main()
