#!/usr/bin/env python3
"""M3 (trace part, extension): per-kernel NCCL analysis of converted Chakra ET traces.

For every GPU NCCL node in every rank's ET (Llama3-70B, Mixtral-8x22B, Mixtral-8x7B) this script:
  * derives the collective type, process group, group size n (membership inferred from usage as in
    analyze_et.py) and scope (intra/inter node, node = rank // G);
  * takes bytes from comm_size. On Mixtral-8x7B, analyze_moe.py verified comm_size == input bytes of
    the kernel (Kineto "In msg nelems" x dtype size) for every NCCL kernel of ranks 0/2/3/6;
  * computes NCCL bus bytes (AG: (n-1)*in, RS: (n-1)/n*in, AR: 2(n-1)/n*in, a2a: (n-1)/n*in),
    per-peer message size (AG: in; RS, AR, a2a: in/n; a2a assumes equal splits), the kernel's hold
    time per MiB of input, and a lower bound on its duration from link peaks:
      intra-node: t_min = bus / NVLINK (GB/s per direction);
      inter-node, two NIC assumptions (the metadata gives only "InfiniBand-100Gbps, Switch"):
        A = one 100 Gb/s NIC per GPU,  B = one 100 Gb/s NIC per node, shared by the node's G GPUs.
        a2a:  remote = (n - m_loc)/n * in (m_loc = group members on this rank's node, incl. itself);
              A: t_min = remote / IB;  B: t_min = G * remote / IB (all G GPUs run the same a2a).
        ring AG/RS/AR spanning nodes: bus bytes cross each node boundary once per ring;
              A: t_min = bus / (m_loc * IB) (one ring per local NIC);
              B: t_min = (G / m_loc) * bus / IB (all G/m_loc groups of this kind cross concurrently).
      "unexplained" = 1 - sum(t_min)/sum(dur): an upper bound on the share of held time spent waiting,
      synchronising or paying latency rather than moving bytes at link peak.
  * size histograms of per-peer message sizes, and the share of ops and of NCCL time below 64 KiB/peer;
  * (MoE) expert GEMM time: GPU GEMM kernels whose CPU ancestor chain (ctrl_deps) contains
    LinearWithGradAccumulationAndAsyncCommunication (Megatron expert linears; the LM head also uses this
    class, see README).

Usage: python3 analyze_nccl_et.py <label> <G> <nvlink_GBps> <ib_GBps> <et files...>
Writes nccl_et_<label>.json next to this script.
"""
import sys, os, re, json, collections, statistics
from multiprocessing import Pool
import chakra_et

LAT_THR = 64 * 1024
BINS = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
BIN_LBL = ["<1K", "1K-4K", "4K-16K", "16K-64K", "64K-256K", "256K-1M", "1M-4M", "4M-16M", "16M-64M", ">=64M"]


def rank_of(p):
    return int(re.search(r"\.(\d+)\.et$", p).group(1))


def scan(path):
    _, nodes = chakra_et.read_et(path)
    rec = []; par = {}; gem = []
    for n in nodes:
        a = n["attr"]
        if a.get("is_cpu_op", True):
            par[n["id"]] = (n["name"], n["ctrl_deps"][0] if n["ctrl_deps"] else None)
            continue
        if "comm_type" in a:
            ct = chakra_et.COLL_TYPES.get(a["comm_type"], str(a["comm_type"]))
            rec.append((ct, str(a.get("pg_name")), a.get("comm_size", 0), n["dur"]))
        elif "gemm" in n["name"].lower():
            gem.append((n["ctrl_deps"][0] if n["ctrl_deps"] else None, n["dur"]))
    ex = 0; allg = 0; cache = {}

    def is_expert(j):
        seen = []; r = False
        for _ in range(16):
            if j is None or j not in par: break
            if j in cache: r = cache[j]; break
            seen.append(j)
            if "LinearWithGradAccumulationAndAsyncCommun" in par[j][0]: r = True; break
            j = par[j][1]
        for x in seen: cache[x] = r
        return r
    for p, d in gem:
        allg += d
        if is_expert(p): ex += d
    return dict(rank=rank_of(path), rec=rec, gemm_us=allg, expert_gemm_us=ex)


def bus_peer(ct, s, n):
    if n <= 1: return 0.0, s
    f = (n - 1) / n
    if ct == "ALL_GATHER": return s * (n - 1), s
    if ct == "REDUCE_SCATTER": return s * f, s / n
    if ct == "ALL_REDUCE": return 2 * s * f, s / n
    if ct == "ALL_TO_ALL": return s * f, s / n
    return s, s


def binof(x):
    for i, b in enumerate(BINS):
        if x < b: return i
    return len(BINS)


def med(x):
    return statistics.median(x) if x else float("nan")


def main():
    label, G, NV, IB = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    files = sys.argv[5:]
    with Pool(min(len(files), 32)) as p:
        R = sorted(p.map(scan, files), key=lambda r: r["rank"])
    members = collections.defaultdict(set)
    for r in R:
        for ct, pg, s, d in r["rec"]: members[pg].add(r["rank"])
    K = []  # pooled kernel records
    for r in R:
        rk = r["rank"]
        for ct, pg, s, d in r["rec"]:
            m = members[pg]; n = len(m); nodes = {x // G for x in m}
            scope = "intra" if len(nodes) == 1 else "inter"
            mloc = sum(1 for x in m if x // G == rk // G)
            bus, peer = bus_peer(ct, s, n)
            if scope == "intra":
                tA = tB = bus / (NV * 1e3)
            elif ct == "ALL_TO_ALL":
                remote = (n - mloc) / n * s
                tA = remote / (IB * 1e3); tB = G * remote / (IB * 1e3)
            else:
                tA = bus / (mloc * IB * 1e3); tB = (G / mloc) * bus / (IB * 1e3)
            K.append(dict(rank=rk, ct=ct, pg=pg, n=n, mloc=mloc, scope=scope, s=s, bus=bus, peer=peer, dur=d, tA=tA, tB=tB))
    print(f"# {label}: {len(R)} ranks, {len(K)} NCCL GPU kernels; G={G} GPUs/node (assumed); NVLink {NV} GB/s/dir; IB {IB} GB/s per NIC")
    tot = sum(k["dur"] for k in K)
    # ---- item 2: bandwidth / hold time per byte
    print("\n## Hold time per byte and link-peak lower bounds, pooled over ranks (type x scope x group size)")
    print("t_min: intra = bus/NVLink; inter A = one NIC per GPU, inter B = one NIC per node (see docstring)")
    print("type | scope | n | m_loc | kernels | % of NCCL time | sum dur ms | us per MiB in (sum/sum) | median us/MiB | "
          "(us/MiB only for kernels >= 1 MiB in; nan otherwise) | us per bus-MiB (sum/sum) | median busBW GB/s | max busBW GB/s | unexplained % (intra or A) | unexplained % (B)")
    g = collections.defaultdict(list)
    for k in K: g[(k["ct"], k["scope"], k["n"], k["mloc"])].append(k)
    summ = {}
    for key, L in sorted(g.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
        sd = sum(k["dur"] for k in L); ss = sum(k["s"] for k in L)
        L2 = [k for k in L if k["s"] > 0 and k["dur"] > 0]
        upm = [k["dur"] / (k["s"] / 2**20) for k in L2 if k["s"] >= 2**20]
        bw = [k["bus"] / k["dur"] / 1e3 for k in L2]
        uA = 1 - sum(k["tA"] for k in L) / sd if sd else float("nan")
        uB = 1 - sum(k["tB"] for k in L) / sd if sd else float("nan")
        print(f"{key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(L)} | {100*sd/tot:.2f} | {sd/1e3:.1f} | "
              f"{sd/(ss/2**20) if ss >= 2**20 * len(L) else float('nan'):.1f} | {med(upm):.1f} | {sd/(sum(k['bus'] for k in L)/2**20) if ss >= 2**20 * len(L) and sum(k['bus'] for k in L) > 0 else float('nan'):.1f} | {med(bw):.2f} | {max(bw) if bw else float('nan'):.2f} | "
              f"{100*uA:.1f} | {100*uB:.1f}")
        summ["|".join(map(str, key))] = dict(kernels=len(L), sum_ms=sd / 1e3, pct_nccl=100 * sd / tot,
                                             us_per_MiB=sd / (ss / 2**20) if ss else None, med_bw=med(bw),
                                             max_bw=max(bw) if bw else None, unexpl_A=100 * uA, unexpl_B=100 * uB)
    for sc in ("intra", "inter"):
        L = [k for k in K if k["scope"] == sc]
        if not L: continue
        sd = sum(k["dur"] for k in L)
        print(f"{sc}: {len(L)} kernels, {sd/1e3:.1f} ms ({100*sd/tot:.1f}% of NCCL time); unexplained at peak: "
              f"A/intra {100*(1-sum(k['tA'] for k in L)/sd):.1f}%, B {100*(1-sum(k['tB'] for k in L)/sd):.1f}%")
    # per-byte hold-time comparison: largest inter-node group vs. each intra-node group >= 1% of NCCL time
    big = [(key, L) for key, L in g.items() if sum(k["dur"] for k in L) >= 0.01 * tot and sum(k["bus"] for k in L) > 0]
    inter = [x for x in big if x[0][1] == "inter"]; intra = [x for x in big if x[0][1] == "intra"]
    if inter and intra:
        ki, Li = max(inter, key=lambda x: sum(k["dur"] for k in x[1]))
        def stats(L):
            L2 = [k for k in L if k["dur"] > 0 and k["bus"] > 0]
            return (sum(k["dur"] for k in L) / (sum(k["bus"] for k in L) / 2**20), med([k["bus"] / k["dur"] / 1e3 for k in L2]),
                    max(k["bus"] / k["dur"] / 1e3 for k in L2))
        a = stats(Li)
        print(f"\n## Per-byte hold time: inter {ki[0]} (n={ki[2]}) vs intra groups (ratio = inter/intra hold time per bus byte)")
        print("intra group | us/bus-MiB inter | us/bus-MiB intra | ratio (sum/sum) | median busBW ratio intra/inter | max busBW ratio intra/inter")
        for kk, L in sorted(intra, key=lambda x: -sum(k["dur"] for k in x[1])):
            b = stats(L)
            print(f"{kk[0]} n={kk[2]} | {a[0]:.1f} | {b[0]:.1f} | {a[0]/b[0]:.1f}x | {b[1]/a[1]:.1f}x | {b[2]/a[2]:.1f}x")
        print(f"link-peak ratio NVLink/IB-NIC = {NV/IB:.0f}x")
    # duration distribution vs. link-peak bound for the big groups
    print("\n## Duration percentiles vs. link-peak bounds (groups >= 5% of NCCL time)")
    print("type | scope | n | in MiB | p0 | p10 | p50 | p90 | p99 (us) | t_min intra/A us | t_min B us | % kernels faster than A bound | % faster than B bound")
    for key, L in sorted(g.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
        if sum(k["dur"] for k in L) < 0.05 * tot: continue
        for sz in sorted({k["s"] for k in L}):
            Ls = sorted((k for k in L if k["s"] == sz), key=lambda k: k["dur"])
            if len(Ls) < 10: continue
            d = [k["dur"] for k in Ls]; q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
            print(f"{key[0]} | {key[1]} | {key[2]} | {sz/2**20:.2f} | {q(0)} | {q(.1)} | {q(.5)} | {q(.9)} | {q(.99)} | "
                  f"{Ls[0]['tA']:.0f} | {Ls[0]['tB']:.0f} | {100*sum(1 for k in Ls if k['dur'] < k['tA'])/len(Ls):.1f} | "
                  f"{100*sum(1 for k in Ls if k['dur'] < k['tB'])/len(Ls):.1f}")
    # empirical: excess over fastest same-size kernel
    print("\n## Excess over the fastest kernel of the same (type, scope, n, bytes), pooled over ranks (empirical wait estimate)")
    print("type | scope | n | in MiB | kernels | min dur us | busBW at min GB/s | median dur us | sum dur ms | excess over min %")
    ex = collections.defaultdict(list)
    for k in K: ex[(k["ct"], k["scope"], k["n"], k["s"])].append(k)
    xs_t = 0.0
    for key, L in sorted(ex.items(), key=lambda x: -sum(k["dur"] for k in x[1])):
        mn = min(k["dur"] for k in L); sd = sum(k["dur"] for k in L); xs = sum(k["dur"] - mn for k in L); xs_t += xs
        if sd < 0.005 * tot: continue
        print(f"{key[0]} | {key[1]} | {key[2]} | {key[3]/2**20:.2f} | {len(L)} | {mn} | {L[0]['bus']/max(mn,1)/1e3:.2f} | "
              f"{med([k['dur'] for k in L]):.0f} | {sd/1e3:.1f} | {100*xs/sd:.1f}")
    print(f"(groups < 0.5% of NCCL time omitted) all NCCL: excess over min = {100*xs_t/tot:.1f}% of {tot/1e3:.1f} ms")
    mins = {key: min(k["dur"] for k in L) for key, L in ex.items()}
    per_rank = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))  # rank -> ct|scope -> [held, moving] ms
    for k in K:
        v = per_rank[k["rank"]][f"{k['ct']}|{k['scope']}"]
        v[0] += k["dur"] / 1e3; v[1] += mins[(k["ct"], k["scope"], k["n"], k["s"])] / 1e3
    # ---- item 3: per-peer size histograms
    print("\n## Per-peer message size histogram (count | % of NCCL time), pooled over ranks")
    print("per-peer: AG = in bytes (each rank's shard); RS, AR = in/n (ring chunk); a2a = in/n (equal splits); bcast = in")
    print("type | scope | n | " + " | ".join(BIN_LBL) + f" | ops <64KiB % | time <64KiB %")
    hist = {}
    for key, L in sorted(collections.OrderedDict(((k["ct"], k["scope"], k["n"]), None) for k in K).items()):
        L = [k for k in K if (k["ct"], k["scope"], k["n"]) == key]
        c = [0] * (len(BINS) + 1); t = [0.0] * (len(BINS) + 1)
        for k in L: b = binof(k["peer"]); c[b] += 1; t[b] += k["dur"]
        lo_c = sum(1 for k in L if k["peer"] < LAT_THR); lo_t = sum(k["dur"] for k in L if k["peer"] < LAT_THR)
        sd = sum(k["dur"] for k in L)
        print(f"{key[0]} | {key[1]} | {key[2]} | " + " | ".join(f"{c[i]} ({100*t[i]/tot:.1f})" if c[i] else "." for i in range(len(c))) +
              f" | {100*lo_c/len(L):.1f} | {100*lo_t/sd if sd else 0:.1f}")
        hist["|".join(map(str, key))] = dict(counts=c, time_pct_of_nccl=[100 * x / tot for x in t])
    lo = [k for k in K if k["peer"] < LAT_THR]
    print(f"\nALL: {len(lo)}/{len(K)} ops = {100*len(lo)/len(K):.1f}% of NCCL ops are < 64 KiB per peer; "
          f"they take {100*sum(k['dur'] for k in lo)/tot:.2f}% of NCCL kernel time")
    for sc in ("intra", "inter"):
        L = [k for k in K if k["scope"] == sc]
        if not L: continue
        l2 = [k for k in L if k["peer"] < LAT_THR]
        print(f"  {sc}: {100*len(l2)/len(L):.1f}% of ops, {100*sum(k['dur'] for k in l2)/max(sum(k['dur'] for k in L),1):.2f}% of {sc} NCCL time")
    # ---- expert GEMM
    eg = sorted(r["expert_gemm_us"] for r in R); ag = sorted(r["gemm_us"] for r in R)
    print(f"\n## GEMM time per rank (ms): all GEMM kernels median {med(ag)/1e3:.1f}; "
          f"GEMMs under LinearWithGradAccumulationAndAsyncCommunication median {med(eg)/1e3:.1f} (min {eg[0]/1e3:.1f}, max {eg[-1]/1e3:.1f})")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"nccl_et_{label}.json")
    json.dump(dict(label=label, G=G, nvlink=NV, ib=IB, groups=summ, hist=hist, bins=BIN_LBL,
                   lat_ops_pct=100 * len(lo) / len(K), lat_time_pct=100 * sum(k["dur"] for k in lo) / tot,
                   expert_gemm_ms={r["rank"]: r["expert_gemm_us"] / 1e3 for r in R},
                   gemm_ms={r["rank"]: r["gemm_us"] / 1e3 for r in R},
                   per_rank_held_moving_ms={rk: dict(v) for rk, v in per_rank.items()}), open(out, "w"), indent=1)
    print(f"(written {out})")


if __name__ == "__main__":
    main()
