#!/usr/bin/env python3
"""M1: classify every NCCL library source file (src/) as scale-up / scale-out / mixed / common
and count code lines (cloc: no blanks, no comments).

Usage:  python3 categorize.py <nccl-checkout> <label>
Needs:  cloc and universal-ctags on PATH, e.g.
        nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl v2.32.3-1
Writes: files_<label>.csv (per-file), split_<label>.csv (per-region of mixed files),
        prints summary tables + interface counts to stdout.

Two views:
  file view  : each file gets exactly one of up/out/mixed/common (MIXED files = SPLITS keys).
  split view : each MIXED file is split by function/struct (universal-ctags ranges) using the
               explicit name lists in SPLITS; unlisted regions fall back to the file's default.
Vendored third-party code (VENDORED) is categorized too but reported separately.
"""
import csv, json, os, re, subprocess, sys
from collections import defaultdict

UP, OUT, MIX, COM = "scale-up", "scale-out", "mixed", "common"
CATS = [UP, OUT, MIX, COM]

KEEP_LANGS = {"C", "C++", "C/C++ Header", "CUDA"}
# Kernel generators are counted (their outputs live only in the build dir); other scripts are not.
GENERATORS = {"src/device/generate.py": COM,              # instantiates all legacy coll kernels
              "src/device/symmetric/generate.py": UP,     # instantiates symmetric (LSA/multimem) kernels
              "src/collectives/device/gen_rules.sh": COM} # v2.18 equivalent of generate.py

# Third-party code shipped inside src/ (copied upstream libraries/headers).
VENDORED = [r"^src/include/nvtx3/",                       # NVTX library
            r"^src/transport/net_ib/gdaki/doca-gpunetio/",# DOCA GPUNetIO (BSD-3 copy)
            r"^src/transport/net_efa_gda/efa-dp-direct/", # Amazon EFA direct data path
            r"^src/transport/net_nd/nd(spi|def)\.h$",     # Microsoft NetworkDirect SPI headers
            r"^src/include/ibvcore\.h$",                  # rdma-core verbs.h copy
            r"^src/include/mlx5/mlx5dvcore\.h$"]          # rdma-core mlx5dv.h copy

# Ordered path rules: first match wins. Anything unmatched is COMMON.
RULES = [
    # ---------------- scale-out (network / RDMA / proxy-driven) ----------------
    (OUT, r"^src/transport/net[^/]*\.cc$"),               # net.cc, net_socket.cc, net_ib.cc(v2.18), stubs
    (OUT, r"^src/transport/(net_ib|net_socket|net_nd|net_efa_gda)/"),
    (OUT, r"^src/transport/coll_net\.cc$"),
    (OUT, r"^src/transport/rma_socket"),                  # socket RMA plugin backend
    (OUT, r"^src/net\.cc$"),                              # v2.18 net/collnet plugin loader
    (OUT, r"^src/misc/profiler\.cc$"),                    # v2.18 proxy (network progress) profiler
    (OUT, r"^src/plugin/(net|gin|rma)(\.cc|/)"),          # net/collnet/gin/rma plugin loaders + version shims
    (OUT, r"^src/include/plugin/(net|gin|rma)/"),
    (OUT, r"^src/include/plugin/nccl_(net|gin|rma)\.h$"),
    (OUT, r"^src/include/plugin/profiler/net_"),          # profiler event defs for net_ib/net_socket
    (OUT, r"^src/include/(nccl_net|net|coll_net|gin|ibvcore|ibvsymbols|ibvwrap|gdrwrap)\.h$"),
    (OUT, r"^src/include/(mlx5|gin)/"),
    (OUT, r"^src/misc/(ibvwrap|ibvsymbols|mlx5dvwrap|mlx5dvsymbols|gdrwrap)\.cc$"),
    (OUT, r"^src/gin/"),                                  # GIN host side (GDAKI / CPU proxy backends)
    (OUT, r"^src/nccl_device/gin_"),
    (OUT, r"^src/include/nccl_device/(gin|gin_barrier|gin_win_stub|net_device)\.h$"),
    (OUT, r"^src/include/nccl_device/impl/gin"),
    (OUT, r"^src/include/nccl_device/gin/"),
    (OUT, r"^src/device/network/"),                       # device-side unpack for net device offload
    (OUT, r"^src/device/symmetric/.*gin"),                # GIN (network) variants of symmetric kernels
    (OUT, r"^src/rma/rma_proxy"), (OUT, r"^src/include/rma/rma_proxy\.h$"),
    (OUT, r"^src/tuning/collnet\.cc$"), (OUT, r"^src/tuning/sym_model/gin\.cc$"),
    (OUT, r"^src/diagnostics/ib_write_bw"),
    # ---------------- scale-up (NVLink / PCIe P2P / SHM / NVLS / CE / LSA) ----------------
    (UP, r"^src/transport/(p2p|shm|nvls|nvls_ub|multicast|mc_arena)\.cc$"),
    (UP, r"^src/include/(p2p|shm|nvls_ub|multicast|mc_arena|ipcsocket|ce_coll|mnnvl|sym_kernels)\.h$"),
    (UP, r"^src/(ce_coll|mnnvl|sym_kernels|cft_dev_runtime)\.cc$"),
    (UP, r"^src/scheduler/symmetric_sched\.cc$"),
    (UP, r"^src/os/(linux|windows)_ipcsocket\.cc$"), (UP, r"^src/misc/ipcsocket\.cc$"),  # cuMem FD passing
    (UP, r"^src/nccl_device/(lsa_barrier|ll_a2a|cft_barrier)\.cc$"),
    (UP, r"^src/include/nccl_device/(lsa_barrier|ll_a2a|cft|cft_barrier|reduce_copy)\.h$"),
    (UP, r"^src/include/nccl_device/impl/(lsa_barrier|ll_a2a|cft|cft_barrier|multimem|reduce_copy)__"),
    (UP, r"^src/device/symmetric/"),                      # LSA / multimem symmetric kernels
    (UP, r"^src/rma/rma_ce\.cc$"), (UP, r"^src/include/rma/rma_ce\.h$"),
    (UP, r"^src/tuning/(nvls|ce_model)\.cc$"), (UP, r"^src/tuning/sym_model/lsa"),
    (UP, r"^src/diagnostics/(p2p\.|device/p2p)"),
    (UP, r"^src/enqueue/task_sched/ce_sched\.cc$"),
]

def names(*xs): return r"^(" + "|".join(xs) + r")\t"
DEVALGO = [(UP, r"\tstruct RunWorkColl<.*NCCL_ALGO_NVLS"),          # NVLS and NVLS_TREE
           (OUT, r"\tstruct RunWorkColl<.*NCCL_ALGO_COLLNET")]      # COLLNET_DIRECT / COLLNET_CHAIN
# MIXED files: path -> (default category for unlisted regions, [(category, regex on "name\tfirst line")])
SPLITS = {
    "src/proxy.cc": (COM, [  # default = proxy service thread / RPC used by every transport's setup
        (OUT, names("NeedProxy", "allocateArgs", "getOpIndex", "printProxyOp", "dumpProxyState",
                    "ncclProxyOpToArgs", "ProxyAppend", "ncclProxyPost", "ncclLocalOpAppend", "SaveProxy",
                    "getNvlsDenseLocalRank", "nvlsDensePeer", "ncclProxySaveOp", "ncclProxyComputeP2p",
                    "removeOp", "progressOps", "ncclProxyGetPostedOps", "ncclDumpProxyState",
                    "proxyCpusetOnceFunc", "setProxyThreadContext", "ncclProxyProgress", "ncclProxyStart",
                    "ncclProxyProgressCreate", "ncclProxyProgressDestroy", "proxyProgressInit", "proxyOpsFree",
                    "ncclProxyShmUnlink")),                       # progress engine: only NET/COLLNET progress fns
        (UP, names("ncclProxyCallBlockingUDS", "ncclProxyClientGetFdBlocking", "ncclProxyClientBatchQueryFdBlocking",
                   "ncclProxyClientQueryFdBlocking", "ncclProxyClientConvertFdBlocking", "proxyQueryFd",
                   "proxyGetFd", "proxyConvertFd", "proxyUDSRecvReq", "ncclProxyServiceUDS"))]),  # cuMem FD passing
    "src/include/proxy.h": (COM, [
        (OUT, names("ncclProxyOp", "ncclProxyEventHandle", "ncclProxySubArgs", "ncclProxyArgs", "ncclProxyOpsPool",
                    "ncclProxyOps", "ncclProxySharedP2p", "ncclProxyPeer", "ncclSharedNetComms",
                    "ncclProxyProgressState")),
        (UP, names("ncclIpcHdr"))]),
    "src/graph/topo.cc": (COM, [
        (OUT, names("ncclTopoGetMinNetBw", "ncclTopoAddNet", "ncclTopoAddGin", "ncclTopoAddRma", "ncclTopoAddNic",
                    "ncclTopoMakeVnic", "ncclTopoForceMerge", "ncclTopoAutoMerge", "ncclTopoGetVNicParent",
                    "ncclTopoMakeVNics", "ncclTopoPopulateNics", "ncclTopoUpdateVNics", "ncclTopoProcessNet",
                    "ncclTopoGetMergePolicy", "ncclTopoGetNetPropertiesSnapshot", "ncclTopoGetLocalNetCountByBw",
                    "getNetDevsPolicyOnce", "ncclTopoSortDevsByRailPlane", "ncclTopoGetLocalNetType",
                    "ncclTopoGetLocalNet", "ncclTopoGetNetCount", "ncclTopoFindLinkWidthRec")),
        (UP, names("ncclTopoAddNvLinks", "ncclTopoAddC2c", "ncclTopoGetNvsCount"))]),
    "src/graph/paths.cc": (COM, [
        (OUT, names("ncclTopoCheckGdr", "ncclTopoIsGdrAvail", "ncclTopoNeedFlush", "ncclTopoCheckNet",
                    "ncclTopoGetIntermediateRank", "ncclPxnDisable", "ncclTopoGetPxnRanks")),
        (UP, names("ncclTopoCheckP2p", "ncclTopoCheckMNNVL", "ncclTopoGetNvbGpus", "ncclTopoSplitNvLink",
                   "ncclTopoPathAllNVLink"))]),
    "src/graph/search.cc": (COM, [
        (OUT, names("ncclTopoSearchTryCollnetDirect", "ncclTopoPrefNetsGpuFirst", "ncclTopoPrefNetsChannelFirst",
                    "ncclTopoSelectNets", "ncclTopoSearchCheckNet", "ncclTopoSearchRecNet", "getNvlsNetDev",
                    "ncclTopoGetNetDev", "getNetIndex", "getNetPaths")),
        (UP, names("ncclTopoSearchTryNvls"))]),
    "src/graph/connect.cc": (COM, [(OUT, names("connectCollNet")), (UP, names("connectNvls"))]),
    "src/transport.cc": (COM, [(OUT, names("ncclTransportCollNetSetup", "ncclTransportCollNetCheck",
                                           "ncclTransportCollNetFree"))]),
    "src/init.cc": (COM, [(OUT, names("collNetTrySetup", "initGdrCopy"))]),
    "src/register/coll_reg.cc": (COM, [(OUT, names("isMloPartBufRdmaCapable")),
                                       (UP, names("ncclRegisterCollNvlsBuffers", "registerCheckP2PConnection"))]),
    "src/dev_runtime.cc": (COM, [  # device-API runtime: windows/teams shared by LSA and GIN
        (OUT, names("symMemoryRegisterGin", "symMemoryRegisterRma", "symMemoryDeregisterRma", "symWindowInitGin",
                    "ncclGinResourcesRequested", "ncclDevCommGdakiDump", "ncclDevCommEfaGdaDump",
                    "ncclDevCommProxyDump", "ncclDevrGetRmaWin")),
        (UP, names("computeLsaSize", "ncclDevrIsOneLsaTeam", "symHasCountedCftMemory", "symMemorySetAccessForVASegment",
                   "symMemoryExportSegmentHandle", "symMemoryImportAndMapSegmentHandle",
                   "symMemoryImportAndMapSegmentsForRank", "symMemoryMapLsaTeam", "symBindTeamMemory",
                   "symUnbindTeamMemory", "symMemoryUnmapLsaRank", "ncclDevrWorldToLsaRank", "ncclDevrGetLsaRankPtr",
                   "ncclDevrGetLsaTeamPtrMC", "ncclGetMultimemDevicePointer", "ncclGetLsaMultimemDevicePointer",
                   "ncclGetLsaDevicePointer", "ncclGetPeerDevicePointer"))]),
}
for p in ["src/device/all_reduce.h", "src/device/all_gather.h", "src/device/reduce_scatter.h",
          "src/collectives/device/all_reduce.h", "src/collectives/device/all_gather.h",
          "src/collectives/device/reduce_scatter.h"]:
    SPLITS[p] = (COM, DEVALGO)

def file_category(path):
    if path in GENERATORS: return GENERATORS[path]
    if path in SPLITS: return MIX
    for cat, rx in RULES:
        if re.search(rx, path): return cat
    return COM

def is_vendored(path): return any(re.search(rx, path) for rx in VENDORED)

def code_line_mask(text):
    """True for lines that contain code outside comments (same notion as cloc)."""
    mask, in_block = [], False
    for line in text.split("\n"):
        i, n, code = 0, len(line), False
        while i < n:
            if in_block:
                j = line.find("*/", i)
                if j < 0: i = n
                else: in_block, i = False, j + 2
                continue
            c = line[i]
            if line.startswith("//", i): break
            if line.startswith("/*", i): in_block, i = True, i + 2; continue
            if c in "\"'":
                code = True; q = c; i += 1
                while i < n and line[i] != q: i += 2 if line[i] == "\\" else 1
                i += 1; continue
            if not c.isspace(): code = True
            i += 1
        mask.append(code)
    return mask

def ctags(path):
    out = subprocess.run(["ctags", "--language-force=C++", "--kinds-C++=fsc", "--fields=+neK",
                          "--output-format=json", "-f", "-", path], capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]

def split_file(root, rel, cloc_code):
    default, rules = SPLITS[rel]
    path = os.path.join(root, rel)
    text = open(path, errors="replace").read()
    lines = text.split("\n")
    owner = [default] * len(lines)
    regions = []
    hits = []
    for t in ctags(path):
        if "end" not in t: continue
        key = f"{t['name']}\t{lines[t['line'] - 1].strip()}"
        for cat, rx in rules:
            if re.search(rx, key):
                hits.append((t["end"] - t["line"], t["line"], t["end"], cat, t["name"])); break
    for _, s, e, cat, name in sorted(hits, reverse=True):   # outer first, inner overrides
        for k in range(s - 1, e): owner[k] = cat
    mask = code_line_mask(text)
    raw = defaultdict(int)
    for m, o in zip(mask, owner):
        if m: raw[o] += 1
    total_raw = sum(raw.values())
    # scale to cloc's number for the file (differences are a handful of lines at most)
    scaled = {c: round(v * cloc_code / total_raw) for c, v in raw.items()} if total_raw else {default: cloc_code}
    scaled[default] = scaled.get(default, 0) + cloc_code - sum(scaled.values())
    for _, s, e, cat, name in sorted(hits, key=lambda h: h[1]):
        regions.append((rel, name, s, e, cat, sum(1 for k in range(s - 1, e) if mask[k] and owner[k] == cat)))
    return scaled, regions, total_raw

def run_cloc(root):
    out = subprocess.run(["cloc", "--by-file", "--csv", "--quiet", "src"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for r in csv.reader(out.splitlines()):
        if len(r) < 5 or r[0] in ("language", "SUM") or not r[1].startswith("src/"): continue
        rows.append((r[1], r[0], int(r[4])))
    return rows

def pct(a, b): return f"{100.0 * a / b:.1f}%" if b else "-"

# ---------------------------------------------------------------- interface counting
def interfaces(root):
    res = []
    incs = []
    for d, _, fs in os.walk(os.path.join(root, "src/include")):
        if "nvtx3" in d: continue
        incs += [os.path.join(d, f) for f in fs if f.endswith(".h")]
    rx = re.compile(r"typedef\s+struct\s*\w*\s*\{(.*?)\}\s*(nccl(Net|CollNet|Gin|Rma|Tuner|Profiler|Env)_v(\d+)_t)\s*;", re.S)
    apis = defaultdict(dict)
    for f in incs:
        txt = open(f, errors="replace").read()
        for m in rx.finditer(txt):
            apis[m.group(3)][int(m.group(4))] = len(re.findall(r"\(\s*\*\s*\w+\s*\)\s*\(", m.group(1)))
    for api in ["Net", "CollNet", "Gin", "Rma", "Tuner", "Profiler", "Env"]:
        if api in apis:
            vs = sorted(apis[api])
            res.append((f"plugin API ncclGin/Net/..: nccl{api}_vX", f"versions={vs}",
                        f"fn-ptrs newest(v{vs[-1]})={apis[api][vs[-1]]}", f"fn-ptrs oldest(v{vs[0]})={apis[api][vs[0]]}"))
    th = open(os.path.join(root, "src/include/transport.h")).read()
    tc = re.search(r"struct ncclTransportComm \{(.*?)\n\};", th, re.S)
    nfp = len(re.findall(r"\(\s*\*\s*\w+\s*\)\s*\(", tc.group(1))) if tc else 0
    tt = open(os.path.join(root, "src/transport.cc")).read()
    arr = re.search(r"ncclTransports\[NTRANSPORTS\]\s*=\s*\{(.*?)\}", tt, re.S)
    trs = re.findall(r"&(\w+)", arr.group(1)) if arr else []
    res.append(("internal ncclTransport contract", f"ncclTransportComm fn-ptrs={nfp} (x2 send/recv) + canConnect",
                f"built-in transports={trs}", ""))
    # public device-API declarations (non-impl headers)
    dd = os.path.join(root, "src/include/nccl_device")
    if os.path.isdir(dd):
        per = defaultdict(int)
        for f in sorted(os.listdir(dd)):
            if not f.endswith(".h"): continue
            rel = "src/include/nccl_device/" + f
            n = len(re.findall(r"NCCL_(?:HOST_)?DEVICE_INLINE[^;{]*\(", open(os.path.join(dd, f), errors="replace").read()))
            per[file_category(rel)] += n
        res.append(("device API public decls (nccl_device/*.h, NCCL_*DEVICE_INLINE)",
                    ", ".join(f"{k}={v}" for k, v in sorted(per.items())), "", ""))
    return res

def main():
    root, label = sys.argv[1], sys.argv[2]
    outdir = os.path.dirname(os.path.abspath(__file__))
    rows = run_cloc(root)
    files, regions = [], []
    for path, lang, code in rows:
        if lang not in KEEP_LANGS and path not in GENERATORS: continue
        cat, vend = file_category(path), is_vendored(path)
        split = {cat: code}
        if cat == MIX:
            split, regs, _ = split_file(root, path, code)
            regions += regs
        files.append(dict(path=path, language=lang, code=code, category=cat, vendored=vend,
                          up=split.get(UP, 0), out=split.get(OUT, 0), common=split.get(COM, 0)))
    files.sort(key=lambda f: (f["category"], -f["code"]))
    with open(os.path.join(outdir, f"files_{label}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "language", "code_lines", "category", "vendored",
                    "split_scale_up", "split_scale_out", "split_common"])
        for f in files:
            w.writerow([f["path"], f["language"], f["code"], f["category"], int(f["vendored"]),
                        f["up"] if f["category"] == MIX else "", f["out"] if f["category"] == MIX else "",
                        f["common"] if f["category"] == MIX else ""])
    with open(os.path.join(outdir, f"split_{label}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "region", "start_line", "end_line", "category", "code_lines_raw"])
        for r in regions: w.writerow(r)

    print(f"# {label}  ({root})")
    for title, sel in [("NCCL-authored (excl. vendored)", lambda f: not f["vendored"]),
                       ("vendored third-party only", lambda f: f["vendored"]),
                       ("everything in src/ (incl. vendored)", lambda f: True)]:
        fs = [f for f in files if sel(f)]
        tot = sum(f["code"] for f in fs)
        print(f"\n## {title}: {len(fs)} files, {tot} code lines")
        print("| category | files | file view LoC | % | split view LoC | % |")
        print("|---|---|---|---|---|---|")
        for c in CATS:
            n = sum(1 for f in fs if f["category"] == c)
            fv = sum(f["code"] for f in fs if f["category"] == c)
            key = {UP: "up", OUT: "out", COM: "common"}.get(c)
            sv = sum(f[key] if f["category"] == MIX else (f["code"] if f["category"] == c else 0)
                     for f in fs) if key else 0
            print(f"| {c} | {n} | {fv} | {pct(fv, tot)} | {sv if key else '-'} | {pct(sv, tot) if key else '-'} |")
        # bounds for mixed: all-to-common vs all-to-transport
        mix = [f for f in fs if f["category"] == MIX]
        print(f"mixed files split into: up={sum(f['up'] for f in mix)} out={sum(f['out'] for f in mix)} "
              f"common={sum(f['common'] for f in mix)}")
    # sensitivity of the transport-specific share (NCCL-authored only)
    fs = [f for f in files if not f["vendored"]]
    tot = sum(f["code"] for f in fs)
    fv = lambda c: sum(f["code"] for f in fs if f["category"] == c)
    sp = lambda k, sel=lambda f: True: sum(
        f[k] if f["category"] == MIX else (f["code"] if f["category"] == {"up": UP, "out": OUT, "common": COM}[k] else 0)
        for f in fs if sel(f))
    proxy_com = sum(f["common"] for f in fs if f["path"] in ("src/proxy.cc", "src/include/proxy.h"))
    noras = lambda f: not f["path"].startswith(("src/ras/", "src/diagnostics"))
    tot_noras = sum(f["code"] for f in fs if noras(f))
    print("\n## sensitivity: transport-specific (up+out) share of NCCL-authored code")
    print(f"  A file view, mixed counted as common : up={fv(UP)} out={fv(OUT)} -> {pct(fv(UP) + fv(OUT), tot)} of {tot}")
    print(f"  B split view (default)               : up={sp('up')} out={sp('out')} -> {pct(sp('up') + sp('out'), tot)} of {tot}")
    print(f"  C split view + whole proxy = out     : up={sp('up')} out={sp('out') + proxy_com} -> "
          f"{pct(sp('up') + sp('out') + proxy_com, tot)} of {tot}")
    print(f"  D split view, excl. src/ras + src/diagnostics*: up={sp('up', noras)} out={sp('out', noras)} -> "
          f"{pct(sp('up', noras) + sp('out', noras), tot_noras)} of {tot_noras}")
    print("\n## top files per category (NCCL-authored)")
    for c in CATS:
        fs = [f for f in files if f["category"] == c and not f["vendored"]]
        print(f"- {c}: " + ", ".join(f"{f['path'][4:]}({f['code']})" for f in fs[:15]))
    print("\n## per-directory LoC by category (NCCL-authored, split view)")
    agg = defaultdict(lambda: defaultdict(int))
    for f in files:
        if f["vendored"]: continue
        d = "/".join(f["path"].split("/")[:2]) if f["path"].count("/") > 1 else "src/*.cc"
        if f["category"] == MIX:
            for k, c in (("up", UP), ("out", OUT), ("common", COM)): agg[d][c] += f[k]
        else: agg[d][f["category"]] += f["code"]
    for d in sorted(agg, key=lambda d: -sum(agg[d].values())):
        print(f"  {d:28s} up={agg[d][UP]:6d} out={agg[d][OUT]:6d} common={agg[d][COM]:6d}")
    print("\n## interfaces")
    for r in interfaces(root): print("  " + " | ".join(x for x in r if x))

if __name__ == "__main__":
    main()
