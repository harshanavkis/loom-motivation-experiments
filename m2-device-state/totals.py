#!/usr/bin/env python3
"""M2: transport state an IBGDA GPU keeps, for realistic configurations.

All constants are from NVSHMEM v3.8.0-0 / DeepEP a56d615 source (see results.md for file:line).
Struct sizes are from sizes.cpp (g++ against the cloned headers). Assumptions that are ours
(not code defaults) are marked ASSUMPTION.
"""

KiB, MiB = 1024, 1024 * 1024

# ---- NVSHMEM IBGDA constants (code) ----
WQEBB = 64            # MLX5_SEND_WQE_BB; ibgda.cpp:2185-2186
CQE = 64              # NVSHMEMI_IBGDA_CQE_SIZE; ibgda.cpp:56, 1438-1439
DBR = 8               # IBGDA_DBRSIZE; ibgda.cpp:66
IBUF_SLOT = 256       # NVSHMEMI_IBGDA_IBUF_SLOT_SIZE; nvshmem_common_ibgda.h:120
FETCH_SLOTS = 1024    # IBGDA_NUM_FETCH_SLOTS_PER_{RC,DCI} default; env_defs.h:196-201
ALIGN_SLACK = 64 * KiB - 1  # ibgda_gpu_mem_alloc: cudaMalloc(size + alignment - 1), IBGDA_GPAGE_SIZE; ibgda.cpp:1027-1031
QP_DESC, CQ_DESC, DCT_AV, KEY = 184, 72, 48, 16   # sizes.out
STATE_CONST = 8384    # nvshmemi_ibgda_device_state_t in __constant__ memory
MAX_CONST_DCTS, MAX_CONST_RKEYS = 128, 64
REGION_SLOTS = 4096   # NVSHMEMI_REGION_MAX_SLOTS_DEFAULT (3.8 only)


def pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def batch_rma_bytes(qp_count, slots=REGION_SLOTS):
    ch = lambda q: (q + 31) // 32
    return (ch(qp_count) + ch(ch(qp_count))) * slots * 4   # nvshmem_common_batch_rma_pending_qps.hpp:25-56


def per_qp(depth):
    """GPU-HBM bytes owned by ONE RC QP (default NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=gpumem)."""
    d = pow2(depth)
    parts = {
        "SQ WQE ring": d * WQEBB,
        "send-CQ ring": d * CQE,
        "internal buf (ibuf)": IBUF_SLOT * (FETCH_SLOTS + 1),
        "SQ doorbell record": DBR,
        "CQ doorbell record": DBR,
        "device QP descriptor (incl. mvars)": QP_DESC,
        "device CQ descriptor": CQ_DESC,
    }
    n_cudamalloc_aligned = 5  # ibuf, wq, sq-dbr, cq, cq-dbr each via ibgda_gpu_mem_alloc(.., 64 KiB)
    return parts, n_cudamalloc_aligned


def nvshmem_ibgda(npes, rc_per_pe, depth=1024, ndci=1, ndct=2, ndev=1, heap_bytes=2 * 1024 * MiB,
                  granularity=512 * MiB):
    """Totals per GPU (PE). ndev = NICs per PE (ASSUMPTION: 1, one rail NIC per GPU)."""
    parts, nalloc = per_qp(depth)
    per_qp_bytes = sum(parts.values())
    n_rc = rc_per_pe * ndev * (npes - 1)          # no loopback QP; ibgda.cpp:3440-3446
    n_rc_slots = rc_per_pe * ndev * npes          # descriptor arrays include self slot; ibgda.cpp:3627-3632
    n_qp = n_rc + ndci * ndev
    hbm = {
        "per-QP rings+ibuf+dbr (RC+DCI)": n_qp * (per_qp_bytes - QP_DESC - CQ_DESC),
        "QP descriptors": (n_rc_slots + ndci * ndev) * QP_DESC,
        "CQ descriptors": (n_rc_slots + ndci * ndev) * CQ_DESC,
        "DCT AV table (global part)": max(0, ndct * ndev * npes - MAX_CONST_DCTS) * DCT_AV,
        "rkey table (global part)": max(0, (heap_bytes // granularity) * npes * ndev - MAX_CONST_RKEYS) * KEY,
        "batch-RMA pending bitmap (3.8)": batch_rma_bytes(ndci * ndev + n_rc_slots),
    }
    total = sum(hbm.values())
    slack = n_rc * nalloc * ALIGN_SLACK
    return dict(npes=npes, rc_per_pe=rc_per_pe, n_rc=n_rc, n_qp=n_qp, uar_pages=n_qp,
                per_qp=per_qp_bytes, hbm=hbm, total=total, slack_upper=slack)


def fmt(b):
    return f"{b / MiB:8.2f} MiB" if b >= MiB else f"{b / KiB:8.1f} KiB"


def show(title, r):
    print(f"\n### {title}: PEs={r['npes']}  RC/peer={r['rc_per_pe']}  RC QPs={r['n_rc']}  "
          f"QPs total={r['n_qp']}  UAR doorbell pages mapped into GPU={r['uar_pages']}")
    for k, v in r["hbm"].items():
        print(f"   {k:34s} {v:>12,d} B  {fmt(v)}")
    print(f"   {'TOTAL GPU HBM (struct/ring bytes)':34s} {r['total']:>12,d} B  {fmt(r['total'])}")
    print(f"   + up to {fmt(r['slack_upper'])} cudaMalloc alignment slack (5 x (64KiB-1) per RC QP)")


# ---- NCCL GIN GDAKI (v2.32.3-1, vendored DOCA GPUNetIO) constants (code) ----
PAGE = 4096                   # priv_get_page_size() = sysconf(_SC_PAGESIZE); doca_gpunetio_high_level.cpp:56-57 (x86: 4 KiB, inferred)
DOCA_QP_DESC, GDAKI_CTX = 296, 88   # sizes_gdaki.out


def align(v, a=PAGE):
    return (v + a - 1) // a * a


def gdaki_per_qp(depth):
    d = pow2(depth)                        # next_power_of_two; doca_gpunetio_high_level.cpp:1400
    return {
        "SQ WQE ring": align(d * 64),                 # calc_qp_external_umem_size, :575-581
        "CQ ring (+8 B DBR)": align(d * 64 + 8),      # calc_cq_external_umem_size, :204-211
        "CQ DBR slab slice": align(8),                # dbr_size_per_qp, :1402
        "SQ DBR slab slice": align(8),                # :1402, :1452-1460
    }


def nccl_gdaki(nranks, contexts, peers_incl_self, depth=1024, nsignals=None, counters=0):
    """Per GPU. peers_incl_self = nranks/rankStride (FULL: nranks; RAIL: nodes). ASSUMPTION ginCommCount=1 (1 NIC/GPU)."""
    q_per_ctx = peers_incl_self + 1                  # +1 self responder QP; gin_host_gdaki.cc:618-627, 873-879
    if counters > 0:
        q_per_ctx *= 2                               # companion QPs; gin_host_gdaki.cc:624-625
    n_qp = contexts * q_per_ctx
    per_qp = sum(gdaki_per_qp(depth).values())
    nsig = nsignals if nsignals is not None else nranks + 4   # DeepEP: reqs.ginSignalCount = num_ranks + 2*2
    hbm = {
        "per-QP SQ/CQ rings + DBRs": n_qp * per_qp,
        "device QP descriptors (nranks slots/ctx)": contexts * nranks * DOCA_QP_DESC,  # doca_gpunetio.cpp:1144
        "GIN GDAKI GPU contexts": contexts * GDAKI_CTX,
        "signal table + signal shadows": 2 * contexts * nsig * 8,   # gin_host_gdaki.cc:674; dev_runtime.cc:1634
        "last_issued/visible_get": 2 * contexts * nranks * 8,       # gin_host_gdaki.cc:978-979
        "rkey arrays (signals, counters)": 2 * nranks * 4,
    }
    return dict(n_qp=n_qp, per_qp=per_qp, hbm=hbm, total=sum(hbm.values()), uar=n_qp)


if __name__ == "__main__":
    parts, _ = per_qp(1024)
    print("Per RC QP (depth 1024):")
    for k, v in parts.items():
        print(f"   {k:36s} {v:>9,d} B")
    print(f"   {'sum':36s} {sum(parts.values()):>9,d} B = {fmt(sum(parts.values()))}")
    print(f"   of which NIC-protocol state excl. ibuf: {sum(parts.values()) - parts['internal buf (ibuf)']:,d} B")

    print("\n## NVSHMEM defaults (RC_PER_PE=2, QP_DEPTH=1024, 1 DCI, 2 DCT, ASSUMPTION 1 NIC/PE)")
    for nodes in (2, 16, 32):
        show(f"NVSHMEM default, {nodes} nodes x 8 GPUs", nvshmem_ibgda(8 * nodes, 2))

    # DeepEP V1 (a56d615) low-latency: all EP ranks are NVSHMEM PEs; NUM_RC_PER_PE = num_experts // num_ranks
    # ASSUMPTION: 256 routed experts (DeepSeek-V3; docs/legacy.md:256) up to EP256; EP cannot exceed the
    # expert count, so EP512 / EP1024 assume 512 / 1024 experts (one per GPU)
    print("\n## DeepEP V1 low-latency (PEs = all ranks; RC/peer = experts / EP; depth 1024)")
    for nodes, experts in ((2, 256), (16, 256), (32, 256), (64, 512), (128, 1024)):
        ep = 8 * nodes
        show(f"DeepEP-V1 LL EP{ep} ({experts} experts)", nvshmem_ibgda(ep, experts // ep))

    # DeepEP V1 normal: NVSHMEM PEs = RDMA ranks (one per node, same local GPU index); default num_qps_per_rank=24.
    # At most 20 nodes = EP160 (LEGACY_NUM_MAX_RDMA_PEERS, csrc/kernels/legacy/compiled.cuh:6; checked in
    # csrc/legacy/buffer.hpp:113 unless low-latency mode). EP256 (32 nodes) was listed here before 2026-10-07:
    # DeepEP V1 refuses that configuration.
    print("\n## DeepEP V1 normal (PEs = nodes; RC/peer = 24 default; depth 1024; at most 20 nodes = EP160)")
    for nodes in (2, 16, 20):
        show(f"DeepEP-V1 normal {nodes} nodes (EP{8 * nodes})", nvshmem_ibgda(nodes, 24))

    print("\n## NCCL GIN GDAKI per-QP (depth 1024 as set by DeepEP; NCCL default 128)")
    for dep in (1024, 128):
        p = gdaki_per_qp(dep)
        print(f"   depth {dep}: " + ", ".join(f"{k}={v:,d}" for k, v in p.items()) + f"  sum={sum(p.values()):,d} B")

    print("\n## DeepEP V2.5 EPBuffer on NCCL GIN GDAKI (depth 1024; contexts = num_allocated_qps; no counters)")
    for label, ctx, rail in (("hybrid, CX-7 (129 ctx)", 129, True), ("hybrid, CX-8 (65 ctx)", 65, True),
                             ("direct (17 ctx)", 17, False)):
        # kNumMaxRanks = 1024 (deep_ep/include/deep_ep/common/compiled.cuh:74); EP512 / EP1024 for the default
        # CX-7 hybrid mode only (need 512 / 1024 experts)
        for nodes in ((2, 16, 32, 64, 128) if ctx == 129 else (2, 16, 32)):
            n = 8 * nodes
            r = nccl_gdaki(n, ctx, nodes if rail else n)
            print(f"\n### DeepEP-V2.5 {label}, EP{n}: QPs={r['n_qp']}  UAR doorbells mapped={r['uar']}")
            for k, v in r["hbm"].items():
                print(f"   {k:40s} {v:>12,d} B  {fmt(v)}")
            print(f"   {'TOTAL GPU HBM':40s} {r['total']:>12,d} B  {fmt(r['total'])}")
