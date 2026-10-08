#!/usr/bin/env python3
"""
Registered (pinned, NIC-visible) GPU memory that DeepEP allocates for its
communication buffers, split by purpose.

Python 3, stdlib only. Every formula is a port of DeepEP source; the comment
above each one cites file:line. Paths are relative to ../src/:
  V1   = DeepEP-v1-last  (commit a56d615, last NVSHMEM/IBGDA version)
  V2.5 = DeepEP          (commit 93eb6eb, NCCL GIN backend)
  NCCL = nccl            (v2.32.3-1, commit 12df1a1)
  NVSHMEM = nvshmem      (v3.8.0-0, commit 270759e)

Categories per GPU:
  (a) send/staging   = buffer whose payload the NIC reads by DMA as the source
                       of an RDMA write (exists only because the peer is
                       reached through a NIC; a load/store fabric stores
                       straight into the peer's receive buffer, which DeepEP
                       itself does for NVLink peers)
  (b) RDMA receive   = landing zone written by remote NICs
  (c) NVLink side    = receive buffers written by NVLink peers (load/store)
  (d) signal/meta    = counters, flags, tails, barrier words
"""

import math

MiB = 1 << 20
GiB = 1 << 30


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def fmt(n):
    if n >= GiB:
        return f"{n / GiB:8.2f} GiB"
    return f"{n / MiB:8.1f} MiB"


# ---------------------------------------------------------------------------
# Model configuration (DeepSeek-V3, DeepEP published settings)
# ---------------------------------------------------------------------------
HIDDEN = 7168
TOPK = 8
GPUS_PER_NODE = 8  # NVLink domain = node


def num_experts_for(ep):
    # 256 routed experts (DeepSeek-V3); one expert per GPU beyond EP256
    return max(256, ep)


# ===========================================================================
# DeepEP V1 (NVSHMEM)
# ===========================================================================
# DeepEP-v1-last/csrc/kernels/legacy/compiled.cuh:5,6,7,9
V1_NUM_MAX_NVL_PEERS = 8
V1_NUM_MAX_RDMA_PEERS = 20
V1_NUM_WORKSPACE_BYTES = 32 * 1024 * 1024
V1_NUM_BUFFER_ALIGNMENT_BYTES = 128
# DeepEP-v1-last/deep_ep/include/deep_ep/common/compiled.cuh:54-55,72
# EP_NUM_TOPK_IDX_BITS defaults to 64 -> topk_idx_t = int64_t
SIZEOF_TOPK_IDX_T = 8
# DeepEP-v1-last/csrc/kernels/legacy/internode.cu:22-23,43-44
# struct SourceMeta { int src_rdma_rank, is_token_in_nvl_rank_bits; } -> 8 B
SOURCE_META_BYTES = 8
SIZEOF_INT, SIZEOF_FLOAT, SIZEOF_INT4 = 4, 4, 16
SIZEOF_BF16, SIZEOF_BF162 = 2, 4
# DeepEP-v1-last/deep_ep/buffers/legacy.py:31 (class default) and
# docs/legacy.md:132 + tests/legacy/test_internode.py:324 (24 SMs)
V1_NUM_SMS = 24
# NVSHMEM heap grows in chunks of NVSHMEM_CUMEM_GRANULARITY, which DeepEP sets
# to 2**29 (DeepEP-v1-last/deep_ep/buffers/legacy.py, `NVSHMEM_CUMEM_GRANULARITY`);
# nvshmem/src/host/mem/mem_heap.cpp:1448-1449 rounds each physical backing up
# to that granularity, and nvshmem/src/host/mem/heap_registration.cpp:496
# registers it with the remote (IB) transport.
NVSHMEM_CUMEM_GRANULARITY = 1 << 29


class V1Config:
    """Port of DeepEP-v1-last/csrc/legacy/config.hpp:24-51 (struct Config)."""

    def __init__(self, num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv):
        self.num_sms = num_sms
        self.num_max_nvl_chunked_send_tokens = nvl_send
        self.num_max_nvl_chunked_recv_tokens = nvl_recv
        self.num_max_rdma_chunked_send_tokens = rdma_send
        # config.hpp:47  recv tokens are aligned up to a multiple of send tokens
        self.num_max_rdma_chunked_recv_tokens = align(rdma_recv, rdma_send)
        assert rdma_send <= self.num_max_rdma_chunked_recv_tokens // 2  # config.hpp:50

    def nvl_hint(self, hidden_bytes, num_ranks):
        """config.hpp:53-73 get_nvl_buffer_size_hint, returned split by term."""
        kNumMaxTopK, kNumMaxScales = 128, 128  # config.hpp:56-57
        num_rdma_ranks = max(num_ranks // V1_NUM_MAX_NVL_PEERS, 1)  # config.hpp:60
        num_nvl_ranks = min(num_ranks, V1_NUM_MAX_NVL_PEERS)  # config.hpp:61
        num_channels = self.num_sms // 2  # config.hpp:62
        r = self.num_max_nvl_chunked_recv_tokens
        meta = num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * SIZEOF_INT  # :65
        data = num_channels * num_nvl_ranks * r * hidden_bytes  # :66
        data += num_channels * num_nvl_ranks * r * SOURCE_META_BYTES  # :67
        data += num_channels * num_nvl_ranks * r * kNumMaxTopK * SIZEOF_TOPK_IDX_T  # :68
        data += num_channels * num_nvl_ranks * r * kNumMaxTopK * SIZEOF_FLOAT  # :69
        data += num_channels * num_nvl_ranks * r * kNumMaxScales * SIZEOF_FLOAT  # :70
        total = align(meta + data, 128)  # :71
        return dict(total=total, data=data, meta=meta, pad=total - meta - data)

    def rdma_hint(self, hidden_bytes, num_ranks):
        """config.hpp:75-100 get_rdma_buffer_size_hint, returned split by term.

        The RDMA region is carved as decoupled SymBuffers (send half + recv
        half): DeepEP-v1-last/csrc/kernels/legacy/buffer.cuh:106-113 and
        internode.cu:526-530, hence the `* 2` on each per-token term.
        """
        if num_ranks <= V1_NUM_MAX_NVL_PEERS:  # config.hpp:77-78
            return dict(total=0, send=0, recv=0, meta=0, slack=0, pad=0)
        kNumMaxTopK, kNumMaxScales = 128, 128  # config.hpp:82-83
        assert num_ranks % V1_NUM_MAX_NVL_PEERS == 0 and self.num_sms % 2 == 0
        num_rdma_ranks = num_ranks // V1_NUM_MAX_NVL_PEERS  # config.hpp:86
        num_channels = self.num_sms // 2  # config.hpp:87
        r = self.num_max_rdma_chunked_recv_tokens
        meta = num_channels * num_rdma_ranks * (V1_NUM_MAX_NVL_PEERS * 2 + 2) * 2 * SIZEOF_INT  # :90
        per_token = (hidden_bytes  # :91
                     + SOURCE_META_BYTES  # :92
                     + kNumMaxTopK * SIZEOF_TOPK_IDX_T  # :93
                     + kNumMaxTopK * SIZEOF_FLOAT  # :94
                     + kNumMaxScales * SIZEOF_FLOAT)  # :95
        data_both_halves = num_channels * num_rdma_ranks * r * per_token * 2
        # :96 one int4 per token per half; used by the uint64 head/tail
        # SymBuffer<uint64_t,false> (internode.cu:529-530) and int4 alignment
        slack = num_channels * num_rdma_ranks * r * SIZEOF_INT4 * 2
        raw = meta + data_both_halves + slack
        total = align(raw, 128)  # :97
        return dict(total=total, send=data_both_halves // 2, recv=data_both_halves // 2,
                    meta=meta, slack=slack, pad=total - raw)


def v1_dispatch_config(num_ranks, num_sms):
    """DeepEP-v1-last/deep_ep/buffers/legacy.py:245-258 get_dispatch_config."""
    m = {8: (6, 256, 6, 128), 16: (36, 288, 20, 128), 32: (32, 288, 8, 128),
         64: (32, 288, 8, 128), 128: (20, 560, 12, 128), 160: (28, 720, 12, 128)}
    return V1Config(num_sms, *m[num_ranks])


def v1_combine_config(num_ranks, num_sms):
    """DeepEP-v1-last/deep_ep/buffers/legacy.py:275-288 get_combine_config."""
    m = {8: (4, 256, 6, 128), 16: (4, 288, 12, 128), 32: (1, 288, 8, 128),
         64: (1, 288, 8, 128), 128: (1, 560, 8, 128), 160: (2, 720, 8, 128)}
    return V1Config(num_sms, *m[num_ranks])


def v1_normal(ep, num_sms=V1_NUM_SMS):
    """V1 normal (high-throughput) mode, sized as in docs/legacy.md:136-147
    (max of the dispatch and combine config hints)."""
    # docs/legacy.md:151-153 get_hidden_bytes = hidden * max(elem_size, 2):
    # sized for BF16 even with FP8 dispatch (combine is BF16)
    hidden_bytes = HIDDEN * 2
    cfgs = (v1_dispatch_config(ep, num_sms), v1_combine_config(ep, num_sms))
    nvl = max((c.nvl_hint(hidden_bytes, ep) for c in cfgs), key=lambda d: d["total"])
    rdma = max((c.rdma_hint(hidden_bytes, ep) for c in cfgs), key=lambda d: d["total"])
    # csrc/legacy/buffer.hpp:101-103,134-135: NVL buffer + 8 barrier ints +
    # 8 buffer ptrs + 8 barrier-signal ptrs (cudaMalloc / fabric, CUDA IPC;
    # NOT registered with the NIC)
    nvl_extra = V1_NUM_MAX_NVL_PEERS * (4 + 8 + 8)
    # csrc/legacy/buffer.hpp:268 nvshmem::alloc(num_rdma_bytes) -> NVSHMEM
    # symmetric heap, registered with the NIC
    nic = rdma["total"]
    return dict(
        a_send=rdma["send"],
        b_rdma_recv=rdma["recv"],
        c_nvl=nvl["data"] + nvl["pad"],
        d_signal=rdma["meta"] + rdma["slack"] + rdma["pad"] + nvl["meta"] + nvl_extra,
        total=rdma["total"] + nvl["total"] + nvl_extra,
        nic=nic,
        nic_heap=align(nic, NVSHMEM_CUMEM_GRANULARITY) if nic else 0,
        detail=dict(rdma=rdma, nvl=nvl),
    )


def v1_low_latency(ep, tokens):
    """Port of DeepEP-v1-last/csrc/legacy/config.hpp:122-188
    (LowLatencyLayout + get_low_latency_rdma_size_hint)."""
    num_experts = num_experts_for(ep)
    hidden = HIDDEN
    num_scales = hidden // 128  # config.hpp:132
    # config.hpp:143 (sized with BF16, see deep_ep/buffers/legacy.py:178)
    per_dispatch_msg = SIZEOF_INT4 + max(hidden * SIZEOF_BF16, hidden + num_scales * SIZEOF_FLOAT)
    # config.hpp:144
    per_combine_msg = num_scales * SIZEOF_BF162 + hidden * SIZEOF_BF16
    # config.hpp:147-151 send buffer, odd/even x2
    dispatch_send = tokens * per_dispatch_msg
    combine_send = num_experts * tokens * per_combine_msg
    send = max(dispatch_send, combine_send)
    # config.hpp:155-159 receive buffer, odd/even x2
    dispatch_recv = num_experts * tokens * per_dispatch_msg
    combine_recv = num_experts * tokens * per_combine_msg
    recv = max(dispatch_recv, combine_recv)
    # config.hpp:162-166 signalling buffer, odd/even x2
    sig = align(num_experts * SIZEOF_INT, 128)
    total_bytes = 2 * send + 2 * recv + 2 * sig
    # config.hpp:185-187 (note: always adds one 128 B block)
    hint = ((total_bytes + V1_NUM_BUFFER_ALIGNMENT_BYTES) // V1_NUM_BUFFER_ALIGNMENT_BYTES) \
        * V1_NUM_BUFFER_ALIGNMENT_BYTES
    # docs/legacy.md:250-256: Buffer(group, 0, num_rdma_bytes, low_latency_mode=True)
    # -> no NVL buffer; whole hint is nvshmem::alloc'ed (buffer.hpp:268) -> NIC registered
    return dict(
        a_send=2 * send,
        b_rdma_recv=2 * recv,
        c_nvl=0,
        d_signal=2 * sig + (hint - total_bytes),
        total=hint,
        nic=hint,
        nic_heap=align(hint, NVSHMEM_CUMEM_GRANULARITY),
        detail=dict(per_dispatch_msg=per_dispatch_msg, per_combine_msg=per_combine_msg,
                    dispatch_send=dispatch_send, combine_send=combine_send,
                    dispatch_recv=dispatch_recv, combine_recv=combine_recv, sig=sig,
                    num_experts=num_experts),
    )


# ===========================================================================
# DeepEP V2.5 (NCCL GIN)
# ===========================================================================
# DeepEP/deep_ep/include/deep_ep/common/compiled.cuh:67-69,74,82
kNumTMAAlignmentBytes = 32
kNumRDMAAlignmentBytes = 64
kNumAllocationAlignmentBytes = 2 * 1024 * 1024
kNumMaxRanks = 1024
kNumMaxSignalBytes = 16 * 1024 * 1024
# DeepEP/csrc/buffers/ep.hpp:41-43
kNumMaxChannels = 8 * 160
SIZEOF_SF_PACK = 4  # compiled.cuh:59-62 union sf_pack_t {float; int}
SIZEOF_MBARRIER = 8


def token_bytes(num_hidden_bytes, num_sf_bytes, num_topk, with_metadata, with_mbarrier=False):
    """DeepEP/deep_ep/include/deep_ep/layout/ep/token.cuh:18-41 TokenLayout::get_num_bytes."""
    # token.cuh:25-26
    num_metadata_bytes = num_topk * (4 + 4) + ((1 + num_topk) * 4 if with_metadata else 0)
    # token.cuh:34-37
    num_data = align(align(num_hidden_bytes, kNumTMAAlignmentBytes)
                     + align(num_sf_bytes, kNumTMAAlignmentBytes)
                     + align(num_metadata_bytes, kNumTMAAlignmentBytes), kNumRDMAAlignmentBytes)
    # token.cuh:38-39
    return num_data + (align(SIZEOF_MBARRIER, kNumTMAAlignmentBytes) if with_mbarrier else 0)


def buf(tok, num_ranks, tokens_per_rank):
    """token.cuh:107-114 BufferLayout::get_num_bytes = tokens_per_rank * token * ranks."""
    return tokens_per_rank * tok * num_ranks


def dispatch_token(hidden, elem_size, num_sf_packs, num_topk):
    """DeepEP/csrc/kernels/ep/dispatch.hpp:26-29 get_dispatch_token_layout."""
    return token_bytes(hidden * elem_size, num_sf_packs * SIZEOF_SF_PACK, num_topk, True)


def combine_token(hidden, num_topk):
    """DeepEP/csrc/kernels/ep/combine.hpp:19-22 get_combine_token_layout (BF16)."""
    return token_bytes(hidden * SIZEOF_BF16, 0, num_topk, False)


def v2_dispatch(T, hidden, num_sf_packs, num_topk, elem, S, U, nvlink):
    """DeepEP/csrc/buffers/ep.hpp:183-211 get_dispatch_buffer_size, split.
    S = num_scaleout_ranks, U = num_scaleup_ranks."""
    tok = dispatch_token(hidden, elem, num_sf_packs, num_topk)
    if S == 1:
        # ep.hpp:193-197 direct: send buffer only if scale-up is NOT NVLink
        send = buf(tok, 0 if nvlink else 1, T)
        recv = buf(tok, S * U, T)
        if nvlink:
            return dict(a_send=0, b_rdma_recv=0, c_nvl=recv, tok=tok)
        # direct RDMA mode: one flat recv buffer, NVLink peers in the node
        # also write into it (dispatch.cuh:367-373); counted as RDMA receive
        return dict(a_send=send, b_rdma_recv=recv, c_nvl=0, tok=tok)
    # ep.hpp:200-209 hybrid
    scaleup_recv = buf(tok, U, S * T)  # ep.hpp:200-201
    scaleout_send = buf(tok, 1, T)  # ep.hpp:202-203
    scaleout_recv = buf(tok, S, T + kNumMaxChannels)  # ep.hpp:204-206
    return dict(a_send=scaleout_send, b_rdma_recv=scaleout_recv, c_nvl=scaleup_recv, tok=tok)


def v2_combine(T, hidden, num_topk, S, U, nvlink, multi_red):
    """DeepEP/csrc/buffers/ep.hpp:213-247 get_combine_buffer_size, split."""
    tok = combine_token(hidden, num_topk)
    n = S * U
    if S == 1:
        # ep.hpp:222-230 direct
        n_layout = min(n, num_topk) if multi_red else num_topk
        send = buf(tok, 0 if nvlink else n, T * (1 if multi_red else num_topk))
        recv = buf(tok, n_layout, T)
        if nvlink:
            return dict(a_send=0, b_rdma_recv=0, c_nvl=recv, tok=tok)
        return dict(a_send=send, b_rdma_recv=recv, c_nvl=0, tok=tok)
    # ep.hpp:233-245 hybrid
    up_layout = min(U, num_topk) if multi_red else num_topk  # ep.hpp:233
    out_layout = min(S, num_topk) if multi_red else num_topk  # ep.hpp:234
    scaleup_recv = buf(tok, up_layout, S * T)  # ep.hpp:235-236
    scaleout_recv = buf(tok, out_layout, T)  # ep.hpp:237-238
    scaleout_send = buf(tok, 1 if multi_red else num_topk, S * (T + kNumMaxChannels))  # ep.hpp:239-242
    return dict(a_send=scaleout_send, b_rdma_recv=scaleout_recv, c_nvl=scaleup_recv, tok=tok)


def sizeof_ep_signals():
    """sizeof(EPSignals), DeepEP/deep_ep/include/deep_ep/layout/ep/workspace.cuh:9-22
    (alignas(64)); BarrierSignals layout/common/barrier.cuh:11-13 (alignas(64))."""
    kNumMaxExperts, kNumMaxChannelsSig = 2048, 1280  # workspace.cuh:10,12
    off = 0
    fields = [
        ("barrier_signals", 64, align(8 + 2 * 4, 64), "nvl"),
        ("notify_reduction", 8, 8 * (kNumMaxRanks + kNumMaxExperts), "other"),
        ("scaleup_rank_expert_count", 8, 8 * 2 * (kNumMaxRanks + kNumMaxExperts), "nvl"),
        ("scaleup_atomic_sender_count", 4, 4 * kNumMaxRanks, "nvl"),
        ("scaleout_rank_expert_count", 4, 4 * 2 * (kNumMaxRanks + kNumMaxExperts), "rdma"),
        ("scaleout_channel_signaled_tail", 8, 8 * kNumMaxChannelsSig * kNumMaxRanks, "rdma"),
        ("channel_scaleup_tail", 4, 4 * kNumMaxChannelsSig * kNumMaxRanks, "nvl"),
        ("lb_chunk_idx_counter", 4, 4, "other"),
    ]
    parts = {}
    for name, al, size, kind in fields:
        off = align(off, al) + size
        parts[name] = (size, kind)
    total = align(off, 64)
    assert total <= kNumMaxSignalBytes  # workspace.cuh:23
    return total, parts


def v2_ep(ep, T, use_fp8=True, num_topk=TOPK, hybrid=True, multi_red=True,
          force_flat_nvlink=False, num_lb_bytes=0):
    """DeepEP/csrc/buffers/ep.hpp:249-283 calculate_buffer_size, plus the
    workspace and LB region that EPBuffer puts in the same NCCL window."""
    # ep.hpp:260 num_topk == 0 -> sized as 32
    num_topk = 32 if num_topk == 0 else num_topk
    # csrc/kernels/comm/context.cpp:52-63 physical / logical domain
    R, N = ep // GPUS_PER_NODE if ep >= GPUS_PER_NODE else 1, min(ep, GPUS_PER_NODE)
    if force_flat_nvlink:
        # Counterfactual: all EP ranks in one load/store (LSA) domain
        R, N = 1, ep
    S, U = (R, N) if hybrid else (1, R * N)
    nvlink = (U == N)  # ep.hpp:265
    elem = 1 if use_fp8 else 2  # ep.hpp:268
    sf = ceil_div(HIDDEN, 32) if use_fp8 else 0  # ep.hpp:269
    d = v2_dispatch(T, HIDDEN, sf, num_topk, elem, S, U, nvlink)
    c = v2_combine(T, HIDDEN, num_topk, S, U, nvlink, multi_red)
    d_tot = d["a_send"] + d["b_rdma_recv"] + d["c_nvl"]
    c_tot = c["a_send"] + c["b_rdma_recv"] + c["c_nvl"]
    buffer_bytes = align(max(d_tot, c_tot), kNumAllocationAlignmentBytes)  # ep.hpp:282
    sig, sig_parts = sizeof_ep_signals()
    # ep.hpp:70-71 workspace aligned to 2 MiB, ep.hpp:73-77 window =
    # workspace + buffer + LB; context.cpp:186-199 one ncclMemAlloc'ed window
    # registered with ncclCommWindowRegister(..., NCCL_WIN_STRICT_ORDERING),
    # which without NCCL_WIN_GIN_ONLY means "register all" (NCCL
    # src/dev_runtime.cc:47-60) incl. GIN/NIC (src/dev_runtime.cc:639-660,870-871)
    workspace = align(sig, kNumAllocationAlignmentBytes)
    total = workspace + buffer_bytes + num_lb_bytes
    dom = d if d_tot >= c_tot else c
    # alternative allocation if the RDMA send/staging regions did not exist
    no_send = align(max(d_tot - d["a_send"], c_tot - c["a_send"]), kNumAllocationAlignmentBytes)
    # ... and if additionally the 2-hop RDMA landing zone did not exist
    no_send_no_hop = align(max(d["c_nvl"], c["c_nvl"]), kNumAllocationAlignmentBytes) \
        if S > 1 else no_send
    return dict(
        a_send=dom["a_send"], b_rdma_recv=dom["b_rdma_recv"], c_nvl=dom["c_nvl"],
        d_signal=workspace, pad=buffer_bytes - max(d_tot, c_tot),
        total=total, nic=total if not force_flat_nvlink else None,
        buffer_bytes=buffer_bytes, dispatch=d, combine=c, d_tot=d_tot, c_tot=c_tot,
        dominant="combine" if c_tot > d_tot else "dispatch",
        S=S, U=U, nvlink=nvlink, sig=sig, sig_parts=sig_parts,
        rdma_only_strict=buffer_bytes - no_send,
        rdma_only_broad=buffer_bytes - no_send_no_hop,
    )


# ===========================================================================
# Output
# ===========================================================================
EPS = [16, 128, 256, 512, 1024]


def header(title):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def row_hdr():
    print(f"{'EP':>5} {'(a) send/staging':>17} {'(b) RDMA recv':>15} {'(c) NVLink side':>16}"
          f" {'(d) signal/meta':>16} {'total':>13} {'NIC-registered':>15} {'only-RDMA (a)':>14}")


def row(ep, r, nic=None, only=None):
    nic = r["nic"] if nic is None else nic
    only = r["a_send"] if only is None else only
    print(f"{ep:>5} {fmt(r['a_send']):>17} {fmt(r['b_rdma_recv']):>15} {fmt(r['c_nvl']):>16}"
          f" {fmt(r['d_signal']):>16} {fmt(r['total']):>13} {fmt(nic):>15} {fmt(only):>14}")


def main():
    print("DeepEP communication-buffer memory per GPU (bytes requested from the allocator)")
    print(f"hidden={HIDDEN}, top-{TOPK}, experts=max(256, EP), {GPUS_PER_NODE} GPUs per NVLink domain,"
          " FP8 dispatch, BF16 combine")
    print("MiB = 2^20 B, GiB = 2^30 B. 'only-RDMA (a)' = send/staging bytes the NIC reads as RDMA source;")
    print("a load/store fabric stores directly into the peer's receive buffer (as DeepEP does for NVLink peers).")

    # ---------------- V1 low latency ----------------
    header("DeepEP V1 low-latency (decode), NVSHMEM IBGDA, 128 tokens/rank "
           "(config.hpp:122-188; docs/legacy.md:246-256)")
    row_hdr()
    for ep in [8] + EPS:
        r = v1_low_latency(ep, 128)
        row(ep, r)
    r = v1_low_latency(16, 128)["detail"]
    print(f"  per-msg bytes: dispatch {r['per_dispatch_msg']} B (BF16-sized, config.hpp:143),"
          f" combine {r['per_combine_msg']} B (config.hpp:144)")
    print("  send buffer (each of 2 odd/even) = max(dispatch_send, combine_send); experts=256:")
    print(f"    dispatch_send = 128 * {r['per_dispatch_msg']} = {r['dispatch_send']:,} B;"
          f" combine_send = 256*128*{r['per_combine_msg']} = {r['combine_send']:,} B")
    print(f"  recv buffer (each of 2) = max(dispatch_recv {r['dispatch_recv']:,}, combine_recv {r['combine_recv']:,}) B")
    for ep in (16, 256, 512, 1024):
        x = v1_low_latency(ep, 128)
        print(f"  EP{ep}: 2*send = {x['a_send']:,} B ({x['a_send'] / 1e6:.1f} MB), 2*recv = {x['b_rdma_recv']:,} B,"
              f" signals+pad = {x['d_signal']:,} B, hint = {x['total']:,} B ({x['total'] / 1e9:.3f} GB);"
              f" NVSHMEM heap backing (512 MiB chunks) >= {fmt(x['nic_heap']).strip()}")
    print("  experts = 256 for EP<=256 so V1 LL size is independent of EP there; it scales with experts beyond.")
    print("  Of (a), 2*dispatch_send = "
          f"{2 * v1_low_latency(16, 128)['detail']['dispatch_send']:,} B is the FP8-cast copy that is also"
          " read for NVLink peers (internode_ll.cu:259-270);")
    print("  the rest is combine staging used only for non-P2P peers (internode_ll.cu:843-853,910-911).")

    # ---------------- V1 normal ----------------
    header(f"DeepEP V1 normal (training/prefill), NVSHMEM, num_sms={V1_NUM_SMS}, hint = max(dispatch, combine config)"
           " (config.hpp:53-100; legacy.py:233-290; docs/legacy.md:136-147)")
    print("  (sizes do not depend on tokens/rank; ring buffers of num_max_*_chunked_recv_tokens per channel)")
    row_hdr()
    for ep in (8, 16, 128, 160):
        r = v1_normal(ep)
        row(ep, r)
    for ep in (16, 128):
        r = v1_normal(ep)
        x = r["detail"]["rdma"]
        print(f"  EP{ep}: RDMA hint {x['total']:,} B = send half {x['send']:,} + recv half {x['recv']:,}"
              f" + meta {x['meta']:,} + int4 slack {x['slack']:,} + pad {x['pad']:,};"
              f" NVL hint {r['detail']['nvl']['total']:,} B; NVSHMEM heap backing >= {fmt(r['nic_heap']).strip()}")
    print("  (a)/(b) per-token terms use kNumMaxTopK=128, kNumMaxScales=128, int64 topk idx (config.hpp:82-83):")
    print("  real tokens use far less of each slot, but the allocation is what is pinned/registered.")
    print("  V1 normal supports at most 20 nodes (compiled.cuh:6; legacy.py config_map up to 160): EP256+ n/a.")
    print("  NVL buffer: cudaMalloc + CUDA IPC (buffer.hpp:132-136), NOT NIC-registered. Workspace 32 MiB cudaMalloc"
          " (buffer.hpp:149) not counted.")
    r20 = [v1_normal(ep, 20) for ep in (16, 128)]
    print(f"  with Buffer.num_sms default 20 (legacy.py:31): EP16 total {fmt(r20[0]['total']).strip()},"
          f" NIC {fmt(r20[0]['nic']).strip()}, (a) {fmt(r20[0]['a_send']).strip()};"
          f" EP128 total {fmt(r20[1]['total']).strip()}, NIC {fmt(r20[1]['nic']).strip()},"
          f" (a) {fmt(r20[1]['a_send']).strip()}")

    # ---------------- V2.5 ----------------
    sig, parts = sizeof_ep_signals()
    for T, label in ((128, "decode"), (4096, "training/prefill")):
        header(f"DeepEP V2.5 EPBuffer, NCCL GIN, hybrid mode (default), allow_multiple_reduction, FP8 dispatch,"
               f" num_topk={TOPK}, {T} tokens/rank ({label})")
        print("  (a)/(b)/(c) = the phase whose layout sets the buffer size (dispatch and combine reuse the same"
              " bytes, ep.hpp:282); (d) = 16 MiB workspace")
        row_hdr()
        for ep in [8] + EPS:
            r = v2_ep(ep, T)
            row(ep, r)
        print("  per phase (dispatch | combine), S=scale-out ranks (nodes), U=scale-up ranks (8):")
        for ep in [8] + EPS:
            r = v2_ep(ep, T)
            d, c = r["dispatch"], r["combine"]
            print(f"   EP{ep:<5} S={r['S']:<3} dispatch[tok {d['tok']} B]: send {fmt(d['a_send'])}"
                  f" rdma-recv {fmt(d['b_rdma_recv'])} nvl-recv {fmt(d['c_nvl'])} = {fmt(r['d_tot'])}"
                  f" | combine[tok {c['tok']} B]: send {fmt(c['a_send'])} rdma-recv {fmt(c['b_rdma_recv'])}"
                  f" nvl-recv {fmt(c['c_nvl'])} = {fmt(r['c_tot'])}")
        print("  allocation attributable to RDMA (buffer minus the same formula with the regions removed):")
        for ep in [8] + EPS:
            r = v2_ep(ep, T)
            flat = v2_ep(ep, T, force_flat_nvlink=True)
            print(f"   EP{ep:<5} buffer {fmt(r['buffer_bytes'])}; strict (send/staging only) {fmt(r['rdma_only_strict'])};"
                  f" broad (+ 2-hop RDMA landing zone) {fmt(r['rdma_only_broad'])};"
                  f" flat load/store counterfactual (direct, is_scaleup_nvlink) buffer {fmt(flat['buffer_bytes'])}"
                  f" (gap {fmt(r['buffer_bytes'] - flat['buffer_bytes'])})")
    print()
    print(f"  workspace: sizeof(EPSignals) = {sig:,} B -> aligned {align(sig, kNumAllocationAlignmentBytes):,} B"
          " (workspace.cuh:9-22, ep.hpp:70-71); constant, independent of EP/tokens:")
    for name, (size, kind) in parts.items():
        print(f"     {name:32s} {size:>12,} B  written by {kind}")

    # Sensitivities
    header("V2.5 sensitivity")
    print(f"{'config':<64} {'EP16':>12} {'EP128':>12} {'EP256':>12} {'EP1024':>12}")

    def sens(label, **kw):
        vals = [v2_ep(ep, **kw) for ep in (16, 128, 256, 1024)]
        print(f"{label + ' total':<64}" + "".join(f"{fmt(v['total']):>13}" for v in vals))
        print(f"{'   of which (a) send/staging':<64}" + "".join(f"{fmt(v['a_send']):>13}" for v in vals))
    sens("T=8192 (V2 README perf: 8K tokens), hybrid", T=8192)
    sens("T=4096, num_topk=0 (sized as 32; tests/ep/test_ep.py:531-539)", T=4096, num_topk=0)
    sens("T=4096, BF16 dispatch (use_fp8_dispatch=False default)", T=4096, use_fp8=False)
    sens("T=4096, allow_multiple_reduction=False", T=4096, multi_red=False)
    sens("T=4096, allow_hybrid_mode=False (direct RDMA, FULL conn.)", T=4096, hybrid=False)
    sens("T=128,  allow_hybrid_mode=False (direct RDMA, FULL conn.)", T=128, hybrid=False)
    sens("T=4096, flat load/store domain (direct, is_scaleup_nvlink)", T=4096, force_flat_nvlink=True)
    sens("T=128,  flat load/store domain (direct, is_scaleup_nvlink)", T=128, force_flat_nvlink=True)


# --------------------------------------------------------------------------------------------------
# Loom, GPU side (rough estimate, owner 2026-10-07: "based on pointers and destinations"). With Loom a
# GPU reaches a peer through an ordinary mapping of the peer's window (paper design.tex "XPU MMU:
# virtual address -> aperture"; design-sw.tex peer mapping engine): no queue pairs, doorbells, or
# staging. What remains per GPU, for each other GPU in the EP group: one pointer (8 B) and the
# last-level page-table entries (8 B per 2 MiB page) covering that peer's receive window. The window is
# sized as DeepEP V1 low-latency's receive buffer (worst case: experts x tokens x message), an upper
# bound. Upper page-table levels (one 4 KiB page per 1 GiB of windows), the receive buffer itself (every
# design needs it, excluded for all bars) and the switch's per-binding state (not on the GPU) are left out.
PAGE = 2 * 1024 * 1024
PTE = 8
PTR = 8


def loom_gpu_state(ep, tokens=128):
    window = v1_low_latency(ep, tokens)["b_rdma_recv"]
    pages = -(-window // PAGE)
    per_peer = PTR + pages * PTE
    return dict(window=window, pages_per_peer=pages, per_peer=per_peer, total=(ep - 1) * per_peer)


def print_loom():
    # An NVLink peer is reached the same way today (CUDA IPC maps the peer's buffer into the GPU's page tables),
    # so this is the GPU-side state of an NVLink peer; Loom gives every peer in the EP group that contract.
    nvl = 7 * loom_gpu_state(256)["per_peer"]
    header(f"NVLink-style peer state, GPU side (rough estimate; NVLink peers of an 8-GPU node: 7 x per peer = {nvl:,} B):"
           " per peer one 8 B pointer + 8 B PTEs (2 MiB pages) over its receive window")
    print("   EP  window per peer   pages/peer   bytes/peer        total")
    for ep in (16, 128, 256, 512, 1024):
        r = loom_gpu_state(ep)
        print(f"{ep:5d}  {fmt(r['window']):>15}  {r['pages_per_peer']:>10,}  {r['per_peer']:>10,} B  {fmt(r['total']):>11}")


if __name__ == "__main__":
    main()
    print_loom()
