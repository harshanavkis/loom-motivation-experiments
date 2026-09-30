# DeepEP and "0-SM communication": what moves bytes, and where SMs are still spent

Source clone: `src/DeepEP-0sm` (separate from `src/DeepEP`). Refs studied:

| ref | commit | what |
|---|---|---|
| `main` (V2.5, NCCL GIN) | `93eb6eb` (V2.5 = `def8651`) | current |
| last V1 (NVSHMEM/IBGDA) | `567632d` (= `b306af0^`, the parent of the V2 release) | V1 normal + low-latency (LL) kernels |
| PR #347 "SM-free normal kernel" | `0b1ccde` (`git fetch origin pull/347/head:pr347`) | **still open, not merged into `antgroup-opt` (`76f0271`)**; that branch only has LL-SBO #483 and LL-Layered #500 |
| PR #453 zero-copy | `6db603e` (`pull/453/head:pr453`) | open |
| `hybrid-ep` branch (NVIDIA) | `10d4dd7` | TMA backend, DOCA/NIXL RDMA, PCIe kernel |

Notation: `file:line` refers to the ref named in that row or paragraph. "(inference)" marks something I derived and did not read directly in code or docs.

---

## 1. Summary table

"GPU-posted WQE" means GPU threads write the NIC work-queue entry and ring the doorbell (IBGDA / NCCL GIN GDAKI / DOCA GPUNetIO). No DeepEP EP path uses CPU-posted WQEs. V2.5 turns the NCCL GIN proxy (CPU) backend off at compile time (`csrc/runtime/jit.hpp:39-40`: `-DNCCL_GIN_GDAKI_ENABLE=1 -DNCCL_GIN_PROXY_ENABLE=0`) and requires GDAKI when it creates the communicator (`csrc/kernels/comm/context.cpp:129-134`).

| variant | mechanism | who moves bytes | SMs used | what the SMs still do | citation |
|---|---|---|---|---|---|
| **V1 normal** (internode, NVLink+RDMA forwarding) | Persistent kernel with 2 SMs per channel (even SM = RDMA sender / NVL receiver, odd SM = forwarder). Ring buffers with head/tail credits. | SM ld/st + TMA for NVLink; NIC via GPU-posted WQE (IBGDA) for RDMA | default `Buffer.num_sms = 20`; README example `set_num_sms(24)`; DeepSeek-V3 report: "20 of the 132 SMs" on H800 | per-token routing, copying into RDMA send buffers, IB-to-NVLink forwarding, NVLink receive, combine reduction, polling ring-buffer head/tail | `567632d:deep_ep/buffer.py:30`, `README.md:134`; `internode.cu:482-506` (warp roles, `num_channels = num_sms/2`); DeepSeek-V3 report §3.2.2, §3.5.1 |
| **V1 LL, no hook** | One kernel runs both SEND and RECV phases | NIC via GPU-posted WQE (IBGDA `nvshmemi_ibgda_put_nbi_warp`, GPU rings BlueFlame doorbell) | `num_sms = ceil(E / ceil(E/num_device_sms))`, so **128 of 132 SMs** for E=256 (96 for E=288), for about 77-194 µs dispatch and 114-369 µs combine | FP8 cast (amax, scale, e4m3) inside the send kernel; one WQE per token-expert; count signalling; receive: poll counts, copy into per-expert packed layout; combine: top-k weighted reduction | `internode_ll.cu:494-501` (SM count), `:213-249` (FP8 cast), `:267` (put), `:355-440` (recv/pack), `:709` (weighted accumulate); `ibgda_device.cuh:130-150` (doorbell) |
| **V1 LL + `return_recv_hook`** ("does not occupy any SM") | Send kernel posts WQEs and **exits**. The NIC finishes the RDMA writes in the background. The later `hook()` launches the RECV-phase kernel. | NIC (GPU-posted WQE) | **0 SMs while bytes are in flight**; about 128 SMs during the short send kernel and again during the recv kernel | Same as above. Only the network flight time is SM-free: FP8 cast, WQE posting, layout and reduction still run on SMs | `567632d:csrc/deep_ep.cpp:1582` (runs on compute stream), `:1649` (SEND phase only), `:1662-1664` (hook = RECV phase); `README.md:7,270-271,290` |
| **V2.5 EP, hybrid mode** (default, main) | Comm kernel with warp roles (notify, scale-out, forward). TMA to local staging and to NVLink peers; GIN put for RDMA. Then a separate **epilogue kernel on all SMs** (dispatch copy/layout, combine reduce). | TMA (issued by SM warps) for NVLink and staging; NIC via GPU-posted WQE (GIN GDAKI) | comm kernel: analytical `num_sms` (§3), e.g. H100 EP16 = 16, EP32/64 = 8, EP128/256 = 4; **epilogues: `get_num_sms()` = all 132** | routing counts, dedup, prefix sums, slot atomics; TMA issue; WQE posting and signal polling; epilogue: scatter into expanded per-expert layout with zero padding (dispatch), top-k reduction (combine) | `deep_ep/buffers/ep.py:431-541`; `hybrid_dispatch.cuh:107-134,329-446,462-594`; `csrc/buffers/ep.hpp:723-743` ("Launch copy kernels with full SMs"), `:943-956`; `README.md:39` ("zero-SM RDMA EP is not supported") |
| **V2.5 EP, direct mode** | Every rank puts to every rank. NVLink via TMA stores, RDMA via GIN put. | TMA (SM) + NIC (GPU-posted WQE) | analytical: H100 EP16 = 20, EP64 = 32, EP256 = 72 (55% of the GPU); B200 EP256 = 136 | same as hybrid | `impls/ep/dispatch.cuh:91-105` (counts), `:288` (TMA load), `:336-345` (slot atomics), `:362,373` (TMA stores), `:385` (`gin.put`); `combine.cuh:138-235` |
| **V2.5 bucket all-gather, NVLink-only** | Host builds per-(peer, bucket) descriptors. `cudaMemcpyBatchAsync` with `cudaMemcpyFlagPreferOverlapWithCompute`. Arrival and completion flags use `cuStreamBatchMemOp` WRITE_VALUE_64 / WAIT_VALUE_64 on peer-mapped signals. | **copy engine** (CE) over NVLink P2P; flags written by stream memops | **0** (`EP_HOST_ASSERT(num_sms == 0)`) | none | `csrc/kernels/driver/driver.cpp:50-65` (CE push), `:67-87` (signals); `csrc/kernels/bucket/all_gather.hpp:62-108`; `csrc/buffers/bucket.hpp:257`; `deep_ep/buffers/bucket.py:290-291` |
| **V2.5 bucket all-gather, RDMA / hybrid** | Pure RDMA: a cooperative kernel on **all device SMs** runs a GIN barrier and then `gin.put`s. Hybrid: `num_rdma_ranks` CTAs (1 per RDMA peer, 1 warp per QP) put chunks with signal increments, and the CE forwards each arrived chunk over NVLink after a `cuStreamWaitValue`. | NIC via GPU-posted WQE for RDMA; CE for the NVLink leg | API says `num_sms=0`, but a short kernel on 132 SMs (pure RDMA) or `num_rdma_ranks` CTAs (hybrid) only posts WQEs and exits. **"0 SM" is exactly true only for the NVLink path.** | barrier, WQE posting | `all_gather.hpp:36-61` (`grid_dim = get_num_sms()`), `:109-229`; `impls/bucket/all_gather/rdma.cuh:38-48`; `hybrid.cuh:55-76` |
| V2.5 reduce-scatter / all-reduce | SM kernels, NVLink multimem (NVLS) reductions | SM ld/st + multimem; NIC (GPU WQE) | at least 4-8 SMs (formula in `bucket.py:236-288`); "0-SM reduce-scatter/all-reduce is not implemented" | reduction | `csrc/buffers/bucket.hpp:127,195` |
| V2.5 PP send/recv (V2.0 README called it "0 SM PP") | TMA-copy the tensor into the registered send slot, then `gin.put` and a signal | TMA (SM) then NIC (GPU WQE) | `num_sms=0` means **all SMs** (`pp.hpp:92`) for a short kernel | staging copy, credit polling | `impls/pp/pp_send_recv.cuh:111-156` (copy at `:142`); `csrc/buffers/pp.hpp:92,109`. V2.5 README dropped the "0 SM" label for PP and Engram (`README.md:33-34`) |
| **PR #347 Normal-SMFree** (AntGroup) | V1 normal kernel split into a SEND-phase and a RECV-phase kernel (hook), run on the compute stream. The RDMA recv buffer is enlarged to hold **all tokens in one shot**, so senders never wait on ring-buffer credits. One channel per SM instead of two. | NIC via **GPU-posted WQE (IBGDA)**; NVLink leg by SM ld/st + TMA in the recv phase | test uses `num_sms = 64` on H20 (78 SMs, per NVIDIA spec) **during each phase**; **0 while the NIC transfers** | send phase: routing, copying into RDMA buffer, WQE posting, tail atomics; recv phase: RDMA-to-NVLink forwarding, NVLink receive, combine reduce. PR text: recv phase of dispatch and send phase of combine are "limited by the NVLink bandwidth" | `pr347:csrc/kernels/utils.cuh:667-687` (hook roles), `internode.cu:495,1296` (1 SM/channel), `:818` (IBGDA put), `deep_ep.cpp:407` (compute stream), `tests/test_internode_hook.py:503`; PR body §2-3.3 |
| **PR #453 Zero-copy + TMA offload** (Tencent) | Buffer fusion: user tensors live in registered comm buffers. The RDMA sender builds an **SGE list** of token rows and posts **multi-SGE RDMA WRITE** WQEs, so the NIC gathers rows directly with no SM copy into a send buffer. NVLink forward via TMA. One channel per SM. | NIC gathers via GPU-posted multi-SGE WQE; TMA (SM) for NVLink | **12 SMs (EP16), 8 SMs (EP32)** vs 24 for the original (H20) | routing, SGE-list construction, WQE posting, credit polling, TMA forwarding, combine reduction | `pr453:csrc/kernels/internode_zcopy.cu:106` (channel = SM), `:244-246`, `:326` (`put_nbi_warp_multi_sge_parallel`), `:574-600` (TMA); `tests/test_internode.py:325-329`; PR body |
| **hybrid-ep branch** (NVIDIA) | Persistent, warp-specialized kernels. Dispatch per block: 1 RDMA (N2N) warp, 1 G2S TMA warp, 2-3 S2G TMA warps. Combine: intra/inter-node reduction warp groups plus G2S and RDMA warps. RDMA through **DOCA GPUNetIO** (GPU builds WQE, `doca_gpu_dev_verbs_submit_db`) or NIXL/UCX GDA. For multi-node, the input tokens are first staged with a **`cudaMemcpyAsync` D2D (CE)** into the RDMA-registered buffer. | TMA (SM-issued); NIC via GPU-posted WQE; CE for input staging | **4/8/16 SMs** (H100, EP16-64); **16/32** (GB200); on B200 EP8, 16 SMs match DeepEP V1 at 36-48 | TMA issue, WQE posting, flag polling, combine reduction, optional fused permute/unpermute blocks | `hybrid_ep_backend.cuh:4405-4407,4650-4653` (warp layout), `:1155-1209` (DOCA WQE + doorbell); `executor/executor.cu:211`; `docs/README_Hybrid-EP.md:62-95,116`; `docs/Hybrid-EP_Implementation.md:466-484` |
| **hybrid-ep PCIe kernel** (no NVLink) | All traffic, **including intra-node**, goes through the RDMA NIC with bounce buffers. Two SMs per channel (sender + receiver). | NIC via GPU-posted WQE (IBGDA) | **24 SMs** (H20 with NVLink disabled, 8x CX7, EP8/EP16, about 48-54 GB/s) | copying into bounce buffers, WQE posting, receive and layout, combine reduction | `csrc/kernels/pcie.cu:292` (2 SM/channel), `:533` (put); `docs/README_PCIe.md:13-35,58-61` ("Future: reduce SM usage by merging sender and receiver") |

---

## 2. V2.5 copy-engine all-gather: the exact API

- **API:** `cudaMemcpyBatchAsync(dsts, srcs, sizes, n, &attrs, &attr_idx, 1, stream)` with `srcAccessOrder = cudaMemcpySrcAccessOrderStream` and `flags = cudaMemcpyFlagPreferOverlapWithCompute` (`driver.cpp:58-64`). It is not `cudaMemcpyAsync`, `cuMemcpyBatchAsync` or `cudaMemcpy3DPeer`. The destinations are peer pointers into the NCCL symmetric window (`context.get_sym_ptr(..., peer_rank_idx)`, `all_gather.hpp:87`). That means P2P/NVLink copies driven by the CE.
- **Synchronization is 0-SM too.** Every rank writes a sequence number into each peer's signal slot (`CU_STREAM_MEM_OP_WRITE_VALUE_64`) and waits `GEQ` on its own slots (`CU_STREAM_MEM_OP_WAIT_VALUE_64`), batched through `cuStreamBatchMemOp` (`driver.cpp:21-48,67-87`). The order is: prologue sync, CE push, epilogue sync (`all_gather.hpp:106-108`). Sequence indices cannot be used under CUDA-graph capture (`driver.cpp:14-19`).
- **RDMA is not done by the CE.** For RDMA peers the bytes go through the NIC, and GPU threads post the WQEs (`rdma.cuh:45`, `hybrid.cuh:73`). In the hybrid path the CE waits on per-chunk RDMA tail signals and then forwards each chunk over NVLink (`all_gather.hpp:205-225`). A host-side cost model picks the chunk count, with about 10 µs of CE launch overhead per wave (`all_gather.hpp:120-135`). So the README's "all-gather is driven by copy engines (num_sms=0)" is fully true only for NVLink. The RDMA leg is GPU-initiated, although the SMs are held only long enough to post the WQEs.

## 3. Analytical SM count (`EPBuffer.get_theoretical_num_sms`, `deep_ep/buffers/ep.py:431-541`)

The model is purely HBM-bandwidth-based. It assumes per-SM HBM read = 180 GB/s and write = 45 GB/s (`deep_ep/utils/envs.py:194-205`, "TODO: architecture-specific"). NVLink GB/s comes from `nvidia-smi nvlink -s` × 0.9, and RDMA GB/s from `ibstat` rate / 8.

- `E[k](g) = g·(1 − C(E − E/g, k)/C(E, k))` is the expected number of distinct groups hit by the top-k.
- Per unit of token data it accumulates `sm_read`, `sm_write`, `rdma_traffic` and `nvlink_traffic`. Hybrid mode adds: read tokens, write the send buffer, local bypass, forward reads, write the scale-up issue.
- The bounding link is RDMA or NVLink. The raw count is `max(B/T·sm_read/180, B/T·sm_write/45, B/T·(sm_write − nvl)/180, B/T·(sm_read + nvl)/45)`, where B/T is the bounding bandwidth over its traffic. The last two terms model combine.
- The raw count is then scaled: `align(max(4, ceil(1.3·n)), 4)`, forced to at least 64 when `prefer_overlap_with_compute=False`, and capped at the device SM count rounded down to even.
- In direct mode the bound is always NVLink (`ep.py:512`), even though RDMA traffic is computed. That looks like a modelling shortcut, and it is why the direct-mode counts come out large (inference).
- **The count covers only the comm kernel.** The dispatch copy epilogue and the combine reduce epilogue always launch on all SMs (`ep.hpp:723-743,943-956`). They run on the comm stream, or on the caller's stream at `.wait()` when deferred (`ep.hpp:772-783`).

I reproduced the formula torch-free in `scratchpad/sm_est.py`. The results below use DeepSeek-V3 settings (256 experts, top-8; hidden size does not enter the formula), 8 GPUs per node, hybrid mode, CX7 400G = 50 GB/s:

| config | H100/H200 SXM (132 SMs, NVLink 430 GB/s) | B200 (148 SMs, NVLink 810 GB/s) | B200 + CX8 800G |
|---|---|---|---|
| EP8 (1x8, NVLink only) | 16 (12%) | 32 (22%) | 32 |
| EP16 (2x8) | **16** (12%) [direct: 20] | 16 [direct: 36] | 28 |
| EP32 (4x8) | 8 (6%) [direct: 24] | 8 [direct: 44] | 12 |
| EP64 (8x8) | **8** (6%) [direct: 32] | 8 [direct: 56] | 12 |
| EP128 (16x8) | 4 [direct: 44] | 4 [direct: 84] | 8 |
| EP256 (32x8) | **4** (3%) [direct: 72, 55%] | 4 [direct: 136, 92%] | 8 |
| 1 GPU/node, pure RDMA, EP2 (e.g. steve + H100 host) | 8 (CX7 400G), 4 (200G) | — | RTX PRO 6000 (188 SMs): 8 at 400G, 12 at 800G |
| same, EP4-EP64 | 4 | — | 4-8 |

- **B200 has 148 SMs, not 188.** 188 is the RTX PRO 6000 Blackwell. That card has no NVLink, so only the pure-RDMA row applies; V2.5 EP has no PCIe-only intra-node path, and the closest is hybrid-ep's PCIe kernel.
- **Where the README's V2.0 table comes from.** With `prefer_overlap_with_compute=False` every row becomes at least 64. The V2.0 formula (`b306af0:deep_ep/buffers/elastic.py:622-632`: read/write constants 200/50, factor 1.25, align 2, dispatch terms only) gives **12 for EP8x2 and 6 for EP8x4**. Those are exactly the V2.0 README numbers ("SM90 CX7 EP 8x2: 90/81 GB/s, 12 SMs; EP 8x4: 61/61 GB/s, 6 SMs"; "SM100 EP8: 64 SMs max-perf, 24 min-SM"; "V3-like training: SM usage 24 → 4-6"; README at `b306af0`). The V2.5 formula is more conservative: 16 and 8.

### Reported SM counts and fractions (with sources)
- **DeepSeek-V3 tech report** (arXiv 2412.19437v2, §3.2.2): "only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink … partition 20 SMs into 10 communication channels". §3.5.1: "we allocate 20 out of the 132 SMs available in the H800", which is **15.2%**. The report lists what the SMs do: IB-to-NVLink forwarding and aggregation; moving data between RDMA buffers and input/output buffers; reduce for combine; fine-grained memory layout. It then asks for a co-processor that unifies the IB and NVLink interfaces. This is almost exactly the Loom thesis and is worth citing.
- **V1 DeepEP:** default 20 SMs (`buffer.py:30`), README example 24 (`README.md:134`). The V1 README performance tables (H800, `567632d:README.md:15-37`) have no SM column. Normal kernels: 153/158 GB/s NVLink at EP8, 43-58 GB/s RDMA at EP16-64. LL kernels: 77-194 µs dispatch, 114-369 µs combine at EP8-256. The LL kernels have "no SM control API" (`README.md:243`) and use about 128 SMs.
- **V2.0 README table:** as above, 12 SMs (9.1%) and 6 SMs (4.5%) on SM90.
- **hybrid-ep:** 4/8/16 SMs on H100 is 3-12%. At EP16, 8 SMs reach 62/77/68 GB/s kernel-only (FP8 dispatch / BF16 dispatch / combine), and 4 SMs reach 34/44/52 GB/s. On B200 EP8, hybrid-ep with 16 SMs gets 410/536/531 GB/s, while DeepEP V1 needs 36-48 SMs for about the same.
- **PR #453:** 24 → 12 SMs (EP16) and 24 → 8 SMs (EP32) on H20, with higher bandwidth.
- **PR #347:** H20, 4096 tokens. Kernel time falls, for example FP8 dispatch at EP32 goes from 3535 to 884 µs (30 → 124 GB/s "RDMA bandwidth"). Caveat: in hook mode this "bandwidth" divides by send+recv kernel time only and excludes the overlapped network time (PR §3.2.1). The cost is a roughly 270 MB RDMA buffer per rank.

---

## 4. Verdict: does "0-SM communication" make sense?

It makes sense for data movement and NIC control. It does not make sense for EP as a whole, because combine contains real arithmetic and dispatch needs a small amount of routing compute. DeepEP's own history supports this split: V2 already separates a k-SM comm kernel from all-SM layout and reduce epilogues.

**Inherently compute** (stays on SMs, or must be fused into neighbouring kernels):
1. **Combine reduction.** A top-k weighted sum per token in BF16/FP32: `combine_utils.cuh:58-165` (V2), and `internode_ll.cu:709` (V1 LL, `accum += bf16 * weight`). A NIC or CE cannot do it. Only in-switch reduction (NVLS multimem, which DeepEP uses for bucket reduce-scatter) or a reduction-capable engine could. Fusing it into the next kernel is the realistic 0-comm-SM option (inference).
2. **FP8 quantization.** V1 LL fused amax, scale and e4m3 conversion into the send kernel (`internode_ll.cu:213-249`). V2 moved it out: the caller passes `(x_fp8, sf)` and the comm kernel only copies the scale factors (`dispatch.cuh:293-307`). This shows quantization is producer-side compute, not communication.
3. **Routing metadata.** Per-expert and per-rank counts with rank deduplication (`dispatch.cuh:91-105`), prefix sums (`:231-256`), and slot allocation by `atomicAdd` (`:336-345`). This is O(tokens × top-k) integer work: small, but on the critical path, because it produces the destination offsets, i.e. the descriptors an engine would need.

**Pure data movement** (an external engine could do it with descriptors):
4. **Token scatter/gather and expert layout.** The dispatch copy epilogue writes the expanded per-expert layout with padding. This is index-driven row movement of 7-14 KB, which is DMA work once the indices exist. PR #453 already has the NIC gather rows through multi-SGE WQEs (`internode_zcopy.cu:326`). A CE could do it through `cudaMemcpyBatchAsync` with one descriptor per token. Whether a CE sustains line rate at 7 KB granularity is unmeasured (inference; this is the key microbenchmark).
5. **Bulk NVLink and staging traffic.** TMA loads and stores (`dispatch.cuh:288,362,373`; `hybrid_dispatch.cuh:594`; hybrid-ep G2S/S2G warps) are SM-issued even though TMA is an asynchronous unit, so a warp and its SM stay resident. The CE already does this for the NVLink all-gather.

**NIC control and synchronization** (pure control; an engine or proxy could do it):
6. **WQE build and doorbell.** GIN GDAKI, IBGDA (`ibgda_device.cuh:130-150`), DOCA (`hybrid_ep_backend.cuh:1155-1209`). GPU-side posting was chosen for latency, and the proxy backend is compiled out.
7. **Flow control and polling.** Head/tail credits, signal waits, barriers (`timeout_while`). PR #347's motivation is that "SMs are often stalled, continuously polling for RDMA buffer availability". This is SM time spent doing nothing, and it disappears when buffers are large (PR #347) or the engine handles flow control.

**Where the SM budget actually goes today** (V2.5 hybrid): k = 4-16 SMs for the comm kernel for its whole duration (3-12% of an H100), plus short all-SM epilogues.
- Rough epilogue estimate (inference): dispatch at EP16 with 8K tokens moves about 0.4 GB read + 0.5 GB write per rank. At about 3 TB/s that is roughly 0.3 ms of the whole GPU, against about 4.4 ms of dispatch at 90 GB/s logical bandwidth. Spread over the dispatch time, that is about 9 extra SM-equivalents.
- Direct mode and V1 are much higher: 20-72 SMs and 15-55%.

The ceiling on the compute speedup from freeing k SMs is about 132/(132−k), assuming perfect overlap and SM-bound compute (inference):
- k = 4: 1.03×
- k = 8: 1.06×
- k = 16: 1.14×
- k = 20 (V3): 1.18×
- k = 24: 1.22×

Offload does not remove the HBM traffic, because NIC DMA reads and writes the same bytes. It does remove occupancy, polling and issue overhead.

## 5. Implications for the "CE / external engine vs. SM-driven GPU-initiated RDMA" experiment

Constraints:
- DeepEP V2.5 EP cannot run on one GPU. NCCL rejects two ranks on the same GPU, and EP=1 has no traffic.
- The ConnectX links on steve are down. That rules out RDMA and loopback QPs, since QPs need an active port; I did not test this (inference).

So the cleanest setup isolates the SM-side cost on a single GPU and emulates the NIC's HBM traffic with the CE over PCIe. On H200 NVL, PCIe Gen5 x16 gives about 50-55 GB/s per direction, roughly the rate a CX7 400G NIC pulls from HBM (inference).

**Single-GPU (steve, H200 NVL, 132 SMs). Same compute kernel in all runs:** DeepSeek-V3 expert GEMMs, e.g. grouped GEMM K=7168, N=4096 (gate+up) and K=2048, N=7168 (down), with M ≈ 512-2048 per expert, in BF16 and FP8 via cuBLAS/`torch._scaled_mm` or DeepGEMM. Optionally add an attention kernel as the micro-batch-overlap partner.
- **C0 baseline:** compute alone on 132 SMs.
- **C1 occupancy only:** compute confined to 132−k SMs with CUDA green contexts (SM-partitioned resources, no root; check the driver/CUDA version on the host). Sweep k ∈ {4, 8, 12, 16, 20, 24, 32, 64}. Fallback: k persistent spinner CTAs, 1 per SM via maximum shared memory, launched first on another stream. This gives the upper bound of what SM-free communication buys.
- **C2 SM-driven comm emulation:** a k-CTA persistent kernel concurrent with compute. It runs DeepEP-like per-token TMA or ld/st gathers of 7168 B FP8 rows (plus SF) into a staging buffer and pushes them to pinned host memory, throttled to 50 GB/s (400G) or 25 GB/s (200G). It spins on flags between chunks, mimicking credit polling.
- **C3 engine-driven:** the same bytes moved by `cudaMemcpyBatchAsync` with `cudaMemcpyFlagPreferOverlapWithCompute` (exactly DeepEP's `copy_engine_push`, `driver.cpp:50-65`), D2H/H2D at the same rate, with completion via `cuStreamWaitValue`. Compute uses all 132 SMs.
- **Report:**
  - C2/C3 compute time is the offload speedup.
  - C3/C0 is the residual HBM/PCIe interference that any external engine, including a NIC, still causes.
  - C1 tells you how much of C2's cost is occupancy versus memory contention.
  - Keep the all-SM epilogue in C2 and C3 (layout or reduce over the received bytes), since offload does not remove it unless the engine does the layout.
- **C4 microbenchmark (key for Loom):** CE throughput versus descriptor size and count (1 KB to 1 MB; 1 to 64K descriptors per batch; D2D and D2H). This decides whether a CE can do per-token layout (item 4) at line rate or whether the layout has to stay on SMs.

**Two-host follow-up (once the links are up):** steve H200 NVL ↔ H100 host running DeepEP V2.5 pure-RDMA EP2 (1 GPU per node).
- Requirements: CUDA 13.1+ for DeepJIT, NCCL with GIN GDAKI, and `ibstat`/`nvidia-smi` must work for auto-detection.
- The analytical count is **8 SMs** at 400G and **4** at 200G. Run dispatch/combine concurrently with the GEMM and sweep the `num_sms` override {4, 8, 16, 24}.
- For the external-engine contrast, compare against NCCL GIN's proxy (CPU-posted WQE) backend. This needs a patch to `jit.hpp:39-40` and `context.cpp:129-134`, and I have not verified that DeepEP's kernels work on proxy GIN. A simpler alternative is NVSHMEM perftest device-initiated put bandwidth (k CTAs) against host-initiated put, concurrent with the GEMM.
- Check the H100's SM count first: 132 on SXM/NVL, 114 on PCIe.
