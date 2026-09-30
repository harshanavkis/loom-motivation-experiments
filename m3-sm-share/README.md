# M3: SM share of GPU-initiated communication (DeepEP)

Detailed evidence (per-variant table with file:line citations, the V2.5 copy-engine API, the full SM-model derivation, and the experiment design): [deepep-0sm.md](deepep-0sm.md).

## Question

How many SMs does DeepEP spend on communication, what do those SMs actually do, and does DeepEP's "0-SM communication" mean that no SMs move bytes? This quantifies the SM cost of GPU-initiated RDMA and of SM-driven NVLink traffic, which Loom's off-accelerator engine is meant to remove.

## Methodology

**Sources:** `../src/DeepEP-0sm`, a separate clone of github.com/deepseek-ai/DeepEP at `93eb6eb` with the PR heads and branches below fetched.

| ref | commit | what |
|---|---|---|
| `main` (V2.5, NCCL GIN) | `93eb6eb` (V2.5 = `def8651`) | current |
| last V1 (NVSHMEM/IBGDA) | `567632d` (= `b306af0^`, parent of the V2 release) | V1 normal + low-latency (LL) kernels |
| PR #347 "SM-free normal kernel" | `0b1ccde` (`git fetch origin pull/347/head:pr347`) | open, not merged into `antgroup-opt` (`76f0271`) |
| PR #453 zero-copy | `6db603e` (`pull/453/head:pr453`) | open |
| `hybrid-ep` branch (NVIDIA) | `10d4dd7` | TMA backend, DOCA/NIXL RDMA, PCIe kernel |

External: DeepSeek-V3 technical report, arXiv 2412.19437v2, §3.2.2 and §3.5.1.

**Tools:** git; Python 3.13.15 (standard library only) for `sm_est.py` and `sm_est_v20.py`.

**What was done:**
- Read the kernels, host code and READMEs of each variant to find who moves the bytes (SM ld/st, TMA, copy engine, NIC via GPU-posted WQE), how many SMs are used, and what those SMs do.
- Re-implemented DeepEP V2.5's analytical SM count (`EPBuffer.get_theoretical_num_sms`, `deep_ep/buffers/ep.py:431-541`) torch-free in `sm_est.py`, and the V2.0 formula (`b306af0:deep_ep/buffers/elastic.py:622-632`) in `sm_est_v20.py`, and evaluated them for H100/H200, B200 and RTX PRO 6000.

**Assumptions and judgment calls:**
1. Model inputs are DeepEP's own constants: per-SM HBM read 180 GB/s and write 45 GB/s (`deep_ep/utils/envs.py:194-205`, marked "TODO: architecture-specific"); V2.0 uses 200/50.
2. NVLink bandwidth: H100/H200 = 18 × 26.562 × 0.9 = 430.3 GB/s; B200 = 18 × 50 × 0.9 = 810 GB/s (DeepEP multiplies `nvidia-smi nvlink -s` by 0.9). RDMA = link rate / 8: CX7 400G = 50 GB/s, CX6 200G = 25, CX8 800G = 100.
3. Workload: DeepSeek-V3, 256 experts, top-8 (hidden size does not enter the formula); 8 GPUs per node; hybrid mode unless marked direct; `prefer_overlap_with_compute=True` (with `False` every count becomes at least 64).
4. SM counts: H100/H200 132, B200 148, RTX PRO 6000 Blackwell 188, H20 78 (NVIDIA spec).
5. The model covers only the comm kernel. The dispatch copy and combine reduce epilogues always launch on all SMs (`csrc/buffers/ep.hpp:723-743, 943-956`) and are not in the counts.
6. Statements marked "(inference)" in deepep-0sm.md are derived, not read in code or docs; this includes the epilogue estimate and the speedup ceiling below.

## Results

**Analytical comm-kernel SM count** (`python3 sm_est.py`; % of the GPU in parentheses):

| config | H100/H200 SXM (132 SMs, NVLink 430 GB/s) | B200 (148 SMs, NVLink 810 GB/s) | B200 + CX8 800G |
|---|---|---|---|
| EP8 (1x8, NVLink only) | 16 (12%) | 32 (22%) | 32 |
| EP16 (2x8) | **16** (12%) [direct: 20] | 16 [direct: 36] | 28 |
| EP32 (4x8) | 8 (6%) [direct: 24] | 8 [direct: 44] | 12 |
| EP64 (8x8) | **8** (6%) [direct: 32] | 8 [direct: 56] | 12 |
| EP128 (16x8) | 4 [direct: 44] | 4 [direct: 84] | 8 |
| EP256 (32x8) | **4** (3%) [direct: 72, 55%] | 4 [direct: 136, 92%] | 8 |
| 1 GPU/node, pure RDMA, EP2 | 8 (CX7 400G), 4 (200G) | — | RTX PRO 6000 (188 SMs): 8 at 400G, 12 at 800G |
| same, EP4-EP64 | 4 | — | 4-8 |

The V2.0 formula (`python3 sm_est_v20.py`) gives 12 for EP8x2 and 6 for EP8x4, matching the V2.0 README ("SM90 CX7 EP 8x2: 90/81 GB/s, 12 SMs; EP 8x4: 61/61 GB/s, 6 SMs"). The V2.5 formula gives 16 and 8.

**Who moves the bytes, per variant** (condensed from deepep-0sm.md §1):

| variant | who moves bytes | SMs used |
|---|---|---|
| V1 normal | SM ld/st + TMA (NVLink); NIC via GPU-posted WQE (IBGDA) | default 20 (`567632d:deep_ep/buffer.py:30`); DeepSeek-V3: 20 of 132 on H800 |
| V1 LL, no hook | NIC via GPU-posted WQE | 128 of 132 for E=256 (`internode_ll.cu:494-501`) |
| V1 LL + `return_recv_hook` | NIC via GPU-posted WQE | 0 while bytes are in flight; about 128 during the send and recv kernels |
| V2.5 EP hybrid (default) | TMA (SM-issued); NIC via GPU-posted WQE (GIN GDAKI) | comm kernel 4-16 (H100); epilogues all 132 |
| V2.5 EP direct | TMA (SM) + NIC via GPU-posted WQE | 20-72 (H100), 136 (B200 EP256) |
| V2.5 bucket all-gather, NVLink-only | copy engine via `cudaMemcpyBatchAsync` + stream memops | 0 (`EP_HOST_ASSERT(num_sms == 0)`) |
| V2.5 bucket all-gather, RDMA/hybrid | NIC via GPU-posted WQE; CE for NVLink leg | API says 0; short kernel on 132 SMs (pure RDMA) or `num_rdma_ranks` CTAs (hybrid) posts WQEs |
| V2.5 PP send/recv | TMA (SM) then NIC | `num_sms=0` means all SMs for a short kernel (`pp.hpp:92`) |
| PR #347 Normal-SMFree | NIC via GPU-posted WQE; SM/TMA for NVLink | 64 of 78 (H20) during each phase; 0 while the NIC transfers |
| PR #453 zero-copy | NIC gathers via multi-SGE WQE; TMA for NVLink | 12 (EP16), 8 (EP32) vs 24 original (H20) |
| hybrid-ep branch | TMA; NIC via DOCA GPUNetIO WQE; CE for staging | 4/8/16 (H100, EP16-64); 16/32 (GB200) |
| hybrid-ep PCIe kernel | NIC via GPU-posted WQE, intra-node too | 24 (H20, NVLink disabled) |

No DeepEP EP path uses CPU-posted WQEs: V2.5 compiles the NCCL GIN proxy backend out (`csrc/runtime/jit.hpp:39-40`) and requires GDAKI (`csrc/kernels/comm/context.cpp:129-134`).

**Reported fractions:** DeepSeek-V3 report: 20 of 132 SMs on H800 = 15.2%. V2.0 README: 12 SMs (9.1%) and 6 SMs (4.5%) on SM90. hybrid-ep: 4/8/16 SMs on H100 = 3-12%.

**What the SMs do** (deepep-0sm.md §4): inherently compute = combine reduction, FP8 quantization, routing metadata; pure data movement = token scatter/gather into expert layout, bulk NVLink/staging traffic; NIC control = WQE build and doorbell, flow control and polling.

**Speedup ceiling from freeing k SMs** (inference; 132/(132−k), perfect overlap, SM-bound compute): k = 4: 1.03×; k = 8: 1.06×; k = 16: 1.14×; k = 20 (V3): 1.18×; k = 24: 1.22×.

## Caveats / fairness

- SM counts for V2.5 are DeepEP's own analytical model, not measured occupancy. It is purely HBM-bandwidth-based with per-SM constants DeepEP marks architecture-unspecific.
- In direct mode the model always takes NVLink as the bound (`ep.py:512`), which is why direct counts are large (inference).
- The counts exclude the all-SM epilogues; the rough epilogue estimate (about 9 extra SM-equivalents for dispatch at EP16, 8K tokens) is an inference.
- The V2.0 match (12 for EP8x2) uses the H100 NVLink value 430.3 GB/s. `sm_est_v20.py` also prints the result with NVLink 160 GB/s ("H800 nvl=160" column), which gives 6 for EP8x2.
- Offload does not remove HBM traffic, because NIC DMA reads and writes the same bytes. It removes occupancy, polling and issue overhead.
- PR #347's reported "RDMA bandwidth" divides by send+recv kernel time only and excludes the overlapped network time; it also needs a roughly 270 MB RDMA buffer per rank.
- PR #347 and PR #453 are unmerged.
- The last-V1 ref here (`567632d`, parent of V2.0) differs from M2's (`a56d615`, parent of V2.5).
- deepep-0sm.md refers to the script as `scratchpad/sm_est.py`; it is `sm_est.py` in this directory.

## Candidate claims for the paper

deepep-0sm.md has no dedicated claims section; these sentences are taken from it.

1. DeepSeek-V3 report (§3.5.1): "we allocate 20 out of the 132 SMs available in the H800", which is **15.2%**. The report lists what the SMs do (IB-to-NVLink forwarding and aggregation; moving data between RDMA buffers and input/output buffers; reduce for combine; fine-grained memory layout) and then asks for a co-processor that unifies the IB and NVLink interfaces.
2. "No DeepEP EP path uses CPU-posted WQEs. V2.5 turns the NCCL GIN proxy (CPU) backend off at compile time."
3. "Where the SM budget actually goes today (V2.5 hybrid): k = 4-16 SMs for the comm kernel for its whole duration (3-12% of an H100), plus short all-SM epilogues." "Direct mode and V1 are much higher: 20-72 SMs and 15-55%."
4. "'0 SM' is exactly true only for the NVLink path." The README's "all-gather is driven by copy engines (num_sms=0)" is fully true only for NVLink; the RDMA leg is GPU-initiated.
5. "It makes sense for data movement and NIC control. It does not make sense for EP as a whole, because combine contains real arithmetic and dispatch needs a small amount of routing compute."
6. "Offload does not remove the HBM traffic, because NIC DMA reads and writes the same bytes. It does remove occupancy, polling and issue overhead."

## Reproduce

Run from this directory (`sm_est_v20.py` reads `sm_est.py` by relative path). Each takes under a second.

```sh
python3 sm_est.py
python3 sm_est_v20.py
```

To check the refs in the source clone:

```sh
for r in main def8651 'b306af0^' pr347 pr453 origin/hybrid-ep origin/antgroup-opt; do git -C ../src/DeepEP-0sm rev-parse --short=7 "$r"; done
```

Expected, one per line: `93eb6eb`, `def8651`, `567632d`, `0b1ccde`, `6db603e`, `10d4dd7`, `76f0271`. To compare the SM-model output with the tables above, read it directly; there is no stored output file.

## GPU interference experiment (steve)

Full methodology, tables and reproduce commands: [gpu-interference/README.md](gpu-interference/README.md). Single H200 NVL, no network. A cuBLAS BF16 expert GEMM runs next to either k SMs moving bytes (DeepEP-style) or the copy engine moving the same bytes at the same rate.

| k SMs held (up-proj GEMM, % of 811.6 TFLOP/s) | 8 | 16 | 20 | 32 |
|---|---|---|---|---|
| SMs copy 50 GB/s (400G line rate) | 90.5 | 72.7 | 48.9 | 65.5 |
| GEMM planned for 132−k SMs, nothing co-running (lower bound) | 90.3 | 88.8 | 79.1 | 75.1 |
| copy engine moves the same 50 GB/s, no SMs held | 99.9 | | | |

- For compute-bound work, the cost is the held SMs, not the bytes: SMs holding and idling cost the same as SMs copying at 50–809 GB/s.
- SM stores to host memory over PCIe leave the GEMM 2–34%, while copy-engine DMA of the same bytes leaves 99.8%.
- Copy engines are not a drop-in replacement. Local capacity is 85.8 GB/s whatever the stream count, and at token granularity (7–14 KB) they move only 10–18 GB/s. The offload engine needs scatter-gather descriptors and NIC-class bandwidth.

## Files

- `README.md`: this file.
- `deepep-0sm.md`: full analysis with file:line citations (variant table, CE all-gather API, SM model, verdict, experiment design).
- `sm_est.py`: torch-free re-implementation of DeepEP V2.5 `get_theoretical_num_sms`, evaluated for H100/H200, B200 and RTX PRO 6000.
- `sm_est_v20.py`: the DeepEP V2.0 SM formula, for comparison with the V2.0 README numbers.
- `gpu-interference/`: the steve SM-vs-copy-engine interference benchmark (own README).
