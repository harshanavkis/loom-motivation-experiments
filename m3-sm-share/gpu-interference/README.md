# M3b: who moves the bytes on the GPU (SMs vs. copy engine), H200

## Question

When a GPU moves communication bytes with its own SMs (as DeepEP-style kernels do: they hold *k* SMs for the whole transfer), how much throughput does concurrent compute lose? And how much does it lose if an engine outside the SMs moves the **same bytes at the same rate**? On the GPU, that engine is the copy engine (CE), which stands in here for an off-accelerator engine such as Loom's. The CE→host case is closest to a NIC or Loom engine DMA-reading HBM over PCIe.

Single GPU, no network. The real-RDMA version (GPU-posted vs. CPU-posted WQEs) needs a GPU host with a cabled ConnectX NIC; none exists yet (see "Not covered").

## Methodology

**Hardware / software**

| item | value |
|---|---|
| host | steve |
| GPU | NVIDIA H200 NVL, 132 SMs, CC 9.0, max SM clock 1785 MHz, power limit 600 W |
| PCIe | Gen5, **×8** (the slot trains ×8, max ×16), so the host path tops out at 29 GB/s |
| driver | 595.71.05 (open kernel module) |
| CUDA / cuBLAS | nixpkgs `cudaPackages`: nvcc 12.9.86, cudart 12.9.79, cuBLAS 12.9.1.4 |
| binding | `numactl --cpunodebind=0 --membind=0` (the GPU sits on NUMA node 0) |

The GPU had no other compute process during any run (checked with `nvidia-smi --query-compute-apps`). Clocks were not locked (no root), so every configuration is interleaved within each rep and the tables report the median of 5 reps.

**Workloads** (fill the GPU, run on one stream):
- `gemm`: cuBLAS BF16 GEMM with FP32 accumulate, [8192 × 7168] × [7168 × 4096]. This is a DeepSeek-V3 expert up/gate projection over 8192 tokens. 100 calls per measurement; baseline 811.6 TFLOP/s.
- `gemm_down`: [8192 × 2048] × [2048 × 7168], the down projection. Baseline 764.0 TFLOP/s.
- `triad`: `a = b + s*c` over 3 × 1 GiB, HBM-bound. Baseline 3991 GB/s.

**What runs next to the workload** (on a second stream):

| mode | what | isolates |
|---|---|---|
| `none` | nothing | baseline |
| `target` | nothing co-runs; cuBLAS is told it has 132−k SMs (`cublasSetSmCountTarget`) | the cost of planning the GEMM for fewer SMs, with no interference (the best case for any SM-partitioned design) |
| `idle` | k CTAs hold k SMs and sleep on a flag; GEMM gets `SmCountTarget = 132−k` | SM occupancy alone, with no memory traffic |
| `smcopy d2d R` | the same k CTAs copy HBM→HBM, paced to R GB/s in total (`max` = unpaced) | the SM-driven data movement of DeepEP-style kernels |
| `smcopy d2h 50` | k CTAs store HBM data to pinned host memory over PCIe | SMs pushing bytes to a PCIe device by stores |
| `ce d2d/d2h R` | `cudaMemcpyBatchAsync` with `cudaMemcpyFlagPreferOverlapWithCompute`, 64 MiB requests, host-paced to R GB/s, at most 4 in flight | the same bytes moved by the copy engine, holding no SMs |

Rates: 50 GB/s is a 400G NIC's line rate (CX-7), 100 GB/s an 800G one (CX-8); k ∈ {4, 8, 16, 20, 32}. DeepEP V2.5 hybrid uses 4–16 SMs on H100, and DeepSeek-V3 used 20.

**Implementation details that matter for correctness:**
1. **One CTA per SM, exclusively.** Each holding CTA takes 200 KB of dynamic shared memory, so no GEMM CTA fits beside it. The kernel records `%smid`, and every run confirms k distinct SMs (`distinct_sms` column).
2. **SMs are taken in pairs** (2-CTA clusters, `--occ-cluster 2`, the default). This was the kinder placement for cuBLAS in the diagnostic below, so it is the conservative choice.
3. **The GEMM is told how many SMs it has.** It gets `cublasSetSmCountTarget(132−k)`, as DeepGEMM is given `num_sms` next to DeepEP. The SM targets are warmed up before any SM-holding kernel runs, and cuBLAS gets a fixed 256 MiB workspace.
4. **The stop flag lives in HBM** and is set by a small copy-engine memcpy. Polling a host-memory flag from 20 SMs perturbed the whole GPU (triad −45%) in the first version, and real comm kernels poll HBM.
5. **The copy engine really is a copy engine.** `ce_check` holds all 132 SMs and confirms that `cudaMemcpyAsync` and `cudaMemcpyBatchAsync` complete, both D2D and D2H. CORRECTION (2026-10-06): `ce_check`'s occupiers fill shared memory but not thread slots, so a copy kernel can still fit beside them. With every SM truly full (`../../m5-rdma-init/steve-cx7/dev_ce_check.cu`), plain `cudaMemcpyAsync` D2D does NOT complete (it is an SM kernel); only `cudaMemcpyBatchAsync` + `PreferOverlapWithCompute` does.
6. The copy CTAs cycle through 32 MiB per CTA with streaming loads and stores (`__ldcs`/`__stcs`), so they touch HBM rather than the 60 MB L2.

## Results

### 1. Compute-bound GEMM: copy engine ≈ free, SMs cost 10–51%

`gemm` throughput as a percentage of the 811.6 TFLOP/s baseline. `[..]` is the achieved communication GB/s.

| k | ideal (132−k)/132 | target (no co-run) | idle | smcopy d2d 50 | smcopy d2d 100 | smcopy d2d max | smcopy d2h 50 |
|---|---|---|---|---|---|---|---|
| 4 | 97.0 | 99.7 | 100.0 | 100.0 [50] | 99.9 [100] | 99.9 [117] | 34.2 [27] |
| 8 | 93.9 | 90.3 | 90.5 | 90.5 [50] | 90.4 [100] | 90.4 [232] | 20.5 [27] |
| 16 | 87.9 | 88.8 | 72.7 | 72.7 [50] | 72.6 [100] | 72.4 [454] | 4.6 [27] |
| 20 | 84.8 | 79.1 | 48.9 | 48.9 [50] | 48.9 [100] | 48.6 [574] | 2.1 [27] |
| 32 | 75.8 | 75.1 | 65.4 | 65.5 [50] | 65.4 [100] | 64.9 [809] | 2.2 [27] |

The copy engine moving the same bytes, holding no SMs:

| workload | ce d2d 50 | ce d2d 100 (achieves 81–83) | ce d2h 50 (achieves 29) |
|---|---|---|---|
| gemm | 99.9 | 99.9 | 99.8 |
| gemm_down | 99.6 | 99.8 | 99.6 |

`gemm_down` shows the same pattern (from [summary_steve.md](summary_steve.md)):

| k | target | idle | smcopy d2d 50 |
|---|---|---|---|
| 8 | 91.9 | 91.7 | 91.6 |
| 16 | 87.1 | 64.0 | 64.0 |
| 20 | 87.1 | 54.3 | 54.3 |
| 32 | 77.9 | 70.1 | 69.9 |

Reading:
- **For compute-bound work the cost is the SMs, not the bytes.** `idle` ≈ `smcopy` at every rate from 50 GB/s to 809 GB/s, while the copy engine moving 50–83 GB/s costs 0.1–0.4%.
- **Even perfect partitioning costs more than k/132.** `target` loses 8–21% at k = 8–20 (up-projection), because the GEMM's waves quantize.
- **Real co-location costs more again, and depends on placement** (k = 16/20: 27–51% lost; non-monotonic, k = 20 is worse than k = 32). `diag_placement_steve.txt` shows why placement matters. With CTAs scattered one per SM pair (`--occ-cluster 1`), k = 8 costs 84.3 ms against 65.4 ms for `target`; taken in pairs, it costs 65.5 ms. At k = 20 both placements take 121.2 ms against 74.9 ms for `target`. Inference: Hopper GEMM kernels launch in thread-block clusters and plan a grid for 132−k SMs, and held SMs spread over several GPCs leave fewer usable cluster slots than planned, which adds a partial wave. VERIFIED 2026-10-06 (`../deepgemm/dg_compare.py`): told 112 SMs, cuBLAS switches to an 8-CTA-cluster kernel (`nvjet_tst_320x128_…_2x4`, grid 112) from a 2-CTA one at 132; DeepSeek's DeepGEMM keeps 2-CTA clusters and loses nothing beyond the SMs it gives up. A design that holds SMs for communication has to co-tune the GEMM's grid with SM placement; one that holds none does not.

### 2. SM stores to host / PCIe are catastrophic; CE DMA over the same link is not

At the same ≈27–29 GB/s over the same PCIe link, SM stores to pinned host memory leave the GEMM 2–34% of its throughput (`smcopy d2h 50`), and triad 5–31%. The copy engine doing the same transfer leaves the GEMM 99.8%. Inference: the SMs' outstanding sysmem writes back up into the GPU memory system once the ×8 link saturates, stalling everyone. Design implication for a GPU-side Loom: the GPU must hand the engine descriptors and let it DMA, rather than push payload with SM stores into a device window.

### 3. HBM-bound triad: the bytes do cost something, whoever moves them

| | throughput vs 3991 GB/s |
|---|---|
| CE d2d 50 GB/s / 67 GB/s (max) | 97.6% / 96.9% |
| CE d2h 29 GB/s | 99.3% |
| idle k = 8 / 20 / 32 | 96.8% / 92.6% / 92.0% |
| smcopy d2d 50 GB/s, k = 8–32 | 93.9–95.9% (k = 4: 76.0%) |
| smcopy d2d unpaced (122–446 GB/s) | 62.4–68.3% |

Offloading does not remove HBM traffic: a CE moving 50–67 GB/s costs triad 2.4–3.1%, roughly its share of HBM bandwidth. It removes the SM cost on top of that.

### 4. Copy-engine limits (why a plain CE is not enough)

- **Capacity:** local D2D is 85.8 GB/s, with 1, 2, 4 or 8 streams alike (`ce_streams_steve.csv`, `asyncEngineCount` = 3). D2H is 29.0 GB/s (PCIe ×8 bound). That covers a 400G NIC but not 800G for a local copy. NVLink peer copies use other engines and were not testable on a single-GPU host.
- **Granularity:** CE throughput by request size with scattered destinations (`cudaMemcpyBatchAsync`), which matters for per-token dispatch:

| size | 512 B | 3.5 KB | 7 KB (FP8 token, hidden 7168) | 14 KB (BF16 token) | 64 KB | 256 KB | 1 MB | 16 MB |
|---|---|---|---|---|---|---|---|---|
| batch d2d GB/s | 0.6 | 5.3 | 10.6 | 18.4 | 46.5 | 70.9 | 81.6 | 85.6 |
| batch d2h GB/s | 0.8 | 5.6 | 10.0 | 14.8 | 22.8 | 27.2 | 28.6 | 29.0 |

At token granularity this measures about 1.5 M copies/s, i.e. 10–18 GB/s. CORRECTION (2026-10-01): the timer starts before the host enqueues the batch, so this is the driver building copies on the CPU, not the engine. With the batch built before a kernel triggers it, the engine runs 21,711 token copies at 64 GB/s (1 KiB) and 84 GB/s (7 KiB) (`../../m5-rdma-init`, `dispatch_ce.cu`). The limit is descriptor generation on the host. `cudaMemcpyAsync` in a loop is 3–4× worse still. An engine that replaces the SMs' scatter/gather needs scatter-gather descriptors, as a NIC's multi-SGE WQEs have (DeepEP PR #453 uses them). One copy per token will not do.
- **Caveat on plain `cudaMemcpyAsync` D2D:** it reaches 1431 GB/s at 16 MB (`ce_size` rows), which is SM-kernel speed. When SMs are free, the driver runs large D2D copies on SMs, and with every SM truly full the same call does not progress (`dev_ce_check.cu`; the earlier `ce_check` result came from occupiers that left thread slots free). Only `cudaMemcpyBatchAsync` with `PreferOverlapWithCompute` was used as "CE" in the tables above.

### 5. CPU-posted RDMA next to compute (real NIC, steve CX-7 loopback)

perftest `ib_write_bw` wrote 64 KiB messages from GPU memory (dma-buf GPUDirect) over steve's CX-7 port-0 ↔ port-1 cable while the workloads ran (`rdma_cpu_posted.sh`). The CPU posted the work requests and the NIC did the DMA; no SMs were used. Setup: [../steve-rdma/README.md](../steve-rdma/README.md).

| workload | no RDMA | NIC reads HBM, 20.5 GB/s | NIC writes HBM, 20.8 GB/s |
|---|---|---|---|
| gemm (TFLOP/s) | 809.6 | 811.2 (100.2%) | 810.9 (100.2%) |
| gemm_down (TFLOP/s) | 763.2 | 766.1 (100.4%) | 765.1 (100.2%) |
| triad (GB/s) | 3993.0 | 3973.7 (99.5%) | 3982.4 (99.7%) |

The posting thread used 1.00 CPU core in both directions: perftest busy-polls its completion queue. So CPU-posted RDMA at 20 GB/s costs GPU compute nothing measurable and costs one host core, which is the B2 trade-off. The GPU-posted counterpart (NVSHMEM IBGDA put kernel on k CTAs next to the GEMM) is not measured yet. IBGDA itself runs on steve (15.7 GB/s at 1 MiB puts, `../steve-rdma/nvshmem-loopback/put_bw_steve.txt`).

## Candidate claims for the paper

1. "On an H200, a DeepEP-style kernel that holds 8–20 SMs to move 50 GB/s (a 400G NIC's line rate) costs a concurrent BF16 expert GEMM 9.5–51% of its throughput; the copy engine moving the same bytes costs 0.1%." (gemm: `smcopy d2d 50` 90.5 / 72.7 / 48.9% at k = 8 / 16 / 20; `ce d2d 50` 99.9%.)
2. "Even with perfect partitioning (the GEMM planned for 132−k SMs, nothing co-running), dedicating 8–20 SMs costs 8–21%." (`target` 90.3 / 88.8 / 79.1% for gemm; 91.9 / 87.1 / 87.1% for gemm_down.)
3. "Copy engines are no drop-in replacement: the host builds their work at token granularity (7–14 KB) at only 10–18 GB/s (the engine itself runs pre-built token copies at 64–84 GB/s), and a single GPU's local copy engines top out at 86 GB/s. The offload engine needs scatter-gather descriptors and NIC-class bandwidth, which is Loom's engine, not the GPU's CE."
4. "Pushing payload to a PCIe device with SM stores stalls the whole GPU (GEMM keeps 2–34%), while DMA of the same bytes costs 0.2%: the GPU should hand descriptors to an engine."

## Caveats / fairness

- Single GPU, local HBM→HBM stands in for NVLink traffic, and HBM→host over PCIe stands in for NIC DMA. No NIC is involved.
- The SM-copy kernel is a plain ld/st copy, not DeepEP's TMA kernels. For compute-bound GEMMs this does not matter (`idle` ≈ `smcopy`); for HBM-bound work TMA may differ.
- The co-location penalty beyond `target` depends on SM placement and cuBLAS's kernel choice, and a tuned GEMM such as DeepGEMM with `num_sms` might do better. The `target` column is the placement-independent lower bound on the SM cost.
- Clocks were not locked, so the numbers are medians of 5 interleaved reps. The rep-to-rep spread ((max−min)/mean) is at most 0.4% for gemm, 0.6% for gemm_down and 0.9% for triad, with medians of 0.2–0.3%. The exception is the `smcopy d2h` rows, at up to 16%; their effect is 3–50× larger than that.
- PCIe on steve trains at ×8, so every D2H number is capped at 29 GB/s.
- The H100 on jamie was not used: it is in Confidential Compute mode and passed through to a SEV-SNP VM (see `~/doctor-cluster-config/hosts/jamie.nix`).

## Not covered yet, and what it needs

- **Two hosts are needed only for NCCL GIN and DeepEP.** NCCL refuses two ranks on one GPU, and DeepEP V2.5 runs on NCCL GIN (it also needs Hopper or newer on both ends, which rules out jack's A40). Everything else runs on steve alone over its CX-7 port-0 ↔ port-1 loopback cable:
  - CPU-posted RDMA from GPU memory (`perftest --use_cuda_dmabuf`);
  - GPU-posted RDMA through NVSHMEM IBGDA, with two PEs on the H200 (NVSHMEM supports several PEs per GPU).

  jamie (H100) plus steve would cover NCCL GIN and DeepEP, but the hosts are too far apart for the cable. That's parked until a longer optical cable (AOC) exists; jamie's config was left unchanged.
- **Loopback on steve, as set up on 2026-09-30:**
  - GPUDirect needs the CX-7's IOMMU groups (105/106) in passthrough. This was set at runtime, and a reboot reverts it.
  - With that, 64 KiB RDMA writes reach 20.5 GB/s GPU → host and 14.9 GB/s GPU → GPU across steve's two sockets (`~/loom-experiments/perftest-cuda/loopback_steve_identity.txt`).
- **Pure GPU-posted (IBGDA)** also needs the NVIDIA driver loaded with `NVreg_RegistryDwords="PeerMappingOverride=1;"`, so the GPU can map the NIC doorbell. steve currently has 0, and reloading the driver needs the owner.
- **NVLink peer copies** (multi-GPU CEs): no multi-GPU host.

## Reproduce

steve cannot see jamie's `/scratch`, but `/home` is NFS-shared. From jamie:

```sh
mkdir -p ~/loom-experiments && cp -r /scratch/harshanavkis/loom-proj/motivation-experiments/m3-sm-share/gpu-interference ~/loom-experiments/
```

On steve (the sweep takes about 25 minutes, the other two about a minute each):

```sh
cd ~/loom-experiments/gpu-interference && ./build.sh
cd ~/loom-experiments/gpu-interference && ./run.sh results_steve.csv && python3 summarize.py results_steve.csv > summary_steve.md
cd ~/loom-experiments/gpu-interference && NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 ./ce_streams > ce_streams_steve.csv
cd ~/loom-experiments/gpu-interference && for cl in 1 2; do NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 ./interfere --diag --occ-cluster $cl; done > diag_placement_steve.txt
```

Check first that the GPU is idle: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` should print nothing.

## Files

- `interfere.cu`: the benchmark (modes, workloads, `ce_check`, CE size sweep, `--diag` placement test).
- `ce_streams.cu`: CE throughput vs. number of streams.
- `build.sh`: builds both with nixpkgs CUDA 12.9.
- `run.sh`: full sweep with NUMA binding; also records GPU info in `results_steve.gpu.txt`.
- `summarize.py`: CSV → markdown tables (medians).
- `results_steve.csv`: raw sweep (590 runs, 5 reps) plus `ce_check` and `ce_size` rows.
- `summary_steve.md`: all tables, including `gemm_down` and triad in full.
- `ce_streams_steve.csv`, `diag_placement_steve.txt`, `results_steve.gpu.txt`: raw outputs.


**Matched to the GPU-initiated runs (2026-10-06, `rdma_cpu_posted_7k.sh`):** 7168 B writes, GPU memory to GPU memory, 90 s, next to the same GEMM (`rdma_gpu2gpu_7k.csv`, `rdma_cpu_7k.txt`): 2.13 M writes/s = 15.3 GB/s, poster = 1.00 CPU core, GEMM 100.1%. GPU-initiated 7 KiB puts in `../gpu-posted` reach 2.3 / 5.3 / 9.3 / 10.0 GB/s with 4 / 8 / 16 / 20 held SMs. This is Figure 3's right axis; the earlier 20.5 GB/s point was 64 KiB GPU -> host and is not comparable.
