# M3c: GPU-posted RDMA (NVSHMEM IBGDA) next to a GEMM, on steve

## Question

When the GPU itself posts RDMA (the SMs build work requests and ring the NIC doorbell, as in NVSHMEM IBGDA / DeepEP), what does it cost concurrent compute? And how fast can SMs post messages at MoE token size?

This is the counterpart of the CPU-posted measurement ([../gpu-interference/README.md](../gpu-interference/README.md) §5). There, the CPU posted and the NIC did the DMA: the GEMM kept 100.2% and one CPU core was busy.

## Methodology

**Setup:**
- steve: H200 NVL (132 SMs), CX-7 port 0 ↔ port 1 loopback, NVSHMEM 3.6.5 with two PEs on the one GPU.
- Pure IBGDA: `NVSHMEM_REMOTE_TRANSPORT=none`, NIC handler on the GPU, 24 RC queue pairs per peer (DeepEP V1's setting).
- Driver and IOMMU prerequisites: [../steve-rdma/README.md](../steve-rdma/README.md).

**What runs:**
- **PE 0** runs a cuBLAS BF16 GEMM (up-projection [8192×7168]×[7168×4096], or down-projection [8192×2048]×[2048×7168], 100 calls). Concurrently, a **put kernel on k CTAs** runs on another stream.
- **Each put CTA holds its SM exclusively**, like a DeepEP communication kernel (200 KB shared memory, 2-CTA clusters). One thread per CTA calls `nvshmem_putmem_nbi` in a loop, paced to a total target rate, with `nvshmem_quiet` every 16 puts. The GEMM is planned for 132−k SMs (`cublasSetSmCountTarget`), as in the SM-copy experiment.
- **PE 1** is only the remote memory. It waits on the host (polling a file) and does not touch the GPU during measurement. In a first attempt PE 1 waited in `nvshmem_barrier_all`, which spins on the GPU; without MPS the two processes time-slice, and the GEMM fell to 362 TFLOP/s. That run was discarded.

**Configurations:**
- `none`: the GEMM alone.
- `target k`: GEMM planned for 132−k SMs, nothing co-running.
- `idle k`: k CTAs hold SMs, no puts.
- `put k msg rate`: msg = 7168 B (an FP8 DeepSeek-V3 token) or 64 KiB, at 5, 10 or 15 GB/s.

k ∈ {4, 8, 16, 20}; 3 reps, medians reported.

**SM time per operation:** `clock64` around each put call and each `quiet`, converted to ns with the kernel's own cycle/`globaltimer` ratio. `quiet` time is reported per put, i.e. the `quiet` total divided by the number of puts.

**Data flow of one put:** the thread reserves a send-queue slot (atomics in HBM), writes the work request into the send queue in HBM, fences, writes the doorbell record, and writes the doorbell into the NIC's BAR (MMIO, cross-socket). The NIC fetches the work request and payload from HBM, the payload travels over the cable into PE 1's HBM buffer, and the NIC writes the completion into the completion queue in HBM. `quiet` polls that queue.

## Results

GEMM throughput as % of the baseline (812.3 TFLOP/s up-projection, 766.1 down-projection). "put" is the range over the three rates, with the achieved GB/s in brackets for 7168 B messages.

| k | target | idle hold | put, 7168 B | put, 64 KiB | down-proj: idle / put |
|---|---|---|---|---|---|
| 4 | 99.7 | 100.0 | 99.9–100.0 [2.3] | 99.8–100.0 [5.0–11.7] | 99.7 / 99.6–99.9 |
| 8 | 90.5 | 90.6 | 90.4–90.5 [5.0–5.3] | 90.4–90.5 [5.0–13.0] | 91.7 / 91.6–91.9 |
| 16 | 88.9 | 72.6 | 72.6–72.7 [5.0–9.3] | 72.6–72.7 [5.0–13.8] | 64.1 / 64.0–64.1 |
| 20 | 79.2 | 48.9 | 48.9 [5.0–10.2] | 48.7–48.8 [5.0–14.0] | 54.2 / 54.1–54.2 |

**SM time per put** (posting thread): **6.0–7.8 µs inside `nvshmem_putmem_nbi`**, plus 3.6–8 µs of `quiet` per put at rates the posting keeps up with. When the NIC is the bottleneck (64 KiB at 15 GB/s), `quiet` grows to 14–81 µs per put.

Reading:
1. **GPU-posted RDMA costs compute exactly what holding the SMs costs.** Put and idle-hold columns agree to within 0.2 points in every row; the puts' own memory traffic is invisible to a compute-bound GEMM. The numbers also match the SM-copy experiment on the same GPU (90.5 / 72.7 / 48.9% at k = 8 / 16 / 20). So the SM cost of GPU-initiated communication is the SMs it occupies, whether they move bytes themselves or drive the NIC.
2. **Posting from SMs is slow per message.** One thread spends about 6–8 µs per put (inflated on steve by the cross-socket doorbell and work-request path; see caveats), so a CTA posts at most about 100–150k messages/s. At 7 KiB tokens, 4 CTAs reach only 2.3 GB/s and 20 CTAs only 10.2 GB/s, about half the loopback's 20 GB/s. Driving the NIC at line rate with token-sized messages takes many SMs, which is why DeepEP posts per warp from many CTAs.
3. **Compared with CPU-posted at the same NIC:** CPU-posted RDMA at 20.5 GB/s left the GEMM at 100.2% and used one CPU core. GPU-posted at 10 GB/s of 7 KiB messages needs about 16–20 held SMs, leaving the GEMM at 48.9–72.7% (79.2–88.9% even with perfect partitioning).

## Caveats

- **Cross-socket:** the H200 (NUMA node 0) and CX-7 (NUMA node 1) are on different sockets, so every doorbell write, work-request fetch and completion write crosses the socket interconnect. The 6–8 µs per put is therefore an upper bound; with a GPU and NIC under the same PCIe switch it would be lower. The *compute* cost (point 1) does not depend on this.
- **One posting thread per CTA with `nbi` puts:** DeepEP posts per warp with many CTAs. More threads per CTA would raise the message rate per held SM, but not change point 1: the cost is still the held SMs.
- **Placement:** the extra loss of `idle` over `target` at k = 16/20 comes from GEMM cluster/wave placement next to held SMs (explained in `../gpu-interference/README.md`). That is part of the real cost of a design that holds SMs.
- **Loopback:** both PEs are on one GPU and one NIC.

## Candidate claims

1. "On an H200, GPU-initiated RDMA (NVSHMEM IBGDA) costs a concurrent BF16 expert GEMM exactly what holding its SMs costs: 90.5 / 72.7 / 48.9% of throughput with 8 / 16 / 20 posting CTAs, identical (±0.2 points) to idle held SMs. The CPU-posted equivalent leaves the GEMM at 100.2% and costs one host core."
2. "A GPU thread spends 6–8 µs of SM time per `nvshmem_putmem_nbi` on our testbed, so 20 CTAs sustain only 10.2 GB/s of 7 KiB token messages."

## Reproduce (on steve, after `setup_root.sh`)

The runnable copy is `~/loom-experiments/gpu-posted/`.

```sh
~/loom-experiments/gpu-posted/build.sh   # = ../../m5-rdma-init/steve-cx7/build_gpu_posted.sh (deployed under this name by scripts/deploy_gpu_host.sh)
sudo ~/loom-experiments/gpu-posted/run.sh 3      # results_steve.csv, about 5 min
```

## Files

- `nvshmem_interfere.cu`: the benchmark.
- `build.sh`: nvcc `-rdc=true` against `nvshmem-3.6.5` (static `libnvshmem_device.a`) and cuBLAS.
- `run.sh`: environment, root for memlock, NUMA binding.
- `results_steve.csv`: all runs (3 reps × 66 configurations).
- `results_steve.gpu.txt`: GPU name, driver, PCIe width.
