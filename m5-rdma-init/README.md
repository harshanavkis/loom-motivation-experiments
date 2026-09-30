# M5: initiation cost of GPU communication (rdma_init) — CPU-posted, GPU-posted, copy engine

## Question

What does it cost to start a transfer, and to learn that it finished, when the work request is posted by the **CPU** (B2, the NCCL proxy model), by the **GPU's SMs** (B1, NVSHMEM IBGDA / DeepEP), or by the **copy engine** (the GPU's own DMA engine, started from the host)? And how does each path behave while a GEMM saturates the GPU, which is what happens when MoE dispatch/combine overlaps expert compute?

These are the `rdma_init` inputs the astra-sim model needs. The model adds `rdma_init` on top of the wire latency, so it must be measured with (almost) no wire. That is the loopback setup here: steve's two CX-7 ports cabled to each other on one host.

Status:
- **CX-7 on steve: done** (`steve-cx7/`).
- **E810 post+poll on the FPGA testbed (amy/clara): not started.**

## Methodology

**Hardware and software:**
- steve: H200 NVL (132 SMs, NUMA node 0, PCIe Gen5 ×8), ConnectX-7 dual port (NUMA node 1), port 0 ↔ port 1 at 200 Gb/s, RoCE v2 (GID 3).
- Driver 595.71.05 with `PeerMappingOverride=1`; IOMMU groups of the GPU and NIC in passthrough.
- CUDA 12.9, perftest `bae0736` (dma-buf GPUDirect), NVSHMEM 3.6.5.
- Full setup: [../m3-sm-share/steve-rdma/README.md](../m3-sm-share/steve-rdma/README.md).

**The GPU and the NIC are on different sockets.** Every NIC↔HBM access and every GPU→NIC doorbell crosses the socket interconnect. That inflates the paths that touch GPU memory, the GPU-posted path most of all (see Caveats).

**Load:** a BF16 cuBLAS GEMM ([8192×7168]×[7168×4096]) runs back to back for the whole measurement:
- For perftest (which launches no kernels), in a separate process.
- For the copy engine, on a second stream of the same process.
- For NVSHMEM, as a separate MPS client. It was verified to run concurrently at full speed: 100% utilization, 433 W, 337 × 100 GEMMs in 20 s, the same rate as alone.

| path | tool | what is timed | statistic |
|---|---|---|---|
| CPU-posted RDMA write | `ib_write_lat` (ping-pong) | one-way = ½ round trip | average over 20,000 |
| CPU-posted RDMA read | `ib_read_lat` | read round trip (post → CQE) | average over 20,000 |
| GPU-posted RDMA read | NVSHMEM `shmem_g_latency` (IBGDA, no host proxy) | one blocking `nvshmem_int_g` round trip (2 threads, 1 get each) | total / 10,000 iterations |
| GPU-posted RDMA put | NVSHMEM `shmem_put_latency` (IBGDA, no host proxy) | `put_nbi` + `quiet` per iteration: post → completion after the remote ACK, from 1 thread / 1 warp / 1 block | total / 10,000 iterations |
| copy engine | `ce_latency.cu` | API call only; issue → last 8 B visible in pinned host memory; issue → `cudaStreamSynchronize` | median over 20,000 (5,000 under load) |
| reference | `ce_latency.cu` | empty kernel launch → `cudaStreamSynchronize` | median |

Notes on the method:
- **GPU memory with `ib_write_lat`:** perftest can't poll GPU memory, so the GPU-memory write test uses write-with-immediate. `host_imm` is the same method on host memory, as a control: 1.48 µs vs. 1.44 µs for the plain write.
- **Excluded: NVSHMEM's ping-pong tests.** Their signal operations go through NVSHMEM's host proxy thread in this setup (with the proxy off the kernel faults on a NULL pointer, Xid 31). They take 50–60 µs and are not GPU-posted numbers.
- **Two PEs on one GPU need MPS.** Without it the two processes time-slice and the ping-pong crawls. `run_put_lat.sh` starts a private MPS daemon.

## Results (8 B unless noted)

| path | operation | idle | GEMM running |
|---|---|---|---|
| CPU-posted | RDMA write one-way, host → host memory | 1.44 µs | 1.42 µs |
| CPU-posted | RDMA write one-way, GPU → GPU memory | 2.64 µs | 2.62 µs |
| CPU-posted | RDMA read round trip, host memory | 2.55 µs | 2.57 µs |
| **CPU-posted** | **RDMA read round trip, GPU memory** | **3.35 µs** | **3.36 µs** |
| **GPU-posted (IBGDA)** | **RDMA read round trip (`nvshmem_int_g`), GPU memory** | **14.06 µs** | **14.39 µs** |
| GPU-posted (IBGDA) | put + completion, 1 thread | 12.60 µs | 14.82 µs (+18%) |
| GPU-posted (IBGDA) | put + completion, 1 warp | 15.90 µs | 19.33 µs (+22%) |
| GPU-posted (IBGDA) | put + completion, 1 block | 15.89 µs | 17.53 µs (+10%) |
| copy engine | API call (`cudaMemcpyAsync` / batch) | 1.76 / 2.02 µs | 2.86 / 3.03 µs |
| copy engine | D2H issue → data visible in host memory | 4.58 µs | 5.57 µs |
| copy engine | D2D issue → sync, batch API + PreferOverlap (median / p99) | 6.61 / 7.06 µs | 7.85 / 8.90 µs |
| "copy" | D2D issue → sync, plain `cudaMemcpyAsync` (median / p99) | 7.26 / 7.78 µs | 8.32 / **526 µs** |
| reference | empty kernel launch → sync | 6.58 µs | 7.83 µs |

Larger sizes are in the CSVs. CPU-posted GPU-memory reads grow to 3.79 µs at 4 KiB and 9.04 µs at 64 KiB. GPU-posted put + completion from one thread grows to 13.1 µs at 4 KiB and 14.5 µs at 32 KiB.

Reading the table:
1. **Same NIC, same cable, same GPU buffers: a GPU-posted read round trip costs 14.06 µs, a CPU-posted one 3.35 µs.** That is 4.2×, or 10.7 µs more per operation. The difference is the work the GPU does itself: building the work request (atomics, stores, fences), writing the doorbell to the NIC, the NIC fetching the work request from HBM, the completion landing in HBM, and the GPU polling it.
2. **Compute makes GPU-posted communication slower; CPU-posted doesn't notice.** A put from a thread, warp or block slows by 10–22% while the GEMM runs (a read by 2%). Every CPU-posted number changes by 0.02 µs or less.
3. **Copy engine initiation costs about 4.6 µs to data visible in host memory, but it can only be started from the host.** That's an API call (1.8–2.0 µs), the driver's command submission, and a doorbell. Under load it stays within about 1 µs of idle, as long as it really is a copy engine: plain `cudaMemcpyAsync` device-to-device is silently executed as an SM kernel and waits behind the GEMM (p99 526 µs).

## Caveats / fairness

- **The cross-socket topology on steve inflates GPU-posted latency more than CPU-posted.** The GPU-posted path crosses the socket interconnect several extra times per operation: doorbell write, work-request fetch from HBM, completion write to HBM. Each GPU↔NIC access across sockets costs about 1 µs here (GPU vs. host memory at an otherwise equal method: 2.64 vs. 1.48 µs one-way). So treat 10.7 µs as an upper bound for well-placed GPUs and NICs, not a typical value. The published NVSHMEM IBGDA inter-node put figure (7.5 µs one-way, 256 B, including the wire; arXiv 2606.05951) is the other end of the bracket.
- **Statistics differ:** perftest reports averages over iterations; NVSHMEM reports total time / iterations; the copy-engine numbers are medians. All the medians and averages here are close, since the distributions are tight (the p99 is in the CSVs).
- **Loopback:** both ends are on one NIC and one host, so the remote side is also on the same GPU/host. Wire time is a few meters of cable (negligible), which is what the simulator's `rdma_init` needs.
- **What "GEMM running" means for NVSHMEM:** it is an MPS co-tenant, which shares SMs the way an in-process overlap would, but the scheduling is not identical.

## Candidate claims for the paper

1. "On the same NIC and GPU buffers, a GPU-posted (IBGDA) RDMA read round trip takes 14.1 µs versus 3.35 µs when the CPU posts it: the GPU pays about 10.7 µs per operation for building, ringing and completing work requests itself." (Mention the cross-socket inflation.)
2. "GPU-posted communication slows down when compute runs beside it (puts +10–22% under a concurrent GEMM); CPU-posted and copy-engine paths don't (within 0.02 µs and about 1 µs respectively). Yet MoE overlaps communication with expert compute by design."
3. "A 'copy' can silently become an SM kernel: plain `cudaMemcpyAsync` device-to-device reaches a p99 of 526 µs behind a GEMM, while the copy-engine path stays at 8.9 µs."

## Reproduce (on steve, after `setup_root.sh` from the steve-rdma README)

The runnable copies live in `~/loom-experiments/latency/` and `~/loom-experiments/nvshmem-loopback/`. The files in `steve-cx7/` are the committed copies, and `nvshmem_env.sh` is `nvshmem-loopback/env.sh`.

```sh
cd ~/loom-experiments/latency && ./build.sh
cd ~/loom-experiments/latency && NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 ./ce_latency --iters 20000 > ce_latency_idle.csv
cd ~/loom-experiments/latency && NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 ./ce_latency --iters 5000 --load > ce_latency_load.csv
~/loom-experiments/latency/lat_cpu_posted.sh idle      # lat_cpu_posted_idle.csv, about 8 min
~/loom-experiments/latency/lat_cpu_posted.sh load      # lat_cpu_posted_load.csv, about 15 min, GEMM in another process
sudo ~/loom-experiments/nvshmem-loopback/run_put_lat.sh        # put_lat_steve.txt
sudo ~/loom-experiments/nvshmem-loopback/run_put_lat.sh load   # put_lat_steve_load.txt, GEMM as an MPS client
```

## Files (`steve-cx7/`)

- `ce_latency.cu`, `build.sh`: the copy-engine latency benchmark.
- `ce_latency_{idle,load}.csv`: its outputs.
- `lat_cpu_posted.sh`: runs perftest write/read latency on the loopback, idle or loaded.
- `lat_cpu_posted_{idle,load}.csv`: its outputs.
- `run_put_lat.sh`: NVSHMEM IBGDA latency (MPS, 2 PEs), idle or loaded.
- `put_lat_steve.txt`, `put_lat_steve_load.txt`: its outputs.
- `nvshmem_env.sh`: the NVSHMEM environment (the MPG / IBGDA / HCA mapping settings).
