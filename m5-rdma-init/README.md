# M5: initiation cost of GPU communication (rdma_init) — CPU-posted, GPU-posted, copy engine

## Question

What does it cost to start a transfer, and to learn that it finished, when the work request is posted by the **CPU** (B2, the NCCL proxy model), by the **GPU's SMs** (B1, NVSHMEM IBGDA / DeepEP), or by the **copy engine** (the GPU's own DMA engine, started from the host)? And how does each path behave while a GEMM saturates the GPU, which is what happens when MoE dispatch/combine overlaps expert compute?

These are the `rdma_init` inputs the astra-sim model needs. The model adds `rdma_init` on top of the wire latency, so it must be measured with (almost) no wire. That is the loopback setup here: steve's two CX-7 ports cabled to each other on one host.

Status:
- **CX-7 on steve: done** (`steve-cx7/`).
- **E810 post+poll on the FPGA testbed: dropped (2026-09-30).** The CX-7 numbers cover rdma_init for §2 and the simulator (the CX-7 is also the stronger baseline); the evaluation uses stock perf_rdma as the baseline NIC.

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

## Follow-up: like-for-like from the kernel's point of view, and where the GPU's time goes

The CPU-posted numbers above start when the CPU already knows the data is ready. In real use (the NCCL proxy model) a kernel produces the data, so it must hand off to a CPU thread and get a completion back. `proxy_b2.cu` measures that from the kernel:
1. A 1-thread kernel stamps `globaltimer` and writes a request flag into pinned host memory.
2. A CPU proxy thread polls it, `ibv_post_send`s an RDMA write (GPU buffer → GPU buffer, dma-buf MRs, mlx5_0 QP → mlx5_1 QP), polls the CQ, and writes a done flag.
3. The kernel spins on the done flag.

That is the same bracket as IBGDA put + `quiet` from a kernel.

| 8 B, "kernel decides to send → kernel knows it's done" | proxy + flags on socket 0 (GPU's) | on socket 1 (NIC's) |
|---|---|---|
| **CPU proxy posts (B2)** | **5.44 µs** (p99 6.40) | 5.82 µs (p99 7.36) |
| **GPU posts (B1, IBGDA put + quiet, 1 thread)** | **12.60 µs** | — |
| CPU time inside `ibv_post_send` | 0.075 µs | 0.33 µs |

Placement, 5 runs each (`proxy_b2_placement.csv`, 2026-10-01): medians 6.27 µs (CPU + memory on the GPU's socket), 5.70 (both on the NIC's socket), 5.70 (CPU on the NIC's, memory on the GPU's), 5.86 (CPU on the GPU's, memory on the NIC's). Runs of one placement spread by up to 1.1 µs, so placement moves the proxy by at most ~0.5 µs; the single-run 5.44 vs 5.82 above is within that noise.
| GPU time inside `nvshmem_putmem_nbi` (from `../m3-sm-share/gpu-posted`) | 6.0–7.8 µs | — |

Larger messages (proxy, socket 0): 64 KiB 8.93 µs, 1 MiB 69.2 µs, 4 MiB 262 µs, i.e. converging to the ~15–16 GB/s cap for NIC writes into HBM.

**Placement check.** Pinning the plain `ib_read_lat` run to each socket changes the CPU-posted GPU-memory read round trip by only 0.3 µs (3.53 vs. 3.27 µs; unpinned 3.35 µs). Host-memory buffers change more (3.40 vs. 2.48 µs). Placement does not explain the B1/B2 gap.

**Where the GPU's posting time goes** (`fence_cost.cu`, one GPU thread, median):

| operation | host page on socket 0 | on socket 1 |
|---|---|---|
| HBM store + `__threadfence_system()` | 0.58 µs | 0.58 µs |
| host-memory store + system fence | 0.58 µs | 0.64 µs |
| host-memory store, no fence (posted) | 0.03 µs | 0.03 µs |
| host-memory load (PCIe round trip) | 0.70 µs | 0.83 µs |

A system-scope fence costs about 0.6 µs, and a dependent read of queue state about as much again. The cross-socket part is only 0.06–0.13 µs per operation. One IBGDA put executes 4–5 atomics on queue-pair state, the WQE stores, 3 system fences, a doorbell-record store and the doorbell (M2). Most of these depend on the previous one, so they form a serial chain of 0.5–1 µs memory operations: that is the measured 6–8 µs. The CPU executes the same steps in its cache in 0.075 µs.

**Interpretation.** GPU-initiated RDMA removes the GPU→CPU handoff, which is worth about 2 µs here (5.44 µs vs. the 3.3–3.5 µs raw CPU round trip). It then pays more than that back in the SM's serialized post path. On this hardware a CPU proxy has lower single-message latency (5.4 vs. 12.6 µs). What GPU initiation buys is no CPU in the loop and parallel posting from many SMs, at the cost of held SMs (`../m3-sm-share/gpu-posted`: the GEMM keeps 90.5 / 72.7 / 48.9% with 8 / 16 / 20 posting CTAs).

## Why DeepEP uses GPU-initiated RDMA anyway

The measurements above are the CPU proxy's best case: one message at a time, whose destination and size the CPU already knows. MoE dispatch/combine is close to the opposite, and that is why DeepEP (V1 on NVSHMEM IBGDA, V2.5 on NCCL GIN GDAKI) posts from the GPU even though each GPU post costs more:

1. **Data-dependent communication known only on the GPU.** The top-k routing (which experts, hence how many bytes to which rank) is computed in a kernel. A CPU poster needs those counts on the host first: a D2H copy + synchronisation + kernel boundary per layer (the NCCL all-to-all pattern needs split sizes on the host). GPU initiation lets the routing kernel send each token the moment its destination is known.
2. **Fusion and overlap.** Layout, FP8 packing and sending live in one kernel. In low-latency mode the send kernel returns while the bytes are in flight (the receive "hook"), so compute continues. A proxy needs a host thread, flags, and a stream arrangement around it.
3. **CUDA Graphs.** Decode steps are captured as graphs. GPU-initiated communication lives inside the graph; a host proxy in the loop does not fit well.
4. **Message rate and fan-out.** Each token goes to up to 8 destinations across many ranks: thousands of small messages per GPU per layer. GPU posting spreads this over many SMs in parallel; a proxy funnels it through a few host threads and a GPU→host FIFO.

So DeepEP pays SM time per post (6–8 µs on this testbed) and holds SMs (M3c) to remove the host from a data-dependent control path. That trade-off is the gap Loom targets: triggered from inside the kernel like IBGDA (no host synchronisation), but executed by an engine outside the SMs (no fence chain, no held SMs).

### Dispatch-shaped benchmark on real hardware (steve)

DeepEP itself cannot run here, since it needs NCCL GIN, which needs two GPUs. Instead, both mechanisms are reproduced with identical routing and payload.

**Common setup:** a kernel computes top-8 routing over 256 experts for T tokens (deterministic hash). There are R = 8 destinations, and each token goes once to each distinct destination rank, as DeepEP deduplicates per rank. Tokens are 7168 B (FP8 DeepSeek-V3). The destinations are 8 regions of the peer's receive buffer (EP-shaped traffic over one peer). Grid: 8/20/32 CTAs × 256 threads. Timed with CUDA events around the kernel: routing → all bytes delivered; median of 20.

- **B1 (`dispatch_ibgda.cu`):** each warp routes its tokens and immediately puts each one to its destination (`nvshmemx_putmem_nbi_warp`, IBGDA, 24 QPs), then `quiet`.
- **B2 (`dispatch_proxy.cu`):**
  - The kernel routes and packs tokens contiguously per destination (SM copies).
  - The last CTA writes the 8 counts plus a flag into pinned host memory and spins on a done flag.
  - A CPU proxy thread (socket 0) reads the counts, posts RDMA writes (GPU → GPU, dma-buf MRs), waits for the last completion, and sets done.
  - Two posting modes: "block" (one write per destination) and "token" (one write per token).

| shape | B1 GPU-initiated | B2 proxy, per destination | B2 proxy, per token |
|---|---|---|---|
| decode, 128 tokens (683 messages, 4.9 MB) | **368–377 µs** (13.0–13.3 GB/s) | 489–518 µs (9.5–10.0 GB/s) | 507–532 µs (9.2–9.7 GB/s) |
| prefill, 4096 tokens (21,711 messages, 156 MB) | 10.69–10.79 ms (14.4–14.6 GB/s) | 10.53–11.14 ms (14.0–14.8 GB/s) | 10.66–11.25 ms |

(Ranges over 8/20/32 CTAs.)

Reading:
- **Decode: GPU-initiated is about 25% faster (120–140 µs).** B1 streams each token as soon as its route is known, so routing, packing and transfer overlap. B2 must finish routing and packing, hand the counts to the host, and only then transfer. This is point 1 above, measured.
- **Prefill: no difference.** Both run into steve's ~15 GB/s ceiling for NIC writes into HBM.
- **The CPU's message rate is not the limiter here.** Per-token posting costs about 20 µs more at 683 messages and about 120 µs at 21,711.
- **CTA count hardly matters for B1**, because it is bandwidth-bound on this loopback. With 7 KiB tokens, a 128-token decode is still 4.9 MB, i.e. bandwidth-dominated.

**Sweep: token size, batch size and GEMM load** (`dispatch_sweep_all.csv`: `--H 1024/7168`, `--load`).
- Batches of 16/32/128/1024/4096 tokens.
- With `--load`, a cuBLAS BF16 GEMM runs continuously on another stream of the same process, planned for 132−ctas SMs, so the dispatch kernel always finds free SMs.
- Median µs; ranges over 8/20 CTAs.

| H | tokens (messages) | B1 idle | B1 GEMM running | B2 packed, idle | B2 per-token, idle | B1 / B2 packed (idle) |
|---|---|---|---|---|---|---|
| 1 KiB | 16 (90) | **75–77** | 81–88 | 159–190 | 196–216 | 0.41–0.47 |
| 1 KiB | 32 (170) | **77–78** | 94–96 | 168–203 | 244–269 | 0.39–0.46 |
| 1 KiB | 128 (683) | 181–186 | 194–207 | 172–218 | 469–536 | 0.85–1.05 |
| 1 KiB | 1024 (5,446) | 1,097–1,100 | 1,115–1,126 | **488–520** | 2,216–2,414 | 2.11–2.25 |
| 1 KiB | 4096 (21,711) | 4,630–4,638 | 4,672–4,693 | **1,576–1,676** | 9,265–9,362 | 2.76–2.94 |
| 7 KiB | 16 (90) | **90–92** | 90–103 | 202–212 | 215–262 | 0.43–0.45 |
| 7 KiB | 32 (170) | **125** | 150–164 | 201–246 | 264–277 | 0.51–0.62 |
| 7 KiB | 128 (683) | 368–382 | 401–429 | 457–464 | 500–541 | 0.79–0.83 |
| 7 KiB | 1024 (5,446) | 2,732–2,746 | 2,862–2,902 | 2,638–2,723 | 2,678–2,778 | 1.00–1.04 |
| 7 KiB | 4096 (21,711) | 10,683–10,705 | 11,216–11,322 | 10,662–11,168 | 10,793–11,282 | 0.96–1.00 |

Three regimes:
1. **Small decode batches (16–32 tokens): GPU-initiated is about 2× faster** (75–125 vs. 160–245 µs). The proxy's fixed path (finish routing, hand off to the host, then send) dominates; this is DeepEP's low-latency regime.
2. **Many small tokens (1 KiB × ≥1024): the proxy with packing wins by 2.1–2.9×.** It sends 8 large writes instead of thousands of 1 KiB puts. Without packing the proxy is about 2× slower than B1: one CPU thread posts about 2.3 M writes/s (21,711 in 9.3 ms), while GPU warps post about 4.7 M puts/s (21,711 in 4.6 ms). Per-message posting is the cost on both sides, and batching wins.
3. **Large tokens and batches:** both are bound by the ~15 GB/s NIC → HBM ceiling.

**Under GEMM load**, B1 slows by up to about 30% at small batches (e.g. 77 → 94 µs, 32 × 1 KiB) and about 5% at large ones. B2 is unchanged within noise (its packing kernel also shares SMs, but its sends do not).

**Implication for Loom:** the best dispatch wants GPU triggering (regime 1) *and* engine-side gathering into few large transfers (regime 2), without SM copies for packing and without the SM post chain (M3c). That is a GPU-triggered engine with scatter-gather descriptors, the same requirement as the copy-engine granularity result in `../m3-sm-share/gpu-interference`.

**Caveats:**
- B2 is the unpipelined NCCL-style handoff. A proxy fed in chunks could overlap too, at the cost of more GPU↔host round trips.
- Everything shares one NIC and one GPU (loopback).
- The IBGDA per-put SM cost (6–8 µs) is hidden here because many warps post in parallel and the NIC is the bottleneck. It shows up in held SMs (M3c), not in dispatch time.

**Like for like: GPU-initiated with the same packing** (`dispatch_ibgda --block`, `dispatch_ibgda_block_H{1024,7168}_load{0,1}.csv`, 2026-10-01). Regime 2 above compared a *packed* proxy with *per-token* GPU puts. With `--block` the kernel routes and packs per destination exactly like `dispatch_pack`, and the last CTA puts one block per destination (8 warp puts) + quiet. 20 CTAs, idle, µs:

| H | tokens | B1 packed | B2 packed | B2 / B1 | B1 per token | B2 per token | B2 / B1 |
|---|---|---|---|---|---|---|---|
| 1 KiB | 16 | 43.1 | 189.8 | 4.4 | 77.2 | 196.1 | 2.5 |
| 1 KiB | 32 | 44.2 | 203.1 | 4.6 | 78.4 | 268.9 | 3.4 |
| 1 KiB | 128 | 80.1 | 172.4 | 2.2 | 180.9 | 468.8 | 2.6 |
| 1 KiB | 1024 | 413.5 | 488.2 | 1.2 | 1,100.4 | 2,413.7 | 2.2 |
| 1 KiB | 4096 | 1,597.4 | 1,575.7 | 0.99 | 4,637.8 | 9,265.2 | 2.0 |
| 7 KiB | 16 | 86.3 | 212.4 | 2.5 | 92.1 | 214.9 | 2.3 |
| 7 KiB | 128 | 367.1 | 464.4 | 1.3 | 367.7 | 499.7 | 1.4 |
| 7 KiB | 4096 | 10,947 | 10,662 | 0.97 | 10,705 | 10,793 | 1.0 |

With the same packing, GPU initiation is never slower beyond 3% (the NIC-bandwidth-bound points) and 1.6–4.6× faster at 16–32 tokens. The "proxy wins by 2.1–2.9×" of regime 2 was packing, not the poster. Figure 1b plots both paths packed.

### Copy engine triggered from a kernel (`ce_triggered.cu`)

A copy engine cannot be *created* from device code, but a pre-enqueued copy can be *triggered* by a kernel.
1. The host enqueues `cuStreamWaitValue32(flag ≥ i)` followed by a copy (`cudaMemcpyBatchAsync` + `PreferOverlapWithCompute`, i.e. a real copy engine), 200 pairs per batch.
2. A 1-thread kernel writes the payload's tail, stamps `globaltimer`, writes flag = i, and spins until the tail appears at the destination.

Everything is timed on one GPU clock: trigger → the GPU front end releases the wait → the copy engine copies → data landed. Median of 1,900 (10 launches × 190).

| size | device → device | device → host (pinned) |
|---|---|---|
| 8 B – 4 KiB | **2.50–2.59 µs** (p99 2.8–3.0) | 10.8–10.9 µs |
| 64 KiB | 3.46 µs | 12.5 µs |
| 1 MiB | 15.3 µs | 47.2 µs |
| 4 MiB | 52.0 µs (about 80 GB/s) | 157 µs (about 27 GB/s, PCIe ×8) |

- **An off-SM engine triggered from a kernel delivers local data in 2.5 µs.** The IBGDA put + completion takes 12.6 µs and the kernel-timed CPU proxy 5.4 µs. The kernel's own cost is one store plus a system fence (about 0.6 µs, `fence_cost.cu`), not the 6–8 µs IBGDA post chain. This is the closest existing analogue of a GPU-triggered Loom engine, but local only: the copy engine cannot reach the network.
- **The device → host figures are probably inflated** by the measurement: the kernel detects arrival by continuously reading pinned host memory over PCIe (about 0.7–0.8 µs per read), which likely slows the copy engine's writes. Do not quote them as copy-engine latency.
- **Trap (the earlier hang):** with CUDA 12's default lazy module loading, the trigger kernel's *first* launch loads its module, and that load synchronises with the device, which is waiting on the kernel. Deadlock. The program sets `CUDA_MODULE_LOADING=EAGER`. The debug programs `trig_dbg*.cu` in `~/loom-experiments/latency` showed that enqueuing waits+copies alone never blocks.

### The NIC's own share and the unified-contract bound (`nic_post.cu`)

Of the proxy's 5.44 µs, the NIC's post → completion is the largest part. `nic_post.cu` times it for the same 8 B GPU → GPU write (one CPU thread posts and polls; ops 2 µs apart; proxy on NUMA 0) while removing the NIC's PCIe reads one at a time. The data is either in GPU memory, in host memory, or inline (inside the work request). The work request is either pushed by BlueFlame (the CPU writes it into the NIC's BAR) or fetched by the NIC (`MLX5_SHUT_UP_BF=1`). rdma-core's `MLX5_POST_SEND_PREFER_BF` defaults to on, so `proxy_b2` already used BlueFlame. Medians of 3 runs (`nic_post_numa0.csv`, each run 20,000 ops; the runs agree within 0.15 µs):

| 8 B, post return → CQE seen | BlueFlame | NIC fetches the work request |
|---|---|---|
| data in GPU memory (as `proxy_b2`) | 4.12 µs | 5.13 µs |
| data in host memory | 3.60 µs | 4.58 µs |
| inline (no payload read) | **2.67 µs** | 3.71 µs |

A work-request fetch costs the NIC 1.0 µs, a payload read from GPU memory 1.45 µs (host memory 0.9 µs). Inline + BlueFlame (the request and its data in one MMIO write) is the NIC's floor: 2.67 µs.

**Unified-contract bound** (Figure 1c band): if a kernel started a remote transfer like a local one, it would pay a store + system fence (0.58 µs, `fence_cost`), the NIC's own time, and one load of a completion word in host memory (0.70 µs). That gives 3.9 µs when the store carries the data (NIC floor) and 5.4 µs when the NIC reads the payload from GPU memory, against 5.44 µs for the proxy (which also needs a core) and 12.6 µs for IBGDA. Not counted: the flight of a GPU store to the NIC (the CPU's MMIO flight is inside the NIC's time here) and any engine work beyond what the CX-7 does.

### GPU-initiated with DeepEP's own post path (`deepep_post.cu`, 2026-10-01)

Everything GPU-initiated above uses NVSHMEM's generic put. DeepEP ships a leaner post path (V1 a56d615, `csrc/kernels/legacy/ibgda_device.cuh`): warp-parallel WQE writes, a gpu-scope `__threadfence()`, gpu-scope release stores for the doorbell record and the doorbell (no system-scope fence), and one doorbell per 4 messages per QP. `deepep_post.cu` runs it, unmodified except one line, on the same IBGDA QPs as NVSHMEM's put. The one line: DeepEP indexes RC QPs PE-major (older NVSHMEM), and 3.6.5 lays them out QP-major (`rcs[id * npes + pe]`). `build_deepep_post.sh` patches that index into a copy. Unpatched, it picks the never-created QP to itself and faults.

Single message, one warp, put → completion on that QP, NVSHMEM default QPs (as the perftest), medians of 3 runs (`deepep_post_lat.csv`):

| 8 B | post (SM time) | put + completion |
|---|---|---|
| NVSHMEM `putmem_nbi_warp` + `quiet` | 7.33 µs | 14.27 µs |
| DeepEP `put_nbi_warp<true>` + per-QP quiet | **3.07 µs** | **9.22 µs** |

Dispatch, per token (DeepEP LL style: QP = destination, message index = slot, 24 QPs/PE, `NVSHMEM_QP_DEPTH=1024`), 20 CTAs, idle, µs (`deepep_post_dispatch_H{1024,7168}.csv`):

| H | tokens | DeepEP | NVSHMEM per token | NVSHMEM packed | best proxy |
|---|---|---|---|---|---|
| 1 KiB | 16 | **35.3** | 82.4 | 43.1 | 189.8 |
| 1 KiB | 32 | **40.6** | 82.8 | 44.2 | 203.1 |
| 1 KiB | 128 | 156.4 | 173.3 | **80.1** | 172.4 |
| 7 KiB | 16 | **56.9** | 90.0 | 86.3 | 212.4 |
| 7 KiB | 128 | **331.9** | 368.8 | 367.1 | 464.4 |

- DeepEP skips the WQ slot check and requires `NVSHMEM_QP_DEPTH >= (tokens + 1) * 2` messages in flight per QP (`deep_ep/buffers/legacy.py`), so the deepep mode runs 16–128 tokens only. A first try at 1024/4096 tokens overran the QPs, reported an impossible 26 GB/s and hung at exit.
- With 24 QPs/PE, `nvshmem_quiet` polls every QP and the NVSHMEM single-message number inflates to ~34 µs. The latency test therefore uses NVSHMEM's default QP count.
- Figure 1 now takes each path at its best measured variant: 1c GPU-initiated = DeepEP (9.2 µs, 3.1 µs SM per post); 1b = min over DeepEP / NVSHMEM per token / NVSHMEM packed, and min over proxy packed / per token. Best GPU vs best proxy at 16–32 tokens: 2.1–5.4×. The unified bound vs the best GPU variant at 16 tokens: 1.2–1.9×.

### Unified-contract bound for dispatch: the GPU-triggered copy engine (`dispatch_ce.cu`)

The same dispatch (routing, token sizes, batches, 8/20 CTAs, GEMM load) started the way a kernel starts a local transfer: the kernel's last CTA writes a flag in HBM that releases copies pre-enqueued on a copy-engine stream (`cuStreamWaitValue32` → `cudaMemcpyBatchAsync`, PreferOverlapWithCompute → `cuStreamWriteValue32` done), then spins on done. The destination is a buffer in the same GPU's HBM (a local peer). "token" = the kernel only routes and the engine copies each token message from the token buffer (no SM copies); "block" = the kernel packs per destination on SMs and the engine runs 8 copies. The copy list is built on the host from the same deterministic routing before the trigger; it stands in for an engine that takes descriptors from the kernel. Timed like B1/B2: CUDA events around the kernel, median of 20.

| H | tokens | CE token, idle | CE token, GEMM | CE block, idle | B1 idle | B2 packed idle |
|---|---|---|---|---|---|---|
| 1 KiB | 16 | **15.5** | 21.1 | 20.0 | 77.2 | 189.8 |
| 1 KiB | 128 | **26.1** | 28.8 | 27.5 | 180.9 | 172.4 |
| 1 KiB | 4096 | **347** | 352 | 375 | 4,638 | 1,576 |
| 7 KiB | 16 | **22.4** | 28.3 | 40.4 | 92.1 | 212.4 |
| 7 KiB | 128 | **73.9** | 77.1 | 92.3 | 367.7 | 464.4 |
| 7 KiB | 4096 | **1,847** | 1,857 | 2,197 | 10,705 | 10,662 |

(20 CTAs, µs; all rows in `dispatch_ce_H{1024,7168}_load{0,1}.csv`.)

- Token mode beats block mode: the engine runs 21,711 pre-built token copies at 64 GB/s (1 KiB) and 84 GB/s (7 KiB). The ~1.5 M copies/s in `../m3-sm-share/gpu-interference` is the driver building the copies on the host (its timer includes the enqueue), not the engine.
- The local copy engine is not held to a NIC. Figure 1b's bound is therefore `max(CE token time, bytes / 14.8 GB/s) + 2.67 µs`: the best NIC → HBM rate in the RDMA sweeps and the NIC floor from `nic_post`. Against the best of B1/B2 it is 4.2× (1 KiB) and 2.0× (7 KiB) faster at 16 tokens, 3.5× / 1.1× at 128, and equal in prefill, where the NIC's bandwidth binds every path.
- Under GEMM load the CE path slows by up to 6 µs at small batches (the routing kernel and the copies share the GPU with the GEMM) and stays at least 3× ahead of B1 (B1 under load: 80.8 vs 21.1 µs at 16 × 1 KiB).
- TRAP: under `--load`, cuBLAS loads a new GEMM kernel the first time an SM target is used. That load waits for an idle device while the copy stream waits on the kernel's flag, and the process deadlocks at 0% GPU utilisation. The load thread now runs every SM target once before the sweep.

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
NUMA=$(nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
cd ~/loom-experiments/latency && $NUMA -N 0 -m 0 ./proxy_b2 20000 > proxy_b2_numa0.csv
cd ~/loom-experiments/latency && $NUMA -N 0 -m 0 ./fence_cost > fence_cost_numa0.csv
cd ~/loom-experiments/latency && $NUMA -N 0 -m 0 ./ce_triggered < /dev/null > ce_triggered_idle.csv
~/loom-experiments/latency/run_nic_post.sh                                           # nic_post_numa0.csv, about 1 min
~/loom-experiments/latency/run_dispatch_ce.sh                                        # dispatch_ce_H*_load*.csv, about 15 min
sudo ~/loom-experiments/gpu-posted/run_dispatch.sh < /dev/null                       # B1 sweep: dispatch_ibgda_H{1024,7168}_load{0,1}.csv
sudo ~/loom-experiments/gpu-posted/run_dispatch_block.sh < /dev/null                 # B1 packed: dispatch_ibgda_block_H*_load*.csv
~/loom-experiments/gpu-posted/build_deepep_post.sh && sudo ~/loom-experiments/gpu-posted/run_deepep_post.sh   # deepep_post_*.csv
~/loom-experiments/latency/run_dispatch_proxy.sh                                      # B2 sweep: dispatch_proxy_H*_load*.csv
cd ~/loom-experiments/latency && $NUMA -N 0 -m 0 ./dispatch_proxy < /dev/null > dispatch_proxy.csv
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
- `proxy_b2.cu`: B2 as used for GPU data (kernel → CPU proxy → RDMA → kernel), kernel-timed.
- `proxy_b2_numa{0,1}.csv`: its outputs with the proxy on each socket.
- `fence_cost.cu`: GPU fence, host-store and host-load costs.
- `ce_triggered.cu`: the kernel-triggered copy engine.
- `ce_triggered_idle.csv`: its outputs.
- `fence_cost_numa{0,1}.csv`: its outputs with the host page on each socket.
- `build.sh`: builds `ce_latency`, `ce_triggered`, `proxy_b2`, `fence_cost`, `dispatch_proxy`, `nic_post` and `dispatch_ce` (the verbs programs link nixpkgs rdma-core and libcuda).
- `nic_post.cu`, `run_nic_post.sh`: the NIC's post → completion with its PCIe reads removed one at a time.
- `nic_post_numa0.csv`: its outputs.
- `dispatch_ce.cu`, `run_dispatch_ce.sh`: the GPU-triggered copy-engine dispatch (the unified-contract bound).
- `dispatch_ce_H{1024,7168}_load{0,1}.csv`: its outputs.
- `dispatch_ibgda.cu`, `run_dispatch.sh`, `build_gpu_posted.sh`: the B1 dispatch benchmark (NVSHMEM; root for memlock). The runnable copy is in `~/loom-experiments/gpu-posted`.
- `dispatch_ibgda.csv`: its outputs.
- `run_dispatch_block.sh`, `dispatch_ibgda_block_H{1024,7168}_load{0,1}.csv`: B1 with `--block` (packed per destination, like B2 block).
- `deepep_post.cu`, `build_deepep_post.sh`, `run_deepep_post.sh`: DeepEP's post path vs NVSHMEM's put (needs DeepEP V1 a56d615 headers in `deepep-include/`, see the build script).
- `deepep_post_lat.csv`, `deepep_post_dispatch_H{1024,7168}.csv`: its outputs.
- `dispatch_proxy.cu`: the B2 dispatch benchmark (verbs + CPU proxy).
- `dispatch_proxy.csv`: its outputs.
- `run_dispatch_proxy.sh`: the B2 sweep (H × load).
- `dispatch_sweep_all.csv`: all sweep outputs, B1 and B2.
