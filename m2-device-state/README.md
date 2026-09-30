# M2: device-side transport state of GPU-initiated RDMA

Detailed evidence (full inventories, per-step instruction paths, file:line citations for every entry): [results.md](results.md).

## Question

When RDMA is GPU-initiated, how much NIC transport state (queue pairs, WQE and CQ rings, doorbell records, mapped doorbell pages) does each GPU hold, and how much of the NIC driver's post/doorbell/poll path runs on SMs? This measures the cost of putting NIC state and the NIC's post path on the accelerator, which Loom moves to an off-accelerator engine.

## Methodology

**Sources** (clones under `../src/`):

| System | Ref | Commit | Date |
|---|---|---|---|
| NVSHMEM | tag `v3.8.0-0` | `270759e5481b16ef5a71930e1d9b8df184cd7072` | 2026-09-22 |
| NCCL (GIN GDAKI, vendored DOCA GPUNetIO) | tag `v2.32.3-1` | `12df1a11afad322be5a204a2db890161cbf8131d` | 2026-09-17 |
| DeepEP main (V2.5, NCCL GIN backend) | `main` HEAD | `93eb6eb238127e96c6d7a4a625a6dad158348509` | 2026-09-30 |
| DeepEP V1 (last NVSHMEM/IBGDA code) | `def8651^`, parent of "DeepEP V2.5 (#763)", which removed V1 | `a56d6156febcd9976e55adc85b5155bfac9f28f8` | 2026-09-16 |

`../src/DeepEP-v1-last` is a git worktree of `../src/DeepEP` at `a56d615`. DeepEP V1 needs NVSHMEM ≥ 3.3.9; the IBGDA QP, CQ and key structs have the same sizes in the 3.6.5 headers (checked by diff), so 3.8 sizes apply except for the 3.8-only batch-RMA bitmap.

**Tools:** g++ (GCC) 15.2.0 from `nix shell nixpkgs#gcc`; rdma-core `mlx5dv.h` from `/nix/store` (see caveat 8 on which version); Python 3.13.15 for `totals.py`. No GPU or NIC was used.

**What was counted and how:**
- Inventory of every structure/buffer the GPU holds or maps, read from source (location, size, multiplicity).
- Struct sizes measured by compiling `sizes.cpp` (NVSHMEM headers + `mlx5dv.h`) and `sizes_gdaki.cpp` (NCCL's vendored DOCA GPUNetIO headers) on the host with `sizeof`/`offsetof`. Output: `sizes.out`.
- Per-configuration totals computed by `totals.py` from code constants and the measured sizes. Output: `totals.out`. Counts are bytes requested from the allocator for rings, doorbell records and descriptors.
- Instruction path of one put read from device code at CUDA/PTX level (atomics, stores, fences, MMIO writes per put).

**Assumptions and judgment calls:**
1. 1 NIC per GPU (`ndev` = 1, `ginCommCount` = 1), i.e. a rail-optimized 8-GPU/8-NIC node; 8 GPUs per node; NVLink domain = node (`lsaSize` 8).
2. NVSHMEM defaults unless DeepEP overrides: `IBGDA_NUM_RC_PER_PE` = 2, `QP_DEPTH` = 1024, 1 DCI, 2 DCTs, rings in GPU memory (`FORCE_NIC_BUF_MEMTYPE=gpumem`), GPU rings the doorbell (`NIC_HANDLER=auto`). Symmetric heap 2 GiB (affects only the rkey table).
3. DeepEP V1 LL: 256 routed experts (DeepSeek-V3), so RC QPs per peer = 256 / EP size. DeepEP V1 normal: 24 QPs per peer (constructor default), NVSHMEM PEs = one per node.
4. DeepEP V2.5: contexts = `num_allocated_qps` = 129 (hybrid, non-CX-8), 65 (hybrid, MT4131, which we take to be ConnectX-8), 17 (direct); queue depth 1024; `ginSignalCount = num_ranks + 4`; no counters; RAIL connection in hybrid mode, FULL in direct mode. Extra contexts created by Engram/PP/Bucket buffers are not counted.
5. GDAKI page size 4 KiB (x86, inferred) for slab alignment.
6. NVSHMEM's 256 KiB `ibuf` per RC QP is transport staging, not NIC-protocol state; it is reported both included and excluded.
7. NVSHMEM `cudaMalloc(size + 64 KiB − 1)` alignment slack is reported separately as an upper bound.

## Results

**Per QP (depth 1024)** (`totals.out`, `sizes.out`):
- NVSHMEM IBGDA RC QP: WQ 65,536 + CQ 65,536 + ibuf 262,400 + DBRs 16 + descriptors 256 = **393,744 B (384.5 KiB)**; of that **131,344 B** is NIC-protocol state excluding ibuf. Up to 5 × (64 KiB − 1) alignment slack more. One NIC doorbell (UAR) page mapped into the GPU per QP.
- NCCL GIN GDAKI QP: SQ 65,536 + CQ 69,632 + 2 DBR slices 4,096 each = **143,360 B (140 KiB)**, plus a 296 B device descriptor and one GPU-mapped doorbell. At NCCL's default depth 128: 28,672 B.
- Device struct sizes: `nvshmemi_ibgda_device_qp_t` 184 B (of which `mvars` 96), `nvshmemi_ibgda_device_cq_t` 72, `nvshmemi_ibgda_device_state_t` 8,384 (8,192 constmem caches); `doca_gpu_dev_verbs_qp` 296, `ncclGinGdakiGPUContext` 88. mlx5 WQEBB and CQE are 64 B.

**NVSHMEM defaults** (GPU HBM, struct and ring bytes):

| Configuration | PEs | RC/peer | RC QPs per GPU | GPU-mapped doorbell pages | GPU HBM transport state | + alloc slack (≤) |
|---|---|---|---|---|---|---|
| NVSHMEM default, 2×8 GPUs | 16 | 2 | 30 | 31 | 11.7 MiB | 9.4 MiB |
| NVSHMEM default, 16×8 | 128 | 2 | 254 | 255 | 95.9 MiB | 79.4 MiB |
| NVSHMEM default, 32×8 | 256 | 2 | 510 | 511 | 192.2 MiB | 159.4 MiB |

**DeepEP V1 on NVSHMEM IBGDA:**

| Mode | Nodes × 8 | NVSHMEM PEs | RC/peer | RC QPs per GPU | Doorbell pages mapped | GPU HBM transport state | + slack (≤) |
|---|---|---|---|---|---|---|---|
| LL | 2 (EP16) | 16 | 16 | 240 | 241 | 90.7 MiB | 75.0 MiB |
| LL | 16 (EP128) | 128 | 2 | 254 | 255 | 95.9 MiB | 79.4 MiB |
| LL | 32 (EP256) | 256 | 1 | 255 | 256 | 96.3 MiB | 79.7 MiB |
| Normal | 2 (EP16) | 2 | 24 | 24 | 25 | 9.4 MiB | 7.5 MiB |
| Normal | 16 (EP128) | 16 | 24 | 360 | 361 | 135.8 MiB | 112.5 MiB |
| Normal | 32 (EP256) | 32 | 24 | 744 | 745 | 280.2 MiB | 232.5 MiB |

**DeepEP V2.5 on NCCL GIN GDAKI** (QPs per GPU = C × (P + 1); number of GPU-mapped doorbell UARs = number of QPs):

| Mode (contexts C) | EP16 (2 nodes) | EP128 (16 nodes) | EP256 (32 nodes) |
|---|---|---|---|
| hybrid, CX-7 default (129) | 387 QPs, **53.6 MiB** | 2,193 QPs, **305 MiB** | 4,257 QPs, **592 MiB** |
| hybrid, CX-8 default (65) | 195 QPs, 27.0 MiB | 1,105 QPs, 154 MiB | 2,145 QPs, 298 MiB |
| direct (17) | 289 QPs, 39.6 MiB | 2,193 QPs, 301 MiB | 4,369 QPs, 599 MiB |

**GPU work per put** (from results.md §5; read from source, not SASS):

| | NVSHMEM IBGDA 3.8 (RC) | DeepEP V1 own IBGDA path | NCCL GIN GDAKI 2.32 |
|---|---|---|---|
| Per-QP GPU HBM (D=1024) | 384.5 KiB (128.3 KiB rings/DBR/desc + 256.3 KiB ibuf) + ≤320 KiB alloc slack | same (NVSHMEM-allocated) | 140 KiB + 296 B desc |
| GPU-mapped NIC doorbell | 1 UAR per QP | same | 1 UAR per QP |
| WQE bytes / store instr. per put | 48 B / 12 × 32-bit | 48 B / 3 × 128-bit | 48 B / 3 × 128-bit (112 B / 7 with signal) |
| Global atomics/CAS per put | 4–5 (+ lock) | 1 add + 1 CAS; every 4th msg + lock CAS + max | 4 (+ unlock) |
| Fences per put | 3 × fence.gpu + 3–4 × fence.cta | 1 × fence.gpu + 2 × fence.cta (on DB) | 4–5 (release/acquire gpu) |
| Doorbell record + MMIO per put | 1 DBR + 1 MMIO (batched ≤32) | 1 DBR + 1 MMIO every 4th msg | 1 DBR + **2 MMIO** |
| Completion | CQ poll on every QP in `quiet` | poll per (peer, qp) | CQ poll on every peer QP in `flush` |

Key sources for the headline numbers (paths as in results.md): NVSHMEM per-QP allocation `ibgda.cpp:2186-2187, 1438-1453, 2280-2285`; QP count `ibgda.cpp:3440-3446`, `env.h:177`; put path `ibgda_device.cuh:2050-2062, 1561-1585, 1551-1558`. GDAKI QP count `gin_host_gdaki.cc:618-627`; ring sizes `doca_gpunetio_high_level.cpp:1400-1402`; put path `doca_gpunetio_dev_verbs_qp.cuh:94-128, 524-549`. DeepEP V2.5 context count `deep_ep/buffers/ep.py:327-335`, `csrc/kernels/comm/context.cpp:122-164`; GDAKI required at `context.cpp:129-134`.

## Caveats / fairness

1. Sizes are bytes requested from the allocator. Resident HBM can be higher (NVSHMEM alignment slack shown separately; NCCL cuMem rounds up to allocation granularity).
2. NVSHMEM's 256 KiB `ibuf` is staging for fetching AMOs/`g`, not NIC-protocol state; without it an RC QP is 128.3 KiB.
3. NIC count multiplies QPs; 1 NIC per GPU is assumed. Mapping MT4131 to ConnectX-8 is an inference.
4. The 256-expert LL configuration for DeepEP V1 is an assumption; the code requires `num_qps_per_rank = num_experts / num_ranks`.
5. The NVSHMEM totals include the 3.8-only batch-RMA bitmap (≤ 416 KiB); DeepEP V1 ran on 3.3.9+ where it does not exist.
6. The host still creates and connects every QP, and holds SRQ/DCT/recv-CQ objects (NVSHMEM); NVSHMEM keeps a minimal proxy thread and NCCL an error-query path. The claim is that the GPU **holds** the per-QP datapath state and **executes** the post/doorbell/poll protocol, not that it does all of RDMA.
7. Both stacks ship CPU-assisted modes (NVSHMEM `NIC_HANDLER=cpu`, DOCA `CPU_PROXY`, NCCL GIN PROXY backend). In the first two the WQE ring and indices stay on the GPU. DeepEP V2.5 compiles the proxy backend out and requires GDAKI.
8. results.md states the mlx5 headers are from rdma-core 64. `run_sizes.sh` picks `ls -d /nix/store/*rdma-core-6*-dev/include | sort | tail -1`, which sorts by store hash and on this machine selects rdma-core 62.0. Compiling `sizes.cpp` against the rdma-core 64.0 headers gives identical output (checked 2026-09-30).
9. Instruction counts are at CUDA/PTX level; spin loops (ready CAS, lock, CQ poll) are counted once and repeat under contention. `quiet`/`flush` cost grows with the number of QPs, per sync rather than per put.
10. The DeepEP "V1" ref here (`a56d615`, parent of V2.5) differs from the one used in M3 (`567632d`, parent of the V2.0 release). Both contain the V1 NVSHMEM code; M2 uses the later one.

## Candidate claims for the paper (from results.md §7)

1. "In DeepEP's current NCCL-GIN (GDAKI) backend with its default 129 GIN contexts, each GPU of a 256-GPU EP job holds **4,257 RC QPs**. That is **≈592 MiB of HBM** for SQ/CQ rings and doorbell records, plus 4,257 NIC doorbell pages mapped into the GPU, all so SMs can drive the NIC." (§4.1: 129 × 33 QPs × 140 KiB; context.cpp:135-137, ep.py:332, gdaki.cc:618-627, hl.cpp:1400-1402)
2. "Posting a single RDMA write from an SM in NCCL GIN GDAKI takes 4 global atomics, 3 WQE stores, a doorbell-record write, 2 MMIO doorbell writes and 4–5 memory fences. NVSHMEM IBGDA takes 4–5 atomics, 12 32-bit WQE stores, 1 MMIO doorbell and 3 device-scope `__threadfence`s. The accelerator is literally running the NIC driver's post-send path." (§1.3, §3.3)
3. "With NVSHMEM IBGDA every RC QP costs the GPU 384 KiB (128 KiB of it WQE/CQ rings). DeepEP V1's normal mode (24 QPs/peer) therefore held **744 QPs ≈ 280 MiB** per GPU at EP256, and even its low-latency mode held ≈ 255 QPs ≈ 96 MiB regardless of EP size." (§2.3)

## Reproduce

Run from this directory (a few seconds each). `run_sizes.sh` builds two host binaries, `sizes` and `sizes_gdaki`, in this directory; the last command removes them.

```sh
python3 totals.py | diff - totals.out
./run_sizes.sh | diff - sizes.out
rm -f sizes sizes_gdaki
```

No output from `diff` means the stored outputs are reproduced (verified 2026-09-30). Drop `| diff - <file>` to see the output itself.

## Files

- `README.md`: this file.
- `results.md`: full inventories, instruction paths, host-side fairness notes, with file:line citations.
- `sizes.cpp`: `sizeof`/`offsetof` of NVSHMEM IBGDA device structs, mlx5 segments and the 3.8 batch-RMA bitmap.
- `sizes_gdaki.cpp`: `sizeof` of NCCL GIN GDAKI / DOCA GPUNetIO device structs.
- `run_sizes.sh`: builds and runs both size programs with nix g++ and rdma-core headers.
- `sizes.out`: output of `run_sizes.sh` (both programs).
- `totals.py`: per-configuration QP counts and HBM totals for NVSHMEM, DeepEP V1 and DeepEP V2.5.
- `totals.out`: output of `totals.py`.
