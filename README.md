# Loom motivation experiments

Each experiment folder has a `README.md` (question, methodology, results, caveats, candidate claims, reproduce commands) and a detailed evidence file with file:line citations.

| # | experiment | folder | status | headline result |
|---|---|---|---|---|
| M1 | NCCL scale-up vs scale-out LoC split | [m1-nccl-loc/](m1-nccl-loc/) | done | NCCL 2.32: 46.9% of 113,222 NCCL-authored lines is transport-specific (36,580 scale-out, 16,566 scale-up; 43–51% across attribution variants), up from 32.4% in 2.18; transport-specific code grew 6.6× vs 3.6× for shared code. |
| M2 | Device-side transport state of GPU-initiated RDMA | [m2-device-state/](m2-device-state/) | done | DeepEP V2.5 on NCCL GIN GDAKI (129 contexts) holds 4,257 RC QPs and ≈592 MiB of HBM rings/doorbell records per GPU at EP256, plus 4,257 GPU-mapped NIC doorbells; one put costs 4 atomics, 3 WQE stores, 1 doorbell-record store, 2 MMIO doorbells and 4–5 fences on the SM. |
| M3 | SM share | [m3-sm-share/](m3-sm-share/) | analysis done; GPU interference (steve) done; real-RDMA variant blocked (no cabled GPU+CX host) | DeepEP V2.5 hybrid comm kernel uses 4–16 SMs (3–12% of an H100); direct/V1 20–72 (15–55%). On an H200, holding 8–20 SMs to move 50 GB/s costs a BF16 expert GEMM 9.5–51% of its throughput; the copy engine moving the same bytes costs 0.1%. GPU-posted RDMA (IBGDA) costs exactly what holding the SMs costs (90.5/72.7/48.9% at k=8/16/20) at 6–8 µs of SM time per put; CPU-posted leaves the GEMM at 100.2% for one host core. Chakra: NCCL 40–44% of a Mixtral step at ~30 SMs. |
| M4 | Who moves the bytes on the CPU/FPGA testbed (dgemm+memcpy vs E810 RDMA) | – | dropped (2026-09-30): the GPU versions in M3 answer this for the paper's target; the evaluation uses stock perf_rdma (Coyote's RDMA stack) as the baseline NIC | – |
| M5 | Initiation cost (rdma_init): CPU-posted vs GPU-posted RDMA vs copy engine | [m5-rdma-init/](m5-rdma-init/) | done (CX-7 on steve; the E810 part was dropped, see M4) | Same NIC and GPU buffers: GPU-posted (IBGDA) read round trip 14.1 µs vs CPU-posted 3.35 µs; GPU-posted puts slow by 10–22% next to a GEMM, CPU-posted is unchanged; plain cudaMemcpyAsync D2D hits p99 526 µs behind a GEMM (runs on SMs), the copy engine 8.9 µs. Dispatch measured to the receiver's signal (16 × 1 KiB, 20 CTAs): local load/store path 4.1 µs and 55 CTA-µs; GPU-initiated RDMA 28.7 µs / 101 CTA-µs (ordered signal) and 34.7 µs / 481 CTA-µs (flush, 313 of them waiting for completions). |

## Source clones (`src/`)

| directory | ref | commit |
|---|---|---|
| `src/nccl` | NCCL `v2.32.3-1` | `12df1a11` |
| `src/nccl-v2.18.5` | NCCL `v2.18.5-1` (git worktree of `src/nccl`) | `559b70f8` |
| `src/nvshmem` | NVSHMEM `v3.8.0-0` | `270759e` |
| `src/DeepEP` | DeepEP `main` | `93eb6eb` |
| `src/DeepEP-v1-last` | DeepEP last commit with V1 code, parent of V2.5 (git worktree of `src/DeepEP`) | `a56d615` |
| `src/DeepEP-0sm` | separate DeepEP clone at `main` with PR/branch refs fetched (`pr347`, `pr453`, `origin/hybrid-ep`, `origin/antgroup-opt`, ...) | `93eb6eb` |

Experiment READMEs reference these clones by relative path (`../src/...`), so run their commands from inside the experiment folder.
