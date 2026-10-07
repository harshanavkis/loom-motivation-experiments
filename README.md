# Loom motivation experiments

Everything behind the Loom paper's Introduction and Section 2 (Background and Motivation): source
analyses, production-trace analyses, and microbenchmarks on an H200 with a ConnectX-7 NIC. This page
is the reproduction guide. Each experiment folder's `README.md` has the methodology, the results,
the caveats and the evidence (file:line citations).

- [Quick start](#quick-start)
- [What produces each figure, table and number](#what-produces-each-figure-table-and-number)
- [The GPU host](#the-gpu-host) (hardware, one-time setup, after every reboot, what each setting is for)
- [Local analyses and traces](#local-analyses-and-traces)
- [Checking a rerun](#checking-a-rerun) and [known gaps](#known-gaps)
- [Layout](#layout) and [source clones](#source-clones-src)

## Quick start

Three machines are involved: a **repo host** (this checkout; nix, no GPU needed), a **GPU host** (an
NVIDIA Hopper GPU and a ConnectX-7 whose two ports are cabled to each other; ours is `steve`), and
the paper (`~/loom-paper`).

```sh
# repo host: everything that needs no GPU, then the figures (2-3 minutes; reproduces the committed outputs)
scripts/fetch_sources.sh                   # src/: NCCL, NVSHMEM, DeepEP, DeepGEMM at the analysed commits
scripts/analyze_local.sh --traces          # M1, M2, M6, Chakra traces, summaries of the GPU runs
scripts/make_plots.sh ~/loom-paper/plots   # Fig 1 (loom-mix) and Fig 3 (loom-costs); prints every number
rm ~/loom-paper/plots/*.png                # the paper uses the PDFs

# GPU runs: deploy, build, set up, run, bring the results back
scripts/deploy_gpu_host.sh steve                          # repo host: sources and scripts -> steve:~/loom-experiments/
ssh steve ~/loom-experiments/paper/build_all.sh           # GPU host, once (~30 min, ~25 GB of nix + docker)
ssh -t steve sudo ~/loom-experiments/steve-rdma/setup_root.sh   # GPU host, after EVERY reboot
ssh steve ~/loom-experiments/paper/run_paper.sh fig3      # Fig 3a/b (~15 min); `text` (~3 h), `extra`, `all`
scripts/collect_results.sh steve                          # repo host: outputs -> the repo (git diff shows changes)
scripts/analyze_local.sh && scripts/make_plots.sh ~/loom-paper/plots
```

`run_paper.sh` needs the GPU to itself: nothing else may hold the GPU or the NIC (check with
`~/loom-experiments/steve-rdma/check.sh`, line "GPU compute processes", want 0).

## What produces each figure, table and number

Paths are relative to this repo. "GPU" means the number comes from a GPU-host run (`run_paper.sh`
stage in brackets), "local" from `analyze_local.sh`, "source" from reading code in `src/`.
`plots/plot_motivation.py` prints every plotted and quoted number; the paper's `% ARGUMENT` comments
cite the same files.

### Figures and tables

| Paper element | Data | Produced by |
|---|---|---|
| Fig 1 (`loom-mix`): Mixtral-8x22B bytes per MoE layer by fabric and kind | `m3-sm-share/chakra/et_summary_Mixtral-8x22B.json` | local (`analyze_et.py`), `plot_motivation.figure_intro` |
| Fig 2 (`tables/divide-code.tex`): DeepEP's split, abridged listings | DeepEP V2.5 source, `src/DeepEP` (`impls/ep/dispatch.cuh`, `csrc/.../all_gather.hpp`) | source |
| Table 1 (`tables/code-state.tex`): NCCL lines, growth, plugin APIs | `m1-nccl-loc/summary_v2.{32.3,18.5}-1.txt`, `split_*.csv` | local (`categorize.py`) |
| Table 1: work per token, SMs held | `m2-device-state/results.md` sections 3.3 and 5; `m3-sm-share/deepep-0sm.md` | source |
| Table 1: GPU state at EP256 | `m2-device-state/totals.out` | local (`totals.py`) |
| Fig 3a (`loom-costs`): dispatch breakdown, 7 KiB, 1/16/128 tokens | `m5-rdma-init/steve-cx7/bd_v2/dispatch_bd_t3_*.csv` -> `summary_t3.csv` | GPU [`fig3`], `summarize_bd.py` |
| Fig 3b: combine breakdown, 14 KiB | `bd_v2/combine_bd_t3_*.csv` -> `summary_combine_t3.csv` | GPU [`fig3`] (`dispatch_bd --combine`) |
| Fig 3a/b dashed line: payload through the NIC | message counts from the runs above x size / 14.8 GB/s | the 14.8 GB/s is `m3-sm-share/steve-rdma/perftest-cuda/loopback_steve_identity.txt` (GPU->GPU) |
| Fig 3c: transport state per GPU, EP16-EP1024 | `m2-device-state/totals.out` (struct sizes from `sizes.out`) | local (`totals.py`, `run_sizes.sh`) |
| Table 2 (`tables/comparison.tex`) | qualitative | - |

### Numbers in the text

| Where | Number | Data | Produced by |
|---|---|---|---|
| Intro, 2.1 | 483 / 471 / 12 / 72 MiB per Mixtral-8x22B layer | `m3-sm-share/chakra/et_summary_Mixtral-8x22B.json` | local |
| Intro, 2.2 | 87 M/s GPU-initiated vs ~3 M/s CPU proxy | `m5-rdma-init/steve-cx7/msgrate_steve.csv` | GPU [`text`] `run_msgrate.sh` |
| Intro, 2.2 | dispatch 2.1-5.4x faster when the GPU posts | `dispatch_sweep_all.csv`, `dispatch_ibgda_block_H*_load0.csv`, `deepep_post_dispatch_H*.csv` | GPU [`text`] `run_dispatch*.sh`, `run_deepep_post.sh` |
| Intro, 2.3 #2 | post chain 3.1 µs (DeepEP) / 7.3 µs (NVSHMEM) of SM time | `deepep_post_lat.csv` | GPU [`text`] `run_deepep_post.sh` |
| 2.2 | single message 5.4 (proxy) / 9.2 (DeepEP) / 14.3 µs (NVSHMEM) | `proxy_b2_numa0.csv`, `deepep_post_lat.csv` | GPU [`text`] |
| Intro, 2.3 #2 | flush holds SMs 10-34x (dispatch), 9-27x (combine) as long as local | `bd_v2/summary_t3.csv`, `summary_combine_t3.csv` | GPU [`fig3`] |
| Intro, 2.3 #3, Table 1 | 4,257 QPs / 592 MiB at EP256; 16,641 / 2.3 GiB at EP1024 | `m2-device-state/totals.out` | local |
| Intro, 2.3 #1 | NCCL 47% fabric-specific, 36.6K / 16.6K lines, 32% of 25K in 2.18, 20 plugin APIs | `m1-nccl-loc/summary_*.txt`, `results.md` | local |
| 2.1 | TMA 45 GB/s per SM, stores 35 GB/s | `m5-rdma-init/steve-cx7/tma_bw.csv` | GPU [`text`] |
| 2.1 | a kernel cannot start a copy engine | `dev_ce_check.csv` | GPU [`text`] |
| 2.1 | copy-engine collective 17 µs + 2 µs per rank | `src/nccl/src/tuning/ce_model.cc` | source |
| 2.1 | Mixtral-8x7B: 83% before the last rank starts, a2a 94 µs, 1.7x NVLink bound | `m3-sm-share/chakra/out_skew_Mixtral-8x7B.txt`, `out_moe_Mixtral-8x7B.txt` | local (`--traces`) |
| 2.1 | NIC writes GPU memory at 14.8 GB/s | `m3-sm-share/steve-rdma/perftest-cuda/loopback_steve_identity.txt` | GPU (`test_loopback.sh`) |
| 2.3 #2 | NVSHMEM puts 10-22% slower beside a GEMM | `put_lat_steve.txt` vs `put_lat_steve_load.txt` | GPU [`text`] `run_put_lat.sh` |
| 2.3 #2 | 88-97% of a token's copies leave an 8-GPU domain from EP64 | `m6-routing/routing_traffic.csv` | local (`routing_traffic.py`) |
| 2.3 #2 | 1-token and 128-token breakdown numbers | `bd_v2/summary_t3.csv`, `summary_combine_t3.csv` | GPU [`fig3`] |
| 2.3 #2 | at 4096 tokens the remote path is the link (10.6 ms vs 0.29 ms) | `bd_v2/summary.csv` | GPU [`text`] `run_dispatch_bd.sh` |
| 2.3 #2 | DeepGEMM keeps 81-94% beside 20 reserved SMs | `m3-sm-share/deepgemm/held_m*.csv`, `locked_m*.csv` | GPU [`text`] (docker) |
| 2.3 #2 | a single message by stores would take 3.9-5.4 µs | `fence_cost_numa0.csv`, `nic_post_numa0.csv` | GPU [`text`] |
| 2.3 #3 | 384.5 KiB per NVSHMEM QP, 140 KiB per GIN QP | `m2-device-state/sizes.out`, `totals.out` | local |

Numbers that only appear in the paper's `% ARGUMENT` comments (GEMM lost per dispatch, cuBLAS next to
held SMs, CPU-posted RDMA beside the GEMM, copy-engine latency) come from the `extra` stage:
`m3-sm-share/gpu-interference/`, `m3-sm-share/gpu-posted/`, `bd_v2/d3*`, `ce_latency_*.csv`,
`lat_cpu_posted_*.csv`, `dispatch_ce_*.csv`.

## The GPU host

### Hardware and software

| | ours (`steve`) | why it matters |
|---|---|---|
| GPU | NVIDIA H200 NVL, PCI `0000:15:00.0`, NUMA 0, PCIe Gen5 x8 | Hopper (TMA, `sm_90`) |
| NIC | ConnectX-7 dual port 200 Gb/s, `0000:95:00.0` / `.1` (`mlx5_0`, `mlx5_1`), NUMA 1 | GPUDirect RDMA and GPU-initiated (IBGDA) |
| cabling | port 0 <-> port 1, RoCE v2, GID index 3 | both ends of every RDMA transfer on one host |
| OS | NixOS (kernel 7.2.2, `intel_iommu=on`), NVIDIA driver 595.71.05, nix with flakes | every tool comes from nixpkgs |
| toolchain (nix) | CUDA 12.9, NVSHMEM 3.6.5 (`cudaPackages.libnvshmem`), rdma-core 62.0, mpich 5.0.1, numactl 2.0.18, perftest `bae0736` | |
| docker | with the NVIDIA CDI devices (`--device nvidia.com/gpu=all`), user in the `docker` group | DeepGEMM (CUDA 12.9 container) |
| sudo | passwordless for the runners | NVSHMEM needs unlimited locked memory; clock locking |

**Testbed caveat.** On steve the GPU and the NIC sit on different sockets. This inflates the NIC's
share of every remote transfer (work-request fetch, payload reads, completions) and caps NIC -> GPU
memory at 14.8 GB/s. It does not inflate the GPU-side costs: `fence_cost.cu` shows the cross-socket
part of the post chain is 0.06-0.13 µs of 6-8 µs. A host with GPU and NIC on one socket gives the
same GPU-side numbers and smaller NIC-side ones.

On another host, set the PCI addresses (`GPU=`, `NICS=`) for `setup_root.sh` / `check.sh`, the NIC
names and GID in `m3-sm-share/steve-rdma/nvshmem-loopback/env.sh` (`NVSHMEM_HCA_PE_MAPPING`,
`NVSHMEM_IB_GID_INDEX`) and in the perftest scripts (`-d mlx5_0/1 -x 3`), and keep processes on the
GPU's NUMA node (the runners use `numactl -N 0 -m 0`).

### One-time setup

`scripts/deploy_gpu_host.sh <host>` copies the sources and scripts into `~/loom-experiments/` on the
host (the layout every runner assumes; see the table in `scripts/deploy_gpu_host.sh`). Then, on the
host, `~/loom-experiments/paper/build_all.sh`:

1. `steve-rdma/build.sh`: the cuBLAS interference benchmark; perftest with CUDA dma-buf support
   (`perftest-cuda/build.sh`; configure must report `cuMemGetHandleForAddressRange ... yes`, so
   `nvidia_peermem` is not needed); NVSHMEM 3.6.5 copied out of the nix store into
   `~/loom-experiments/nvshmem-3.6.5` (prebuilt, not compiled; its rpaths point into `/nix/store`, so
   keep the same nixpkgs revision); mpich.
2. `latency/build.sh`: the CPU-proxy, copy-engine, NIC-post, fence and TMA programs.
3. `gpu-posted/fetch_deepep_include.sh` (DeepEP V1 `a56d615` headers), `gpu-posted/build.sh`
   (`nvshmem_interfere`, `dispatch_ibgda`), `build_deepep_post.sh` (ports DeepEP's post path to
   NVSHMEM 3.6.5: one line changes the RC QP index from PE-major to QP-major; unpatched it posts to a
   QP that does not exist), `build_dispatch_bd.sh`.
4. `docker build -t loom-deepgemm ~/loom-experiments/deepgemm` (DeepGEMM pinned to `057ca59`).

### After every reboot (root): `sudo ~/loom-experiments/steve-rdma/setup_root.sh`

Nothing it does persists. It ends by running `check.sh`, which must show:

```
IOMMU groups (want: identity for all three):   ... identity  x3
NVIDIA driver (want: PeerMappingOverride=1):   RegistryDwords "PeerMappingOverride=1;"
```

What it does, and why:

| Step | Command (abridged) | Without it |
|---|---|---|
| NVIDIA driver option `PeerMappingOverride=1` | `options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"` in `/run/modprobe.d/`, then reload `nvidia` | the GPU cannot map the NIC's doorbell page, so GPU-initiated (IBGDA) posting is impossible. It must be a modprobe config file: a udev rule reloads the driver behind your back (`nvidia-container-toolkit-cdi-generator`) |
| GPU's IOMMU group -> `identity` | unbind from `nvidia`, `echo identity > /sys/kernel/iommu_groups/<g>/type`, rebind | IBGDA puts hang: the GPU's doorbell writes to the NIC's BAR fault in the IOMMU (DMAR write fault from `15:00.0`) |
| both NIC ports' IOMMU groups -> `identity` | the same through `mlx5_core` | GPUDirect RDMA fails: `local protection error` (syndrome 0x51) and a DMAR read fault from `95:00.0`, because the NIC is handed the GPU BAR's bus address while its own group is translated |

The group numbers are looked up from the PCI addresses (each device must be alone in its group).
Undo: reboot.

### What every NVSHMEM run sets

`nvshmem-loopback/env.sh` (sourced by every runner):

| Variable | Value | Purpose |
|---|---|---|
| `NVSHMEM_HOME` | `~/loom-experiments/nvshmem-3.6.5` | the copied NVSHMEM |
| `NVSHMEM_BOOTSTRAP` | `PMI` | mpiexec starts the PEs |
| `NVSHMEM_DISABLE_P2P`, `NVSHMEM_DISABLE_NVLS` | `1` | both PEs share one GPU: force the NIC path |
| `NVSHMEM_IB_ENABLE_IBGDA`, `NVSHMEM_IBGDA_NIC_HANDLER` | `1`, `gpu` | GPU-initiated posting, GPU rings the doorbell |
| `NVSHMEM_HCA_PE_MAPPING` | `mlx5_0:1:1,mlx5_1:1:1` | PE 0 on port 0, PE 1 on port 1 (the loopback cable) |
| `NVSHMEM_IB_GID_INDEX` | `3` | RoCE v2 GID |
| `NVSHMEM_SYMMETRIC_SIZE` | `2G` | heap for the dispatch buffers |
| `LD_LIBRARY_PATH` | nix rdma-core 62.0 + `/run/opengl-driver/lib` | update the store path if nixpkgs moves: `nix build --no-link --print-out-paths 'nixpkgs#rdma-core^out'` |
| `CUDA_VISIBLE_DEVICES`, `NVSHMEM_DISABLE_NCCL` | `0`, `1` | |

Per run the runners add `NVSHMEM_REMOTE_TRANSPORT=none` (pure IBGDA, no host proxy),
`NVSHMEM_IBGDA_NUM_RC_PER_PE` (24, DeepEP V1's setting), `NVSHMEM_QP_DEPTH` (1024; 8192 for
`dispatch_bd`, because DeepEP's post path skips the queue-slot check), `NVSHMEM_DISABLE_CUDA_VMM=1`
(`dispatch_bd`: the receiver maps PE 1's buffer by CUDA IPC). They run as root (`ulimit -l
unlimited`), start two PEs with `mpiexec -n 2`, pin to NUMA node 0, and give the outputs back to the
invoking user. `run_msgrate.sh` and `run_put_lat.sh` start MPS so the two PEs share the GPU.

### Containers

One image is built and used, for DeepSeek's FP8 GEMM (DeepGEMM needs CUDA >= 12.9; the host's nix
toolchain is used for everything else):

```sh
# build (build_all.sh does this): CUDA 12.9.1 devel, Ubuntu 24.04, PyTorch cu129, DeepGEMM 057ca59 from source, ~18 GB
docker build -t loom-deepgemm ~/loom-experiments/deepgemm          # Dockerfile: m3-sm-share/deepgemm/Dockerfile
docker run --rm loom-deepgemm cat /opt/DeepGEMM.commit             # check: 057ca5964aae...
# run (run_paper.sh text does this): GPUs through CDI, results written into the mounted directory
cd ~/loom-experiments/deepgemm
docker run --rm --device nvidia.com/gpu=all --ipc=host -v $PWD:/work -w /work loom-deepgemm \
  bash -c "cat /opt/DeepGEMM.commit; python dg_held.py --dtypes fp8 --ks 4,8,16,20 --reps 5 --iters 100 --m-per-group 1024" > held_m1024.csv
./run_locked.sh     # the same with the SM clock locked at 1410 MHz (sudo nvidia-smi -lgc; reset on exit)
```

`--device nvidia.com/gpu=all` needs the NVIDIA container toolkit's CDI spec on the host (NixOS:
`hardware.nvidia-container-toolkit.enable = true`). No other container is part of the experiments;
make sure no other container holds the GPU before a run (`docker ps`, `nvidia-smi`).

### The runs

`~/loom-experiments/paper/run_paper.sh <stage>` (read it for the exact commands):

| Stage | Runs | Time |
|---|---|---|
| `fig3` | `dispatch_bd` local / ordered / flush, dispatch (7 KiB) and combine (14 KiB), 1/16/128 tokens, 20 runs each | ~15 min |
| `text` | message rate; DeepEP post path; NVSHMEM put latency idle and beside a GEMM; GPU vs proxy dispatch sweeps (+ `dispatch_sweep_all.csv`); proxy, fence, NIC post, TMA, device copy engine; the 16-4096-token breakdown sweep; DeepGEMM next to reserved SMs (free and clock-locked) | ~3 h |
| `extra` | ARGUMENT-only numbers: cuBLAS beside held SMs, CPU-posted RDMA beside the GEMM, NVSHMEM puts beside the GEMM, compute lost per dispatch (D3), copy-engine and CPU-posted latency, cuBLAS vs DeepGEMM | ~2.5 h |

`dispatch_bd` (`m5-rdma-init/steve-cx7/dispatch_bd.cu`) is the benchmark behind Fig 3: DeepEP V2.5's
token flow on both branches, timed until the receiver sees the arrival signal. One-off runs:
`sudo bash ~/loom-experiments/gpu-posted/bd_one.sh <args>`; options are listed at the top of the source
and in `m5-rdma-init/README.md`.

## Local analyses and traces

`scripts/analyze_local.sh` needs only nix and `src/` (`scripts/fetch_sources.sh`):

- **M1** (`m1-nccl-loc/categorize.py`): NCCL lines by fabric, v2.32.3-1 and v2.18.5-1 (cloc, ctags).
- **M2** (`m2-device-state/`): `run_sizes.sh` compiles the NVSHMEM and NCCL GDAKI structs to get
  their sizes; `totals.py` counts QPs and HBM for each configuration.
- **M6** (`m6-routing/routing_traffic.py`): DeepSeek-V3 routing, copies inside vs outside the
  scale-up domain (50,000 tokens, seed 1).
- **M5 summaries**: medians and breakdowns of the GPU runs (`summarize_bd.py`, `summarize_d3.py`).
- **Chakra** (`--traces`, `m3-sm-share/chakra/`): needs the MLCommons Chakra Open Trace Library zips
  (access through the MLCommons Chakra working group's Google Drive) in `~/chakra-traces/`:
  `Mixtral-20260930T115950Z-1-00{1,2}.zip`, `Llama3-20260930T120123Z-1-00{1..5}.zip`.
  `extract.sh` unpacks them into `/scratch/$USER/chakra-traces/`.

## Checking a rerun

- **Local analyses** rewrite the committed outputs in place: after `scripts/analyze_local.sh --traces`,
  `git status` is clean (verified 2026-10-07).
- **GPU runs** are measurements: expect the medians to move by a few percent between sessions (the
  1/128-token dispatch reproduced within 2-5% across three sessions on 2026-10-06/07). Compare with
  `git diff` after `scripts/collect_results.sh`, then `scripts/make_plots.sh` prints the numbers the
  text quotes.

## Known gaps

- `m3-sm-share/chakra`: `nemo_raw/device_{1,4,5,7}.json` (Mixtral-8x7B Kineto device traces) are not
  in the downloaded zips; their origin is not recorded. Without them `analyze_local.sh --traces`
  skips the Mixtral-8x7B device-trace analyses (the 83% / 94 µs / 1.7x numbers of Section 2.1) and
  keeps the committed outputs.
- No recorded commands for `proxy_b2_numa1.csv`, `fence_cost_numa1.csv`, `proxy_b2_placement.csv`
  (socket-placement checks; presumably `numactl -N 1 -m 1` / the four placements), the early
  `dispatch_ibgda.csv` / `dispatch_proxy.csv` (format without H / load columns), and
  `m3-sm-share/deepgemm/v1`, `v2` (earlier runs). None of them feeds a figure or a number in the text.
- The DeepGEMM image installs the newest torch for CUDA 12.9 (unpinned).

## Layout

| # | experiment | folder | headline |
|---|---|---|---|
| M1 | NCCL lines by fabric | [m1-nccl-loc/](m1-nccl-loc/) | NCCL 2.32: 46.9% of 113,222 lines are fabric-specific (36,580 scale-out, 16,566 scale-up), up from 32.4% in 2.18 |
| M2 | Transport state of GPU-initiated RDMA | [m2-device-state/](m2-device-state/) | DeepEP V2.5 holds 4,257 QPs / 592 MiB per GPU at EP256 and 16,641 / 2.3 GiB at EP1024; one put = 4 atomics, 3 WQE stores, 2 doorbells, 4-5 fences on the SM |
| M3 | SM share: 0-SM study, GEMM beside held SMs, DeepGEMM, Chakra | [m3-sm-share/](m3-sm-share/) | 20 reserved SMs cost DeepGEMM 6-19%; GPU-posted RDMA costs what holding the SMs costs |
| M4 | (dropped 2026-09-30) | - | - |
| M5 | Initiation cost and the dispatch / combine breakdown | [m5-rdma-init/](m5-rdma-init/) | 1-token dispatch: local 4.9 µs, remote 18.8 (ordered) / 32.0 µs (flush); a flush holds its SMs 10-34x as long |
| M6 | DeepSeek-V3 routing vs scale-up domain size | [m6-routing/](m6-routing/) | 88-97% of a token's copies leave an 8-GPU domain from EP64 on |
| | reproduction scripts | [scripts/](scripts/) | `fetch_sources`, `deploy_gpu_host`, `gpu-host/{build_all,run_paper}`, `collect_results`, `analyze_local`, `make_plots` |
| | figures | [plots/plot_motivation.py](plots/plot_motivation.py) | Fig 1 and Fig 3 |

## Source clones (`src/`)

`scripts/fetch_sources.sh` recreates them (gitignored). Experiment READMEs reference them by relative
path (`../src/...`).

| directory | ref | commit |
|---|---|---|
| `src/nccl` | NCCL `v2.32.3-1` | `12df1a11` |
| `src/nccl-v2.18.5` | NCCL `v2.18.5-1` (git worktree of `src/nccl`) | `559b70f8` |
| `src/nvshmem` | NVSHMEM `v3.8.0-0` (M2 struct sizes only; the runs use the nixpkgs 3.6.5 build) | `270759e` |
| `src/DeepEP` | DeepEP `main` (V2.5, NCCL GIN) | `93eb6eb` |
| `src/DeepEP-v1-last` | DeepEP's last V1 commit (NVSHMEM), parent of V2.5 (worktree of `src/DeepEP`) | `a56d615` |
| `src/DeepEP-0sm` | DeepEP `main` with PRs 347 / 453 and the `hybrid-ep`, `antgroup-opt` branches | `93eb6eb` |
| `src/DeepGEMM` | DeepGEMM `main` | `057ca59` |
