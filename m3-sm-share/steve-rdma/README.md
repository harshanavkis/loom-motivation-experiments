# GPU RDMA experiments on steve (one host, one GPU, CX-7 loopback)

> This is the committed copy. The runnable copy is `~/loom-experiments/steve-rdma/` (NFS home, visible on steve), and its scripts refer to `~/loom-experiments/{gpu-interference,perftest-cuda,nvshmem-loopback}`. To recreate that layout: `mkdir -p ~/loom-experiments && cp -r steve-rdma ~/loom-experiments/ && cp -r steve-rdma/perftest-cuda steve-rdma/nvshmem-loopback gpu-interference ~/loom-experiments/`.

This setup measures what RDMA costs a GPU when the **CPU** posts the work requests (NCCL-proxy style, the NIC does the DMA) versus when the **GPU** posts them (NVSHMEM IBGDA / DeepEP style, the SMs build WQEs and ring the doorbell), next to GPU compute. It uses one machine: steve's H200, with its dual-port ConnectX-7 cabled port 0 ↔ port 1. The remote side of every transfer is just memory, on the same host.

All scripts run **directly on steve**. `~/loom-experiments` is the NFS home, so the same files are visible from jamie.

## What works (2026-09-30)

| | result |
|---|---|
| RDMA host → host over the cable | 22,069 MiB/s (64 KiB writes) |
| GPUDirect: NIC reads HBM (GPU → host) | 20,509 MiB/s |
| GPUDirect: NIC writes HBM (→ GPU) | 14,944 MiB/s; about 15 GB/s is the cross-socket cap for NIC writes into HBM on steve |
| GPU-posted RDMA, NVSHMEM IBGDA, 2 PEs on the H200 | 15.7 GB/s at 1 MiB puts (0.32 GB/s at 4 KiB) |
| CPU-posted RDMA next to a BF16 GEMM, at 20.5 GB/s | GEMM keeps 100.2% of 809.6 TFLOP/s, triad 99.5%; the poster uses 1.00 CPU core |

GPU-posted RDMA next to the GEMM was done afterwards (`../gpu-posted`). NCCL GIN and DeepEP V2.5 still cannot run here, because NCCL refuses two ranks on one GPU; they need a second Hopper host.

## Hardware facts that matter

| item | value |
|---|---|
| GPU | H200 NVL at PCI `0000:15:00.0`, NUMA node 0, PCIe Gen5 ×8 (host transfers top out at 29 GB/s) |
| NIC | CX-7 dual port at `0000:95:00.0` (`mlx5_0`, `ens3931f0np0`) and `0000:95:00.1` (`mlx5_1`, `ens3931f1np1`), NUMA node 1, 200 Gb/s |
| loopback | cable port 0 ↔ port 1. RoCE v2, GID index 3 = the IPv4 link-local addresses (169.254.169.187 / 169.254.48.90), no switch, no config |
| not to use | `irdma0` / `ens4055f0np0` is steve's uplink to the university network; never run RDMA tests on it |

The GPU and NIC sit on different sockets, so all GPUDirect traffic crosses the socket interconnect. Reads from HBM are near line rate; NIC writes into HBM cap near 15 GB/s.

## Steps

### 0. Once: cable

Connect steve's two CX-7 ports to each other. `check.sh` should show both `mlx5_0` and `mlx5_1` as `4: ACTIVE 200 Gb/sec`.

### 1. Once, or when tools change: build (no root)

```sh
~/loom-experiments/steve-rdma/build.sh
```

The script does everything in "Building from scratch" below, and skips steps whose output already exists.

## Building from scratch

Nothing is installed system-wide. Every tool comes from nixpkgs through `nix build` / `nix shell`, and every build output lives under `~/loom-experiments`.

### Toolchain (nixpkgs)

`nixpkgs#…` resolves through steve's flake registry to its system nixpkgs: rev `cf446cc34f1e64dc465606e821f37418b46e220d` (26.05.20260914), the same revision as jamie. CUDA packages are unfree, so every command that touches them needs `export NIXPKGS_ALLOW_UNFREE=1` and `--impure`.

| nixpkgs attribute | version | used for |
|---|---|---|
| `cudaPackages.cuda_nvcc` | 12.9.86 | compiling `interfere.cu`, `ce_streams.cu` (`-arch=sm_90`) |
| `cudaPackages.cuda_cudart` (output `out`) | 12.9.79 | CUDA runtime headers and libs; `cuda.h` for perftest |
| `cudaPackages.cuda_cccl` | 12.9 | CUB/Thrust headers nvcc expects |
| `cudaPackages.libcublas` (outputs `include`, `lib`) | 12.9.1.4 | the GEMM workload |
| `cudaPackages.libnvshmem` | 3.6.5-0 | NVSHMEM runtime + prebuilt perftests (`bin/perftest/device/pt-to-pt/shmem_put_bw`) |
| `rdma-core` (outputs `dev`, `out`) | 62.0 | `libibverbs`, `libmlx5` (perftest build, NVSHMEM dlopen) |
| `pciutils` (output `out`) | 3.15.0 | perftest configure requirement |
| `autoconf automake libtool gnumake gcc pkg-config` | — | perftest's autotools build |
| `mpich` | 5.0.1 | `mpiexec -n 2` launcher for the two NVSHMEM PEs (PMI bootstrap) |
| `numactl` | 2.0.18 | pin runs to the GPU's NUMA node 0 |

The NVIDIA driver (595.71.05, open kernel module) and `libcuda.so`/`libnvidia-ml.so` come from steve's NixOS system under `/run/opengl-driver/lib`. That path is on every rpath and `LD_LIBRARY_PATH` below.

Useful pattern for finding an output path:

```sh
export NIXPKGS_ALLOW_UNFREE=1
nix build --impure --no-link --print-out-paths 'nixpkgs#rdma-core^dev'
```

`^dev`/`^out` select an output. Without it you may get `man` first, e.g. `pciutils` lists `man` before `out`.

### a. GPU interference benchmark: `~/loom-experiments/gpu-interference/`

Sources: `interfere.cu`, `ce_streams.cu`, `summarize.py`, `run.sh`, `rdma_cpu_posted.sh`. The committed copy lives in the motivation-experiments repo on jamie (`/scratch/harshanavkis/loom-proj/motivation-experiments/m3-sm-share/gpu-interference/`); copy it into `~/loom-experiments/gpu-interference/` if the home copy is missing. `gpu-interference/build.sh` does:

```sh
export NIXPKGS_ALLOW_UNFREE=1
P() { nix build --impure --no-link --print-out-paths "nixpkgs#cudaPackages.$1" | head -1; }
CUDART=$(P cuda_cudart); CCCL=$(P cuda_cccl); CUBLAS_INC=$(P 'libcublas^include'); CUBLAS_LIB=$(P 'libcublas^lib')
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I$CUDART/include -I$CUBLAS_INC/include -I$CCCL/include \
  -L$CUDART/lib -L$CUBLAS_LIB/lib -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker $CUDART/lib:$CUBLAS_LIB/lib:/run/opengl-driver/lib \
  interfere.cu -o interfere -lcublas -lcudart
# ce_streams.cu the same way, without cuBLAS
```

### b. perftest with CUDA (dma-buf GPUDirect): `~/loom-experiments/perftest-cuda/`

Source: `git clone https://github.com/linux-rdma/perftest.git src && git -C src checkout bae0736291f14c45769135a78819021d2c99caae` (`build.sh` clones it if `src/` is missing). `perftest-cuda/build.sh` does:

```sh
export NIXPKGS_ALLOW_UNFREE=1
P() { nix build --impure --no-link --print-out-paths "nixpkgs#$1" | head -1; }
CUDART=$(P cudaPackages.cuda_cudart); RC_DEV=$(P 'rdma-core^dev'); RC=$(P 'rdma-core^out'); PCI=$(P 'pciutils^out')
cd src
nix shell --impure nixpkgs#autoconf nixpkgs#automake nixpkgs#libtool nixpkgs#gnumake nixpkgs#gcc nixpkgs#pkg-config -c bash -c "
  ./autogen.sh
  ./configure --disable-cudart --prefix=$PWD/../install CUDA_H_PATH=$CUDART/include/cuda.h \
    CPPFLAGS='-I$CUDART/include -I$RC_DEV/include -I$PCI/include' \
    LDFLAGS='-L$RC/lib -L$PCI/lib -L/run/opengl-driver/lib -Wl,-rpath,$RC/lib:$PCI/lib:/run/opengl-driver/lib'
  make -j32 && make install"
```

Notes:
- `configure` must report `checking for cuMemGetHandleForAddressRange in -lcuda... yes`; that is the dma-buf GPUDirect path, so `nvidia_peermem` is not needed.
- `--disable-cudart` skips perftest's optional GPU validation kernel, which would need nvcc under the cudart prefix.
- The explicit `-I`/`-L` flags are needed because `nix shell` does not export headers to gcc the way `nix-shell -p` does.
- The older `~/loom-experiments/perftest-build` is not usable on steve: it links jamie's rdma-core store path.

### c. NVSHMEM: `~/loom-experiments/nvshmem-3.6.5/`

This is the prebuilt NVIDIA redistributable from nixpkgs, copied out of the store so the NFS home carries it, with no build step:

```sh
# fast: on jamie, whose store already has it
cp -rL /nix/store/*cuda12.9-libnvshmem-3.6.5-0 ~/loom-experiments/nvshmem-3.6.5 && chmod -R u+w ~/loom-experiments/nvshmem-3.6.5
# or on steve (downloads ~2 GB):
N=$(NIXPKGS_ALLOW_UNFREE=1 nix build --impure --no-link --print-out-paths nixpkgs#cudaPackages.libnvshmem | head -1)
cp -rL $N ~/loom-experiments/nvshmem-3.6.5 && chmod -R u+w ~/loom-experiments/nvshmem-3.6.5
```

The binaries' rpaths point into `/nix/store`. They resolve on steve because both hosts use the same nixpkgs revision; `ldd ~/loom-experiments/nvshmem-3.6.5/bin/perftest/device/pt-to-pt/shmem_put_bw | grep "not found"` must print nothing.

At runtime NVSHMEM dlopens `libibverbs`/`libmlx5` and `libnvidia-ml`, which are not on NixOS's default search path. `nvshmem-loopback/env.sh` sets:

```sh
export LD_LIBRARY_PATH=/nix/store/6f9hdh53xy45pdybibsz28z5iwxji35a-rdma-core-62.0/lib:/run/opengl-driver/lib
```

That rdma-core path is `nix build --no-link --print-out-paths 'nixpkgs#rdma-core^out'` + `/lib`; update it if nixpkgs moves.

### d. MPICH launcher

```sh
nix build --no-link --print-out-paths nixpkgs#mpich     # prefetch
nix shell nixpkgs#mpich -c mpiexec -n 2 <binary>         # use
```

`run_put_bw.sh` resolves the path with `nix build` as the user and then calls `$MPI/bin/mpiexec` as root.

### e. numactl

The GPU is on NUMA node 0, so the benchmarks run under `nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 …`. `run.sh` and `rdma_cpu_posted.sh` do this.

### 2. After every reboot: root setup

```sh
sudo ~/loom-experiments/steve-rdma/setup_root.sh
```

It does three things, none of which persists across a reboot:

1. **NVIDIA driver option `PeerMappingOverride=1`**, so the GPU can map the NIC's doorbell page (needed for IBGDA). The option has to go through `/run/modprobe.d/nvidia-peermapping.conf`, not the `modprobe` command line. A udev rule (`/etc/udev/rules.d/99-local.rules`) restarts `nvidia-container-toolkit-cdi-generator` on every `nvidia` module event, and that reloads the driver behind your back, without the option.
2. **IOMMU passthrough (`identity`) for the GPU's group and both CX-7 groups.** steve boots with `intel_iommu=on`, which puts every device in a translated (`DMA-FQ`) domain.
   - **Without it on the NIC:** GPU-memory RDMA fails with `local protection error (syndrome 0x51)` and `DMAR: [DMA Read] Request device [0000:95:00.0] fault addr 0x224041000000`.
   - **Without it on the GPU:** GPU-posted puts hang, with `DMAR: [DMA Write] Request device [0000:15:00.0] fault addr 0x2cfff0109000`. That address is the doorbell inside `mlx5_0`'s BAR0.
3. **Stopping and restarting ollama** around the driver reload.

The script ends by running `check.sh`.

### 3. Check (no root)

```sh
~/loom-experiments/steve-rdma/check.sh
```

Everything should read: both links `ACTIVE`; all three IOMMU groups `identity`; `RegistryDwords "PeerMappingOverride=1;"`; 0 GPU compute processes (run experiments only on an idle GPU); all three artifacts `present`.

### 4. Experiments

| what | command (on steve) | root | output |
|---|---|---|---|
| RDMA loopback, host and GPU memory, 5 cases | `~/loom-experiments/perftest-cuda/test_loopback.sh` | no | stdout (keep it with `\| tee`) |
| GPU-posted RDMA bandwidth (NVSHMEM IBGDA, 2 PEs) | `sudo ~/loom-experiments/nvshmem-loopback/run_put_bw.sh` | yes (memlock) | `nvshmem-loopback/put_bw_steve.txt` |
| SM vs copy-engine interference (no network) | `cd ~/loom-experiments/gpu-interference && ./run.sh results_steve.csv && python3 summarize.py results_steve.csv` | no | `results_steve.csv`, about 25 min |
| CPU-posted RDMA next to the GEMM | `~/loom-experiments/gpu-interference/rdma_cpu_posted.sh` | no | `gpu-interference/rdma_{none,gpu2host,host2gpu}.csv`, `rdma_cpu.txt`, about 4 min |

To read the CPU-posted result, compare the `run,...,none,...` metric of the same workload across the three CSVs. `rdma_cpu.txt` has the perftest bandwidth and the posting thread's CPU use.

## Troubleshooting (every trap hit so far)

| symptom | cause | fix |
|---|---|---|
| perftest: `Port number 1 state is Down` | no cable, or the mlx5 driver is still rebinding | cable port 0 ↔ port 1; wait about 5 s after `setup_root.sh` |
| perftest with `--use_cuda_dmabuf`: `local protection error`, DMAR DMA Read fault from `95:00.x` | NIC IOMMU group translated | `setup_root.sh` (NIC groups → identity) |
| NVSHMEM puts hang, DMAR DMA Write fault from `15:00.0` at an address in `mlx5_0` BAR0 | GPU IOMMU group translated | `setup_root.sh` (GPU group → identity) |
| `RegistryDwords: ""` after a reload with the option on the command line | nvidia-ctk reloaded the driver first | use the `/run/modprobe.d` file (the script does this) |
| NVSHMEM: `libibverbs not found on the system` / segfault after `NVML library not found` | NixOS: libraries aren't in the default path | `nvshmem-loopback/env.sh` puts nixpkgs rdma-core and `/run/opengl-driver/lib` on `LD_LIBRARY_PATH` |
| NVSHMEM: `ibv_create_cq for recv_cq failed`, then `Cannot allocate memory` | locked-memory limit is 8 MiB | run as root with `ulimit -l unlimited` (`run_put_bw.sh` does) |
| NVSHMEM: `transport create ep failed` in ibrc | host-side IB transport not needed for device puts | `NVSHMEM_REMOTE_TRANSPORT=none` |
| old perftest (`perftest-build`): `libmlx5.so.1: cannot open shared object` on steve | that build links jamie's rdma-core path | use `perftest-cuda` |
| `vfio-pci 0000:60:00.0: resetting` spam in dmesg | thore's TeeDisk VM (NVMe passthrough) | not ours; ignore |

## NVSHMEM configuration (`nvshmem-loopback/env.sh`)

Two PEs on one GPU ("MPG"); NVSHMEM detects this itself. `NVSHMEM_DISABLE_P2P=1` forces traffic through the NIC. `NVSHMEM_IB_ENABLE_IBGDA=1` with `NVSHMEM_IBGDA_NIC_HANDLER=gpu` makes the GPU post the work requests and ring the doorbell. `NVSHMEM_HCA_PE_MAPPING=mlx5_0:1:1,mlx5_1:1:1` puts PE 0 on port 0 and PE 1 on port 1, and `NVSHMEM_IB_GID_INDEX=3` selects RoCE v2. The launcher is `mpiexec -n 2` from nixpkgs MPICH, through NVSHMEM's default PMI bootstrap.

## Undo

Reboot steve. Alternatively: `echo DMA-FQ` into each of the three groups (with the device unbound), `rm /run/modprobe.d/nvidia-peermapping.conf`, and reload the NVIDIA driver.
