#!/usr/bin/env bash
# Run on steve (no root). Builds/fetches every tool the experiments use, into
# ~/loom-experiments (NFS home, so jamie sees the same files). Idempotent.
set -euo pipefail
H=$HOME/loom-experiments
export NIXPKGS_ALLOW_UNFREE=1

echo "== GPU interference benchmark (interfere, ce_streams)"
$H/gpu-interference/build.sh

echo "== perftest with CUDA / dma-buf GPUDirect"
if [ ! -d $H/perftest-cuda/src ]; then
  git clone https://github.com/linux-rdma/perftest.git $H/perftest-cuda/src
  git -C $H/perftest-cuda/src checkout bae0736291f14c45769135a78819021d2c99caae
fi
$H/perftest-cuda/build.sh

echo "== NVSHMEM 3.6.5 (IBGDA perftests), copied out of the nix store into the home dir"
if [ ! -x $H/nvshmem-3.6.5/bin/perftest/device/pt-to-pt/shmem_put_bw ]; then
  N=$(nix build --impure --no-link --print-out-paths nixpkgs#cudaPackages.libnvshmem | head -1)   # ~2 GB download
  cp -rL "$N" $H/nvshmem-3.6.5 && chmod -R u+w $H/nvshmem-3.6.5
fi

echo "== MPICH (mpiexec launches the 2 NVSHMEM PEs)"
nix build --no-link --print-out-paths nixpkgs#mpich | head -1

echo "done; next: sudo $H/steve-rdma/setup_root.sh (after every reboot), then $H/steve-rdma/check.sh"
