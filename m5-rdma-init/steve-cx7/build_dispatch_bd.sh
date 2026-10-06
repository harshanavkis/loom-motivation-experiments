#!/usr/bin/env bash
# Build dispatch_bd.cu on steve (relocatable device code, static NVSHMEM 3.6.5 device lib), next to
# deepep_post.cu: deepep-include/ and the ported ibgda_device_nv365.cuh come from build_deepep_post.sh.
set -euo pipefail
cd "$(dirname "$0")"
export NIXPKGS_ALLOW_UNFREE=1
NV=$HOME/loom-experiments/nvshmem-3.6.5
P() { nix build --impure --no-link --print-out-paths "nixpkgs#cudaPackages.$1" | head -1; }
CUDART=$(P cuda_cudart); CCCL=$(P cuda_cccl); CUBLAS_INC=$(P 'libcublas^include'); CUBLAS_LIB=$(P 'libcublas^lib')
RC_DEV=$(nix build --no-link --print-out-paths 'nixpkgs#rdma-core^dev' | head -1)   # infiniband/mlx5dv.h
test -f deepep-include/legacy/ibgda_device_nv365.cuh || { echo "run build_deepep_post.sh first"; exit 1; }
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -rdc=true -arch=sm_90 \
  --expt-relaxed-constexpr --extended-lambda --diag-suppress=128,2417 -DDISABLE_AGGRESSIVE_PTX_INSTRS \
  -I"$NV/include" -I"$CUDART/include" -I"$CCCL/include" -I"$RC_DEV/include" -I"$CUBLAS_INC/include" \
  -Ideepep-include/legacy -Ideepep-include \
  -L"$NV/lib" -L"$CUDART/lib" -L"$CUBLAS_LIB/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$NV/lib:$CUDART/lib:$CUBLAS_LIB/lib:/run/opengl-driver/lib" \
  dispatch_bd.cu -o dispatch_bd -lnvshmem_host -lnvshmem_device -lcublas -lcudart -lcuda
echo built ./dispatch_bd
