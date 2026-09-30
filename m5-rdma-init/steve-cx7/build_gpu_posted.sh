#!/usr/bin/env bash
# Build nvshmem_interfere.cu (relocatable device code, static NVSHMEM device lib). Run on steve.
set -euo pipefail
cd "$(dirname "$0")"
export NIXPKGS_ALLOW_UNFREE=1
NV=$HOME/loom-experiments/nvshmem-3.6.5
P() { nix build --impure --no-link --print-out-paths "nixpkgs#cudaPackages.$1" | head -1; }
CUDART=$(P cuda_cudart); CCCL=$(P cuda_cccl); CUBLAS_INC=$(P 'libcublas^include'); CUBLAS_LIB=$(P 'libcublas^lib')
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -rdc=true -arch=sm_90 \
  -I"$NV/include" -I"$CUDART/include" -I"$CUBLAS_INC/include" -I"$CCCL/include" \
  -L"$NV/lib" -L"$CUDART/lib" -L"$CUBLAS_LIB/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$NV/lib:$CUDART/lib:$CUBLAS_LIB/lib:/run/opengl-driver/lib" \
  nvshmem_interfere.cu -o nvshmem_interfere -lnvshmem_host -lnvshmem_device -lcublas -lcudart -lcuda
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -rdc=true -arch=sm_90 \
  -I"$NV/include" -I"$CUDART/include" -I"$CCCL/include" -L"$NV/lib" -L"$CUDART/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$NV/lib:$CUDART/lib:/run/opengl-driver/lib" \
  -I"$CUBLAS_INC/include" -L"$CUBLAS_LIB/lib" -Xlinker -rpath -Xlinker "$CUBLAS_LIB/lib" dispatch_ibgda.cu -o dispatch_ibgda -lnvshmem_host -lnvshmem_device -lcublas -lcudart -lcuda
echo built ./nvshmem_interfere ./dispatch_ibgda
