#!/usr/bin/env bash
# Build interfere.cu with nixpkgs CUDA 12.9 (no system installs). Run on the GPU host (steve).
set -euo pipefail
cd "$(dirname "$0")"
export NIXPKGS_ALLOW_UNFREE=1
P() { nix build --impure --no-link --print-out-paths "nixpkgs#cudaPackages.$1" | head -1; }
CUDART=$(P cuda_cudart); CCCL=$(P cuda_cccl)
CUBLAS_INC=$(P 'libcublas^include'); CUBLAS_LIB=$(P 'libcublas^lib')
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I"$CUDART/include" -I"$CUBLAS_INC/include" -I"$CCCL/include" \
  -L"$CUDART/lib" -L"$CUBLAS_LIB/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$CUDART/lib:$CUBLAS_LIB/lib:/run/opengl-driver/lib" \
  interfere.cu -o interfere -lcublas -lcudart
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I"$CUDART/include" -I"$CCCL/include" -L"$CUDART/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$CUDART/lib:/run/opengl-driver/lib" ce_streams.cu -o ce_streams -lcudart
echo built ./interfere ./ce_streams
