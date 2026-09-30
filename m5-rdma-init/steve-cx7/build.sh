#!/usr/bin/env bash
# Build ce_latency.cu with nixpkgs CUDA 12.9. Run on steve.
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
  ce_latency.cu -o ce_latency -lcublas -lcudart
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I"$CUDART/include" -I"$CCCL/include" -L"$CUDART/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$CUDART/lib:/run/opengl-driver/lib" \
  ce_triggered.cu -o ce_triggered -lcudart -lcuda
RC_DEV=$(nix build --no-link --print-out-paths 'nixpkgs#rdma-core^dev' | head -1); RC=$(nix build --no-link --print-out-paths 'nixpkgs#rdma-core^out' | head -1)
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I"$CUDART/include" -I"$CCCL/include" -I"$RC_DEV/include" -L"$CUDART/lib" -L"$RC/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$CUDART/lib:$RC/lib:/run/opengl-driver/lib" \
  proxy_b2.cu -o proxy_b2 -libverbs -lcudart -lcuda
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -arch=sm_90 \
  -I"$CUDART/include" -I"$CCCL/include" -L"$CUDART/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$CUDART/lib:/run/opengl-driver/lib" fence_cost.cu -o fence_cost -lcudart
echo built ./ce_latency ./ce_triggered ./proxy_b2 ./fence_cost
