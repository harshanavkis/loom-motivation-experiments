#!/usr/bin/env bash
# Build and run sizes.cpp with a nix-provided g++ and rdma-core headers (no system installs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/../src/nvshmem/src/include
RDMA=$(ls -d /nix/store/*rdma-core-6*-dev/include | sort | tail -1)
nix shell nixpkgs#gcc -c g++ -std=c++17 -D__constant__= -I"$SRC" -I"$RDMA" \
    "$HERE/sizes.cpp" -o "$HERE/sizes" && "$HERE/sizes"
N=$HERE/../src/nccl/src
nix shell nixpkgs#gcc -c g++ -std=c++17 -I"$N/transport/net_ib/gdaki/doca-gpunetio/include/common" -I"$N/include" \
    "$HERE/sizes_gdaki.cpp" -o "$HERE/sizes_gdaki" && "$HERE/sizes_gdaki"
