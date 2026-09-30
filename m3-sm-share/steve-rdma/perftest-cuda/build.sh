#!/usr/bin/env bash
# perftest with CUDA (GPUDirect via dma-buf) built against nixpkgs rdma-core + CUDA 12.9; run on the GPU host.
set -euo pipefail
cd "$(dirname "$0")/src"
export NIXPKGS_ALLOW_UNFREE=1
P() { nix build --impure --no-link --print-out-paths "nixpkgs#$1" | head -1; }
CUDART=$(P cudaPackages.cuda_cudart); RC_DEV=$(P 'rdma-core^dev'); RC=$(P 'rdma-core^out'); PCI=$(P 'pciutils^out')
nix shell --impure nixpkgs#autoconf nixpkgs#automake nixpkgs#libtool nixpkgs#gnumake nixpkgs#gcc nixpkgs#pkg-config -c bash -c "
  make distclean >/dev/null 2>&1 || true
  ./autogen.sh >/dev/null 2>&1
  ./configure --disable-cudart --prefix=$PWD/../install CUDA_H_PATH=$CUDART/include/cuda.h \
    CPPFLAGS='-I$CUDART/include -I$RC_DEV/include -I$PCI/include' \
    LDFLAGS='-L$RC/lib -L$PCI/lib -L/run/opengl-driver/lib -Wl,-rpath,$RC/lib:$PCI/lib:/run/opengl-driver/lib' > ../configure.log
  grep -i -E 'cuda|dmabuf' ../configure.log || true
  make -j32 > ../make.log 2>&1 && make install > /dev/null
"
echo built $(dirname "$PWD")/install/bin
