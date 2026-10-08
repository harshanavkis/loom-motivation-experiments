#!/usr/bin/env bash
# Build everything the paper's GPU benchmarks need, on the GPU host, as the normal user (no root).
# Prerequisite: scripts/deploy_gpu_host.sh <host> from the repo. Idempotent.
#   ~/loom-experiments/paper/build_all.sh [--no-docker]
set -euo pipefail
H=$HOME/loom-experiments
export NIXPKGS_ALLOW_UNFREE=1
$H/steve-rdma/build.sh                         # interfere/ce_streams, perftest-cuda, NVSHMEM 3.6.5 copy, mpich
echo "== latency programs (CPU proxy, copy engine, NIC post, fences, TMA)"
(cd $H/latency && ./build.sh)
echo "== GPU-initiated RDMA programs"
cd $H/gpu-posted
[ -f deepep-include/legacy/ibgda_device.cuh ] || ./fetch_deepep_include.sh   # DeepEP V1 a56d615 headers
./build.sh                                     # nvshmem_interfere, dispatch_ibgda
./build_deepep_post.sh                         # ports DeepEP's post path to NVSHMEM 3.6.5, builds deepep_post
./build_dispatch_bd.sh                         # dispatch_bd (Fig 3a/b)
if [ "${1:-}" != --no-docker ]; then
  echo "== DeepGEMM image (DeepGEMM 057ca59, CUDA 12.9; ~18 GB)"
  docker build -t loom-deepgemm $H/deepgemm
fi
echo "== pin the nix store paths the built tools load at run time (a garbage collection, e.g. at reboot, removed"
echo "   them once: binaries then fail with 'libmlx5.so.1: cannot open shared object file')"
mkdir -p $H/.gcroots
for f in $H/latency/* $H/gpu-posted/* $H/gpu-interference/* $H/perftest-cuda/install/bin/* $H/nvshmem-3.6.5/lib/*.so*; do
  [ -f "$f" ] && [ -x "$f" -o "${f%.so*}" != "$f" ] || continue
  readelf -d "$f" 2>/dev/null | grep -o '/nix/store/[^:/]*' || true
done | sort -u > $H/.gcroots/paths.txt
grep -o '/nix/store/[^:/]*' $H/nvshmem-loopback/env.sh >> $H/.gcroots/paths.txt || true
for t in mpich numactl; do nix build --no-link --print-out-paths nixpkgs#$t | grep -v -- -man | head -1 >> $H/.gcroots/paths.txt; done
sort -u $H/.gcroots/paths.txt | while read p; do
  nix-store --realise "$p" --add-root "$H/.gcroots/$(basename "$p")" --indirect > /dev/null 2>&1 \
    || echo "   not pinned (gone and not substitutable): $p"   # e.g. openmpi/ucx/pmix of NVSHMEM's unused plugins
done
echo "   pinned $(sort -u $H/.gcroots/paths.txt | wc -l) store paths under $H/.gcroots"
echo "built; next: sudo $H/steve-rdma/setup_root.sh (after every reboot), then $H/paper/run_paper.sh fig3"
