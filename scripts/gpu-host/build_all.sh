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
echo "built; next: sudo $H/steve-rdma/setup_root.sh (after every reboot), then $H/paper/run_paper.sh fig3"
