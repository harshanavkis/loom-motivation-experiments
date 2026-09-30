#!/usr/bin/env bash
# CPU-posted RDMA (the NIC DMAs, no SMs) next to GPU compute, on steve's CX-7 loopback
# (port 0 <-> port 1). perftest's client thread posts the WQEs on the CPU; the GPU
# only runs the workload. Needs the CX-7 IOMMU groups in passthrough (see README).
set -uo pipefail
cd "$(dirname "$0")"
PT=~/loom-experiments/perftest-cuda/install/bin/ib_write_bw
GPU="--use_cuda=0 --use_cuda_dmabuf"
NUMA="numactl --cpunodebind=0 --membind=0"
run_bench() { NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c $NUMA ./interfere --only-none --reps 5 --out "$1" > /dev/null; }
traffic() {  # name server_mem client_mem : client mlx5_0 -> server mlx5_1, 64 KiB writes, 90 s
  local name=$1 sm=$2 cm=$3
  $PT -d mlx5_1 -x 3 -s 65536 -t 32 -D 90 -p 18520 $sm > rdma_$name.srv.txt 2>&1 &
  sleep 1
  $PT -d mlx5_0 -x 3 -s 65536 -t 32 -D 90 -p 18520 $cm 127.0.0.1 > rdma_$name.cli.txt 2>&1 &
  local cli=$!
  sleep 5
  local t0; t0=$(awk '{print $14+$15}' /proc/$cli/stat)
  local s0; s0=$(date +%s.%N)
  run_bench rdma_$name.csv
  local t1; t1=$(awk '{print $14+$15}' /proc/$cli/stat)
  local s1; s1=$(date +%s.%N)
  echo "$name: poster CPU = $(awk -v a=$t0 -v b=$t1 -v x=$s0 -v y=$s1 -v hz=$(getconf CLK_TCK) 'BEGIN{printf "%.2f", (b-a)/hz/(y-x)}') cores" | tee -a rdma_cpu.txt
  wait
  grep -E "^ 65536" rdma_$name.cli.txt | tee -a rdma_cpu.txt
}
: > rdma_cpu.txt
run_bench rdma_none.csv
traffic gpu2host "" "$GPU"      # NIC reads HBM (payload leaves the GPU)
traffic host2gpu "$GPU" ""      # NIC writes HBM (payload arrives in the GPU)
