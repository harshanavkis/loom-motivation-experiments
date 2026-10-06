#!/usr/bin/env bash
# CPU-posted RDMA next to the GEMM, matched to the GPU-initiated held-SM runs
# (../gpu-posted: 7 KiB puts, GPU memory -> GPU memory over steve's CX-7 loopback):
# perftest ib_write_bw, 7168 B writes, both buffers in GPU memory (dma-buf), 90 s.
# Output: rdma_gpu2gpu_7k.csv (GEMM next to the traffic) + rdma_cpu_7k.txt (rate, poster CPU).
set -uo pipefail
cd "$(dirname "$0")"
PT=~/loom-experiments/perftest-cuda/install/bin/ib_write_bw
GPU="--use_cuda=0 --use_cuda_dmabuf"
NUMA="numactl --cpunodebind=0 --membind=0"
run_bench() { NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c $NUMA ./interfere --only-none --reps 5 --out "$1" > /dev/null; }
: > rdma_cpu_7k.txt
$PT -d mlx5_1 -x 3 -s 7168 -t 128 -D 90 -p 18522 $GPU > rdma_gpu2gpu_7k.srv.txt 2>&1 &
sleep 1
$PT -d mlx5_0 -x 3 -s 7168 -t 128 -D 90 -p 18522 $GPU 127.0.0.1 > rdma_gpu2gpu_7k.cli.txt 2>&1 &
cli=$!
sleep 5
t0=$(awk '{print $14+$15}' /proc/$cli/stat); s0=$(date +%s.%N)
run_bench rdma_gpu2gpu_7k.csv
t1=$(awk '{print $14+$15}' /proc/$cli/stat); s1=$(date +%s.%N)
echo "gpu2gpu_7k: poster CPU = $(awk -v a=$t0 -v b=$t1 -v x=$s0 -v y=$s1 -v hz=$(getconf CLK_TCK) 'BEGIN{printf "%.2f", (b-a)/hz/(y-x)}') cores" | tee -a rdma_cpu_7k.txt
wait
grep -E "^ 7168" rdma_gpu2gpu_7k.cli.txt | tee -a rdma_cpu_7k.txt
