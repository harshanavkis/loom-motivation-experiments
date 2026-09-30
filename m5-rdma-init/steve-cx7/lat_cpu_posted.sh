#!/usr/bin/env bash
# CPU-posted RDMA one-way latency on steve's CX-7 loopback (port 0 -> port 1):
# ib_write_lat ping-pong (reports half the round trip), host vs GPU memory,
# optionally with a GEMM running in another process (perftest launches no kernels).
# Usage: lat_cpu_posted.sh [load]      output: lat_cpu_posted_{idle,load}.csv
set -uo pipefail
cd "$(dirname "$0")"
PT=~/loom-experiments/perftest-cuda/install/bin/ib_write_lat
MODE=${1:-idle}; OUT=lat_cpu_posted_$MODE.csv
LOADPID=
if [ "$MODE" = load ]; then
  ~/loom-experiments/gpu-interference/interfere --load-seconds 600 > load.log 2>&1 & LOADPID=$!
  sleep 8
fi
echo "mem,size,t_min_us,t_typical_us,t_avg_us,p99_us,p99.9_us" > $OUT
# host      : RDMA write, receiver polls the last byte in host memory (perftest default)
# host_imm  : RDMA write-with-immediate, receiver waits for the CQE (same method as gpu_imm)
# gpu_imm   : GPU memory (dma-buf GPUDirect) on both ends; perftest cannot poll GPU memory,
#             so it requires --write_with_imm
for mem in host host_imm gpu_imm; do
  case $mem in host) X="";; host_imm) X="--write_with_imm";; gpu_imm) X="--write_with_imm --use_cuda=0 --use_cuda_dmabuf";; esac
  for sz in 8 64 512 4096 65536; do
    $PT -d mlx5_1 -x 3 -s $sz -n 20000 -p 18530 $X > /tmp/lat_srv.txt 2>&1 &
    sleep 1
    r=$($PT -d mlx5_0 -x 3 -s $sz -n 20000 -p 18530 $X 127.0.0.1 2>&1 | awk -v s=$sz '$1==s {print $3","$5","$6","$8","$9}')
    wait
    echo "$mem,$sz,${r:-FAILED}" | tee -a $OUT
  done
done
# RDMA read round trip (same mechanism as NVSHMEM shmem_g_latency: post a read, wait for its CQE)
RD=~/loom-experiments/perftest-cuda/install/bin/ib_read_lat
for mem in host gpu; do
  X=""; [ $mem = gpu ] && X="--use_cuda=0 --use_cuda_dmabuf"
  for sz in 8 64 512 4096 65536; do
    $RD -d mlx5_1 -x 3 -s $sz -n 20000 -p 18531 $X > /tmp/lat_srv.txt 2>&1 &
    sleep 1
    r=$($RD -d mlx5_0 -x 3 -s $sz -n 20000 -p 18531 $X 127.0.0.1 2>&1 | awk -v s=$sz '$1==s {print $3","$5","$6","$8","$9}')
    wait
    echo "read_$mem,$sz,${r:-FAILED}" | tee -a $OUT
  done
done
[ -n "$LOADPID" ] && kill $LOADPID 2>/dev/null
echo "wrote $OUT"
