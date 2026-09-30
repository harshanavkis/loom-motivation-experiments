#!/usr/bin/env bash
# GPU-posted RDMA latency (NVSHMEM IBGDA) on steve: 2 PEs on the H200 over the CX-7
# port0 <-> port1 loopback. The ping-pong tests report the FULL round trip per iteration
# (put + signal, then wait for the peer's put + signal); halve for one-way.
#   shmem_put_ping_pong_latency : nvshmem_int_put_nbi + fence + signal_op, from one thread
#   shmem_p_ping_pong_latency   : single-element nvshmem_p
# Needs unlimited locked memory (IBGDA pins its CQs in host memory), hence root.
# Usage on steve:  sudo ~/loom-experiments/nvshmem-loopback/run_put_lat.sh [load]
#   load: a BF16 GEMM runs continuously in another MPS client (really concurrent) during
#         the pure-IBGDA tests; output put_lat_steve_load.txt
set -uo pipefail
D=/home/harshanavkis/loom-experiments/nvshmem-loopback
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
ulimit -l unlimited
export HOME=/home/harshanavkis
source $D/env.sh
# host ibrc transport stays ON: signal ops need it (with it off, the ping-pong kernels read NULL, Xid 31)
# 2 PEs on one GPU must run their kernels CONCURRENTLY (ping-pong spins on the peer).
# Without MPS the two processes time-slice (~ms per switch) and the test crawls/hangs.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/loom-mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/loom-mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control' EXIT
MODE=${1:-idle}
OUT=$D/put_lat_steve.txt; [ "$MODE" = load ] && OUT=$D/put_lat_steve_load.txt
if [ "$MODE" = load ]; then
  /home/harshanavkis/loom-experiments/gpu-interference/interfere --load-seconds 400 > $D/load.log 2>&1 &
  LOADPID=$!
  trap 'kill $LOADPID 2>/dev/null; wait $LOADPID 2>/dev/null; echo quit | nvidia-cuda-mps-control' EXIT
  sleep 10
fi
{
  echo "# $(date -Is) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  grep RegistryDwords: /proc/driver/nvidia/params
  # pure IBGDA (no host proxy): put_nbi + quiet per iteration (post -> completion after the
  # remote ACK, ~1 round trip) and a blocking get (RDMA read round trip); one GPU thread
  for t in shmem_put_latency shmem_g_latency; do
    echo "## $t (REMOTE_TRANSPORT=none: pure IBGDA)"
    NVSHMEM_REMOTE_TRANSPORT=none timeout 120 $MPI/bin/mpiexec -n 2 $NVSHMEM_HOME/bin/perftest/device/pt-to-pt/$t -b 8 -e 65536 -f 8 -n 10000 -w 1000
  done
  # ping-pong with signal ops: on this setup the signal goes through the host proxy
  # (with the proxy off the kernel faults on a NULL pointer, Xid 31) -> not pure GPU-posted
  [ "$MODE" = load ] || for t in shmem_put_ping_pong_latency shmem_p_ping_pong_latency; do
    echo "## $t (ibrc proxy on: signal ops proxy-assisted)"
    timeout 120 $MPI/bin/mpiexec -n 2 $NVSHMEM_HOME/bin/perftest/device/pt-to-pt/$t -b 8 -e 65536 -f 8 -n 10000 -w 1000
  done
} > $OUT 2>&1
echo "exit=$?  output: $OUT"
chown harshanavkis $OUT
