#!/usr/bin/env bash
# GPU-initiated (IBGDA) vs CPU proxy (NVSHMEM IBRC) message rate on steve, the comparison
# NVIDIA's IBGDA blog makes: NVSHMEM's own perftests, CTA sweep, one QP per CTA.
#   shmem_p_bw:   scalar 4 B puts from 1024 threads per CTA, 1 MiB per iteration
#   shmem_put_bw: block puts, one thread per CTA, 8 B - 64 KiB
# Root (memlock), MPS (2 PEs on one GPU). Usage on steve: sudo .../run_msgrate.sh
set -uo pipefail
D=/home/harshanavkis/loom-experiments/nvshmem-loopback
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
ulimit -l unlimited
export HOME=/home/harshanavkis
source $D/env.sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/loom-mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/loom-mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null
trap 'echo quit | nvidia-cuda-mps-control' EXIT
P=$NVSHMEM_HOME/bin/perftest/device/pt-to-pt
OUT=$D/msgrate_steve.csv
{
  echo "test,transport,ctas,size,GBps"
  for c in 1 2 4 8 16 32 64; do
    for tr in ibgda ibrc; do
      E="NVSHMEM_IBGDA_NUM_RC_PER_PE=$c"; [ $tr = ibrc ] && E="NVSHMEM_IB_ENABLE_IBGDA=0"
      env $E timeout 300 $MPI/bin/mpiexec -n 2 $P/shmem_p_bw -c $c -t 1024 -b 1048576 -e 1048576 2>/dev/null |
        awk -v t=$tr -v c=$c '/^[0-9]/{print "p_bw," t "," c "," $1 "," $3}'
    done
  done
  for c in 4 64; do
    for tr in ibgda ibrc; do
      E="NVSHMEM_IBGDA_NUM_RC_PER_PE=$c"; [ $tr = ibrc ] && E="NVSHMEM_IB_ENABLE_IBGDA=0"
      env $E timeout 300 $MPI/bin/mpiexec -n 2 $P/shmem_put_bw -c $c -t 1 -b 8 -e 65536 -f 2 2>/dev/null |
        awk -v t=$tr -v c=$c '/^[0-9]/{print "put_bw," t "," c "," $1 "," $3}'
    done
  done
} > $OUT
echo "exit=$?  output: $OUT"; chown harshanavkis $OUT
