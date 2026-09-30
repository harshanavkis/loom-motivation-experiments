#!/usr/bin/env bash
# MoE dispatch, GPU-initiated (IBGDA), on steve (see dispatch_ibgda.cu).
# Root: unlimited locked memory for IBGDA. Usage: sudo ~/loom-experiments/gpu-posted/run_dispatch.sh
set -uo pipefail
D=/home/harshanavkis/loom-experiments/gpu-posted
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
NUMA=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
ulimit -l unlimited
export HOME=/home/harshanavkis
source $HOME/loom-experiments/nvshmem-loopback/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none       # only put_nbi + quiet: pure IBGDA, no host proxy
export NVSHMEM_IBGDA_NUM_RC_PER_PE=24      # DeepEP V1's setting; default 2 would make k CTAs share 2 QPs
export NVSHMEM_SYMMETRIC_SIZE=2G

OUT=$D/dispatch_ibgda.csv
timeout 1800 $MPI/bin/mpiexec -n 2 $NUMA --cpunodebind=0 --membind=0 $D/dispatch_ibgda > $OUT 2> $D/dispatch_ibgda.err
echo "exit=$?  output: $OUT"
chown harshanavkis $OUT $D/dispatch_ibgda.err
