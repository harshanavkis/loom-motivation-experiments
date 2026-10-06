# One dispatch_bd run with the NVSHMEM loopback environment (root for memlock). Usage on steve:
#   sudo bash ~/loom-experiments/gpu-posted/bd_one.sh --path ordered --qp dest --nq 3 --post lane --H 7168 --tokens 1,128 --iters 20
# usage: sudo bash /tmp/bd_one.sh <dispatch_bd args...>
set -uo pipefail
D=/home/harshanavkis/loom-experiments/gpu-posted
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
NUMA=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
ulimit -l unlimited
export HOME=/home/harshanavkis
source $HOME/loom-experiments/nvshmem-loopback/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IBGDA_NUM_RC_PER_PE=24 NVSHMEM_QP_DEPTH=8192 NVSHMEM_DISABLE_CUDA_VMM=1
timeout 300 $MPI/bin/mpiexec -n 2 $NUMA --cpunodebind=0 --membind=0 $D/dispatch_bd "$@" < /dev/null
