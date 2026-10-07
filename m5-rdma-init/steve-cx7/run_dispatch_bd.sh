#!/usr/bin/env bash
# Dispatch time breakdown on steve (see dispatch_bd.cu): every path / QP variant x H x {idle, GEMM}.
# Root: unlimited locked memory for IBGDA.
# Usage on steve: sudo ~/loom-experiments/gpu-posted/run_dispatch_bd.sh [extra dispatch_bd args, e.g. --ctas 2,4,8,20]
set -uo pipefail
U=${SUDO_USER:-$(logname)}; UH=$(getent passwd "$U" | cut -d: -f6)   # the invoking user: runs nix, owns the outputs
D=$UH/loom-experiments/gpu-posted
MPI=$(sudo -u "$U" nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
NUMA=$(sudo -u "$U" nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
ulimit -l unlimited
export HOME=$UH
source $HOME/loom-experiments/nvshmem-loopback/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none       # pure IBGDA, no host proxy
export NVSHMEM_IBGDA_NUM_RC_PER_PE=24      # as run_dispatch.sh (DeepEP V1's setting)
export NVSHMEM_QP_DEPTH=8192               # DeepEP skips the WQ slot check: >= messages in flight per QP (~2.7k at 4096 tokens)
export NVSHMEM_DISABLE_CUDA_VMM=1          # heap from cudaMalloc, so PE 1 can export it to PE 0's receiver by CUDA IPC
OUT=${OUT:-$D/dispatch_bd}
# name:args. QPs: one per destination (8), three per destination (24, V1 low latency: one per
# local expert), one per warp (24, V2.5: QPs follow SMs)
VARIANTS=${VARIANTS:-"local:--path_local flush:--path_flush ordered:--path_ordered flush-warp:--path_flush_--qp_warp ordered-dest3:--path_ordered_--qp_dest_--nq_3"}
for H in 1024 7168; do for LOAD in 0 1; do for V in $VARIANTS; do
  NAME=${V%%:*}; ARGS=${V#*:}; ARGS=${ARGS//_/ }
  X=""; [ $LOAD = 1 ] && X="--load"
  F=${OUT}_${NAME}_H${H}_load${LOAD}.csv
  timeout 1800 $MPI/bin/mpiexec -n 2 $NUMA --cpunodebind=0 --membind=0 $D/dispatch_bd $ARGS --H $H $X "$@" < /dev/null > $F 2> ${F%.csv}.err
  echo "variant=$NAME H=$H load=$LOAD exit=$?  output: $F"
  chown "$U" $F ${F%.csv}.err
done; done; done
