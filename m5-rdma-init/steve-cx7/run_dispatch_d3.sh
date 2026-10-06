#!/usr/bin/env bash
# D3 on steve (see dispatch_bd.cu --d3): compute lost per dispatch, local / ordered / flush (per-lane puts).
# Root: unlimited locked memory for IBGDA. Usage on steve: sudo ~/loom-experiments/gpu-posted/run_dispatch_d3.sh
set -uo pipefail
D=/home/harshanavkis/loom-experiments/gpu-posted
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
NUMA=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
ulimit -l unlimited
export HOME=/home/harshanavkis
source $HOME/loom-experiments/nvshmem-loopback/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IBGDA_NUM_RC_PER_PE=24 NVSHMEM_QP_DEPTH=8192 NVSHMEM_DISABLE_CUDA_VMM=1
OUT=${OUT:-$D/bd_v2/d3}
VARIANTS=${VARIANTS:-"local:--path_local ordered-destL3:--path_ordered_--qp_dest_--nq_3_--post_lane flush-warpL:--path_flush_--qp_warp_--post_lane"}
for H in 1024 7168; do for V in $VARIANTS; do
  NAME=${V%%:*}; ARGS=${V#*:}; ARGS=${ARGS//_/ }
  F=${OUT}_${NAME}_H${H}.csv; : > $F
  # dispatch periods just above the slowest path's dispatch time: the lost fraction must stand well
  # above the filler's window-to-window noise (~0.1%), which a dispatch every ms at 2 CTAs does not
  TPS="16:50,100 128:200,400"; [ $H = 7168 ] && TPS="16:100,200 128:500,1000"
  for TP in $TPS; do
    timeout 1200 $MPI/bin/mpiexec -n 2 $NUMA --cpunodebind=0 --membind=0 $D/dispatch_bd $ARGS --H $H --tokens ${TP%%:*} \
      --ctas 20 --d3 ${TP#*:} --reps 5 --window-ms 500 "$@" < /dev/null >> $F 2>> ${F%.csv}.err
    echo "variant=$NAME H=$H tokens=${TP%%:*} exit=$?  output: $F"
  done
  chown harshanavkis $F ${F%.csv}.err
done; done
