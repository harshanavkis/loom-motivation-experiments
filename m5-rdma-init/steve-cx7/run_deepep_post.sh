#!/usr/bin/env bash
# DeepEP's post path vs NVSHMEM's generic put on steve (see deepep_post.cu): single-message
# latency, then the per-token dispatch, both modes. Root: unlimited locked memory for IBGDA.
# Usage on steve: sudo ~/loom-experiments/gpu-posted/run_deepep_post.sh
set -uo pipefail
D=/home/harshanavkis/loom-experiments/gpu-posted
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
NUMA=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
ulimit -l unlimited
export HOME=/home/harshanavkis
source $HOME/loom-experiments/nvshmem-loopback/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none       # pure IBGDA, no host proxy
export NVSHMEM_QP_DEPTH=1024               # DeepEP's default (deep_ep/buffers/legacy.py)
run() { timeout 600 $MPI/bin/mpiexec -n 2 $NUMA --cpunodebind=0 --membind=0 $D/deepep_post "$@" < /dev/null; }
# latency: NVSHMEM's default QPs per PE, as the shmem_put_latency perftest (put_lat_steve.txt);
# with 24, nvshmem_quiet polls every QP and the NVSHMEM baseline inflates to ~34 us
OUT=$D/deepep_post_lat.csv
{ for rep in 1 2 3; do for m in nvshmem deepep; do run --test lat --mode $m; done; done; } > $OUT 2> ${OUT%.csv}.err
echo "lat exit=$?  output: $OUT"
export NVSHMEM_IBGDA_NUM_RC_PER_PE=24      # dispatch: as run_dispatch.sh (DeepEP V1's setting)
for H in 1024 7168; do
  OUT=$D/deepep_post_dispatch_H$H.csv
  { for m in nvshmem deepep; do run --test dispatch --mode $m --H $H; done; } > $OUT 2> ${OUT%.csv}.err
  echo "dispatch H=$H exit=$?  output: $OUT"
done
chown harshanavkis $D/deepep_post_*
