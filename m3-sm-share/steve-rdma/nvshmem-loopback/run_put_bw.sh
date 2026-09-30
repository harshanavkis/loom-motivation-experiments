#!/usr/bin/env bash
# GPU-posted RDMA (NVSHMEM IBGDA) on steve: 2 PEs on the H200 over the CX-7
# port0 <-> port1 loopback. Needs unlimited locked memory (IBGDA pins its CQs in
# host memory; the default 8 MiB limit makes ibv_create_cq fail), hence sudo.
# Usage on steve:  sudo ~/loom-experiments/nvshmem-loopback/run_put_bw.sh
set -uo pipefail
D=/home/harshanavkis/loom-experiments/nvshmem-loopback
MPI=$(sudo -u harshanavkis nix build --no-link --print-out-paths nixpkgs#mpich | head -1)
ulimit -l unlimited
export HOME=/home/harshanavkis
source $D/env.sh
export NVSHMEM_REMOTE_TRANSPORT=none      # device-side puts only: IBGDA, no host ibrc
OUT=$D/put_bw_steve.txt
{
  echo "# $(date -Is) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  grep RegistryDwords: /proc/driver/nvidia/params
  NVSHMEM_DEBUG=INFO timeout 240 $MPI/bin/mpiexec -n 2 \
    $NVSHMEM_HOME/bin/perftest/device/pt-to-pt/shmem_put_bw -b 4096 -e 4194304 -f 4
} > $OUT 2>&1
echo "exit=$?  output: $OUT"
chown harshanavkis $OUT
