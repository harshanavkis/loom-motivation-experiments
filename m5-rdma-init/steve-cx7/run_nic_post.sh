#!/usr/bin/env bash
# nic_post variants on steve, proxy on NUMA 0 as in proxy_b2_numa0.csv, 3 runs each.
# Output: nic_post_numa0.csv (src,bf,size,post_med_us,cqe_med_us,cqe_p99_us).
set -euo pipefail
cd "$(dirname "$0")"
NUMA=$(nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
run() { env "$@" "$NUMA" -N 0 -m 0 ./nic_post "$SRC" 20000; }
{
  echo "src,bf,size,post_med_us,cqe_med_us,cqe_p99_us"
  for rep in 1 2 3; do
    for bf in 0 1; do                               # MLX5_SHUT_UP_BF=1: doorbell, the NIC fetches the WQE
      SRC=gpu    run MLX5_SHUT_UP_BF=$bf              # payload read from GPU memory (= proxy_b2 with bf=0)
      SRC=host   run MLX5_SHUT_UP_BF=$bf              # payload read from host memory
      SRC=inline run MLX5_SHUT_UP_BF=$bf              # no payload read: data inside the WQE
    done
  done
} > nic_post_numa0.csv
cat nic_post_numa0.csv
