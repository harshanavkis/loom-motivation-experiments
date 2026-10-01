#!/usr/bin/env bash
# Local-peer dispatch sweeps on steve (GPU-triggered copy engine, then SM stores): H in {1024, 7168} x {idle, GEMM load}. No root needed.
set -uo pipefail
cd "$(dirname "$0")"
NUMA=$(nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
for H in 1024 7168; do for LOAD in 0 1; do
  X=""; [ $LOAD = 1 ] && X="--load"
  timeout 900 $NUMA -N 0 -m 0 ./dispatch_ce --H $H $X < /dev/null > dispatch_ce_H${H}_load${LOAD}.csv 2>&1
  echo "H=$H load=$LOAD exit=$?"
  timeout 900 $NUMA -N 0 -m 0 ./dispatch_ce --sm --H $H $X < /dev/null > dispatch_sm_H${H}_load${LOAD}.csv 2>&1
  echo "sm H=$H load=$LOAD exit=$?"
done; done
