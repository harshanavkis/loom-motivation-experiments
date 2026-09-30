#!/usr/bin/env bash
# Full sweep on steve (H200 NVL). The GPU sits on NUMA node 0; bind CPU and host memory there.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-results_steve.csv}
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,power.limit --format=csv > "${OUT%.csv}.gpu.txt"
NIXPKGS_ALLOW_UNFREE=1 nix shell nixpkgs#numactl -c numactl --cpunodebind=0 --membind=0 ./interfere --reps 5 --out "$OUT"
