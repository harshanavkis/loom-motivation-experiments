#!/usr/bin/env bash
# dg_held.py with the GPU clock locked (owner, 2026-10-06), so runs are comparable. Usage on steve:
#   ~/loom-experiments/deepgemm/run_locked.sh   (needs passwordless sudo for nvidia-smi; resets on exit)
set -uo pipefail
cd "$(dirname "$0")"
CLK=${CLK:-1410}   # 1500 MHz touched the 600 W power cap under the FP8 GEMM; 1410 leaves margin
sudo -n nvidia-smi -lgc $CLK,$CLK
trap 'sudo -n nvidia-smi -rgc' EXIT
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > locked_clocks.csv &
MON=$!
for M in 1024 4096; do
  docker run --rm --device nvidia.com/gpu=all --ipc=host -v $PWD:/work -w /work loom-deepgemm \
    bash -c "cat /opt/DeepGEMM.commit; echo '# locked SM clock $CLK MHz'; python dg_held.py --dtypes fp8 --ks 4,8,16,20 --reps 5 --iters 100 --m-per-group $M" \
    > locked_m$M.csv 2> locked_m$M.err
  echo "m$M exit=$?"
done
kill $MON
