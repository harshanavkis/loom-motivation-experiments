#!/usr/bin/env bash
# Copy the GPU host's benchmark outputs back into the repo, next to the committed ones (git diff then
# shows what a rerun changed). Usage (repo root or anywhere):
#   scripts/collect_results.sh steve            # copy
#   scripts/collect_results.sh steve --dry-run  # list what would change
set -euo pipefail
HOST=${1:?usage: collect_results.sh <ssh host> [--dry-run]}; shift
R="$(dirname "$(readlink -f "$0")")/.."
CX=$R/m5-rdma-init/steve-cx7; M3=$R/m3-sm-share
RS=(rsync -ai --ignore-missing-args "$@")
get() { local dst=$1; shift; local src=(); for f in "$@"; do src+=("$HOST:loom-experiments/$f"); done; "${RS[@]}" "${src[@]}" "$dst/"; }

# Fig 3a/b and the dispatch breakdown sweeps (raw CSVs; summaries are rebuilt by scripts/analyze_local.sh)
get $CX/bd_v2 'gpu-posted/bd_v2/*.csv' 'gpu-posted/bd_v2/*.err'
# text numbers (Sections 1-2)
get $CX nvshmem-loopback/msgrate_steve.csv 'nvshmem-loopback/put_lat_steve*.txt' \
        'gpu-posted/deepep_post_*.csv' 'gpu-posted/dispatch_ibgda_block_H*.csv' paper/dispatch_sweep_all.csv \
        latency/proxy_b2_numa0.csv latency/fence_cost_numa0.csv latency/nic_post_numa0.csv latency/tma_bw.csv \
        latency/dev_ce_check.csv latency/ce_triggered_idle.csv \
        'latency/ce_latency_*.csv' 'latency/lat_cpu_posted_*.csv' 'latency/dispatch_ce_H*.csv' 'latency/dispatch_sm_H*.csv'
get $M3/deepgemm 'deepgemm/held_m*.csv' 'deepgemm/locked_m*.csv' deepgemm/locked_clocks.csv deepgemm/compare.csv
# ARGUMENT-only numbers
get $M3/gpu-interference 'gpu-interference/results_steve.*' gpu-interference/summary_steve.md gpu-interference/ce_streams_steve.csv \
        'gpu-interference/rdma_*.csv' 'gpu-interference/rdma_cpu*.txt' gpu-interference/rdma_gpu2gpu_7k.cli.txt
get $M3/gpu-posted 'gpu-posted/results_steve.*'
get $M3/steve-rdma/nvshmem-loopback nvshmem-loopback/put_bw_steve.txt
