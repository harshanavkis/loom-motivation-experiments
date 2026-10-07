#!/usr/bin/env bash
# Run the GPU benchmarks behind the paper's Sections 1-2, on the GPU host, as the normal user (the
# NVSHMEM runners are started with sudo). Prerequisites: build_all.sh, and after every reboot
# `sudo ~/loom-experiments/steve-rdma/setup_root.sh` (check.sh must show identity x3 and
# PeerMappingOverride=1). Nothing else may run on the GPU. Outputs stay in each run directory;
# scripts/collect_results.sh <host> copies them into the repo.
#   run_paper.sh fig3    Fig 3a/b: dispatch + combine breakdown, 1/16/128 tokens       (~15 min)
#   run_paper.sh text    every measured number in the text of Sections 1-2             (~3 h)
#   run_paper.sh extra   numbers kept in the paper's % ARGUMENT comments only           (~2.5 h)
#   run_paper.sh all
set -uo pipefail
H=$HOME/loom-experiments
STAGE=${1:?usage: run_paper.sh fig3|text|extra|all}
NUMA=$(nix build --no-link --print-out-paths nixpkgs#numactl | grep -v -- -man | head -1)/bin/numactl
say() { echo "== $(date +%T) $*"; }
# remote paths as plotted: ordered = DeepEP V1 low latency (3 QPs per destination, signal behind each
# QP's data), flush = DeepEP V2.5 normal (QPs follow warps, wait for all completions, then signal)
PATHS=("local:--path local" "ordered-destL3:--path ordered --qp dest --nq 3 --post lane" "flush-warpL:--path flush --qp warp --post lane")

fig3() {
  mkdir -p $H/gpu-posted/bd_v2
  for V in "${PATHS[@]}"; do
    N=${V%%:*}; A=${V#*:}
    say "dispatch $N";  F=$H/gpu-posted/bd_v2/dispatch_bd_t3_${N}_H7168_load0.csv
    sudo bash $H/gpu-posted/bd_one.sh $A --H 7168 --tokens 1,16,128 --iters 20 > $F 2> ${F%.csv}.err
    say "combine $N";   F=$H/gpu-posted/bd_v2/combine_bd_t3_${N}_H14336_load0.csv
    sudo bash $H/gpu-posted/bd_one.sh $A --combine --H 14336 --tokens 1,16,128 --iters 20 > $F 2> ${F%.csv}.err
  done
}

text() {
  say "message rate, GPU-initiated vs CPU proxy (Section 2.2, intro)"
  sudo $H/nvshmem-loopback/run_msgrate.sh
  say "DeepEP post path vs NVSHMEM put: SM time per post, single message, decode dispatch (Sections 2.2, 2.3 #2)"
  sudo $H/gpu-posted/run_deepep_post.sh
  say "NVSHMEM put latency idle and beside a GEMM (Section 2.3 #2: 10-22%)"
  sudo $H/nvshmem-loopback/run_put_lat.sh && sudo $H/nvshmem-loopback/run_put_lat.sh load
  say "dispatch, GPU-initiated vs CPU proxy (Section 2.2: 2.1-5.4x)"
  sudo $H/gpu-posted/run_dispatch.sh < /dev/null
  sudo $H/gpu-posted/run_dispatch_block.sh < /dev/null
  $H/latency/run_dispatch_proxy.sh
  (cd $H && cat gpu-posted/dispatch_ibgda_H{1024,7168}_load{0,1}.csv latency/dispatch_proxy_H{1024,7168}_load{0,1}.csv > paper/dispatch_sweep_all.csv)
  say "CPU proxy single put, fences, NIC post, TMA, device-launched copy engine (Sections 2.1-2.3)"
  cd $H/latency
  $NUMA -N 0 -m 0 ./proxy_b2 20000 > proxy_b2_numa0.csv
  $NUMA -N 0 -m 0 ./fence_cost > fence_cost_numa0.csv
  ./run_nic_post.sh
  $NUMA -N 0 -m 0 ./tma_bw > tma_bw.csv
  $NUMA -N 0 -m 0 ./dev_ce_check > dev_ce_check.csv
  $NUMA -N 0 -m 0 ./ce_triggered < /dev/null > ce_triggered_idle.csv
  say "dispatch breakdown sweep, 16-4096 tokens, 2-20 CTAs (Section 2.3 #2: training size, SM time)"
  mkdir -p $H/gpu-posted/bd_v2
  sudo env OUT=$H/gpu-posted/bd_v2/dispatch_bd $H/gpu-posted/run_dispatch_bd.sh --ctas 2,4,8,20
  sudo env OUT=$H/gpu-posted/bd_v2/dispatch_bd VARIANTS="local:--path_local flush-warpL:--path_flush_--qp_warp_--post_lane ordered-destL3:--path_ordered_--qp_dest_--nq_3_--post_lane flush-destL3:--path_flush_--qp_dest_--nq_3_--post_lane" \
    $H/gpu-posted/run_dispatch_bd.sh --ctas 2,4,8,20
  say "DeepGEMM next to 20 reserved SMs (Section 2.3 #2: 81-94%)"
  cd $H/deepgemm
  for M in 1024 4096; do
    docker run --rm --device nvidia.com/gpu=all --ipc=host -v $PWD:/work -w /work loom-deepgemm \
      bash -c "cat /opt/DeepGEMM.commit; python dg_held.py --dtypes fp8 --ks 4,8,16,20 --reps 5 --iters 100 --m-per-group $M" > held_m$M.csv
  done
  ./run_locked.sh        # same with the SM clock locked at 1410 MHz (sudo nvidia-smi -lgc)
}

extra() {
  say "cuBLAS GEMM beside held SMs / copy engine (M3)"
  (cd $H/gpu-interference && ./run.sh results_steve.csv && python3 summarize.py results_steve.csv > summary_steve.md)
  (cd $H/gpu-interference && $NUMA --cpunodebind=0 --membind=0 ./ce_streams > ce_streams_steve.csv)
  $H/gpu-interference/rdma_cpu_posted.sh && $H/gpu-interference/rdma_cpu_posted_7k.sh
  say "NVSHMEM puts beside the GEMM (M3)"
  sudo $H/gpu-posted/run.sh 3
  say "compute lost per dispatch (D3) and vs dispatch rate"
  sudo $H/gpu-posted/run_dispatch_d3.sh
  sudo env FILLER=gemm-up WINDOW_MS=1000 OUT=$H/gpu-posted/bd_v2/d3gemm $H/gpu-posted/run_dispatch_d3.sh
  sudo env FILLER=gemm-up WINDOW_MS=1000 REPS=3 TPS_1K="16:100,150,200,300,500,1000 128:200,300,400,600,1000,2000" \
    TPS_7K="16:100,150,200,300,500,1000 128:500,700,1000,1500,2000,3000" OUT=$H/gpu-posted/bd_v2/d3rate $H/gpu-posted/run_dispatch_d3.sh
  say "copy engine latency, CPU-posted latency, copy-engine dispatch"
  (cd $H/latency && $NUMA --cpunodebind=0 --membind=0 ./ce_latency --iters 20000 > ce_latency_idle.csv \
                 && $NUMA --cpunodebind=0 --membind=0 ./ce_latency --iters 5000 --load > ce_latency_load.csv)
  $H/latency/lat_cpu_posted.sh idle && $H/latency/lat_cpu_posted.sh load
  $H/latency/run_dispatch_ce.sh
  say "cuBLAS vs DeepGEMM in one harness"
  (cd $H/deepgemm && docker run --rm --device nvidia.com/gpu=all --ipc=host -v $PWD:/work -w /work loom-deepgemm \
     python dg_compare.py --profile --reps 5 > compare.csv)
}

$H/steve-rdma/check.sh
case $STAGE in
  fig3) fig3 ;; text) text ;; extra) extra ;; all) fig3; text; extra ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
say "done ($STAGE); on the repo host: scripts/collect_results.sh <host>"
