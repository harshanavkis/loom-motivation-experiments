#!/usr/bin/env bash
# Copy every benchmark source and script to the GPU host's run layout, ~/loom-experiments/<dir>/
# (the layout every runner assumes). Results already on the host are never touched: only the files
# listed here are sent. Usage (repo root or anywhere):
#   scripts/deploy_gpu_host.sh steve            # copy
#   scripts/deploy_gpu_host.sh steve --dry-run  # show what would change
set -euo pipefail
HOST=${1:?usage: deploy_gpu_host.sh <ssh host> [--dry-run]}; shift
R="$(dirname "$(readlink -f "$0")")/.."
CX=$R/m5-rdma-init/steve-cx7; M3=$R/m3-sm-share
RS=(rsync -ai --mkpath "$@")
put() { local dst=$1; shift; "${RS[@]}" "$@" "$HOST:loom-experiments/$dst/"; }

# host setup and checks (sudo steve-rdma/setup_root.sh after every reboot)
put steve-rdma      $M3/steve-rdma/{README.md,build.sh,check.sh,setup_root.sh}
put perftest-cuda   $M3/steve-rdma/perftest-cuda/{build.sh,test_loopback.sh}
# NVSHMEM loopback environment and the NVSHMEM perftest runners
put nvshmem-loopback $M3/steve-rdma/nvshmem-loopback/{env.sh,run_put_bw.sh} $CX/{run_msgrate.sh,run_put_lat.sh}
# M3: cuBLAS GEMM beside held SMs / copy engine / CPU-posted RDMA
put gpu-interference $M3/gpu-interference/{build.sh,run.sh,summarize.py,interfere.cu,ce_streams.cu,rdma_cpu_posted.sh,rdma_cpu_posted_7k.sh}
# GPU-initiated RDMA: DeepEP post path, dispatch_ibgda, dispatch_bd (Fig 3a/b), NVSHMEM puts beside a GEMM
put gpu-posted      $M3/gpu-posted/{nvshmem_interfere.cu,run.sh} \
                    $CX/{bd_one.sh,build_deepep_post.sh,build_dispatch_bd.sh,deepep_post.cu,dispatch_bd.cu,dispatch_ibgda.cu,fetch_deepep_include.sh,run_deepep_post.sh,run_dispatch.sh,run_dispatch_bd.sh,run_dispatch_block.sh,run_dispatch_d3.sh}
"${RS[@]}" $CX/build_gpu_posted.sh "$HOST:loom-experiments/gpu-posted/build.sh"   # builds nvshmem_interfere + dispatch_ibgda
put gpu-posted/diag $CX/diag/{README.md,gt_res.cu,gt_hist.cu,gpudirect_ordering.cu}
# latency / copy-engine / CPU-proxy programs
put latency         $CX/{build.sh,ce_latency.cu,ce_triggered.cu,dev_ce_check.cu,dispatch_ce.cu,dispatch_proxy.cu,fence_cost.cu,nic_post.cu,proxy_b2.cu,tma_bw.cu,lat_cpu_posted.sh,run_dispatch_ce.sh,run_dispatch_proxy.sh,run_nic_post.sh}
# DeepSeek's FP8 GEMM next to reserved SMs (docker)
put deepgemm        $M3/deepgemm/{Dockerfile,dg_held.py,dg_compare.py,run_locked.sh}
# one-command build and paper runs
put paper           $R/scripts/gpu-host/{build_all.sh,run_paper.sh}
