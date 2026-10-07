#!/usr/bin/env bash
# Every analysis that needs no GPU: source analyses (M1, M2), trace analyses (Chakra), routing (M6), and
# the summaries of the GPU runs (M5). Rewrites the committed outputs in place, so `git status` afterwards
# shows exactly what a rerun changed (a clean tree = reproduced).
# Needs: nix; src/ (scripts/fetch_sources.sh) for M1/M2; the Chakra traces (see README) for --traces.
#   scripts/analyze_local.sh            M1, M2, M6, M5 summaries
#   scripts/analyze_local.sh --traces   + Chakra (Fig 1, Section 2 trace numbers; ~3 min)
set -euo pipefail
R="$(dirname "$(readlink -f "$0")")/.."
NP() { nix shell --impure --expr "(builtins.getFlake \"nixpkgs\").legacyPackages.x86_64-linux.python3.withPackages (p: with p; [$1])" -c python3 "${@:2}"; }
say() { echo "== $*"; }

say "M1: NCCL lines by fabric (Table 1, Section 2.3 #1)"
cd $R/m1-nccl-loc
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl v2.32.3-1 > summary_v2.32.3-1.txt
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl-v2.18.5 v2.18.5-1 > summary_v2.18.5-1.txt

say "M2: transport state per GPU (Fig 3c, Table 1, Section 2.3 #3)"
cd $R/m2-device-state
nix build --no-link 'nixpkgs#rdma-core^dev'   # run_sizes.sh needs mlx5dv.h in the local store
./run_sizes.sh > sizes.out && rm -f sizes sizes_gdaki
python3 totals.py > totals.out

say "M6: DeepSeek-V3 routing, copies leaving the scale-up domain (Section 2.3 #2)"
cd $R/m6-routing && NP numpy routing_traffic.py 50000 > /dev/null

say "M5: dispatch / combine breakdown summaries (Fig 3a/b, Section 2.3 #2)"
cd $R/m5-rdma-init/steve-cx7
python3 summarize_bd.py $(ls bd_v2/dispatch_bd_*.csv | grep -v -e _t1_ -e _t3_) --csv bd_v2/summary.csv > bd_v2/summary.txt
python3 summarize_bd.py bd_v2/dispatch_bd_t1_*.csv --csv bd_v2/summary_t1.csv > /dev/null
python3 summarize_bd.py bd_v2/dispatch_bd_t3_*.csv --csv bd_v2/summary_t3.csv > /dev/null
python3 summarize_bd.py bd_v2/combine_bd_t3_*.csv --csv bd_v2/summary_combine_t3.csv > /dev/null
python3 summarize_d3.py bd_v2/d3gemm_*.csv --csv bd_v2/d3gemm_summary.csv > bd_v2/d3gemm_summary.txt

if [ "${1:-}" = --traces ]; then
  say "Chakra traces (Fig 1, Section 2.1 trace numbers)"
  cd $R/m3-sm-share/chakra
  T=${T:-/scratch/$USER/chakra-traces}
  [ -d "$T/Mixtral" ] || DST=$T ./extract.sh
  python3 analyze_et.py Mixtral-8x7B  8 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et        > out_et_Mixtral-8x7B.txt
  python3 analyze_et.py Mixtral-8x22B 8 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et > out_et_Mixtral-8x22B.txt
  python3 analyze_et.py Llama3-70B    8 $T/Llama3/Llama3-70B/16TP/rank.*.et              > out_et_Llama3-70B.txt
  python3 analyze_nccl_et.py Mixtral-8x7B  8 450 12.5 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et         > out_nccl_et_Mixtral-8x7B.txt
  python3 analyze_nccl_et.py Mixtral-8x22B 8 450 12.5 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et  > out_nccl_et_Mixtral-8x22B.txt
  python3 analyze_nccl_et.py Llama3-70B    8 450 12.5 $T/Llama3/Llama3-70B/16TP/rank.*.et               > out_nccl_et_Llama3-70B.txt
  D=$T/Mixtral/Mixtral-8x7B/nemo_raw
  if ls $D/device_{0..7}.json > /dev/null 2>&1; then   # device_{1,4,5,7}.json are not in the public zips (see README)
    python3 analyze_kineto.py 8 $D/device_{0..7}.json > out_kineto_Mixtral-8x7B.txt
    python3 analyze_moe.py 450 $T/Mixtral/Mixtral-8x7B $D/device_{0..7}.json > out_moe_Mixtral-8x7B.txt
    python3 project_offsm.py > out_projection.txt
    python3 analyze_skew.py 450 $D > out_skew_Mixtral-8x7B.txt
  else
    echo "   skipped the Mixtral-8x7B device-trace analyses: $D/device_{0..7}.json incomplete"
  fi
fi
cd $R && git status --short -- . ':!src'
