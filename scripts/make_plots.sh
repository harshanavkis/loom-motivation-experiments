#!/usr/bin/env bash
# Draw Fig 1 (loom-mix) and Fig 3 (loom-costs) from the committed result files, and print every number
# the text quotes. Usage: scripts/make_plots.sh [out dir, default plots/out]   e.g. ~/loom-paper/plots
set -euo pipefail
R="$(dirname "$(readlink -f "$0")")/.."
OUT=$(readlink -f "${1:-$R/plots/out}"); mkdir -p "$OUT"
cd $R/plots
nix shell --impure --expr '(builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux.python3.withPackages (p: with p; [matplotlib seaborn pandas])' \
  -c python3 plot_motivation.py "$OUT"
echo "wrote $OUT/loom-mix.{pdf,png} $OUT/loom-costs.{pdf,png}"
