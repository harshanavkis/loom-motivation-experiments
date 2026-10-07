#!/usr/bin/env bash
# Recreate src/ (gitignored): the exact library checkouts every source analysis (M1, M2, M3) used.
# Usage (repo root or anywhere): scripts/fetch_sources.sh        Idempotent: existing checkouts are kept.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p src && cd src
clone() { [ -d "$2" ] || git clone ${3:-} "$1" "$2"; }

clone https://github.com/NVIDIA/nccl nccl
git -C nccl checkout -q 12df1a11afad322be5a204a2db890161cbf8131d                       # v2.32.3-1
[ -d nccl-v2.18.5 ] || git -C nccl worktree add --detach ../nccl-v2.18.5 559b70f86c190a0d8f67f0d7a0f2c9810dd1e8c7   # v2.18.5-1

clone https://github.com/NVIDIA/nvshmem nvshmem
git -C nvshmem checkout -q 270759e5481b16ef5a71930e1d9b8df184cd7072                    # v3.8.0-0 (M2 header sizes only)

clone https://github.com/deepseek-ai/DeepEP DeepEP
git -C DeepEP checkout -q 93eb6eb238127e96c6d7a4a625a6dad158348509                     # main = V2.5 (NCCL GIN)
[ -d DeepEP-v1-last ] || git -C DeepEP worktree add --detach ../DeepEP-v1-last a56d6156febcd9976e55adc85b5155bfac9f28f8   # last V1 (NVSHMEM)

clone https://github.com/deepseek-ai/DeepEP DeepEP-0sm                                  # M3 0-SM study: PRs and branches
git -C DeepEP-0sm checkout -q 93eb6eb238127e96c6d7a4a625a6dad158348509
git -C DeepEP-0sm fetch -q origin pull/347/head:pr347 pull/453/head:pr453 '+refs/heads/*:refs/remotes/origin/*' || true

clone https://github.com/deepseek-ai/DeepGEMM DeepGEMM --recursive
git -C DeepGEMM checkout -q 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7 && git -C DeepGEMM submodule update -q --init --recursive

for d in nccl nccl-v2.18.5 nvshmem DeepEP DeepEP-v1-last DeepEP-0sm DeepGEMM; do printf '%-16s %s\n' "$d" "$(git -C $d rev-parse --short HEAD)"; done
