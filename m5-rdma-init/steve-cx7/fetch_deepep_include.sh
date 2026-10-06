#!/usr/bin/env bash
# Recreate deepep-include/ (DeepEP V1 a56d615 headers, unmodified), which deepep_post.cu and
# dispatch_bd.cu build against (see build_deepep_post.sh, which then writes the one-line port
# ibgda_device_nv365.cuh). Run on steve in ~/loom-experiments/gpu-posted/.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d)
git clone -q https://github.com/deepseek-ai/DeepEP "$T/DeepEP"
git -C "$T/DeepEP" checkout -q a56d615
mkdir -p deepep-include/legacy deepep-include/deep_ep/common
cp "$T"/DeepEP/csrc/kernels/legacy/{compiled,utils,ibgda_device}.cuh deepep-include/legacy/
cp "$T"/DeepEP/deep_ep/include/deep_ep/common/{compiled,exception}.cuh deepep-include/deep_ep/common/
rm -rf "$T"
echo "deepep-include/ ready; next: ./build_deepep_post.sh (ports the QP index) and ./build_dispatch_bd.sh"
