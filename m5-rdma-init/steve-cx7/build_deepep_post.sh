#!/usr/bin/env bash
# Build deepep_post.cu on steve (relocatable device code, static NVSHMEM 3.6.5 device lib).
# deepep-include/ holds DeepEP V1 a56d615 headers, unmodified:
#   legacy/{compiled,utils,ibgda_device}.cuh  <- csrc/kernels/legacy/
#   deep_ep/common/{compiled,exception}.cuh   <- deep_ep/include/deep_ep/common/
set -euo pipefail
cd "$(dirname "$0")"
export NIXPKGS_ALLOW_UNFREE=1
NV=$HOME/loom-experiments/nvshmem-3.6.5
P() { nix build --impure --no-link --print-out-paths "nixpkgs#cudaPackages.$1" | head -1; }
CUDART=$(P cuda_cudart); CCCL=$(P cuda_cccl)
# One port to NVSHMEM 3.6.5: its RC QPs are laid out QP-major (rcs[id * npes + pe], see its
# ibgda_get_rc), DeepEP a56d615 indexes them PE-major for an older NVSHMEM. Patch only that
# index, into deepep-include/legacy/ibgda_device_nv365.cuh (the rest of the post path as is).
python3 - <<'PY'
src = open('deepep-include/legacy/ibgda_device.cuh').read()
old = '.rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];'
assert old in src
new = '.rcs[(id % (num_rc_per_pe * state->num_devices_initialized)) * nvshmemi_device_state_d.npes + pe];'
open('deepep-include/legacy/ibgda_device_nv365.cuh', 'w').write(src.replace(old, new))
PY
RC_DEV=$(nix build --no-link --print-out-paths 'nixpkgs#rdma-core^dev' | head -1)   # infiniband/mlx5dv.h
# DeepEP setup.py flags (+ --expt-relaxed-constexpr, which torch cpp_extension adds)
nix shell --impure nixpkgs#cudaPackages.cuda_nvcc -c nvcc -O3 -std=c++17 -rdc=true -arch=sm_90 \
  --expt-relaxed-constexpr --extended-lambda --diag-suppress=128,2417 --ptxas-options=--register-usage-level=10 \
  -DDISABLE_AGGRESSIVE_PTX_INSTRS \
  -I"$NV/include" -I"$CUDART/include" -I"$CCCL/include" -I"$RC_DEV/include" -Ideepep-include/legacy -Ideepep-include \
  -L"$NV/lib" -L"$CUDART/lib" -L/run/opengl-driver/lib \
  -Xlinker -rpath -Xlinker "$NV/lib:$CUDART/lib:/run/opengl-driver/lib" \
  deepep_post.cu -o deepep_post -lnvshmem_host -lnvshmem_device -lcudart -lcuda
echo built ./deepep_post
