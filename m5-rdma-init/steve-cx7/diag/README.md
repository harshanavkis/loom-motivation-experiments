# Diagnostics behind dispatch_bd.cu (steve, H200 NVL)

Build each with nixpkgs CUDA on steve, e.g.
`nvcc -O3 -std=c++17 -arch=sm_90 -I$CUDART/include -L$CUDART/lib -Xlinker -rpath -Xlinker $CUDART/lib:/run/opengl-driver/lib gt_res.cu -o gt_res`
with `CUDART=$(NIXPKGS_ALLOW_UNFREE=1 nix build --impure --no-link --print-out-paths nixpkgs#cudaPackages.cuda_cudart | head -1)` and nvcc from `nix shell --impure nixpkgs#cudaPackages.cuda_nvcc`.

- `gt_res.cu`: smallest `%globaltimer` step and the SM clock (`clock64` vs `%globaltimer` over 10 ms). Result: 32 ns, 1.780 GHz.
- `gt_hist.cu`: histogram of `%globaltimer` steps seen by one thread. Result: 32 ns (82%) and 64 ns (18%). The 256 ns-quantized stamps first seen in dispatch_bd were NOT the timer: they came from reading it right after `__syncthreads()` (see the TRAP in ../../README.md).
- `gpudirect_ordering.cu`: `cudaDevAttrGPUDirectRDMAWritesOrdering` and the flush options. Result: 100 (owner) and 1, so a kernel on the owning GPU sees the NIC's writes in order: a receiver that sees the signal sees the data.
