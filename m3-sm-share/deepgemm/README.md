# DeepGEMM next to SMs reserved for communication (Fig 3d)

## Question

DeepSeek-V3 reserves 20 of an H800's 132 SMs for its all-to-all kernels (report §3.2.2, §3.5.1), and its persistent DeepEP kernels run beside GEMMs planned for the remaining SMs. What does that reservation cost DeepSeek's own expert GEMM? The earlier answer, 51 points at 20 SMs (`../gpu-interference`), came from cuBLAS. cuBLAS's clustered kernels fit scattered held SMs poorly, so that number may not apply to DeepSeek's setup.

## Method

- **GEMM**: DeepGEMM `m_grouped_fp8_fp4_gemm_nt_contiguous`, FP8 with 1×128 / 128×128 scales (DeepSeek's format), at commit `057ca59`. Shapes: 8 local experts × ~1024 or ~4096 tokens, each 0.7–1.3× the mean as in DeepGEMM's tests; K = 7168, N = 4096, DeepSeek-V3's up/gate projection. Throughput is counted over valid rows only.
- **Modes**, each relative to the same GEMM alone on all 132 SMs:
  - `partitioned`: `deep_gemm.set_num_sms(132 − k)`, nothing else running.
  - `held`: the same, plus k CTAs holding k SMs for the whole run. Each holder takes 200 KB of shared memory, so it gets one SM to itself, and they launch in 2-CTA clusters, sleeping on a flag. This mirrors DeepEP's persistent normal kernels. With cuBLAS, an idle holder cost a GEMM what GPU-initiated puts cost (`../gpu-posted`).
  - `ce50` / `cemax`: the GEMM on all SMs while the copy engine copies 64 MiB HBM→HBM on another stream (`cudaMemcpyBatchAsync` + `PreferOverlapWithCompute`). `ce50` is paced to 50 GB/s by a host thread, a 400 Gb/s NIC's line rate; `cemax` is unpaced, ~85 GB/s.
- **Runs**: 100 GEMMs per measurement, 5 repetitions, median. Every SM count is compiled and run once before any holder exists, and modules load eagerly.
- **Environment**: steve's H200 NVL, in a container built from `Dockerfile`: CUDA 12.9, PyTorch cu129, DeepGEMM from source. DeepGEMM `main` requires CUDA ≥ 12.9 and elfutils headers.

## Results (FP8, % of alone, median of 5; `held_m{1024,4096}.csv`)

| tokens per expert | alone | ce50 | cemax | k = 4 held (part.) | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| ~1024 | 992 TFLOP/s | 99.6 | 95.1 | 92.9 (91.5) | 92.3 (92.6) | 86.6 (86.8) | 86.3 (87.2) |
| ~4096 | 1052 TFLOP/s | 97.4 | 94.9 | 95.5 (94.3) | 95.1 (95.3) | 91.4 (91.4) | 89.2 (89.7) |
| share of SMs left | | | | 97.0 | 93.9 | 87.9 | 84.8 |

- **Held = partitioned within ~1 point at every k.** DeepGEMM shows none of cuBLAS's placement penalty (cuBLAS BF16: 79.1% partitioned, 48.9% held at k = 20).
- **Reserving 20 SMs costs DeepSeek's FP8 expert GEMM 11–19%**, about the 15% of SMs given up. Run-to-run variation is several points: the ~1024 shape at k = 20 measured 81–82% in two earlier runs (`v1/`, `v2/`) and 86% in this one. Clocks are not locked on this shared host. The ~1024 shape at k = 4 is unstable (86–96%), probably a different kernel configuration for that SM count.
- **The copy engine at a NIC's line rate costs 0.4–2.6%.** Unpaced it costs ~5%: FP8 GEMMs lean on HBM bandwidth more than cuBLAS's BF16 one, which kept 99.9% beside 50 GB/s.
- **At 20 SMs, the zero-SM path is therefore worth 8–13 points of this GEMM's throughput, not 51.**
- **DeepGEMM's BF16 path** (`v2/`) runs at 545–636 TFLOP/s, below cuBLAS's 811, and barely notices fewer SMs (≥ 92% at k = 20). It is not used.

## Reproduce (on steve)

```sh
cd ~/loom-experiments/deepgemm && docker build -t loom-deepgemm .
for M in 1024 4096; do
  docker run --rm --device nvidia.com/gpu=all --ipc=host -v $PWD:/work -w /work loom-deepgemm \
    bash -c "cat /opt/DeepGEMM.commit; python dg_held.py --dtypes fp8 --ks 4,8,16,20 --reps 5 --iters 100 --m-per-group $M" > held_m$M.csv
done
```

## Files

- `Dockerfile`, `dg_held.py`: the image and the benchmark.
- `held_m{1024,4096}.csv`: the plotted run (FP8, with `ce50`/`cemax`).
- `v1/`: the first run (FP8 + BF16, no copy engine). `v2/`: FP8 + BF16 with an unpaced copy engine.
