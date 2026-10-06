#!/usr/bin/env python3
"""DeepGEMM next to SMs held for communication: Fig 3d's setup with DeepSeek's own GEMM.

The expert up/gate GEMMs of one MoE layer on one GPU (DeepSeek-V3 shapes: 8 local experts x
~1024 tokens, K = 7168, N = 4096), as DeepGEMM's m-grouped contiguous GEMM, BF16 or FP8 (DeepSeek's
format, 1x128 / 128x128 scales). Three modes per k:
  alone        num_sms = 132, nothing else running (the baseline)
  partitioned  deep_gemm.set_num_sms(132 - k), nothing else running
  held         set_num_sms(132 - k), and k CTAs hold k SMs for the whole run, as DeepEP's persistent
               normal kernels do beside DeepGEMM: 200 KB of shared memory each (one per SM, no GEMM
               CTA beside them), 2-CTA clusters, sleeping on a flag (an idle holder costs a GEMM what
               GPU-initiated puts cost: ../gpu-posted)
  ce<R>        num_sms = 132, while the copy engine copies 64 MiB HBM -> HBM on another stream
               (cudaMemcpyBatchAsync + PreferOverlapWithCompute), the zero-SM path: paced to R GB/s by a
               host thread (50 = a 400G NIC's line rate, as ../gpu-interference), or unpaced (cemax)
Every num_sms is compiled and run once before any holder exists: a kernel loaded while holders spin
could wait for an idle device. Output: one CSV row per (rep, dtype, k, mode).
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault('CUDA_MODULE_LOADING', 'EAGER')
import torch
import deep_gemm
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, '/opt/DeepGEMM/tests')
from generators import MajorTypeAB, QuantConfig, generate_m_grouped_contiguous  # noqa: E402

HOLD_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void hold(volatile int* stop, int* started) {
  extern __shared__ char reserve[];   // never used: only keeps the SM to this CTA
  if (threadIdx.x == 0) {
    atomicAdd(started, 1);
    while (*stop == 0) __nanosleep(2000);
  }
}
// n back-to-back copies on the copy engine (a plain cudaMemcpyAsync D2D may run as an SM kernel)
void ce_copies(torch::Tensor dst, torch::Tensor src, int64_t n) {
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.srcLocHint.type = cudaMemLocationTypeDevice; attr.dstLocHint.type = cudaMemLocationTypeDevice;
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  for (int64_t i = 0; i < n; i++) {
    void* d = dst.data_ptr(); void* sp = src.data_ptr(); size_t bytes = dst.numel() * dst.element_size();
    size_t aidx = 0, fail = 0;
    TORCH_CHECK(cudaMemcpyBatchAsync(&d, &sp, &bytes, 1, &attr, &aidx, 1, &fail, s) == cudaSuccess, "batch copy failed");
  }
}
void launch_hold(torch::Tensor stop, torch::Tensor started, int64_t k, int64_t smem) {
  TORCH_CHECK(k % 2 == 0, "k must be even (2-CTA clusters)");
  cudaFuncSetAttribute(hold, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)k); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, hold, (volatile int*)stop.data_ptr<int>(), started.data_ptr<int>()) == cudaSuccess,
              "hold launch failed");
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtypes', default='bf16,fp8')
    ap.add_argument('--ks', default='4,8,16,20')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--m-per-group', type=int, default=1024)
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--ce-rates', default='50,0', help='copy-engine GB/s beside the GEMM; 0 = unpaced')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',')]
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    ext = load_inline('loom_hold', cpp_sources='void launch_hold(torch::Tensor, torch::Tensor, int64_t, int64_t);\n'
                      'void ce_copies(torch::Tensor, torch::Tensor, int64_t);',
                      cuda_sources=HOLD_SRC, functions=['launch_hold', 'ce_copies'], extra_cuda_cflags=['-arch=sm_90a'])
    ce_src = torch.empty(64 << 20, dtype=torch.uint8, device='cuda')
    ce_dst = torch.empty_like(ce_src)
    ce_stream = torch.cuda.Stream()
    deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    n, k_dim = 4096, 7168
    stop = torch.zeros(1, dtype=torch.int32, device='cuda')
    started = torch.zeros(1, dtype=torch.int32, device='cuda')
    hold_stream, gemm_stream = torch.cuda.Stream(), torch.cuda.Stream()

    gemms = {}
    for dtype in args.dtypes.split(','):
        torch.manual_seed(0)
        import random; random.seed(0)
        m, a, b, layout, d, ref_d, valid = generate_m_grouped_contiguous(
            args.groups, args.m_per_group, n, k_dim, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=False, use_bf16=(dtype == 'bf16'), quant_config=None if dtype == 'bf16' else QuantConfig())
        if dtype == 'bf16':
            fn = lambda a=a, b=b, d=d, layout=layout: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
        else:
            recipe, recipe_a, recipe_b = QuantConfig().get_recipes()
            fn = lambda a=a, b=b, d=d, layout=layout: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a, b, d, layout, disable_ue8m0_cast=True, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
        flop = 2.0 * float(valid.sum().item()) * n * k_dim   # valid rows only (padding rows are zeros)
        # correctness once, at full SM count
        deep_gemm.set_num_sms(nsm); fn(); torch.cuda.synchronize()
        diff = (d[valid].float() - ref_d[valid].float()).abs().mean() / ref_d[valid].float().abs().mean()
        print(f'# {dtype}: m={m} (valid {int(valid.sum())}), n={n}, k={k_dim}, {flop / 1e9:.1f} GFLOP, mean rel err {diff:.4f}',
              flush=True)
        gemms[dtype] = (fn, flop)
        for s in [nsm] + [nsm - x for x in ks]:   # compile and load every SM count first
            deep_gemm.set_num_sms(s); fn()
        torch.cuda.synchronize()

    def timed(fn, iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(gemm_stream):
            fn()   # warm
            a.record()
            for _ in range(iters):
                fn()
            b.record()
        b.synchronize()
        return a.elapsed_time(b) / iters * 1e3   # us per GEMM

    def held(fn, k, iters):
        stop.zero_(); started.zero_(); torch.cuda.synchronize()
        with torch.cuda.stream(hold_stream):
            ext.launch_hold(stop, started, k, 200 * 1024)
        t0 = time.time()
        while int(started.item()) < k:   # .item() runs on the default stream: holders are on their own
            if time.time() - t0 > 10:
                raise RuntimeError(f'holders did not start ({int(started.item())}/{k})')
            time.sleep(0.001)
        us = timed(fn, iters)
        stop.fill_(1)
        torch.cuda.synchronize()
        return us

    def with_ce(fn, iters, rate_gbps):
        # copies throughout the timed GEMMs; returns (us per GEMM, copy-engine GB/s achieved)
        if rate_gbps == 0:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            n = 80
            with torch.cuda.stream(ce_stream):
                a.record(); ext.ce_copies(ce_dst, ce_src, n); b.record()
            us = timed(fn, iters)
            b.synchronize()
            return us, n * ce_src.numel() / (a.elapsed_time(b) * 1e-3) / 1e9
        done, count = threading.Event(), [0]
        def pace():
            period = ce_src.numel() / (rate_gbps * 1e9)
            with torch.cuda.stream(ce_stream):
                t0 = time.perf_counter()
                while not done.is_set():
                    ext.ce_copies(ce_dst, ce_src, 1); count[0] += 1
                    while time.perf_counter() < t0 + count[0] * period and not done.is_set():
                        time.sleep(period / 20)
            ce_stream.synchronize()
        th = threading.Thread(target=pace); t0 = time.perf_counter(); th.start()
        time.sleep(0.005)                  # the copies are running before the timed GEMMs start
        us = timed(fn, iters)
        done.set(); th.join()
        return us, count[0] * ce_src.numel() / (time.perf_counter() - t0) / 1e9

    def alone(fn):
        deep_gemm.set_num_sms(nsm)
        return timed(fn, args.iters)

    # every mode is compared with the GEMM alone measured right before and right after it (their mean):
    # a baseline taken once per repetition drifted by up to ~9 points within the repetition
    print('rep,dtype,k,mode,num_sms,us_per_gemm,tflops,pct_of_alone', flush=True)
    for rep in range(args.reps):
        for dtype, (fn, flop) in gemms.items():
            base = alone(fn)
            print(f'{rep},{dtype},0,alone,{nsm},{base:.1f},{flop / base / 1e6:.1f},100.0', flush=True)
            for r in (int(x) for x in args.ce_rates.split(',')):
                b0 = alone(fn)
                us, gbps = with_ce(fn, args.iters, r)
                b = (b0 + alone(fn)) / 2
                mode = f'ce{r}' if r else 'cemax'
                print(f'{rep},{dtype},0,{mode},{nsm},{us:.1f},{flop / us / 1e6:.1f},{100 * b / us:.1f}', flush=True)
                print(f'# {mode} {gbps:.1f} GB/s', flush=True)
            for k in ks:
                for mode in ('partitioned', 'held'):
                    b0 = alone(fn)
                    deep_gemm.set_num_sms(nsm - k)
                    us = timed(fn, args.iters) if mode == 'partitioned' else held(fn, k, args.iters)
                    b = (b0 + alone(fn)) / 2
                    print(f'{rep},{dtype},{k},{mode},{nsm - k},{us:.1f},{flop / us / 1e6:.1f},{100 * b / us:.1f}', flush=True)


if __name__ == '__main__':
    main()
