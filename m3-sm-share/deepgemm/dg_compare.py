#!/usr/bin/env python3
"""cuBLAS vs DeepGEMM next to held SMs, in ONE harness: why did 20 held SMs cost cuBLAS 51 points
(../gpu-interference) and DeepGEMM ~13 (dg_held.py)?

Same process, same holders (k CTAs, 200 KB shared memory, 2-CTA clusters, sleeping; their %smid is
recorded), same timing. GEMMs:
  cublas-bf16    torch.mm, [8192 x 7168] x [7168 x 4096] BF16 (the ../gpu-interference shape), SM count
                 set with cublasSetSmCountTarget on PyTorch's cuBLAS handle (ctypes)
  dg-fp8, dg-bf16  DeepGEMM m-grouped contiguous, 8 experts x ~1024 tokens, 7168 -> 4096
Modes per k: partitioned (GEMM planned for 132 - k, nothing else) and held (planned for 132 - k, k
holders resident). --profile prints each GEMM's kernels (name, grid, block, shared memory) at 132
and 132 - 20 SMs from a PyTorch profiler trace.
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile
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
__global__ void hold(volatile int* stop, int* started, int* smids) {
  extern __shared__ char reserve[];
  if (threadIdx.x == 0) {
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    smids[blockIdx.x] = (int)s;
    atomicAdd(started, 1);
    while (*stop == 0) __nanosleep(2000);
  }
}
void launch_hold(torch::Tensor stop, torch::Tensor started, torch::Tensor smids, int64_t k, int64_t smem) {
  cudaFuncSetAttribute(hold, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)k); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, hold, (volatile int*)stop.data_ptr<int>(), started.data_ptr<int>(),
                                 smids.data_ptr<int>()) == cudaSuccess, "hold launch failed");
}
'''


def cublas_lib():
    import nvidia.cublas
    for d in nvidia.cublas.__path__:
        p = os.path.join(d, 'lib', 'libcublas.so.12')
        if os.path.exists(p):
            return ctypes.CDLL(p)
    raise RuntimeError('libcublas.so.12 not found')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gemms', default='cublas-bf16,dg-fp8,dg-bf16')
    ap.add_argument('--ks', default='8,16,20')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',')]
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    ext = load_inline('loom_hold2', cpp_sources='void launch_hold(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);',
                      cuda_sources=HOLD_SRC, functions=['launch_hold'], extra_cuda_cflags=['-arch=sm_90a'])
    blas = cublas_lib()
    deep_gemm.set_mk_alignment_for_contiguous_layout(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    stop = torch.zeros(1, dtype=torch.int32, device='cuda')
    started = torch.zeros(1, dtype=torch.int32, device='cuda')
    smids = torch.zeros(64, dtype=torch.int32, device='cuda')
    hold_stream, gemm_stream = torch.cuda.Stream(), torch.cuda.Stream()

    def set_sms(name, s):
        if name.startswith('cublas'):
            with torch.cuda.stream(gemm_stream):
                h = ctypes.c_void_p(torch.cuda.current_blas_handle())
            st = blas.cublasSetSmCountTarget(h, ctypes.c_int(0 if s == nsm else s))
            assert st == 0, f'cublasSetSmCountTarget -> {st}'
        else:
            deep_gemm.set_num_sms(s)

    gemms = {}
    for name in args.gemms.split(','):
        torch.manual_seed(0)
        import random; random.seed(0)
        if name == 'cublas-bf16':
            M, K, N = 8192, 7168, 4096
            A = torch.full((M, K), 0.01, dtype=torch.bfloat16, device='cuda')
            B = torch.full((K, N), 0.01, dtype=torch.bfloat16, device='cuda')
            C = torch.empty((M, N), dtype=torch.bfloat16, device='cuda')
            fn = lambda A=A, B=B, C=C: torch.mm(A, B, out=C)
            flop = 2.0 * M * N * K
        else:
            bf16 = name == 'dg-bf16'
            m, a, b, layout, d, ref_d, valid = generate_m_grouped_contiguous(
                8, 1024, 4096, 7168, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=False, use_bf16=bf16,
                quant_config=None if bf16 else QuantConfig())
            if bf16:
                fn = lambda a=a, b=b, d=d, layout=layout: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
            else:
                fn = lambda a=a, b=b, d=d, layout=layout: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a, b, d, layout, disable_ue8m0_cast=True)
            flop = 2.0 * float(valid.sum().item()) * 4096 * 7168
        for s in [nsm] + [nsm - k for k in ks]:
            set_sms(name, s)
            with torch.cuda.stream(gemm_stream):
                fn()
        torch.cuda.synchronize()
        gemms[name] = (fn, flop)
        print(f'# {name}: {flop / 1e9:.1f} GFLOP', flush=True)

    if args.profile:
        for name, (fn, _) in gemms.items():
            for s in (nsm, nsm - 20):
                set_sms(name, s)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.cuda.stream(gemm_stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.synchronize()
                path = tempfile.mktemp(suffix='.json')
                prof.export_chrome_trace(path)
                seen = set()
                for e in json.load(open(path))['traceEvents']:
                    if e.get('cat') == 'kernel':
                        a = e.get('args', {})
                        key = (e['name'], str(a.get('grid')))
                        if key in seen:
                            continue
                        seen.add(key)
                        print(f'# profile {name} sms={s}: grid={a.get("grid")} block={a.get("block")} '
                              f'smem={a.get("shared memory")} regs={a.get("registers per thread")} '
                              f'dur={e.get("dur")}us name={e["name"][:160]}', flush=True)

    def timed(fn, iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(gemm_stream):
            fn()
            a.record()
            for _ in range(iters):
                fn()
            b.record()
        b.synchronize()
        return a.elapsed_time(b) / iters * 1e3

    def held(fn, k, iters):
        stop.zero_(); started.zero_(); smids.fill_(-1); torch.cuda.synchronize()
        with torch.cuda.stream(hold_stream):
            ext.launch_hold(stop, started, smids, k, 200 * 1024)
        t0 = time.time()
        while int(started.item()) < k:
            if time.time() - t0 > 10:
                raise RuntimeError('holders did not start')
            time.sleep(0.001)
        us = timed(fn, iters)
        sm = sorted(smids[:k].tolist())
        stop.fill_(1)
        torch.cuda.synchronize()
        return us, sm

    print('rep,gemm,k,mode,us_per_gemm,tflops,pct_of_alone,distinct_sms,smids', flush=True)
    for rep in range(args.reps):
        for name, (fn, flop) in gemms.items():
            set_sms(name, nsm)
            base = timed(fn, args.iters)
            print(f'{rep},{name},0,alone,{base:.1f},{flop / base / 1e6:.1f},100.0,0,', flush=True)
            for k in ks:
                set_sms(name, nsm - k)
                us = timed(fn, args.iters)
                print(f'{rep},{name},{k},partitioned,{us:.1f},{flop / us / 1e6:.1f},{100 * base / us:.1f},0,', flush=True)
                us, sm = held(fn, k, args.iters)
                print(f'{rep},{name},{k},held,{us:.1f},{flop / us / 1e6:.1f},{100 * base / us:.1f},{len(set(sm))},'
                      f'{" ".join(map(str, sm))}', flush=True)
            set_sms(name, nsm)


if __name__ == '__main__':
    main()
