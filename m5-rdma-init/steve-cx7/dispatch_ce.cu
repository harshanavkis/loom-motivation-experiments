// MoE dispatch, GPU-triggered copy engine: the unified-contract bound for
// dispatch_proxy.cu (B2) and ../gpu-posted/dispatch_ibgda.cu (B1), with the same routing,
// token size, grid and GEMM load. The kernel starts the transfer the way it starts a local
// one: its last CTA writes a flag in HBM that releases copies pre-enqueued on a copy-engine
// stream (cuStreamWaitValue32 -> cudaMemcpyBatchAsync, PreferOverlapWithCompute ->
// cuStreamWriteValue32 done), then spins on done. The destination is a receive buffer in
// the same GPU's HBM (a local peer). Modes as in dispatch_proxy: "block" = the kernel packs
// tokens per destination (SM copies) and the engine runs one copy per destination; "token" =
// the kernel only routes and the engine runs one copy per token message, straight from the
// token buffer. The copy sizes and slots are computed on the host from the same
// deterministic routing: they stand in for an engine that takes them from the kernel's
// trigger. Timed with CUDA events around the kernel: routing -> all bytes delivered.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <unistd.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m; cuGetErrorString(r_, &m); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m); exit(1); } } while (0)

static const int R = 8, HMAX = 7168, E = 256, TOPK = 8, TMAX = 4096;

__host__ __device__ __forceinline__ unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__host__ __device__ unsigned rank_mask(int t, int E_, int R_, int topk) {
  unsigned mask = 0; int got = 0; unsigned long long used[4] = {0, 0, 0, 0};
  for (int j = 0; got < topk; j++) {
    int e = hash(t * 131 + j) % E_;
    if (used[e >> 6] >> (e & 63) & 1) continue;
    used[e >> 6] |= 1ull << (e & 63); got++;
    mask |= 1u << (e / (E_ / R_));
  }
  return mask;
}

// block mode: route + pack per destination, as dispatch_pack in dispatch_proxy.cu
// token mode (send == nullptr): route only; masks are stored so the routing is not elided
__global__ void dispatch_trigger(const int4* tok, int4* send, unsigned* masks, int T, int H, int* counters,
                                 int* blocks_done, volatile unsigned* flag, volatile unsigned* done, unsigned iter) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5, h16 = H / 16;
  for (int t = warp; t < T; t += nwarps) {
    unsigned mask = 0;
    if (lane == 0) mask = rank_mask(t, E, R, TOPK);
    mask = __shfl_sync(0xffffffff, mask, 0);
    if (!send) { if (lane == 0) masks[t] = mask; continue; }
    while (mask) {
      int r = __ffs(mask) - 1; mask &= mask - 1;
      int slot = 0;
      if (lane == 0) slot = atomicAdd(&counters[r], 1);
      slot = __shfl_sync(0xffffffff, slot, 0);
      int4* d = send + ((size_t)r * T + slot) * h16; const int4* s = tok + (size_t)t * h16;
      for (int i = lane; i < h16; i += 32) d[i] = s[i];
    }
  }
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(blocks_done, 1) == gridDim.x - 1) {  // last CTA starts the engine
    __threadfence_system();
    *flag = iter; __threadfence_system();
    while (*done != iter) {}
  }
}

struct Load {  // background GEMM on another stream (same process), planned for 132 - reserve SMs
  std::atomic<bool> stop{false}, ready{false}; std::atomic<int> reserve{0}; std::thread th;
  void start() {
    th = std::thread([this] {
      cublasHandle_t h; cublasCreate(&h); cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); cublasSetStream(h, s);
      void* ws; cudaMalloc(&ws, 256ull << 20); cublasSetWorkspace(h, ws, 256ull << 20);
      const int M = 8192, N = 4096, K = 7168; __nv_bfloat16 *A, *B, *C;
      cudaMalloc(&A, (size_t)M * K * 2); cudaMalloc(&B, (size_t)K * N * 2); cudaMalloc(&C, (size_t)M * N * 2);
      cudaMemset(A, 0x3c, (size_t)M * K * 2); cudaMemset(B, 0x3c, (size_t)K * N * 2);
      const float al = 1.f, be = 0.f;
      // run every SM target once first: a GEMM kernel cuBLAS first uses mid-sweep is loaded
      // then, and that load waits for an idle device while the copy stream waits on the kernel
      for (int r : {8, 20}) {
        cublasSetSmCountTarget(h, 132 - r);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }
      cudaStreamSynchronize(s); ready = true;
      while (!stop) {
        cublasSetSmCountTarget(h, 132 - reserve);
        for (int i = 0; i < 10; i++)
          cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaStreamSynchronize(s);
      }
    });
  }
};

int main(int argc, char** argv) {
  // kernels load lazily at first launch, and that load waits for the device: it would
  // deadlock against the pre-enqueued stream wait (see ce_triggered.cu)
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  const int iters = 20; int H = 7168; bool load = false;
  for (int i = 1; i < argc; i++) { if (!strcmp(argv[i], "--H")) H = atoi(argv[++i]); else if (!strcmp(argv[i], "--load")) load = true; }
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  char *tok, *send, *recv; size_t blk = (size_t)R * TMAX * HMAX;
  CK(cudaMalloc(&tok, (size_t)TMAX * HMAX)); CK(cudaMalloc(&send, blk)); CK(cudaMalloc(&recv, blk));
  CK(cudaMemset(tok, 1, (size_t)TMAX * HMAX));
  int *counters, *blocks_done; unsigned *masks, *flag, *done;
  CK(cudaMalloc(&counters, R * 4)); CK(cudaMalloc(&blocks_done, 4)); CK(cudaMalloc(&masks, TMAX * 4));
  CK(cudaMalloc(&flag, 4)); CK(cudaMalloc(&done, 4)); CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(done, 0, 4));
  cudaStream_t ks, ce; CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ce, cudaStreamNonBlocking));
  cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.srcLocHint.type = cudaMemLocationTypeDevice; attr.dstLocHint.type = cudaMemLocationTypeDevice;
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;   // copy engine, not an SM kernel
  Load L; if (load) { L.start(); while (!L.ready) usleep(1000); usleep(500000); }
  printf("# GPU-triggered copy-engine dispatch (local HBM peer), H=%d E=%d R=%d topk=%d load=%d\n", H, E, R, TOPK, (int)load);
  printf("test,H,load,mode,tokens,ctas,messages,median_us,p10_us,p90_us,GBps\n");
  unsigned iter = 0;
  for (int per_token = 0; per_token < 2; per_token++)
    for (int T : {16, 32, 128, 1024, 4096}) {
      // the copies the engine runs, from the same routing the kernel computes
      std::vector<void*> ds, ss; std::vector<size_t> sz; int count[R] = {0}, msgs = 0;
      for (int t = 0; t < T; t++)
        for (unsigned m = rank_mask(t, E, R, TOPK); m; m &= m - 1) {
          int r = __builtin_ctz(m);
          if (per_token) {
            ds.push_back(recv + ((size_t)r * T + count[r]) * H); ss.push_back(tok + (size_t)t * H); sz.push_back(H);
          }
          count[r]++; msgs++;
        }
      if (!per_token)
        for (int r = 0; r < R; r++)
          if (count[r]) { size_t off = (size_t)r * T * H; ds.push_back(recv + off); ss.push_back(send + off); sz.push_back((size_t)count[r] * H); }
      for (int ctas : {8, 20}) {
        L.reserve = ctas;
        std::vector<float> v;
        for (int it = 0; it < iters + 3; it++) {
          iter++;
          CK(cudaMemset(counters, 0, R * 4)); CK(cudaMemset(blocks_done, 0, 4)); CK(cudaDeviceSynchronize());
          CU(cuStreamWaitValue32((CUstream)ce, (CUdeviceptr)flag, iter, CU_STREAM_WAIT_VALUE_GEQ));
          size_t aidx = 0, fail = 0;
          CK(cudaMemcpyBatchAsync(ds.data(), ss.data(), sz.data(), ds.size(), &attr, &aidx, 1, &fail, ce));
          CU(cuStreamWriteValue32((CUstream)ce, (CUdeviceptr)done, iter, CU_STREAM_WRITE_VALUE_DEFAULT));
          CK(cudaEventRecord(ea, ks));
          dispatch_trigger<<<ctas, 256, 0, ks>>>((const int4*)tok, per_token ? nullptr : (int4*)send, masks, T, H,
                                                 counters, blocks_done, flag, done, iter);
          CK(cudaEventRecord(eb, ks)); CK(cudaEventSynchronize(eb)); CK(cudaGetLastError());
          CK(cudaStreamSynchronize(ce));
          float ms; CK(cudaEventElapsedTime(&ms, ea, eb)); if (it >= 3) v.push_back(ms * 1e3f);
        }
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        printf("ce,%d,%d,%s,%d,%d,%d,%.1f,%.1f,%.1f,%.2f\n", H, (int)load, per_token ? "token" : "block", T, ctas, msgs, med,
               v[v.size() / 10], v[v.size() * 9 / 10], (double)msgs * H / (med * 1e-6) / 1e9);
        fflush(stdout);
      }
    }
  if (load) { L.stop = true; L.th.join(); }
  return 0;
}
