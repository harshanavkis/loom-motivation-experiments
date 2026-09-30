// Copy-engine initiation latency on one GPU. A copy-engine transfer can only be
// started from the host (a kernel cannot launch one), so its initiation path is:
// host API call -> driver writes a command into the channel's push buffer ->
// doorbell -> the copy engine fetches the command -> copies.
// Measured (median / p99 over --iters, spin-wait scheduling, host steady_clock):
//   api        : time spent inside the API call only (cudaMemcpyAsync / cudaMemcpyBatchAsync)
//   visible    : D2H only; issue -> the last 8 B of the payload is visible in pinned host
//                memory (host spins on it). No stream synchronisation in the path.
//   sync       : issue -> cudaStreamSynchronize returns
//   kernel     : reference: empty kernel launch -> cudaStreamSynchronize returns
// --load runs a BF16 GEMM continuously on another stream of the same process.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <thread>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "%s:%d %s: status %d\n", __FILE__, __LINE__, #x, (int)s_); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
__global__ void empty_kernel() {}

struct Stat { double med, p99; };
static Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v[(size_t)(v.size() * 0.99)]};
}

static void copy(bool batch, void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t s) {
  if (!batch) { CK(cudaMemcpyAsync(dst, src, n, kind, s)); return; }
  void* d[1] = {dst}; void* sr[1] = {(void*)src}; size_t sz[1] = {n};
  cudaMemcpyAttributes a = {};
  a.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  a.srcLocHint.type = (kind == cudaMemcpyHostToDevice) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  a.dstLocHint.type = (kind == cudaMemcpyDeviceToHost) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  a.flags = cudaMemcpyFlagPreferOverlapWithCompute;   // forces a copy engine (no SM copy kernel)
  size_t idx = 0, fail = 0;
  CK(cudaMemcpyBatchAsync(d, sr, sz, 1, &a, &idx, 1, &fail, s));
}

int main(int argc, char** argv) {
  int iters = 20000; bool load = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--load")) load = true;
  }
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const size_t maxn = 1 << 16;
  char *dsrc, *ddst, *hsrc, *hdst;
  CK(cudaMalloc(&dsrc, 2 * maxn)); CK(cudaMalloc(&ddst, maxn));
  CK(cudaHostAlloc(&hsrc, maxn, 0)); CK(cudaHostAlloc(&hdst, maxn, 0));
  // two device sources whose last 8 B differ, so each D2H copy changes what the host sees
  std::vector<char> pat(maxn, 0);
  CK(cudaMemcpy(dsrc, pat.data(), maxn, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dsrc + maxn, pat.data(), maxn, cudaMemcpyHostToDevice));
  cudaStream_t s, ls; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ls, cudaStreamNonBlocking));

  // optional background GEMM on another stream (same process, so it really shares the GPU)
  std::atomic<bool> stop{false}; std::thread loader;
  cublasHandle_t h; __nv_bfloat16 *A, *B, *C; const int M = 8192, N = 4096, K = 7168;
  if (load) {
    CB(cublasCreate(&h)); CB(cublasSetStream(h, ls));
    CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&B, (size_t)K * N * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
    CK(cudaMemset(A, 0x3c, (size_t)M * K * 2)); CK(cudaMemset(B, 0x3c, (size_t)K * N * 2));
    loader = std::thread([&] {
      const float al = 1.f, be = 0.f;
      while (!stop) {
        for (int i = 0; i < 20; i++)
          CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C,
                          CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CK(cudaStreamSynchronize(ls));
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  printf("# device=%s load=%d iters=%d (latencies in us, median / p99)\n", p.name, load, iters);
  printf("test,api,dir,size,median_us,p99_us\n");
  // reference: kernel launch + sync
  {
    std::vector<double> v;
    for (int i = 0; i < iters; i++) { auto t0 = clk::now(); empty_kernel<<<1, 1, 0, s>>>(); CK(cudaStreamSynchronize(s)); v.push_back(us(t0, clk::now())); }
    Stat st = stat(v); printf("kernel,launch,-,0,%.2f,%.2f\n", st.med, st.p99);
  }
  const size_t sizes[] = {8, 64, 512, 4096, 65536};
  for (int batch = 0; batch < 2; batch++) {
    const char* api = batch ? "cudaMemcpyBatchAsync" : "cudaMemcpyAsync";
    for (size_t n : sizes) {
      // D2H: issue -> payload visible in pinned host memory (spin on the last 8 bytes)
      std::vector<double> va, vv;
      volatile unsigned long long* tail = (volatile unsigned long long*)(hdst + n - 8);
      for (int i = 0; i < iters; i++) {
        unsigned long long want = (unsigned long long)(i + 1);
        // stage the expected tail value into the device source first (not timed)
        char* src = dsrc + (i & 1) * maxn;
        CK(cudaMemcpyAsync(src + n - 8, &want, 8, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s));
        auto t0 = clk::now();
        copy(batch, hdst, src, n, cudaMemcpyDeviceToHost, s);
        auto t1 = clk::now();
        while (*tail != want) {}
        auto t2 = clk::now();
        va.push_back(us(t0, t1)); vv.push_back(us(t0, t2));
        CK(cudaStreamSynchronize(s));
      }
      Stat a = stat(va), v = stat(vv);
      printf("api,%s,d2h,%zu,%.2f,%.2f\n", api, n, a.med, a.p99);
      printf("visible,%s,d2h,%zu,%.2f,%.2f\n", api, n, v.med, v.p99);
      // issue -> cudaStreamSynchronize, all three directions
      struct { const char* name; void* d; const void* sr; cudaMemcpyKind k; } dirs[] = {
          {"d2d", ddst, dsrc, cudaMemcpyDeviceToDevice}, {"d2h", hdst, dsrc, cudaMemcpyDeviceToHost},
          {"h2d", ddst, hsrc, cudaMemcpyHostToDevice}};
      for (auto& d : dirs) {
        std::vector<double> vs;
        for (int i = 0; i < iters; i++) {
          auto t0 = clk::now(); copy(batch, d.d, d.sr, n, d.k, s); CK(cudaStreamSynchronize(s)); vs.push_back(us(t0, clk::now()));
        }
        Stat st = stat(vs); printf("sync,%s,%s,%zu,%.2f,%.2f\n", api, d.name, n, st.med, st.p99);
      }
      fflush(stdout);
    }
  }
  if (load) { stop = true; loader.join(); }
  return 0;
}
