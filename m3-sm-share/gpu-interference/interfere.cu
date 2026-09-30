// GPU interference microbenchmark: how much does a compute workload lose when
// the same bytes are moved by SMs (a kernel holding k SMs) instead of by the
// copy engines?  Single GPU, no network.  See README.md for the methodology.
//
// Modes (run concurrently with a workload that fills the rest of the GPU):
//   none    : workload alone
//   idle    : k CTAs hold k SMs (one CTA per SM, forced by shared memory) and
//             spin on a flag; no memory traffic -> pure SM-occupancy cost
//   smcopy  : the same k CTAs copy src -> dst (d2d: HBM->HBM, d2h: HBM->pinned
//             host over PCIe), optionally paced to a target rate
//   ce      : cudaMemcpyBatchAsync (copy engine, PreferOverlapWithCompute) moves
//             the same bytes, paced by the host to the same target rate
// Workloads: gemm (cuBLAS BF16, DeepSeek-V3 expert up-projection shape) and
// triad (HBM-bound a = b + s*c).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <set>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "%s:%d %s: status %d\n", __FILE__, __LINE__, #x, (int)s_); exit(1); } } while (0)

static const int OCC_THREADS = 1024;
static const int OCC_SMEM = 200 * 1024;          // > half of an SM's 228 KB: one CTA per SM, no GEMM CTA fits beside it
static const size_t REGION = 32ull << 20;         // bytes each copy CTA cycles through (defeats the 60 MB L2)
static const size_t SUBCHUNK = 1ull << 20;        // pacing granularity per CTA
static const size_t CE_CHUNK = 64ull << 20;       // copy-engine request size
static const size_t COPY_BYTES = 1ull << 30;      // src / dst buffers
static const int MAX_CTAS = 160;
static int g_occ_cluster = 2;  // default: take SMs in pairs (kinder to clustered GEMMs, i.e. conservative)  // --occ-cluster: launch occupier CTAs in clusters (take SMs in groups)

struct HostCtl { volatile int started[MAX_CTAS]; };  // pinned+mapped: each CTA writes its slot once
// The stop flag lives in HBM and is set by a copy-engine memcpy: polling host memory from
// SMs (non-posted PCIe reads) perturbs the whole GPU and is not what real comm kernels do.

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

__global__ void __launch_bounds__(OCC_THREADS, 1)
occupier(HostCtl* ctl, const volatile int* stop_flag, int copy, const int4* __restrict__ src, int4* __restrict__ dst,
         unsigned long long ns_per_subchunk, unsigned long long* bytes_out, unsigned* smids) {
  extern __shared__ int4 reserve[];  // never used; only holds the SM
  __shared__ int stop;
  if (threadIdx.x == 0) {
    smids[blockIdx.x] = smid();
    stop = 0;
    ctl->started[blockIdx.x] = 1;
    __threadfence_system();
  }
  __syncthreads();
  if (!copy) {
    if (threadIdx.x == 0) while (!*stop_flag) __nanosleep(1000);
    return;
  }
  const size_t region16 = REGION / 16, sub16 = SUBCHUNK / 16;
  const int4* s = src + blockIdx.x * region16;
  int4* d = dst + blockIdx.x * region16;
  const size_t st = blockDim.x;
  unsigned long long n = 0, t0 = gtimer();
  size_t off = 0;
  while (true) {
    const int4* ss = s + off; int4* dd = d + off;
    size_t i = threadIdx.x;
    for (; i + 3 * st < sub16; i += 4 * st) {
      int4 a = __ldcs(ss + i), b = __ldcs(ss + i + st), c = __ldcs(ss + i + 2 * st), e = __ldcs(ss + i + 3 * st);
      __stcs(dd + i, a); __stcs(dd + i + st, b); __stcs(dd + i + 2 * st, c); __stcs(dd + i + 3 * st, e);
    }
    for (; i < sub16; i += st) __stcs(dd + i, __ldcs(ss + i));
    n++;
    off = (off + sub16) % region16;
    __syncthreads();
    if (threadIdx.x == 0) {
      if (ns_per_subchunk)
        while (gtimer() - t0 < n * ns_per_subchunk && !*stop_flag) __nanosleep(500);
      stop = *stop_flag;
    }
    __syncthreads();
    if (stop) break;
  }
  if (threadIdx.x == 0) bytes_out[blockIdx.x] = n * SUBCHUNK;
}

__global__ void triad(float4* __restrict__ a, const float4* __restrict__ b, const float4* __restrict__ c, size_t n, float s) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 x = b[i], y = c[i];
    a[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
  }
}

struct Ctx {
  int nsm;
  cudaStream_t work, comm;
  cublasHandle_t blas;
  // gemm: C[M,N] = A[M,K] * B[K,N], BF16 in/out, FP32 accumulate
  int M = 8192, N = 4096, K = 7168, gemm_iters = 100;   // up/gate proj: [T,7168]x[7168,2*2048]
  int dN = 7168, dK = 2048, down_iters = 100;             // down proj:    [T,2048]x[2048,7168]
  __nv_bfloat16 *A, *B, *C;
  // triad
  size_t tri_n = (1ull << 30) / sizeof(float4); int tri_iters = 100;
  float4 *ta, *tb, *tc;
  // copies
  char *src, *ddst, *hdst, *hdst_dev;
  HostCtl* ctl; HostCtl* ctl_dev;
  int* stop_dev; int* one_h; int* zero_h; cudaStream_t ctls;
  unsigned long long* bytes_dev; unsigned* smids_dev;
};

enum Mode { NONE, TARGET, IDLE, SMCOPY, CE };
static const char* mode_name[] = {"none", "target", "idle", "smcopy", "ce"};
struct Cfg { std::string workload; Mode mode; bool d2h; int k; double target_GBps; };
struct Res { double work_ms, metric, comm_GBps; int distinct_sms; };

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void run_workload(Ctx& c, const std::string& w) {
  if (w == "gemm" || w == "gemm_down") {
    const float alpha = 1.f, beta = 0.f;
    int n = w == "gemm" ? c.N : c.dN, k = w == "gemm" ? c.K : c.dK;
    for (int i = 0; i < c.gemm_iters; i++)
      CB(cublasGemmEx(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, c.M, k, &alpha, c.B, CUDA_R_16BF, n,
                      c.A, CUDA_R_16BF, k, &beta, c.C, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT));
  } else {
    for (int i = 0; i < c.tri_iters; i++) triad<<<c.nsm * 16, 256, 0, c.work>>>(c.ta, c.tb, c.tc, c.tri_n, 1.5f);
  }
  CK(cudaGetLastError());
}

static double work_metric(Ctx& c, const std::string& w, double ms) {
  if (w == "gemm") return 2.0 * c.M * c.N * c.K * c.gemm_iters / (ms * 1e-3) / 1e12;  // TFLOP/s
  if (w == "gemm_down") return 2.0 * c.M * c.dN * c.dK * c.gemm_iters / (ms * 1e-3) / 1e12;
  return 3.0 * c.tri_n * sizeof(float4) * c.tri_iters / (ms * 1e-3) / 1e9;          // GB/s
}

static void ce_copy(Ctx& c, char* dst, const char* src, size_t n, bool d2h) {
  void* dsts[1] = {dst}; void* srcs[1] = {(void*)src}; size_t sizes[1] = {n};
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.srcLocHint.type = cudaMemLocationTypeDevice; attr.srcLocHint.id = 0;
  attr.dstLocHint.type = d2h ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice; attr.dstLocHint.id = 0;
  attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  size_t idx = 0, fail = 0;
  CK(cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, &idx, 1, &fail, c.comm));
}

static void set_stop(Ctx& c, bool v) {
  CK(cudaMemcpyAsync(c.stop_dev, v ? c.one_h : c.zero_h, sizeof(int), cudaMemcpyHostToDevice, c.ctls));
  CK(cudaStreamSynchronize(c.ctls));
}

static void start_occupier(Ctx& c, int k, bool copy, bool d2h, double target_GBps) {
  set_stop(c, false);
  for (int i = 0; i < MAX_CTAS; i++) c.ctl->started[i] = 0;
  unsigned long long ns = 0;
  if (copy && target_GBps > 0) ns = (unsigned long long)(SUBCHUNK * (double)k / (target_GBps * 1e9) * 1e9);
  cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
  lc.gridDim = dim3(k); lc.blockDim = dim3(OCC_THREADS); lc.dynamicSmemBytes = OCC_SMEM; lc.stream = c.comm;
  la[0].id = cudaLaunchAttributeClusterDimension; la[0].val.clusterDim.x = g_occ_cluster; la[0].val.clusterDim.y = 1; la[0].val.clusterDim.z = 1;
  lc.attrs = la; lc.numAttrs = 1;
  CK(cudaLaunchKernelEx(&lc, occupier, c.ctl_dev, (const volatile int*)c.stop_dev, (int)copy, (const int4*)c.src,
                        (int4*)(d2h ? c.hdst_dev : c.ddst), ns, c.bytes_dev, c.smids_dev));
  double t0 = now_s();
  for (;;) {
    int n = 0;
    for (int i = 0; i < k; i++) n += c.ctl->started[i];
    if (n == k) break;
    if (now_s() - t0 > 5) { fprintf(stderr, "occupier: only %d/%d CTAs resident\n", n, k); exit(1); }
  }
}

static Res run(Ctx& c, const Cfg& g) {
  Res r = {0, 0, 0, 0};
  cudaEvent_t ws, we, cs, ce;
  CK(cudaEventCreate(&ws)); CK(cudaEventCreate(&we)); CK(cudaEventCreate(&cs)); CK(cudaEventCreate(&ce));
  // the GEMM plans for the SMs it actually gets, as DeepGEMM does with num_sms
  CB(cublasSetSmCountTarget(c.blas, (g.mode == TARGET || g.mode == IDLE || g.mode == SMCOPY) ? c.nsm - g.k : 0));

  size_t ce_issued = 0; double ce_t0 = 0; std::vector<cudaEvent_t> inflight;
  auto ce_pump = [&](bool force) {
    // keep at most 4 requests queued; pace issue to target_GBps if set
    for (auto it = inflight.begin(); it != inflight.end();) {
      if (cudaEventQuery(*it) == cudaSuccess) { CK(cudaEventDestroy(*it)); it = inflight.erase(it); } else ++it;
    }
    while (inflight.size() < 4) {
      if (!force && g.target_GBps > 0 && (now_s() - ce_t0) * g.target_GBps * 1e9 < (double)ce_issued) break;
      size_t off = ce_issued % COPY_BYTES;
      ce_copy(c, (g.d2h ? c.hdst : c.ddst) + off, c.src + off, CE_CHUNK, g.d2h);
      cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); CK(cudaEventRecord(e, c.comm));
      inflight.push_back(e);
      ce_issued += CE_CHUNK;
      if (force) break;
    }
  };

  if (g.mode == IDLE || g.mode == SMCOPY) {
    CK(cudaEventRecord(cs, c.comm));
    start_occupier(c, g.k, g.mode == SMCOPY, g.d2h, g.target_GBps);
  } else if (g.mode == CE) {
    CK(cudaEventRecord(cs, c.comm));
    ce_t0 = now_s();
    ce_pump(true);
  }
  volatile bool finished = false;
  std::thread watchdog([&] {
    double t0 = now_s();
    while (!finished) {
      if (now_s() - t0 > 30) { fprintf(stderr, "watchdog: %s/%s k=%d stuck, releasing occupier\n", g.workload.c_str(), mode_name[g.mode], g.k); set_stop(c, true); break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  CK(cudaEventRecord(ws, c.work));
  run_workload(c, g.workload);
  CK(cudaEventRecord(we, c.work));
  while (cudaEventQuery(we) == cudaErrorNotReady) {
    if (g.mode == CE) ce_pump(false); else std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  set_stop(c, true);
  bool has_comm = g.mode == IDLE || g.mode == SMCOPY || g.mode == CE;
  if (has_comm) CK(cudaEventRecord(ce, c.comm));
  CK(cudaStreamSynchronize(c.comm));
  CK(cudaStreamSynchronize(c.work));
  finished = true; watchdog.join();
  for (auto e : inflight) CK(cudaEventDestroy(e));

  float ms; CK(cudaEventElapsedTime(&ms, ws, we));
  r.work_ms = ms; r.metric = work_metric(c, g.workload, ms);
  if (has_comm) {
    float cms; CK(cudaEventElapsedTime(&cms, cs, ce));
    double bytes = 0;
    if (g.mode == CE) bytes = (double)ce_issued;
    else if (g.mode == SMCOPY) {
      std::vector<unsigned long long> b(g.k);
      CK(cudaMemcpy(b.data(), c.bytes_dev, g.k * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      for (auto x : b) bytes += (double)x;
    }
    r.comm_GBps = bytes / (cms * 1e-3) / 1e9;
    if (g.mode != CE) {
      std::vector<unsigned> s(g.k);
      CK(cudaMemcpy(s.data(), c.smids_dev, g.k * sizeof(unsigned), cudaMemcpyDeviceToHost));
      r.distinct_sms = (int)std::set<unsigned>(s.begin(), s.end()).size();
    }
  }
  CK(cudaEventDestroy(ws)); CK(cudaEventDestroy(we)); CK(cudaEventDestroy(cs)); CK(cudaEventDestroy(ce));
  return r;
}

// Does a copy make progress while every SM is held by an occupier?  If yes, it runs on a copy engine.
static void ce_check(Ctx& c, FILE* out) {
  const size_t n = 256ull << 20;
  for (int api = 0; api < 2; api++)
    for (int d2h = 0; d2h < 2; d2h++) {
      start_occupier(c, c.nsm, false, false, 0);
      cudaEvent_t e; CK(cudaEventCreate(&e));
      char* dst = d2h ? c.hdst : c.ddst;
      if (api == 0) CK(cudaMemcpyAsync(dst, c.src, n, d2h ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, c.work));
      else { std::swap(c.comm, c.work); ce_copy(c, dst, c.src, n, d2h); std::swap(c.comm, c.work); }
      CK(cudaEventRecord(e, c.work));
      double t0 = now_s(); bool done = false;
      while (now_s() - t0 < 2.0) if (cudaEventQuery(e) == cudaSuccess) { done = true; break; }
      set_stop(c, true);
      CK(cudaDeviceSynchronize());
      CK(cudaEventDestroy(e));
      fprintf(out, "ce_check,%s,%s,completed_while_all_%d_SMs_held=%d\n", api ? "cudaMemcpyBatchAsync" : "cudaMemcpyAsync",
              d2h ? "d2h" : "d2d", c.nsm, done);
      fflush(out);
    }
}

// Copy-engine throughput vs request size: can a copy engine do per-token scatter?
static void ce_sizes(Ctx& c, FILE* out, int reps) {
  const size_t sizes[] = {512, 1024, 2048, 3584, 7168, 14336, 65536, 262144, 1 << 20, 16 << 20};
  for (int d2h = 0; d2h < 2; d2h++)
    for (size_t s : sizes) {
      size_t count = std::min<size_t>((256ull << 20) / s, 16384);
      std::vector<void*> dsts(count), srcs(count); std::vector<size_t> szs(count, s);
      char* dbase = d2h ? c.hdst : c.ddst;
      for (size_t i = 0; i < count; i++) { srcs[i] = c.src + i * s; dsts[i] = dbase + ((i * 2 * s) % (COPY_BYTES - s)); }
      cudaMemcpyAttributes attr = {};
      attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      attr.srcLocHint.type = cudaMemLocationTypeDevice;
      attr.dstLocHint.type = d2h ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
      attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
      cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      for (int api = 0; api < 2; api++)
        for (int r = 0; r < reps; r++) {
          CK(cudaEventRecord(a, c.comm));
          if (api == 0) {
            size_t idx = 0, fail = 0;
            CK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), szs.data(), count, &attr, &idx, 1, &fail, c.comm));
          } else {
            for (size_t i = 0; i < count; i++)
              CK(cudaMemcpyAsync(dsts[i], srcs[i], s, d2h ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, c.comm));
          }
          CK(cudaEventRecord(b, c.comm));
          CK(cudaEventSynchronize(b));
          float ms; CK(cudaEventElapsedTime(&ms, a, b));
          fprintf(out, "ce_size,%s,%s,%zu,%zu,%d,%.4f,%.2f,%.4f\n", api ? "cudaMemcpyAsync_loop" : "cudaMemcpyBatchAsync",
                  d2h ? "d2h" : "d2d", s, count, r, ms, count * (double)s / (ms * 1e-3) / 1e9, count / (ms * 1e-3) / 1e6);
        }
      fflush(out);
      CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
}

int main(int argc, char** argv) {
  int reps = 5; std::string out_path = "results.csv"; std::vector<int> ks = {4, 8, 16, 20, 32};
  std::vector<double> rates = {50, 100, 0};  // GB/s; 0 = unpaced
  bool quick = false, diag = false, only_none = false;  // --only-none: workloads alone (external traffic runs beside)
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--reps") reps = atoi(argv[++i]);
    else if (a == "--out") out_path = argv[++i];
    else if (a == "--quick") quick = true;
    else if (a == "--occ-cluster") g_occ_cluster = atoi(argv[++i]);
    else if (a == "--diag") { diag = true; }
    else if (a == "--only-none") only_none = true;
  }
  if (quick) { reps = 1; ks = {8, 20}; rates = {50}; }

  Ctx c;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  c.nsm = p.multiProcessorCount;
  CK(cudaFuncSetAttribute(occupier, cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM));
  CK(cudaStreamCreateWithFlags(&c.work, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&c.comm, cudaStreamNonBlocking));
  CB(cublasCreate(&c.blas)); CB(cublasSetStream(c.blas, c.work));
  // fixed workspace: cuBLAS must never cudaMalloc/cudaFree while an occupier holds SMs
  // (cudaFree synchronizes the device and would wait for the occupier forever)
  void* ws; const size_t ws_bytes = 256ull << 20; CK(cudaMalloc(&ws, ws_bytes));
  CB(cublasSetWorkspace(c.blas, ws, ws_bytes));
  CK(cudaMalloc(&c.A, (size_t)c.M * c.K * 2)); CK(cudaMalloc(&c.B, (size_t)c.K * c.dN * 2)); CK(cudaMalloc(&c.C, (size_t)c.M * c.dN * 2));
  CK(cudaMemset(c.A, 0x3c, (size_t)c.M * c.K * 2)); CK(cudaMemset(c.B, 0x3c, (size_t)c.K * c.dN * 2));
  CK(cudaMalloc(&c.ta, c.tri_n * 16)); CK(cudaMalloc(&c.tb, c.tri_n * 16)); CK(cudaMalloc(&c.tc, c.tri_n * 16));
  CK(cudaMemset(c.tb, 0, c.tri_n * 16)); CK(cudaMemset(c.tc, 0, c.tri_n * 16));
  CK(cudaMalloc(&c.src, COPY_BYTES)); CK(cudaMalloc(&c.ddst, COPY_BYTES)); CK(cudaMemset(c.src, 1, COPY_BYTES));
  CK(cudaHostAlloc(&c.hdst, COPY_BYTES, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&c.hdst_dev, c.hdst, 0));
  memset(c.hdst, 0, COPY_BYTES);
  CK(cudaHostAlloc(&c.ctl, sizeof(HostCtl), cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&c.ctl_dev, c.ctl, 0));
  CK(cudaMalloc(&c.stop_dev, sizeof(int))); CK(cudaHostAlloc(&c.one_h, 4, 0)); CK(cudaHostAlloc(&c.zero_h, 4, 0));
  *c.one_h = 1; *c.zero_h = 0; CK(cudaStreamCreateWithFlags(&c.ctls, cudaStreamNonBlocking));
  CK(cudaMalloc(&c.bytes_dev, MAX_CTAS * 8)); CK(cudaMalloc(&c.smids_dev, MAX_CTAS * 4));
  if (*std::max_element(ks.begin(), ks.end()) * REGION > COPY_BYTES) { fprintf(stderr, "k too large for buffers\n"); return 1; }

  if (diag) {
    for (int k : ks) { CB(cublasSetSmCountTarget(c.blas, c.nsm - k)); run_workload(c, "gemm"); }
    CB(cublasSetSmCountTarget(c.blas, 0)); run_workload(c, "gemm"); CK(cudaDeviceSynchronize());
    for (int k : {8, 16, 20, 32}) {
      Res t = run(c, {"gemm", TARGET, false, k, 0}), r = run(c, {"gemm", IDLE, false, k, 0});
      std::vector<unsigned> sm(k); CK(cudaMemcpy(sm.data(), c.smids_dev, k * 4, cudaMemcpyDeviceToHost));
      std::sort(sm.begin(), sm.end());
      printf("cluster=%d k=%d target_only=%.1fms idle=%.1fms smids:", g_occ_cluster, k, t.work_ms, r.work_ms);
      for (auto x : sm) printf(" %u", x);
      printf("\n");
    }
    return 0;
  }
  FILE* out = fopen(out_path.c_str(), "w");
  fprintf(out, "# device=%s sms=%d cc=%d.%d gemm=%dx%dx%d gemm_down=%dx%dx%d bf16 triad_bytes=%zu occ_cluster=%d\n", p.name, c.nsm,
          p.major, p.minor, c.M, c.N, c.K, c.M, c.dN, c.dK, c.tri_n * 16, g_occ_cluster);
  if (!only_none) ce_check(c, out);

  std::vector<Cfg> cfgs;
  for (std::string w : {"gemm", "gemm_down", "triad"}) {
    cfgs.push_back({w, NONE, false, 0, 0});
    if (only_none) continue;
    if (w != "triad") for (int k : ks) cfgs.push_back({w, TARGET, false, k, 0});
    for (int k : ks) cfgs.push_back({w, IDLE, false, k, 0});
    for (int d2h = 0; d2h < 2; d2h++)
      for (double r : rates) {
        if (d2h && r > 60) continue;  // PCIe Gen5 x16 tops out near 55 GB/s
        for (int k : ks) cfgs.push_back({w, SMCOPY, (bool)d2h, k, r});
        cfgs.push_back({w, CE, (bool)d2h, 0, r});
      }
  }
  // warm-up: every SM target the sweep will use, with no occupier running (cuBLAS heuristics, clocks)
  for (int k : ks) { CB(cublasSetSmCountTarget(c.blas, c.nsm - k)); run_workload(c, "gemm"); run_workload(c, "gemm_down"); }
  CB(cublasSetSmCountTarget(c.blas, 0)); run_workload(c, "gemm"); run_workload(c, "gemm_down"); run_workload(c, "triad");
  CK(cudaDeviceSynchronize());
  fprintf(out, "run,rep,workload,mode,dir,k,target_GBps,sm_target,work_ms,metric,unit,comm_GBps,distinct_sms\n");
  for (int rep = 0; rep < reps; rep++)
    for (auto& g : cfgs) {
      Res r = run(c, g);
      fprintf(out, "run,%d,%s,%s,%s,%d,%.0f,%d,%.3f,%.2f,%s,%.2f,%d\n", rep, g.workload.c_str(), mode_name[g.mode],
              g.mode == NONE || g.mode == TARGET || g.mode == IDLE ? "-" : (g.d2h ? "d2h" : "d2d"), g.k, g.target_GBps,
              (g.mode == TARGET || g.mode == IDLE || g.mode == SMCOPY) ? c.nsm - g.k : c.nsm, r.work_ms, r.metric,
              g.workload == "triad" ? "GBps" : "TFLOPs", r.comm_GBps, r.distinct_sms);
      fflush(out);
    }
  if (!only_none) ce_sizes(c, out, quick ? 1 : 3);
  fclose(out);
  printf("wrote %s\n", out_path.c_str());
  return 0;
}
