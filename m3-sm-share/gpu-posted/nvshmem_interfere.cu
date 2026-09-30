// GPU-posted RDMA next to compute (step 2), on steve's CX-7 loopback.
// 2 NVSHMEM PEs on the one H200 (MPG). PE 0 runs a cuBLAS BF16 GEMM and, concurrently,
// a put kernel on k CTAs: one thread per CTA posts nvshmem_putmem_nbi (IBGDA: the SM
// builds the WQE and rings the NIC doorbell) paced to a target rate, with nvshmem_quiet
// every QUIET_EVERY puts. PE 1 is only the remote memory. Each put CTA holds its SM
// exclusively (200 KB smem, 2-CTA clusters), as a DeepEP comm kernel does, and the GEMM
// is planned for 132-k SMs (cublasSetSmCountTarget), exactly as in gpu-interference/.
// Modes: none | target k (GEMM planned for 132-k SMs, nothing co-running) |
//        idle k (CTAs hold SMs, no puts) | put k msg rate.
// Per put CTA: SM cycles spent inside the put call and inside quiet -> ns via the
// kernel's own clock64/globaltimer ratio.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <set>
#include <chrono>
#include <thread>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "%s:%d %s: status %d\n", __FILE__, __LINE__, #x, (int)s_); exit(1); } } while (0)

static const int PUT_THREADS = 256;
static const int PUT_SMEM = 200 * 1024;
static const size_t REGION = 32ull << 20;   // per-CTA source/destination window
static const int MAX_CTAS = 64;
static const int QUIET_EVERY = 16;

struct Stats { unsigned long long n, put_cyc, quiet_cyc, cyc, ns; };

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

__global__ void __launch_bounds__(PUT_THREADS, 1)
putter(const volatile int* stop, volatile int* started, char* dst, const char* src, size_t msg,
       unsigned long long ns_per_msg, Stats* st, unsigned* smids) {
  extern __shared__ char reserve[];  // only holds the SM
  if (threadIdx.x) return;           // one posting thread per CTA (the SM stays held)
  smids[blockIdx.x] = smid();
  started[blockIdx.x] = 1; __threadfence_system();
  Stats s = {0, 0, 0, 0, 0};
  const unsigned long long c0 = clock64(), t0 = gtimer();
  if (msg == 0) {                    // idle hold
    while (!*stop) __nanosleep(1000);
  } else {
    char* d = dst + blockIdx.x * REGION; const char* sr = src + blockIdx.x * REGION;
    size_t off = 0;
    while (!*stop) {
      unsigned long long a = clock64();
      nvshmem_putmem_nbi(d + off, sr + off, msg, 1);
      s.put_cyc += clock64() - a;
      s.n++;
      off += msg; if (off + msg > REGION) off = 0;
      if (s.n % QUIET_EVERY == 0) { a = clock64(); nvshmem_quiet(); s.quiet_cyc += clock64() - a; }
      if (ns_per_msg) while (gtimer() - t0 < s.n * ns_per_msg && !*stop) {}
    }
    unsigned long long a = clock64(); nvshmem_quiet(); s.quiet_cyc += clock64() - a;
  }
  s.cyc = clock64() - c0; s.ns = gtimer() - t0;
  st[blockIdx.x] = s;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

enum Mode { NONE, TARGET, IDLE, PUT };
static const char* mode_name[] = {"none", "target", "idle", "put"};
struct Cfg { std::string w; Mode m; int k; size_t msg; double rate; };

int main(int argc, char** argv) {
  int reps = 3;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
  nvshmem_init();
  int me = nvshmem_my_pe();
  CK(cudaSetDevice(0));
  // symmetric windows (both PEs allocate; PE 1 only receives)
  const int KMAX = 20;
  // PE 1 must not touch the GPU while PE 0 measures: without MPS the two processes time-slice,
  // and nvshmem_barrier_all spins on the GPU. PE 1 waits on the host for this file instead.
  const char* done_file = "/tmp/loom_nvshmem_interfere.done";
  if (me == 0) unlink(done_file);
  char* sym = (char*)nvshmem_malloc(2 * KMAX * REGION);
  if (!sym) { fprintf(stderr, "nvshmem_malloc failed\n"); return 1; }
  char* src = sym; char* dst = sym + KMAX * REGION;
  if (me != 0) {
    while (access(done_file, F_OK) != 0) usleep(100000);
    nvshmem_barrier_all(); nvshmem_free(sym); nvshmem_finalize(); return 0;
  }

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); const int nsm = p.multiProcessorCount;
  cudaStream_t work, comm, ctl;
  CK(cudaStreamCreateWithFlags(&work, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&comm, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&ctl, cudaStreamNonBlocking));
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetStream(h, work));
  void* ws; CK(cudaMalloc(&ws, 256ull << 20)); CB(cublasSetWorkspace(h, ws, 256ull << 20));
  const int M = 8192; __nv_bfloat16 *A, *B, *C;
  CK(cudaMalloc(&A, (size_t)M * 7168 * 2)); CK(cudaMalloc(&B, (size_t)7168 * 7168 * 2)); CK(cudaMalloc(&C, (size_t)M * 7168 * 2));
  CK(cudaMemset(A, 0x3c, (size_t)M * 7168 * 2)); CK(cudaMemset(B, 0x3c, (size_t)7168 * 7168 * 2));
  auto gemm = [&](const std::string& w) {  // gemm: up-proj 8192x4096x7168; gemm_down: 8192x7168x2048
    const float al = 1.f, be = 0.f; int n = w == "gemm" ? 4096 : 7168, k = w == "gemm" ? 7168 : 2048;
    for (int i = 0; i < 100; i++)
      CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, M, k, &al, B, CUDA_R_16BF, n, A, CUDA_R_16BF, k, &be, C,
                      CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  auto flops = [&](const std::string& w) { return w == "gemm" ? 2.0 * M * 4096 * 7168 * 100 : 2.0 * M * 7168 * 2048 * 100; };

  int *stop_d, *one_h, *zero_h; volatile int* started_h; int* started_d;
  CK(cudaMalloc(&stop_d, 4)); CK(cudaHostAlloc(&one_h, 4, 0)); CK(cudaHostAlloc(&zero_h, 4, 0)); *one_h = 1; *zero_h = 0;
  CK(cudaHostAlloc((void**)&started_h, MAX_CTAS * 4, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&started_d, (void*)started_h, 0));
  Stats* st_d; unsigned* smids_d; CK(cudaMalloc(&st_d, MAX_CTAS * sizeof(Stats))); CK(cudaMalloc(&smids_d, MAX_CTAS * 4));
  CK(cudaFuncSetAttribute(putter, cudaFuncAttributeMaxDynamicSharedMemorySize, PUT_SMEM));
  auto set_stop = [&](int v) { CK(cudaMemcpyAsync(stop_d, v ? one_h : zero_h, 4, cudaMemcpyHostToDevice, ctl)); CK(cudaStreamSynchronize(ctl)); };

  std::vector<int> ks = {4, 8, 16, 20};
  std::vector<size_t> msgs = {7168, 65536};
  std::vector<double> rates = {5, 10, 15};   // GB/s total; ~15-20 GB/s is the loopback's GPU->GPU ceiling
  // warm-up every SM target with nothing co-running
  for (int k : ks) { CB(cublasSetSmCountTarget(h, nsm - k)); gemm("gemm"); gemm("gemm_down"); }
  CB(cublasSetSmCountTarget(h, 0)); gemm("gemm"); gemm("gemm_down"); CK(cudaDeviceSynchronize());

  std::vector<Cfg> cfgs;
  for (std::string w : {"gemm", "gemm_down"}) {
    cfgs.push_back({w, NONE, 0, 0, 0});
    for (int k : ks) cfgs.push_back({w, TARGET, k, 0, 0});
    for (int k : ks) cfgs.push_back({w, IDLE, k, 0, 0});
    for (int k : ks) for (size_t m : msgs) for (double r : rates) cfgs.push_back({w, PUT, k, m, r});
  }
  printf("# device=%s sms=%d quiet_every=%d NVSHMEM_IBGDA_NUM_RC_PER_PE=%s\n", p.name, nsm, QUIET_EVERY,
         getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE") ? getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE") : "default");
  printf("run,rep,workload,mode,k,msg,target_GBps,sm_target,work_ms,TFLOPs,comm_GBps,puts,put_ns,quiet_ns_per_put,distinct_sms\n");
  cudaEvent_t ws_e, we_e; CK(cudaEventCreate(&ws_e)); CK(cudaEventCreate(&we_e));
  for (int rep = 0; rep < reps; rep++)
    for (auto& g : cfgs) {
      bool hold = g.m == IDLE || g.m == PUT;
      CB(cublasSetSmCountTarget(h, g.m == NONE ? 0 : nsm - g.k));
      if (hold) {
        set_stop(0);
        for (int i = 0; i < MAX_CTAS; i++) started_h[i] = 0;
        unsigned long long ns = g.m == PUT ? (unsigned long long)(g.msg * (double)g.k / (g.rate * 1e9) * 1e9) : 0;
        cudaLaunchConfig_t lc = {}; cudaLaunchAttribute la[1];
        lc.gridDim = dim3(g.k); lc.blockDim = dim3(PUT_THREADS); lc.dynamicSmemBytes = PUT_SMEM; lc.stream = comm;
        la[0].id = cudaLaunchAttributeClusterDimension; la[0].val.clusterDim.x = 2; la[0].val.clusterDim.y = 1; la[0].val.clusterDim.z = 1;
        lc.attrs = la; lc.numAttrs = 1;
        CK(cudaLaunchKernelEx(&lc, putter, (const volatile int*)stop_d, (volatile int*)started_d, dst, (const char*)src,
                              g.m == PUT ? g.msg : (size_t)0, ns, st_d, smids_d));
        double t0 = now_s(); int n = 0;
        while ((n = [&] { int c = 0; for (int i = 0; i < g.k; i++) c += started_h[i]; return c; }()) < g.k)
          if (now_s() - t0 > 5) { fprintf(stderr, "only %d/%d put CTAs resident\n", n, g.k); return 1; }
      }
      CK(cudaEventRecord(ws_e, work)); gemm(g.w); CK(cudaEventRecord(we_e, work));
      CK(cudaEventSynchronize(we_e));
      double comm_gbps = 0, put_ns = 0, quiet_ns = 0; unsigned long long puts = 0; int distinct = 0;
      if (hold) {
        set_stop(1); CK(cudaStreamSynchronize(comm));
        std::vector<Stats> s(g.k); std::vector<unsigned> sm(g.k);
        CK(cudaMemcpy(s.data(), st_d, g.k * sizeof(Stats), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sm.data(), smids_d, g.k * 4, cudaMemcpyDeviceToHost));
        distinct = (int)std::set<unsigned>(sm.begin(), sm.end()).size();
        double bytes = 0, pc = 0, qc = 0, cyc_per_ns = 0, secs = 0;
        for (auto& x : s) { puts += x.n; bytes += x.n * (double)g.msg; pc += x.put_cyc; qc += x.quiet_cyc;
                            cyc_per_ns += (double)x.cyc / x.ns / g.k; secs = std::max(secs, x.ns * 1e-9); }
        if (g.m == PUT && puts) { comm_gbps = bytes / secs / 1e9; put_ns = pc / puts / cyc_per_ns; quiet_ns = qc / puts / cyc_per_ns; }
      }
      float ms; CK(cudaEventElapsedTime(&ms, ws_e, we_e));
      printf("run,%d,%s,%s,%d,%zu,%.0f,%d,%.3f,%.2f,%.2f,%llu,%.1f,%.1f,%d\n", rep, g.w.c_str(), mode_name[g.m], g.k, g.msg,
             g.rate, g.m == NONE ? nsm : nsm - g.k, ms, flops(g.w) / (ms * 1e-3) / 1e12, comm_gbps, puts, put_ns, quiet_ns, distinct);
      fflush(stdout);
    }
  { FILE* f = fopen(done_file, "w"); if (f) fclose(f); }
  nvshmem_barrier_all();
  nvshmem_free(sym);
  nvshmem_finalize();
  return 0;
}
