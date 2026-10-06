// Can a KERNEL start a copy-engine transfer? We start a 256 MiB device-to-device copy while
// every SM is FULL (no thread slots left: two 1024-thread CTAs per SM, 2 x 1024 = the H200's
// 2048 threads per SM, all spinning on an HBM stop flag; one of those CTAs is the parent that
// starts the copy). A copy that completes then cannot have run on SMs, so it ran on a copy
// engine. The same copies with the SMs free give the reference times (an SM copy kernel moves
// 256 MiB in ~0.2 ms; a copy engine needs ~3 ms at ~86 GB/s).
//   dev_memcpy : device-side cudaMemcpyAsync (CUDA dynamic parallelism, CDP2), fire-and-forget
//   dev_graph  : device graph launch of a graph holding one memcpy node (instantiated on the
//                host with cudaGraphInstantiateFlagDeviceLaunch; addresses and size fixed there).
//                A device graph launch is only allowed from a kernel that itself runs in a graph.
//   host_batch : control, host cudaMemcpyBatchAsync + PreferOverlapWithCompute
//   host_plain : control, host cudaMemcpyAsync D2D
// Completion = the HOST checks all 256 MiB (via a D2H copy, copy engine) while the SMs are still
// held. Output: mode,sms,complete,parent_saw_tail_us,device_call_err.
// An earlier version held SMs with 32-thread CTAs (shared memory only): a copy kernel can still
// fit beside those, so it wrongly looked like the device-side copies ran on a copy engine.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t N = 256ull << 20;
static const int THREADS = 1024, SMEM = 100 * 1024;   // two CTAs fill an SM's 2048 thread slots

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void __launch_bounds__(THREADS, 1) occupier(volatile int* started, const volatile int* stop) {
  extern __shared__ int hold[];
  if (threadIdx.x == 0) { started[blockIdx.x] = 1; __threadfence_system(); }
  while (!*stop) __nanosleep(2000);   // every warp stays resident
}

// result[0] = microseconds until the parent saw the copy's last 8 B (-1: not within 1 s),
// result[1] = the device call's cudaError_t
__device__ void hold_and_watch(volatile unsigned long long* tail, unsigned long long expect, volatile long long* result,
                               const volatile int* stop) {
  if (threadIdx.x == 0) {
    unsigned long long t0 = gtimer(), t; result[0] = -1;
    while ((t = gtimer()) - t0 < 1000000000ull) if (*tail == expect) { result[0] = (long long)((t - t0) / 1000); break; }
  }
  while (!*stop) __nanosleep(2000);   // keep this SM full until the host has checked the copy
}
__global__ void __launch_bounds__(THREADS, 1) parent_memcpy(char* dst, const char* src, volatile unsigned long long* tail,
    unsigned long long expect, volatile long long* result, const volatile int* stop) {
  extern __shared__ int hold[];
  if (threadIdx.x == 0) result[1] = cudaMemcpyAsync(dst, src, N, cudaMemcpyDeviceToDevice, cudaStreamFireAndForget);
  hold_and_watch(tail, expect, result, stop);
}
__global__ void __launch_bounds__(THREADS, 1) parent_graph(cudaGraphExec_t g, volatile unsigned long long* tail,
    unsigned long long expect, volatile long long* result, const volatile int* stop) {
  extern __shared__ int hold[];
  if (threadIdx.x == 0) result[1] = cudaGraphLaunch(g, cudaStreamGraphFireAndForget);
  hold_and_watch(tail, expect, result, stop);
}

int main() {
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  for (auto f : {(const void*)occupier, (const void*)parent_memcpy, (const void*)parent_graph})
    CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  char *src, *dst; CK(cudaMalloc(&src, N)); CK(cudaMalloc(&dst, N));
  std::vector<unsigned char> h(N);
  int *started, *stop; long long* result;
  const int nocc_max = 2 * nsm;
  CK(cudaHostAlloc((void**)&started, nocc_max * 4, cudaHostAllocMapped)); CK(cudaMalloc(&stop, 4));
  CK(cudaHostAlloc((void**)&result, 16, cudaHostAllocMapped));
  int* started_d; long long* result_d;
  CK(cudaHostGetDevicePointer((void**)&started_d, started, 0)); CK(cudaHostGetDevicePointer((void**)&result_d, result, 0));
  cudaStream_t so, sp, sc, sh; for (auto s : {&so, &sp, &sc, &sh}) CK(cudaStreamCreateWithFlags(s, cudaStreamNonBlocking));
  cudaGraph_t graph; cudaGraphExec_t gexec; cudaGraphNode_t node;
  CK(cudaGraphCreate(&graph, 0));
  CK(cudaGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, N, cudaMemcpyDeviceToDevice));
  CK(cudaGraphInstantiate(&gexec, graph, cudaGraphInstantiateFlagDeviceLaunch));
  CK(cudaGraphUpload(gexec, sp));
  // the parent of the graph mode runs inside a host graph
  cudaGraphExec_t pge; {
    cudaGraph_t pg;
    CK(cudaStreamBeginCapture(sp, cudaStreamCaptureModeRelaxed));
    parent_graph<<<1, THREADS, SMEM, sp>>>(gexec, (unsigned long long*)(dst + N - 8), 0x5a5a5a5a5a5a5a5aull, result_d, stop);
    CK(cudaStreamEndCapture(sp, &pg)); CK(cudaGraphInstantiate(&pge, pg, 0));
  }
  CK(cudaDeviceSynchronize());
  printf("# 256 MiB D2D copy; held = every SM full (%d CTAs x %d threads); free = no other work\n", nocc_max, THREADS);
  printf("mode,sms,complete,parent_saw_tail_us,device_call_err\n");
  const char* names[] = {"dev_memcpy", "dev_graph", "host_batch", "host_plain"};
  for (int held = 1; held >= 0; held--)
    for (int mode = 0; mode < 4; mode++) {
      CK(cudaMemset(src, 0x5a, N)); CK(cudaMemset(dst, 0, N)); CK(cudaMemset(stop, 0, 4));
      memset((void*)started, 0, nocc_max * 4); result[0] = -2; result[1] = 0;
      CK(cudaDeviceSynchronize());
      const bool dev = mode < 2;
      const int nocc = held ? (dev ? nocc_max - 1 : nocc_max) : 0;
      if (nocc) occupier<<<nocc, THREADS, SMEM, so>>>(started_d, stop);
      for (bool all = false; !all; ) { all = true; for (int i = 0; i < nocc; i++) if (!((volatile int*)started)[i]) all = false; }
      if (mode == 0) parent_memcpy<<<1, THREADS, SMEM, sp>>>(dst, src, (unsigned long long*)(dst + N - 8), 0x5a5a5a5a5a5a5a5aull, result_d, stop);
      if (mode == 1) CK(cudaGraphLaunch(pge, sp));
      if (mode == 2) {
        cudaMemcpyAttributes a = {}; a.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        a.srcLocHint.type = cudaMemLocationTypeDevice; a.dstLocHint.type = cudaMemLocationTypeDevice;
        a.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        void* ds[1] = {dst}; void* ss[1] = {src}; size_t sz[1] = {N}; size_t idx = 0, fail = 0;
        CK(cudaMemcpyBatchAsync(ds, ss, sz, 1, &a, &idx, 1, &fail, sc));
      }
      if (mode == 3) CK(cudaMemcpyAsync(dst, src, N, cudaMemcpyDeviceToDevice, sc));
      CK(cudaGetLastError());
      usleep(300000);   // 0.3 s: ~100x what either engine needs
      // check all of dst from the host while the SMs are still held (D2H runs on a copy engine)
      CK(cudaMemcpyAsync(h.data(), dst, N, cudaMemcpyDeviceToHost, sh)); CK(cudaStreamSynchronize(sh));
      size_t ok = 0; for (size_t i = 0; i < N; i++) ok += h[i] == 0x5a;
      int one = 1; CK(cudaMemcpyAsync(stop, &one, 4, cudaMemcpyHostToDevice, sh));
      CK(cudaDeviceSynchronize());
      printf("%s,%s,%.3f,%lld,%lld\n", names[mode], held ? "held" : "free", (double)ok / N, dev ? result[0] : -1, result[1]);
      fflush(stdout);
    }
  return 0;
}
