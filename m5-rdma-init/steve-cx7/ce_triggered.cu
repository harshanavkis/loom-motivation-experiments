// Kernel-triggered copy-engine latency. The host pre-enqueues on stream `ce`:
//   for i in 1..N: cuStreamWaitValue32(flag >= i) ; copy(dst <- src, n bytes)
// A 1-thread kernel on another stream then, per iteration: writes value i into the last
// 8 B of src, stamps %globaltimer, sets flag = i, and spins until dst's last 8 B == i.
// So one GPU clock times: trigger written -> the GPU front end sees the semaphore ->
// copy engine runs -> data lands (dst in pinned host memory: read back over PCIe; or HBM).
// This is the GPU-triggered counterpart of an IBGDA put (kernel posts, polls completion).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m; cuGetErrorString(r_, &m); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void trigger(volatile unsigned* flag, volatile unsigned long long* src_tail,
                        volatile unsigned long long* dst_tail, int first, int iters, unsigned long long* ns) {
  for (int i = first; i < first + iters; i++) {
    *src_tail = i;
    __threadfence_system();
    unsigned long long t0 = gtimer();
    *flag = i;
    __threadfence_system();
    unsigned long long spins = 0;
    while (*dst_tail != (unsigned long long)i && ++spins < (1ull << 26)) {}
    ns[i - first] = (spins >= (1ull << 26)) ? ~0ull : gtimer() - t0;   // ~0 = copy never landed
    if (spins >= (1ull << 26)) return;
  }
}

int main(int argc, char** argv) {
  // CUDA 12 loads kernels lazily at their first launch, and that load synchronises with the
  // device, which deadlocks against the pre-enqueued stream waits. Load modules eagerly.
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  // BATCH wait+copy pairs are pre-enqueued per kernel launch: enqueuing all of them before
  // the trigger kernel runs deadlocks once the stream's command queue is full.
  const int BATCH = 200, LAUNCHES = 10, iters = BATCH * LAUNCHES;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const size_t sizes[] = {8, 512, 4096, 65536, 1 << 20, 4 << 20};
  const size_t maxn = 4 << 20;
  char *src, *ddst, *hdst; unsigned* flag; unsigned long long* ns_d;
  CK(cudaMalloc(&src, maxn)); CK(cudaMalloc(&ddst, maxn)); CK(cudaMalloc(&flag, 4)); CK(cudaMalloc(&ns_d, BATCH * 8));
  CK(cudaHostAlloc(&hdst, maxn, cudaHostAllocMapped));
  char* hdst_dev; CK(cudaHostGetDevicePointer((void**)&hdst_dev, hdst, 0));
  cudaStream_t ce, ks; CK(cudaStreamCreateWithFlags(&ce, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
  printf("# device=%s iters=%d  (trigger->data-landed, GPU globaltimer, us)\n", p.name, iters);
  printf("test,dir,size,median_us,p99_us\n");
  for (int d2h = 0; d2h < 2; d2h++)
    for (size_t n : sizes) {
      char* dst = d2h ? hdst : ddst; char* dst_k = d2h ? hdst_dev : ddst;
      CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(src, 0, maxn)); CK(cudaMemset(ddst, 0, maxn)); memset(hdst, 0, maxn);
      CK(cudaDeviceSynchronize());
      std::vector<unsigned long long> v;
      for (int L = 0; L < LAUNCHES; L++) {
      const int first = 1 + L * BATCH;
      for (int i = first; i < first + BATCH; i++) {
        CU(cuStreamWaitValue32((CUstream)ce, (CUdeviceptr)flag, (cuuint32_t)i, CU_STREAM_WAIT_VALUE_GEQ));
        void* ds[1] = {dst}; void* ss[1] = {src}; size_t sz[1] = {n};
        cudaMemcpyAttributes a = {};
        a.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        a.srcLocHint.type = cudaMemLocationTypeDevice;
        a.dstLocHint.type = d2h ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
        a.flags = cudaMemcpyFlagPreferOverlapWithCompute;   // copy engine, not an SM kernel
        size_t idx = 0, fail = 0;
        CK(cudaMemcpyBatchAsync(ds, ss, sz, 1, &a, &idx, 1, &fail, ce));
      }
      trigger<<<1, 1, 0, ks>>>(flag, (unsigned long long*)(src + n - 8), (unsigned long long*)(dst_k + n - 8), first, BATCH, ns_d);
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(ks)); CK(cudaStreamSynchronize(ce));
      std::vector<unsigned long long> b(BATCH); CK(cudaMemcpy(b.data(), ns_d, BATCH * 8, cudaMemcpyDeviceToHost));
      if (b[0] == ~0ull || b[BATCH - 1] == ~0ull) { fprintf(stderr, "size %zu %s launch %d: copy never landed (first=%llu last=%llu)\n", n, d2h ? "d2h" : "d2d", L, b[0], b[BATCH-1]); exit(2); }
      v.insert(v.end(), b.begin() + 10, b.end());   // drop the first 10 of each launch
      }
      std::sort(v.begin(), v.end());
      size_t m = v.size() / 2, q = (size_t)(v.size() * 0.99);
      printf("triggered,%s,%zu,%.2f,%.2f\n", d2h ? "d2h" : "d2d", n, v[m] / 1e3, v[q] / 1e3);
      fflush(stdout);
    }
  return 0;
}
