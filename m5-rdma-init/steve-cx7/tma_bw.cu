// Copy bandwidth vs the number of SMs, the way EP libraries move tokens: each of k CTAs (one
// per SM) copies its own 32 MiB region, GPU memory to GPU memory on one H200.
//   tma   : one thread issues TMA bulk copies (cp.async.bulk) through shared memory, as DeepEP /
//           NCCL EP do: load a 32 KiB chunk into one of 4 stages, store it out, keep loads ahead
//   store : 1024 threads copy with 16 B loads and stores (DeepEP V1's warp copies)
// Local HBM -> HBM is an upper bound for a peer over NVLink. Output: mode,ctas,GBps (copied).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t REGION = 32ull << 20;
static const unsigned CHUNK = 32 * 1024;
static const int STAGES = 4;

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(b)));
}
__device__ __forceinline__ void tma_load(void* dst_smem, const void* src, unsigned bytes, unsigned long long* b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst_smem)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned phase) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
               :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_store(void* dst, const void* src_smem, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(smem_u32(src_smem)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__global__ void __launch_bounds__(32, 1) copy_tma(const char* src, char* dst) {
  extern __shared__ __align__(128) char buf[];
  __shared__ __align__(8) unsigned long long bar[STAGES];
  if (threadIdx.x != 0) return;
  const char* s = src + blockIdx.x * REGION; char* d = dst + blockIdx.x * REGION;
  const int n = REGION / CHUNK;
  for (int i = 0; i < STAGES; i++) mbar_init(&bar[i]);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  unsigned phase[STAGES] = {0, 0, 0, 0};
  for (int i = 0; i < STAGES && i < n; i++) tma_load(buf + i * CHUNK, s + (size_t)i * CHUNK, CHUNK, &bar[i]);
  for (int i = 0; i < n; i++) {
    const int st = i % STAGES;
    mbar_wait(&bar[st], phase[st]); phase[st] ^= 1;
    tma_store(d + (size_t)i * CHUNK, buf + st * CHUNK, CHUNK);
    const int j = i - 1 + STAGES;   // refill the previous stage once its store has read shared memory
    if (i >= 1 && j < n) {
      asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      const int pst = (i - 1) % STAGES;
      tma_load(buf + pst * CHUNK, s + (size_t)j * CHUNK, CHUNK, &bar[pst]);
    }
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__global__ void __launch_bounds__(1024, 1) copy_store(const int4* src, int4* dst) {
  extern __shared__ char hold[];   // one CTA per SM, like the TMA kernel
  const size_t n = REGION / 16;
  const int4* s = src + blockIdx.x * n; int4* d = dst + blockIdx.x * n;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

int main() {
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int SMEM = STAGES * CHUNK + 64 * 1024;   // > half an SM's shared memory: one CTA per SM
  CK(cudaFuncSetAttribute(copy_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(copy_store, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  char *src, *dst; CK(cudaMalloc(&src, nsm * REGION)); CK(cudaMalloc(&dst, nsm * REGION));
  CK(cudaMemset(src, 7, nsm * REGION));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  printf("# copy bandwidth vs SMs, %zu MiB per CTA, GPU memory -> GPU memory, one H200 (%d SMs)\n", REGION >> 20, nsm);
  printf("mode,ctas,GBps\n");
  for (int mode = 0; mode < 2; mode++)
    for (int k : {1, 2, 4, 8, 16, 32, 64, nsm}) {
      std::vector<float> v;
      for (int r = 0; r < 7; r++) {
        CK(cudaEventRecord(a));
        if (mode == 0) copy_tma<<<k, 32, SMEM>>>(src, dst);
        else copy_store<<<k, 1024, SMEM>>>((const int4*)src, (int4*)dst);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (r >= 2) v.push_back(ms);
      }
      std::sort(v.begin(), v.end());
      // spot check the last region's last bytes
      char chk; CK(cudaMemcpy(&chk, dst + (size_t)k * REGION - 1, 1, cudaMemcpyDeviceToHost));
      if (chk != 7) { fprintf(stderr, "copy check failed (mode %d, k %d)\n", mode, k); return 1; }
      CK(cudaMemset(dst, 0, nsm * REGION));
      printf("%s,%d,%.1f\n", mode ? "store" : "tma", k, (double)k * REGION / (v[v.size() / 2] * 1e-3) / 1e9);
      fflush(stdout);
    }
  return 0;
}
