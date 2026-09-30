// MoE dispatch, GPU-initiated (B1, DeepEP style): one kernel computes top-k routing and
// puts every token straight to its destination with IBGDA (warp-level nbi puts), then quiet.
// 2 NVSHMEM PEs on steve's H200 over the CX-7 loopback; the R "destination ranks" are R
// regions of PE 1's receive buffer (EP-shaped traffic over one peer). Timed with CUDA events
// around the kernel: routing -> all puts complete. Tokens: H bytes (7168 = FP8 DeepSeek-V3).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// routing shared with dispatch_proxy.cu: token t picks TOPK distinct experts of E (hash),
// destination rank = expert / (E / R); the token is sent once per distinct rank
__device__ __forceinline__ unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__device__ unsigned rank_mask(int t, int E, int R, int topk) {
  unsigned mask = 0; int got = 0; unsigned long long used[4] = {0, 0, 0, 0};
  for (int j = 0; got < topk; j++) {
    int e = hash(t * 131 + j) % E;
    if (used[e >> 6] >> (e & 63) & 1) continue;
    used[e >> 6] |= 1ull << (e & 63); got++;
    mask |= 1u << (e / (E / R));
  }
  return mask;
}

__global__ void dispatch(const char* tok, char* recv, int T, int H, int E, int R, int topk, int* counters) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int t = warp; t < T; t += nwarps) {
    unsigned mask = 0;
    if (lane == 0) mask = rank_mask(t, E, R, topk);
    mask = __shfl_sync(0xffffffff, mask, 0);
    while (mask) {
      int r = __ffs(mask) - 1; mask &= mask - 1;
      int slot = 0;
      if (lane == 0) slot = atomicAdd(&counters[r], 1);
      slot = __shfl_sync(0xffffffff, slot, 0);
      nvshmemx_putmem_nbi_warp(recv + ((size_t)r * T + slot) * H, tok + (size_t)t * H, H, 1);
    }
  }
  nvshmem_quiet();
}

int main(int argc, char** argv) {
  int iters = 20;
  nvshmem_init();
  int me = nvshmem_my_pe();
  CK(cudaSetDevice(0));
  const int H = 7168, E = 256, R = 8, TOPK = 8, TMAX = 4096;
  char* tok = (char*)nvshmem_malloc((size_t)TMAX * H);
  char* recv = (char*)nvshmem_malloc((size_t)R * TMAX * H);
  const char* done_file = "/tmp/loom_dispatch_ibgda.done";
  if (me == 0) unlink(done_file);
  nvshmem_barrier_all();
  if (me != 0) {  // PE 1 is only memory; keep off the GPU (no MPS: processes would time-slice)
    while (access(done_file, F_OK) != 0) usleep(100000);
    nvshmem_barrier_all(); nvshmem_finalize(); return 0;
  }
  int* counters; CK(cudaMalloc(&counters, R * sizeof(int)));
  CK(cudaMemset(tok, 1, (size_t)TMAX * H));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  printf("# B1 GPU-initiated dispatch (IBGDA, warp nbi puts), H=%d E=%d R=%d topk=%d, %s QPs/peer\n", H, E, R, TOPK,
         getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE") ? getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE") : "default");
  printf("test,tokens,ctas,warps_per_cta,messages,median_us,p10_us,p90_us,GBps\n");
  for (int T : {128, 4096})
    for (int ctas : {8, 20, 32}) {
      std::vector<float> v; int msgs = 0;
      for (int it = 0; it < iters + 3; it++) {
        CK(cudaMemset(counters, 0, R * sizeof(int))); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a)); dispatch<<<ctas, 256>>>(tok, recv, T, H, E, R, TOPK, counters); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (it >= 3) v.push_back(ms * 1e3f);
        std::vector<int> c(R); CK(cudaMemcpy(c.data(), counters, R * 4, cudaMemcpyDeviceToHost));
        msgs = 0; for (int x : c) msgs += x;
      }
      std::sort(v.begin(), v.end());
      double med = v[v.size() / 2];
      printf("ibgda,%d,%d,8,%d,%.1f,%.1f,%.1f,%.2f\n", T, ctas, msgs, med, v[v.size() / 10], v[v.size() * 9 / 10],
             (double)msgs * H / (med * 1e-6) / 1e9);
      fflush(stdout);
    }
  { FILE* f = fopen(done_file, "w"); if (f) fclose(f); }
  nvshmem_barrier_all();
  nvshmem_finalize();
  return 0;
}
