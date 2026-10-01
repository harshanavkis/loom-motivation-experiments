// GPU-initiated RDMA with DeepEP's own post path next to NVSHMEM's generic put, on the
// same IBGDA QPs (2 NVSHMEM PEs on steve's H200 over the CX-7 loopback, as dispatch_ibgda.cu).
// DeepEP V1 (a56d615) csrc/kernels/legacy/ibgda_device.cuh, with one change (the RC QP index,
// ported to NVSHMEM 3.6.5's layout; see build_deepep_post.sh). The post path is as is: warp-parallel
// WQE writes, a gpu-scope __threadfence, gpu-scope release stores for the doorbell record
// and the doorbell, and (kAlwaysDoPostSend = false) one doorbell per 4 messages per QP.
//   --test lat:      one warp, per iteration: put (8 B or a token) -> completion on that QP.
//                    post = the put call itself (SM time to post), total = put + completion.
//                    nvshmem: nvshmemx_putmem_nbi_warp + nvshmem_quiet;
//                    deepep:  nvshmemi_ibgda_put_nbi_warp<true> + nvshmemi_ibgda_quiet.
//   --test dispatch: the dispatch_ibgda.cu dispatch (same routing, one put per token message).
//                    deepep: put_nbi_warp<false> on QP r (DeepEP LL: qp = destination, message
//                    index = slot), then the last CTA rings each QP for what is left and waits.
//                    DeepEP skips the WQ slot check and requires NVSHMEM_QP_DEPTH >= (tokens + 1) * 2
//                    in flight per QP (deep_ep/buffers/legacy.py), i.e. its low-latency (decode)
//                    regime: the deepep mode runs 16-128 tokens only (<= ~85 messages per QP).
// Timed with %globaltimer (lat) or CUDA events around the kernel (dispatch).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "compiled.cuh"
#include "ibgda_device_nv365.cuh"   // ibgda_device.cuh with the QP index ported to 3.6.5

namespace dl = deep_ep::legacy;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int R = 8, E = 256, TOPK = 8, HMAX = 7168, TMAX = 4096;

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// routing shared with dispatch_ibgda.cu / dispatch_proxy.cu
__device__ __forceinline__ unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__device__ unsigned rank_mask(int t) {
  unsigned mask = 0; int got = 0; unsigned long long used[4] = {0, 0, 0, 0};
  for (int j = 0; got < TOPK; j++) {
    int e = hash(t * 131 + j) % E;
    if (used[e >> 6] >> (e & 63) & 1) continue;
    used[e >> 6] |= 1ull << (e & 63); got++;
    mask |= 1u << (e / (E / R));
  }
  return mask;
}

__global__ void probe(int* out) {
  out[0] = dl::ibgda_get_state()->use_async_postsend;
  out[1] = dl::ibgda_get_state()->num_rc_per_pe;
}

template <bool kDeepEP>
__global__ void put_lat(char* dst, const char* src, size_t n, int iters, unsigned long long* post_ns, unsigned long long* total_ns) {
  const int lane = threadIdx.x & 31;
  for (int i = 0; i < iters; i++) {
    unsigned long long t0 = gtimer();
    if (kDeepEP) dl::nvshmemi_ibgda_put_nbi_warp<true>((uint64_t)dst, (uint64_t)src, n, 1, 0, lane, 0);
    else nvshmemx_putmem_nbi_warp(dst, src, n, 1);
    __syncwarp();
    unsigned long long t1 = gtimer();
    if (lane == 0) { if (kDeepEP) dl::nvshmemi_ibgda_quiet(1, 0); else nvshmem_quiet(); }
    __syncwarp();
    unsigned long long t2 = gtimer();
    if (lane == 0) { post_ns[i] = t1 - t0; total_ns[i] = t2 - t0; }
  }
}

template <bool kDeepEP>
__global__ void dispatch(const char* tok, char* recv, int T, int H, int* counters, int* blocks_done) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int t = warp; t < T; t += nwarps) {
    unsigned mask = 0;
    if (lane == 0) mask = rank_mask(t);
    mask = __shfl_sync(0xffffffff, mask, 0);
    while (mask) {
      int r = __ffs(mask) - 1; mask &= mask - 1;
      int slot = 0;
      if (lane == 0) slot = atomicAdd(&counters[r], 1);
      slot = __shfl_sync(0xffffffff, slot, 0);
      char* d = recv + ((size_t)r * T + slot) * H; const char* s = tok + (size_t)t * H;
      if (kDeepEP) dl::nvshmemi_ibgda_put_nbi_warp<false>((uint64_t)d, (uint64_t)s, H, 1, r, lane, slot);
      else nvshmemx_putmem_nbi_warp(d, s, H, 1);
    }
  }
  if (!kDeepEP) { nvshmem_quiet(); return; }
  // doorbells are batched per QP: the last CTA rings each QP for its remaining WQEs and waits
  __shared__ int last;
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(blocks_done, 1) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  const int w = threadIdx.x >> 5;
  if (w < R && lane == 0) {
    auto qp = dl::ibgda_get_rc(1, w);
    dl::ibgda_post_send(qp, dl::ld_na_relaxed(&qp->mvars.tx_wq.ready_head));
    dl::nvshmemi_ibgda_quiet(1, w);
  }
}

int main(int argc, char** argv) {
  bool deepep = false, lat = true; int H = 7168;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--test")) lat = !strcmp(argv[++i], "lat");
    else if (!strcmp(argv[i], "--mode")) deepep = !strcmp(argv[++i], "deepep");
    else if (!strcmp(argv[i], "--H")) H = atoi(argv[++i]);
  }
  nvshmem_init();
  int me = nvshmem_my_pe();
  CK(cudaSetDevice(0));
  char* tok = (char*)nvshmem_malloc((size_t)TMAX * HMAX);
  char* recv = (char*)nvshmem_malloc((size_t)R * TMAX * HMAX);
  const char* done_file = "/tmp/loom_deepep_post.done";
  if (me == 0) unlink(done_file);
  nvshmem_barrier_all();
  if (me != 0) {  // PE 1 is only memory; keep it off the GPU (no MPS)
    while (access(done_file, F_OK) != 0) usleep(100000);
    nvshmem_barrier_all(); nvshmem_finalize(); return 0;
  }
  CK(cudaMemset(tok, 1, (size_t)TMAX * HMAX));
  int* info; CK(cudaMallocManaged(&info, 8)); probe<<<1, 1>>>(info); CK(cudaDeviceSynchronize());
  const char* mode = deepep ? "deepep" : "nvshmem";
  printf("# %s post path, use_async_postsend=%d num_rc_per_pe=%d\n", mode, info[0], info[1]);
  if (lat) {
    const int iters = 10000, warm = 1000;
    unsigned long long *post, *total; CK(cudaMalloc(&post, iters * 8)); CK(cudaMalloc(&total, iters * 8));
    printf("test,mode,size,post_med_us,total_med_us,total_p10_us,total_p90_us\n");
    for (size_t n : {(size_t)8, (size_t)7168}) {
      if (deepep) put_lat<true><<<1, 32>>>(recv, tok, n, iters, post, total);
      else put_lat<false><<<1, 32>>>(recv, tok, n, iters, post, total);
      CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      std::vector<unsigned long long> p(iters), t(iters);
      CK(cudaMemcpy(p.data(), post, iters * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t.data(), total, iters * 8, cudaMemcpyDeviceToHost));
      std::sort(p.begin() + warm, p.end()); std::sort(t.begin() + warm, t.end());
      auto q = [&](std::vector<unsigned long long>& v, double f) { return v[warm + (size_t)((iters - warm) * f)] / 1e3; };
      printf("lat,%s,%zu,%.2f,%.2f,%.2f,%.2f\n", mode, n, q(p, 0.5), q(t, 0.5), q(t, 0.1), q(t, 0.9));
      fflush(stdout);
    }
  } else {
    const int iters = 20;
    int *counters, *blocks_done; CK(cudaMalloc(&counters, R * 4)); CK(cudaMalloc(&blocks_done, 4));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    printf("test,mode,H,tokens,ctas,messages,median_us,p10_us,p90_us,GBps\n");
    for (int T : {16, 32, 128, 1024, 4096})
      for (int ctas : {8, 20}) {
        if (deepep && T > 128) continue;   // beyond DeepEP's in-flight limit per QP
        std::vector<float> v; int msgs = 0;
        for (int it = 0; it < iters + 3; it++) {
          CK(cudaMemset(counters, 0, R * 4)); CK(cudaMemset(blocks_done, 0, 4)); CK(cudaDeviceSynchronize());
          CK(cudaEventRecord(a));
          if (deepep) dispatch<true><<<ctas, 256>>>(tok, recv, T, H, counters, blocks_done);
          else dispatch<false><<<ctas, 256>>>(tok, recv, T, H, counters, blocks_done);
          CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
          float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (it >= 3) v.push_back(ms * 1e3f);
          std::vector<int> c(R); CK(cudaMemcpy(c.data(), counters, R * 4, cudaMemcpyDeviceToHost));
          msgs = 0; for (int x : c) msgs += x;
        }
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        printf("dispatch,%s,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.2f\n", mode, H, T, ctas, msgs, med, v[v.size() / 10],
               v[v.size() * 9 / 10], (double)msgs * H / (med * 1e-6) / 1e9);
        fflush(stdout);
      }
  }
  { FILE* f = fopen(done_file, "w"); if (f) fclose(f); }
  nvshmem_barrier_all();
  nvshmem_finalize();
  return 0;
}
