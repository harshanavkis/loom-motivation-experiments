// Copy-engine throughput vs number of concurrent streams (each stream can map to a
// different copy engine).  Answers: can copy engines alone sustain a CX-8 800G
// (100 GB/s) or NVLink-class rate?  64 MiB requests, cudaMemcpyBatchAsync with
// PreferOverlapWithCompute, d2d (HBM->HBM) and d2h (HBM->pinned host).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

int main() {
  const size_t chunk = 64ull << 20, per_stream = 16;  // 1 GiB per stream per measurement
  const int max_streams = 8, reps = 3;
  char *src, *ddst, *hdst;
  CK(cudaMalloc(&src, chunk * max_streams)); CK(cudaMalloc(&ddst, chunk * max_streams));
  CK(cudaHostAlloc(&hdst, chunk * max_streams, 0));
  CK(cudaMemset(src, 1, chunk * max_streams));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("# device=%s asyncEngineCount=%d\n", p.name, p.asyncEngineCount);
  printf("dir,streams,rep,GBps\n");
  std::vector<cudaStream_t> st(max_streams);
  for (auto& s : st) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int d2h = 0; d2h < 2; d2h++)
    for (int ns : {1, 2, 4, 8})
      for (int r = 0; r < reps; r++) {
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a, st[0]));
        for (int i = 1; i < ns; i++) CK(cudaStreamWaitEvent(st[i], a));
        for (size_t j = 0; j < per_stream; j++)
          for (int i = 0; i < ns; i++) {
            void* dsts[1] = {(d2h ? hdst : ddst) + i * chunk}; void* srcs[1] = {src + i * chunk}; size_t sz[1] = {chunk};
            cudaMemcpyAttributes at = {};
            at.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            at.srcLocHint.type = cudaMemLocationTypeDevice;
            at.dstLocHint.type = d2h ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
            at.flags = cudaMemcpyFlagPreferOverlapWithCompute;
            size_t idx = 0, fail = 0;
            CK(cudaMemcpyBatchAsync(dsts, srcs, sz, 1, &at, &idx, 1, &fail, st[i]));
          }
        for (int i = 1; i < ns; i++) { cudaEvent_t e; CK(cudaEventCreate(&e)); CK(cudaEventRecord(e, st[i])); CK(cudaStreamWaitEvent(st[0], e)); CK(cudaEventDestroy(e)); }
        CK(cudaEventRecord(b, st[0])); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("%s,%d,%d,%.1f\n", d2h ? "d2h" : "d2d", ns, r, ns * per_stream * chunk / (ms * 1e-3) / 1e9);
      }
  return 0;
}
