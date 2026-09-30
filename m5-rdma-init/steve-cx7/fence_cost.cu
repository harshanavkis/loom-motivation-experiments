// What does the GPU pay for the ordering steps a GPU-initiated post needs?
// IBGDA writes the WQE into HBM, fences at system scope, then writes the doorbell (an MMIO
// store into the NIC's BAR, i.e. a store that leaves the GPU over PCIe) and fences again.
// Timed from one thread with %globaltimer, per iteration (median of N):
//   hbm_store+fence_sys  : st.global to HBM, then __threadfence_system()
//   sys_store+fence_sys  : st to pinned HOST memory (leaves the GPU over PCIe), then fence
//   sys_store only       : st to pinned host memory, no fence (posted write)
//   sys_load             : ld from pinned host memory (a PCIe round trip)
// Run under numactl -m 0 / -m 1 to put the host page on the GPU's or the NIC's socket.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e_)); return 1; } } while (0)
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void k(volatile unsigned* hbm, volatile unsigned* sys, int n, unsigned long long* out) {
  unsigned long long t; unsigned x = 0;
  for (int i = 0; i < n; i++) { t = gt(); hbm[0] = i; __threadfence_system(); out[0 * n + i] = gt() - t; }
  for (int i = 0; i < n; i++) { t = gt(); sys[0] = i; __threadfence_system(); out[1 * n + i] = gt() - t; }
  for (int i = 0; i < n; i++) { t = gt(); sys[16 * (i & 63)] = i; out[2 * n + i] = gt() - t; }
  for (int i = 0; i < n; i++) { t = gt(); x += sys[16 * (i & 63)]; out[3 * n + i] = gt() - t; }
  if (x == 0xdeadbeef) out[0] = 0;
}

int main() {
  const int n = 20000;
  unsigned *hbm, *sys_h, *sys_d; unsigned long long* out;
  CK(cudaMalloc(&hbm, 64)); CK(cudaHostAlloc((void**)&sys_h, 4096, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&sys_d, sys_h, 0)); CK(cudaMalloc(&out, 4ull * n * 8));
  k<<<1, 1>>>(hbm, sys_d, n, out); CK(cudaDeviceSynchronize());
  std::vector<unsigned long long> v(4ull * n); CK(cudaMemcpy(v.data(), out, v.size() * 8, cudaMemcpyDeviceToHost));
  const char* name[] = {"hbm_store+fence_sys", "sys_store+fence_sys", "sys_store_only", "sys_load"};
  printf("test,median_ns,p99_ns\n");
  for (int j = 0; j < 4; j++) {
    std::vector<unsigned long long> s(v.begin() + j * n + 100, v.begin() + (j + 1) * n);
    std::sort(s.begin(), s.end());
    printf("%s,%llu,%llu\n", name[j], s[s.size() / 2], s[(size_t)(s.size() * 0.99)]);
  }
  return 0;
}
