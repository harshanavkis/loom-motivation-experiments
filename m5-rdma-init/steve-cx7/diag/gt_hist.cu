#include <cstdio>
#include <map>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void k(unsigned long long* d, int n) {   // the first n nonzero steps seen by one thread
  unsigned long long a = gt(); int i = 0;
  while (i < n) { unsigned long long b = gt(); if (b != a) { d[i++] = b - a; a = b; } }
}
int main() {
  const int n = 20000; unsigned long long* d; cudaMallocManaged(&d, n * 8);
  k<<<1, 1>>>(d, n); cudaDeviceSynchronize();
  std::map<unsigned long long, int> h; for (int i = 0; i < n; i++) h[d[i]]++;
  for (auto& [s, c] : h) if (c > n / 1000) printf("step %llu ns: %d\n", s, c);
}
