#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void k(unsigned long long* out) {
  // smallest nonzero step of %globaltimer, and clock64 ticks per ns over ~10 ms
  unsigned long long a = gt(), mn = ~0ull; int changes = 0;
  while (changes < 2000) { unsigned long long b = gt(); if (b != a) { if (b - a < mn) mn = b - a; a = b; changes++; } }
  unsigned long long g0 = gt(); long long c0 = clock64();
  while (gt() - g0 < 10000000ull) {}
  unsigned long long g1 = gt(); long long c1 = clock64();
  out[0] = mn; out[1] = g1 - g0; out[2] = c1 - c0;
}
int main() {
  unsigned long long* o; cudaMallocManaged(&o, 24); k<<<1, 1>>>(o); cudaDeviceSynchronize();
  printf("globaltimer min step %llu ns; clock64 %.3f GHz\n", o[0], (double)o[2] / o[1]);
}
