#include <cstdio>
#include <cuda_runtime.h>
int main(){int v=-1; cudaDeviceGetAttribute(&v, cudaDevAttrGPUDirectRDMAWritesOrdering, 0); printf("GPUDirectRDMAWritesOrdering=%d (0 none, 100 owner, 200 all devices)\n", v);
int f=-1; cudaDeviceGetAttribute(&f, cudaDevAttrGPUDirectRDMAFlushWritesOptions, 0); printf("FlushWritesOptions=%d\n", f);}
