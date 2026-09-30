// CPU-posted RDMA as it is actually used for GPU data (the NCCL proxy model, "B2"),
// timed the same way as the GPU-posted IBGDA put + quiet: from a kernel's point of view.
//   kernel (1 thread):  t0 = globaltimer; req = i (store into pinned host memory);
//                       spin until done == i;  latency = globaltimer - t0
//   proxy (CPU thread): spin until req == i; ibv_post_send(RDMA WRITE, GPU src -> GPU dst,
//                       signaled); poll the CQ; done = i (store into pinned host memory)
// Loopback on one host: QP on mlx5_0 connected to a QP on mlx5_1 (RoCE v2, GID 3); both
// buffers are GPU memory registered via dma-buf. Also reports the proxy's own time spent
// in ibv_post_send and CQ polling. Place the process with numactl (proxy CPU + host flags).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <x86intrin.h>
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m; cuGetErrorString(r_, &m); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m); exit(1); } } while (0)
#define IB(x) do { if (!(x)) { perror(#x); fprintf(stderr, "%s:%d\n", __FILE__, __LINE__); exit(1); } } while (0)

static const int GID = 3;

struct End { ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_qp* qp; ibv_gid gid; ibv_mr* mr; };

static End open_end(const char* name, void* gpu_buf, size_t len) {
  int n; ibv_device** l = ibv_get_device_list(&n); End e = {};
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(l[i]), name)) e.ctx = ibv_open_device(l[i]);
  IB(e.ctx); IB(e.pd = ibv_alloc_pd(e.ctx)); IB(e.cq = ibv_create_cq(e.ctx, 256, nullptr, nullptr, 0));
  ibv_qp_init_attr qa = {}; qa.send_cq = qa.recv_cq = e.cq; qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = 128; qa.cap.max_recv_wr = 16; qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
  IB(e.qp = ibv_create_qp(e.pd, &qa));
  IB(ibv_query_gid(e.ctx, 1, GID, &e.gid) == 0);
  int fd = -1;
  CU(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)gpu_buf, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  IB(e.mr = ibv_reg_dmabuf_mr(e.pd, 0, len, (uint64_t)gpu_buf, fd,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  return e;
}

static void connect(End& a, const End& b) {  // move a's QP to RTS, targeting b's QP
  ibv_port_attr pa; IB(ibv_query_port(a.ctx, 1, &pa) == 0);
  ibv_qp_attr x = {}; x.qp_state = IBV_QPS_INIT; x.port_num = 1; x.pkey_index = 0;
  x.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS) == 0);
  x = {}; x.qp_state = IBV_QPS_RTR; x.path_mtu = pa.active_mtu; x.dest_qp_num = b.qp->qp_num; x.rq_psn = 0;
  x.max_dest_rd_atomic = 1; x.min_rnr_timer = 12;
  x.ah_attr.is_global = 1; x.ah_attr.grh.dgid = b.gid; x.ah_attr.grh.sgid_index = GID; x.ah_attr.grh.hop_limit = 64;
  x.ah_attr.port_num = 1;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
  x = {}; x.qp_state = IBV_QPS_RTS; x.timeout = 14; x.retry_cnt = 7; x.rnr_retry = 7; x.sq_psn = 0; x.max_rd_atomic = 1;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                   IBV_QP_MAX_QP_RD_ATOMIC) == 0);
}

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void requester(volatile unsigned* req, volatile unsigned* done, int iters, unsigned long long* ns) {
  for (int i = 1; i <= iters; i++) {
    unsigned long long t0 = gtimer();
    *req = i; __threadfence_system();
    while (*done != (unsigned)i) {}
    ns[i - 1] = gtimer() - t0;
  }
}

int main(int argc, char** argv) {
  int iters = 20000;
  if (argc > 1) iters = atoi(argv[1]);
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  const size_t maxn = 4 << 20;
  char *src, *dst; CK(cudaMalloc(&src, maxn)); CK(cudaMalloc(&dst, maxn));
  End a = open_end("mlx5_0", src, maxn), b = open_end("mlx5_1", dst, maxn);
  connect(a, b); connect(b, a);
  unsigned *req_h, *done_h, *req_d, *done_d;   // flags in pinned host memory, mapped for the kernel
  CK(cudaHostAlloc((void**)&req_h, 64, cudaHostAllocMapped)); CK(cudaHostAlloc((void**)&done_h, 64, cudaHostAllocMapped));
  CK(cudaHostGetDevicePointer((void**)&req_d, req_h, 0)); CK(cudaHostGetDevicePointer((void**)&done_d, done_h, 0));
  unsigned long long* ns_d; CK(cudaMalloc(&ns_d, (size_t)iters * 8));
  // TSC frequency for the proxy's own timings
  auto c0 = __rdtsc(); auto w0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  double tsc_per_us = (__rdtsc() - c0) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();

  printf("# CPU-posted RDMA write GPU->GPU via a CPU proxy thread, timed from a kernel (GPU globaltimer)\n");
  printf("test,size,median_us,p99_us,proxy_post_us,proxy_poll_us\n");
  const size_t sizes[] = {8, 64, 512, 4096, 65536, 1 << 20, 4 << 20};
  for (size_t n : sizes) {
    *(volatile unsigned*)req_h = 0; *(volatile unsigned*)done_h = 0;
    std::atomic<bool> fail{false}; double post_cyc = 0, poll_cyc = 0;
    std::thread proxy([&] {
      for (int i = 1; i <= iters; i++) {
        while (*(volatile unsigned*)req_h != (unsigned)i) {}
        ibv_sge sg = {(uint64_t)src, (uint32_t)n, a.mr->lkey};
        ibv_send_wr wr = {}, *bad; wr.wr_id = i; wr.sg_list = &sg; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED; wr.wr.rdma.remote_addr = (uint64_t)dst; wr.wr.rdma.rkey = b.mr->rkey;
        auto t = __rdtsc();
        if (ibv_post_send(a.qp, &wr, &bad)) { fail = true; *(volatile unsigned*)done_h = i; return; }
        auto t2 = __rdtsc(); post_cyc += t2 - t;
        ibv_wc wc; int k;
        while ((k = ibv_poll_cq(a.cq, 1, &wc)) == 0) {}
        poll_cyc += __rdtsc() - t2;
        if (k < 0 || wc.status != IBV_WC_SUCCESS) { fprintf(stderr, "wc status %d\n", k < 0 ? -1 : wc.status); fail = true; }
        _mm_sfence();
        *(volatile unsigned*)done_h = i;
      }
    });
    requester<<<1, 1>>>(req_d, done_d, iters, ns_d);
    CK(cudaDeviceSynchronize()); proxy.join();
    if (fail) { printf("proxy,%zu,FAILED\n", n); continue; }
    std::vector<unsigned long long> v(iters); CK(cudaMemcpy(v.data(), ns_d, (size_t)iters * 8, cudaMemcpyDeviceToHost));
    std::sort(v.begin() + 100, v.end());
    size_t m = 100 + (iters - 100) / 2, q = 100 + (size_t)((iters - 100) * 0.99);
    printf("proxy,%zu,%.2f,%.2f,%.3f,%.2f\n", n, v[m] / 1e3, v[q] / 1e3, post_cyc / iters / tsc_per_us, poll_cyc / iters / tsc_per_us);
    fflush(stdout);
  }
  return 0;
}
