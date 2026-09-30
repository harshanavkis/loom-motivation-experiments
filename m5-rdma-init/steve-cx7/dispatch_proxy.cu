// MoE dispatch, CPU-proxy (B2, NCCL-proxy style), same routing and token size as
// ../gpu-posted/dispatch_ibgda.cu. One kernel computes top-k routing and packs each token
// into a contiguous per-destination block of a send buffer (SM copies); the last CTA then
// writes the R counts + a flag into pinned host memory and spins on a done flag. A CPU proxy
// thread reads the counts (the host round trip that data-dependent routing forces on a CPU
// poster), posts the RDMA writes (GPU send buffer -> GPU receive buffer, mlx5_0 -> mlx5_1
// loopback) and sets done after the last completion. Modes: "block" = one write per
// destination (packed), "token" = one write per token message (the proxy's message rate).
// Timed with CUDA events around the kernel: routing -> all bytes delivered.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <x86intrin.h>
#include <chrono>
#include <infiniband/verbs.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m; cuGetErrorString(r_, &m); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m); exit(1); } } while (0)
#define IB(x) do { if (!(x)) { perror(#x); fprintf(stderr, "%s:%d\n", __FILE__, __LINE__); exit(1); } } while (0)

static const int GID = 3, R = 8, H = 7168, E = 256, TOPK = 8, TMAX = 4096;

struct End { ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_qp* qp; ibv_gid gid; ibv_mr* mr; };
static End open_end(const char* name, void* buf, size_t len) {
  int n; ibv_device** l = ibv_get_device_list(&n); End e = {};
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(l[i]), name)) e.ctx = ibv_open_device(l[i]);
  IB(e.ctx); IB(e.pd = ibv_alloc_pd(e.ctx)); IB(e.cq = ibv_create_cq(e.ctx, 4096, nullptr, nullptr, 0));
  ibv_qp_init_attr qa = {}; qa.send_cq = qa.recv_cq = e.cq; qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = 2048; qa.cap.max_recv_wr = 16; qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
  IB(e.qp = ibv_create_qp(e.pd, &qa)); IB(ibv_query_gid(e.ctx, 1, GID, &e.gid) == 0);
  int fd = -1; CU(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)buf, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  IB(e.mr = ibv_reg_dmabuf_mr(e.pd, 0, len, (uint64_t)buf, fd, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  return e;
}
static void connect(End& a, const End& b) {
  ibv_port_attr pa; IB(ibv_query_port(a.ctx, 1, &pa) == 0);
  ibv_qp_attr x = {}; x.qp_state = IBV_QPS_INIT; x.port_num = 1;
  x.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS) == 0);
  x = {}; x.qp_state = IBV_QPS_RTR; x.path_mtu = pa.active_mtu; x.dest_qp_num = b.qp->qp_num; x.max_dest_rd_atomic = 1; x.min_rnr_timer = 12;
  x.ah_attr.is_global = 1; x.ah_attr.grh.dgid = b.gid; x.ah_attr.grh.sgid_index = GID; x.ah_attr.grh.hop_limit = 64; x.ah_attr.port_num = 1;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
  x = {}; x.qp_state = IBV_QPS_RTS; x.timeout = 14; x.retry_cnt = 7; x.rnr_retry = 7; x.max_rd_atomic = 1;
  IB(ibv_modify_qp(a.qp, &x, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
}

__device__ __forceinline__ unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__device__ unsigned rank_mask(int t, int E_, int R_, int topk) {
  unsigned mask = 0; int got = 0; unsigned long long used[4] = {0, 0, 0, 0};
  for (int j = 0; got < topk; j++) {
    int e = hash(t * 131 + j) % E_;
    if (used[e >> 6] >> (e & 63) & 1) continue;
    used[e >> 6] |= 1ull << (e & 63); got++;
    mask |= 1u << (e / (E_ / R_));
  }
  return mask;
}

__global__ void dispatch_pack(const int4* tok, int4* send, int T, int* counters, int* blocks_done,
                              volatile int* counts_h, volatile unsigned* req_h, volatile unsigned* done_h, unsigned iter) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5, h16 = H / 16;
  for (int t = warp; t < T; t += nwarps) {
    unsigned mask = 0;
    if (lane == 0) mask = rank_mask(t, E, R, TOPK);
    mask = __shfl_sync(0xffffffff, mask, 0);
    while (mask) {
      int r = __ffs(mask) - 1; mask &= mask - 1;
      int slot = 0;
      if (lane == 0) slot = atomicAdd(&counters[r], 1);
      slot = __shfl_sync(0xffffffff, slot, 0);
      int4* d = send + ((size_t)r * T + slot) * h16; const int4* s = tok + (size_t)t * h16;
      for (int i = lane; i < h16; i += 32) d[i] = s[i];
    }
  }
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(blocks_done, 1) == gridDim.x - 1) {  // last CTA hands off to the proxy
    for (int r = 0; r < R; r++) counts_h[r] = atomicAdd(&counters[r], 0);
    __threadfence_system();
    *req_h = iter; __threadfence_system();
    while (*done_h != iter) {}
  }
}

int main(int argc, char** argv) {
  const int iters = 20;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  char *tok, *send, *recv; size_t blk = (size_t)R * TMAX * H;
  CK(cudaMalloc(&tok, (size_t)TMAX * H)); CK(cudaMalloc(&send, blk)); CK(cudaMalloc(&recv, blk));
  CK(cudaMemset(tok, 1, (size_t)TMAX * H));
  End a = open_end("mlx5_0", send, blk), b = open_end("mlx5_1", recv, blk);
  connect(a, b); connect(b, a);
  int *counters, *blocks_done; CK(cudaMalloc(&counters, R * 4)); CK(cudaMalloc(&blocks_done, 4));
  int *counts_h, *counts_d; unsigned *req_h, *done_h, *req_d, *done_d;
  CK(cudaHostAlloc((void**)&counts_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&counts_d, counts_h, 0));
  CK(cudaHostAlloc((void**)&req_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&req_d, req_h, 0));
  CK(cudaHostAlloc((void**)&done_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&done_d, done_h, 0));
  cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  auto c0 = __rdtsc(); auto w0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const double tsc_per_us = (__rdtsc() - c0) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
  printf("# B2 CPU-proxy dispatch (verbs), H=%d E=%d R=%d topk=%d; kernel routes+packs, proxy posts\n", H, E, R, TOPK);
  printf("test,mode,tokens,ctas,messages,median_us,p10_us,p90_us,GBps,proxy_post_us\n");
  unsigned iter = 0;
  for (int per_token = 0; per_token < 2; per_token++)
    for (int T : {128, 4096})
      for (int ctas : {8, 20, 32}) {
        std::vector<float> v; int msgs = 0; double post_us = 0;
        for (int it = 0; it < iters + 3; it++) {
          iter++;
          CK(cudaMemset(counters, 0, R * 4)); CK(cudaMemset(blocks_done, 0, 4)); CK(cudaDeviceSynchronize());
          std::atomic<bool> fail{false}; double post_cyc = 0;
          std::thread proxy([&, iter] {
            while (*(volatile unsigned*)req_h != iter) {}
            int n_wr = 0, outstanding = 0; unsigned long long t0 = __rdtsc();   // outstanding = signaled WRs in flight
            for (int r = 0; r < R; r++) {
              int c = ((volatile int*)counts_h)[r];
              int pieces = per_token ? c : (c > 0);
              for (int p = 0; p < pieces; p++) {
                size_t off = ((size_t)r * T + (per_token ? p : 0)) * H, len = per_token ? H : (size_t)c * H;
                ibv_sge sg = {(uint64_t)(send + off), (uint32_t)len, a.mr->lkey};
                ibv_send_wr wr = {}, *bad; wr.sg_list = &sg; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
                n_wr++; wr.send_flags = (n_wr % 64 == 0) ? IBV_SEND_SIGNALED : 0;
                wr.wr.rdma.remote_addr = (uint64_t)(recv + off); wr.wr.rdma.rkey = b.mr->rkey;
                if (ibv_post_send(a.qp, &wr, &bad)) { fail = true; break; }
                if (n_wr % 64 == 0) {   // every 64th WR is signaled; keep < 30*64 WRs in the 2048-deep SQ
                  outstanding++;
                  ibv_wc wc[16]; int k = ibv_poll_cq(a.cq, 16, wc); if (k > 0) outstanding -= k;
                  while (outstanding >= 30) { k = ibv_poll_cq(a.cq, 16, wc); if (k > 0) outstanding -= k; }
                }
              }
            }
            // final signaled zero-length write: its completion means all earlier writes landed
            ibv_send_wr wr = {}, *bad; wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = (uint64_t)recv; wr.wr.rdma.rkey = b.mr->rkey;
            if (ibv_post_send(a.qp, &wr, &bad)) fail = true;
            outstanding++;
            while (outstanding > 0) {   // drain: the last completion is the zero-length write's
              ibv_wc wc; int k = ibv_poll_cq(a.cq, 1, &wc);
              if (k < 0 || (k == 1 && wc.status != IBV_WC_SUCCESS)) { fail = true; break; }
              outstanding -= k;
            }
            post_cyc = __rdtsc() - t0;
            _mm_sfence(); *(volatile unsigned*)done_h = iter;
          });
          CK(cudaEventRecord(ea));
          dispatch_pack<<<ctas, 256>>>((const int4*)tok, (int4*)send, T, counters, blocks_done, counts_d, req_d, done_d, iter);
          CK(cudaEventRecord(eb)); CK(cudaEventSynchronize(eb)); CK(cudaGetLastError());
          proxy.join();
          if (fail) { fprintf(stderr, "proxy failed\n"); return 1; }
          float ms; CK(cudaEventElapsedTime(&ms, ea, eb)); if (it >= 3) { v.push_back(ms * 1e3f); post_us += post_cyc; }
          msgs = 0; for (int r = 0; r < R; r++) msgs += counts_h[r];
        }
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        printf("proxy,%s,%d,%d,%d,%.1f,%.1f,%.1f,%.2f,%.1f\n", per_token ? "token" : "block", T, ctas, msgs, med,
               v[v.size() / 10], v[v.size() * 9 / 10], (double)msgs * H / (med * 1e-6) / 1e9, post_us / iters / tsc_per_us);
        fflush(stdout);
      }
  return 0;
}
