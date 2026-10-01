// The NIC's own post -> completion time for one 8 B RDMA WRITE into GPU memory, with the
// PCIe reads the NIC does before it can send removed one at a time:
//   payload fetch: src in GPU memory, in host memory, or inline (the CPU copies the data
//                  into the work request, so the NIC reads no payload);
//   WQE fetch:     doorbell only (the NIC reads the work request from host memory), or
//                  BlueFlame (the CPU writes the whole work request into the NIC's BAR).
// BlueFlame is chosen by the mlx5 provider (rdma-core providers/mlx5/qp.c): a single post
// uses it unless MLX5_SHUT_UP_BF=1, and a non-inline one also needs MLX5_POST_SEND_PREFER_BF,
// which defaults to on. So proxy_b2 already used BlueFlame. The run script sets SHUT_UP_BF.
// Loopback on one host as in proxy_b2: mlx5_0 -> mlx5_1 (RoCE v2, GID 3), dst = GPU memory
// via dma-buf. One CPU thread posts, then polls the CQ; ops are spaced 2 us apart (about the
// proxy_b2 handoff). Times are rdtsc: post = ibv_post_send, cqe = post return -> CQE seen
// (the quantity proxy_b2 reports as proxy_poll_us). Usage: nic_post gpu|host|inline [iters]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
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
static const int ACC = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

struct End { ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_qp* qp; ibv_gid gid; ibv_mr* mr; };

// buf is GPU memory (registered via dma-buf) or, with host = true, host memory
static End open_end(const char* name, void* buf, size_t len, bool host) {
  int n; ibv_device** l = ibv_get_device_list(&n); End e = {};
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(l[i]), name)) e.ctx = ibv_open_device(l[i]);
  IB(e.ctx); IB(e.pd = ibv_alloc_pd(e.ctx)); IB(e.cq = ibv_create_cq(e.ctx, 256, nullptr, nullptr, 0));
  ibv_qp_init_attr qa = {}; qa.send_cq = qa.recv_cq = e.cq; qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = 128; qa.cap.max_recv_wr = 16; qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
  qa.cap.max_inline_data = 64;
  IB(e.qp = ibv_create_qp(e.pd, &qa));
  IB(ibv_query_gid(e.ctx, 1, GID, &e.gid) == 0);
  if (host) {
    IB(e.mr = ibv_reg_mr(e.pd, buf, len, ACC));
  } else {
    int fd = -1;
    CU(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)buf, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    IB(e.mr = ibv_reg_dmabuf_mr(e.pd, 0, len, (uint64_t)buf, fd, ACC));
  }
  return e;
}

static void connect(End& a, const End& b) {  // move a's QP to RTS, targeting b's QP
  ibv_port_attr pa; IB(ibv_query_port(a.ctx, 1, &pa) == 0);
  ibv_qp_attr x = {}; x.qp_state = IBV_QPS_INIT; x.port_num = 1; x.pkey_index = 0;
  x.qp_access_flags = ACC;
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

int main(int argc, char** argv) {
  if (argc < 2 || (strcmp(argv[1], "gpu") && strcmp(argv[1], "host") && strcmp(argv[1], "inline"))) {
    fprintf(stderr, "usage: %s gpu|host|inline [iters]\n", argv[0]); return 1;
  }
  const char* src_kind = argv[1];
  bool host_src = strcmp(src_kind, "gpu") != 0, inl = !strcmp(src_kind, "inline");
  int iters = argc > 2 ? atoi(argv[2]) : 20000;
  const size_t n = 8, len = 1 << 16;
  CK(cudaSetDevice(0)); CK(cudaFree(0));
  char *src, *dst; CK(cudaMalloc(&dst, len));
  if (host_src) { IB(posix_memalign((void**)&src, 4096, len) == 0); memset(src, 1, len); }
  else CK(cudaMalloc(&src, len));
  End a = open_end("mlx5_0", src, len, host_src), b = open_end("mlx5_1", dst, len, false);
  connect(a, b); connect(b, a);
  auto c0 = __rdtsc(); auto w0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  double tsc_per_us = (__rdtsc() - c0) / std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
  auto env_on = [](const char* k, bool dflt) { const char* v = getenv(k); return v ? strcmp(v, "0") != 0 : dflt; };
  const char* bf = !env_on("MLX5_SHUT_UP_BF", false) && (inl || env_on("MLX5_POST_SEND_PREFER_BF", true)) ? "on" : "off";

  std::vector<double> post(iters), cqe(iters);
  for (int i = 0; i < iters; i++) {
    ibv_sge sg = {(uint64_t)src, (uint32_t)n, a.mr->lkey};
    ibv_send_wr wr = {}, *bad; wr.wr_id = i; wr.sg_list = &sg; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | (inl ? IBV_SEND_INLINE : 0);
    wr.wr.rdma.remote_addr = (uint64_t)dst; wr.wr.rdma.rkey = b.mr->rkey;
    auto t0 = __rdtsc();
    IB(ibv_post_send(a.qp, &wr, &bad) == 0);
    auto t1 = __rdtsc();
    ibv_wc wc; int k;
    while ((k = ibv_poll_cq(a.cq, 1, &wc)) == 0) {}
    auto t2 = __rdtsc();
    if (k < 0 || wc.status != IBV_WC_SUCCESS) { fprintf(stderr, "wc status %d\n", k < 0 ? -1 : wc.status); return 1; }
    post[i] = (t1 - t0) / tsc_per_us; cqe[i] = (t2 - t1) / tsc_per_us;
    while ((__rdtsc() - t2) / tsc_per_us < 2.0) {}
  }
  auto stat = [&](std::vector<double>& v, double q) {
    std::sort(v.begin() + 100, v.end()); return v[100 + (size_t)((iters - 100) * q)]; };
  double pm = stat(post, 0.5), cm = stat(cqe, 0.5), cp = stat(cqe, 0.99);
  printf("%s,%s,%zu,%.3f,%.2f,%.2f\n", src_kind, bf, n, pm, cm, cp);
  return 0;
}
