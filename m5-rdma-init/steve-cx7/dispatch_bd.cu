// MoE dispatch, time breakdown with one end point: from the sender kernel's first instruction
// until the receiver sees the arrival signal. Where the time goes, and where the SM time goes,
// on the local load/store path and on the remote GPU-initiated RDMA path.
// Routing and token sizes as dispatch_ibgda.cu / dispatch_ce.cu: top-8 of 256 experts, R = 8
// destinations, one message per token and destination. 256-thread CTAs (8 warps); a warp
// handles one token at a time. The token flow is DeepEP V2.5's (impls/ep/dispatch.cuh):
// TMA-load the token into shared memory, route it, then
//   --path local   : each lane TMA-stores the token into its destination's slot (V2.5's
//                    NVLink branch: get_sym_ptr -> tma_store_1d). The destination is the same
//                    GPU's HBM (a local peer). End: every CTA waits for its stores and fences
//                    (system scope); the last CTA stores the signal with release semantics
//                    (DeepEP V1's NVLink path: st_release_sys_global).
//   --path flush   : TMA-store the token into the send buffer, wait for that store, then one put
//                    per destination with DeepEP's IBGDA post path. End: every CTA rings every
//                    QP and waits for all its completions (V2.5's GIN barrier flush); then the
//                    last CTA sends one signal per destination, an RDMA atomic add.
//   --path ordered : as flush without the completion wait: the last CTA posts a signal on each
//                    QP behind that QP's data, and the RC QP keeps it behind (DeepEP V1 low-
//                    latency dispatch: amo_nonfetch_add after the puts, on the expert's QP).
//   QPs: --qp warp  : QP = global warp % NVSHMEM_IBGDA_NUM_RC_PER_PE (V2.5 maps QPs to SMs and
//                     channels, get_qp_mapping in comm/barrier.cuh); flush only.
//        --qp dest --nq N : N QPs per destination, slot s on QP s % N (V1 low latency: one QP per
//                     local expert); ordered needs this, one signal per QP.
//   Posting: --post warp : one warp put per destination, in turn (DeepEP V1's warp put);
//            --post lane : each lane posts its destination's put in parallel (V2.5's shape), with
//                          a one-thread version of DeepEP's post path (put_nbi_thread).
// The RDMA paths run 2 NVSHMEM PEs on steve's H200 over the CX-7 loopback; the receive buffer
// is PE 1's, which PE 0 maps by CUDA IPC (needs NVSHMEM_DISABLE_CUDA_VMM=1). Receiver: one
// 1024-thread CTA on its own stream, started before the sender: warp 0 polls the R signal
// words, the other warps poll the last 8 B of every message (the run's nonce, written into
// each token), stamping %globaltimer (32 ns steps). Sender and receiver share that clock.
// The sender stamps each CTA's phases and each message's post (or store issue), and sums per
// warp: route, load (TMA load wait), smem (wait for the previous token's stores to free shared
// memory), stage (store into the send buffer and wait), post (DeepEP put calls), store (TMA
// store issue). --load runs a cuBLAS GEMM on the SMs the dispatch does not use.
// Output: one CSV row per run; summarize_bd.py takes medians and prints the breakdown.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <string>
#include <unistd.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "compiled.cuh"
#include "ibgda_device_nv365.cuh"   // DeepEP V1's ibgda_device.cuh, QP index ported to NVSHMEM 3.6.5

namespace dl = deep_ep::legacy;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m; cuGetErrorString(r_, &m); \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, m); exit(1); } } while (0)

static const int R = 8, E = 256, TOPK = 8, HMAX = 7168, TMAX = 4096, WARPS = 8, MAXCTA = 64;
static const int SMEM = WARPS * HMAX;   // one token buffer per warp
enum { LOCAL = 0, FLUSH = 1, ORDERED = 2 };
static const char* PATH_NAME[] = {"local", "flush", "ordered"};
enum { A_ROUTE, A_LOAD, A_SMEM, A_STAGE, A_POST, A_STORE, NACT };

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// A timer read waits for nothing it does not depend on: after BAR.SYNC.DEFER_BLOCKING (what
// __syncthreads compiles to) a warp reads the timer when it reaches the barrier, not when the
// barrier completes (measured: stamps up to 31 us early). These read it under a predicate on
// a value that only exists once the barrier (bar.red) or the memory operation has completed.
__device__ __forceinline__ unsigned long long gtimer_dep(int dep) {
  unsigned long long t = 0;
  asm volatile("{\n .reg .pred p;\n setp.ne.s32 p, %1, -1;\n @p mov.u64 %0, %%globaltimer;\n}" : "+l"(t) : "r"(dep));
  return t;
}
__device__ __forceinline__ unsigned long long sync_gtimer() { return gtimer_dep(__syncthreads_count(1)); }

// routing shared with dispatch_ibgda.cu / dispatch_ce.cu
__host__ __device__ __forceinline__ unsigned hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__host__ __device__ unsigned rank_mask(int t) {
  unsigned mask = 0; int got = 0; unsigned long long used[4] = {0, 0, 0, 0};
  for (int j = 0; got < TOPK; j++) {
    int e = hash(t * 131 + j) % E;
    if (used[e >> 6] >> (e & 63) & 1) continue;
    used[e >> 6] |= 1ull << (e & 63); got++;
    mask |= 1u << (e / (E / R));
  }
  return mask;
}

// TMA (as tma_bw.cu) and memory-order helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(b)));
}
__device__ __forceinline__ void tma_load(void* dst_smem, const void* src, unsigned bytes, unsigned long long* b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst_smem)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned phase) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
               :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_store(void* dst, const void* src_smem, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(smem_u32(src_smem)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void tma_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void tma_wait() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void st_release_sys(int* p, int v) { asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int ld_acquire_sys(const int* p) { int v; asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned long long ld_volatile(const void* p) {
  unsigned long long v; asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}

// DeepEP V1's put (nvshmemi_ibgda_put_nbi_warp) for one thread: V2.5 posts a token's puts from
// parallel lanes (gin.put per lane in impls/ep/dispatch.cuh), V1's post path only has a warp put.
// Same helpers, same WQE, same doorbell batching; a message may span registration chunks (<= 3 WQEs).
__device__ __forceinline__ void put_nbi_thread(uint64_t rptr, uint64_t lptr, size_t bytes, int pe, int qp_id, int message_idx) {
  auto qp = dl::ibgda_get_rc(pe, qp_id);
  __be32 lkey[3], rkey[3]; uint64_t laddr[3], raddr[3], len[3]; int n = 0;
  while (bytes > 0) {
    laddr[n] = lptr;
    len[n] = min((uint64_t)bytes, dl::ibgda_get_lkey_and_rkey(lptr, &lkey[n], rptr, pe, &raddr[n], &rkey[n], qp->dev_idx));
    bytes -= len[n]; lptr += len[n]; rptr += len[n]; n++;
  }
  const uint64_t base = dl::ibgda_reserve_wqe_slots(qp, n);
  for (int i = 0; i < n; i++) {
    void* wqe = dl::ibgda_get_wqe_ptr(qp, base + i);
    dl::ibgda_write_rdma_write_wqe(qp, laddr[i], lkey[i], raddr[i], rkey[i], len[i], base + i, &wqe);
  }
  dl::ibgda_submit_requests<false>(qp, base, n, message_idx);
}

struct Stamps { unsigned long long *cta_start, *cta_loop, *cta_drain, *cta_exit, *signal, *warp, *post, *wend; int* who; };

template <int kPath>
__global__ void __launch_bounds__(256, 1)
dispatch(const char* tok, char* send, char* recv, int* sig, int T, int H, int* counters, int* ctas_done, Stamps st,
         int qp_warp, int nq, int nqp, int post_lane) {
  extern __shared__ __align__(128) char smem[];
  __shared__ __align__(8) unsigned long long bar[WARPS];
  __shared__ int last, fence_dep;   // fence_dep: a word to load after a fence, so the timer waits for it
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int gw = blockIdx.x * WARPS + w, nw = gridDim.x * WARPS;
  const int first = w * gridDim.x + blockIdx.x;   // tokens go round the SMs first, as DeepEP V2.5
  if (threadIdx.x == 0) { st.cta_start[blockIdx.x] = gtimer(); fence_dep = 1; }
  __syncthreads();
  char* buf = smem + (size_t)w * HMAX;
  if (lane == 0) mbar_init(&bar[w]);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncwarp();
  unsigned phase = 0;
  int nmsg = 0;                      // this warp's messages (doorbell batching with a QP per warp)
  unsigned long long acc[NACT] = {};
  for (int t = first; t < T; t += nw) {
    const unsigned long long a = gtimer();
    tma_wait_read(); __syncwarp();   // the previous token's stores have read shared memory
    const unsigned long long b = gtimer(); acc[A_SMEM] += b - a;
    if (lane == 0) tma_load(buf, tok + (size_t)t * H, H, &bar[w]);
    unsigned mask = 0;
    if (lane == 0) mask = rank_mask(t);
    mask = __shfl_sync(0xffffffff, mask, 0);
    int slot = -1;                   // lane r < R: this token's slot at destination r
    if (lane < R && (mask >> lane & 1)) slot = atomicAdd(&counters[lane], 1);
    __syncwarp();
    const unsigned long long c = gtimer(); acc[A_ROUTE] += c - b;
    if (lane == 0) mbar_wait(&bar[w], phase);
    phase ^= 1; __syncwarp();
    const unsigned long long d = gtimer(); acc[A_LOAD] += d - c;
    if (kPath == LOCAL) {
      if (slot >= 0) {
        tma_store(recv + ((size_t)lane * T + slot) * H, buf, H);
        st.post[lane * T + slot] = gtimer(); st.who[lane * T + slot] = gw;
      }
      __syncwarp();
      acc[A_STORE] += gtimer() - d;
    } else {
      char* s = send + (size_t)t * H;
      if (lane == 0) { tma_store(s, buf, H); tma_wait(); }
      __syncwarp();
      const unsigned long long e = gtimer(); acc[A_STAGE] += e - d;
      if (post_lane) {               // V2.5: each lane posts its destination's put, in parallel
        if (slot >= 0) {
          const int qp = qp_warp ? gw % nqp : lane * nq + slot % nq;
          const int idx = qp_warp ? nmsg + __popc(mask & ((1u << lane) - 1)) : slot / nq;
          put_nbi_thread((uint64_t)(recv + ((size_t)lane * T + slot) * H), (uint64_t)s, H, 1, qp, idx);
          st.post[lane * T + slot] = gtimer(); st.who[lane * T + slot] = gw;
        }
        nmsg += __popc(mask);
        __syncwarp();
      } else for (unsigned m = mask; m; m &= m - 1) {   // one warp put per destination, in turn
        const int r = __ffs(m) - 1, sl = __shfl_sync(0xffffffff, slot, r);
        // QP per warp (V2.5: QPs follow SMs) or nq QPs per destination (V1 low latency: per local expert)
        const int qp = qp_warp ? gw % nqp : r * nq + sl % nq, idx = qp_warp ? nmsg++ : sl / nq;
        dl::nvshmemi_ibgda_put_nbi_warp<false>((uint64_t)(recv + ((size_t)r * T + sl) * H), (uint64_t)s, H, 1, qp, lane, idx);
        if (lane == 0) { st.post[r * T + sl] = gtimer(); st.who[r * T + sl] = gw; }
      }
      acc[A_POST] += gtimer() - e;
    }
  }
  if (lane == 0) { for (int i = 0; i < NACT; i++) st.warp[(size_t)gw * NACT + i] = acc[i]; st.wend[gw] = gtimer(); }
  unsigned long long t = sync_gtimer();   // every warp has left the loop
  if (threadIdx.x == 0) st.cta_loop[blockIdx.x] = t;
  if (kPath == LOCAL) {
    tma_wait();                      // this thread's stores are complete
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); t = gtimer_dep(*(volatile int*)&fence_dep); }
  } else if (kPath == FLUSH) {
    for (int q = threadIdx.x; q < nqp; q += blockDim.x) {   // ring every QP for what is left, wait for all its completions
      auto qp = dl::ibgda_get_rc(1, q);
      dl::ibgda_post_send(qp, dl::ld_na_relaxed(&qp->mvars.tx_wq.ready_head));
      dl::nvshmemi_ibgda_quiet(1, q);
    }
    t = sync_gtimer();
  }
  if (threadIdx.x == 0) {
    st.cta_drain[blockIdx.x] = t;
    __threadfence();
    last = atomicAdd(ctas_done, 1) == gridDim.x - 1;
  }
  __syncthreads();
  if (last) {                        // every CTA has issued (local, ordered) or flushed (flush) its writes
    if (kPath == LOCAL) {
      if (threadIdx.x < R) st_release_sys(sig + threadIdx.x, 1);   // in parallel, as DeepEP's barrier
    } else if (kPath == FLUSH && threadIdx.x < R) {   // the data has landed: one signal per destination
      dl::nvshmemi_ibgda_amo_nonfetch_add(sig + threadIdx.x, 1, 1, threadIdx.x % nqp);
    } else if (kPath == ORDERED && threadIdx.x < R * nq) {   // one signal per QP, behind that QP's data
      __threadfence();
      dl::nvshmemi_ibgda_amo_nonfetch_add(sig + threadIdx.x, 1, 1, threadIdx.x);
    }
    t = sync_gtimer();
    if (threadIdx.x == 0) *st.signal = t;
  }
  t = sync_gtimer();
  if (threadIdx.x == 0) st.cta_exit[blockIdx.x] = t;
}

// after a remote run, outside the timing: reap the completions left (the signal's, and ordered's data)
__global__ void reap(int nqp) {
  if ((int)threadIdx.x < nqp) {
    auto qp = dl::ibgda_get_rc(1, threadIdx.x);
    dl::ibgda_post_send(qp, dl::ld_na_relaxed(&qp->mvars.tx_wq.ready_head));
    dl::nvshmemi_ibgda_quiet(1, threadIdx.x);
  }
}

__global__ void set_nonce(char* tok, int T, int H, unsigned long long nonce) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < T) *(unsigned long long*)(tok + (size_t)t * H + H - 8) = nonce;
}

__global__ void __launch_bounds__(1024, 1)
receiver(const char* recv, const int* sig, int ns, int T, int H, const int* cnt, unsigned long long nonce, unsigned long long* arr,
         unsigned long long* sig_ts, int* missing, int* timed_out, volatile int* ready_h, int ready_val) {
  __shared__ int pre[R + 1];
  if (threadIdx.x == 0) { pre[0] = 0; for (int r = 0; r < R; r++) pre[r + 1] = pre[r] + cnt[r]; }
  __syncthreads();
  const int M = pre[R], NP = blockDim.x - 32, me = (int)threadIdx.x - 32;
  unsigned todo = 0;                 // bit k: message me + k * NP (at most 22 at 4096 tokens)
  if (me >= 0) for (int k = 0; me + k * NP < M; k++) todo |= 1u << k;
  bool sig_todo = (int)threadIdx.x < ns;   // ns signals: signal j covers destination j / g, slots = j mod g
  const int g = ns / R;
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence_system(); *ready_h = ready_val; }
  const unsigned long long t0 = gtimer();
  while (todo || sig_todo) {
    for (unsigned m = todo; m; m &= m - 1) {
      const int k = __ffs(m) - 1, i = me + k * NP;
      int r = 0; while (i >= pre[r + 1]) r++;
      const size_t idx = (size_t)r * T + (i - pre[r]);
      if (ld_volatile(recv + idx * H + H - 8) == nonce) { arr[idx] = gtimer(); todo &= ~(1u << k); }
    }
    if (sig_todo && ld_acquire_sys(sig + threadIdx.x) >= 1) {
      sig_ts[threadIdx.x] = gtimer(); sig_todo = false;
      const int r = threadIdx.x / g; int miss = 0;   // the messages this signal covers, not visible when it is
      for (int s = threadIdx.x % g; s < cnt[r]; s += g) miss += ld_volatile(recv + ((size_t)r * T + s) * H + H - 8) != nonce;
      missing[threadIdx.x] = miss;
    }
    if (gtimer() - t0 > 2000000000ull) { atomicExch(timed_out, 1); break; }
  }
}

struct Load {  // background GEMM on another stream (same process), planned for 132 - reserve SMs
  std::atomic<bool> stop{false}, ready{false}; std::atomic<int> reserve{0}; std::thread th; std::vector<int> warm;
  void start() {
    th = std::thread([this] {
      cublasHandle_t h; cublasCreate(&h); cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); cublasSetStream(h, s);
      void* ws; cudaMalloc(&ws, 256ull << 20); cublasSetWorkspace(h, ws, 256ull << 20);
      const int M = 8192, N = 4096, K = 7168; __nv_bfloat16 *A, *B, *C;
      cudaMalloc(&A, (size_t)M * K * 2); cudaMalloc(&B, (size_t)K * N * 2); cudaMalloc(&C, (size_t)M * N * 2);
      cudaMemset(A, 0x3c, (size_t)M * K * 2); cudaMemset(B, 0x3c, (size_t)K * N * 2);
      const float al = 1.f, be = 0.f;
      // run every SM target once before any dispatch: a GEMM kernel first used mid-sweep is
      // loaded then, that load waits for an idle device, and the receiver spins (see dispatch_ce.cu)
      for (int r : warm) {
        cublasSetSmCountTarget(h, 132 - r);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }
      cudaStreamSynchronize(s); ready = true;
      while (!stop) {
        cublasSetSmCountTarget(h, 132 - reserve);
        for (int i = 0; i < 10; i++)
          cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &be, C, CUDA_R_16BF, N,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaStreamSynchronize(s);
      }
    });
  }
};

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v; std::string x(s); size_t p = 0;
  while (p < x.size()) { size_t q = x.find(',', p); if (q == std::string::npos) q = x.size(); v.push_back(atoi(x.substr(p, q - p).c_str())); p = q + 1; }
  return v;
}

int main(int argc, char** argv) {
  int H = 7168, path = LOCAL, iters = 20, qp_warp = 0, nq = 1, post_lane = 0; bool load = false;
  std::vector<int> Ts = {16, 32, 128, 1024, 4096}, CTAs = {20};
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--H")) H = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--load")) load = true;
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tokens")) Ts = parse_list(argv[++i]);
    else if (!strcmp(argv[i], "--ctas")) CTAs = parse_list(argv[++i]);
    else if (!strcmp(argv[i], "--qp")) qp_warp = !strcmp(argv[++i], "warp");   // warp | dest
    else if (!strcmp(argv[i], "--nq")) nq = atoi(argv[++i]);                     // QPs per destination (--qp dest)
    else if (!strcmp(argv[i], "--post")) post_lane = !strcmp(argv[++i], "lane");  // lane (V2.5) | warp (one warp put at a time)
    else if (!strcmp(argv[i], "--path")) { const char* p = argv[++i]; path = !strcmp(p, "flush") ? FLUSH : !strcmp(p, "ordered") ? ORDERED : LOCAL; }
  }
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);   // no lazy kernel load while the receiver spins
  nvshmem_init();
  const int nqp = qp_warp ? (getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE") ? atoi(getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE")) : 2) : R * nq;
  const int ns = path == ORDERED ? R * nq : R;   // signal words
  if ((path == ORDERED && qp_warp) || ns > 32 || nqp > 32) { fprintf(stderr, "bad QP setup\n"); exit(1); }
  const int me = nvshmem_my_pe();
  CK(cudaSetDevice(0));
  char* tok = (char*)nvshmem_malloc((size_t)TMAX * HMAX);
  char* send = (char*)nvshmem_malloc((size_t)TMAX * HMAX);       // staging: one slot per token
  char* recv = (char*)nvshmem_malloc((size_t)R * TMAX * HMAX);   // destination r, slot s at (r * T + s) * H
  int* sig = (int*)nvshmem_malloc(256);                           // R signal words
  const char* ipc_file = "/tmp/loom_dispatch_bd.ipc";
  const char* done_file = "/tmp/loom_dispatch_bd.done";
  if (me == 0) unlink(done_file);
  if (me == 1) {   // export the heap allocation holding recv and sig to PE 0
    CUdeviceptr base; size_t size; CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)recv));
    if ((CUdeviceptr)sig < base || (CUdeviceptr)sig + 256 > base + size) { fprintf(stderr, "recv and sig in different allocations\n"); exit(1); }
    cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, (void*)base));
    size_t off[2] = {(CUdeviceptr)recv - base, (CUdeviceptr)sig - base};
    FILE* f = fopen(ipc_file, "wb"); fwrite(&h, sizeof h, 1, f); fwrite(off, sizeof off, 1, f); fclose(f);
  }
  nvshmem_barrier_all();
  if (me != 0) {   // PE 1 is only memory; keep it off the GPU (no MPS: processes would time-slice)
    while (access(done_file, F_OK) != 0) usleep(100000);
    nvshmem_barrier_all(); nvshmem_finalize(); return 0;
  }
  char* recv_rx; int* sig_rx;   // PE 1's receive buffer and signals, as PE 0's receiver sees them
  {
    cudaIpcMemHandle_t h; size_t off[2]; FILE* f = fopen(ipc_file, "rb");
    if (!f || fread(&h, sizeof h, 1, f) != 1 || fread(off, sizeof off, 1, f) != 1) { fprintf(stderr, "no IPC handle from PE 1\n"); exit(1); }
    fclose(f);
    void* p; CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
    recv_rx = (char*)p + off[0]; sig_rx = (int*)((char*)p + off[1]);
  }
  char* recv_local; int* sig_local;
  CK(cudaMalloc(&recv_local, (size_t)R * TMAX * HMAX)); CK(cudaMalloc(&sig_local, 256));
  char* const recv_tx = path == LOCAL ? recv_local : recv;     // the sender's view
  int* const sig_tx = path == LOCAL ? sig_local : sig;
  char* const recv_obs = path == LOCAL ? recv_local : recv_rx; // the receiver's view
  int* const sig_obs = path == LOCAL ? sig_local : sig_rx;
  CK(cudaMemset(tok, 1, (size_t)TMAX * HMAX));
  CK(cudaMemset(recv_obs, 0, (size_t)R * TMAX * HMAX));

  Stamps st; unsigned long long *arr, *sig_ts; int *counters, *ctas_done, *cnt_d, *missing, *timed_out;
  CK(cudaMalloc(&st.cta_start, MAXCTA * 8)); CK(cudaMalloc(&st.cta_loop, MAXCTA * 8)); CK(cudaMalloc(&st.cta_drain, MAXCTA * 8));
  CK(cudaMalloc(&st.cta_exit, MAXCTA * 8)); CK(cudaMalloc(&st.signal, 8)); CK(cudaMalloc(&st.warp, MAXCTA * WARPS * NACT * 8));
  CK(cudaMalloc(&st.post, (size_t)R * TMAX * 8)); CK(cudaMalloc(&st.who, (size_t)R * TMAX * 4)); CK(cudaMalloc(&st.wend, MAXCTA * WARPS * 8)); CK(cudaMalloc(&arr, (size_t)R * TMAX * 8)); CK(cudaMalloc(&sig_ts, 32 * 8));
  CK(cudaMalloc(&counters, R * 4)); CK(cudaMalloc(&ctas_done, 4)); CK(cudaMalloc(&cnt_d, R * 4));
  CK(cudaMalloc(&missing, 32 * 4)); CK(cudaMalloc(&timed_out, 4));
  int *ready_h, *ready_d; CK(cudaHostAlloc((void**)&ready_h, 64, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&ready_d, ready_h, 0));
  *ready_h = 0;
  CK(cudaFuncSetAttribute(dispatch<LOCAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(dispatch<FLUSH>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(dispatch<ORDERED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  cudaStream_t ks, rs; CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));
  cudaEvent_t ea, eb; CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  CK(cudaDeviceSynchronize());
  Load L; if (load) { for (int c : CTAs) L.warm.push_back(c + 1); L.start(); while (!L.ready) usleep(1000); usleep(500000); }

  printf("# dispatch breakdown, path=%s H=%d E=%d R=%d topk=%d load=%d, 256-thread CTAs, QPs: %s (%d), %d signals; times in us from the sender's first CTA start\n",
         PATH_NAME[path], H, E, R, TOPK, (int)load, path == LOCAL ? "none" : qp_warp ? "per warp" : "per destination", path == LOCAL ? 0 : nqp, ns);
  printf("path,qp,nqp,H,load,tokens,ctas,run,msgs,event_us,loop_end_us,post_last_us,arr_first_us,arr_last_us,drain_end_us,signal_us,"
         "seen_us,oneway_med_us,oneway_p90_us,missing,cta_loop_us,cta_drain_us,cta_tail_us,"
         "w_route_us,w_load_us,w_smem_us,w_stage_us,w_post_us,w_store_us\n");
  unsigned long long nonce = 0; int ready_val = 0;
  for (int T : Ts) {
    int cnt[R] = {0}, M = 0;
    for (int t = 0; t < T; t++) for (unsigned m = rank_mask(t); m; m &= m - 1) { cnt[__builtin_ctz(m)]++; M++; }
    CK(cudaMemcpy(cnt_d, cnt, sizeof cnt, cudaMemcpyHostToDevice));
    for (int ctas : CTAs) {
      L.reserve = ctas + 1;   // the dispatch and the receiver
      for (int it = 0; it < iters + 3; it++) {
        nonce++; ready_val++;
        CK(cudaMemset(counters, 0, R * 4)); CK(cudaMemset(ctas_done, 0, 4)); CK(cudaMemset(sig_obs, 0, 256));
        CK(cudaMemset(timed_out, 0, 4)); CK(cudaMemset(arr, 0, (size_t)R * TMAX * 8)); CK(cudaMemset(sig_ts, 0, 32 * 8));
        set_nonce<<<(T + 255) / 256, 256, 0, ks>>>(tok, T, H, nonce);
        CK(cudaDeviceSynchronize());
        receiver<<<1, 1024, 0, rs>>>(recv_obs, sig_obs, ns, T, H, cnt_d, nonce, arr, sig_ts, missing, timed_out, ready_d, ready_val);
        for (long spin = 0; *(volatile int*)ready_h != ready_val; spin++)
          if (spin > 2000000000L) { fprintf(stderr, "receiver did not start\n"); return 1; }
        CK(cudaEventRecord(ea, ks));
        if (path == LOCAL) dispatch<LOCAL><<<ctas, 256, SMEM, ks>>>(tok, send, recv_tx, sig_tx, T, H, counters, ctas_done, st, qp_warp, nq, nqp, post_lane);
        else if (path == FLUSH) dispatch<FLUSH><<<ctas, 256, SMEM, ks>>>(tok, send, recv_tx, sig_tx, T, H, counters, ctas_done, st, qp_warp, nq, nqp, post_lane);
        else dispatch<ORDERED><<<ctas, 256, SMEM, ks>>>(tok, send, recv_tx, sig_tx, T, H, counters, ctas_done, st, qp_warp, nq, nqp, post_lane);
        CK(cudaEventRecord(eb, ks));
        CK(cudaStreamSynchronize(ks)); CK(cudaStreamSynchronize(rs)); CK(cudaGetLastError());
        if (path != LOCAL) { reap<<<1, 32, 0, ks>>>(nqp); CK(cudaStreamSynchronize(ks)); }
        int to; CK(cudaMemcpy(&to, timed_out, 4, cudaMemcpyDeviceToHost));
        if (to) { fprintf(stderr, "receiver timed out (path %s T %d ctas %d run %d)\n", PATH_NAME[path], T, ctas, it); return 1; }
        int got[R]; CK(cudaMemcpy(got, counters, sizeof got, cudaMemcpyDeviceToHost));
        for (int r = 0; r < R; r++) if (got[r] != cnt[r]) { fprintf(stderr, "destination %d got %d messages, expected %d\n", r, got[r], cnt[r]); return 1; }
        if (it < 3) continue;
        // copy the stamps back and reduce them
        std::vector<unsigned long long> cs(ctas), cl(ctas), cd(ctas), ce(ctas), wa((size_t)ctas * WARPS * NACT),
            P((size_t)R * T), A((size_t)R * T), Z(ns);
        unsigned long long sg; int miss[32];
        CK(cudaMemcpy(cs.data(), st.cta_start, ctas * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cl.data(), st.cta_loop, ctas * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cd.data(), st.cta_drain, ctas * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ce.data(), st.cta_exit, ctas * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(wa.data(), st.warp, wa.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&sg, st.signal, 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(P.data(), st.post, P.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(A.data(), arr, A.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Z.data(), sig_ts, ns * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(miss, missing, ns * 4, cudaMemcpyDeviceToHost));
        float ev; CK(cudaEventElapsedTime(&ev, ea, eb));
        const unsigned long long k0 = *std::min_element(cs.begin(), cs.end());
        auto us = [&](unsigned long long t) { return ((double)t - (double)k0) / 1e3; };
        unsigned long long p_last = 0, a_first = ~0ull, a_last = 0; std::vector<double> ow; ow.reserve(M);
        for (int r = 0; r < R; r++)
          for (int s = 0; s < cnt[r]; s++) {
            const size_t i = (size_t)r * T + s;
            p_last = std::max(p_last, P[i]); a_first = std::min(a_first, A[i]); a_last = std::max(a_last, A[i]);
            ow.push_back(((double)A[i] - (double)P[i]) / 1e3);
          }
        std::sort(ow.begin(), ow.end());
        if (getenv("BD_DEBUG") && it == 3) {
          std::vector<int> who((size_t)R * T); CK(cudaMemcpy(who.data(), st.who, who.size() * 4, cudaMemcpyDeviceToHost));
          for (int c = 0; c < ctas; c++) fprintf(stderr, "cta %d start %.2f loop %.2f drain %.2f exit %.2f\n", c, us(cs[c]), us(cl[c]), us(cd[c]), us(ce[c]));
          std::vector<unsigned long long> we((size_t)ctas * WARPS); CK(cudaMemcpy(we.data(), st.wend, we.size() * 8, cudaMemcpyDeviceToHost));
          for (size_t g = 0; g < we.size(); g++) fprintf(stderr, "warp %zu cta %zu end %.2f\n", g, g / WARPS, us(we[g]));
          for (int r = 0; r < R; r++) for (int s2 = 0; s2 < cnt[r]; s2++) { size_t i = (size_t)r * T + s2;
            fprintf(stderr, "msg r%d s%d warp %d post %.2f arr %.2f\n", r, s2, who[i], us(P[i]), us(A[i])); }
        }
        double c_loop = 0, c_drain = 0, c_tail = 0, wsum[NACT] = {0};
        for (int c = 0; c < ctas; c++) { c_loop += (cl[c] - cs[c]) / 1e3; c_drain += (cd[c] - cl[c]) / 1e3; c_tail += (ce[c] - cd[c]) / 1e3; }
        for (size_t i = 0; i < wa.size(); i++) wsum[i % NACT] += wa[i] / 1e3;
        int misses = 0; for (int j = 0; j < ns; j++) misses += miss[j];
        printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
               PATH_NAME[path], path == LOCAL ? "none" : qp_warp ? (post_lane ? "warpL" : "warp") : (post_lane ? "destL" : "dest"), path == LOCAL ? 0 : nqp, H, (int)load, T, ctas, it - 3, M, ev * 1e3,
               us(*std::max_element(cl.begin(), cl.end())), us(p_last), us(a_first), us(a_last),
               us(*std::max_element(cd.begin(), cd.end())), us(sg), us(*std::max_element(Z.begin(), Z.end())),
               ow[ow.size() / 2], ow[ow.size() * 9 / 10], misses, c_loop, c_drain, c_tail,
               wsum[A_ROUTE], wsum[A_LOAD], wsum[A_SMEM], wsum[A_STAGE], wsum[A_POST], wsum[A_STORE]);
        fflush(stdout);
      }
    }
  }
  if (load) { L.stop = true; L.th.join(); }
  { FILE* f = fopen(done_file, "w"); if (f) fclose(f); }
  nvshmem_barrier_all();
  nvshmem_finalize();
  return 0;
}
