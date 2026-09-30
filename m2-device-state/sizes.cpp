// M2: sizeof/offsetof checks for NVSHMEM IBGDA device-visible transport state.
// Compiled on the host with g++ against the cloned NVSHMEM headers + rdma-core mlx5dv.h
// (no GPU needed). See run_sizes.sh.
#include <cstddef>
#include <cstdio>
#include <infiniband/mlx5dv.h>
#include "device_host_transport/nvshmem_common_ibgda.h"
#include "device_host_transport/nvshmem_common_batch_rma_pending_qps.hpp"

#define P(T) printf("%-52s %6zu\n", #T, sizeof(T))
#define O(T, f) printf("  offsetof(%s, %s) = %zu\n", #T, #f, offsetof(T, f))

int main() {
    printf("== NVSHMEM IBGDA device structs (nvshmem_common_ibgda.h) ==\n");
    P(nvshmemi_ibgda_device_qp_t);
    P(nvshmemi_ibgda_device_qp_management_t);
    P(nvshmemi_ibgda_device_cq_t);
    P(nvshmemi_ibgda_device_dct_t);  // == struct mlx5_wqe_av
    P(nvshmemi_ibgda_device_key_t);
    P(nvshmemi_ibgda_device_local_only_mhandle_t);
    P(nvshmemi_ibgda_device_state_t);
    O(nvshmemi_ibgda_device_state_t, constmem);
    O(nvshmemi_ibgda_device_state_t, globalmem);
    printf("  sizeof(state.constmem) = %zu\n", sizeof(((nvshmemi_ibgda_device_state_t *)0)->constmem));
    O(nvshmemi_ibgda_device_qp_t, tx_wq);
    O(nvshmemi_ibgda_device_qp_t, mvars);

    printf("== mlx5 WQE / CQE building blocks (rdma-core mlx5dv.h) ==\n");
    P(struct mlx5_wqe_ctrl_seg);
    P(struct mlx5_wqe_raddr_seg);
    P(struct mlx5_wqe_data_seg);
    P(struct mlx5_wqe_atomic_seg);
    P(struct mlx5_wqe_inl_data_seg);
    P(struct mlx5_wqe_av);
    P(struct mlx5_cqe64);
    printf("  MLX5_SEND_WQE_BB = %d\n", (int)MLX5_SEND_WQE_BB);

    printf("== NVSHMEM 3.8 batch-RMA pending-QP bitmap: bytes for qp_count, 4096 slots ==\n");
    unsigned qps[] = {1 + 2 * 2, 1 + 24 * 2, 1 + 24 * 16, 1 + 24 * 32, 1 + 16 * 16, 1 + 2 * 128, 1 + 1 * 256};
    for (unsigned q : qps) {
        size_t sz = 0;
        nvshmemi_batch_rma_pending_qps_storage_size(q, 4096, &sz);
        printf("  qp_count=%5u -> %zu B\n", q, sz);
    }
    return 0;
}
