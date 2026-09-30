// M2: sizeof checks for NCCL GIN GDAKI (vendored DOCA GPUNetIO) device-visible structs.
#include <cstdio>
#include <cstddef>
#include "doca_gpunetio_verbs_dev.h"
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
#define P(T) printf("%-44s %6zu\n", #T, sizeof(T))
int main() {
    P(struct doca_gpu_dev_verbs_qp);
    P(struct doca_gpu_dev_verbs_cq);
    P(struct doca_gpu_dev_verbs_wqe);
    P(struct doca_gpunetio_ib_mlx5_cqe64);
    P(struct ncclGinGdakiGPUContext);
    P(struct ncclGinGdakiMemHandle);
    printf("  offsetof(qp, cq_sq) = %zu\n", offsetof(struct doca_gpu_dev_verbs_qp, cq_sq));
    return 0;
}
