# NVSHMEM on steve: 2 PEs on the one H200, GPU-initiated RDMA (IBGDA) over the
# CX-7 port0 <-> port1 loopback cable. Source this, then launch with mpiexec -n 2.
export NVSHMEM_HOME=$HOME/loom-experiments/nvshmem-3.6.5
export CUDA_VISIBLE_DEVICES=0
export NVSHMEM_BOOTSTRAP=PMI
export NVSHMEM_DISABLE_P2P=1                      # same GPU: force traffic through the NIC
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_IB_ENABLE_IBGDA=1                  # GPU-initiated (IBGDA) remote transport
export NVSHMEM_IBGDA_NIC_HANDLER=gpu              # the GPU rings the doorbell (needs PeerMappingOverride=1)
export NVSHMEM_HCA_PE_MAPPING=mlx5_0:1:1,mlx5_1:1:1   # PE 0 -> port 0, PE 1 -> port 1
export NVSHMEM_IB_GID_INDEX=3                     # RoCE v2, IPv4 link-local
export NVSHMEM_SYMMETRIC_SIZE=2G
export LD_LIBRARY_PATH=/nix/store/6f9hdh53xy45pdybibsz28z5iwxji35a-rdma-core-62.0/lib:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # nixpkgs rdma-core (libibverbs, libmlx5) for dlopen
export NVSHMEM_DISABLE_NCCL=1
