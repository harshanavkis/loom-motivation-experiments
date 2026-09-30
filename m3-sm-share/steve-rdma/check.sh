#!/usr/bin/env bash
# Run on steve (no root). Prints whether every prerequisite for the GPU RDMA
# experiments is in place; see README.md for how to fix each one.
set -uo pipefail
NIC0=0000:95:00.0; NIC1=0000:95:00.1; GPU=0000:15:00.0
ok() { printf '  %-44s %s\n' "$1" "$2"; }
grp() { basename "$(readlink /sys/bus/pci/devices/$1/iommu_group)"; }

echo "links (CX-7 port0 <-> port1 loopback cable):"
for d in mlx5_0 mlx5_1; do ok "$d state" "$(cat /sys/class/infiniband/$d/ports/1/state) $(cat /sys/class/infiniband/$d/ports/1/rate)"; done
for d in mlx5_0 mlx5_1; do ok "$d GID 3" "$(cat /sys/class/infiniband/$d/ports/1/gids/3)"; done

echo "IOMMU groups (want: identity for all three):"
for d in $NIC0 $NIC1 $GPU; do ok "$d group $(grp $d)" "$(cat /sys/kernel/iommu_groups/$(grp $d)/type)"; done

echo "NVIDIA driver (want: PeerMappingOverride=1):"
ok "RegistryDwords" "$(grep -o '"[^"]*"' <(grep '^RegistryDwords:' /proc/driver/nvidia/params))"
ok "GPU" "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
ok "GPU compute processes" "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"

echo "limits:"
ok "memlock (ulimit -l, KiB)" "$(ulimit -l)  (NVSHMEM needs unlimited -> its runner uses sudo)"

echo "built artifacts:"
H=$HOME/loom-experiments
for f in gpu-interference/interfere perftest-cuda/install/bin/ib_write_bw nvshmem-3.6.5/bin/perftest/device/pt-to-pt/shmem_put_bw; do
  [ -x "$H/$f" ] && ok "$f" "present" || ok "$f" "MISSING (run build.sh)"
done
