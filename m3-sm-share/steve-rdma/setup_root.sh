#!/usr/bin/env bash
# Run on steve as root, once after every reboot (nothing here persists):
#   sudo ~/loom-experiments/steve-rdma/setup_root.sh
# 1. NVIDIA driver option PeerMappingOverride=1: lets the GPU map the NIC doorbell (IBGDA).
# 2. IOMMU passthrough (identity) for the GPU and both CX-7 functions: GPUDirect RDMA
#    (NIC <-> HBM) and GPU -> NIC doorbell writes otherwise fault in the IOMMU (DMAR).
# Undo: reboot (or write DMA-FQ back to each group and remove /run/modprobe.d/nvidia-peermapping.conf).
set -euo pipefail
NICS="0000:95:00.0 0000:95:00.1"; GPU=0000:15:00.0
grp() { basename "$(readlink /sys/bus/pci/devices/$1/iommu_group)"; }
typ() { cat /sys/kernel/iommu_groups/$(grp $1)/type; }

systemctl stop ollama || true

# 1. driver option. A udev rule restarts nvidia-ctk (CDI generator) on every nvidia module
#    event and that reloads the driver behind our back, so the option must live in a
#    modprobe config file rather than on the modprobe command line.
mkdir -p /run/modprobe.d
echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"' > /run/modprobe.d/nvidia-peermapping.conf
if ! grep -q 'PeerMappingOverride=1' /proc/driver/nvidia/params; then
  rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true
  modprobe nvidia
fi

# 2a. GPU group -> identity (the device must be unbound from the driver meanwhile)
if [ "$(typ $GPU)" != identity ]; then
  rmmod nvidia_drm nvidia_modeset 2>/dev/null || true
  echo $GPU > /sys/bus/pci/drivers/nvidia/unbind
  echo identity > /sys/kernel/iommu_groups/$(grp $GPU)/type
  echo $GPU > /sys/bus/pci/drivers/nvidia/bind
fi
modprobe nvidia_uvm; modprobe nvidia_modeset; modprobe nvidia_drm

# 2b. CX-7 groups -> identity
for d in $NICS; do
  if [ "$(typ $d)" != identity ]; then
    echo $d > /sys/bus/pci/drivers/mlx5_core/unbind
    echo identity > /sys/kernel/iommu_groups/$(grp $d)/type
    echo $d > /sys/bus/pci/drivers/mlx5_core/bind
  fi
done

systemctl start ollama || true
sleep 5   # links come back up after the mlx5 rebind
sudo -u harshanavkis /home/harshanavkis/loom-experiments/steve-rdma/check.sh
