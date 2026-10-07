# M2 — NIC transport state kept on/for the GPU by GPU-initiated networking (from source)

All numbers are from source code. Struct sizes were measured with `sizes.cpp` (g++ 15.2 from
`nix shell nixpkgs#gcc`, NVSHMEM headers from the clone, rdma-core-64 `mlx5dv.h`; output in
`sizes.out`). The configuration arithmetic is in `totals.py` (output in `totals.out`). No GPU or NIC was touched.

## 0. Versions

| System | Ref | Commit | Date |
|---|---|---|---|
| NVSHMEM | tag `v3.8.0-0` (newest release tag) | `270759e5481b16ef5a71930e1d9b8df184cd7072` | 2026-09-22 |
| NCCL | tag `v2.32.3-1` (newest release tag) | `12df1a11afad322be5a204a2db890161cbf8131d` | 2026-09-17 |
| DeepEP main (V2.5, NCCL GIN backend) | `main` HEAD | `93eb6eb238127e96c6d7a4a625a6dad158348509` | 2026-09-30 |
| DeepEP V1 (last NVSHMEM/IBGDA code) | `def8651^` = parent of "DeepEP V2.5 (#763)", the commit that removed V1 | `a56d6156febcd9976e55adc85b5155bfac9f28f8` | 2026-09-16 |
| (DeepEP last V1 *tag*, for reference) | `v1.2.1` | `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` | 2025-09-15 |

Clones are in `../src/{nvshmem,nccl,DeepEP}`. `../src/DeepEP-v1-last` is a git worktree of DeepEP at `a56d615`.
DeepEP V1 needs NVSHMEM ≥ 3.3.9 (`DeepEP-v1-last/docs/nvshmem.md:18,29`). The IBGDA device structs are
ABI-versioned `_v1` structs with `static_assert`ed sizes (`nvshmem_common_ibgda.h:158,182,209,223`). A diff
against the NVSHMEM 3.6.5 headers in `/nix/store` shows the QP, CQ and key structs have the same sizes. The only
exception is the 3.8-only batch-RMA bitmap (§1.1), which is not in 3.6.5.

Paths below are relative to each repo root. `ibgda.cpp` = `src/modules/transport/ibgda/ibgda.cpp`,
`dev.cuh` = `src/include/non_abi/device/pt-to-pt/ibgda_device.cuh`, `common.h` =
`src/include/device_host_transport/nvshmem_common_ibgda.h`, `env.h` = `src/modules/transport/common/env_defs.h`.

---

## 1. NVSHMEM IBGDA (v3.8.0-0)

### 1.1 Inventory: state the GPU holds or maps

Default placement: `NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=gpumem` (env.h:189). The ring, CQ and doorbell-record buffers go
through `ibgda_nic_control_alloc` → `ibgda_gpu_mem_alloc` → `cudaMalloc` (ibgda.cpp:1177-1186, 1030).
`NVSHMEM_IBGDA_NIC_HANDLER=auto`, which means GPU SMs ring the doorbell (env.h:202-208). D = `NVSHMEM_QP_DEPTH`,
default 1024 (env.h:121), rounded up to a power of 2.

| # | Structure / buffer | What it is | Where it lives | Size per instance (B) | Multiplicity | Source |
|---|---|---|---|---|---|---|
| 1 | SQ WQE ring (`wq_mobject`) | mlx5 send queue. The GPU writes WQEs here. | GPU HBM (cudaMalloc), NIC-registered umem | D×64 = **65,536** | per QP (RC and DCI). DCIs share one allocation. | ibgda.cpp:2186-2187 (size), 2100, 2132-2137 (alloc + NIC reg), 2369 (RC), 2371-2386 (DCIs share one slab) |
| 2 | SQ doorbell record (`dbr_mobject`) | 4 B SQ producer counter that the NIC reads (8 B allocated) | GPU HBM | **8** | per QP | ibgda.cpp:66, 2103, 2139-2147; device ptr ibgda.cpp:3910-3911 |
| 3 | Send CQ ring (`cq_mobject`) | 64 B CQEs, pre-set to 0xff. The GPU polls them. | GPU HBM, NIC-registered | D×64 = **65,536** | 1 per QP (each RC/DCI gets its own CQ) | ibgda.cpp:1438-1453 (1444 alloc, 1447 memset 0xff); one CQ per QP: 2352 |
| 4 | CQ doorbell record | CQ consumer counter | GPU HBM | **8** | per CQ | ibgda.cpp:1441, 1456-1461 |
| 5 | Internal buffer `ibuf` | Landing/source slots for fetch AMOs/`g` plus 1 non-fetch slot. DeepEP uses it as the AMO source. | GPU HBM (always cudaMalloc), MR-registered | 256×(1024+1) = **262,400** | per RC QP. DCIs have one per DCI in a shared buffer. | ibgda.cpp:2280-2285, 2022-2049; common.h:120; env.h:196-201 |
| 6 | BlueFlame/UAR doorbell (`uar_mobject`) | NIC BAR page. The GPU does an 8 B MMIO store to it. | **NIC BAR mapped into GPU VA** (`cudaHostRegister(..., IoMemory)` + `cudaHostGetDevicePointer`) | registered length 8 B (NC-dedicated UAR) or 2^`log_bf_reg_size` B (BF UAR). Device-dependent, not in source. | 1 UAR per QP | ibgda.cpp:1327-1394 (alloc), 1213-1250 (GPU map; 1225-1227 IoMemory), 3914 |
| 7 | `nvshmemi_ibgda_device_qp_t` | Device QP descriptor: qpn, ring pointers, dbrec, bf, cq, ibuf lkey/rkey, plus **`mvars`** (below) | GPU HBM (`globalmem.rcs`/`dcis`) | **184** | per RC slot (`rc_per_pe × ndev × npes`, self slot included) + per DCI | common.h:187-209; ibgda.cpp:3618-3669 (cudaMalloc 3666), 3310 |
| 7a | └ `mvars` (`qp_management_v1`) | Driver indices kept by the GPU: `post_send_lock`, `resv_head`, `ready_head`, `prod_idx`, `cons_idx`, `get_head`, `get_tail`, `ibuf.head/tail` | GPU HBM (inside #7) | **96** | per QP | common.h:164-183 |
| 8 | `nvshmemi_ibgda_device_cq_t` | Device CQ descriptor: cqe ptr, dbrec, pointers to prod/cons/resv/ready indices, cqn, ncqes, qpn | GPU HBM (`globalmem.cqs`) | **72** | per DCI + per RC slot | common.h:145-158; ibgda.cpp:3680-3733 |
| 9 | `nvshmemi_ibgda_device_state_t` | Global transport state: QP-map policy, counts, flags, plus **constmem** caches of 64 lkeys, 64 rkeys and 128 DCT AVs | GPU **`__constant__`** memory | **8,384** (8,192 of it constmem caches) | 1 per PE (per GPU context) | common.h:289-345; src/device/init/init_device.cu:40 |
| 10 | DCT address vectors (`mlx5_wqe_av`) | Remote DCT addressing for DC sends | constmem for the first 128, then GPU HBM | **48** | `NUM_DCT(2) × ndev × npes` | common.h:213; ibgda.cpp:2997-3064; env.h:160 |
| 11 | lkey table `nvshmemi_ibgda_device_key_t` | Local MR key per symmetric-heap chunk | constmem for the first 64, then GPU HBM | **16** (4 B key + pad + 8 B `next_addr`) | `heap/granularity × ndev` | common.h:229-232; ibgda.cpp:808-866 (constmem copy 851) |
| 12 | rkey table | Remote MR key per chunk **per peer** | constmem for the first 64, then GPU HBM | **16** | `heap/granularity × npes × ndev` (granularity 512 MiB, `NVSHMEM_CUMEM_GRANULARITY`, src/include/host/env/env_defs.h:237) | ibgda.cpp:4679-4775 |
| 13 | local-only mhandle list | lkeys of non-heap registered buffers (linked list the GPU walks) | GPU HBM | **96** | per registered local buffer | common.h:215-224; ibgda.cpp:756-805; dev.cuh:1990-2003 |
| 14 | `qp_group_switches` | Round-robin counters for picking a QP (atomicAdd on every put) | GPU HBM | 4 × (num_qp_groups+1) | per PE | ibgda.cpp:3749-3777 (3760, 3770); dev.cuh:1864 |
| 15 | batch-RMA pending-QP bitmap (**new in 3.8**) | Bitmap of QPs with deferred doorbells per region slot | GPU HBM | (⌈Q/32⌉+⌈⌈Q/32⌉/32⌉)×4096×4, with Q = DCIs + RC slots | per PE | ibgda.cpp:4060-4070; nvshmem_common_batch_rma_pending_qps.hpp:25-56; nvshmemi_region_constants.h:9 |
| — | NIC-side QPC/CQC (created by DEVX), CQ's own UAR (not GPU-mapped), SRQ + recv CQ (depth 16384), DCT objects, PD | NIC context and host-side verbs objects | NIC ICM / host memory | — | per device / per QP | ibgda.cpp:1519-1523, 2224-2246, 2584-2600 |

`mlx5` segment sizes (sizes.out): ctrl 16, raddr 16, data 16, atomic 16, AV 48, CQE 64 B. WQEBB = 64 B.

**Placement detail that inflates HBM use:** every `ibgda_gpu_mem_alloc(size, 64 KiB)` calls
`cudaMalloc(size + 65535)` so it can align the pointer (ibgda.cpp:1026-1030, `IBGDA_GPAGE_SIZE` = 64 KiB at ibgda.cpp:80-81).
An RC QP makes 5 such allocations: ibuf, WQ, SQ-DBR, CQ, CQ-DBR (ibgda.cpp:2029, 2132, 2140, 1444, 1456). The 8 B doorbell
records therefore each ask cudaMalloc for about 64 KiB. Treat this slack as an upper bound. Whether it is all
physically committed depends on the CUDA allocator (inferred).

**Per RC QP at depth 1024 (GPU HBM, struct and ring bytes):** WQ 65,536 + CQ 65,536 + ibuf 262,400 + DBRs 16 +
descriptors 256 = **393,744 B (384.5 KiB)**. Of that, **131,344 B (128.3 KiB)** is pure NIC-protocol state (rings,
doorbell records, descriptors), not counting ibuf. The cudaMalloc alignment slack adds up to 5×(64 KiB−1) ≈ 320 KiB more.
Each QP also holds **one NIC doorbell page mapped into the GPU**.

### 1.2 How many QPs, and totals

- RC QPs per PE = `IBGDA_NUM_RC_PER_PE × ndev × (npes − 1)`. There is no loopback QP (ibgda.cpp:3440-3446). QPs go to
  **every** other PE, including NVLink-reachable ones in the same node. `IBGDA_NUM_RC_PER_PE` defaults to 2 (env.h:177).
  It is forced to 8 on data-direct NICs unless the user set it (ibgda.cpp:5258-5266).
- Descriptor arrays are sized with the self slot: `rc_per_pe × ndev × npes` (ibgda.cpp:3627-3632).
- DCIs: `NVSHMEM_IBGDA_NUM_DCI` = 1 (env.h:162). DCTs: 2 (env.h:160). Each DCI has its own CQ, UAR and ibuf slice.
- ASSUMPTION: 1 NIC per PE (`ndev` = 1), i.e. a rail-optimized 8-GPU/8-NIC node. Symmetric heap = 2 GiB (only the
  rkey table uses this, and it is negligible).

`totals.py` output (GPU HBM, struct and ring bytes, excluding cudaMalloc slack):

| Configuration | PEs | RC/peer | RC QPs per GPU | GPU-mapped doorbell pages | GPU HBM transport state | + alloc slack (≤) |
|---|---|---|---|---|---|---|
| NVSHMEM default, 2×8 GPUs | 16 | 2 | 30 | 31 | 11.7 MiB | 9.4 MiB |
| NVSHMEM default, 16×8 | 128 | 2 | 254 | 255 | 95.9 MiB | 79.4 MiB |
| NVSHMEM default, 32×8 | 256 | 2 | 510 | 511 | 192.2 MiB | 159.4 MiB |

Formula: `total ≈ (n_rc + n_dci) × (2·D·64 + 256·1025 + 16) + (rc_slots + n_dci) × (184 + 72) + DCT/rkey overflow + bitmap`.
It grows **linearly in peers × QPs-per-peer**, about 384.5 KiB per QP at D = 1024.

### 1.3 The GPU-side instruction path of ONE put (`nvshmemi_ibgda_rma_nbi`, thread scope, RC QP, one chunk)

Default build: `NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=OFF` (cmake_config/NVSHMEMEnv.cmake:88). So WQE stores are volatile 32-bit
`WRITE_ONCE`s and `IBGDA_MEMBAR` is a real `__threadfence()` (dev.cuh:33-40, 216-243, 255-306).

| Step | What the SM thread does | Global-memory ops | Source |
|---|---|---|---|
| 1 | Read transport state from `__constant__` (`ibgda_get_state`) | const loads | dev.cuh:198-201 |
| 2 | Pick QP: `rc_map_type=none` (default, env.h:181) → `atomicAdd(qp_group_switches[0])` when >1 default RC | **1 atomic** | dev.cuh:1860-1867, 1900 |
| 3 | Translate addresses into keys: lkey lookup (constmem or global), rkey lookup indexed `[chunk][pe][dev]`, load `peer_heap_base_remote[pe]` | 3–5 loads | dev.cuh:1955-2048; call sites 2251, 2256 |
| 4 | Reserve WQE slot: `atomicAdd(mvars.resv_head, num_wqes)` (device scope, because RC QPs are shared among CTAs) | **1 atomic** | dev.cuh:2050-2062, 2274 |
| 5 | Check the ring is not full: once `wqe_idx ≥ D`, `ibgda_poll_cq` reads `cons_idx`, spins on `prod_idx`, reads CQE `wqe_counter`, does `atomicMax(cons_idx)`, then `fence.acq_rel.cta` | ≥3 loads + **1 atomic** + 1 fence (after the first D WQEs) | dev.cuh:1736-1753, 501-633 (atomicMax 599) |
| 6 | Build the WQE in the SQ ring: ctrl(16) + raddr(16) + data(16) = **48 B written as 12 × 32-bit stores** (`ibgda_store_wqe_segment` loops per word). Sets `CQ_UPDATE` on the last WQE of the group. | **12 stores** | dev.cuh:681-737, 308-333; call 2293 |
| 7 | If the source buffer is in sysmem: `__threadfence_system()` | (0–1 fence) | dev.cuh:2337-2339 |
| 8 | Publish ready: `__threadfence()`, then spin `atomicCAS(ready_head, base, base+n)` until earlier reservations are published, then `fence.acq_rel.cta` | **1 fence.gpu + ≥1 CAS + 1 fence.cta** | dev.cuh:1654-1666 |
| 9 | Decide whether to ring the doorbell: read `resv_head`; ring if no concurrent submitter or a 32-request batch boundary was crossed (`NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=32`, env.h:192) | 1 load | dev.cuh:1671-1690 |
| 10 | `ibgda_post_send`: take the per-QP spin lock (`atomicCAS(post_send_lock)` + `fence.acq_rel.cta`), `atomicMax(prod_idx)` | **1 CAS + 1 atomic + 1 fence** | dev.cuh:1561-1585, 415-427 |
| 11 | `__threadfence()`, write the **doorbell record** (big-endian 16-bit PI, 4 B store to GPU HBM that the NIC reads) | **1 fence + 1 store** | dev.cuh:1578-1579, 1527-1549 |
| 12 | `__threadfence()`, then an **8 B MMIO store to the BlueFlame/UAR page** (NIC BAR) holding ctrl-seg qpn+PI | **1 fence + 1 MMIO store** | dev.cuh:1580-1581, 1551-1558 |
| 13 | Release lock: `fence.acq_rel.cta` + store 0 | 1 fence + 1 store | dev.cuh:429-441 |
| 14 | Completion (`nvshmem_quiet`): for **every** RC QP to every PE plus DCIs, poll the CQ (#5 logic) | O(rc_per_pe×npes) CQ polls | dev.cuh:3672-3737, 1719-1734 |

**Count for one put that rings the doorbell:** about **4–5 global atomics/CAS** (QP pick, slot reserve,
ready CAS, lock CAS, prod_idx max, plus 1 more when checking a full ring), **12 WQE stores (48 B)** plus 1 doorbell-record store,
**1 MMIO doorbell store over PCIe**, 2 lock/unlock ops, **3 `__threadfence()` (gpu scope) + 3–4 `fence.acq_rel.cta`**,
and 3–5 key/address loads. The CQ-full check adds a CQ poll after the ring wraps. `quiet` polls every QP's CQ.
With 32-request batching, steps 10–13 are amortized when many threads submit concurrently. Steps 2–9 happen on every put.

### 1.4 Host-side resources IBGDA still needs (fairness)

- The **host builds everything**: DEVX `CREATE_CQ` and `CREATE_QP` commands, umem registration, UAR allocation, QP state
  transitions RST→INIT→RTR→RTS, an all-to-all of QP numbers over the bootstrap, and MR registration (ibgda.cpp:1472-1567,
  2298-2471, 1598-2002, 3413-3523). The GPU never creates QPs. It *drives* already-created QPs.
- **SRQ and recv CQ** (depth 16384) and the **DCT, SRQ and CQs are host-memory verbs objects** (ibgda.cpp:2224-2246, 2584-2600).
- A **host proxy thread still runs** (`NVSHMEMI_PROXY_MINIMAL`) for global exit and device timeouts, unless
  `NVSHMEM_DISABLE_LOCAL_ONLY_PROXY=1` (src/host/init/init.cu:1883-1903; env.h:124-128; `no_proxy` at ibgda.cpp:5323).
- **CPU doorbell fallback**: `NVSHMEM_IBGDA_NIC_HANDLER=cpu|cpu_host_memory`. The GPU still builds WQEs and does
  `atomicMax_system` on a host-visible `prod_idx`. A CPU progress loop then writes the DBR and BlueFlame
  (dev.cuh:1587-1605; ibgda.cpp:537-599, 5307-5308). This is used when the UAR cannot be GPU-mapped (ibgda.cpp:1383-1390).
  **Even in this mode, the WQE ring, the indices and the CQ polling stay on the GPU.**
- `NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=hostmem` moves the rings, CQs and DBRs to pinned host memory
  (ibgda.cpp:1117-1175). The GPU still writes and polls them, now over PCIe.

---

## 2. DeepEP V1 (a56d615, last commit with the NVSHMEM backend) on NVSHMEM IBGDA

### 2.1 What DeepEP V1 configures (`deep_ep/buffers/legacy.py`)

| Setting | Value | Source |
|---|---|---|
| `NVSHMEM_IB_ENABLE_IBGDA` | 1 | legacy.py:109 |
| `NVSHMEM_IBGDA_NUM_RC_PER_PE` | `num_qps_per_rank`. Constructor default 24. **LL mode must use num_local_experts** = num_experts / num_ranks. | legacy.py:38, 52-53, 110; docs/legacy.md:256; tests/legacy/test_low_latency.py:266; test_internode.py:324-325 (24) |
| `NVSHMEM_QP_DEPTH` | 1024 (the default is kept explicitly, "larger than on-flight WRs so we can skip WQ slot check") | legacy.py:112-114 |
| `NVSHMEM_DISABLE_P2P` | 0 when NVLink is allowed for LL | legacy.py:108 |
| `NVSHMEM_MAX_TEAMS` = 7, `DISABLE_NVLS` = 1, `CUMEM_GRANULARITY` = 2^29, `DISABLE_MNNVL` = 1 | "Reduce gpu memory usage" | legacy.py:116-126 |
| NVSHMEM PE set | **LL: all EP ranks**. **Normal: RDMA ranks only** (one per node, same local GPU index). | csrc/legacy/buffer.hpp:262-265 |

### 2.2 DeepEP V1's own device transport code (`csrc/kernels/legacy/ibgda_device.cuh`, 496 lines)

DeepEP reimplements the IBGDA put, AMO and quiet on top of NVSHMEM's device structs, so it is itself a NIC driver
running in SM code:
- `ibgda_get_rc` indexes `rcs[pe × num_rc_per_pe × ndev + id]` (:81-86). `qp_id` = destination local-expert index in LL
  (internode_ll.cu:266, 911), i.e. one QP per (peer, local expert).
- `ibgda_write_rdma_write_wqe` stores 48 B as **3 × 16 B `st.relaxed.gpu.L1::no_allocate.v4` vector stores** (:281-320;
  utils.cuh:249-253). The AMO WQE is 4 × 16 B (:383-426).
- `ibgda_reserve_wqe_slots`: a single `atomicAdd(resv_head)`, with **no ring-full check** (relies on depth 1024; :251-254).
- `ibgda_submit_requests`: `__threadfence()`, spin `atomicCAS(ready_head)`, and a doorbell every 4th message unless
  `kAlwaysDoPostSend` (:143-168). `ibgda_post_send`: lock CAS + `fence.acq_rel.cta`, `atomicMax(prod_idx)`,
  DBR `st.release.gpu` (4 B), BF `st.release.gpu` (8 B MMIO), unlock (:88-141).
- Its own `ibgda_poll_cq` / `nvshmemi_ibgda_quiet`, which are not thread-safe (:462-494).
- Per warp-put (`nvshmemi_ibgda_put_nbi_warp`, :335-381): lane 0 does 1 atomicAdd, lanes compute keys, ≤32 lanes each write one
  48 B WQE (3 vector stores), then 1 fence + ≥1 CAS, and every 4th message 1 lock CAS + 1 atomicMax + DBR store + MMIO store + unlock.

### 2.3 Totals for DeepEP V1 (same NVSHMEM 3.8 accounting; ASSUMPTION: 256 routed experts, DeepSeek-V3, 1 NIC/GPU)

| Mode | Nodes × 8 | NVSHMEM PEs | RC/peer | RC QPs per GPU | Doorbell pages mapped | GPU HBM transport state | + slack (≤) |
|---|---|---|---|---|---|---|---|
| LL | 2 (EP16) | 16 | 16 | 240 | 241 | 90.7 MiB | 75.0 MiB |
| LL | 16 (EP128) | 128 | 2 | 254 | 255 | 95.9 MiB | 79.4 MiB |
| LL | 32 (EP256) | 256 | 1 | 255 | 256 | 96.3 MiB | 79.7 MiB |
| Normal | 2 (EP16) | 2 | 24 | 24 | 25 | 9.4 MiB | 7.5 MiB |
| Normal | 16 (EP128) | 16 | 24 | 360 | 361 | 135.8 MiB | 112.5 MiB |
| Normal | 20 (EP160) | 20 | 24 | 456 | 457 | 171.9 MiB | – |

DeepEP V1's normal kernels run on at most 20 nodes = EP160 (`LEGACY_NUM_MAX_RDMA_PEERS = 20`,
`csrc/kernels/legacy/compiled.cuh:6`, checked in `csrc/legacy/buffer.hpp:113` unless low-latency mode).
Until 2026-10-07 this table listed "Normal, 32 (EP256): 744 QPs, 280.2 MiB", a configuration V1 refuses.
With more experts, LL grows to 511 QPs / 192.6 MiB at EP512 (512 experts) and 1,023 / 385.2 MiB at
EP1024 (1,024 experts); EP cannot exceed the expert count (`totals.out`).

In LL mode, RC QPs per GPU ≈ num_experts × (N−1)/N ≈ **255 for 256 experts regardless of EP size**. Every QP has
its own 64 KiB WQ ring, 64 KiB CQ ring, 256 KiB ibuf and GPU-mapped doorbell page. Normal mode grows as 24 × (nodes − 1).
The 3.8-only batch bitmap is included (≤ 416 KiB). Under NVSHMEM 3.3.9–3.6 the totals are the same minus that bitmap.


---

## 3. NCCL GIN, GDAKI backend (v2.32.3-1): now the **primary** GPU-initiated path, because DeepEP main runs on it

Paths: `gdaki.cc` = `src/transport/net_ib/gdaki/gin_host_gdaki.cc`, `gin_gdaki.h` = `src/include/nccl_device/gin/gdaki/gin_gdaki.h`,
`DOCA/` = `src/transport/net_ib/gdaki/doca-gpunetio/` (DOCA GPUNetIO vendored inside NCCL), `hl.cpp` = `DOCA/src/doca_gpunetio_high_level.cpp`,
`qp.cuh` = `DOCA/include/device/doca_gpunetio_dev_verbs_qp.cuh`, `cmn.cuh` = `DOCA/include/device/doca_gpunetio_dev_verbs_common.cuh`,
`onesided.cuh` = `DOCA/include/device/doca_gpunetio_dev_verbs_onesided.cuh`.
Struct sizes come from `sizes_gdaki.cpp` (in `sizes.out`).

### 3.1 How many QPs

- A GIN **context** (DeepEP calls it a "QP") holds **one RC QP per connected peer, including a self-loopback slot, plus one self-responder QP**:
  `nqps_for_comm_this_rank = nContexts × nranks / rankStride`, `nqps = nContexts × (nranks + 1)` (gdaki.cc:618-627, 799-879).
  If `ginCounterCount > 0` every QP also gets a **companion QP**, which doubles the count (gdaki.cc:624-625, 884-895). DeepEP uses no counters.
- Connected peers: `rankStride = requestedStride / connectedStride`. **FULL** → stride 1 (every rank, NVLink peers included).
  **RAIL** → stride = `lsaSize` (NVLink-domain size), i.e. one peer per node on the same rail. **CUSTOM_STRIDE** → user stride
  (src/gin/gin_host.cc:309-350; src/include/nccl_device/impl/core__funcs.h:67-73 `ncclTeamRail`).
- Contexts = `ginContextCount` rounded up to a multiple of `ginCommCount` (= number of local GIN NICs, env `NCCL_GIN_NCONNECTIONS`)
  (gin_host.cc:258-266, 161-179). Queue depth = `ginQueueDepth`, else `NCCL_GIN_GDAKI_QP_DEPTH` (default **128**), rounded up to a power of 2
  (gdaki.cc:58, 785; hl.cpp:1400). Defaults in `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER`: 4 contexts, 0 signals, 0 counters, connection NONE
  (src/include/nccl_device/host.h:82-108).

### 3.2 Inventory (default `NCCL_GIN_GDAKI_NIC_HANDLER=0` = AUTO → GPU_SM_DB when the UAR can be GPU-registered, else CPU_PROXY; hl.cpp:1406-1424)

| Structure / buffer | What | Where | Size per instance (B) | Multiplicity | Source |
|---|---|---|---|---|---|
| SQ WQE ring | mlx5 SQ. `doca_gpu_dev_verbs_wqe` = 64 B WQEBB | GPU HBM (`doca_gpu_mem_alloc` GPU slab, NIC umem) | align4K(D×64) = 65,536 @D=1024 (8,192 @128) | per QP | hl.cpp:575-581, 1402, 1448-1450; sizes.out |
| CQ ring (+ embedded 8 B DBR) | 64 B CQEs, one CQ per QP, D entries | GPU HBM slab | align4K(D×64+8) = **69,632** @1024 (12,288 @128) | per QP | hl.cpp:204-211, 1401, 1439-1441, 1470-1473 |
| CQ DBR slice | CQ doorbell record | GPU HBM slab, page-aligned | 4,096 (8 B used) | per QP | hl.cpp:1402, 1443-1446, 363-369 |
| SQ DBR slice | SQ doorbell record | GPU HBM (host memory only in CPU-proxy mode) | 4,096 (8 B used) | per QP | hl.cpp:1402, 1452-1460, 699-702 |
| UAR doorbell | NIC BAR page (NC-dedicated preferred, else NC or BlueFlame) | **NIC BAR mapped into GPU** via `cudaHostRegister(..., IoMemory)`, 8 B registered | 8 B | 1 UAR per QP | hl.cpp:103-134, 1476; DOCA/src/doca_gpunetio.cpp:682-705; doca_internal.hpp:64 |
| `doca_gpu_dev_verbs_qp` | Device QP: `sq_rsvd_index`, `sq_ready_index`, `sq_wqe_pi`, `sq_lock`, ring/DBR/DB pointers, precomputed qpn/ds words, embedded CQ state (`cqe_ci`, `cqe_rsvd`...) | GPU HBM, one array per context | **296** (embedded `doca_gpu_dev_verbs_cq` 64) | `nranks` slots per context (null slots included) | DOCA/include/common/doca_gpunetio_verbs_dev.h:155-199; doca_gpunetio.cpp:1105-1175 (alloc 1144) |
| `ncclGinGdakiGPUContext` | Per-context pointers to QP arrays, counter/signal tables, sink lkey, get trackers | GPU (cuMem) | **88** | per context | gin_gdaki_device_host_common.h:25-37; gdaki.cc:672, 987-1045 |
| Signal table | 8 B signals, NIC-atomic targets, plus rkey array (4 B × nranks) | GPU (cuMem, MR-registered) | 8 × nSignals + 4 × nranks | per context (× nSignals) | gdaki.cc:674, 741-745, 1026-1030 |
| Signal shadows | GPU-side shadow per signal per context | GPU (devr buffer) | 8 | nContexts × nSignals | src/dev_runtime.cc:1628-1640 |
| Counter table + companion QPs | Only when `ginCounterCount > 0` | GPU | 8/counter | — | gdaki.cc:673, 1019-1023 |
| `last_issued_get` / `last_visible_get` | Per-peer get ordering state | GPU (`ncclCudaCalloc`) | 8 + 8 | per context × nranks | gdaki.cc:978-979 |
| Window mem handle `ncclGinGdakiMemHandle` | lkey + pointer to per-peer **rkey array** (4 B × nranks) | GPU | 16 + 4×nranks | per registered window | gin_gdaki_device_host_common.h:39-42; gin_gdaki.h:70-71 |
| Sink buffer | 8 B local target for signal AMOs | GPU (cuMem) | 8 (cuMem rounds up to allocation granularity) | per GIN connection | gdaki.cc:969-975; src/include/alloc.h:331-332 |

**Per GDAKI QP at depth 1024: 143,360 B (140 KiB) of GPU-HBM rings and doorbell records, plus a 296 B device descriptor and one GPU-mapped doorbell.** At NCCL's default depth 128 it is 28,672 B.

### 3.3 GPU-side path of ONE `ncclGin::put` on GDAKI (thread coop, `NCCL_GIN_RESOURCE_SHARING_GPU`, the default DeepEP uses)

| Step | Work done by the SM thread | Ops | Source |
|---|---|---|---|
| 1 | Load context → `gdqp + peer`, rkey[peer], lkey (+ signal rkey) | 3–4 loads | gin_gdaki.h:58-90, 561-575 |
| 2 | Optional `fence.release.sys` if system scope was requested | (0–1 fence) | gin_gdaki.h:92-96 |
| 3 | Reserve slots: device-scope `atomicAdd(sq_rsvd_index, n)`. After the ring wraps, poll the CQ to check a slot is free. | **1 atomic** (+ CQ poll) | qp.cuh:94-128; cmn.cuh:334-346 |
| 4 | Build WQE: RDMA-WRITE ctrl+raddr+data = **48 B as 3 × `st.weak.cs.v2.b64`**. Put-with-signal adds an ATOMIC_FA WQE (4 × 16 B) = **112 B, 7 stores, 2 WQEBBs**. | 3 (or 7) stores | qp.cuh:45-48, 709-745, 985-1024; onesided.cuh:49-104, 291-360 |
| 5 | Mark ready: `fence.release.gpu`, spin `atomicCAS(sq_ready_index)`, `fence.acquire.gpu` | **1 CAS + 2 fences** | qp.cuh:156-204 (GPU branch 185-192), 231-250 |
| 6 | Submit (GPU_SM_DB): spin `atomicCAS(sq_lock)` + acquire fence, `atomic_max(sq_wqe_pi)` | **1 CAS + 1 atomic + 1 fence** | qp.cuh:524-533; cmn.cuh:316-330, 363-372 |
| 7 | **Doorbell #1**: 8 B MMIO store to the UAR ("early ring") | **1 MMIO store** | qp.cuh:534-535, 423-431, 359-392 |
| 8 | **DBR** update: 4 B big-endian PI, system-scope relaxed store | 1 store | qp.cuh:537-538, 280-294 |
| 9 | Fence + **Doorbell #2** (8 B MMIO), then unlock (release store) | **1 fence + 1 MMIO store + 1 store** | qp.cuh:540-549; cmn.cuh:381-390 |
| 10 | Completion: `flush` walks **every peer QP** of the context and polls its CQ | O(nranks) CQ polls | gin_gdaki.h:392-437; onesided.cuh:897-921 |

**Count for one put (GPU sharing mode):** 4 global atomics/CAS (reserve, ready CAS, lock CAS, PI max) + unlock,
3 WQE vector stores (48 B), or 7 (112 B) with a signal, 1 DBR store, **2 MMIO doorbell writes**, and 4–5 fences.
With `NCCL_GIN_RESOURCE_SHARING_THREAD` (→ DOCA EXCLUSIVE) the atomics and lock become plain loads and stores (cmn.cuh:317-320, 335-338, 364-365).
**The WQE build, DBR write, MMIO doorbells and CQ polling remain on the SM in every mode.**

### 3.4 Host side (fairness)

- The host creates everything: DOCA verbs/DEVX QPs, CQs, umems and UARs, QP connection through `allToAll` and RTR/RTS (gdaki.cc:467-586, 897-966).
  The CQ's completion channel and periodic error polling (`NCCL_GIN_ERROR_QUERY_SEC` = 10) run on the host (gdaki.cc:62, 771-779 comp channel, 1420-1440 error query).
- **CPU-proxy fallbacks** exist. (a) The DOCA `CPU_PROXY` NIC handler, used when the UAR cannot be GPU-mapped: the DBR/DB go to host memory and a CPU
  rings the doorbell. The GPU still writes the WQEs (hl.cpp:1414-1424, 1452-1460; qp.cuh:638-652). (b) The separate **GIN PROXY backend**
  (`NCCL_GIN_TYPE_PROXY`): the GPU only does a system-scope `fetch_add` on a per-peer PI and writes a descriptor (GFD) into a host-visible
  queue. A CPU proxy thread posts the verbs (src/include/nccl_device/gin/proxy/gin_proxy.h:109-130; src/gin/gin_host_proxy.cc:478).
  This is the existing "state off the GPU" design point, but it goes through the CPU. **DeepEP main requires GDAKI** (below).

---

## 4. DeepEP main (V2.5, 93eb6eb) on NCCL GIN GDAKI

- NVSHMEM is gone. All RDMA goes through `ncclGin` (`deep_ep/include/deep_ep/comm/handle.cuh:21-245`, default `NCCL_GIN_RESOURCE_SHARING_GPU` at :29).
  DeepEP has **no own WQE/doorbell code any more**: grep for mlx5/wqe/doorbell finds nothing in main. Direct-mode NVLink peers also go through the NIC
  (handle.cuh:222 "local or NVLink put will also go through NIC via this API").
- Device comm requirements (`csrc/kernels/comm/context.cpp:122-164`): `ginType = GDAKI` (asserted available, :129-134),
  `ginContextCount = num_allocated_qps`, `ginExclusiveContexts = true`, `ginQueueDepth = 1024` (`kDefaultQPDepth`, compiled.cuh:81),
  `ginSignalCount = num_ranks + 4`, connection = **RAIL** in hybrid mode (default `allow_hybrid_mode=True`) and **FULL** in direct mode (:141-146).
- `num_allocated_qps` (deep_ep/buffers/ep.py:327-335): hybrid → **65** if the NIC is MT4131 (ConnectX-8, `check_fast_rdma_atomic_support`,
  deep_ep/utils/envs.py:238-257), else **129**. Direct → **17**. Max 1024 (compiled.cuh:80). Per-kernel QP use is
  `get_theoretical_num_qps`: hybrid `num_sms×16+1`, direct `min(num_sms, 9)`, capped at the allocated count (ep.py:543-560). All allocated contexts'
  QPs exist on the GPU whether or not a kernel uses them.
- Other buffers (Engram, PP, Bucket) create additional contexts `kNumMaxQPs / num_rdma_ranks` (csrc/buffers/pp.hpp:55, bucket.hpp:86). They are **not counted** below.

### 4.1 Totals (totals.py). ASSUMPTIONS: 8 GPUs/node, NVLink domain = node (lsaSize 8), 1 GIN NIC per GPU (ginCommCount = 1), 4 KiB host page

QPs per GPU = C × (P + 1), with P = connected peers including self (RAIL: nodes; FULL: N).

| Mode (contexts C) | EP16 (2 nodes) | EP128 (16 nodes) | EP256 (32 nodes) |
|---|---|---|---|
| hybrid, CX-7 default (129) | 387 QPs, **53.6 MiB** | 2,193 QPs, **305 MiB** | 4,257 QPs, **592 MiB** |
| hybrid, CX-8 default (65) | 195 QPs, 27.0 MiB | 1,105 QPs, 154 MiB | 2,145 QPs, 298 MiB |
| direct (17) | 289 QPs, 39.6 MiB | 2,193 QPs, 301 MiB | 4,369 QPs, 599 MiB |

(GPU HBM for rings, DBR slices, QP descriptors, contexts, signal tables and get trackers. It is dominated by 140 KiB/QP. The number of GPU-mapped
doorbell UARs equals the number of QPs.)

---

## 5. Side-by-side: bytes and GPU ops per put

| | NVSHMEM IBGDA 3.8 (RC) | DeepEP V1 own IBGDA path | NCCL GIN GDAKI 2.32 |
|---|---|---|---|
| Per-QP GPU HBM (D=1024) | 384.5 KiB (128.3 KiB rings/DBR/desc + 256.3 KiB ibuf) + ≤320 KiB alloc slack | same (NVSHMEM-allocated) | 140 KiB + 296 B desc |
| GPU-mapped NIC doorbell | 1 UAR per QP | same | 1 UAR per QP |
| WQE bytes / store instr. per put | 48 B / 12 × 32-bit | 48 B / 3 × 128-bit | 48 B / 3 × 128-bit (112 B / 7 with signal) |
| Global atomics/CAS per put | 4–5 (+ lock) | 1 add + 1 CAS; every 4th msg + lock CAS + max | 4 (+ unlock) |
| Fences per put | 3 × fence.gpu + 3–4 × fence.cta | 1 × fence.gpu + 2 × fence.cta (on DB) | 4–5 (release/acquire gpu) |
| Doorbell record + MMIO per put | 1 DBR + 1 MMIO (batched ≤32) | 1 DBR + 1 MMIO every 4th msg | 1 DBR + **2 MMIO** |
| Completion | CQ poll on every QP in `quiet` | poll per (peer, qp) | CQ poll on every peer QP in `flush` |

---

## 6. Fairness caveats and uncertainties

1. **Sizes are for struct/ring bytes requested from the allocator.** NVSHMEM asks `cudaMalloc` for size + 64 KiB−1 per aligned buffer.
   NCCL cuMem allocations round up to the allocation granularity. Actual resident HBM can be higher. We show NVSHMEM slack separately as an upper bound.
2. **NVSHMEM's 256 KiB `ibuf` per RC QP** is transport-internal staging for fetching AMOs/`g`, not NIC-protocol state proper. We report it separately (128.3 KiB without it).
3. **NIC count** matters: ndev/ginCommCount multiplies QPs. We assumed 1 NIC per GPU. DeepEP's CX-7/CX-8 context count is taken from code; mapping MT4131 to ConnectX-8 is our inference.
4. **LL expert count** (256, DeepSeek-V3) is our assumption for DeepEP V1. The code requires `num_qps_per_rank = num_experts / num_ranks`.
5. The **3.8-only batch-RMA bitmap** is included for NVSHMEM. DeepEP V1 ran on 3.3.9+, where it does not exist (≤ 416 KiB difference).
6. The host still creates and connects every QP and holds SRQ/DCT/recv-CQ objects (NVSHMEM). NVSHMEM also keeps a minimal proxy thread, NCCL an error-query path.
   The claim is "the GPU **holds** the per-QP datapath state and **executes** the post/doorbell/poll protocol", not "the GPU does all of RDMA".
7. Both stacks already ship CPU-assisted modes (NVSHMEM `NIC_HANDLER=cpu`, DOCA `CPU_PROXY`, NCCL GIN PROXY backend). They trade latency for SM work.
   In the first two, the WQE ring and indices **stay on the GPU**.
8. Instruction counts are read from source (CUDA/PTX level), not from SASS. Spin loops (ready CAS, lock, CQ poll) are counted once. Under contention they repeat.
9. NVSHMEM `quiet` and GIN `flush` cost scales with the number of QPs (peers × QPs-per-peer). That is a per-sync cost that grows with EP size, not a per-put cost.

---

## 7. Candidate one-sentence claims (numbers above)

1. "In DeepEP's current NCCL-GIN (GDAKI) backend with its default 129 GIN contexts, each GPU of a 256-GPU EP job holds **4,257 RC QPs**.
   That is **≈592 MiB of HBM** for SQ/CQ rings and doorbell records, plus 4,257 NIC doorbell pages mapped into the GPU, all so SMs can drive the NIC."
   (§4.1: 129 × 33 QPs × 140 KiB; context.cpp:135-137, ep.py:332, gdaki.cc:618-627, hl.cpp:1400-1402)
2. "Posting a single RDMA write from an SM in NCCL GIN GDAKI takes 4 global atomics, 3 WQE stores, a doorbell-record write, 2 MMIO doorbell writes
   and 4–5 memory fences. NVSHMEM IBGDA takes 4–5 atomics, 12 32-bit WQE stores, 1 MMIO doorbell and 3 device-scope `__threadfence`s.
   The accelerator is literally running the NIC driver's post-send path." (§1.3, §3.3)
3. "With NVSHMEM IBGDA every RC QP costs the GPU 384 KiB (128 KiB of it WQE/CQ rings). DeepEP V1's normal mode (24 QPs/peer) therefore held **361 QPs ≈ 136 MiB** per GPU at EP128 (it runs on at most 20 nodes; the earlier "744 QPs at EP256" was a configuration V1 refuses),
   and even its low-latency mode held ≈ 255 QPs ≈ 96 MiB regardless of EP size." (§2.3)

