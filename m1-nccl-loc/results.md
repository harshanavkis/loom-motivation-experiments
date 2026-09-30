# M1: How much of NCCL is scale-up code, scale-out code, and shared code?

## What was measured

| | newest release | older release (about 3 years earlier) |
|---|---|---|
| tag | `v2.32.3-1` (highest `v2.x.y-1` in `git ls-remote --tags`) | `v2.18.5-1` |
| commit | `12df1a11afad322be5a204a2db890161cbf8131d` (2026-09-17) | `559b70f86c190a0d8f67f0d7a0f2c9810dd1e8c7` (2023-08-23) |
| checkout | `../src/nccl` | `../src/nccl-v2.18.5` (git worktree) |

- **Tools.** cloc 2.08 counts code lines (no blank lines, no comments). universal-ctags 6.2.1 gives the function and struct line ranges used to split mixed files. Both run through `nix shell nixpkgs#cloc nixpkgs#universal-ctags`, with no system install. The script's own comment-stripping counter matches cloc exactly on every split file; it was checked against all 13 mixed files in v2.32.
- **Scope.** Only `src/`, which is the libnccl library. I counted C, C++, C/C++ headers and CUDA, plus the kernel generators `src/device/generate.py`, `src/device/symmetric/generate.py` and v2.18's `gen_rules.sh`.
  - Excluded: `plugins/` (example and reference plugins), `contrib/`, `bindings/`, `docs/`, `pkg/`, CMake and Makefiles, and `misc/generate_git_version.py`.
  - There are no tests under `src/`.
  - No generated code is checked in. The generators write device and host tables and `rules.mk` into the build directory, so I counted the generators and not their outputs.
- **Vendored third-party code** is reported separately and left out of the headline numbers:
  - NVTX (`include/nvtx3/`)
  - DOCA GPUNetIO (`net_ib/gdaki/doca-gpunetio/`, BSD-3 copy)
  - Amazon `efa-dp-direct/`
  - Microsoft `net_nd/ndspi.h` and `nddef.h`
  - The rdma-core copies `ibvcore.h` and `mlx5dvcore.h`
- **Re-running.** From this directory:
  `nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl v2.32.3-1`, and the same with `../src/nccl-v2.18.5 v2.18.5-1`.
- **Outputs.**
  - `summary_<tag>.txt` is the full stdout.
  - `files_<tag>.csv` has one row per file: path, language, code lines, category, vendored flag, and the split of mixed files.
  - `split_<tag>.csv` lists every function or struct region that was reassigned inside a mixed file.

## Headline results (NCCL-authored code, vendored excluded)

"File view" puts each file in exactly one bucket. "Split view" divides the 13 mixed files (11 in v2.18) function by function, using the explicit name lists in `SPLITS`. Anything not listed stays in the file's default bucket, which is common in every case.

| category | v2.18.5 files | v2.18.5 file view | v2.18.5 split view | v2.32.3 files | v2.32.3 file view | v2.32.3 split view |
|---|---|---|---|---|---|---|
| scale-up | 7 | 1,465 (5.9%) | **1,745 (7.0%)** | 61 | 14,956 (13.2%) | **16,566 (14.6%)** |
| scale-out | 15 | 4,821 (19.3%) | **6,347 (25.4%)** | 140 | 33,507 (29.6%) | **36,580 (32.3%)** |
| mixed | 11 | 6,948 (27.8%) | – | 13 | 14,994 (13.2%) | – |
| common | 75 | 11,775 (47.1%) | **16,917 (67.6%)** | 235 | 49,765 (44.0%) | **60,076 (53.1%)** |
| **total** | 108 | 25,009 | 25,009 | 449 | 113,222 | 113,222 |

**Mixed files after splitting.**
- v2.32: 1,610 lines went to scale-up, 3,073 to scale-out and 10,311 to common.
- v2.18: 280 lines went to scale-up, 1,526 to scale-out and 5,142 to common.

**Vendored code** (reported separately, not in the table above):
- v2.32 has 30,183 vendored lines: 25,230 scale-out (DOCA GPUNetIO, EFA, NetworkDirect, verbs headers) and 4,953 NVTX, counted as common.
- Including vendored code, v2.32's `src/` has 143,405 lines, of which scale-out is 61,810 (43.1%, split view).
- v2.18 has 4,728 vendored lines: 844 scale-out (`ibvcore.h`) and 3,884 NVTX.

### How much the share of transport-specific code (scale-up plus scale-out) depends on the judgment calls

| variant | v2.18.5 | v2.32.3 |
|---|---|---|
| A. File view, mixed files counted as common (lower bound) | 25.1% | 42.8% |
| B. Split view (the default) | 32.4% | 46.9% |
| C. B, plus the whole proxy (`proxy.cc` and `proxy.h`) counted as scale-out | 35.5% | 47.9% |
| D. B, excluding RAS and diagnostics (`src/ras`, `src/diagnostics*`) | 32.4% (neither exists) | 51.2% of 100,865 |

**Growth from v2.18.5 to v2.32.3** (split view, arithmetic on the numbers above):

| | v2.18.5 | v2.32.3 | growth |
|---|---|---|---|
| total | 25,009 | 113,222 | 4.5× |
| scale-up | 1,745 | 16,566 | 9.5× |
| scale-out | 6,347 | 36,580 | 5.8× |
| transport-specific (scale-up + scale-out) | 8,092 | 53,146 | 6.6× |
| common | 16,917 | 60,076 | 3.6× |

Scale-out code is 3.6× the size of scale-up code in v2.18 and 2.2× in v2.32.

### Where the code sits in v2.32.3 (split view)
- `src/transport`: 3,082 scale-up, 18,468 scale-out, 125 common.
- `src/include`: 3,866 scale-up, 7,098 scale-out, 11,484 common.
- Top-level `src/*.cc`: 3,101 scale-up, 1,040 scale-out, 10,970 common.
- `src/device`: 3,302 scale-up, 1,574 scale-out, 5,654 common.
- `src/graph`: 391 scale-up, 1,310 scale-out, 5,098 common.
- `src/plugin`: 2,420 scale-out, 2,710 common.
- `src/rma`: 401 scale-up, 1,269 scale-out, 221 common.
- `src/gin`: 1,156 scale-out.
- `src/ras`: 10,766 common.

`summary_v2.32.3-1.txt` has the full table.

### Largest files per category (v2.32.3; full lists in `files_v2.32.3-1.csv`)
- **Scale-up:**
  - `ce_coll.cc` (1,868) and `transport/p2p.cc` (1,281)
  - `diagnostics/p2p.cc` (848)
  - `nccl_device/impl/cft__funcs.h` (794)
  - `transport/nvls.cc` (692)
  - `impl/reduce_copy__funcs.h` (628) and `impl/multimem__funcs.h` (605)
  - The `device/symmetric/*.cuh` LSA kernels (389 to 532 each)
  - `rma/rma_ce.cc` (401), `transport/shm.cc` (376) and `nvls_ub.cc` (374)
- **Scale-out:**
  - `transport/net.cc` (1,833) and `transport/coll_net.cc` (1,575)
  - `impl/gin__funcs.h` (1,483)
  - `net_ib/connect.cc` (1,431)
  - `net_ib/p2p_resiliency_recovery.cc` (1,263) and `net_ib/p2p_resiliency.cc` (1,024)
  - `net_ib/gdaki/gin_host_gdaki.cc` (1,178)
  - `net_nd/connect.cc` (931), `net_ib/p2p.cc` (789) and `net_ib/gin.cc` (782)
  - `gin/gpi/gin_gpi.h` (761) and `misc/gdrwrap.cc` (703)
- **Mixed (split):** `init.cc` (3,266), `graph/topo.cc` (2,140), `dev_runtime.cc` (1,970), `proxy.cc` (1,902), `graph/search.cc` (1,296), `graph/paths.cc` (910), `device/all_reduce.h` (661), `device/all_gather.h` (588), `device/reduce_scatter.h` (526), `register/coll_reg.cc` (498), `graph/connect.cc` (439), `transport.cc` (421) and `include/proxy.h` (377).
- **Common:** `enqueue/enqueue.cc` (2,927), `ras/client_support.cc` (1,984), `ras/diagnostics_pci.cc` (1,471), `bootstrap.cc` (1,319), `os/windows.cc` (1,270), `graph/xml.cc` (1,245), `plugin/profiler.cc` (1,244), `device/prims_simple.h` (1,185) and `device/reduce_kernel.h` (1,110).
- **v2.18.5:**
  - Scale-up is just `p2p.cc` (608), `shm.cc` (368), `nvls.cc` (294) and `ipcsocket.cc` (147), plus small headers.
  - Scale-out is `net_ib.cc` (1,138), `transport/net.cc` (1,073), `coll_net.cc` (698), `net_socket.cc` (540) and the `net.cc` plugin loader (319), plus the gdr and ibv wrappers.

## The transport interfaces (counted by `categorize.py`)

| contract | v2.18.5 | v2.32.3 |
|---|---|---|
| `ncclNet_vX` (scale-out plugin API) | v4–v6 (3 versions); v6 has 16 function pointers | v6–v12 (7 versions); v12 has 21 function pointers (v6 has 16) |
| `ncclCollNet_vX` (SHARP / CollNet plugin API) | v4–v6 (3); v6 has 14 | v6–v12 (7); v12 has 18 |
| `ncclGin_vX` (GPU-initiated networking plugin API) | – | v13–v14 (2); v14 has 16 (v13 has 20) |
| `ncclRma_vX` (host RMA plugin API) | – | v14–v16 as structs; v16 has 21. The loader also accepts v13, which is `typedef ncclGin_v13_t` (`plugin/rma.cc`: `{16,15,14,13}`) |
| scale-up plugin API | none | **none**. P2P, SHM and NVLS are hard-wired into NCCL |
| internal `ncclTransportComm` | 8 function pointers × {send, recv}, plus `canConnect` | 10 function pointers × {send, recv}, plus `canConnect` |
| built-in `ncclTransports[]` | p2p, shm, net, collNet | p2p, shm, net, collNet (NVLS is set up separately; its struct has only `free`) |
| public device-API declarations (`nccl_device/*.h`) | – | 82 scale-up (LSA, multimem, CFT, LL-A2A, reduce-copy), 86 scale-out (GIN, GIN barrier, net device), 158 shared (team, comm, pointer, coop, host) |

In v2.32 the loader has to adapt 7 + 7 + 2 + 4 = 20 versioned scale-out plugin struct definitions, up from 3 + 3 = 6 in v2.18. The version lists come from the `ncclNetVersion`, `ncclGinVersion` and `ncclRmaVersion` arrays in `src/plugin/*.cc`. Scale-up has no plugin contract at all.

## Judgment calls (all encoded in `categorize.py`: `RULES`, `SPLITS` and `VENDORED`)

1. **Proxy (`proxy.cc`, `proxy.h`): split three ways.**
   - I read the code before deciding. In v2.32, only NET and COLLNET set `proxyProgress` by default. P2P sets it only under the opt-in `NCCL_P2P_USE_CUDA_MEMCPY=1`, and SHM's is NULL. In v2.18, both P2P and SHM had opt-in CE-memcpy progress (`P2P_USE_CUDA_MEMCPY` and `SHM_USE_CUDA_MEMCPY`, both default 0).
   - The **progress engine** (op posting, `ncclProxySaveOp`, progress thread, ops pool, and the `ncclProxyOp`/`Args`/`SubArgs` structs) therefore counts as **scale-out**: 819 lines in v2.32.
   - The **service/RPC thread** (connection pool, `ncclProxyConnect`, `CallAsync`, `ncclProxyService`, create and destroy) runs the setup, free and register calls of *every* transport, so it counts as **common**.
   - The **UDS file-descriptor-passing paths** (`ncclProxyServiceUDS`, `proxyGetFd`, `proxyQueryFd`, `…ClientGetFdBlocking`, v2.18's `ConvertFd`) exchange cuMem POSIX handles, so they count as **scale-up**: 193 lines.
   - Variant C shows what happens if the whole proxy is counted as scale-out: about +1 percentage point in v2.32 and +3 in v2.18.
2. **Topology and graph code:** split by function.
   - NIC, vNIC, merge, GDR, PXN, net-search and CollNet-search functions count as scale-out.
   - NVLink, C2C, NVB, MNNVL, `CheckP2p`, `SplitNvLink`, NVLS search and `connectNvls` count as scale-up.
   - Generic PCI/CPU topology, path computation, ring/tree search and the XML code stay common.
   - `graph/xml.cc` and `trees.cc`/`rings.cc` are entirely common.
3. **Device collective kernels:**
   - The `RunWorkColl<…, NCCL_ALGO_NVLS / NVLS_TREE, …>` specializations count as scale-up. NVLS_TREE is a hybrid of NVLS inside the node and a tree across the network, and I put it on the NVLS side.
   - The `NCCL_ALGO_COLLNET_*` specializations count as scale-out.
   - Ring, tree and PAT, and all of `prims_simple/ll/ll128.h`, `reduce_kernel.h`, `op128.h`, `common.h` and `sendrecv.h`, are **common**. The legacy device primitives are transport-agnostic by design, because every transport exposes the same connection FIFO to the kernel.
4. **Device API (LSA/GIN) in v2.32:**
   - LSA barrier, multimem, `reduce_copy` (`ncclLsaReduce*`), LL-A2A, CFT (the NVLink-fabric logical-endpoint feature, gated on the CUDA 13.3 `LOGICAL_ENDPOINT` attributes), `cft_dev_runtime.cc`, and the symmetric kernels and scheduler count as **scale-up**.
   - Everything under `gin*`, `nccl_device/gin/*` (GDAKI, proxy, GPI, EFA-GDA), `net_device.h`, `src/gin/`, the `*gin*` symmetric kernels and `tuning/sym_model/gin.cc` counts as **scale-out**.
   - The shared device-API infrastructure (`core*`, `comm*`, `ptr*`, `coop`, `utility`, `vector*`, `host.h`, `barrier*` which composes LSA and GIN barriers, `devcomm/*` ABI shims, `dev_runtime_segments.cc`) counts as **common**.
   - `dev_runtime.cc` is split: the LSA map/export/import and multimem-pointer functions go to scale-up, and GIN/RMA registration and GDAKI/EFA/proxy dumps go to scale-out.
   - A few symmetric kernels are hybrid "Rail" variants that also use GIN, and I left them in scale-up. That slightly overstates scale-up.
5. **Files I kept common even though they have branches for both stacks:**
   - `init.cc`: only `initGdrCopy`, and v2.18's `collNetTrySetup` (165 code lines), are carved out.
   - `enqueue.cc`, `group.cc`, `register.cc`, `transport.cc` (except the CollNet setup functions), `comm.h`, `device.h`.
   - This choice is conservative: it favours common, and so understates the split.
6. **cuMem and IPC:**
   - Only IPC-specific code counts as scale-up: `ipcsocket` and the proxy UDS paths.
   - `allocator.cc` (`ncclMemAlloc`, 382 lines) and `mem_manager.cc` (856 lines), which manage every allocation, and `cudawrap.cc` stay common.
   - Moving the first two to scale-up would add 1,238 lines, about +1.1 percentage points.
7. **Network-adjacent helpers:**
   - `gdrwrap.cc` counts as scale-out: GDRCopy is used only by net, coll_net, rma_proxy, gin_host_proxy and rma_socket.
   - `misc/socket.cc`, `crypt.cc` (TLS for bootstrap), `bootstrap.cc`, `shmutils.cc` (shared by proxy, p2p, shm, nvls and net) and the RAS subsystem (its own socket overlay network for health monitoring) stay common.
8. **RMA (host one-sided operations):**
   - `rma_ce.cc` counts as scale-up.
   - `rma_proxy*`, `plugin/rma*` and `transport/rma_socket*` count as scale-out.
   - `rma.cc`, the dispatcher between the two, is common.
9. **Windows NetworkDirect port (`transport/net_nd/`):** 4,014 NCCL-authored lines, per `files_v2.32.3-1.csv`, counted in scale-out. It is a Windows-only reimplementation of the IB transport. If you think it should not count, subtract it from scale-out.
10. **Tuning and diagnostics:**
    - `tuning/collnet.cc` counts as scale-out.
    - `tuning/nvls.cc`, `ce_model.cc` and `sym_model/lsa*` count as scale-up.
    - `diagnostics/p2p*` counts as scale-up and `ib_write_bw*` as scale-out.
    - v2.18's monolithic `graph/tuning.cc` stays common. This makes the v2.18 transport share look slightly smaller than a like-for-like split would: in v2.32 the equivalent transport-specific tuning code is only 821 lines.

**Uncertainty.** The v2.32 transport-specific share is **43–51%** depending on these calls (variants A–D), with **47%** as the best estimate. For v2.18 it is **25–36%**, with **32%** as the best estimate. The direction of growth holds under every variant.

## Candidate one-sentence claims for §2

1. "In NCCL 2.32, 47% of the library's 113 K lines of NCCL-authored code (43–51% depending on how shared files are attributed) is specific to one transport stack: 36.6 K lines for scale-out (IB/RoCE, socket, CollNet, GIN, proxy) and 16.6 K for scale-up (P2P, SHM, NVLS, CE, LSA). Only 60 K lines are shared."
   - Numbers: split view 16,566 + 36,580 = 53,146 of 113,222; range from variants A to D.
2. "Between NCCL 2.18 (2023) and 2.32 (2026), transport-specific code grew 6.6× (8.1 K to 53.1 K lines) while shared code grew only 3.6× (16.9 K to 60.1 K). Stack-specific code went from about a third to about half of the codebase (32% to 47%)."
3. "The two stacks also expose two unrelated contracts. Scale-out is reached through four versioned plugin APIs (ncclNet v6–v12, ncclCollNet v6–v12, ncclGin v13–v14, ncclRma v13–v16: 20 struct versions, up from 6 in 2.18). Scale-up has no plugin API and is hard-wired, and its GPU-side device API is separate: 82 LSA/multimem declarations versus 86 for GIN."
