# M1: NCCL scale-up vs scale-out lines of code

Detailed evidence, per-file lists and the reasoning behind every judgment call: [results.md](results.md).

## Question

How much of NCCL's library code is specific to the scale-up stack (P2P/NVLink, SHM, NVLS, CE, LSA), how much to the scale-out stack (IB/RoCE, socket, CollNet, GIN, proxy), and how much is shared, and how has that changed over three years? If a large and growing share is stack-specific, NCCL is in practice two communication stacks behind one API, which is the split Loom proposes to replace with one interface.

## Methodology

**Sources** (clones under `../src/`):

| | newest release | older release |
|---|---|---|
| tag | `v2.32.3-1` (highest `v2.x.y-1` in `git ls-remote --tags`) | `v2.18.5-1` |
| commit | `12df1a11afad322be5a204a2db890161cbf8131d` (2026-09-17) | `559b70f86c190a0d8f67f0d7a0f2c9810dd1e8c7` (2023-08-23) |
| checkout | `../src/nccl` (clone of github.com/NVIDIA/nccl) | `../src/nccl-v2.18.5` (git worktree of `../src/nccl`) |

**Tools:** cloc 2.08 and Universal Ctags 6.2.1, both from `nix shell nixpkgs#cloc nixpkgs#universal-ctags`; Python 3.13.15 runs `categorize.py`.

**What was counted:**
- Code lines as cloc counts them: no blank lines, no comment lines.
- Scope is `src/` only (libnccl): C, C++, C/C++ headers, CUDA, plus the kernel generators `src/device/generate.py`, `src/device/symmetric/generate.py` and v2.18's `gen_rules.sh`. Excluded: `plugins/`, `contrib/`, `bindings/`, `docs/`, `pkg/`, build files, `misc/generate_git_version.py`. There are no tests under `src/`, and no generated code is checked in.
- Vendored third-party code (NVTX, DOCA GPUNetIO copy, Amazon `efa-dp-direct/`, Microsoft NetworkDirect headers, rdma-core `ibvcore.h`/`mlx5dvcore.h`) is categorized but reported separately and left out of the headline.

**How files are categorized** (`RULES`, `SPLITS`, `VENDORED` in `categorize.py`):
- *File view:* ordered path rules put each file in exactly one of scale-up, scale-out, mixed, common. Unmatched files are common.
- *Split view (default):* the 13 mixed files (11 in v2.18) are split function by function using ctags line ranges and explicit name lists. Regions not listed stay in the file's default bucket, which is common in every case. The script's own comment-stripping counter matches cloc exactly on every split file.

**Judgment calls** (full list with reasons in [results.md](results.md#judgment-calls-all-encoded-in-categorizepy-rules-splits-and-vendored)):
1. Proxy split three ways: progress engine = scale-out (819 lines in v2.32; only NET and COLLNET set `proxyProgress` by default), service/RPC thread = common, UDS fd-passing for cuMem handles = scale-up (193 lines).
2. Topology/graph code (`topo.cc`, `paths.cc`, `search.cc`, `connect.cc`) split by function.
3. Device collectives: `NCCL_ALGO_NVLS`/`NVLS_TREE` specializations = scale-up, `NCCL_ALGO_COLLNET_*` = scale-out; ring/tree/PAT and the legacy primitives = common.
4. v2.32 device API: LSA, multimem, reduce-copy, LL-A2A, CFT, symmetric kernels = scale-up; everything GIN = scale-out; shared device-API infrastructure = common. Hybrid "Rail" symmetric kernels that also use GIN are left in scale-up (slightly overstates scale-up).
5. Files with branches for both stacks (`init.cc` except `initGdrCopy`/`collNetTrySetup`, `enqueue.cc`, `group.cc`, `register.cc`, `transport.cc` except CollNet setup, `comm.h`, `device.h`) stay common. This understates the split.
6. `allocator.cc` (382) and `mem_manager.cc` (856) stay common; moving them to scale-up would add 1,238 lines (about +1.1 points).
7. `gdrwrap.cc` = scale-out; `socket.cc`, `crypt.cc`, `bootstrap.cc`, `shmutils.cc` and RAS = common.
8. RMA: `rma_ce.cc` = scale-up, `rma_proxy*`/`plugin/rma*`/`rma_socket*` = scale-out, dispatcher `rma.cc` = common.
9. The Windows NetworkDirect port (`transport/net_nd/`, 4,014 NCCL-authored lines) is counted as scale-out. Subtract it if you think it should not count.
10. v2.18's monolithic `graph/tuning.cc` stays common, which makes the v2.18 transport share slightly smaller than a like-for-like split.

## Results

NCCL-authored code, vendored excluded (from `summary_v2.18.5-1.txt`, `summary_v2.32.3-1.txt`):

| category | v2.18.5 files | v2.18.5 file view | v2.18.5 split view | v2.32.3 files | v2.32.3 file view | v2.32.3 split view |
|---|---|---|---|---|---|---|
| scale-up | 7 | 1,465 (5.9%) | **1,745 (7.0%)** | 61 | 14,956 (13.2%) | **16,566 (14.6%)** |
| scale-out | 15 | 4,821 (19.3%) | **6,347 (25.4%)** | 140 | 33,507 (29.6%) | **36,580 (32.3%)** |
| mixed | 11 | 6,948 (27.8%) | – | 13 | 14,994 (13.2%) | – |
| common | 75 | 11,775 (47.1%) | **16,917 (67.6%)** | 235 | 49,765 (44.0%) | **60,076 (53.1%)** |
| **total** | 108 | 25,009 | 25,009 | 449 | 113,222 | 113,222 |

Mixed files after splitting: v2.32 gave 1,610 lines to scale-up, 3,073 to scale-out, 10,311 to common; v2.18 gave 280, 1,526 and 5,142.

Sensitivity of the transport-specific (scale-up + scale-out) share to the judgment calls:

| variant | v2.18.5 | v2.32.3 |
|---|---|---|
| A. File view, mixed files counted as common (lower bound) | 25.1% | 42.8% |
| B. Split view (the default) | 32.4% | 46.9% |
| C. B, plus the whole proxy (`proxy.cc`, `proxy.h`) counted as scale-out | 35.5% | 47.9% |
| D. B, excluding RAS and diagnostics (`src/ras`, `src/diagnostics*`) | 32.4% (neither exists) | 51.2% of 100,865 |

Growth v2.18.5 to v2.32.3 (split view):

| | v2.18.5 | v2.32.3 | growth |
|---|---|---|---|
| total | 25,009 | 113,222 | 4.5× |
| scale-up | 1,745 | 16,566 | 9.5× |
| scale-out | 6,347 | 36,580 | 5.8× |
| transport-specific (scale-up + scale-out) | 8,092 | 53,146 | 6.6× |
| common | 16,917 | 60,076 | 3.6× |

Scale-out code is 3.6× the size of scale-up code in v2.18 and 2.2× in v2.32.

Vendored code (not in the tables above): v2.32 has 30,183 vendored lines (25,230 scale-out, 4,953 NVTX counted as common). Including it, v2.32's `src/` has 143,405 lines, of which scale-out is 61,810 (43.1%, split view). v2.18 has 4,728 vendored lines (844 scale-out, 3,884 NVTX).

Transport interfaces (counted by `categorize.py`; the v13 RMA entry and the 20-version total come from reading `src/plugin/*.cc`, see [results.md](results.md#the-transport-interfaces-counted-by-categorizepy)):

| contract | v2.18.5 | v2.32.3 |
|---|---|---|
| `ncclNet_vX` | v4–v6 (3 versions) | v6–v12 (7) |
| `ncclCollNet_vX` | v4–v6 (3) | v6–v12 (7) |
| `ncclGin_vX` | – | v13–v14 (2) |
| `ncclRma_vX` | – | v14–v16 as structs; loader also accepts v13 (= `ncclGin_v13_t`) |
| scale-up plugin API | none | none |
| public device-API declarations (`nccl_device/*.h`) | – | 82 scale-up, 86 scale-out, 158 shared |

The v2.32 loader adapts 7 + 7 + 2 + 4 = 20 versioned scale-out plugin structs, up from 6 in v2.18.

Uncertainty: v2.32 transport-specific share is 43–51% (variants A–D), best estimate 47%; v2.18 is 25–36%, best estimate 32%. The direction of growth holds under every variant.

## Caveats / fairness

- Lines of code measure implementation size, not complexity or maintenance cost.
- Judgment calls 5 and 6 favour common, so the default split understates stack-specific code; calls 1, 4 and 9 are the ones a reviewer could argue inflate it. Variants A–D bracket the effect.
- Part of v2.32's scale-out growth is Windows NetworkDirect (4,014 lines) and IB resiliency code, not new GPU-initiated paths.
- RAS (10,766 lines in v2.32, all common) inflates the shared denominator; variant D removes it.
- v2.18's tuning code is not split, so the v2.18 share is slightly low relative to v2.32.
- Only `src/` is counted: external net plugins (e.g. vendor IB/EFA plugins) are not, which would add further scale-out code.

## Candidate claims for the paper (from results.md)

1. "In NCCL 2.32, 47% of the library's 113 K lines of NCCL-authored code (43–51% depending on how shared files are attributed) is specific to one transport stack: 36.6 K lines for scale-out (IB/RoCE, socket, CollNet, GIN, proxy) and 16.6 K for scale-up (P2P, SHM, NVLS, CE, LSA). Only 60 K lines are shared."
2. "Between NCCL 2.18 (2023) and 2.32 (2026), transport-specific code grew 6.6× (8.1 K to 53.1 K lines) while shared code grew only 3.6× (16.9 K to 60.1 K). Stack-specific code went from about a third to about half of the codebase (32% to 47%)."
3. "The two stacks also expose two unrelated contracts. Scale-out is reached through four versioned plugin APIs (ncclNet v6–v12, ncclCollNet v6–v12, ncclGin v13–v14, ncclRma v13–v16: 20 struct versions, up from 6 in 2.18). Scale-up has no plugin API and is hard-wired, and its GPU-side device API is separate: 82 LSA/multimem declarations versus 86 for GIN."

## Reproduce

Run from this directory. Each command rewrites `files_<tag>.csv` and `split_<tag>.csv` here and prints the summary to stdout (under a minute each; nix may first download cloc and ctags).

```sh
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl v2.32.3-1
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl-v2.18.5 v2.18.5-1
```

To check the stored summaries (no output from `diff` means identical; verified 2026-09-30, and the regenerated CSVs were byte-identical too):

```sh
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl v2.32.3-1 | diff - summary_v2.32.3-1.txt
nix shell nixpkgs#cloc nixpkgs#universal-ctags -c python3 categorize.py ../src/nccl-v2.18.5 v2.18.5-1 | diff - summary_v2.18.5-1.txt
```

## Files

- `README.md`: this file.
- `results.md`: full write-up with per-directory and per-file breakdowns, interface table and all judgment calls.
- `categorize.py`: classifier and counter (path rules, per-function splits, vendored list, interface counting).
- `summary_v2.32.3-1.txt`, `summary_v2.18.5-1.txt`: stdout of `categorize.py` for each tag.
- `files_v2.32.3-1.csv`, `files_v2.18.5-1.csv`: one row per file (path, language, code lines, category, vendored flag, split of mixed files).
- `split_v2.32.3-1.csv`, `split_v2.18.5-1.csv`: every function/struct region reassigned inside a mixed file.
