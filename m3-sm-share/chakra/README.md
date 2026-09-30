# M3 (trace part): communication share, scale-up vs. scale-out, and NCCL SM-time in MLCommons Chakra traces

## Question

Loom's thesis is that a GPU today runs two separate communication stacks: scale-up (intra-node NVLink/P2P) and scale-out (inter-node RDMA). It also holds that communication both costs SMs and sits on the critical path. From real training traces we want to know:

1. What fraction of step time is communication, split into scale-up and scale-out collectives?
2. Do both stacks appear in the same layer or step, e.g. MoE with TP collectives over NVLink and EP all-to-all over RDMA?
3. Where kernel launch geometry is available, what fraction of SM-time is spent in NCCL kernels?

## Trace provenance and format

The source is the MLCommons Chakra Open Trace Library, downloaded from Google Drive as split zips into `/home/harshanavkis/chakra-traces/`. The zips were not modified. The parts contain disjoint files under common top-level directories (`Llama3/`, `Mixtral/`), so all parts are extracted into one directory, `/scratch/harshanavkis/chakra-traces/`, using `extract.sh`. The host has no `unzip`, so the script uses Python's `zipfile`.

| set | files | format | what it carries |
|---|---|---|---|
| Llama3-70B | `Llama3/Llama3-70B/16TP/rank.{0..15}.et` (~615 MB each) + `chakra_metadata.yaml` | converted **Chakra ET protobuf** (schema `1.1.1-chakra.0.0.4`) | per GPU op: name, stream, `duration_micros`; NCCL ops: `comm_type`, `comm_size` (bytes), `pg_name`. **No start timestamps (all 0), no grid/block dims, no PG rank lists.** |
| Mixtral-8x22B | `Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.{0..31}.et` (~88 MB each) + `chakra_metadata.yaml` | converted Chakra ET protobuf | same as above |
| Mixtral-8x7B | `Mixtral/Mixtral-8x7B/chakra_trace.{0..7}.et` | converted Chakra ET protobuf | same as above |
| Mixtral-8x7B raw | nested zip `nemo-chakra-mixtral-8x7B-traces.zip` → `nemo_raw/host_{0..7}.json` and `nemo_raw/device_{0,2,3,6}.json` | host: PyTorch ET JSON; device: **Kineto/PyTorch-profiler JSON** | Kineto has kernel timestamps, durations, `grid`/`block`, stream, and on NCCL kernels `Collective name`, `Process Group Ranks`, and msg sizes. `distributedInfo.pg_config` has every process group's rank list. Device traces exist **only for ranks 0, 2, 3, 6**. |

Duplicates I verified with `cmp` (byte-identical) and ignored:
- `Llama3/Llama3-70B/llama3-70B_chakra.3.et` is the same file as `16TP/rank.3.et`.
- `Copy of nemo-chakra-mixtral-8x7B-traces.zip` is the same file as the original zip.
- `Mixtral/This_is_trash/{host_3,host_5,device_0}.json` are the same files as the `nemo_raw` copies. The two `.sh` files there are the `trace_link`/`converter` commands used to produce the `.et` files.

### Configuration

| | Llama3-70B | Mixtral-8x22B | Mixtral-8x7B |
|---|---|---|---|
| ranks (trace files) | 16 | 32 | 8 |
| GPU | H200 (metadata) | H200 (metadata) | H200, 132 SMs (Kineto `deviceProperties`) |
| GPUs/node | 8 (metadata: DGX-H200, NVL8) | 8 (metadata) | 8 (**assumed**; no metadata yaml) |
| nodes | 2 (metadata `node_used: 2`; 16/8) | **4 inferred** (32 ranks / 8). Metadata says `node_used: 2`, which cannot hold 32 ranks at 8 GPUs/node, so I treat the yaml value as a copy-paste error. | 1 (inferred: world_size 8; output path `results_1_H200_2TP_1CP_4EP_1DP_1PP`) |
| parallelism | TP16, PP1, DP1 (metadata) | TP4, EP8, PP1 (metadata); DP=8 over the EP groups (from PGs) | TP2, EP4, CP1, PP1 (from Kineto `traceName`) |
| inter-node fabric | InfiniBand 100 Gb/s, switch (metadata) | same (metadata) | n/a (single node) |
| micro-batches in the step | 32 (32 DataLoader calls in the ET; global 32 / micro 1) | 4 (4 DataLoader calls) | 8 (8 DataLoader calls) |
| traced | 1 training step (`ProfilerStep#0`) | 1 step | 1 step |

The Llama3 TP16 group spans both nodes, so every TP collective in that run crosses InfiniBand. This is an unusual configuration and a pathological one. It explains the extreme communication share below.

## Methodology

**Decoding.** `chakra_et.py` is a minimal pure-Python protobuf wire-format decoder for `et_def.proto` (length-delimited `GlobalMetadata`, then `Node` messages). No protobuf package is needed. `analyze_et.py` scans all rank files of a model in parallel. It keeps GPU nodes only (`is_cpu_op == false`, node types COMP or COMM_*). NCCL ops are the nodes that carry `comm_type`, and all of them are `COMM_COLL_NODE`. The EP all-to-all appears as `ncclDevKernel_SendRecv` with `comm_type = ALL_TO_ALL`.

**Process-group membership (ET traces).** The converted ETs keep only `pg_name`. Membership of a pg_name is **inferred** as the set of ranks whose ET issues a collective on it. This relies on torch/Megatron giving `new_group` a globally consistent name counter, so every group has a distinct name on all ranks. I validated the inference on Mixtral-8x7B: the inferred groups exactly match the explicit `pg_config` rank lists in rank 0's Kineto trace:
- pg 5 = [0,2,4,6]
- pg 22 = [0,1]
- pg 57 = [0,2,4,6]
- pg 1 = [0,2,4,6]
- pg 52 = [0,1]
- pg 0 = all

**Intra vs. inter.** node(rank) = rank // 8. This assumes the standard contiguous torchrun rank-to-node placement. A group is intra-node if all its members are on one node; otherwise it is inter-node. For Mixtral-8x22B the classification does not depend on the exact G:
- TP groups {4k..4k+3} are intra-node for any G that is a multiple of 4.
- The EP and DP groups {j, j+4, ..., j+28} span nodes for any G < 32.

A handful of NCCL nodes have `pg_name = None`: 1–3 kernels on 7 Llama ranks and 1 kernel on 8x22B rank 20, together less than 0.02% of comm time. They are classified by the same rule and do not affect any number.

**Communication time (ET).** This is the sum of NCCL kernel `duration_micros`. The ETs have no timestamps, so there is no overlap analysis and no true step time for Llama3 or 8x22B. Instead I report three things:
- **comm / GPU-kernel-sum**: NCCL kernel time divided by the sum of all GPU op durations over all streams.
- **per-stream busy time**: kernels on one stream serialize, so the busiest stream is a lower bound on GPU step time.
- **ET window**: `finish_ts - start_ts` in ms, an upper bound on step time. On 8x7B the ET window is 10.03 s while the Kineto `ProfilerStep#0` is 5.87 s, so this bound is loose.

**Kineto analysis (Mixtral-8x7B ranks 0, 2, 3, 6; `analyze_kineto.py`).**
- **Window**: all GPU activity is clipped to the GPU-side `ProfilerStep#0` annotation.
- **Communication wall time**: the union of NCCL kernel intervals. Compute wall time is the union of non-NCCL kernels plus memcpy/memset.
- **Overlap**: |comm ∩ compute|. Exposed communication is comm wall time minus overlap.
- **SM-time**: each kernel is charged min(gridX·gridY·gridZ, 132 SMs) × duration. This is exact for NCCL, which launches one CTA per channel and one CTA per SM. For compute kernels it is an upper bound on residency.

I report NCCL SM-time two ways:
- as a share of all kernel SM-time;
- as a share of SM capacity, 132 × step time.

Each Kineto JSON (about 155 MB) is loaded whole with the stdlib `json` module; the host has 1.5 TB RAM.

**Important semantic caveat.** NCCL kernel duration covers the time the kernel is resident: waiting for peers plus moving data. It is not wire time. That is the right quantity for "SMs held by communication" and for "communication on the critical path". It overstates the network transfer itself, especially under rank skew; see 8x7B rank 6 below.

## Results

### Mixtral-8x7B (single node, TP2 + EP4, Kineto; all communication is scale-up)

From `out_kineto_Mixtral-8x7B.txt`:

| rank | step (ms) | NCCL wall % of step | % of NCCL overlapped with compute | exposed comm % of step | NCCL % of kernel SM-time | NCCL % of SM capacity |
|---|---|---|---|---|---|---|
| 0 | 5872.0 | 40.3 | 3.3 | 39.0 | 19.0 | 9.15 |
| 2 | 5873.2 | 44.1 | 2.7 | 43.0 | 20.9 | 10.28 |
| 3 | 5873.3 | 44.4 | 2.6 | 43.3 | 20.4 | 10.01 |
| 6 | 5873.5 | 13.8 | 1.5 | 13.6 | 7.0 | 2.94 |

Rank 0 breakdown (sum of kernel durations; all groups intra-node):

| type | pg | kernels | ms | % step | CTAs/kernel |
|---|---|---|---|---|---|
| ALL_TO_ALL (EP, SendRecv) | 57 [0,2,4,6] | 1024 | 1723.2 | 29.3 | 32 |
| ALL_GATHER (TP 22, DP 5) | 22, 5 | 1322 | 346.6 | 5.9 | 24 |
| REDUCE_SCATTER | 22, 5 | 1058 | 308.3 | 5.3 | 24 |
| ALL_REDUCE | 0, 1, 22, 52 | 291 | 31.6 | 0.5 | 1–2 (one with 24) |

Observations:
- While an NCCL kernel runs it holds about 28–31 SMs (the "avg SMs held" line in the output).
- Compute wall time is about 42% of the step on every rank.
- Rank 6 has the least NCCL time and the lowest GPU busy (55.6% vs. 81–85%). This suggests the other ranks spend part of their NCCL time waiting for rank 6, which may be host-bound. That is an inference.
- The ET (`analyze_et.py`) gives the same rank 0 NCCL sum, 2408.0 ms vs. 2409.7 ms from Kineto. This cross-validates the two readers.

### Mixtral-8x22B (32 ranks, 4 nodes inferred, TP4 intra + EP8/DP8 inter)

From `out_et_Mixtral-8x22B.txt`, representative ranks 0 and 17:

| type | class | pg | kernels | rank 0 ms | rank 17 ms | rank 0 % of comm |
|---|---|---|---|---|---|---|
| ALL_TO_ALL (EP8, SendRecv) | **inter** | 173 / 174 | 896 | 9639.7 | 9817.3 | 69.9 |
| ALL_GATHER (TP4) | **intra** | 58 / 62 | 1132 | 2420.3 | 2249.3 | 17.6 |
| ALL_GATHER (DP8, optimizer) | inter | 9 / 11 | 2 | 930.7 | 1023.1 | 6.8 |
| REDUCE_SCATTER (DP8, grads) | inter | 9 / 11 | 1 | 443.8 | 557.2 | 3.2 |
| REDUCE_SCATTER (TP4) | intra | 58 / 62 | 904 | 198.8 | 198.2 | 1.4 |
| ALL_REDUCE | intra / inter | various | 237 / 6 | 122.2 / 29.3 | 4.5 / 48.1 | 1.1 |

Across all 32 ranks:
- NCCL time / GPU-kernel-sum is 0.885–0.889 (median 0.887).
- The inter-node share of NCCL time is 0.789–0.834 (median 0.808).

Rank 0 detail:
- Stream busy: EP-a2a stream 9639.7 ms; TP-collective stream 2732.6 ms; compute stream (7) 1769.2 ms; DP stream 1374.5 ms.
- So step time is at least 9.64 s (the a2a stream alone) and at most 19.52 s (the ET window).
- NCCL sum / ET window = 0.706.

**Both stacks in every layer.** Per micro-batch per layer, the trace issues TP all-gather/reduce-scatter on the intra-node group {4k..4k+3} and EP all-to-all on the inter-node group {j, j+4, ..., j+28}. The count matches exactly: 896 all-to-all kernels = 4 micro-batches × 56 layers × 4 (dispatch + combine, forward + backward). This is inferred from counts, because the ET has no timestamps.

### Llama3-70B (16 ranks, 2 nodes, TP16 spanning both nodes; all communication is scale-out)

From `out_et_Llama3-70B.txt`. Only two NCCL groups exist: pg 83 (TP, all 16 ranks) and pg 0 (world). Both are inter-node.

| type | rank 0 kernels | rank 0 ms | % of comm | % of GPU-kernel-sum |
|---|---|---|---|---|
| ALL_GATHER (TP/SP) | 15456 | 96637.1 | 60.3 | 55.3 |
| REDUCE_SCATTER (TP/SP) | 10304 | 63527.1 | 39.6 | 36.3 |
| ALL_REDUCE | 66 | 128.1 | 0.1 | 0.1 |

Across ranks:
- NCCL time / GPU-kernel-sum is 0.916–0.917.
- The inter-node share is 1.000.

Rank 0 detail:
- Stream busy: comm stream 56 = 160278.7 ms; compute stream 7 = 14565.4 ms.
- ET window = 196679 ms, so step time is between 160.3 s and 196.7 s.

This run is dominated by TP over 100 Gb/s InfiniBand, so treat it as an outlier configuration rather than a typical one.

## Caveats

- **Timestamps.** The converted ETs have no timestamps, so overlap and true step time are available only for Mixtral-8x7B (Kineto, 4 of 8 ranks). For Llama3 and 8x22B, the "fraction of step" figures are fractions of summed GPU kernel time, with step-time bounds given separately.
- **Launch geometry.** Grid/CTA counts exist only in the 8x7B Kineto traces, which cover a single node. There is **no SM-time measurement for any inter-node (RDMA) collective**. I did not extrapolate the 24/32 CTA counts to 8x22B or Llama3, because NCCL's channel count depends on topology.
- **Kernel duration is not wire time.** NCCL kernel duration includes peer-wait and skew. 8x7B rank 6 shows that the per-rank communication share can vary by about 3× within the same step.
- **Placement assumptions.** 8 GPUs per node and contiguous rank-to-node placement are assumptions. For 8x22B the metadata's `node_used: 2` contradicts the 32 rank files. The intra/inter classification of the 8x22B TP and EP groups holds for any contiguous node size that is a multiple of 4 and below 32.
- **Inferred membership.** PG membership in the ET traces is inferred from usage. It was validated against explicit rank lists only for 8x7B.
- **Unusual configurations.** These are academic-cluster traces (Georgia Tech PACE, NeMo/Megatron, H200 with 100 Gb/s IB), each with one traced step. Llama3-70B at TP16 across nodes is unusual, and the IB bandwidth is low for H200 systems. The communication shares are therefore likely higher than in tuned production runs.

## Candidate one-sentence claims (numbers from the outputs above)

1. "In a Mixtral-8x7B training step on 8 H200s (TP2 × EP4), NCCL kernels ran for 40–44% of the 5.87 s step on three of four profiled ranks (13.8% on the fourth). Only 1.5–3.3% of that time overlapped compute, and while running they held about 30 of 132 SMs, which is 19–21% of all kernel SM-time (7% on the fourth rank)."
2. "In Mixtral-8x22B on 32 H200s (TP4 × EP8), every MoE layer uses both stacks: tensor-parallel all-gather/reduce-scatter inside a node and expert-parallel all-to-all across 4 nodes. NCCL kernels account for 88.7% (median) of GPU kernel time, and 81% of that is inter-node."
3. "In Mixtral-8x22B the inter-node expert all-to-all alone keeps a stream busy for 9.6 s per step, 5.4× the 1.77 s of the main compute stream. For Llama3-70B with TP16 spanning two nodes, 100% of communication is inter-node and makes up 91.7% of GPU kernel time."

## Reproduce

```sh
cd /scratch/harshanavkis/loom-proj/motivation-experiments/m3-sm-share/chakra
./extract.sh                                   # -> /scratch/harshanavkis/chakra-traces (≈1 min)
T=/scratch/harshanavkis/chakra-traces
python3 analyze_et.py Mixtral-8x7B  8 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et        > out_et_Mixtral-8x7B.txt
python3 analyze_et.py Mixtral-8x22B 8 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et > out_et_Mixtral-8x22B.txt
python3 analyze_et.py Llama3-70B    8 $T/Llama3/Llama3-70B/16TP/rank.*.et              > out_et_Llama3-70B.txt   # ~1 min, 16 procs
D=$T/Mixtral/Mixtral-8x7B/nemo_raw
python3 analyze_kineto.py 8 $D/device_0.json $D/device_2.json $D/device_3.json $D/device_6.json > out_kineto_Mixtral-8x7B.txt
```

The scripts use only the Python standard library (tested with the system `python3`). `analyze_et.py` also writes per-rank aggregates to `et_summary_<model>.json`.
