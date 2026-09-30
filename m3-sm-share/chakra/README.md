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

## Extension: per-layer MoE breakdown, waiting vs. moving, message sizes, SM-holding projection

This extension adds three scripts. Their outputs are saved next to them:

| script | inputs | output |
|---|---|---|
| `analyze_moe.py` | 8x7B Kineto `device_{0,2,3,6}.json` + 8x7B ETs (size cross-check) | `out_moe_Mixtral-8x7B.txt`, `moe_summary_Mixtral-8x7B.json` |
| `analyze_nccl_et.py` | all ETs of one model | `out_nccl_et_<model>.txt`, `nccl_et_<model>.json` |
| `project_offsm.py` | the two JSON summaries + testbed constants (hard-coded, listed in its docstring) | `out_projection.txt` |

No NeMo traces were present in `/home/harshanavkis/chakra-traces/nemo/`, either at the start or at the end of this work. The only Kineto/NeMo data is the 8x7B `nemo_raw` set that was already used.

### E1. Methodology

**Kernel → launching op (Kineto).** Each kernel's `correlation` id is joined to its `cuda_runtime` launch. The launch is then placed inside the enclosing `cpu_op` / `user_annotation` ranges on the same thread (a sweep-line nesting stack). This yields names such as `_LayerNormLinear`, `LinearWithGradAccumulationAndAsyncCommunication` and `_AllToAll`, which the Kineto kernel records do not carry.

**Layer segmentation (8x7B Kineto).** Megatron/TE kernels serve as anchors:
- A forward layer runs from the `rmsnorm_fwd` launched by `_LayerNormLinear` (input norm fused into QKV) to the next such kernel, or to the final-norm `rmsnorm_fwd`. The final norm is recognised as an `_RMSNorm` that directly follows another `_RMSNorm`.
- A backward layer runs from the end of the previous `_LayerNormLinearBackward` `rmsnorm_bwd_finalize` (or of the final-norm backward) to the end of its own.
- A kernel belongs to the layer in which it starts.

The script finds exactly 256 forward and 256 backward layer instances per rank (8 micro-batches × 32 layers). Every instance contains **exactly two** EP all-to-all kernels, each of 32 MiB. Together the instances cover 90.8–91.8% of `ProfilerStep#0`; the rest is embedding, LM head, loss and optimizer.

**Class assignment inside a layer.**
- **a2a_1 / a2a_2**: `SendRecv` kernels on the EP group (pg 57 or 58, 4 ranks), numbered by call order.
  - Forward: a2a_1 is the **dispatch** (tokens → experts) and a2a_2 is the **combine**. The CPU stacks confirm the order: router topk and bincount, then `_AllToAll`, then expert MLP, then `_AllToAll`.
  - Backward: a2a_1 is the backward of the combine (output grads → experts) and a2a_2 is the backward of the dispatch.
- **TP-MoE**: AG/RS on the 2-rank TP group that start between a2a_1 and a2a_2. These are the sequence-parallel gather and scatter of the dispatched tokens.
- **TP-attn**: all other TP AG/RS.
- **AR**: 1-CTA all-reduces (router aux-loss).
- **DP**: 4-rank data-parallel AG/RS.
- **expert_GEMM**: GEMM kernels launched under `LinearWithGradAccumulationAndAsyncCommunication` (Megatron `SequentialMLP` expert linears; attention uses TE `_LayerNormLinear`/`_Linear`).
- **MoE_other**: other compute between the two a2a (SwiGLU, cat/split).
- **dense_other**: everything else.
- **idle**: the layer span minus the union of all GPU activity.

Percentages are shares of the layer span. Because comm/compute overlap is only 1.5–3.3% of comm time, the class shares add up to about 100%.

**Bytes.** Bytes come from Kineto `In msg nelems` × dtype size. `analyze_moe.py` matched them per process group, in call order, against the Chakra ET `comm_size` of the same rank. All 3697 NCCL kernels of rank 0 match exactly, and so do those of ranks 2, 3 and 6, with durations within 1 µs. So **ET `comm_size` = input bytes**: the per-rank shard for AG, and the full buffer for RS, AR and a2a. I rely on this for 8x22B and Llama3.

**Bus bytes** follow the nccl-tests convention:
- AG: (n−1)·in
- RS: (n−1)/n·in
- AR: 2(n−1)/n·in
- a2a: (n−1)/n·in (equal splits: `In split size` is `[]` on every a2a, and all a2a in a layer have the same size)

**Achieved bandwidth** is bus bytes / kernel duration. **Per held SM** is that value divided by grid CTAs. NCCL uses one CTA per channel and one SM each; the grids are 32 for a2a and 24 for AG/RS.

**Link peaks.**
- NVLink: 450 GB/s per direction.
- InfiniBand: the metadata (`chakra_metadata.yaml`, both multi-node sets) says only `Inter-Node: type: InfiniBand-100Gbps, topology: Switch`. So 100 Gb/s = **12.5 GB/s** per NIC is verified. The number of NICs per node is **not** in the metadata, so I evaluate two bounds:
  - **A**: one 100 Gb/s NIC per GPU.
  - **B**: one 100 Gb/s NIC per node, shared by 8 GPUs.

**Lower bound on kernel duration, t_min.**
- Intra-node: bus / 450 GB/s.
- Inter-node a2a: remote bytes = (n − m_loc)/n · in, where m_loc = group members on this node (2 of 8 for 8x22B EP). A: remote / 12.5 GB/s. B: 8 × remote / 12.5 GB/s.
- Inter-node rings: A: bus / (m_loc · 12.5 GB/s). B: (8 / m_loc) · bus / 12.5 GB/s.

From these I derive two estimates of waiting:
- **Unexplained** = 1 − Σt_min / Σdur. This is an upper bound on the share of held time spent waiting, synchronising or paying latency rather than moving bytes at peak.
- **Excess over min**: for each (type, scope, n, bytes), the fastest kernel in the trace counts as "moving" time, and excess = Σ(dur − min) / Σdur. This is an empirical estimate of waiting and skew that does not depend on any peak number.

**Per-peer message size.**
- AG: `in` (each rank's shard travels to every peer).
- RS and AR: `in/n` (ring chunk).
- a2a: `in/n` (equal splits: verified on 8x7B, **assumed** on 8x22B, whose ET has no split info).
- broadcast: `in`.

"Latency-bound" means < 64 KiB per peer.

**Derived numbers.** Ratios and sums quoted in the prose, such as "6.3× the wire bound" or "47.5–55.1% of the layer", are plain arithmetic on values the scripts print. Everything in the tables is printed directly by the scripts.

### E2. Per-layer MoE breakdown (Mixtral-8x7B, Kineto, `out_moe_Mixtral-8x7B.txt`)

The table shows medians over the 256 layer instances of each pass, as ms (% of layer span). Excluding micro-batch 0 changes the medians by < 0.3 ms (see the output).

| class | fwd r0 | fwd r2 | fwd r3 | fwd r6 | bwd r0 | bwd r2 | bwd r3 | bwd r6 |
|---|---|---|---|---|---|---|---|---|
| layer span | 9.84 | 9.83 | 9.83 | 9.82 | 10.20 | 10.22 | 10.23 | 10.20 |
| a2a_1 (fwd: dispatch) | 3.80 (38.6) | 4.47 (46.2) | 3.88 (40.2) | 0.08 (0.8) | 2.23 (22.2) | 2.33 (22.9) | 1.84 (17.8) | 0.08 (0.8) |
| a2a_2 (fwd: combine) | 0.87 (8.9) | 0.86 (8.9) | 0.84 (8.5) | 0.08 (0.9) | 1.11 (10.7) | 1.11 (10.7) | 1.07 (10.3) | 0.08 (0.8) |
| TP-MoE (AG+RS) | 0.32 (3.2) | 0.32 (3.2) | 0.95 (9.7) | 0.31 (3.2) | 0.32 (3.1) | 0.33 (3.2) | 0.83 (8.2) | 0.31 (3.1) |
| TP-attn (AG+RS) | 0.18 (1.9) | 0.18 (1.9) | 0.20 (2.0) | 0.18 (1.8) | 0.40 (4.0) | 0.40 (3.9) | 0.43 (4.1) | 0.27 (2.6) |
| expert GEMM | 1.91 (19.4) | 1.91 (19.4) | 1.94 (19.7) | 1.94 (19.7) | 3.76 (36.8) | 3.76 (36.6) | 3.82 (37.2) | 3.83 (37.4) |
| MoE other compute | 0.29 (2.9) | 0.29 (2.9) | 0.28 (2.9) | 0.29 (2.9) | 0.43 (4.2) | 0.42 (4.2) | 0.43 (4.2) | 0.43 (4.2) |
| dense/other compute | 0.86 (8.8) | 0.86 (8.8) | 0.86 (8.8) | 0.87 (8.9) | 1.78 (17.1) | 1.77 (17.1) | 1.78 (17.1) | 1.73 (16.9) |
| GPU idle | 1.49 (14.8) | 0.86 (8.4) | 0.67 (6.7) | 5.95 (61.1) | 0.21 (2.0) | 0.21 (2.1) | 0.20 (1.9) | 3.41 (33.6) |

AR (aux-loss) and DP collectives are ≤ 0.2% of a layer.

Per-step totals over all layer instances on rank 0:
- a2a: 1285.0 + 438.2 ms
- TP-MoE: 292.3 ms
- TP-attn: 275.6 ms
- expert GEMM: 1451.8 ms
- idle: 786.4 ms

Observations:
- **Every MoE layer, forward and backward, issues 2 EP all-to-alls and 2 TP-MoE collectives.** This is exact for all 2048 layer instances (4 ranks × 512).
- **Both passes are dominated by communication.** On ranks 0/2/3, the two a2a take 47.5–55.1% of a forward MoE layer and 28–34% of a backward one. Expert GEMMs take 19–20% (forward) and 37% (backward). NCCL wall time (union) is a median 53.8–61.1% of forward layer spans on ranks 0/2/3.
- **The dispatch a2a is mostly waiting for a straggler.** The layer span is the same on all four ranks, at 9.8 ms forward. Rank 6 has 61% GPU idle in forward layers, and its a2a take the minimum time (0.08 ms). Ranks 0/2/3 spend 3.8–4.5 ms in the same dispatch. My inference: rank 6 is host/launch-bound, and the other ranks' dispatch kernels sit resident on 32 SMs until rank 6 joins.
- **A2a time grows with layer index.** Rank 0's per-layer-index medians of forward a2a time rise from 2.47 ms (L1) to about 5 ms (L9–L32).

### E3. Waiting vs. moving in NCCL kernels

#### Mixtral-8x7B (Kineto, 4 ranks pooled; all intra-node NVLink; `out_moe_Mixtral-8x7B.txt`)

| collective (group) | CTAs = SMs held | kernels | Σ dur ms | median busBW GB/s | max busBW GB/s | median GB/s per held SM | unexplained at 450 GB/s | excess over fastest same-size kernel |
|---|---|---|---|---|---|---|---|---|
| a2a EP (n=4, 32 MiB) | 32 | 4096 | 6199.8 | 24.1 | 315.7 | 0.75 | 96.3% | 94.7% (min 79.7 µs, median 1043.5 µs) |
| AllGather TP (n=2) | 24 | 5216 | 1252.8 | 186.7 | 228.1 | 7.78 | 78.4% | 62.1% (32 MiB), 41.0% (16 MiB) |
| ReduceScatter TP (n=2) | 24 | 4160 | 763.8 | 185.0 | 212.8 | 7.71 | 69.7% | 24.0% (64 MiB), 45.0% (32 MiB) |
| ReduceScatter DP (n=4) | 24 | 72 | 136.8 | 32.4 | – | 1.35 | 92.2% | 87.4% (80 MiB) |
| AllGather DP (n=4) | 24 | 72 | 33.3 | 217.4 | – | 9.06 | 67.9% | 49.6% (20 MiB) |
| small AllReduce/Broadcast (≤ 16 KiB) | 1–2 | 1170 | ≈110 | ≈0 | – | ≈0 | ≈100% | 90.1–99.2% (AR), 24.2% (bcast) |

Across all 14786 NCCL kernels (8496.3 ms), data movement at the 450 GB/s peak explains **8.9%** of the time, so 91.1% is unexplained. Relative to the fastest same-size kernel, **82.9%** is excess. Reaching the NVLink peak needs 14.1 GB/s per SM with 32 CTAs, or 18.8 with 24. The a2a median is 0.75 GB/s per held SM, and even the fastest a2a (315.7 GB/s, 70% of peak) is only 9.9 GB/s per SM.

The ET-only version (`out_nccl_et_Mixtral-8x7B.txt`, all 8 ranks) agrees:
- a2a: unexplained 96.1%, excess over min 94.5%.
- all NCCL: unexplained 91.6%, excess over min 83.9%.

#### Mixtral-8x22B (ET only, 32 ranks pooled; `out_nccl_et_Mixtral-8x22B.txt`)

| collective | scope | kernels | % NCCL time | µs per bus-MiB | median busBW GB/s | max busBW GB/s | unexplained (intra peak, or A) | unexplained (B) | excess over min |
|---|---|---|---|---|---|---|---|---|---|
| a2a EP8 (24 MiB; 6 of 7 peers remote) | inter | 28671 | 69.35 | 511.9 | 2.31 | 25.85 | 86.0% | −12.4% | 92.1% |
| AllGather TP4 | intra | 36224 | 17.21 | 42.0 | 209.72 | 298.41 | 94.5% | – | 90.6% (24 MiB), 92.1% (12 MiB) |
| AllGather DP8 (318 MiB) | inter | 64 | 7.14 | 222.4 | 4.73 | 5.30 | 81.1% | −50.9% | 11.1% |
| ReduceScatter DP8 | inter | 32 | 3.67 | 228.6 | 4.54 | 5.29 | 81.7% | −46.8% | 13.3% |
| ReduceScatter TP4 | intra | 28928 | 1.43 | 4.1 | 263.98 | 291.50 | 43.0% | – | 7.6% (96 MiB), 15.1% (48 MiB) |

- Intra-node NCCL time is 90.7% unexplained at peak. Inter-node time is 85.5% unexplained under A.
- **Assumption B (one 100 Gb/s NIC per node) is contradicted by the trace.** 69.4% of the 24 MiB a2a kernels finish faster than B's 12.08 ms floor, and every DP ring kernel beats B's floor. Assumption A is consistent: its floor for the a2a is 1.51 ms, and 0.1% of kernels beat it.
- a2a duration percentiles: p0 852 µs, p10 6481, p50 9527, p90 17082, p99 26916 µs.

**Are the RDMA-path kernels held longer per byte? Yes.** The inter-node EP a2a holds its kernel for 511.9 µs per bus-MiB:
- 12.2× the intra-node TP all-gather (42.0 µs/MiB);
- 125× the TP reduce-scatter (4.1 µs/MiB).

At the median, the per-byte gap is 91–114× (busBW 209.7 and 264.0 vs. 2.31 GB/s). For the fastest kernels it is 11.3–11.5×. For scale, the link-peak ratio is 36×. So the median gap is larger than the link difference. The caveat is that the TP AG itself is 90.6–92.1% excess-over-min (p90 6.3–8.0 ms against a 137–253 µs minimum): it waits for ranks delayed by the a2a. The ET has no grids, so **per-held-SM bandwidth cannot be computed for any inter-node kernel.**

#### Llama3-70B (TP16 over 2 nodes; all inter-node; `out_nccl_et_Llama3-70B.txt`)

- AG (8 MiB shard) and RS (128 MiB) run at a median 21.2–21.3 GB/s bus (max 25.2–25.9), which is 51.5–52.0 µs per bus-MiB.
- Unexplained under A is 79.6–79.8%. B is again contradicted: 98.7–99.1% of kernels beat its floor.
- **Excess over min is only 20.0–21.3%.** Unlike the MoE runs, these kernels are consistently slow, not waiting. p0 is 4.99 ms and p50 5.95 ms, against A's 1.26 ms floor. The time is transfer over the inter-node path at about 21 GB/s bus, which is 2 NICs' worth under A (inference).

### E4. Message-size distribution (`out_nccl_et_<model>.txt`, "Per-peer message size histogram")

Per-peer sizes and their share of NCCL kernel time, pooled over ranks:

| model | collective (scope, n) | per-peer size bin | ops | % of model's NCCL time |
|---|---|---|---|---|
| 8x7B | a2a (intra, 4) | 4–16 MiB (8 MiB) | 8192 | 65.6 |
| 8x7B | AG / RS (intra, 2) | 16–64 MiB | 10432 / 8320 | 18.2 / 11.3 |
| 8x7B | AR (intra, 2) | < 1 KiB | 2048 | 2.5 |
| 8x22B | a2a (inter, 8) | 1–4 MiB (3 MiB) | 28671 | 69.3 |
| 8x22B | AG TP (intra, 4) | 4–16 MiB / 16–64 MiB | 21888 / 14336 | 8.5 / 8.7 |
| 8x22B | AG / RS DP (inter, 8) | ≥ 64 MiB | 64 / 32 | 7.1 / 3.7 |
| 8x22B | AR (intra 4 / inter 8) | < 1 KiB (mostly) | 7168 / 128 | 0.1 / 0.4 |
| Llama3 | AG / RS (inter, 16) | 4–16 MiB | 247290 / 164861 | 60.2 / 39.7 |

Latency-bound share (< 64 KiB per peer):

| model | % of NCCL ops | % of NCCL time | intra ops / time | inter ops / time |
|---|---|---|---|---|
| Mixtral-8x7B | 7.9 | 3.06 | 7.9 / 3.06 | – |
| Mixtral-8x22B | 7.7 | 0.83 | 10.4 / 0.61 | 0.9 / 0.89 |
| Llama3-70B | 0.3 | 0.06 | – | 0.3 / 0.06 |

In these Megatron runs, every sub-64 KiB operation is a tiny all-reduce or broadcast (aux-loss, grad-norm, flags). No bulk collective is latency-bound, and the smallest bulk per-peer message is 3 MiB (the 8x22B a2a). Latency-optimised paths are therefore not the lever for these traces; SM residency during large, skewed transfers is. This holds for these configurations only. Token-dropless or fine-grained, overlapped MoE kernels would issue much smaller messages.

### E5. PROJECTION: compute lost to SM-holding, and what an off-SM engine recovers (`out_projection.txt`)

> Everything in this subsection is a **projection**. It combines trace inputs with the H200 testbed curve: expert GEMM throughput with k SMs held = 90.5/72.7/48.9% (measured co-location) and 90.3/88.8/79.1% (perfect partitioning) for k = 8/16/20.

**Assumptions.**
- 8x7B NCCL kernels hold 32 (a2a) or 24 (AG/RS) SMs. The curve stops at k = 20, so θ(20) serves as a **lower bound** on the penalty. This assumes GEMM throughput does not recover as k grows.
- S1 "as traced": only the traced comm/compute overlap O is slowed, so the loss is O·(1−θ).
- S2 "overlapped MoE": expert GEMMs are scheduled concurrently with the MoE comm (EP a2a + TP), as an overlapped MoE implementation would do. The co-runnable GEMM work is W = min(E, θ·C), and the loss is W·(1/θ − 1).
- C is either the traced held time ("held", including waiting) or the sum of fastest-same-size durations ("moving").
- An off-SM engine (θ = 1) recovers the whole loss.

| rank (8x7B) | step ms | E expert GEMM ms | C held / moving ms | S1 lost ms | S2-held lost ms (% step) | S2-moving lost ms (% step) | NCCL SM-time as full-GPU ms (% step) |
|---|---|---|---|---|---|---|---|
| 0 | 5872 | 1452 | 2015 / 352 | 16–39 | 384–1030 (6.5–17.5%) | 74–180 (1.3–3.1%) | 537 (9.1%) |
| 2 | 5873 | 1451 | 2370 / 352 | 15–36 | 384–1211 (6.5–20.6%) | 74–180 (1.3–3.1%) | 604 (10.3%) |
| 3 | 5873 | 1472 | 2372 / 352 | 14–34 | 389–1212 (6.6–20.6%) | 74–180 (1.3–3.1%) | 588 (10.0%) |
| 6 | 5874 | 1475 | 664 / 352 | 3–6 | 139–339 (2.4–5.8%) | 74–180 (1.3–3.1%) | 173 (2.9%) |

The "moving" C is identical across ranks by construction: per-size minima × identical kernel counts.

- **k sweep (rank 0, S2-held).** k = 8 gives 152–156 ms, k = 16 gives 183–545 ms, and k = 20 gives 384–1030 ms. Even an 8-SM comm kernel, such as a lean GPU-initiated design, costs about 10% of expert GEMM time when co-scheduled.
- **Mixtral-8x22B** (no grids, so k is swept; medians over 32 ranks). E = 1121 ms (includes the LM head). C held/moving = 12215 / 1149 ms. S2-held losses are 118–120, 141–421 and 296–1172 ms for k = 8, 16 and 20, which is 10–11%, 13–38% and 26–104% of E.
- **Posting cost vs. held time** (8x22B inter-node a2a, one put per remote peer, 6 peers, 896 a2a per rank per step). IBGDA would spend 36–48 µs of SM time per a2a, or 32.3–43.0 SM-ms per step. The traced NCCL kernel is resident for a mean 10749 µs per a2a on all its CTAs. A CPU proxy would spend 0.45 µs of CPU per a2a (0.40 ms per step). Chunking multiplies the per-put costs.

**Reading.**
- In the run as traced, communication is barely overlapped, so SM-holding costs little compute directly (S1: < 0.7% of the step). The cost appears as exposed communication, 39–43% of the step (earlier section).
- Once the communication is overlapped with expert compute (what a Loom-style or any overlapped MoE design needs), SMs held by the comm kernel cost 1.3–3.1% of the step if the kernels only moved data, and 6.5–20.6% if they also hold SMs while waiting, as they do in this trace. An off-SM engine recovers that range.

### E6. Caveats (extension)

- **Kineto coverage.** Only 4 of 8 ranks and one step. Rank 6 is a straggler, so the per-rank a2a medians differ by 50× between ranks. Which rank waits depends on who is late, but the SMs are held either way.
- **Layer attribution.** Anchors are TE RMSNorm kernels and Megatron autograd names specific to this NeMo/Megatron build. Expert GEMMs are identified by the `LinearWithGradAccumulationAndAsyncCommunication` launcher. In the ET-based totals for 8x22B that class also includes the LM head. On Llama3 (dense) the same class totals 147 ms per rank, which is an estimate of that overcount's scale for a TP16 head. On 8x7B the ET total (1474.5 ms median) exceeds the in-layer Kineto total (1451.8 ms on rank 0) by about 1.5%.
- **Link peaks and bounds.** 450 GB/s is the NVLink spec; nccl-tests typically reaches less, so "unexplained" is an upper bound on waiting. "Excess over min" can overstate the moving time, because the minimum kernel may itself include some waiting, which makes excess a lower bound on waiting. It can also understate it, if the fastest kernel benefited from peers that had already posted. NCCL kernel names such as `RING_LL` are the default specialisation and do not identify the runtime protocol.
- **NIC count.** 12.5 GB/s per NIC is from the metadata, but the NIC count is not. The trace rules out one NIC per node (B). A (one per GPU) fits all but 0.1% of the 8x22B a2a kernels. The faster ones suggest that some a2a traffic leaves through additional NICs, e.g. NCCL PXN (inference).
- **a2a equal splits** are verified on 8x7B and assumed for 8x22B. One 8x22B a2a kernel (13.4 ms) is on a pg that the usage-based membership inference sees on a single rank. It is listed as "intra, n=1" and ignored.
- **Projection.** The testbed curve covers k ≤ 20, whereas the traced kernels hold 24–32 SMs, so the projection uses θ(20) as a bound. The GEMM shapes in the trace (e.g., FC1 fwd grid 56×2 = 112 CTAs, less than 132 SMs) differ from the testbed GEMM. S2 assumes a scheduler that co-runs expert GEMMs with communication, which the traced run does not do.

### E7. New candidate claims (numbers from `out_moe_*.txt`, `out_nccl_et_*.txt`, `out_projection.txt`)

4. "In Mixtral-8x7B (TP2 × EP4 on 8 H200s), every MoE layer, forward and backward, issues two expert-parallel all-to-alls and two tensor-parallel collectives. In a median forward layer (9.8 ms), the dispatch all-to-all alone holds 32 SMs for 3.8–4.5 ms (39–46%) on three of four profiled ranks, while the expert GEMMs take 1.9 ms (19%). The same 32 MiB all-to-all completes in 80 µs (316 GB/s) when no peer is late; 94.7% of all-to-all kernel time is excess over that minimum, i.e., SMs held while waiting."
5. "Across 14,786 NCCL kernels in the Mixtral-8x7B step, data movement at NVLink peak explains only 8.9% of kernel residency. The median all-to-all moves 0.75 GB/s per held SM, versus the 14 GB/s per SM needed to saturate NVLink with its 32 CTAs."
6. "In Mixtral-8x22B on 32 H200s, the inter-node (RDMA-path) expert all-to-all holds its kernel 12× longer per byte than the intra-node tensor-parallel all-gather (512 vs. 42 µs per MiB) and 125× longer than the reduce-scatter. Its median (9.5 ms) is 6.3× the 100 Gb/s wire bound and 11× its own fastest instance. Only 0.06–3.1% of NCCL time, in any of the three traces, is in latency-bound (< 64 KiB per peer) messages." A companion projection, clearly labelled: "if expert GEMMs were overlapped with this communication while it holds ≥ 20 SMs, 6.5–20.6% of the Mixtral-8x7B step would be lost to slowed GEMMs (1.3–3.1% if the kernels only moved data), which an off-SM engine would recover."

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
# extension (E1-E5); args: NVLink GB/s/dir, [IB GB/s per NIC]
python3 analyze_moe.py 450 $T/Mixtral/Mixtral-8x7B $D/device_0.json $D/device_2.json $D/device_3.json $D/device_6.json > out_moe_Mixtral-8x7B.txt   # ~10 s
python3 analyze_nccl_et.py Mixtral-8x7B  8 450 12.5 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et         > out_nccl_et_Mixtral-8x7B.txt
python3 analyze_nccl_et.py Mixtral-8x22B 8 450 12.5 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et  > out_nccl_et_Mixtral-8x22B.txt
python3 analyze_nccl_et.py Llama3-70B    8 450 12.5 $T/Llama3/Llama3-70B/16TP/rank.*.et               > out_nccl_et_Llama3-70B.txt     # ~35 s, 16 procs
python3 project_offsm.py > out_projection.txt   # reads moe_summary_Mixtral-8x7B.json, nccl_et_Mixtral-8x22B.json
```

The scripts use only the Python standard library (tested with the system `python3`). `analyze_et.py` also writes per-rank aggregates to `et_summary_<model>.json`.
