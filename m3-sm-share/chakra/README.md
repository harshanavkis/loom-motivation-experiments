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
| Mixtral-8x7B raw | nested zip `nemo-chakra-mixtral-8x7B-traces.zip` → `nemo_raw/host_{0..7}.json` and `nemo_raw/device_{0,2,3,6}.json`; `device_{1,4,5,7}.json` were added later (same run), so device traces now exist for **all 8 ranks** | host: PyTorch ET JSON; device: **Kineto/PyTorch-profiler JSON** | Kineto has kernel timestamps, durations, `grid`/`block`, stream, and on NCCL kernels `Collective name`, `Process Group Ranks`, and msg sizes. `distributedInfo.pg_config` has every process group's rank list. Host ET `record_param_comms` nodes carry the ProcessGroupNCCL sequence number (used in E8). |

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

**Kineto analysis (Mixtral-8x7B, all 8 ranks; `analyze_kineto.py`).**
- **Window**: all GPU activity is clipped to the GPU-side `ProfilerStep#0` annotation.
- **Communication wall time**: the union of NCCL kernel intervals. Compute wall time is the union of non-NCCL kernels plus memcpy/memset.
- **Overlap**: |comm ∩ compute|. Exposed communication is comm wall time minus overlap.
- **SM-time**: each kernel is charged min(gridX·gridY·gridZ, 132 SMs) × duration. This is exact for NCCL, which launches one CTA per channel and one CTA per SM. For compute kernels it is an upper bound on residency.

I report NCCL SM-time two ways:
- as a share of all kernel SM-time;
- as a share of SM capacity, 132 × step time.

Each Kineto JSON (about 155 MB) and host ET JSON (about 226 MB) is stream-parsed by `jstream.py` (stdlib `json.JSONDecoder.raw_decode` over 16 MiB chunks; only the event categories a script uses are kept). Earlier versions loaded the whole file with `json.load`; with the streaming loader, the outputs of `analyze_kineto.py`, `analyze_moe.py` and `project_offsm.py` for ranks 0/2/3/6 are byte-identical to the earlier ones (checked with `diff`).

**Important semantic caveat.** NCCL kernel duration covers the time the kernel is resident: waiting for peers plus moving data. It is not wire time. That is the right quantity for "SMs held by communication" and for "communication on the critical path". It overstates the network transfer itself, especially under rank skew; see 8x7B rank 6 below and E8, which splits each kernel at the arrival of the last peer.

## Results

### Mixtral-8x7B (single node, TP2 + EP4, Kineto; all communication is scale-up)

From `out_kineto_Mixtral-8x7B.txt` (all 8 ranks):

| rank | step (ms) | NCCL wall % of step | % of NCCL overlapped with compute | exposed comm % of step | NCCL % of kernel SM-time | NCCL % of SM capacity |
|---|---|---|---|---|---|---|
| 0 | 5872.0 | 40.3 | 3.3 | 39.0 | 19.0 | 9.15 |
| 1 | 5873.1 | 38.5 | 2.1 | 37.7 | 17.8 | 8.35 |
| 2 | 5873.2 | 44.1 | 2.7 | 43.0 | 20.9 | 10.28 |
| 3 | 5873.3 | 44.4 | 2.6 | 43.3 | 20.4 | 10.01 |
| 4 | 5873.4 | 38.3 | 2.7 | 37.2 | 18.7 | 8.92 |
| 5 | 5873.5 | 45.2 | 2.7 | 43.9 | 19.9 | 9.69 |
| 6 | 5873.5 | 13.8 | 1.5 | 13.6 | 7.0 | 2.94 |
| 7 | 5873.6 | 36.9 | 4.6 | 35.2 | 14.5 | 6.69 |

Rank 0 breakdown (sum of kernel durations; all groups intra-node):

| type | pg | kernels | ms | % step | CTAs/kernel |
|---|---|---|---|---|---|
| ALL_TO_ALL (EP, SendRecv) | 57 [0,2,4,6] | 1024 | 1723.2 | 29.3 | 32 |
| ALL_GATHER (TP 22, DP 5) | 22, 5 | 1322 | 346.6 | 5.9 | 24 |
| REDUCE_SCATTER | 22, 5 | 1058 | 308.3 | 5.3 | 24 |
| ALL_REDUCE | 0, 1, 22, 52 | 291 | 31.6 | 0.5 | 1–2 (one with 24) |

Observations:
- While an NCCL kernel runs it holds about 28–31 SMs on ranks 0–6 and 23.9 on rank 7 (the "avg SMs held" line in the output).
- Compute wall time is 41.5–42.3% of the step on every rank.
- Rank 6 has the least NCCL time and the lowest GPU busy (55.6% vs. 77.5–85.9%). Rank 7, its TP partner, is second lowest in NCCL time. E8 measures directly that the other ranks spend most of their NCCL time waiting for ranks 6 and 7, and gives evidence that rank 6 is host-bound (the host-bound part is an inference).
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

- **Timestamps.** The converted ETs have no timestamps, so overlap and true step time are available only for Mixtral-8x7B (Kineto, all 8 ranks). For Llama3 and 8x22B, the "fraction of step" figures are fractions of summed GPU kernel time, with step-time bounds given separately.
- **Launch geometry.** Grid/CTA counts exist only in the 8x7B Kineto traces, which cover a single node. There is **no SM-time measurement for any inter-node (RDMA) collective**. I did not extrapolate the 24/32 CTA counts to 8x22B or Llama3, because NCCL's channel count depends on topology.
- **Kernel duration is not wire time.** NCCL kernel duration includes peer-wait and skew. 8x7B rank 6 shows that the per-rank communication share can vary by about 3× within the same step. E8 measures the effect on matched instances: 82.8% of 8x7B NCCL kernel time passes before the last peer arrives.
- **Placement assumptions.** 8 GPUs per node and contiguous rank-to-node placement are assumptions. For 8x22B the metadata's `node_used: 2` contradicts the 32 rank files. The intra/inter classification of the 8x22B TP and EP groups holds for any contiguous node size that is a multiple of 4 and below 32.
- **Inferred membership.** PG membership in the ET traces is inferred from usage. It was validated against explicit rank lists only for 8x7B.
- **Unusual configurations.** These are academic-cluster traces (Georgia Tech PACE, NeMo/Megatron, H200 with 100 Gb/s IB), each with one traced step. Llama3-70B at TP16 across nodes is unusual, and the IB bandwidth is low for H200 systems. The communication shares are therefore likely higher than in tuned production runs.

## Candidate one-sentence claims (numbers from the outputs above)

1. "In a Mixtral-8x7B training step on 8 H200s (TP2 × EP4), NCCL kernels ran for 37–45% of the 5.87 s step on seven of eight ranks (13.8% on the eighth, a straggler). Only 1.5–4.6% of that time overlapped compute, and while running they held about 24–31 of 132 SMs, which is 14.5–20.9% of all kernel SM-time on those seven ranks (7.0% on the straggler)." Caveat from E8: most of this kernel time is spent waiting for late peers, not moving data.
2. "In Mixtral-8x22B on 32 H200s (TP4 × EP8), every MoE layer uses both stacks: tensor-parallel all-gather/reduce-scatter inside a node and expert-parallel all-to-all across 4 nodes. NCCL kernels account for 88.7% (median) of GPU kernel time, and 81% of that is inter-node."
3. "In Mixtral-8x22B the inter-node expert all-to-all alone keeps a stream busy for 9.6 s per step, 5.4× the 1.77 s of the main compute stream. For Llama3-70B with TP16 spanning two nodes, 100% of communication is inter-node and makes up 91.7% of GPU kernel time."

## Extension: per-layer MoE breakdown, waiting vs. moving, message sizes, SM-holding projection

This extension adds four scripts and a shared loader. Their outputs are saved next to them:

| script | inputs | output |
|---|---|---|
| `analyze_moe.py` | 8x7B Kineto `device_{0..7}.json` + 8x7B ETs (size cross-check) | `out_moe_Mixtral-8x7B.txt`, `moe_summary_Mixtral-8x7B.json` |
| `analyze_nccl_et.py` | all ETs of one model | `out_nccl_et_<model>.txt`, `nccl_et_<model>.json` |
| `project_offsm.py` | the two JSON summaries + testbed constants (hard-coded, listed in its docstring) | `out_projection.txt` |
| `analyze_skew.py` (E8) | 8x7B Kineto `device_{0..7}.json` + host ETs `host_{0..7}.json`; reuses `analyze_moe.classify()` | `out_skew_Mixtral-8x7B.txt` |
| `jstream.py` | – | streaming JSON reader used by all Kineto/host-ET readers |

E1–E5 were first run on ranks 0/2/3/6, the only device traces available then. All tables below are now re-run on all 8 ranks with unchanged methodology; the 4-rank outputs reproduce byte-for-byte with the current scripts.

No NeMo traces were present in `/home/harshanavkis/chakra-traces/nemo/`, either at the start or at the end of this work. The only Kineto/NeMo data is the 8x7B `nemo_raw` set that was already used.

### E1. Methodology

**Kernel → launching op (Kineto).** Each kernel's `correlation` id is joined to its `cuda_runtime` launch. The launch is then placed inside the enclosing `cpu_op` / `user_annotation` ranges on the same thread (a sweep-line nesting stack). This yields names such as `_LayerNormLinear`, `LinearWithGradAccumulationAndAsyncCommunication` and `_AllToAll`, which the Kineto kernel records do not carry.

**Layer segmentation (8x7B Kineto).** Megatron/TE kernels serve as anchors:
- A forward layer runs from the `rmsnorm_fwd` launched by `_LayerNormLinear` (input norm fused into QKV) to the next such kernel, or to the final-norm `rmsnorm_fwd`. The final norm is recognised as an `_RMSNorm` that directly follows another `_RMSNorm`.
- A backward layer runs from the end of the previous `_LayerNormLinearBackward` `rmsnorm_bwd_finalize` (or of the final-norm backward) to the end of its own.
- A kernel belongs to the layer in which it starts.

The script finds exactly 256 forward and 256 backward layer instances on each of the 8 ranks (8 micro-batches × 32 layers). Every instance contains **exactly two** EP all-to-all kernels, each of 32 MiB. Together the instances cover 90.8–91.8% of `ProfilerStep#0`; the rest is embedding, LM head, loss and optimizer.

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

Percentages are shares of the layer span. Because comm/compute overlap is only 1.5–4.6% of comm time, the class shares add up to about 100%.

**Bytes.** Bytes come from Kineto `In msg nelems` × dtype size. `analyze_moe.py` matched them per process group, in call order, against the Chakra ET `comm_size` of the same rank. All 3697 NCCL kernels of every rank match exactly (29,576 kernels over the 8 ranks), with durations within 1 µs. So **ET `comm_size` = input bytes**: the per-rank shard for AG, and the full buffer for RS, AR and a2a. I rely on this for 8x22B and Llama3.

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

Each rank's value is the median over its 256 layer instances of that pass. The table gives the median of the 8 per-rank medians, the range over the 8 ranks, and ranks 6 and 7 separately, because they set most of the ranges. Values are ms (% of layer span); the median and range of the % column are taken separately over ranks. The across-rank columns are printed in the "across ranks" section of `out_moe_Mixtral-8x7B.txt`, and the r6/r7 values come from the per-rank sections. Excluding micro-batch 0 changes any per-rank median by at most 0.34 ms (rank 4, forward idle).

| class | fwd: median of 8 ranks | fwd: range over ranks | fwd r6 / r7 | bwd: median of 8 ranks | bwd: range over ranks | bwd r6 / r7 |
|---|---|---|---|---|---|---|
| layer span | 9.83 | 9.82–9.85 | 9.82 / 9.85 | 10.21 | 10.18–10.23 | 10.20 / 10.23 |
| a2a_1 (fwd: dispatch) | 3.77 (36.9) | 0.08–4.47 (0.8–46.2) | 0.08 (0.8) / 0.08 (0.9) | 1.81 (17.3) | 0.08–2.33 (0.8–22.9) | 0.08 (0.8) / 0.08 (0.8) |
| a2a_2 (fwd: combine) | 0.85 (8.8) | 0.08–0.88 (0.9–9.0) | 0.08 (0.9) / 0.08 (0.9) | 1.10 (10.5) | 0.08–1.13 (0.8–10.8) | 0.08 (0.8) / 0.08 (0.8) |
| TP-MoE (AG+RS) | 0.63 (6.5) | 0.31–1.53 (3.2–15.0) | 0.31 (3.2) / 1.53 (15.0) | 0.58 (5.7) | 0.31–1.43 (3.1–13.6) | 0.31 (3.1) / 1.43 (13.6) |
| TP-attn (AG+RS) | 0.19 (1.9) | 0.18–1.25 (1.8–12.8) | 0.18 (1.8) / 1.25 (12.8) | 0.40 (3.9) | 0.27–1.49 (2.6–14.6) | 0.27 (2.6) / 1.49 (14.6) |
| AR (aux-loss) | 0.02 (0.2) | 0.01–0.76 (0.1–7.7) | 0.01 (0.1) / 0.76 (7.7) | 0 | 0 | 0 / 0 |
| expert GEMM | 1.92 (19.5) | 1.90–1.94 (19.3–19.7) | 1.94 (19.7) / 1.92 (19.5) | 3.78 (36.9) | 3.74–3.83 (36.5–37.4) | 3.83 (37.4) / 3.80 (37.0) |
| MoE other compute | 0.29 (2.9) | 0.28–0.29 (2.9) | 0.29 (2.9) / 0.28 (2.9) | 0.42 (4.2) | 0.42–0.43 (4.1–4.2) | 0.43 (4.2) / 0.42 (4.1) |
| dense/other compute | 0.86 (8.8) | 0.86–0.87 (8.8–8.9) | 0.87 (8.9) / 0.86 (8.8) | 1.77 (17.1) | 1.73–1.81 (16.9–17.7) | 1.73 (16.9) / 1.81 (17.7) |
| GPU idle | 1.21 (12.1) | 0.67–5.95 (6.7–61.1) | 5.95 (61.1) / 2.81 (28.6) | 0.21 (2.1) | 0.20–3.41 (1.9–33.6) | 3.41 (33.6) / 0.20 (2.0) |

DP collectives are 0 at the median of every rank. The TP-MoE median is bimodal on ranks 0–5: 0.32 ms forward on the even ranks (0, 2, 4) and 0.95 ms on the odd ranks (1, 3, 5). In E8, the even rank of each of these pairs is the later arriver that causes 71.5–81.2% of the pair's TP-MoE wait time, so the odd rank waits longer.

Per-step totals over all layer instances on rank 0:
- a2a: 1285.0 + 438.2 ms
- TP-MoE: 292.3 ms
- TP-attn: 275.6 ms
- expert GEMM: 1451.8 ms
- idle: 786.4 ms

Observations:
- **Every MoE layer, forward and backward, issues 2 EP all-to-alls and 2 TP-MoE collectives.** This is exact for all 4096 layer instances (8 ranks × 512).
- **Both passes are dominated by communication on six of eight ranks.** On ranks 0–5, the two a2a take 45.3–55.1% of a forward MoE layer and 23.0–33.6% of a backward one (sums of the two per-rank medians). On all 8 ranks, expert GEMMs take 19.3–19.7% (forward) and 36.5–37.4% (backward). NCCL wall time (union) is a median 51.9–61.1% of forward layer spans on ranks 0–5, 40.4% on rank 7 and 6.9% on rank 6.
- **The dispatch a2a is mostly waiting for a straggler.** The layer span is the same on all eight ranks, 9.82–9.85 ms forward. Ranks 6 and 7, which form one TP pair with one rank in each EP group, are the only ranks whose a2a take the minimum time (0.08 ms). Rank 6 has 61% GPU idle in forward layers. Rank 7 instead spends its forward layers in TP collectives and the aux-loss AR (1.53 + 1.25 + 0.76 ms), waiting for rank 6. Ranks 0–5 spend 3.7–4.5 ms in the same dispatch. E8 measures this directly on matched instances and shows that ranks 6 and 7 are the last arrivers.
- **A2a time grows with layer index.** Rank 0's per-layer-index medians of forward a2a time rise from 2.47 ms (L1) to about 5 ms (L9–L32).

### E3. Waiting vs. moving in NCCL kernels

#### Mixtral-8x7B (Kineto, all 8 ranks pooled; all intra-node NVLink; `out_moe_Mixtral-8x7B.txt`)

"Max busBW" is the largest per-rank maximum in the per-rank table of the output.

| collective (group) | CTAs = SMs held | kernels | Σ dur ms | median busBW GB/s | max busBW GB/s | median GB/s per held SM | unexplained at 450 GB/s | excess over fastest same-size kernel |
|---|---|---|---|---|---|---|---|---|
| a2a EP (n=4, 32 MiB) | 32 | 8192 | 11755.1 | 24.2 | 316.2 | 0.76 | 96.1% | 94.5% (min 79.6 µs, median 1040.0 µs) |
| AllGather TP (n=2) | 24 | 10432 | 3267.0 | 167.9 | 228.1 | 7.00 | 83.4% | 70.9% (32 MiB), 54.9% (16 MiB) |
| ReduceScatter TP (n=2) | 24 | 8320 | 2035.3 | 183.6 | 212.8 | 7.65 | 77.3% | 40.3% (64 MiB), 61.0% (32 MiB) |
| ReduceScatter DP (n=4) | 24 | 144 | 261.0 | 35.3 | 289.4 | 1.47 | 91.8% | 86.6% (80 MiB), 87.6% (125 MiB) |
| AllGather DP (n=4) | 24 | 144 | 66.1 | 206.3 | 301.0 | 8.59 | 67.6% | 49.9% (20 MiB), 49.2% (31 MiB) |
| small AllReduce/Broadcast (< 64 KiB per peer) | 1–2 | 2334 | ≈550 | ≈0 | – | ≈0 | ≈100% | 74.0–99.2% (AR), 69.3% (bcast) |

Across all 29,574 NCCL kernels (17,936.7 ms), data movement at the 450 GB/s peak explains **8.4%** of the time, so 91.6% is unexplained. Relative to the fastest same-size kernel, **83.8%** is excess. Reaching the NVLink peak needs 14.1 GB/s per SM with 32 CTAs, or 18.8 with 24. The a2a median is 0.76 GB/s per held SM, and even the fastest a2a (316.2 GB/s, 70% of peak) is only 9.9 GB/s per SM.

The ET-only version (`out_nccl_et_Mixtral-8x7B.txt`, all 8 ranks) agrees with the 8-rank Kineto numbers to within 0.1 points:
- a2a: unexplained 96.1%, excess over min 94.5%.
- all NCCL: unexplained 91.6%, excess over min 83.9%.

"Excess over the fastest kernel" is an indirect estimate of waiting. E8 measures waiting directly: each matched instance is split at the moment the last peer's kernel starts.

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

> **Caveat added after E8 (do not quote the median gap as a transport cost).** E8 shows that on Mixtral-8x7B, 82.8% of NCCL kernel time (93.6% for the all-to-all) is arrival skew, i.e. waiting for the last peer. The 8x22B ET has no timestamps, so its medians cannot be split into skew and transfer, and they very likely contain skew too. The skew-free comparison is the fastest instance: the inter-node a2a moves 25.85 GB/s at its minimum (852 µs) against 298 GB/s for the intra-node AG (253 µs), an 11.5× gap. That is *below* the 36× link-peak ratio, so at its best the inter-node path is roughly link-bound. These traces therefore support the *structure* claim (both fabrics in every layer, with the inter-node all-to-all as the largest NCCL component) but not a claim that the remote path loses beyond its link bandwidth.

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
| 0 | 5872 | 1452 | 2015 / 351 | 16–39 | 384–1030 (6.5–17.5%) | 73–180 (1.3–3.1%) | 537 (9.1%) |
| 1 | 5873 | 1447 | 1948 / 351 | 10–25 | 382–996 (6.5–17.0%) | 73–180 (1.3–3.1%) | 491 (8.4%) |
| 2 | 5873 | 1451 | 2370 / 351 | 15–36 | 384–1211 (6.5–20.6%) | 73–180 (1.3–3.1%) | 604 (10.3%) |
| 3 | 5873 | 1472 | 2372 / 351 | 14–34 | 389–1212 (6.6–20.6%) | 73–180 (1.3–3.1%) | 588 (10.0%) |
| 4 | 5873 | 1441 | 2054 / 351 | 13–31 | 381–1050 (6.5–17.9%) | 73–180 (1.3–3.1%) | 524 (8.9%) |
| 5 | 5873 | 1462 | 2197 / 351 | 15–37 | 386–1123 (6.6–19.1%) | 73–180 (1.3–3.1%) | 569 (9.7%) |
| 6 | 5874 | 1475 | 664 / 351 | 3–6 | 139–339 (2.4–5.8%) | 73–180 (1.3–3.1%) | 173 (2.9%) |
| 7 | 5874 | 1462 | 1286 / 351 | 21–51 | 269–657 (4.6–11.2%) | 73–180 (1.3–3.1%) | 393 (6.7%) |

The "moving" C is identical across ranks by construction: per-size minima × identical kernel counts. With 8 ranks pooled, the minima are slightly lower than with 4, so C moving changed from 352 to 351 ms.

- **k sweep (rank 0, S2-held).** k = 8 gives 152–156 ms, k = 16 gives 183–545 ms, and k = 20 gives 384–1030 ms. Even an 8-SM comm kernel, such as a lean GPU-initiated design, costs about 10% of expert GEMM time when co-scheduled.
- **Mixtral-8x22B** (no grids, so k is swept; medians over 32 ranks). E = 1121 ms (includes the LM head). C held/moving = 12215 / 1149 ms. S2-held losses are 118–120, 141–421 and 296–1172 ms for k = 8, 16 and 20, which is 10–11%, 13–38% and 26–104% of E.
- **Posting cost vs. held time** (8x22B inter-node a2a, one put per remote peer, 6 peers, 896 a2a per rank per step). IBGDA would spend 36–48 µs of SM time per a2a, or 32.3–43.0 SM-ms per step. The traced NCCL kernel is resident for a mean 10749 µs per a2a on all its CTAs. A CPU proxy would spend 0.45 µs of CPU per a2a (0.40 ms per step). Chunking multiplies the per-put costs.

**Reading.**
- In the run as traced, communication is barely overlapped, so SM-holding costs little compute directly (S1: < 0.9% of the step). The cost appears as exposed communication: 35–44% of the step on seven of eight ranks, and 13.6% on rank 6 (earlier section).
- Once the communication is overlapped with expert compute (what a Loom-style or any overlapped MoE design needs), SMs held by the comm kernel cost 1.3–3.1% of the step if the kernels only moved data. They cost 6.5–20.6% on ranks 0–5 (4.6–11.2% on rank 7, 2.4–5.8% on rank 6) if they also hold SMs while waiting, as they do in this trace. An off-SM engine recovers that range.
- E8 shows that the waiting is arrival skew, caused mostly by one host-starved rank and propagated through its TP partner. Removing the straggler would also shrink the gap between "held" and "moving", so the upper end of the range is specific to this skewed run (inference).

### E6. Caveats (extension)

- **Kineto coverage.** All 8 ranks, but only one step. Ranks 6 and 7 are the late arrivers (E8), so the per-rank a2a medians differ by up to 54× between ranks (4.47 vs. 0.08 ms forward dispatch). Which rank waits depends on who is late, but the SMs are held either way.
- **Layer attribution.** Anchors are TE RMSNorm kernels and Megatron autograd names specific to this NeMo/Megatron build. Expert GEMMs are identified by the `LinearWithGradAccumulationAndAsyncCommunication` launcher. In the ET-based totals for 8x22B that class also includes the LM head. On Llama3 (dense) the same class totals 147 ms per rank, which is an estimate of that overcount's scale for a TP16 head. On 8x7B the ET total (1474.5 ms median) exceeds the in-layer Kineto total (1451.8 ms on rank 0) by about 1.5%.
- **Link peaks and bounds.** 450 GB/s is the NVLink spec; nccl-tests typically reaches less, so "unexplained" is an upper bound on waiting. "Excess over min" can overstate the moving time, because the minimum kernel may itself include some waiting, which makes excess a lower bound on waiting. It can also understate it, if the fastest kernel benefited from peers that had already posted. NCCL kernel names such as `RING_LL` are the default specialisation and do not identify the runtime protocol.
- **NIC count.** 12.5 GB/s per NIC is from the metadata, but the NIC count is not. The trace rules out one NIC per node (B). A (one per GPU) fits all but 0.1% of the 8x22B a2a kernels. The faster ones suggest that some a2a traffic leaves through additional NICs, e.g. NCCL PXN (inference).
- **a2a equal splits** are verified on 8x7B and assumed for 8x22B. One 8x22B a2a kernel (13.4 ms) is on a pg that the usage-based membership inference sees on a single rank. It is listed as "intra, n=1" and ignored.
- **Projection.** The testbed curve covers k ≤ 20, whereas the traced kernels hold 24–32 SMs, so the projection uses θ(20) as a bound. The GEMM shapes in the trace (e.g., FC1 fwd grid 56×2 = 112 CTAs, less than 132 SMs) differ from the testbed GEMM. S2 assumes a scheduler that co-runs expert GEMMs with communication, which the traced run does not do.

### E7. New candidate claims (numbers from `out_moe_*.txt`, `out_nccl_et_*.txt`, `out_projection.txt`)

4. "In Mixtral-8x7B (TP2 × EP4 on 8 H200s), every MoE layer, forward and backward, issues two expert-parallel all-to-alls and two tensor-parallel collectives. In a median forward layer (9.8 ms), the dispatch all-to-all alone holds 32 SMs for 3.7–4.5 ms (37–46%) on six of eight ranks, while the expert GEMMs take 1.9 ms (19%). The same 32 MiB all-to-all completes in 80 µs (316 GB/s) when no peer is late; 94.5% of all-to-all kernel time is excess over that minimum, i.e., SMs held while waiting." (E8 confirms the waiting directly: it is arrival skew.)
5. "Across 29,574 NCCL kernels in the Mixtral-8x7B step (all 8 ranks), data movement at NVLink peak explains only 8.4% of kernel residency. The median all-to-all moves 0.76 GB/s per held SM, versus the 14 GB/s per SM needed to saturate NVLink with its 32 CTAs."
6. "In Mixtral-8x22B on 32 H200s, the inter-node (RDMA-path) expert all-to-all holds its kernel 12× longer per byte than the intra-node tensor-parallel all-gather (512 vs. 42 µs per MiB) and 125× longer than the reduce-scatter. Its median (9.5 ms) is 6.3× the 100 Gb/s wire bound and 11× its own fastest instance. Only 0.06–3.1% of NCCL time, in any of the three traces, is in latency-bound (< 64 KiB per peer) messages." A companion projection, clearly labelled: "if expert GEMMs were overlapped with this communication while it holds ≥ 20 SMs, 6.5–20.6% of the Mixtral-8x7B step on six of eight ranks would be lost to slowed GEMMs (1.3–3.1% if the kernels only moved data), which an off-SM engine would recover."

### E8. Arrival skew vs. communication cost, all 8 ranks (`analyze_skew.py`, `out_skew_Mixtral-8x7B.txt`)

A reviewer can argue that "NCCL kernels mostly wait" only means load imbalance. To separate arrival skew from the cost of communication itself, this section matches every collective instance across the ranks of its process group and splits each rank's kernel at the moment the last member's kernel starts. Every number below is printed by `analyze_skew.py`; ratios in the prose are plain arithmetic on printed values, and inferences are marked.

#### E8.1 Method

- **Inputs.** `device_{0..7}.json` (Kineto) and `host_{0..7}.json` (host ET), stream-parsed. Every rank has 3697 NCCL kernels. Kernel classes come from `analyze_moe.classify()`, i.e. the E1 rules unchanged; kernels outside layer instances are labelled "outside layers: <role>".
- **Matching.** An instance is (process group, k), where k counts this rank's NCCL kernels on that pg in start order. Each pg uses exactly one stream per rank. The checks are:
  - every member has the same kernel count;
  - type, In/Out nelems and dtype are equal on all members;
  - the ProcessGroupNCCL **sequence number** is equal on all members. It is taken from the host ET: Kineto kernel `External id` → `record_param_comms` cpu_op → `Record function id` = host-ET `rf_id` → the `(seq, isP2P)` input.
- **Decomposition.** It uses instances whose kernels start inside each member's GPU `ProfilerStep#0`, excluding broadcasts. Per instance:
  - LA = last arrival = max start; FE = first end = min end.
  - Per rank: wait_i = LA − start_i (skew) and post_i = end_i − LA.
  - The movement bound mb is bus bytes / 450 GB/s, with the E1 bus-bytes convention.
  - Time-weighted shares of Σdur:
    - **skew** = Σwait;
    - **moving@peak** = Σmin(post, mb);
    - **post-arrival excess** = Σmax(0, post − mb).
  - The three shares sum to 100%. post < mb never occurs (0 of 29,560 rank-kernels), so the clipping has no effect.
- **SM-time.** SMs held = min(grid CTAs, 132), as in `analyze_kineto.py`, multiplied by each component and expressed as a share of 132 × the rank's GPU `ProfilerStep#0`.
- **Stragglers.** For each instance: the last arriver and the wait it imposes on the other members. Two indicators of host-boundness:
  - launch lead = kernel start − end of the launching runtime call;
  - GPU busy share.

#### E8.2 Clock alignment: verified; no offset applied

- `baseTimeNanoseconds` is identical on all 8 ranks (1743521598000000000). Kineto `ts` are µs since that base, from the shared host clock.
- `ProfilerStep#0` starts within **34.9 µs** across ranks on the CPU and within **44.2 µs** on the GPU. The GPU step ends spread over 1553.7 µs, rising with rank number. This spread is not a clock offset; the causality bounds below rule out offsets of this size. It occurs at the very end of the step, where the last NCCL operations are two 8-rank broadcasts from rank 0 on pg 0. Those broadcasts finish in the same rank order 0 → 7, as a ring broadcast would: the first ends at +109, +218, …, +736 µs relative to rank 0's start (printed in the output). The exact mechanism behind the step-end order is an inference.
- CPU → GPU on each rank: min(kernel start − launch-call start) is 5.1–5.4 µs on every rank. No kernel starts before its launch.
- **Cross-rank causality.** In AllGather, ReduceScatter, AllReduce and all-to-all, no member can finish before every member has started. Of the 12,638 non-broadcast matched instances, **0** have first_end < last_arrival. The minimum first_end − last_arrival is 5.6 µs (AR), 73.7 µs (RS), 73.8 µs (AG) and 79.6 µs (a2a).
- **Feasible offsets.** Writing measured = true + x_r, causality gives x_i − x_j ≤ min(end_i − start_j). Solving over all pairs (Floyd–Warshall), every rank's offset relative to rank 0 lies in **[−15.3, +15.3] µs**: rank 1 [−5.6, +6.5], up to rank 5 [−15.3, +14.1]. Zero is feasible for every rank, with no negative cycle. Zero is also feasible in the first and second halves of the step separately, so no drift is visible at this resolution.
- **No offset is applied.** The ±15 µs uncertainty is about 1% of the median a2a arrival spread (1180 µs).
- The 2 broadcasts are excluded. The second has first_end − last_arrival = −652 µs: the root (rank 0) finishes before rank 7 starts, which is legal for a broadcast.

#### E8.3 Matching: exact

- The trace has 15 NCCL process groups:
  - world pg 0;
  - pgs 1/3 (AR) and 5/7 (DP) on {0,2,4,6} / {1,3,5,7};
  - TP 22–25 and 52–55 on {0,1} … {6,7};
  - EP 57/58.
- On every pg, all members have the same kernel count.
- **0** type/nelems/dtype mismatches.
- **0** sequence-number mismatches. On every pg, the sequence numbers are consecutive in kernel order, with no gaps, and identical on every member (e.g., EP 10241..11264, TP 21763..24131).
- Every Kineto NCCL kernel links to a host-ET `record_param_comms`, with matching pg and In-nelems.
- This gives 12,640 instances, and no instance has members that disagree on the E1 class label.
- Type and bytes alone could not catch an off-by-one shift here, because every a2a is 32 MiB. The sequence numbers and the causality check do catch it.

#### E8.4 Decomposition (8 ranks pooled; 17,921.6 of 17,936.7 in-step NCCL ms covered)

| class | instances | Σ dur ms | **skew %** | moving@peak % | post-arrival excess % | median dur µs | median post-arrival µs | movement bound µs | median arrival spread (LA − first start) µs |
|---|---|---|---|---|---|---|---|---|---|
| EP a2a fwd dispatch | 512 | 5407.8 | **96.5** | 2.1 | 1.4 | 3369.3 | 94.1 | 55.9 | 4063.9 |
| EP a2a fwd combine | 512 | 1449.9 | **87.1** | 7.9 | 5.0 | 840.8 | 93.9 | 55.9 | 782.2 |
| EP a2a bwd a2a_1 (combine-grad) | 512 | 3162.8 | **94.0** | 3.6 | 2.4 | 1611.0 | 94.0 | 55.9 | 2168.8 |
| EP a2a bwd a2a_2 (dispatch-grad) | 512 | 1734.6 | **89.2** | 6.6 | 4.2 | 1065.2 | 93.9 | 55.9 | 1056.5 |
| EP a2a, all | 2048 | 11755.1 | **93.6** | 3.9 | 2.5 | 1040.0 | 94.0 | 55.9 | 1179.8 |
| TP-MoE (fwd + bwd) | 4096 | 3151.5 | **61.6** | 19.4 | 19.1 | 164.4 | 150.9 | 74.6 | 475.7 |
| TP-attn (fwd + bwd) | 5120 | 2021.4 | **50.4** | 18.9 | 30.7 | 99.6 | 88.7 | 37.3 | 29.5 |
| AR (aux-loss and others) | 1142 | 537.2 | **96.8** | 0.0 | 3.2 | 10.8 | 6.9 | 0.0 | 221.5 |
| DP (in and outside layers) | 72 | 327.1 | **78.6** | 13.1 | 8.3 | 688.9 | 223.4 | 140.1 | 1111.3 |
| **all NCCL** | 12638 | 17921.6 | **82.8** | 8.4 | 8.8 | 163.7 | 93.9 | 55.9 | 298.4 |

Notes on the table:
- The forward/backward splits for TP, AR and DP are in the output. TP collectives outside layer instances (160 instances, 129.4 ms, 77.0% skew) are counted only in the "all NCCL" row.
- In 2-rank groups, half of the rank-kernels belong to the last arriver, whose wait is 0. Their median wait is therefore about 0 by construction, so the table gives the arrival spread instead (mean waits are in the output).

**32 MiB EP all-to-all, post-arrival time:**
- **minimum 79.6 µs, median 94.0 µs** (p10 82.1, p90 96.5; median/min = 1.18), against a 55.9 µs movement bound (24 MiB bus bytes at 450 GB/s).
- At the median, the post-arrival a2a runs at 267.8 GB/s bus, 60% of the NVLink peak.
- In each instance, the first member finishes a median 82.5 µs after the last arrival (min 79.6) and the last member a median 95.9 µs after it (min 93.1).
- For comparison, the median kernel lasts 1040.0 µs and the median arrival spread is 1179.8 µs (p90 4363.7 µs).
- The fastest-kernel minimum used in E3 (79.6 µs) equals the minimum post-arrival time: the fastest kernels belong to the last arriver.

**Reading (plain).**
- **In this trace the waiting is mostly skew.** 82.8% of NCCL kernel time passes before the last peer has arrived (93.6% for the expert all-to-all, 96.5% for the forward dispatch).
- The cost of communication itself, from the last arrival to the end, is 17.2% of NCCL kernel time: 8.4 points at the NVLink movement bound and 8.8 points above it.
- For the 32 MiB a2a, the post-arrival time is 1.7× the movement bound. The TP collectives are less skewed (50–62%), and their post-arrival time is 2.0–2.4× the bound, 184–227 GB/s bus at the median post-arrival time.
- So the reviewer's reading is right for the waiting: it is arrival skew. As E8.6 shows, compute and expert work are balanced across ranks to within 2.3%, so the skew does not come from imbalanced expert or token load. It comes from one host-starved rank and its TP partner (inference).

#### E8.5 SM-time held while waiting vs. while communicating (% of 132 SMs × step)

| rank | NCCL SM-time in step | waiting for peers | post-arrival (moving@peak + excess) | waiting share of NCCL SM-time |
|---|---|---|---|---|
| 0 | 9.15 | 7.83 | 1.31 (0.64 + 0.67) | 85.6% |
| 1 | 8.35 | 7.09 | 1.26 (0.64 + 0.62) | 84.9% |
| 2 | 10.28 | 8.95 | 1.33 (0.64 + 0.69) | 87.0% |
| 3 | 10.01 | 8.70 | 1.31 (0.64 + 0.67) | 86.9% |
| 4 | 8.92 | 7.59 | 1.32 (0.64 + 0.68) | 85.1% |
| 5 | 9.69 | 8.39 | 1.30 (0.64 + 0.66) | 86.6% |
| 6 | 2.94 | 1.68 | 1.26 (0.64 + 0.62) | 57.2% |
| 7 | 6.69 | 5.53 | 1.16 (0.64 + 0.52) | 82.6% |
| **all 8** | **8.25** | **6.97** | **1.28 (0.64 + 0.64)** | **84.5%** |

- Over the node, NCCL kernels that are resident but waiting for peers hold **6.97%** of all SM capacity. Kernels communicating after all peers have arrived hold **1.28%**, 5.4× less.
- The expert all-to-all alone holds 5.68% of capacity while waiting and 0.39% (0.24 + 0.15) after arrival. TP-MoE holds 0.75% vs. 0.47%, and TP-attn 0.39% vs. 0.39%.
- The post-arrival share is almost the same on every rank, 1.16–1.33%. The waiting share varies 5× (1.68–8.95%) and is smallest on the late rank.

#### E8.6 Stragglers: the TP pair {6, 7}, with rank 6 the root

| group | class | instances | last arriver (count) | share of the group's wait caused by the last arriver |
|---|---|---|---|---|
| EP {0,2,4,6} | a2a | 1024 | **r6 732**, r0 188, r4 69, r2 35 | **r6 74.7%**, r0 16.0%, r4 5.9%, r2 3.4% |
| EP {1,3,5,7} | a2a | 1024 | **r7 693**, r1 220, r5 59, r3 52 | **r7 65.6%**, r1 24.2%, r5 5.7%, r3 4.5% |
| TP {6,7} | TP-MoE / TP-attn / AR | 1024 / 1280 / 256 | **r6** 948 / 1140 / 255 | **r6** 90.6% / 98.7% / 100% |
| DP {0,2,4,6} / {1,3,5,7} | AG+RS in layers | 30 / 30 | r6 30 / r7 18 | r6 100% / r7 91.7% |

- **Consistently the same ranks.** Rank 6 is the last arriver in 177–188 of 256 instances in every one of the four a2a sub-classes of its EP group, and rank 7 in 164–183 of 256 in the other group.
- Over all classes, of the 14,840 ms of total wait (summed over ranks), rank 6 causes 39.3% and rank 7 24.5% as last arriver. Ranks 0 and 1 cause 9.6% and 10.8%, and ranks 2–5 cause 2.6–6.6%.
- The earlier suspicion about rank 6 is confirmed, and rank 7, its TP partner, is added.
- TP pairs without ranks 6/7 still show 46–57% skew. The two members of such a pair sit in different EP groups, which release them at different times (inference).

**Why ranks 6 and 7 are late:**

| rank | GPU busy % of step | non-NCCL kernel ms | expert GEMM ms | median launch lead of compute kernels µs | compute kernels starting < 20 µs after launch returns | wait in TP AG/RS (for its TP partner), ms |
|---|---|---|---|---|---|---|
| 0 | 80.8 | 2460.6 | 1451.8 | 2869.0 | 17.8% | 304.0 |
| 1 | 79.2 | 2441.0 | 1446.9 | 1886.9 | 18.0% | 426.8 |
| 2 | 84.7 | 2454.0 | 1451.5 | 4044.6 | 11.5% | 120.3 |
| 3 | 85.4 | 2474.5 | 1472.1 | 4149.8 | 10.5% | 371.1 |
| 4 | 78.8 | 2440.3 | 1441.5 | 2578.3 | 20.4% | 96.4 |
| 5 | 85.9 | 2465.9 | 1462.0 | 4053.2 | 11.2% | 522.1 |
| 6 | **55.6** | 2468.2 | 1474.9 | **145.3** | **47.2%** | 69.6 |
| 7 | 77.5 | 2485.1 | 1462.5 | 874.9 | 24.2% | **1149.0** |

- **Not compute or expert-load imbalance.** Every rank runs 2440–2485 ms of non-NCCL kernels and 1441.5–1474.9 ms of expert GEMMs (within 2.3%). Every a2a has equal 32 MiB splits (`In split size` = `[]`).
- **Rank 6 is host-starved (inference).** Its GPU is busy 55.6% of the step. Its compute kernels start a median 145 µs after their launch call returns, and 47.2% start within 20 µs. On ranks 0–5 the host runs a median 1.9–4.1 ms ahead of the GPU. So rank 6's GPU runs only as fast as its host thread issues work. The trace does not show why the host is slow (e.g., CPU contention or extra host work in that process).
- **Rank 7 inherits the delay (inference).** It is less host-starved (875 µs lead) but spends 1149 ms waiting for rank 6 in their TP collectives, the most of any rank. It therefore arrives late in EP group {1,3,5,7}.
- **Weaker per-kernel signal.** The a2a kernel's own launch lead is a median 462 µs for last arrivers vs. 3131 µs for the others. The GPU idle gap right before an a2a is about 3 µs on every rank: the a2a starts right after the preceding compute, and on rank 6 that compute was itself issued late.

#### E8.7 Caveats

- **Arrival = the kernel's start on the GPU.** Before LA, a kernel can already exchange data with peers that are present (a2a pairs, ring neighbours), so not all skew time is idle. However, post-arrival time exceeds the movement bound in every rank-kernel, so all data movement at peak fits after the last arrival. At most 8.4% of NCCL time can be movement at peak in any case (E3).
- **Movement bound.** It uses 450 GB/s and the bus-bytes convention. Real NVLink throughput is lower, so "excess" overstates protocol and sync overhead, and "moving@peak" understates movement.
- **Clock resolution.** The feasible offset is ±15 µs. This is negligible against the millisecond-scale a2a skew but comparable to the tiny all-reduces (7–13 µs), so the AR split is only indicative.
- **Scope.** One step, one node, one configuration. Whether a host-bound straggler is typical cannot be judged from one trace. Without the straggler, NCCL residency would shrink toward the post-arrival time, and so would the SM-time held while waiting (inference).
- **Held vs. lost.** "SMs held while waiting" are grid CTAs resident on SMs. In this run, comm overlapped compute for only 1.5–4.6% of comm time, so this is capacity held, not compute lost. E5 projects the loss under overlap.
- **Exclusions.** Broadcasts (2 instances, 15.1 ms of in-step kernel time) are excluded.

#### E8.8 Candidate claims

7. "In an 8-GPU Mixtral-8x7B training step (TP2 × EP4 over NVLink), we matched all 12,638 NCCL collective instances across ranks by process-group sequence number, with clocks consistent to ±15 µs, and split every kernel at the arrival of its last peer. 82.8% of NCCL kernel time, and 93.6% of the expert all-to-all time, passes before the last peer arrives. Once all peers are present, the 32 MiB all-to-all completes in a median 94 µs (minimum 80 µs), 1.7× its 56 µs NVLink movement bound."
8. "The waiting is skew, not data movement, but it still occupies SMs. NCCL kernels waiting for late peers hold 7.0% of the node's SM capacity over the step (5.7% in the expert all-to-all alone), 5.4× the 1.3% held after all peers have arrived. The late arrivals do not come from imbalanced expert work, which is within 2.3% across ranks. They come from one host-starved rank (last arriver in 72% of its group's all-to-alls; GPU busy 56% vs. 78–86% on the others) and its tensor-parallel partner, which inherits the delay."

Use claim 8 only with its framing: SM residency during skew is a cost of SM-resident communication, which an off-SM engine avoids, but fixing the straggler would also remove most of it. The trace does not support a claim that the post-arrival communication itself is expensive (1.28% of SM capacity).

## Reproduce

```sh
cd /scratch/harshanavkis/loom-proj/motivation-experiments/m3-sm-share/chakra
./extract.sh                                   # -> /scratch/harshanavkis/chakra-traces (≈1 min)
T=/scratch/harshanavkis/chakra-traces
python3 analyze_et.py Mixtral-8x7B  8 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et        > out_et_Mixtral-8x7B.txt
python3 analyze_et.py Mixtral-8x22B 8 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et > out_et_Mixtral-8x22B.txt
python3 analyze_et.py Llama3-70B    8 $T/Llama3/Llama3-70B/16TP/rank.*.et              > out_et_Llama3-70B.txt   # ~1 min, 16 procs
D=$T/Mixtral/Mixtral-8x7B/nemo_raw      # needs device_{0..7}.json; device_{1,4,5,7}.json are not produced by extract.sh (added later)
python3 analyze_kineto.py 8 $D/device_{0..7}.json > out_kineto_Mixtral-8x7B.txt                                  # ~17 s
# extension (E1-E5); args: NVLink GB/s/dir, [IB GB/s per NIC]
python3 analyze_moe.py 450 $T/Mixtral/Mixtral-8x7B $D/device_{0..7}.json > out_moe_Mixtral-8x7B.txt               # ~11 s, 8 procs
python3 analyze_nccl_et.py Mixtral-8x7B  8 450 12.5 $T/Mixtral/Mixtral-8x7B/chakra_trace.*.et         > out_nccl_et_Mixtral-8x7B.txt
python3 analyze_nccl_et.py Mixtral-8x22B 8 450 12.5 $T/Mixtral/Mixtral-8x22B/mixtral-8x22_chakra.*.et  > out_nccl_et_Mixtral-8x22B.txt
python3 analyze_nccl_et.py Llama3-70B    8 450 12.5 $T/Llama3/Llama3-70B/16TP/rank.*.et               > out_nccl_et_Llama3-70B.txt     # ~35 s, 16 procs
python3 project_offsm.py > out_projection.txt   # reads moe_summary_Mixtral-8x7B.json, nccl_et_Mixtral-8x22B.json
# E8: arrival skew vs. communication cost; args: NVLink GB/s/dir, nemo_raw dir, [ranks, default 0..7]
python3 analyze_skew.py 450 $D > out_skew_Mixtral-8x7B.txt                                                        # ~9 s, 8 procs
```

`{0..7}` is shell brace expansion (bash/zsh). Running `analyze_moe.py` on a 4-rank subset overwrites `moe_summary_Mixtral-8x7B.json`, so re-run it on all 8 ranks before `project_offsm.py`.

The scripts use only the Python standard library (tested with the system `python3`, 3.13). `analyze_et.py` also writes per-rank aggregates to `et_summary_<model>.json`. All JSON traces are stream-parsed by `jstream.py`; peak memory is about 100 MB for streaming one file, plus whatever events a script keeps.
