#!/usr/bin/env python3
"""Loom motivation figures, in the style of jigsaw-plotting-scripts/plot_hw_exp.py.

Figure 1 (intro, single column, loom-mix): Mixtral-8x22B bytes per MoE layer by fabric and kind
of traffic (Chakra ET). Figure 3 (Section 2, 1x3, loom-costs, since 2026-10-07): (a) dispatch and
(b) combine breakdown until the receiver sees the signal, 1 / 16 / 128 tokens, local vs remote
(ordered, flush), with the payload's time through the NIC (dispatch_bd.cu, bd_v2/summary_t3.csv and
summary_combine_t3.csv); (c) transport state per GPU vs EP size (m2-device-state/totals.out).
Dropped (owner): the message-rate panel (2026-10-07), the 16 x 1 KiB breakdown and the SM-time panel
(2026-10-06), the GEMM panels; their numbers are printed below for the text. panel_msgrate and
panel_routing are kept but not drawn.
Older layout notes follow:
Figure 1 (intro, 1x3): (a) Mixtral-8x22B bytes per MoE layer by fabric (Chakra ET),
(b) MoE dispatch latency, GPU-initiated vs CPU proxy, both packed (steve H200 + CX-7),
(c) message rate of 4 B puts vs CTAs issuing, GPU-initiated vs CPU proxy, with the NIC's rated
limit (the old (c), kernel -> 8 B delivered, panel_initiator, is no longer plotted); was: the
the unified-contract bound as a band (kernel store + fence, the NIC's own work, one load).
Figure 2 (Section 2, single column): compute left vs SMs held for communication.

Every number is read from the committed result files of this repository.
Usage: python3 plot_motivation.py <output dir>   (writes .pdf and .png)
"""
import io
import os
import re
import sys
import json
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAKRA = os.path.join(ROOT, 'm3-sm-share', 'chakra')
INTERF = os.path.join(ROOT, 'm3-sm-share', 'gpu-interference')
POSTED = os.path.join(ROOT, 'm3-sm-share', 'gpu-posted')
CX7 = os.path.join(ROOT, 'm5-rdma-init', 'steve-cx7')
DEEPGEMM = os.path.join(ROOT, 'm3-sm-share', 'deepgemm')
BD = os.path.join(CX7, 'bd_v2')
M2 = os.path.join(ROOT, 'm2-device-state')
ROUTING = os.path.join(ROOT, 'm6-routing')
# dispatch_bd variants plotted: summary name -> (label, color). Remote: per-lane DeepEP puts (V2.5's
# shape); flush = V2.5's normal dispatch (QPs follow the warps), ordered = V1 low latency (3 QPs per
# destination, a signal behind each QP's data)
BD_PATHS = {'local': ('local peer', 'tab:green'),
            'ordered-destL24': ('remote: ordered signal', 'lightskyblue'),
            'flush-warpL24': ('remote: flush', 'tab:blue')}

# ===== FONT AND STYLE SETTINGS (as in jigsaw plot_hw_exp.py) =====
SINGLE_COLUMN_FIGURE_SIZE = (10, 5)
SINGLE_COLUMN_FONT_SIZES = {'label': 18, 'tick': 16, 'legend': 15, 'annotation': 16}
E2E_COMBINED_FIGURE_SIZE = (48, 7)
MOTIVATION_COMBINED_FONT_SIZES = {'label': 40, 'tick': 36, 'legend': 34, 'annotation': 36}

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10

# variant -> (plot label, color); jigsaw convention: system green, baseline blue,
# CPU path red, the remaining engine orange
VARIANTS = {
    'loom': ('loom', 'tab:green'),
    'gpu-initiated': ('gpu-initiated', 'tab:blue'),
    'cpu-proxy': ('cpu-proxy', 'tab:red'),
    'copy-engine': ('copy-engine', 'tab:orange'),
    'sm-copy': ('sm-copy', 'gray'),
    'partitioned': ('partitioned (no co-run)', 'black'),
}
PASTEL = sns.color_palette("pastel")
LOOMGONE = '#F6D5D3'   # the paper's loomgone (main.tex): what Loom removes


def top_label(ax, text, size):
    ax.text(0.5, 1.0, text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=size, color='navy', fontweight='bold', clip_on=False)


def save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        kw = {'dpi': 300} if ext == 'png' else {'metadata': {'CreationDate': None}}   # reproducible PDF bytes
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), bbox_inches='tight', **kw)


# ---------------------------------------------------------------- data loading
def mixtral_8x22b_layer_bytes():
    """Median over the 32 ranks of bytes each GPU sends per MoE layer, by fabric.
    224 layer instances per step = 4 micro-batches x 56 layers (896 EP all-to-alls =
    224 x 4 matches exactly). Bytes sent: all-gather (n-1) x in, reduce-scatter
    (n-1)/n x in, all-to-all (n-1)/n x in, of which 1 peer is on the same node and 6
    are remote (8 ranks over 4 nodes; 8 GPUs per node assumed)."""
    d = json.load(open(os.path.join(CHAKRA, 'et_summary_Mixtral-8x22B.json')))
    layers = 4 * 56
    per = {'tp_ag': [], 'tp_rs': [], 'a2a_local': [], 'a2a_remote': []}
    for r in d['ranks']:
        ag = rs = a2a = 0
        for k, (n, _us, b) in r['comm'].items():
            ctype, pg = k.split('|')
            scope = d['pg_class'][pg]
            if ctype == 'ALL_GATHER' and scope == 'intra':
                ag += 3 * b            # TP4 ring all-gather sends (n-1) x input
            elif ctype == 'REDUCE_SCATTER' and scope == 'intra':
                rs += 0.75 * b         # (n-1)/n x input
            elif ctype == 'ALL_TO_ALL' and scope == 'inter':
                a2a += b
        per['tp_ag'].append(ag / layers)
        per['tp_rs'].append(rs / layers)
        per['a2a_local'].append(a2a / 8 / layers)       # 1 of 8 chunks to the same-node peer
        per['a2a_remote'].append(a2a * 6 / 8 / layers)  # 6 of 8 chunks to remote peers
    return {k: statistics.median(v) / 2**20 for k, v in per.items()}


def dispatch_sweep(ctas=20):
    """Median dispatch latency (us), idle, each path at its best measured variant per point:
    GPU-initiated = min(NVSHMEM per token, NVSHMEM packed (--block), DeepEP's post path per
    token (deepep_post, <= 128 tokens)); CPU proxy = min(packed, per token)."""
    rows = []
    for line in open(os.path.join(CX7, 'dispatch_sweep_all.csv')):
        f = line.strip().split(',')
        if f[0] == 'ibgda' and len(f) >= 8:
            rows.append(('gpu-initiated', 'nvshmem-token', int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[7])))
        elif f[0] == 'proxy' and len(f) >= 9:
            rows.append(('cpu-proxy', f[3], int(f[1]), int(f[2]), int(f[4]), int(f[5]), float(f[7])))
    for H in (1024, 7168):
        for line in open(os.path.join(CX7, f'dispatch_ibgda_block_H{H}_load0.csv')):
            f = line.strip().split(',')
            if f[0] == 'ibgda_block':
                rows.append(('gpu-initiated', 'nvshmem-block', int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[7])))
        for line in open(os.path.join(CX7, f'deepep_post_dispatch_H{H}.csv')):
            f = line.strip().split(',')
            if f[0] == 'dispatch' and f[1] == 'deepep':
                rows.append(('gpu-initiated', 'deepep-token', int(f[2]), 0, int(f[3]), int(f[4]), float(f[6])))
    df = pd.DataFrame(rows, columns=['variant', 'impl', 'H', 'load', 'tokens', 'ctas', 'us'])
    df = df[(df.load == 0) & (df.ctas == ctas)]
    return df.loc[df.groupby(['variant', 'H', 'tokens']).us.idxmin()].reset_index(drop=True)




def initiator_latency():
    """Kernel -> data delivered for an 8 B transfer (us), idle, steve."""
    out = {}
    for line in open(os.path.join(CX7, 'proxy_b2_numa0.csv')):
        f = line.strip().split(',')
        if f[0] == 'proxy' and f[1] == '8':
            out['cpu-proxy'] = float(f[2])
    # GPU-initiated: DeepEP's post path (put + completion, one warp), median of the runs
    runs = [line.strip().split(',') for line in open(os.path.join(CX7, 'deepep_post_lat.csv'))]
    runs = [f for f in runs if f[0] == 'lat' and f[1] == 'deepep' and f[2] == '8']
    out['gpu-initiated'] = statistics.median(float(f[4]) for f in runs)
    out['gpu-initiated-post'] = statistics.median(float(f[3]) for f in runs)
    for line in open(os.path.join(CX7, 'ce_triggered_idle.csv')):
        f = line.strip().split(',')
        if f[0] == 'triggered' and f[1] == 'd2d' and f[2] == '8':
            out['copy-engine'] = float(f[3])
    return out


def unified_bound():
    """Unified-contract bound for the same 8 B transfer (us), from measured parts only and
    timed like the bars (until the kernel knows the data landed): the kernel's store +
    system fence, the NIC's own post -> completion time (nic_post, median of the runs), and
    one kernel load of a completion word in pinned host memory. Low edge: the store carries
    the data (NIC floor: inline + BlueFlame, no PCIe read). High edge: the NIC reads the
    payload from GPU memory (BlueFlame, as proxy_b2). Drops the host handoff and the SM post
    chain, nothing else."""
    cost = {}
    for line in open(os.path.join(CX7, 'fence_cost_numa0.csv')):
        f = line.strip().split(',')
        if f[0] in ('hbm_store+fence_sys', 'sys_load'):
            cost[f[0]] = float(f[1]) / 1000
    nic = pd.read_csv(os.path.join(CX7, 'nic_post_numa0.csv'))
    nic = nic[(nic['size'] == 8) & (nic.bf == 'on')].groupby('src').cqe_med_us.median()
    kernel = cost['hbm_store+fence_sys'] + cost['sys_load']
    return kernel + nic['inline'], kernel + nic['gpu']


# ConnectX-7 rated RDMA message rate (NVIDIA ConnectX-7 datasheet: 330-370 M msgs/s), the
# hardware ceiling, drawn the way NVIDIA's IBGDA blog draws the ConnectX-6's 215 M/s
CX7_RATED_MPPS = 330


def message_rate():
    """Scalar 4 B nvshmem_p rate (M ops/s) vs CTAs issuing (1024 threads each, one QP per CTA):
    GPU-initiated (IBGDA) and CPU proxy (IBRC), NVSHMEM's shmem_p_bw on steve."""
    m = pd.read_csv(os.path.join(CX7, 'msgrate_steve.csv'))
    m = m[m.test == 'p_bw'].copy()
    m['mops'] = m.GBps * 1e3 / 4
    return m


def held_sm_curves(ks=(4, 8, 16, 20)):
    """GEMM (up-projection) throughput as % of its baseline vs SMs held."""
    def med_runs(path, cols):
        recs = []
        for line in open(path):
            f = line.strip().split(',')
            if f[0] == 'run' and f[1] != 'rep':
                recs.append(f[1:])
        return pd.DataFrame(recs, columns=cols)
    a = med_runs(os.path.join(INTERF, 'results_steve.csv'),
                 ['rep', 'workload', 'mode', 'dir', 'k', 'target', 'sm_target', 'ms', 'metric', 'unit', 'comm', 'distinct'])
    a = a[a.workload == 'gemm']
    a[['k', 'target', 'metric']] = a[['k', 'target', 'metric']].astype(float)
    base = a[a['mode'] == 'none'].metric.median()
    pct = lambda q: 100 * q.metric.median() / base
    curves = {
        'partitioned': [pct(a[(a['mode'] == 'target') & (a.k == k)]) for k in ks],
        'sm-copy': [pct(a[(a['mode'] == 'smcopy') & (a.dir == 'd2d') & (a.k == k) & (a.target == 50)]) for k in ks],
        'copy-engine': pct(a[(a['mode'] == 'ce') & (a.dir == 'd2d') & (a.target == 50)]),
    }
    b = med_runs(os.path.join(POSTED, 'results_steve.csv'),
                 ['rep', 'workload', 'mode', 'k', 'msg', 'target', 'sm_target', 'ms', 'tflops', 'comm', 'puts', 'put_ns', 'quiet_ns', 'distinct'])
    b = b[b.workload == 'gemm']
    b[['k', 'msg', 'target', 'tflops']] = b[['k', 'msg', 'target', 'tflops']].astype(float)
    bbase = b[b['mode'] == 'none'].tflops.median()
    curves['gpu-initiated'] = [100 * b[(b['mode'] == 'put') & (b.k == k) & (b.msg == 7168) & (b.target == 10)].tflops.median() / bbase for k in ks]
    b['comm'] = b.comm.astype(float)
    curves['gpu-initiated-GBps'] = [b[(b['mode'] == 'put') & (b.k == k) & (b.msg == 7168) & (b.target == 10)].comm.median() for k in ks]
    def none_metric(path):
        v = [float(f[9]) for f in (l.strip().split(',') for l in open(path))
             if f[0] == 'run' and f[1] != 'rep' and f[2] == 'gemm' and f[3] == 'none']
        return statistics.median(v)
    # CPU-posted RDMA matched to the GPU-initiated runs: 7 KiB writes, GPU memory -> GPU memory
    curves['cpu-proxy'] = 100 * none_metric(os.path.join(INTERF, 'rdma_gpu2gpu_7k.csv')) / none_metric(os.path.join(INTERF, 'rdma_none.csv'))
    rate = [l.split() for l in open(os.path.join(INTERF, 'rdma_cpu_7k.txt')) if l.split() and l.split()[0] == '7168'][0]
    curves['cpu-proxy-GBps'] = float(rate[4]) * 1e6 * 7168 / 1e9   # perftest MsgRate [Mpps] x 7 KiB
    return list(ks), curves


def dispatch_breakdown():
    """dispatch_bd.cu medians (bd_v2/summary.csv): 20 CTAs, idle, every size."""
    d = pd.read_csv(os.path.join(BD, 'summary.csv'))
    return d[(d.ctas == 20) & (d.load == 0) & d.path.isin(list(BD_PATHS))]


def reap_t3(kind):
    """The ordered sender's deferred cost: collecting every completion after it has posted the signal
    (dispatch_bd --reap-inline, bd_v2/summary_t3reap.csv / summary_combine_t3reap.csv): median us per token count."""
    f = 'summary_t3reap.csv' if kind == 'dispatch' else 'summary_combine_t3reap.csv'
    d = pd.read_csv(os.path.join(BD, f))
    return d[(d.ctas == 20) & (d.load == 0)].set_index('tokens').reap


def breakdown_t3(kind):
    """dispatch_bd.cu, 1 / 16 / 128 tokens in one session per kind, 20 CTAs, idle: kind 'dispatch' =
    7 KiB FP8 tokens (bd_v2/summary_t3.csv), 'combine' = 14 KiB BF16 expert outputs sent back
    (--combine, bd_v2/summary_combine_t3.csv; paths named c-*). Path names normalized to BD_PATHS."""
    f = 'summary_t3.csv' if kind == 'dispatch' else 'summary_combine_t3.csv'
    d = pd.read_csv(os.path.join(BD, f))
    d['path'] = d.path.str.replace(r'^c-', '', regex=True)
    return d[(d.ctas == 20) & (d.load == 0) & d.path.isin(list(BD_PATHS))]


def transport_state():
    """Per-GPU HBM (MiB) and QPs from m2-device-state/totals.out (source analysis, 1 NIC per GPU)."""
    series = {'DeepEP-V1 LL ': 'v1-ll', 'DeepEP-V1 normal ': 'v1-normal', 'DeepEP-V2.5 hybrid, CX-7': 'v2.5'}
    rows, cur = [], None
    for line in open(os.path.join(M2, 'totals.out')):
        if line.startswith('### '):
            cur = None
            for k, v in series.items():
                if line.startswith('### ' + k):
                    qps = re.search(r'QPs total=(\d+)', line) or re.search(r'QPs=(\d+)', line)
                    cur = [v, int(re.search(r'EP(\d+)', line).group(1)), int(qps.group(1))]
        elif cur and 'TOTAL GPU HBM' in line:
            rows.append(cur + [float(re.search(r'([\d.]+) MiB', line).group(1))])
            cur = None
    return pd.DataFrame(rows, columns=['series', 'ep', 'qps', 'mib'])


def staging_buffers():
    """Pinned send/staging buffers per GPU (GiB) that exist only because peers are reached by RDMA (the NIC
    reads the payload from registered memory): column (a) of m2-device-state/buffers.out, the strict reading
    (receive buffers, two-hop landing zones and V2.5's hierarchy inflation are left out). V1 LL and V2.5 at
    128 tokens per rank (decode), V1 normal is token-independent."""
    sec = {'DeepEP V1 low-latency': 'v1-ll', 'DeepEP V1 normal': 'v1-normal'}
    rows, cur = [], None
    for line in open(os.path.join(M2, 'buffers.out')):
        if line.startswith('DeepEP '):
            cur = next((v for k, v in sec.items() if line.startswith(k)), None)
            if line.startswith('DeepEP V2.5') and '128 tokens/rank' in line:
                cur = 'v2.5'
            continue
        m = re.match(r'^\s+(\d+)\s+([\d.]+) (MiB|GiB)', line)
        if cur and m:
            rows.append([cur, int(m.group(1)), float(m.group(2)) / (1024 if m.group(3) == 'MiB' else 1)])
    return pd.DataFrame(rows, columns=['series', 'ep', 'staging_gib'])


def loom_gpu_state():
    """Loom's GPU-side state per GPU (MiB), rough estimate from buffers.out: one pointer + page-table entries
    over every peer's receive window (see buffers.py loom_gpu_state)."""
    rows, on = [], False
    for line in open(os.path.join(M2, 'buffers.out')):
        if line.startswith('Loom, GPU side'):
            on = True
            continue
        m = re.match(r'^\s+(\d+)\s+[\d.]+ (?:MiB|GiB)\s+[\d,]+\s+[\d,]+ B\s+([\d.]+) (MiB|GiB)', line)
        if on and m:
            rows.append([int(m.group(1)), float(m.group(2)) * (1024 if m.group(3) == 'GiB' else 1)])
    return pd.DataFrame(rows, columns=['ep', 'mib'])


def dispatch_rate():
    """--d3 --d3-filler gemm-up interval sweep (bd_v2/d3rate_<variant>_H*.csv): per (path, H, tokens,
    requested period) the medians over 3 repetitions of the ACHIEVED rate (dispatches / measured
    window, per ms) and of the expert GEMM's throughput lost (%)."""
    rows = []
    names = {'local': 'local', 'ordered-destL3': 'ordered-destL24', 'flush-warpL': 'flush-warpL24'}
    for f, p in names.items():
        for H in (1024, 7168):
            for line in open(os.path.join(BD, f'd3rate_{f}_H{H}.csv')):
                x = line.strip().split(',')
                if x[0] == 'd3' and x[1] != 'path':
                    rows.append((p, H, int(x[5]), int(x[7]), float(x[9]) / float(x[10]) * 1e3, 100 * float(x[14])))
    d = pd.DataFrame(rows, columns=['path', 'H', 'tokens', 'period_us', 'rate_per_ms', 'lost_pct'])
    return d.groupby(['path', 'H', 'tokens', 'period_us'], as_index=False).median()


def deepgemm_held(ks=(4, 8, 16, 20)):
    """DeepGEMM FP8 expert GEMM (8 experts x M tokens, 7168 -> 4096) next to k SMs held for
    communication and planned for 132 - k (dg_held.py): medians of 5, % of the same GEMM alone on
    all SMs; plus the copy engine moving 50 GB/s beside it on all SMs."""
    out = {}
    for M in (1024, 4096):
        rows = [l.strip().split(',') for l in open(os.path.join(DEEPGEMM, f'held_m{M}.csv'))
                if l.count(',') == 7 and l[0].isdigit()]
        df = pd.DataFrame(rows, columns=['rep', 'dtype', 'k', 'mode', 'num_sms', 'us', 'tflops', 'pct'])
        df = df[df.dtype == 'fp8']
        df[['k', 'pct']] = df[['k', 'pct']].astype(float)
        out[M] = {m: [df[(df['mode'] == m) & (df.k == k)].pct.median() for k in ks] for m in ('held', 'partitioned')}
        out[M]['ce50'] = df[df['mode'] == 'ce50'].pct.median()
    return list(ks), out


# ---------------------------------------------------------------- panels
def panel_layer_bytes(ax, fs, title):
    b = mixtral_8x22b_layer_bytes()
    segs = [('TP all-gather (host-queued)', [b['tp_ag'], 0], PASTEL[0], ''),
            ('TP reduce-scatter (host-queued)', [b['tp_rs'], 0], PASTEL[1], '///'),
            ('EP all-to-all (kernel-initiated)', [b['a2a_local'], b['a2a_remote']], PASTEL[3], '\\\\')]
    y = np.arange(2)
    left = np.zeros(2)
    for name, vals, color, hatch in segs:
        ax.barh(y, vals, 0.55, left=left, color=color, hatch=hatch, edgecolor='black', linewidth=1, label=name)
        left += np.array(vals)
    for i, tot in enumerate(left):
        ax.text(tot + 8, y[i], f'{tot:.0f} MiB', va='center', ha='left', fontsize=fs['annotation'])
    ax.set_yticks(y)
    ax.set_yticklabels(['NVLink\n(scale-up)', 'RDMA\n(scale-out)'], fontsize=fs['tick'])
    ax.invert_yaxis()
    ax.set_xlabel('Bytes sent per MoE layer per GPU [MiB]', fontsize=fs['label'])
    ax.tick_params(axis='x', labelsize=fs['tick'])
    ax.set_xlim(0, max(left) * 1.95)   # room right of the bars for the legend
    ax.grid(True, alpha=0.3, axis='x', color='gray', linestyle='-')
    ax.legend(loc='lower right', ncol=1, frameon=True, fontsize=fs['legend'])
    if title:
        top_label(ax, title, fs['annotation'])




def panel_msgrate(ax, fs, title):
    m = message_rate()
    ctas = sorted(m.ctas.unique())
    x = {c: i for i, c in enumerate(ctas)}
    for tr, v, mk in (('ibgda', 'gpu-initiated', 'o'), ('ibrc', 'cpu-proxy', 's')):
        s = m[m.transport == tr].sort_values('ctas')
        ax.plot([x[c] for c in s.ctas], s.mops, color=VARIANTS[v][1], marker=mk, markersize=12, linewidth=2,
                markeredgecolor='k', alpha=0.9, label=VARIANTS[v][0])
    ax.axhline(CX7_RATED_MPPS, color=VARIANTS['loom'][1], linewidth=3, linestyle='--',
               label='NIC rated limit')
    ax.set_yscale('log')
    ax.set_ylim(1, 1e5)           # headroom so the legend sits above the curves
    ax.set_xticks(range(len(ctas)))
    ax.set_xticklabels([str(c) for c in ctas], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlabel('CTAs issuing 4 B puts (one SM each)', fontsize=fs['label'])
    ax.set_ylabel('Rate [M ops/s]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=fs['legend'] - 4, columnspacing=0.8, handlelength=1.5)
    top_label(ax, title, fs['annotation'])




BD_SEGS = (('send', 'send loop', PASTEL[0], ''),
           ('drain', 'waiting for own writes', PASTEL[3], '//'),
           ('signal', 'signal issued', PASTEL[1], '..'),
           ('flight', 'signal in flight', PASTEL[7], '\\\\'))


def panel_breakdown(axes, fs, title, kind, H, tokens=(1, 16, 128), legend=True):
    """Initiation: how long the SENDER of one dispatch (or combine) is busy until it is done, i.e. has issued
    the arrival signal (send loop + waiting for its own writes + issuing the signal), stacked, one sub-axis per
    batch size (owner 2026-10-07: show initiation, not delivery; the receiver-side 'signal in flight' is left
    out and printed for the text). Dashed: the payload's time through the NIC, what a flushing sender waits
    out. Narrow bars: what a remote sender spends beyond the local sender (labels: remote / local). The
    ordered path reaps its completions after the dispatch (not in its bar)."""
    d = breakdown_t3(kind)
    d = d.assign(sender_us=d.send + d.drain + d.signal)
    x = np.arange(len(BD_PATHS))
    for ax, T in zip(axes, tokens):
        t = d[(d.H == H) & (d.tokens == T)].set_index('path')
        bottom = np.zeros(len(x))
        for col, lab, color, hatch in BD_SEGS[:3]:
            vals = np.array([t.loc[p, col] for p in BD_PATHS])
            ax.bar(x, vals, 0.65, bottom=bottom, color=color, hatch=hatch, edgecolor='black', linewidth=1, label=lab)
            bottom += vals
        # the ordered sender collects its completions later (before it reuses its send buffer and queue slots)
        reap = float(reap_t3(kind).loc[T])
        io = list(BD_PATHS).index('ordered-destL24')
        ax.bar(x[io], reap, 0.65, bottom=t.loc['ordered-destL24', 'sender_us'], color='white', hatch='++',
               edgecolor='gray', linewidth=1, label='completions\ncollected later')
        t = t.assign(total_us=t.sender_us)
        t.loc['ordered-destL24', 'total_us'] += reap
        top = max(t.total_us)
        for i, p in enumerate(BD_PATHS):
            ax.text(x[i], t.loc[p, 'total_us'] + 0.04 * top, f"{t.loc[p, 'total_us']:.0f}", ha='center',
                    va='bottom', fontsize=fs['annotation'] - 6)
        link_us = nic_payload_us(kind, H, T)   # what a flushing sender waits out: drawn over the flush bar only
        ifl = list(BD_PATHS).index('flush-warpL24')
        if link_us < top * 1.3:
            ax.plot([x[ifl] - 0.4, x[ifl] + 0.4], [link_us, link_us], color='black', linewidth=4, linestyle='--',
                    label='payload through NIC')
        floor = t.loc['local', 'sender_us']
        for i, p in enumerate(BD_PATHS):
            if p == 'local':
                continue
            extra = t.loc[p, 'total_us'] - floor
            if extra > 0:
                ax.bar(x[i] + 0.43, extra, 0.16, bottom=floor, color=LOOMGONE, edgecolor='#a04040', linewidth=1.5,
                       label='removed by a\nload/store contract' if i == 1 else None)
            ax.text(x[i] + 0.43, max(t.loc[p, 'total_us'], floor) + 0.16 * top, f"{t.loc[p, 'total_us'] / floor:.1f}x",
                    ha='center', va='bottom', fontsize=fs['annotation'] - 12, color='#a04040')
        ax.set_xticks(x)
        ax.set_xlim(-0.55, 2.75)
        ax.set_xticklabels(['local', 'ord.', 'flush'], fontsize=fs['tick'] - 8)
        ax.tick_params(axis='y', labelsize=fs['tick'] - 6)
        ax.set_ylim(0, top * (5.6 if legend and ax is axes[0] else 1.35))   # headroom for the legend
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlabel(f'{T} token{"s" if T > 1 else ""}', fontsize=fs['label'] - 4)
    axes[0].set_ylabel('Sender busy [us]', fontsize=fs['label'])
    if legend:
        h, l = axes[0].get_legend_handles_labels()
        for ax in axes[1:]:   # the payload line may be drawn only in the larger panels
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        axes[0].legend(h, l, loc='upper left', ncol=1, frameon=True, fontsize=fs['legend'] - 10, handlelength=1.4,
                       labelspacing=0.25, borderpad=0.3)
    mid = axes[len(axes) // 2]
    mid.text(0.5, 1.0, title, transform=mid.transAxes, ha='center', va='bottom',
             fontsize=fs['annotation'], color='navy', fontweight='bold', clip_on=False)


def nic_to_hbm_gbps():
    """The NIC -> GPU memory rate the testbed reaches: the fastest 4096 x 7 KiB dispatch of the GPU vs proxy
    sweeps (dispatch_sweep_all.csv; every path is NIC-bound there), payload bytes / time. 14.8 GB/s on steve
    (21,711 messages x 7168 B in 10.53 ms); recomputed after every rerun, e.g. after moving the NIC."""
    d = dispatch_sweep()
    us = d[(d.H == 7168) & (d.tokens == 4096)].us.min()
    for f in sorted(os.listdir(BD)):   # messages in a 4096-token dispatch, from any dispatch_bd run at that size
        if f.startswith('dispatch_bd_ordered-destL3_H7168_load0') and f.endswith('.csv'):
            rows = [l for l in open(os.path.join(BD, f)) if l.strip() and not l.startswith(('#', 'WARN'))]
            r = pd.read_csv(io.StringIO(''.join(rows)))
            return float(r[r.tokens == 4096].msgs.iloc[0]) * 7168 / (us * 1e3)


def nic_payload_us(kind, H, T):
    """Payload bytes of one dispatch / combine (messages from the run) through the NIC at nic_to_hbm_gbps(), in us."""
    return msgs_per_dispatch(kind, T) * H / (NIC_GBPS * 1e3)


def msgs_per_dispatch(kind, T):
    prefix = 'dispatch_bd_t3_flush' if kind == 'dispatch' else 'combine_bd_t3_flush'
    for f in sorted(os.listdir(BD)):
        if f.startswith(prefix) and f.endswith('.csv'):
            rows = [l for l in open(os.path.join(BD, f)) if l.strip() and not l.startswith(('#', 'WARN'))]
            d = pd.read_csv(io.StringIO(''.join(rows)))
            return int(d[d.tokens == T].msgs.iloc[0])


def panel_transport_state(ax, fs, title):
    """GPU memory each GPU holds for GPU-initiated RDMA vs EP size, stacked: queue-pair state (rings, doorbell
    records, descriptors; m2-device-state/totals.out) and the pinned staging buffers the NIC reads payloads
    from (buffers.out, decode: 128 tokens per rank). Labels: QPs, each with a NIC doorbell page mapped into the
    GPU. 256 experts up to EP256, 512 / 1024 at EP512 / EP1024; DeepEP V1 normal runs on at most EP160."""
    d = transport_state()
    b = staging_buffers()
    eps = [16, 128, 256, 512, 1024]
    series = (('v1-ll', 'DeepEP V1 low-latency', 'lightskyblue'),
              ('v1-normal', 'DeepEP V1 normal (max EP160)', 'white'),
              ('v2.5', 'DeepEP V2.5', 'tab:blue'))
    w = 0.21
    for j, (k, lab, color) in enumerate(series):
        s_ = d[d.series == k].set_index('ep')
        st = b[b.series == k].set_index('ep')
        have = [e for e in eps if e in s_.index and e in st.index]
        xs = np.array([eps.index(e) for e in have]) + (j - 1.5) * w
        q = np.array([s_.loc[e, 'mib'] / 1024 for e in have])
        g = np.array([st.loc[e, 'staging_gib'] for e in have])
        ax.bar(xs, q, w, color=color, edgecolor='black', linewidth=1, label=lab)
        ax.bar(xs, g, w, bottom=q, color=color, hatch='xx', edgecolor='black', linewidth=1, alpha=0.55)
        for xi, e, top in zip(xs, have, q + g):
            ax.text(xi, top + 0.1, f"{s_.loc[e, 'qps']:,}", ha='center', va='bottom', rotation=90,
                    fontsize=fs['annotation'] - 12)
    lm = loom_gpu_state().set_index('ep')   # Loom: pointers + page-table entries, no queues, no staging
    xs = np.arange(len(eps)) + 1.5 * w
    vals = np.array([lm.loc[e, 'mib'] / 1024 for e in eps])
    ax.bar(xs, vals, w, color=VARIANTS['loom'][1], edgecolor='black', linewidth=1, label='Loom (pointers + page tables)')
    for xi, e in zip(xs, eps):
        v = lm.loc[e, 'mib']
        ax.text(xi, 0.1, f"{v:.1f} MiB" if v >= 1 else f"{v * 1024:.0f} KiB", ha='center', va='bottom', rotation=90,
                fontsize=fs['annotation'] - 12, color=VARIANTS['loom'][1])
    ax.bar([0], [0], 0, color='white', hatch='xx', edgecolor='black', label='+ pinned staging buffers')
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([f'{e}' for e in eps], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(0, 11)            # headroom so the legend sits above the bars
    ax.set_yticks([0, 2, 4])
    ax.set_xlabel('GPUs in the expert-parallel group', fontsize=fs['label'])
    ax.set_ylabel('Memory per GPU [GiB]', fontsize=fs['label'] - 6)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', ncol=1, frameon=True, fontsize=fs['legend'] - 8, handlelength=1.4,
              labelspacing=0.3, borderpad=0.3)
    top_label(ax, title, fs['annotation'])


def panel_routing(ax, fs, title):
    """Destination GPUs per token of a DeepSeek-V3 dispatch, inside vs outside the sender's scale-up
    domain, vs EP size, for 8-GPU (HGX) and 64-GPU (NVL72) domains (m6-routing/routing_traffic.csv)."""
    d = pd.read_csv(os.path.join(ROUTING, 'routing_traffic.csv'))
    eps = sorted(d.ep.unique())
    w = 0.38
    for j, (dom, lab, hatch) in enumerate(((8, 'HGX, 8 GPUs', ''), (64, 'NVL72, 64 GPUs', '//'))):
        s_ = d[d.domain == dom].set_index('ep')
        xs = np.arange(len(eps)) + (j - 0.5) * w
        loc = np.array([s_.loc[e, 'dst_gpus'] - s_.loc[e, 'remote_gpus'] for e in eps])
        rem = np.array([s_.loc[e, 'remote_gpus'] for e in eps])
        ax.bar(xs, loc, w, color=PASTEL[2], hatch=hatch, edgecolor='black', linewidth=1)
        ax.bar(xs, rem, w, bottom=loc, color='tab:blue', hatch=hatch, edgecolor='black', linewidth=1)
        for xi, e in zip(xs, eps):
            ax.text(xi, s_.loc[e, 'dst_gpus'] + 0.3, f"{100 * s_.loc[e, 'remote_share']:.0f}%", ha='center',
                    va='bottom', rotation=90, fontsize=fs['annotation'] - 10)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PASTEL[2], edgecolor='black', label='inside the domain'),
               Patch(facecolor='tab:blue', edgecolor='black', label='outside (remote)'),
               Patch(facecolor='white', edgecolor='black', label='domain: HGX, 8 GPUs'),
               Patch(facecolor='white', edgecolor='black', hatch='//', label='domain: NVL72, 64 GPUs')]
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([f'EP{e}' for e in eps], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(0, 15.5)          # headroom so the legend sits above the bars
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlabel('GPUs in the expert-parallel group', fontsize=fs['label'])
    ax.set_ylabel('GPUs per token', fontsize=fs['label'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(handles=handles, loc='upper left', ncol=2, frameon=True, fontsize=fs['legend'] - 8, handlelength=1.4,
              labelspacing=0.3, borderpad=0.3, columnspacing=0.8)
    top_label(ax, title, fs['annotation'])


def figure_intro(out_dir):
    """Introduction (single column): Mixtral-8x22B bytes per MoE layer by fabric and kind."""
    fs = {'label': 20, 'tick': 18, 'legend': 16, 'annotation': 18}   # ~8 pt at column width
    fig, ax = plt.subplots(figsize=(8, 3.1))
    panel_layer_bytes(ax, fs, None)
    plt.tight_layout()
    save(fig, out_dir, 'loom-mix')
    plt.close(fig)


def figure_costs(out_dir):
    """Section 2 (one row): dispatch and combine breakdowns (1 / 16 / 128 tokens) and transport state
    per GPU. Dropped 2026-10-07 (owner): the message-rate panel (GPU-initiated vs CPU proxy; its
    numbers stay in the text and are printed below)."""
    fs = MOTIVATION_COMBINED_FONT_SIZES
    fig = plt.figure(figsize=E2E_COMBINED_FIGURE_SIZE)
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.99, top=0.88, bottom=0.2, wspace=0.16, width_ratios=[1.3, 1.3, 1])
    axd = [fig.add_subplot(c) for c in gs[0, 0].subgridspec(1, 3, wspace=0.42)]
    axc = [fig.add_subplot(c) for c in gs[0, 1].subgridspec(1, 3, wspace=0.42)]
    panel_breakdown(axd, fs, '(a) Dispatch initiation, 7 KiB tokens (lower ↓)', 'dispatch', 7168)
    panel_breakdown(axc, fs, '(b) Combine initiation, 14 KiB tokens (lower ↓)', 'combine', 14336, legend=False)
    panel_transport_state(fig.add_subplot(gs[0, 2]), fs, '(c) Transport state per GPU (lower ↓)')
    save(fig, out_dir, 'loom-costs')
    plt.close(fig)


NIC_GBPS = None   # set in main() from nic_to_hbm_gbps()


def main():
    global NIC_GBPS
    NIC_GBPS = nic_to_hbm_gbps()
    print(f'NIC -> HBM rate (payload line): {NIC_GBPS:.2f} GB/s')
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'plots', 'out')
    os.makedirs(out_dir, exist_ok=True)
    figure_intro(out_dir)
    figure_costs(out_dir)
    # print the plotted numbers so the paper text can be checked against them
    print('layer bytes MiB:', {k: round(v, 1) for k, v in mixtral_8x22b_layer_bytes().items()})
    print('initiator us:', initiator_latency())
    print(dispatch_sweep().sort_values(['variant', 'H', 'tokens']).to_string(index=False))
    print('unified bound us (store carries data, NIC reads payload):', unified_bound())
    print('DeepGEMM held', deepgemm_held())
    ks, c = held_sm_curves()
    print('held SMs', ks, {k: (np.round(v, 1).tolist() if isinstance(v, list) else round(v, 1)) for k, v in c.items()})
    b = dispatch_breakdown()
    print(b[['path', 'H', 'tokens', 'send', 'drain', 'signal', 'flight', 'seen_us', 'cta_total_us', 'cta_drain_us']]
          .sort_values(['path', 'H', 'tokens']).to_string(index=False))
    for kind, H in (('dispatch', 7168), ('combine', 14336)):
        print(f'{kind} 1 / 16 / 128 tokens, one session (Fig 3a/b):')
        bk = breakdown_t3(kind); bk = bk.assign(sender_us=bk.send + bk.drain + bk.signal)
        print(bk[['path', 'tokens', 'send', 'drain', 'signal', 'sender_us', 'flight', 'seen_us', 'cta_total_us',
                  'cta_drain_us']].sort_values(['tokens', 'path']).to_string(index=False))
        print('  payload through the NIC us:', {T: round(nic_payload_us(kind, H, T), 1) for T in (1, 16, 128)})
    m = message_rate()
    print('message rate (text only):', m.groupby('transport').mops.max().round(1).to_dict())
    print('transport state (Fig 3c):')
    print(transport_state().to_string(index=False))
    print('Loom GPU-side state (Fig 3c, green), MiB:')
    print(loom_gpu_state().to_string(index=False))
    print('pinned staging buffers (Fig 3c, hatched):')
    print(staging_buffers().to_string(index=False))
    print(dispatch_rate().sort_values(['H', 'tokens', 'path', 'rate_per_ms']).to_string(index=False))


if __name__ == '__main__':
    main()
