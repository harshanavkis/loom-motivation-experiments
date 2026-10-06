#!/usr/bin/env python3
"""Loom motivation figures, in the style of jigsaw-plotting-scripts/plot_hw_exp.py.

Figure 1 (intro, single column, loom-mix): Mixtral-8x22B bytes per MoE layer by fabric and kind
of traffic (Chakra ET). Figure 3 (Section 2, 1x3, loom-costs): (a) message rate, GPU-initiated vs CPU
proxy; (b) a 16 x 1 KiB dispatch to local and remote peers until the receiver sees the signal
(dispatch_bd.cu); (c) SM time per dispatch vs batch size. The CPU proxy appears in (a) only (owner,
2026-10-06). The GEMM panels (shared SMs: GEMM lost vs dispatch rate; reserved SMs: DeepGEMM next to
held SMs) were dropped from the figure the same day (owner): their numbers are printed for the text.
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


def top_label(ax, text, size):
    ax.text(0.5, 1.0, text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=size, color='navy', fontweight='bold', clip_on=False)


def save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        kw = {'dpi': 300} if ext == 'png' else {}
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




def panel_dispatch_time(ax, fs, title, H=1024, T=16):
    """Where the time of one dispatch goes, until the receiver sees the signal (stacked)."""
    d = dispatch_breakdown()
    d = d[(d.H == H) & (d.tokens == T)].set_index('path')
    segs = (('send', 'send loop', PASTEL[0], ''),
            ('drain', 'waiting for own writes', PASTEL[3], '//'),
            ('signal', 'signal issued', PASTEL[1], '..'),
            ('flight', 'signal in flight', PASTEL[7], '\\\\'))
    x = np.arange(len(BD_PATHS))
    bottom = np.zeros(len(x))
    for col, lab, color, hatch in segs:
        vals = np.array([d.loc[p, col] for p in BD_PATHS])
        ax.bar(x, vals, 0.6, bottom=bottom, color=color, hatch=hatch, edgecolor='black', linewidth=1, label=lab)
        bottom += vals
    for i, p in enumerate(BD_PATHS):
        ax.text(x[i], d.loc[p, 'seen_us'] + 0.8, f"{d.loc[p, 'seen_us']:.1f}", ha='center', va='bottom',
                fontsize=fs['annotation'])
    ax.set_xticks(x)
    ax.set_xticklabels(['local', 'remote:\nordered', 'remote:\nflush'], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(0, 80)            # headroom so the legend sits above the bars
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel('Latency [us]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=fs['legend'] - 2, columnspacing=0.8, handlelength=1.4)
    top_label(ax, title, fs['annotation'])








def panel_dispatch_sm(ax, fs, title, H=1024):
    """SM time per dispatch (CTA-us) vs tokens, 20 CTAs, idle; dashed: the part flush spends waiting."""
    d = dispatch_breakdown()
    d = d[d.H == H]
    toks = sorted(d.tokens.unique())
    x = {t: i for i, t in enumerate(toks)}
    short = {'local': 'local', 'ordered-destL24': 'remote: ordered', 'flush-warpL24': 'remote: flush'}
    for p, (_, color) in BD_PATHS.items():
        s_ = d[d.path == p].sort_values('tokens')
        ax.plot([x[t] for t in s_.tokens], s_.cta_total_us, color=color, marker='o', markersize=12, linewidth=3,
                markeredgecolor='k', alpha=0.9, label=short[p])
    s_ = d[d.path == 'flush-warpL24'].sort_values('tokens')
    ax.plot([x[t] for t in s_.tokens], s_.cta_drain_us, color=BD_PATHS['flush-warpL24'][1], marker='o', markersize=10,
            markerfacecolor='white', linewidth=2, linestyle='--', label='flush: waiting')
    ax.set_yscale('log')
    ax.set_ylim(10, 1e8)          # headroom so the legend sits above the curves
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels([str(t) for t in toks], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlabel('Tokens dispatched (1 KiB, top-8)', fontsize=fs['label'])
    ax.set_ylabel('SM time [SM-us]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=fs['legend'] - 2, columnspacing=0.8, handlelength=1.4)
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
    """Section 2 (one row): message rate; a dispatch until the receiver sees the signal; SM time per
    dispatch."""
    fs = MOTIVATION_COMBINED_FONT_SIZES
    fig, axes = plt.subplots(1, 3, figsize=E2E_COMBINED_FIGURE_SIZE)
    panel_msgrate(axes[0], fs, '(a) Message rate (higher is better ↑)')
    panel_dispatch_time(axes[1], fs, '(b) 16 × 1 KiB dispatch (lower is better ↓)')
    panel_dispatch_sm(axes[2], fs, '(c) SM time per dispatch (lower is better ↓)')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.18, wspace=0.28)
    save(fig, out_dir, 'loom-costs')
    plt.close(fig)


def main():
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
    print(dispatch_rate().sort_values(['H', 'tokens', 'path', 'rate_per_ms']).to_string(index=False))


if __name__ == '__main__':
    main()
