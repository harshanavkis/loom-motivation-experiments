#!/usr/bin/env python3
"""Loom motivation figures, in the style of jigsaw-plotting-scripts/plot_hw_exp.py.

Figure 1 (intro, 1x3): (a) Mixtral-8x22B bytes per MoE layer by fabric (Chakra ET),
(b) MoE dispatch latency, GPU-initiated vs CPU proxy (steve H200 + CX-7),
(c) kernel -> data delivered for one 8 B remote transfer, per initiation path, plus the
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
    """Median dispatch latency (us), idle, from dispatch_sweep_all.csv."""
    rows = []
    for line in open(os.path.join(CX7, 'dispatch_sweep_all.csv')):
        f = line.strip().split(',')
        if f[0] == 'ibgda' and len(f) >= 8:
            H, load, T, c, med = int(f[1]), int(f[2]), int(f[3]), int(f[4]), float(f[7])
            rows.append(('gpu-initiated', H, load, T, c, med))
        elif f[0] == 'proxy' and len(f) >= 9 and f[3] == 'block':
            H, load, T, c, med = int(f[1]), int(f[2]), int(f[4]), int(f[5]), float(f[7])
            rows.append(('cpu-proxy', H, load, T, c, med))
    df = pd.DataFrame(rows, columns=['variant', 'H', 'load', 'tokens', 'ctas', 'us'])
    return df[(df.load == 0) & (df.ctas == ctas)]


def dispatch_bound(ctas=20):
    """Unified-contract bound for the same dispatches (us), idle: the GPU-triggered copy
    engine (dispatch_ce token mode: the kernel only routes, the engine copies each token, no
    SM copies), but no faster than the NIC can deliver the bytes (the best NIC -> HBM rate in
    the RDMA sweeps), plus the NIC's own floor (nic_post inline + BlueFlame). Returns the
    measured copy-engine rows and the bound."""
    rows = []
    for H in (1024, 7168):
        for line in open(os.path.join(CX7, f'dispatch_ce_H{H}_load0.csv')):
            f = line.strip().split(',')
            if f[0] == 'ce' and f[3] == 'token' and int(f[5]) == ctas:
                rows.append((H, int(f[4]), int(f[6]), float(f[7])))
    ce = pd.DataFrame(rows, columns=['H', 'tokens', 'messages', 'us'])
    rate = max(float(line.split(',')[10]) for line in open(os.path.join(CX7, 'dispatch_sweep_all.csv'))
               if line.split(',')[0] in ('ibgda', 'proxy') and len(line.split(',')) >= 11) * 1e3   # bytes/us
    nic = pd.read_csv(os.path.join(CX7, 'nic_post_numa0.csv'))
    floor = nic[(nic['size'] == 8) & (nic.bf == 'on') & (nic.src == 'inline')].cqe_med_us.median()
    ce['bound'] = np.maximum(ce.us, ce.messages * ce.H / rate) + floor
    return ce, rate, floor


def initiator_latency():
    """Kernel -> data delivered for an 8 B transfer (us), idle, steve."""
    out = {}
    for line in open(os.path.join(CX7, 'proxy_b2_numa0.csv')):
        f = line.strip().split(',')
        if f[0] == 'proxy' and f[1] == '8':
            out['cpu-proxy'] = float(f[2])
    txt = open(os.path.join(CX7, 'put_lat_steve.txt')).read()
    m = re.search(r'## shmem_put_latency.*?\n.*?^8\s+Thread\s+([\d.]+)', txt, re.S | re.M)
    out['gpu-initiated'] = float(m.group(1))
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
    # CPU-posted RDMA (perftest, 20.5 GB/s GPU->host) next to the same GEMM
    def none_metric(path):
        v = [float(f[9]) for f in (l.strip().split(',') for l in open(path))
             if f[0] == 'run' and f[1] != 'rep' and f[2] == 'gemm' and f[3] == 'none']
        return statistics.median(v)
    curves['cpu-proxy'] = 100 * none_metric(os.path.join(INTERF, 'rdma_gpu2host.csv')) / none_metric(os.path.join(INTERF, 'rdma_none.csv'))
    return list(ks), curves


# ---------------------------------------------------------------- panels
def panel_layer_bytes(ax, fs):
    b = mixtral_8x22b_layer_bytes()
    segs = [('TP all-gather', [b['tp_ag'], 0], PASTEL[0], ''),
            ('TP reduce-scatter', [b['tp_rs'], 0], PASTEL[1], '///'),
            ('EP all-to-all', [b['a2a_local'], b['a2a_remote']], PASTEL[3], '\\\\')]
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
    ax.set_xlim(0, max(left) * 1.3)
    ax.grid(True, alpha=0.3, axis='x', color='gray', linestyle='-')
    ax.legend(loc='lower right', ncol=1, frameon=True, fontsize=fs['legend'])
    top_label(ax, '(a) Mixtral-8x22B: both fabrics in every layer', fs['annotation'])


def panel_dispatch(ax, fs):
    df = dispatch_sweep()
    ce, _, _ = dispatch_bound()
    toks = sorted(df.tokens.unique())
    x = {t: i for i, t in enumerate(toks)}
    markers = {7168: 'o', 1024: 's'}
    # operating windows, as shaded x ranges
    for lo, hi, name in ((-0.3, x[128] + 0.3, 'decode'), (x[1024] - 0.3, len(toks) - 0.7, 'prefill')):
        ax.axvspan(lo, hi, color='gray', alpha=0.12, zorder=0)
        ax.text((lo + hi) / 2, 3.5, name, ha='center', va='bottom', fontsize=fs['legend'], color='dimgray')
    for v in ('gpu-initiated', 'cpu-proxy'):
        label, color = VARIANTS[v]
        for H, mk in markers.items():
            s = df[(df.variant == v) & (df.H == H)].sort_values('tokens')
            ax.plot([x[t] for t in s.tokens], s.us, color=color, marker=mk, markersize=12,
                    linewidth=2, markeredgecolor='k', alpha=0.9)
    for H, mk in markers.items():
        s = ce[ce.H == H].sort_values('tokens')
        ax.plot([x[t] for t in s.tokens], s.us, color=VARIANTS['loom'][1], linewidth=1.5,
                linestyle=':', marker=mk, markersize=8, markerfacecolor='none', alpha=0.8)
        ax.plot([x[t] for t in s.tokens], s.bound, color=VARIANTS['loom'][1], marker=mk, markersize=12,
                linewidth=4, markeredgecolor='k', alpha=0.9)
    ax.set_yscale('log')
    ax.set_ylim(3, 1e6)           # room below for the window labels, above for the legend
    ax.set_xlim(-0.5, len(toks) - 0.5)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels([str(t) for t in toks], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.set_xlabel('Tokens dispatched (top-8, 8 destinations)', fontsize=fs['label'])
    ax.set_ylabel('Dispatch latency [us]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3)
    handles = [Line2D([0], [0], color=VARIANTS[v][1], linewidth=2) for v in ('gpu-initiated', 'cpu-proxy')]
    labels = [VARIANTS[v][0] for v in ('gpu-initiated', 'cpu-proxy')]
    handles += [Line2D([0], [0], color=VARIANTS['loom'][1], linewidth=4),
                Line2D([0], [0], color=VARIANTS['loom'][1], linewidth=1.5, linestyle=':')]
    labels += ['unified (bound)', 'copy engine']
    handles += [Line2D([0], [0], color='gray', linestyle='', marker=m, markersize=12, markeredgecolor='k') for m in markers.values()]
    labels += ['7 KiB tokens', '1 KiB tokens']
    ax.legend(handles, labels, loc='upper left', ncol=3, frameon=True, fontsize=fs['legend'],
              columnspacing=0.8, handlelength=1.5)
    top_label(ax, '(b) MoE dispatch (lower is better ↓)', fs['annotation'])


def panel_initiator(ax, fs):
    lat = initiator_latency()
    order = ['cpu-proxy', 'gpu-initiated', 'copy-engine']
    notes = {'cpu-proxy': '0.075 us post\n+ 1 core',
             'gpu-initiated': '6-8 us SM\nper post',
             'copy-engine': 'local peers\nonly'}
    colors = {'cpu-proxy': PASTEL[3], 'gpu-initiated': PASTEL[0], 'copy-engine': PASTEL[1]}
    hatches = {'cpu-proxy': '', 'gpu-initiated': '///', 'copy-engine': '\\\\'}
    xs = np.arange(len(order))
    for i, v in enumerate(order):
        ax.bar(xs[i], lat[v], 0.6, color=colors[v], hatch=hatches[v], edgecolor='black', linewidth=1)
        ax.text(xs[i], lat[v] + 0.3, f'{lat[v]:.1f}', ha='center', va='bottom', fontsize=fs['annotation'])
    # unified-contract bound as a band behind the bars: composed from measured parts
    lo, hi = unified_bound()
    ax.axhspan(lo, hi, facecolor=PASTEL[2], alpha=0.5, zorder=0, linestyle='--', edgecolor='black',
               linewidth=1, label='unified (bound)')
    ax.set_xlim(-0.5, len(order) - 0.1)   # room right of the bars for the band's edge values
    ax.text(len(order) - 0.15, hi, f'{hi:.1f}', ha='right', va='bottom', fontsize=fs['annotation'])
    ax.text(len(order) - 0.15, lo, f'{lo:.1f}', ha='right', va='top', fontsize=fs['annotation'])
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{VARIANTS[v][0]}\n{notes[v]}' for v in order], fontsize=fs['legend'])
    ax.legend(loc='upper right', frameon=True, fontsize=fs['legend'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(0, max(lat.values()) * 1.5)
    ax.set_ylabel('Latency [us]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3, axis='y', color='gray', linestyle='-')
    top_label(ax, '(c) Kernel → 8 B delivered (lower is better ↓)', fs['annotation'])


def figure1(out_dir):
    fs = MOTIVATION_COMBINED_FONT_SIZES
    fig, axes = plt.subplots(1, 3, figsize=E2E_COMBINED_FIGURE_SIZE)
    panel_layer_bytes(axes[0], fs)
    panel_dispatch(axes[1], fs)
    panel_initiator(axes[2], fs)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.18, wspace=0.25)
    save(fig, out_dir, 'loom-motivation')
    plt.close(fig)


def figure2(out_dir):
    fs = SINGLE_COLUMN_FONT_SIZES
    ks, c = held_sm_curves()
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN_FIGURE_SIZE)
    x = range(len(ks))
    for v, mk, ls, ms, lw in (('gpu-initiated', 'o', '-', 16, 4), ('sm-copy', 's', '--', 8, 2),
                              ('partitioned', '^', ':', 12, 2)):
        label, color = VARIANTS[v]
        ax.plot(x, c[v], color=color, marker=mk, markersize=ms, linewidth=lw, linestyle=ls,
                markeredgecolor='k', alpha=0.9, label=label)
    label, color = VARIANTS['cpu-proxy']
    ax.axhline(c['cpu-proxy'], color=color, linewidth=2, linestyle=':', label=f'{label} (holds a CPU core)')
    # unified-contract bound: the kernel starts the transfer like a local copy-engine copy
    ax.axhline(c['copy-engine'], color=VARIANTS['loom'][1], linewidth=3, linestyle='-.',
               label='unified contract (bound): copy engine')
    ax.fill_between(x, c['gpu-initiated'], c['copy-engine'], color=PASTEL[2], alpha=0.5,
                    label='reclaimed vs gpu-initiated')
    gain = c['copy-engine'] - c['gpu-initiated'][-1]
    ax.annotate(f'+{gain:.0f} pts', xy=(x[-1], c['gpu-initiated'][-1] + gain / 2), xytext=(-12, 0),
                textcoords='offset points', ha='right', va='center', fontsize=fs['annotation'])
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in ks], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(40, 105)
    ax.set_xlabel('SMs held for communication (of 132)', fontsize=fs['label'])
    ax.set_ylabel('Expert GEMM throughput [%]', fontsize=fs['label'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', ncol=1, frameon=True, fontsize=fs['legend'])
    top_label(ax, '(Higher is better ↑)', fs['annotation'])
    plt.tight_layout()
    save(fig, out_dir, 'loom-held-sms')
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'plots', 'out')
    os.makedirs(out_dir, exist_ok=True)
    figure1(out_dir)
    figure2(out_dir)
    # print the plotted numbers so the paper text can be checked against them
    print('layer bytes MiB:', {k: round(v, 1) for k, v in mixtral_8x22b_layer_bytes().items()})
    print('initiator us:', initiator_latency())
    print('unified bound us (store carries data, NIC reads payload):', unified_bound())
    ce, rate, floor = dispatch_bound()
    print(f'dispatch bound: NIC rate {rate / 1e3:.2f} GB/s, floor {floor:.2f} us')
    print(ce.to_string(index=False))
    ks, c = held_sm_curves()
    print('held SMs', ks, {k: (np.round(v, 1).tolist() if isinstance(v, list) else round(v, 1)) for k, v in c.items()})
    d = dispatch_sweep()
    print(d.sort_values(['variant', 'H', 'tokens']).to_string(index=False))


if __name__ == '__main__':
    main()
