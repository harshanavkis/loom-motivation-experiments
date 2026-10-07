#!/usr/bin/env python3
"""How much of an MoE dispatch leaves the scale-up domain, from DeepSeek-V3's routing alone.

Routing (DeepSeek-V3 report, Section 2.1.2 / 3.2.2): 256 routed experts in 8 groups of 32
consecutive experts; each token picks its top-4 groups and its top-8 experts within them. With
independent random affinities this is: 4 of 8 groups uniformly, then 8 of their 128 experts
uniformly. Experts are placed contiguously, 256 / EP per GPU; GPUs fill scale-up domains of D
GPUs in order. The source GPU of a token is uniform. A token is sent once to every GPU that hosts
one of its experts (DeepEP sends one copy per destination GPU).

Per token: destination GPUs, of which outside the source's scale-up domain, and the number of
distinct remote domains (one copy per domain if a GPU there forwards it, as DeepEP V1 normal /
V2.5 hybrid do). Output: routing_traffic.csv (means over N tokens).
Usage: python3 routing_traffic.py [N]   (default 50000 tokens, seed 1: reproduces routing_traffic.csv)
"""
import csv
import sys
import numpy as np

E, GROUPS, TOPG, TOPK = 256, 8, 4, 8


def sample(ep, d, n, rng):
    per_gpu = E // ep
    gsize = E // GROUPS
    # 4 of 8 groups, then 8 experts among the chosen groups' 128 experts
    groups = np.argsort(rng.random((n, GROUPS)), axis=1)[:, :TOPG]
    pick = np.argsort(rng.random((n, TOPG * gsize)), axis=1)[:, :TOPK]
    experts = groups[np.arange(n)[:, None], pick // gsize] * gsize + pick % gsize
    gpus = experts // per_gpu
    src = rng.integers(0, ep, n)
    dst, rem, rdom = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        g = np.unique(gpus[i])
        far = g[g // d != src[i] // d]
        dst[i], rem[i], rdom[i] = len(g), len(far), len(np.unique(far // d))
    return dst.mean(), rem.mean(), rdom.mean()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000   # the committed CSV
    rng = np.random.default_rng(1)
    rows = []
    for d, name in ((8, 'HGX (8 GPUs)'), (64, 'NVL72 (64 GPUs for EP)')):
        for ep in (16, 32, 64, 128, 256):
            if ep <= d:
                dst, rem, rdom = sample(ep, ep, n, rng)
            else:
                dst, rem, rdom = sample(ep, d, n, rng)
            rows.append(dict(domain=d, domain_name=name, ep=ep, dst_gpus=round(dst, 3), remote_gpus=round(rem, 3),
                             remote_domains=round(rdom, 3), remote_share=round(rem / dst, 4)))
            print(rows[-1])
    with open('routing_traffic.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
