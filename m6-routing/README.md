# M6: how much of an MoE dispatch leaves the scale-up domain

**Question.** Under DeepSeek-V3's routing, what share of a token's copies goes to GPUs outside the
sender's scale-up domain, as the expert-parallel (EP) group grows past the domain?

**Method** (`routing_traffic.py`, no GPU). DeepSeek-V3 routing (technical report, Sections 2.1.2 and
3.2.2): 256 routed experts in 8 groups of 32; each token picks its top-4 groups and its top-8 experts
within them. With independent random affinities that is 4 of 8 groups uniformly, then 8 of their 128
experts uniformly. Experts are placed contiguously (256 / EP per GPU), GPUs fill scale-up domains of D
GPUs in order, and the sending GPU is uniform. A token is sent once to every GPU that hosts one of its
experts (as DeepEP). Domains: D = 8 (HGX node) and D = 64 (an NVL72 rack, 64 GPUs used for EP because
256 experts do not split over 72). 50,000 tokens, seed 1.

**Results** (`routing_traffic.csv`):

| EP | GPUs per token | outside an 8-GPU domain | outside a 64-GPU domain |
|---|---|---|---|
| 16 | 5.35 | 50% | 0% |
| 32 | 6.60 | 75% | 0% |
| 64 | 7.36 | 88% | 0% |
| 128 | 7.78 | 94% | 50% |
| 256 | 8.00 | 97% | 75% |

The share is about 1 - D/EP: routing has no preference for the sender's domain. `remote_domains` is
the number of distinct remote domains per token (copies if a GPU in each domain forwards, as DeepEP
V1 normal and V2.5 hybrid do).

**Reproduce:**
```sh
cd m6-routing && nix shell --impure --expr '(builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux.python3.withPackages (p: with p; [numpy])' -c python3 routing_traffic.py   # = routing_traffic.csv
```
