# Faithful re-implementation of deep_ep/buffers/ep.py:431-541 (EPBuffer.get_theoretical_num_sms), torch-free
import math
def align(x,y): return -(-x//y)*y
def est(num_experts, num_topk, num_rdma, num_nvl, hybrid, dev_sms, nvlink_gbs, rdma_gbs,
        sm_read_gbs=180, sm_write_gbs=45, prefer_overlap=True):
    num_ranks = num_rdma*num_nvl
    so, su = (num_rdma, num_nvl) if hybrid else (1, num_ranks)
    def etk(g): return g*(1-math.comb(num_experts-num_experts//g, num_topk)/math.comb(num_experts, num_topk))
    eso = etk(so) if so>1 else 0
    et = etk(num_ranks)
    r=w=rt=nt=0
    r += 1/et
    if so>1:
        w += 1/et
        w += (1/et)*(eso/so)
        rt += (1/et)*(eso*(1-1/so))
        r += eso/et
        w += 1
        nt += 1-1/su
    else:
        if num_rdma>1: w += 1/et
        w += num_nvl/num_ranks
        nt += num_nvl/num_ranks*(1-1/num_nvl)
        rt += (num_ranks-num_nvl)/num_ranks
    if so>1 and rt/rdma_gbs > nt/nvlink_gbs: bt,bg,bn=rt,rdma_gbs,'RDMA'
    else: bt,bg,bn=nt,nvlink_gbs,'NVL'
    n = dev_sms
    if bt>0:
        terms = [bg/bt*r/sm_read_gbs, bg/bt*w/sm_write_gbs, bg/bt*(w-nt)/sm_read_gbs, bg/bt*(r+nt)/sm_write_gbs]
        n = max(terms)
    raw = n
    n = align(max(4, math.ceil(n*1.3)),4)
    if not prefer_overlap: n = max(n,64)
    n = min(n, dev_sms//2*2)
    return n, raw, bn, et, eso
import itertools
H = dict(name='H100/H200 (132 SM, NVL4 18x26.56GB/s*0.9, CX7 400G)', sms=132, nvl=18*26.562*0.9, rdma=400/8)
B = dict(name='B200 (148 SM, NVL5 18x50GB/s*0.9, CX7 400G)', sms=148, nvl=18*50*0.9, rdma=400/8)
B8 = dict(name='B200 w/ CX8 800G', sms=148, nvl=18*50*0.9, rdma=800/8)
for g in (H,B,B8):
    print('==', g['name'], f"nvlink_gbs={g['nvl']:.1f} rdma_gbs={g['rdma']}")
    for (nr,nn) in [(1,8),(2,8),(4,8),(8,8),(16,8),(32,8)]:
        for hyb in (True, False):
            if nr==1 and not hyb: continue
            n,raw,bn,et,eso = est(256,8,nr,nn,hyb,g['sms'],g['nvl'],g['rdma'])
            print(f"  EP{nr*nn:<4} ({nr}x{nn}) {'hybrid' if hyb else 'direct'}: num_sms={n:3d} ({100*n/g['sms']:.0f}% of GPU) raw={raw:.1f} bound={bn} E[topk ranks]={et:.2f} E[topk nodes]={eso:.2f}")
print('== pure RDMA, 1 GPU per node (nvlink_gbs=0 irrelevant since NVL traffic=0)')
for sms,name,rg in [(132,'H100/H200 CX7 400G',50),(132,'H100/H200 CX6 200G',25),(188,'RTX PRO 6000 CX7 400G',50),(188,'RTX PRO 6000 CX8 800G',100)]:
    for nr in (2,4,16,64):
        n,raw,bn,et,eso = est(256,8,nr,1,True,sms,1e-9,rg)
        n2,raw2,*_ = est(256,8,nr,1,True,sms,1e-9,rg,prefer_overlap=False)
        print(f"  {name} EP{nr} ({nr}x1): num_sms={n} (raw {raw:.1f}); prefer_overlap=False -> {n2}")
