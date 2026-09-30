import math
exec(open('sm_est.py').read().split("import itertools")[0])
def est20(num_experts, num_topk, num_rdma, num_nvl, dev_sms, nvlink_gbs, rdma_gbs, rd=200, wr=50):
    num_ranks=num_rdma*num_nvl; so,su=num_rdma,num_nvl
    def etk(g): return g*(1-math.comb(num_experts-num_experts//g, num_topk)/math.comb(num_experts, num_topk))
    eso=etk(so) if so>1 else 0; et=etk(num_ranks); r=w=rt=nt=0; r+=1/et
    if so>1:
        w+=1/et; w+=(1/et)*(eso/so); rt+=(1/et)*(eso*(1-1/so)); r+=eso/et; w+=1; nt+=1-1/su
    else:
        w+=num_nvl/num_ranks; nt+=num_nvl/num_ranks*(1-1/num_nvl)
    if so>1 and rt/rdma_gbs>nt/nvlink_gbs: bt,bg=rt,rdma_gbs
    else: bt,bg=nt,nvlink_gbs
    n=max(bg/bt*r/rd, bg/bt*w/wr)
    return min(align(max(4,math.ceil(n*1.25)),2),dev_sms)
for nr in (2,4,8):
    print('V2.0 formula H800-like (NVL 160GB/s*0.9? use 400*.9), EP8x%d:'%nr, est20(256,8,nr,8,132,18*26.562*0.9,50), ' H800 nvl=160:', est20(256,8,nr,8,132,160,50))
