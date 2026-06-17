import torch
import torch.nn.functional as F

A = torch.load("/workspace/router_dump/router_atom.pt", map_location="cpu", weights_only=False)
S = torch.load("/workspace/router_dump/router_sgl.pt",  map_location="cpu", weights_only=False)

def match(lid):
    ha = F.normalize(A[lid]["hidden"].float(), dim=1)
    hs = F.normalize(S[lid]["hidden"].float(), dim=1)
    M = ha @ hs.t()                      # [64,64] cross-engine cosine
    # best SGL match for each ATOM token
    best, idx = M.max(dim=1)
    # logits agreement on matched pairs
    la = A[lid]["router_logits"].float()
    ls = S[lid]["router_logits"].float()
    lcos = F.cosine_similarity(la, ls[idx], dim=1)
    # are the matched tokens a permutation (unique)?
    uniq = torch.unique(idx).numel()
    return best.mean().item(), best.min().item(), uniq, lcos.mean().item()

print(f"{'L':>3} | best-match hidden cos (mean/min) | uniqSGL | logits cos(mean)")
for lid in [3,5,8,12,20,30,40,50,60]:
    if lid not in A or "hidden" not in A[lid] or "hidden" not in S.get(lid,{}): continue
    bm, bmin, uq, lc = match(lid)
    print(f"{lid:>3} |   {bm:6.4f} / {bmin:6.4f}            |  {uq:>3}/64 |   {lc:6.4f}")
