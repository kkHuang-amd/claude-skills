import torch, sys
import torch.nn.functional as F

def load(d):
    return torch.load(f"/workspace/moe1_dump_{d}/full.pt", map_location="cpu", weights_only=False)

def analyze(name, d):
    ids = d["topk_ids"].long()          # [T, k]
    w   = d["topk_weights"].float()     # [T, k]
    h   = d["hidden"].float()           # [T, H]
    T, k = ids.shape
    H = h.shape[1]
    flat = ids.reshape(-1)
    uniq, cnt = torch.unique(flat, return_counts=True)
    n_active = uniq.numel()
    cnt_sorted = torch.sort(cnt, descending=True).values
    total = flat.numel()
    # concentration: fraction of selections covered by top-10 experts
    top10 = cnt_sorted[:10].sum().item() / total
    # pairwise cosine similarity of the T hidden vectors (token diversity)
    hn = F.normalize(h, dim=1)
    sim = hn @ hn.t()                    # [T,T]
    off = sim[~torch.eye(T, dtype=torch.bool)]
    # norms
    norms = h.norm(dim=1)
    print(f"\n===== {name} =====")
    print(f"  tokens={T} topk={k} total_sel={total}  H={H}")
    print(f"  ACTIVE EXPERTS = {n_active}  (uniform-expect ~{int(256*(1-(1-1/256)**total))})")
    print(f"  expert count: max={cnt.max().item()} mean={cnt.float().mean():.2f} "
          f"top10_share={top10*100:.1f}%")
    print(f"  top-12 expert loads: {cnt_sorted[:12].tolist()}")
    print(f"  hidden norms: min={norms.min():.2f} max={norms.max():.2f} "
          f"mean={norms.mean():.2f} std={norms.std():.2f}")
    print(f"  intra-batch cosine (token similarity): mean={off.mean():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}  >0.9 frac={(off>0.9).float().mean()*100:.1f}%")
    print(f"  topk_weight: mean={w.mean():.4f} sum/token mean={w.sum(1).mean():.4f}")
    return n_active

a = analyze("ATOM", load("atom"))
s = analyze("SGLANG", load("sgl"))
print(f"\n>>> active experts  ATOM={a}  SGL={s}")
