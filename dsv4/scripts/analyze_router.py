import torch
import torch.nn.functional as F

A = torch.load("/workspace/router_dump/router_atom.pt", map_location="cpu", weights_only=False)
S = torch.load("/workspace/router_dump/router_sgl.pt",  map_location="cpu", weights_only=False)
N_ROUTED = 384

def active(ids):
    f = ids.reshape(-1).long()
    f = f[f < N_ROUTED]            # ignore fused-shared expert id(s)
    return torch.unique(f).numel()

def ref_select(logits, bias, k=6):
    """DSv4 routing: select top-k by sqrt(softplus(logits)) + correction_bias."""
    sc = torch.sqrt(F.softplus(logits.float()))
    sel = sc + (bias.float() if bias is not None else 0.0)
    return torch.topk(sel, k, dim=-1).indices

def hidden_stats(h):
    h = h.float()
    n = h.norm(dim=1)
    hn = F.normalize(h, dim=1)
    sim = hn @ hn.t()
    off = sim[~torch.eye(h.shape[0], dtype=torch.bool)]
    return n.mean().item(), n.std().item(), off.mean().item()

def meta(d, lid, key, default=None):
    return d[lid].get("meta", {}).get(key, default)

layers = sorted(set(A.keys()) & set(S.keys()))
hash_layers = [l for l in layers if meta(A, l, "is_hash")]
print(f"layers={len(layers)} hash_layers={hash_layers}")
print(f"ATOM meta@layer5: {A[5].get('meta')}")
print(f"SGL  meta@layer5: {S[5].get('meta')}")

# validate ref_select reproduces each engine's actual top-k (overlap) on its own logits
def overlap_with_actual(d, lid, k=6):
    if "router_logits" not in d[lid] or "topk_ids" not in d[lid]:
        return None
    ref = ref_select(d[lid]["router_logits"], d[lid].get("correction_bias"), k)
    act = d[lid]["topk_ids"].long()
    ak = act.shape[1]
    inter = 0; tot = 0
    for i in range(ref.shape[0]):
        rset = set(ref[i].tolist()); aset = set(x for x in act[i].tolist() if x < N_ROUTED)
        inter += len(rset & aset); tot += len(aset)
    return inter / max(tot, 1)

print("\n=== validation: ref_select vs engine actual topk (mean per-token overlap frac) ===")
for lid in [3, 5, 10, 30, 60]:
    print(f"  layer{lid}: ATOM={overlap_with_actual(A,lid):.3f}  SGL={overlap_with_actual(S,lid):.3f}")

print("\n=== per-layer active experts & hidden homogeneity (identical 64 prompts) ===")
print(f"{'L':>3} {'hash':>4} | {'act_A':>5} {'act_S':>5} | {'refA':>5} {'refS':>5} | "
      f"{'hnstdA':>7} {'hnstdS':>7} | {'cosA':>5} {'cosS':>5}")
sel_layers = [0,1,2,3,4,5,8,12,20,30,40,50,60]
for lid in sel_layers:
    if lid not in layers: continue
    actA = active(A[lid]["topk_ids"]); actS = active(S[lid]["topk_ids"])
    refA = active(ref_select(A[lid]["router_logits"], A[lid].get("correction_bias"))) \
        if "router_logits" in A[lid] else -1
    refS = active(ref_select(S[lid]["router_logits"], S[lid].get("correction_bias"))) \
        if "router_logits" in S[lid] else -1
    _, hnA, cosA = hidden_stats(A[lid]["hidden"]) if "hidden" in A[lid] else (0,-1,-1)
    _, hnS, cosS = hidden_stats(S[lid]["hidden"]) if "hidden" in S[lid] else (0,-1,-1)
    ish = "Y" if meta(A,lid,"is_hash") else ""
    print(f"{lid:>3} {ish:>4} | {actA:>5} {actS:>5} | {refA:>5} {refS:>5} | "
          f"{hnA:>7.3f} {hnS:>7.3f} | {cosA:>5.3f} {cosS:>5.3f}")

# averages over non-hash layers
nh = [l for l in layers if not meta(A,l,"is_hash")]
import statistics as st
aA = st.mean(active(A[l]["topk_ids"]) for l in nh)
aS = st.mean(active(S[l]["topk_ids"]) for l in nh)
print(f"\nNON-HASH avg active: ATOM={aA:.1f}  SGL={aS:.1f}")
