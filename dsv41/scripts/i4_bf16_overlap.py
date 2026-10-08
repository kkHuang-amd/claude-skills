#!/usr/bin/env python3
"""I4 bf16-logits accuracy proxy: how much does storing indexer logits in bf16 change the top-k set,
compared with the change the FP4 q/k quantization already makes?
  HIP_VISIBLE_DEVICES=0 python3 i4_bf16_overlap.py
Synthetic V4.1-like logits: logit[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]), H=32, D=128, random
gaussian q/k (no attention structure, so dense near-ties at the threshold: a pessimistic case).
Reference = top-k of the exact fp32 logits. Reports set overlap and "regret" = 1 - (sum of exact logits
over the chosen set) / (sum over the exact top-k set).
Env: ROWS (default 256), LCS (default 16384,131072), TOPK (default 512), KEY_SCALE (std of k, default 1).
"""
import os

import torch

dev = "cuda"
ROWS = int(os.environ.get("ROWS", 256))
LCS = [int(x) for x in os.environ.get("LCS", "16384,131072").split(",")]
TOPK = int(os.environ.get("TOPK", 512))
H, D = 32, 128
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)


def fake_fp4(x):
    """MXFP4 fake-quant: e2m1 codes, one power-of-two scale per 32 elements (ue8m0)."""
    shp = x.shape
    x = x.reshape(-1, 32).float()
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    y = (x / scale).abs()
    idx = (y[..., None] - E2M1).abs().argmin(dim=-1)
    return (E2M1[idx] * x.sign() * scale).reshape(shp)


def logits_of(q, k, w):
    out = torch.zeros(q.shape[0], k.shape[0], device=dev)
    for h in range(H):
        out += w[:, h : h + 1] * torch.relu(q[:, h] @ k.T)
    return out


def score(exact, chosen_idx, ref_idx):
    ref_sum = exact.gather(1, ref_idx).sum(1)
    got_sum = exact.gather(1, chosen_idx).sum(1)
    regret = (1 - got_sum / ref_sum).mean().item()
    a = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, ref_idx, True)
    overlap = a.gather(1, chosen_idx).float().mean().item()
    return overlap, regret


def main():
    ks = float(os.environ.get("KEY_SCALE", 1.0))
    print(f"rows={ROWS} topk={TOPK} H={H} D={D} key_scale={ks}")
    for lc in LCS:
        torch.manual_seed(0)
        q = torch.randn(ROWS, H, D, device=dev)
        k = torch.randn(lc, D, device=dev) * ks
        w = torch.randn(ROWS, H, device=dev) / H**0.5
        exact = logits_of(q, k, w)
        ref = exact.topk(TOPK, dim=1).indices
        fp4 = logits_of(fake_fp4(q), fake_fp4(k), w)
        rows = {
            "fp32 logits, fp4 q/k (today)": fp4,
            "bf16 logits, fp4 q/k": fp4.bfloat16(),
            "fp16 logits, fp4 q/k": fp4.half(),
            "bf16 logits, exact q/k": exact.bfloat16(),
        }
        today = fp4.topk(TOPK, dim=1).indices
        for name, lg in rows.items():
            idx = lg.topk(TOPK, dim=1).indices
            ov, rg = score(exact, idx, ref)
            ov_t, _ = score(exact, idx, today)
            print(f"lc={lc:6d} {name:30s} overlap vs exact {ov:.4f}  regret {rg:.2e}  overlap vs today {ov_t:.4f}")
        del exact, fp4
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
