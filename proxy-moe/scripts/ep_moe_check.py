"""Run the production aiter.fused_moe on the proxy EP8 MoE shape with expert_mask.

Inputs mirror SGLang's Quark w4a4 MXFP4 path on gfx950 (uint8 weights shuffled (16,16),
e8m0_shuffle'd scales, is_shuffled, GateMode.SEPARATED, global topk ids + expert_mask).
Weights are random bytes: both kernel configs read the same tensors, so comparing two runs
(selected via AITER_CONFIG_FMOE) checks layout/numerics agreement, not model accuracy.

Usage:
  AITER_CONFIG_FMOE=<csv[:csv]> python3 ep_moe_check.py --tag default --out-dir DIR
  python3 ep_moe_check.py --compare DIR/default.pt DIR/tuned.pt
"""

import argparse
import os

import torch

H, I, E_LOCAL, E_GLOBAL, TOPK = 16384, 4096, 64, 512, 16


def build(m_list, seed=0):
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    g = torch.Generator(device="cuda").manual_seed(seed)

    def rbytes(shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=g)

    def rscale(shape):
        s = torch.randint(120, 125, shape, dtype=torch.uint8, device="cuda", generator=g)
        s0, s1, _ = s.shape
        return e8m0_shuffle(s.view(s0 * s1, -1)).view(s0, s1, -1)

    w1 = shuffle_weight(rbytes((E_LOCAL, 2 * I, H // 2)), (16, 16)).view(torch.float4_e2m1fn_x2)
    w2 = shuffle_weight(rbytes((E_LOCAL, H, I // 2)), (16, 16)).view(torch.float4_e2m1fn_x2)
    w1.is_shuffled = True
    w2.is_shuffled = True
    w1s = rscale((E_LOCAL, 2 * I, H // 32))
    w2s = rscale((E_LOCAL, H, I // 32))
    mask = torch.zeros(E_GLOBAL, dtype=torch.int32, device="cuda")
    mask[:E_LOCAL] = 1
    cases = {}
    for m in m_list:
        x = (torch.randn((m, H), device="cuda", generator=g) / 10).to(torch.bfloat16)
        score = torch.randn((m, E_GLOBAL), device="cuda", generator=g)
        w, ids = torch.topk(torch.softmax(score, -1), TOPK, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).float()
        cases[m] = (x, w, ids.to(torch.int32))
    return w1, w2, w1s, w2s, mask, cases


def run(args):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    m_list = [int(v) for v in args.m.split(",")]
    w1, w2, w1s, w2s, mask, cases = build(m_list)
    outs = {}
    for m, (x, w, ids) in cases.items():
        def call():
            return fused_moe(
                x, w1, w2, w, ids, expert_mask=mask, quant_type=QuantType.per_1x32,
                activation=ActivationType.Silu, w1_scale=w1s, w2_scale=w2s,
                gate_mode=GateMode.SEPARATED.value,
            )
        for _ in range(3):
            out = call()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(args.iters):
            call()
        en.record()
        torch.cuda.synchronize()
        us = st.elapsed_time(en) * 1000 / args.iters
        outs[m] = out.float().cpu()
        print(f"[{args.tag}] M={m} us={us:.1f} finite={bool(torch.isfinite(out).all())} "
              f"absmean={out.float().abs().mean():.4g}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(outs, os.path.join(args.out_dir, f"{args.tag}.pt"))


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    for m in sorted(a):
        x, y = a[m].double(), b[m].double()
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        rel = ((x - y).norm() / x.norm().clamp_min(1e-12)).item()
        print(f"M={m} cos={cos:.6f} rel_l2={rel:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="2048,4096,8192")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="/tmp/ep_moe_check")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a)
