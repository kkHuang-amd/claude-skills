"""E20 probe: isolate MXFP4-activation (a4w4) vs MXFP8-activation (a8w4) quantization
error on REAL DeepSeek-R1-0528-MXFP4 layer-3 expert weights.

Weights are natively fp4 in the checkpoint, so the *weight* is identical on both
paths; the ONLY difference between a4w4 (gfx950) and a8w4 (gfx1250) is how the
ACTIVATION is quantized (fp4 vs fp8). Measures each path's FFN-output error vs the
bf16 (fp4-weight, full-precision-activation) reference.

Result (2026-07-08, gfx1250): a4w4 mean rel_l2 ~0.151, a8w4 ~0.036 (ratio 0.24) =>
a8w4 is 4x MORE accurate than a4w4 => MoE quant is NOT the gfx1250 accuracy gap.

Run:  HIP_VISIBLE_DEVICES=0 ENABLE_CK=0 python3 moe_quant_probe.py
Adjust MODEL path if the checkpoint moved.
"""
import json
import torch
from safetensors import safe_open

MODEL = "/dockerx/data/models/DeepSeek-R1-0528-MXFP4"
LAYER = 3
N_EXPERTS = 8
T = 512  # tokens

_MXFP4_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def dequant_mxfp4(weight_u8, scale_e8m0):
    N, kp = weight_u8.shape
    K = kp * 2
    lut = torch.tensor(_MXFP4_VALUES, device=weight_u8.device, dtype=torch.float32)
    lo = (weight_u8 & 0xF).long()
    hi = (weight_u8 >> 4).long()
    vals = torch.empty(N, K, device=weight_u8.device, dtype=torch.float32)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    scale = torch.exp2(scale_e8m0.to(torch.float32) - 127.0)
    scale = torch.where(scale_e8m0 == 255, torch.zeros_like(scale), scale)
    scale = scale.view(N, K // 32, 1)
    return (vals.view(N, K // 32, 32) * scale).view(N, K).to(torch.bfloat16)


def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (torch.norm(a - b) / torch.norm(b)).item()


def load_expert(f, e):
    g = dequant_mxfp4(f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.gate_proj.weight").cuda(),
                      f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.gate_proj.weight_scale").cuda())
    u = dequant_mxfp4(f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.up_proj.weight").cuda(),
                      f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.up_proj.weight_scale").cuda())
    d = dequant_mxfp4(f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.down_proj.weight").cuda(),
                      f.get_tensor(f"model.layers.{LAYER}.mlp.experts.{e}.down_proj.weight_scale").cuda())
    return g, u, d


def q_fp4_roundtrip(x):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    xq, xs = dynamic_mxfp4_quant(x)
    return dequant_mxfp4(xq, xs)


def q_fp8_roundtrip(x):
    from aiter.ops.triton.quant import dynamic_mxfp8_quant
    from aiter import dtypes
    xq, xs = dynamic_mxfp8_quant(x, quant_dtype=dtypes.fp8)
    N, K = x.shape
    xf = xq.to(torch.float32).view(N, K // 32, 32)
    sc = torch.exp2(xs.to(torch.float32) - 127.0).view(N, K // 32, 1)
    return (xf * sc).view(N, K).to(torch.bfloat16)


def ffn(x, g, u, d):
    gate = x @ g.T.float()
    up = x @ u.T.float()
    act = torch.nn.functional.silu(gate) * up
    return act @ d.T.float()


def main():
    torch.manual_seed(0)
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    shard = idx[f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight"]
    f = safe_open(f"{MODEL}/{shard}", framework="pt", device="cpu")

    x = torch.randn(T, 7168, device="cuda", dtype=torch.bfloat16)
    x[::37] *= 6.0  # activation outliers (quant stressor)

    x_a4 = q_fp4_roundtrip(x).float()
    x_a8 = q_fp8_roundtrip(x).float()
    xf = x.float()
    print(f"[activation-only roundtrip] a4w4-act rel_l2={rel_l2(x_a4, xf):.4e}  "
          f"a8w4-act rel_l2={rel_l2(x_a8, xf):.4e}")

    e4, e8 = [], []
    for e in range(N_EXPERTS):
        g, u, d = load_expert(f, e)
        gf, uf, df = g.float(), u.float(), d.float()
        ref = ffn(xf, gf, uf, df)
        y4 = ffn(x_a4, gf, uf, df)
        y8 = ffn(x_a8, gf, uf, df)
        r4, r8 = rel_l2(y4, ref), rel_l2(y8, ref)
        e4.append(r4); e8.append(r8)
        print(f"  expert {e}: a4w4 rel_l2={r4:.4e}   a8w4 rel_l2={r8:.4e}")

    import statistics
    print(f"\n[FFN output vs bf16 ref, {N_EXPERTS} experts, T={T}]")
    print(f"  a4w4 (fp4 act, gfx950):  mean rel_l2 = {statistics.mean(e4):.4e}")
    print(f"  a8w4 (fp8 act, gfx1250): mean rel_l2 = {statistics.mean(e8):.4e}")
    print(f"  ratio a8w4/a4w4 = {statistics.mean(e8)/statistics.mean(e4):.3f} "
          "(<1 => fp8 act more accurate)")


if __name__ == "__main__":
    main()
