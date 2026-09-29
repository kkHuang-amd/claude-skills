# Per-layer check of the FP8 [32,32] -> MXFP4 shared-expert requant used by
# --enforce-shared-experts-fusion (fp8_utils.quantize_block_fp8_weight_to_mxfp4).
# Compares dequant(requant) against the FP8 dequant for every shared expert
# (layers.* and mtp.*), for the current SGLang quantizer and alternatives:
#   cur  = sglang MXFP4QuantizeUtil (scale ceil(log2(amax/6)), ties toward zero)
#   rne  = same scale, round-to-nearest-even
#   ocp  = aiter dynamic_mxfp4_quant scale (amax mantissa round, floor-2), RNE
#   mse  = per 1x32 block best of {ceil, ceil-1} scale by block MSE, RNE
# Metrics: rel = ||Wq-W||/||W||, gain = <Wq,W>/<W,W> (shrinkage if < 1).
# Env: MODEL (checkpoint dir), SRC (sglang python dir), LAYERS ("all" or "0,1,39").
# Run on a free GPU:  HIP_VISIBLE_DEVICES=6 python3 shared_expert_requant_check.py
# Output: one line per (layer, matrix) + summary to stdout.
import json
import os
import re
import sys

import torch
from safetensors import safe_open

MODEL = os.environ.get("MODEL", "/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash")
sys.path.insert(0, os.environ.get("SRC", "/sgl-workspace/sglang-rolao-opt/python"))
from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
    block_quant_dequant,
    quantize_block_fp8_weight_to_mxfp4,
)
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil  # noqa: E402

GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def fp4_rne(y):
    g = GRID.to(y.device)
    d = (y.abs().unsqueeze(-1) - g).abs()
    d = d + (torch.arange(8, device=y.device) % 2) * 1e-7
    return torch.sign(y) * g[d.argmin(-1)]


def q_with_exp(xb, e):
    return fp4_rne(xb / torch.exp2(e)) * torch.exp2(e)


def variants(w):
    xb = w.float().reshape(-1, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(2.0**-126)
    e_ceil = torch.ceil(torch.log2(amax / 6.0)).clamp_min(-127)
    out = {"rne": q_with_exp(xb, e_ceil)}
    ai = ((amax.view(torch.int32) + 0x200000) & 0xFF800000).view(torch.float32)
    out["ocp"] = q_with_exp(xb, torch.floor(torch.log2(ai)) - 2)
    a, b = q_with_exp(xb, e_ceil), q_with_exp(xb, e_ceil - 1)
    ea = ((a - xb) ** 2).sum(-1, keepdim=True)
    eb = ((b - xb) ** 2).sum(-1, keepdim=True)
    out["mse"] = torch.where(eb < ea, b, a)
    return {k: v.reshape(w.shape) for k, v in out.items()}


def metrics(q, w):
    w = w.float()
    return (
        ((q - w).norm() / w.norm()).item(),
        ((q * w).sum() / (w * w).sum()).item(),
    )


def mlp(x, w1, w3, w2, limit=10.0):
    g = (x @ w1.t()).clamp(max=limit)
    u = (x @ w3.t()).clamp(-limit, limit)
    return (torch.nn.functional.silu(g) * u) @ w2.t()


def main():
    wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    names = sorted(
        {k[: -len(".weight")] for k in wm if re.search(r"shared_experts\.w[123]\.weight$", k)}
    )
    sel = os.environ.get("LAYERS", "all")
    if sel != "all":
        keep = {int(x) for x in sel.split(",")}
        names = [n for n in names if int(n.split(".")[1]) in keep]
    dev = "cuda"
    tot = {}
    per_layer = {}
    for n in names:
        with safe_open(f"{MODEL}/{wm[n + '.weight']}", "pt", device=dev) as f:
            qw = f.get_tensor(n + ".weight")
        with safe_open(f"{MODEL}/{wm[n + '.scale']}", "pt", device=dev) as f:
            sc = f.get_tensor(n + ".scale")
        w = block_quant_dequant(qw, sc.to(torch.float32), [32, 32], torch.float32)
        p4, s4 = quantize_block_fp8_weight_to_mxfp4(qw, sc, [32, 32])
        cur = MXFP4QuantizeUtil.dequantize(
            p4.view(torch.uint8), torch.float32, s4.view(torch.uint8), [32]
        ).reshape(w.shape)
        vs = {"ref": w, "cur": cur, **variants(w)}
        res = {k: metrics(v, w) for k, v in vs.items() if k != "ref"}
        per_layer.setdefault(n.rsplit(".", 1)[0], {})[n.rsplit(".", 1)[1]] = vs
        for k, (r, g) in res.items():
            tot.setdefault(k, []).append((r, g))
        print(
            f"{n:34s} {tuple(qw.shape)} {qw.dtype} sc={sc.dtype} "
            + " ".join(f"{k}:rel={r:.4f},gain={g:.4f}" for k, (r, g) in res.items()),
            flush=True,
        )
    torch.manual_seed(0)
    x = torch.randn(256, next(iter(per_layer.values()))["w1"]["ref"].shape[1], device=dev)
    out_tot = {}
    for lname, m in per_layer.items():
        if len(m) != 3:
            continue
        outs = {k: mlp(x, m["w1"][k], m["w3"][k], m["w2"][k]) for k in m["w1"]}
        ref = outs.pop("ref")
        line = []
        for k, y in outs.items():
            r, g = metrics(y, ref)
            out_tot.setdefault(k, []).append((r, g))
            line.append(f"{k}:rel={r:.4f},gain={g:.4f}")
        print(f"MLP {lname:30s} " + " ".join(line), flush=True)
        per_layer[lname] = None
    print("SUMMARY MLP output (mean over layers, x~N(0,1))")
    for k, v in out_tot.items():
        print(
            f"  {k}: rel={sum(a for a, _ in v) / len(v):.4f} "
            f"gain={sum(b for _, b in v) / len(v):.4f} n={len(v)}"
        )
    print("SUMMARY (mean over matrices)")
    for k, v in tot.items():
        r = sum(x[0] for x in v) / len(v)
        g = sum(x[1] for x in v) / len(v)
        print(f"  {k}: rel={r:.4f} gain={g:.4f} n={len(v)}")


if __name__ == "__main__":
    main()
