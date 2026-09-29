#!/usr/bin/env python3
# Standalone aiter fused_moe a8w4 (fp8 act x mxfp4 weight, per_1x32) repro for the DSV4.1-Flash MoE shapes that
# GPU-fault in SGLang TP2 prefill graph capture. Reuses aiter/op_tests/test_moe_2stage.py::test_fmoe (defs only)
# with the AOT-cache guard off so untuned shapes JIT-compile.
#   AITER=/sgl-workspace/aiter  SHAPE=tp2ep1|tp2ep2|tp4ep4  TOKENS="2048 4096"  GATE=interleave|separated
# Run one process per case (a fault kills the process):  HIP_VISIBLE_DEVICES=4 python3 repro_moe_tp2.py
# Prints one "RESULT ..." line per token count; logs go wherever the caller redirects.
import os
import sys

AITER = os.environ.get("AITER", "/sgl-workspace/aiter")
SHAPES = {  # (model_dim, inter_dim per rank, local experts, topk)
    "tp2ep1": (5120, 1152, 384, 6),
    "tp2ep2": (5120, 2304, 192, 6),
    "tp4ep4": (5120, 2304, 96, 6),
}
shape = os.environ.get("SHAPE", "tp2ep1")
model_dim, inter_dim, E, topk = SHAPES[shape]
tokens = [int(t) for t in os.environ.get("TOKENS", "2048 4096").split()]

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
src_path = os.path.join(AITER, "op_tests", "test_moe_2stage.py")
src = open(src_path).read()
src = src[: src.index("args = parser.parse_args()")]
sys.path.insert(0, os.path.dirname(src_path))
sys.argv = [src_path]
g = {"__name__": "moe2stage_defs", "__file__": src_path}
exec(compile(src, src_path, "exec"), g)

import torch  # noqa: E402

aiter, dtypes, GateMode = g["aiter"], g["dtypes"], g["GateMode"]
gate = GateMode.INTERLEAVE.value if os.environ.get("GATE", "interleave") == "interleave" else GateMode.SEPARATED.value
torch.set_default_device("cuda")
for t in tokens:
    print(f"CASE shape={shape} token={t} dim={model_dim} inter={inter_dim} E={E} topk={topk} gate={gate}", flush=True)
    ret = g["test_fmoe"](
        dtypes.bf16, t, model_dim, inter_dim, E, topk, aiter.ActivationType.Silu, gate,
        aiter.QuantType.per_1x32, dtypes.fp8, dtypes.fp4x2,
        use_g1u1=True, preshuffle=True, strict_accuracy=False, check_aot_cache=False,
    )
    torch.cuda.synchronize()
    keep = {k: ret.get(k) for k in ("us", "err", "kernelName1", "kernelName2")} if isinstance(ret, dict) else ret
    print(f"RESULT shape={shape} token={t} ok {keep}", flush=True)
