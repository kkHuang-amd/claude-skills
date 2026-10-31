#!/usr/bin/env python3
"""Run the HIP KDA verify kernel (fp32 ring) eagerly N times at batch B, for rocprofv3.
Usage: SGLANG_ROCM_KDA_VERIFY_HIP_FP32_RING=1 kda_verify_one.py [B=16] [N=50]"""
import sys
import torch

sys.path.insert(0, "/sgl-workspace/sglang-k31-amd/test/registered/kernels/ops/attention")
from test_kda_verify_gfx950 import _inputs, _run  # noqa: E402
from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import kda_verify_gfx950  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
inp = _inputs(B, 12, False, seed=0)
inp["rawv"], inp["rawk"] = inp["rawv"].float(), inp["rawk"].float()
for _ in range(N):
    _run(kda_verify_gfx950, inp, 12, -5.0)
torch.cuda.synchronize()
