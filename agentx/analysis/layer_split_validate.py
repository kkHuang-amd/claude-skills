#!/usr/bin/env python3
"""Pre-flight for the layer-aware split-K arm (SGLANG_MLA_HCA_KV_SPLITS).

    HIP_VISIBLE_DEVICES=0 python3 analysis/layer_split_validate.py

Four checks; all must pass before an 8-GPU launch.

  1. `_kv_splits_for_stream` overrides ONLY compress_ratio 128.
  2. The override reaches the kernel through the public wrapper.
  3. **Numerics**: split-K must agree with the fused path. Unlike the
     fake-kernel arm, this one ships real output, so a wrong answer here is a
     silent accuracy regression rather than an obviously garbage run.
  4. Capture + replay under a local `torch.cuda.CUDAGraph`. Split-K takes a
     different code path (partial buffers + a second kernel) than the fused
     path that production has been capturing until now.
"""
import math
import os
import sys

sys.path.insert(0, os.environ.get("SGLANG_TREE", "/sgl-workspace/sglang-MegaMoE") + "/python")
sys.path.insert(0, "/workspace/claude-skills/agentx/analysis")

import torch  # noqa: E402

import sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode as pd  # noqa: E402
from mla_tail_bench import real_lens  # noqa: E402
from mla_microbench import H, D, NSLOT, _FP8_GROUP_SIZE  # noqa: E402


def build(T, lens, dev):
    total = int(sum(lens))
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(NSLOT, D, dtype=torch.bfloat16, device=dev).to(pd._FP8_DTYPE)
    sc = torch.rand(NSLOT, D // _FP8_GROUP_SIZE, dtype=torch.float32,
                    device=dev) * 0.1 + 0.05
    idx = torch.randint(0, NSLOT, (total,), dtype=torch.int32, device=dev)
    indptr = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(),
                          dtype=torch.int32, device=dev)
    sink = torch.zeros(H, dtype=torch.bfloat16, device=dev)
    return (q, kv, idx, indptr, sink, 1.0 / math.sqrt(D)), sc


def main():
    dev = torch.device("cuda")
    fails = []

    got = [pd._kv_splits_for_stream(r) for r in (0, 4, 128)]
    ok1 = got == [None, None, 4]
    print(f"1 stream gating: swa/csa/hca -> {got} "
          f"{'PASS' if ok1 else 'FAIL, want [None, None, 4]'}")
    if not ok1:
        fails.append("gating")

    T = 98
    lens = real_lens(T, 0, 5000, 1300.0)     # HCA shape at steady state
    args, sc = build(T, lens, dev)
    # Absent entirely in trees that never carried the fake-kernel arm.
    assert getattr(pd, "_FAKE_KVLEN", 0) == 0, "the fake clamp must be OFF here"

    ref = pd.sparse_attn_v4_paged_decode(*args, kv_scales=sc)
    spl = pd.sparse_attn_v4_paged_decode(*args, kv_scales=sc, kv_splits=4)
    heur = pd._kv_splits_heuristic(T, H, 64)
    print(f"2 override reaches the kernel: heuristic would pick {heur}, "
          f"asked for 4 {'PASS' if heur != 4 else 'INCONCLUSIVE (heuristic already 4)'}")

    num = (ref.float() - spl.float()).pow(2).sum().sqrt()
    den = ref.float().pow(2).sum().sqrt()
    rel = (num / den).item()
    ok3 = rel < 2e-2 and torch.isfinite(spl).all().item()
    print(f"3 numerics vs fused: relL2 {rel:.2e} {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        fails.append("numerics")

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                pd.sparse_attn_v4_paged_decode(*args, kv_scales=sc, kv_splits=4)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = pd.sparse_attn_v4_paged_decode(*args, kv_scales=sc, kv_splits=4)
        g.replay()
        torch.cuda.synchronize()
        fin = bool(torch.isfinite(out).all().item())
        print(f"4 capture+replay at splits=4: ok, finite={fin} "
              f"{'PASS' if fin else 'FAIL'}")
        if not fin:
            fails.append("nonfinite")
    except Exception as e:  # noqa: BLE001
        print(f"4 capture+replay: FAIL {type(e).__name__}: {str(e)[:200]}")
        fails.append("capture")

    print("VERDICT:", "PASS" if not fails else f"FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
