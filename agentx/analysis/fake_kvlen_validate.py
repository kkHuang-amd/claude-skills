#!/usr/bin/env python3
"""Pre-flight for the fake-kernel arm (SGLANG_MLA_FAKE_KVLEN).

    HIP_VISIBLE_DEVICES=0 python3 analysis/fake_kvlen_validate.py

Three checks, all of which must pass before any 8-GPU launch:

  1. correctness of `_fake_clamp_indptr` on a hand-checkable case;
  2. the clamp actually removes work (eager us with the clamp << without);
  3. a local `torch.cuda.CUDAGraph` capture + replay around the real kernel
     with the clamp active succeeds. Launch #4 of this project died with
     hipErrorStreamCaptureUnsupported because a probe wrote a host scalar into
     a device tensor; nothing goes to the node again without this check.
"""
import math
import sys

sys.path.insert(0, "/sgl-workspace/sglang-MegaMoE/python")
sys.path.insert(0, "/workspace/claude-skills/agentx/analysis")

import torch  # noqa: E402

import sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode as pd  # noqa: E402
from mla_microbench import H, D, NSLOT, _FP8_GROUP_SIZE  # noqa: E402

CAP = 128


def real_lens(T, gen):
    """Per-token kv_len matching the measured production distribution:
    p50 ~1,000, p99 ~3,100, max ~5,000 (HCA layers, context/128)."""
    x = torch.distributions.LogNormal(math.log(1000.0), 0.62).sample((T,))
    return x.clamp(64, 5000).to(torch.int64).tolist()


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


def timed(args, sc, iters=30, warmup=8):
    for _ in range(warmup):
        pd._sparse_attn_v4_paged_decode_triton(*args, kv_scales=sc)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        pd._sparse_attn_v4_paged_decode_triton(*args, kv_scales=sc)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters


def main():
    dev = torch.device("cuda")
    fails = []

    # 1. indptr arithmetic
    src = torch.tensor([0, 5, 8, 8 + 200], dtype=torch.int32, device=dev)
    got = pd._fake_clamp_indptr(src, CAP).tolist()
    want = [0, 5, 8, 136]
    print(f"1 indptr clamp: got {got} want {want} "
          f"{'PASS' if got == want else 'FAIL'}")
    if got != want:
        fails.append("indptr")

    # 2. the clamp removes work
    T = 98
    lens = real_lens(T, None)
    args, sc = build(T, lens, dev)
    pd._FAKE_KVLEN = 0
    us_off = timed(args, sc)
    pd._FAKE_KVLEN = CAP
    us_on = timed(args, sc)
    ok2 = us_on < 0.5 * us_off
    print(f"2 eager us/call: off {us_off:.1f} on {us_on:.1f} "
          f"({us_on / us_off:.2f}x) {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append("no-work-removed")

    # 3. capture + replay with the clamp active
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                pd._sparse_attn_v4_paged_decode_triton(*args, kv_scales=sc)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = pd._sparse_attn_v4_paged_decode_triton(*args, kv_scales=sc)
        g.replay()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all().item())
        print(f"3 capture+replay: ok, output finite={finite} "
              f"{'PASS' if finite else 'FAIL'}")
        if not finite:
            fails.append("nonfinite")
    except Exception as e:  # noqa: BLE001
        print(f"3 capture+replay: FAIL {type(e).__name__}: {str(e)[:200]}")
        fails.append("capture")

    print("VERDICT:", "PASS" if not fails else f"FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
