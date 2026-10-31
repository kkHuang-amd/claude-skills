#!/usr/bin/env python3
"""KDA spec-verify microbench on one gfx950 GPU (server must be down).
Per-layer us for K3.1 TP8 shapes (H=12, K=V=128, T=8) at B = 4 / 16 / 48.
Variants: {HIP kda_verify_gfx950, Triton fused_kda_conv_gating_verify} x {ring bf16, ring fp32, snapshot}.
Timing: 20 calls captured in one CUDA graph, replayed; reports per-call us.
Production path (Triton conv1d_update + fused_sigmoid_gating + ...) is not callable standalone;
compare against the trace (c16: conv 5.7 + recurrence 26.3 us per layer)."""
import sys
import torch

sys.path.insert(0, "/sgl-workspace/sglang-k31-amd/test/registered/kernels/ops/attention")
from test_kda_verify_gfx950 import _inputs, _run, _T, _K  # noqa: E402
from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import fused_kda_conv_gating_verify  # noqa: E402
from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import kda_verify_gfx950, supports_kda_verify_gfx950  # noqa: E402

H, LB, CALLS, REPS = 12, -5.0, 20, 20


def bench(fn, inp, snap):
    for _ in range(3):
        _run(fn, inp, H, LB, snapshots=snap)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(CALLS):
                _run(fn, inp, H, LB, snapshots=snap)
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(REPS):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / (REPS * CALLS)


def supported_hip(inp, snap):
    return supports_kda_verify_gfx950(
        num_q_heads=H, num_v_heads=H, head_k_dim=_K, head_v_dim=_K, T=_T,
        mixed_qkv=inp["mixed"], conv_weight=inp["w"], a=inp["a"], b=inp["b"], ssm_states=inp["ssm"],
        replayssm_rawv=None if snap is not None else inp["rawv"],
        intermediate_states_buffer=snap, intermediate_conv_window=inp["win"].transpose(-1, -2))


if "--fb" in sys.argv:
    # 5d-1: fused f_b vs (f_b GEMM + verify kernel), both with the fp32 ring
    import functools
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

    def bench_fn(f):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                for _ in range(CALLS):
                    f()
        torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(REPS):
            g.replay()
        en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en) * 1e3 / (REPS * CALLS)

    print(f"{'B':>3} {'tiny_gemm f_b':>14s} {'verify':>8s} {'sum':>8s} {'fused':>8s}")
    for B in (4, 16, 48):
        inp = _inputs(B, H, False, seed=0)
        inp["rawv"] = inp["rawv"].float(); inp["rawk"] = inp["rawk"].float()
        f_a = torch.randn(B * _T, _K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H * _K, _K, device="cuda", dtype=torch.bfloat16) * 0.1
        t_g = bench_fn(lambda: kimi_k3_tiny_gemm(f_a, w))
        t_v = bench_fn(lambda: _run(kda_verify_gfx950, inp, H, LB))
        fused_inp = dict(inp, a=f_a)
        fused = functools.partial(kda_verify_gfx950, f_b_weight=w)
        t_f = bench_fn(lambda: _run(fused, fused_inp, H, LB))
        print(f"{B:>3} {t_g:14.2f} {t_v:8.2f} {t_g + t_v:8.2f} {t_f:8.2f}", flush=True)
    sys.exit(0)

print(f"{'B':>3} {'variant':28s} {'us/layer':>9s}  state MB  floor_us(8TB/s read)")
for B in (4, 16, 48):
    for mode in ("ring_bf16", "ring_fp32", "snapshot"):
        inp = _inputs(B, H, False, seed=0)
        snap = None
        if mode == "ring_fp32":
            inp["rawv"] = inp["rawv"].float(); inp["rawk"] = inp["rawk"].float()
        if mode == "snapshot":
            snap = torch.zeros(B + 2, _T, H, _K, _K, device="cuda", dtype=torch.float32)
        state_mb = B * H * _K * _K * 4 / 1e6
        for name, fn in (("hip", kda_verify_gfx950), ("triton_fused", fused_kda_conv_gating_verify)):
            tag = f"{name}/{mode}"
            if name == "hip" and not supported_hip(inp, snap):
                print(f"{B:>3} {tag:28s} {'unsupported':>9s}"); continue
            try:
                us = bench(fn, inp, snap)
                print(f"{B:>3} {tag:28s} {us:9.2f}  {state_mb:7.1f}  {state_mb/8e6*1e6:6.2f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{B:>3} {tag:28s} ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
