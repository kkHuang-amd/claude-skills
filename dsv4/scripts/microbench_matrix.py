import os
os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")   # force flydsl kernel
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from aiter.fused_moe import fused_moe
from aiter import QuantType, ActivationType
from aiter.ops.flydsl.moe_common import GateMode
import torch.profiler as P

dev = "cuda"
SGL = torch.load("/workspace/moe1_dump_sgl/full.pt")
ATOM = torch.load("/workspace/moe1_dump_atom/full.pt")
g = lambda d, k: None if d.get(k) is None else d[k].to(dev)

# common hidden for ALL runs (timing is independent of hidden values)
HID = g(SGL, "hidden")

def routing(d):
    return g(d, "topk_ids"), g(d, "topk_weights")
def active(tid):
    f = tid.reshape(-1).long(); f = f[f < 384]
    return torch.unique(f).numel()

R216 = routing(SGL)   # SGL's data routing
R103 = routing(ATOM)  # ATOM's data routing
print(f"routings: SGL-data active={active(R216[0])}  ATOM-data active={active(R103[0])}")

def weights(d):
    return dict(w1=g(d,"w13"), w2=g(d,"w2"), w1_scale=g(d,"w13_scale"), w2_scale=g(d,"w2_scale"))

WSGL = weights(SGL); WATOM = weights(ATOM)

def bench(label, W, R, pad):
    tid, tw = R
    def call():
        return fused_moe(hidden_states=HID, topk_weight=tw, topk_ids=tid,
            quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
            a1_scale=None, a2_scale=None, gate_mode=GateMode.INTERLEAVE.value,
            swiglu_limit=10.0, intermediate_pad=pad, **W)
    try:
        for _ in range(15): call()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  [{label}] ERROR {e}"); return
    with P.profile(activities=[P.ProfilerActivity.CUDA]) as prof:
        for _ in range(50): call()
        torch.cuda.synchronize()
    m1 = m2 = 0.0
    for e in prof.key_averages():
        if "mfma_moe1_silu" in e.key: m1 = e.device_time_total / e.count
        elif "mfma_moe2" in e.key:    m2 = e.device_time_total / e.count
    print(f"  [{label:38}] moe1={m1:6.1f}us  moe2={m2:6.1f}us")

print("\n=== matrix (same kernel; vary weights / routing / pad) ===")
print("-- effect of the FIX (SGL weights, SGL-data routing 216) --")
bench("SGL-w  R216  pad=0   (orig/bug)", WSGL, R216, 0)
bench("SGL-w  R216  pad=128 (FIXED)",    WSGL, R216, 128)
print("-- STRUCTURAL: same routing+pad, SGL-w vs ATOM-w --")
bench("ATOM-w R216  pad=128",            WATOM, R216, 128)
bench("SGL-w  R103  pad=128",            WSGL, R103, 128)
bench("ATOM-w R103  pad=128 (ATOM base)",WATOM, R103, 128)
print("-- effect of ROUTING (SGL-w, pad=128): 216 vs 103 --")
print("   (compare 'SGL-w R216 pad=128' vs 'SGL-w R103 pad=128' above)")
