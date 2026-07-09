"""gfx1250 absorb-BMM quantization EMULATION (bf16 numerics; no fp4 kernel needed).

gfx1250 A0 has NO fp4-activation scaled-WMMA, so it CANNOT run the real a4w4 absorb
gemm (`batched_gemm_afp4wfp4_pre_quant`) — it would crash. This tool instead EMULATES
the numerical effect of quantizing the MLA absorb weights (and, optionally, activations)
by doing an MXFP4 (and MXFP8) quant->dequant back to bf16, then letting the normal bf16
BMM run. That reproduces gfx950's quantized-absorb *numerics* on gfx1250 without any fp4
kernel, so we can test the "quantization-matching" thesis (does making gfx1250's absorb
LESS precise move it toward gfx950's 0.93?).

Modes (env `SGLANG_ABSORB_QUANT_EMUL`):
  - unset/"off" : no-op.
  - "w_fp4"     : w_kc/w_vc -> MXFP4 q-dq (bf16). Absorb becomes a16w4 (fp4 weight, bf16
                  act). Robust, structure-independent (only touches self.w_kc/self.w_vc).
  This tool ships the WEIGHT-side emulation (the main lever: gfx950's absorb uses fp4
  weights). The ACTIVATION side (a4w4 = also fp4 act, a8w4 = fp8 act) needs to wrap the
  absorb BMM inside forward_absorb_*; see HANDOVER_gfx1250_absorb_quant.md for the
  skeleton (the exact wrap point can shift by commit, so wire it against the live
  forward_mla.py). mxfp4_qdq / mxfp8_qdq below are provided for that.

Run on gfx1250 (eager or cuda-graph both fine; this only changes weights at load):
  PYTHONPATH=/path:$PYTHONPATH SGLANG_ABSORB_QUANT_EMUL=w_fp4 bash run_ds-r1.sh
  then GSM8K as usual. Look for [absorb_emul] in the log. Compare vs baseline (unset).
"""
import os

_MODE = os.environ.get("SGLANG_ABSORB_QUANT_EMUL", "off")

if _MODE not in ("off", "", None):
    import threading, time

    def mxfp4_qdq(w):
        """MXFP4 (e2m1 + per-32 e8m0 scale) quantize->dequantize; returns same dtype."""
        import torch
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            device=w.device, dtype=torch.float32)
        od = w.dtype
        x = w.float()
        shp = x.shape
        K = shp[-1]
        assert K % 32 == 0, f"absorb weight last dim {K} not %32"
        x = x.reshape(-1, K // 32, 32)
        absmax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(torch.floor(torch.log2((absmax / 6.0).clamp(min=1e-30))))
        scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
        xs = x / scale
        sign = torch.sign(xs)
        idx = (xs.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
        q = grid[idx] * sign
        deq = (q * scale).reshape(shp)
        deq = torch.where(torch.isfinite(deq), deq, torch.zeros_like(deq))
        return deq.to(od)

    def mxfp8_qdq(x):
        """MXFP8 (e4m3 + per-32 e8m0 scale) quantize->dequantize; returns same dtype."""
        import torch
        od = x.dtype
        f = x.float()
        shp = f.shape
        K = shp[-1]
        assert K % 32 == 0
        f = f.reshape(-1, K // 32, 32)
        absmax = f.abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(torch.floor(torch.log2((absmax / 448.0).clamp(min=1e-30))))
        scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
        q = (f / scale).to(torch.float8_e4m3fn).float()
        return (q * scale).reshape(shp).to(od)

    def _install():
        import torch
        for _ in range(600):
            try:
                from sglang.srt.models import deepseek_v2 as dsv2
                break
            except Exception:
                time.sleep(0.3)
        else:
            return
        AttnCls = getattr(dsv2, "DeepseekV2AttentionMLA", None) or getattr(
            dsv2, "DeepseekV2Attention", None)
        if AttnCls is None or not hasattr(AttnCls, "forward"):
            print("[absorb_emul] could not find attention class", flush=True)
            return

        _orig = AttnCls.forward
        done = set()

        def _qdq_weight(t):
            # t: (num_heads, A, B). MXFP4 q-dq along the last dim; if last dim not %32,
            # try the second-to-last by transposing.
            if t is None:
                return t
            try:
                if t.shape[-1] % 32 == 0:
                    return mxfp4_qdq(t)
                if t.shape[-2] % 32 == 0:
                    return mxfp4_qdq(t.transpose(-1, -2)).transpose(-1, -2).contiguous()
            except Exception as e:
                print("[absorb_emul] qdq skip:", repr(e), flush=True)
            return t

        def hooked(self, *args, **kwargs):
            if id(self) not in done:
                done.add(id(self))
                try:
                    if _MODE in ("w_fp4", "a4w4", "a8w4"):
                        for nm in ("w_kc", "w_vc"):
                            w = getattr(self, nm, None)
                            if isinstance(w, torch.Tensor) and w.dtype in (
                                    torch.bfloat16, torch.float16, torch.float32):
                                setattr(self, nm, _qdq_weight(w))
                        lid = getattr(self, "layer_id", "?")
                        print(f"[absorb_emul] layer {lid}: w_kc/w_vc -> MXFP4 q-dq "
                              f"(mode={_MODE})", flush=True)
                except Exception as e:
                    print("[absorb_emul] weight patch err:", repr(e), flush=True)
            return _orig(self, *args, **kwargs)

        AttnCls.forward = hooked
        print(f"[absorb_emul] installed (mode={_MODE}); ACTIVATION-side quant is NOT wired "
              "here — see HANDOVER_gfx1250_absorb_quant.md", flush=True)

    threading.Thread(target=_install, daemon=True).start()
