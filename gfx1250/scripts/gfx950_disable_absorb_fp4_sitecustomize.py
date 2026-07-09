"""E25 ablation (run on gfx950): make the MLA absorb BMM run in bf16 (like gfx1250)
instead of gfx950's fp4, WITHOUT touching the MoE (stays a4w4).

gfx950 normally calls quark_post_load_weights() to dynamically quantize the bf16
kv_b_proj weight into mxfp4 w_kc/w_vc, so the absorb BMM (q_nope@w_kc, attn@w_vc) runs
fp4. gfx1250 skips that and runs the absorb in bf16. This hook monkeypatches
quark_post_load_weights to return **bf16** w_kc/w_vc (None scales), so gfx950's absorb
matches gfx1250. Downstream `w_kc.dtype==uint8` branches then fall to the bf16 bmm path.

Purpose: single-node A/B on gfx950. If GSM8K drops 0.93 -> ~0.85 with this ON, the
fp4-absorb (quant-matching) is the gfx1250 gap. If it stays 0.93, absorb isn't it.

Usage (gfx950):
  place as sitecustomize.py on PYTHONPATH, launch the normal gfx950 recipe with
    PYTHONPATH=/path:$PYTHONPATH SGLANG_DISABLE_QUARK_ABSORB_FP4=1
  then run GSM8K as usual. Look for [gfx950_abl] in the log to confirm it patched.
"""
import os

if os.environ.get("SGLANG_DISABLE_QUARK_ABSORB_FP4", "0") in ("1", "true", "True"):
    import threading, time

    def _install():
        import torch
        for _ in range(600):
            try:
                from sglang.srt.layers.quantization.quark import utils as Q
                import sglang.srt.models.deepseek_common.deepseek_weight_loader as W
                break
            except Exception:
                time.sleep(0.3)
        else:
            return

        from sglang.srt.layers.quantization.quark.utils import mxfp4_to_f32, e8m0_to_f32

        def bf16_split(self_attn, w, quant_format):
            # Mimic gfx1250: keep w_kc / w_vc in bf16, no mxfp4 quant, no scales.
            if w.dtype == torch.uint8:
                # static-mxfp4 checkpoint weight -> dequant to bf16 first
                w = mxfp4_to_f32(w, True).to(torch.bfloat16)
                s = self_attn.kv_b_proj.weight_scale.repeat_interleave(32, dim=-1)
                s = e8m0_to_f32(s).to(torch.bfloat16)
                w = w * s
            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            return w_kc, None, w_vc, None

        Q.quark_post_load_weights = bf16_split
        # deepseek_weight_loader did `from ...quark.utils import quark_post_load_weights`
        # (bound name in its own module namespace) — patch that too.
        if hasattr(W, "quark_post_load_weights"):
            W.quark_post_load_weights = bf16_split
        print("[gfx950_abl] quark_post_load_weights -> bf16 absorb (mimic gfx1250)", flush=True)

    threading.Thread(target=_install, daemon=True).start()
