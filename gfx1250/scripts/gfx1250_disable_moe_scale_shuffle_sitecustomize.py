"""E25b confirmation switch: disable the gfx1250 MoE B-scale n32k4 shuffle.

The current quark_w4a4_mxfp4_moe.py `process_weights_after_loading` shuffles the MoE
B-scale to n32k4 UNCONDITIONALLY for gfx1250 (`if _is_gfx1250: moe_shuffle_scale(...)`),
with no env gate. This hook makes that a no-op (identity), so the B-scale stays RAW.

Purpose: test the "0.000 = weight/scale MISMATCH (not 'no shuffle')" theory (E25b).
Combine with `SGLANG_MOE_SHUFFLE_GFX1250=0` to get BOTH weight and scale RAW (internally
CONSISTENT), which should reproduce the earlier ~0.82 if the old code was both-raw:

  both RAW (consistent):     SGLANG_DISABLE_MOE_SCALE_SHUFFLE_GFX1250=1 SGLANG_MOE_SHUFFLE_GFX1250=0
    -> expect ~0.82 (confirms mismatch theory, option a) OR still 0.000 (=> the kernel
       requires shuffled B/B-scale, so the old 0.82 must have shuffled BOTH; option b).
  both SHUFFLED (the fix):   (default; SGLANG_MOE_SHUFFLE_GFX1250=1, this env unset) -> 0.85
  MISMATCH (weight raw only): SGLANG_MOE_SHUFFLE_GFX1250=0 (this env unset)          -> 0.000

Usage (gfx1250):
  PYTHONPATH=/path:$PYTHONPATH SGLANG_DISABLE_MOE_SCALE_SHUFFLE_GFX1250=1 \
    SGLANG_MOE_SHUFFLE_GFX1250=0 bash run_ds-r1.sh
  then GSM8K. Look for [no_scale_shuffle] in the log.
"""
import os

if os.environ.get("SGLANG_DISABLE_MOE_SCALE_SHUFFLE_GFX1250", "0") in ("1", "true", "True"):
    import threading, time

    def _install():
        for _ in range(600):
            try:
                from sglang.srt.layers.quantization.quark.schemes import (
                    quark_w4a4_mxfp4_moe as M,
                )
                break
            except Exception:
                time.sleep(0.3)
        else:
            return

        def _identity(src, *args, **kwargs):
            # return the scale unchanged (raw); mimic "no n32k4 B-scale shuffle"
            return src

        # patch the name bound in the quark MoE scheme module (it does
        # `from aiter.ops.shuffle import moe_shuffle_scale`)
        if hasattr(M, "moe_shuffle_scale"):
            M.moe_shuffle_scale = _identity
            print("[no_scale_shuffle] gfx1250 MoE B-scale shuffle -> identity (raw)",
                  flush=True)
        else:
            print("[no_scale_shuffle] moe_shuffle_scale not found in scheme module",
                  flush=True)

    threading.Thread(target=_install, daemon=True).start()
