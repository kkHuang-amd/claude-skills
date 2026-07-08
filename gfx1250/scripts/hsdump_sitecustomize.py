"""Cross-node per-layer residual-stream dump (post-attention split version).

Env-gated (HS_DUMP=1). Records, for exactly the FIRST prefill pass, per decoder layer:
  - input       : residual-stream hidden entering the layer
  - post_attn   : the attention-sublayer output (o_proj result, BEFORE MoE)  <-- key signal
  - post_layer  : the full layer output residual (attn + MoE)                <-- MoE control
each as {norm, slice64}. Dumps JSON to $HS_DUMP_OUT and stops.

WHY post_attn matters (see HANDOVER_crossnode_dump.md):
  gfx950 (a4w4) vs gfx1250 (a8w4) MoE differs BY CONSTRUCTION, so post_layer /
  MoE-sublayer deltas are expected to diverge cross-node — that is the control, not a
  bug. The attention sublayer is the SAME bf16 code on both nodes, so post_attn is the
  clean "non-MoE" quantity. Also: layers 0..first_k_dense_replace-1 are DENSE (no MoE),
  so their residual should match cross-node up to the TP/bf16 floor (~1e-2). Divergence
  in the dense layers or in post_attn beyond that floor = a real non-MoE difference.

Launch (EAGER + no radix + skip warmup):
  PYTHONPATH=/path/to/scripts:$PYTHONPATH HS_DUMP=1 HS_DUMP_OUT=/tmp/hs_dump_<node>.json \
    bash run_ds-r1.sh   # with --disable-cuda-graph --disable-radix-cache --skip-server-warmup
then send the ONE fixed prompt (see HANDOVER doc), greedy, max_new_tokens=1.
"""
import os

if os.environ.get("HS_DUMP", "0") in ("1", "true", "True"):
    import threading, time, json

    OUT = os.environ.get("HS_DUMP_OUT", "/tmp/hs_dump.json")
    N = int(os.environ.get("HS_DUMP_NLAYERS", "61"))
    STATE = {"done": False}

    def _install():
        import torch
        for _ in range(600):
            try:
                from sglang.srt.models import deepseek_v2 as dsv2
                Layer = dsv2.DeepseekV2DecoderLayer
                break
            except Exception:
                time.sleep(0.5)
        else:
            return

        AttnCls = None
        for name in ("DeepseekV2AttentionMLA", "DeepseekV2Attention"):
            AttnCls = getattr(dsv2, name, None)
            if AttnCls is not None:
                break

        rec_layers = []   # per-layer dicts (aligned by prefill order)
        attn_outs = []    # per-layer post-attn stats (aligned by prefill order)

        def slice_stats(t):
            f = t.detach().float().reshape(-1, t.shape[-1])
            last = f[-1]
            return {"norm": float(torch.norm(last).item()),
                    "slice64": [round(x, 5) for x in last[:64].tolist()]}

        # ---- attention-module hook: capture post-attention (pre-MoE) output ----
        if AttnCls is not None and hasattr(AttnCls, "forward"):
            _a_orig = AttnCls.forward

            def a_hooked(self, *args, **kwargs):
                out = _a_orig(self, *args, **kwargs)
                if not STATE["done"] and len(attn_outs) < N:
                    try:
                        t = out[0] if isinstance(out, (tuple, list)) else out
                        attn_outs.append(slice_stats(t))
                    except Exception:
                        attn_outs.append(None)
                return out
            AttnCls.forward = a_hooked

        # ---- decoder-layer hook: input + post_layer, merge post_attn ----
        _l_orig = Layer.forward

        def l_hooked(self, positions, hidden_states, forward_batch, residual, *a, **k):
            pre = None
            try:
                pre = slice_stats(hidden_states)
            except Exception:
                pass
            out = _l_orig(self, positions, hidden_states, forward_batch, residual, *a, **k)
            if not STATE["done"] and len(rec_layers) < N:
                try:
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    res = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                    post = hs if res is None else (hs + res)
                    lid = int(getattr(self, "layer_id", len(rec_layers)))
                    post_attn = attn_outs[len(rec_layers)] if len(attn_outs) > len(rec_layers) else (
                        attn_outs[-1] if attn_outs else None)
                    rec_layers.append({"layer": lid, "input": pre,
                                       "post_attn": post_attn,
                                       "post_layer": slice_stats(post)})
                    if len(rec_layers) >= N:
                        json.dump({"layers": rec_layers}, open(OUT, "w"), indent=2)
                        print(f"[hs_dump] wrote {OUT} layers={len(rec_layers)} "
                              f"(post_attn captured={sum(1 for r in rec_layers if r['post_attn'])})",
                              flush=True)
                        STATE["done"] = True
                except Exception as e:
                    print("[hs_dump] layer err:", repr(e), flush=True)
            return out

        Layer.forward = l_hooked
        print(f"[hs_dump] installed (attn_hook={'yes' if AttnCls else 'NO'}) N={N}", flush=True)

    threading.Thread(target=_install, daemon=True).start()
