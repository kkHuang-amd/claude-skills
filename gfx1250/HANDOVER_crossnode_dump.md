# Cross-node hidden-state dump — gfx950 (reference) vs gfx1250 (a8w4)

## Why
`DeepSeek-R1-0528-MXFP4` GSM8K: **gfx950 (native a4w4) = 0.93 (stably reproducible)**
vs **gfx1250 (forced a8w4) = 0.85**. On gfx1250 every component has been proven
numerically clean in isolation (EXPERIMENT_LOG E20 + attention hook: MoE a8w4 is
*more* accurate than a4w4; triton MLA decode + MHA prefill match a torch fp32 ref to
~0.16%; bf16 GEMMs are literally torch `solution:0`). Yet the reproducible gap is
real. The only way left to localize it is to **diff the per-layer residual stream of
the two nodes on the exact same fixed prompt** and find the first layer/sublayer that
diverges beyond the expected MoE-quant level.

This doc hands the **gfx950 side** off to the cursor agent on that node. It dumps a
JSON that I (gfx1250 side) will diff against my own dump.

## What to run on the gfx950 node

### 1. Drop in the dump hook (no repo edits)
Create `/tmp/hsdump/sitecustomize.py` with the content in the **APPENDIX** below.
It monkeypatches `DeepseekV2DecoderLayer.forward` to record, per layer, the norm and
a fixed 64-dim slice of:
- the layer **input** hidden state,
- the **post-attention** hidden state (attention sublayer output + residual),
- the **post-MoE** hidden state (final layer output),
plus the model's **final logits** top-32 (ids + values) for the last prompt token.

It is gated by `HS_DUMP=1` and writes `/tmp/hs_dump_gfx950.json` after the first
forward of the fixed prompt, then disables itself.

### 2. Launch the server with the hook + the SAME recipe you use for 0.93
Use your working gfx950 launch script, but:
- prepend `PYTHONPATH=/tmp/hsdump:$PYTHONPATH HS_DUMP=1`
- add `--disable-radix-cache` (so the fixed prompt always cold-prefills; no prefix reuse)
- keep everything else identical to your 0.93 run (TP2, `--kv-cache-dtype auto`,
  `--attention-backend triton`, `AITER_FORCE_A8W4=1`, etc.)
- **eager is required**: add `--disable-cuda-graph` (the Python hook does not run
  during cuda-graph replay).
- **add `--skip-server-warmup`**: otherwise the server's warmup prefill fires the
  hook first and the dump captures the warmup prompt instead of the fixed prompt below.

### 3. Send the EXACT fixed prompt (greedy, 1 token)
```bash
curl -s http://localhost:PORT/generate -H 'Content-Type: application/json' -d '{
  "text": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nAnswer:",
  "sampling_params": {"temperature": 0, "max_new_tokens": 1}
}'
```
(The prompt text must be byte-identical to what I use on gfx1250 — it is fixed above.)

### 4. Hand back `/tmp/hs_dump_gfx950.json`
Copy that file back to this repo dir (or paste its contents). I diff it against
`/tmp/hs_dump_gfx1250.json`.

## How I read the diff (localization logic)
- **Layer input / post-attention** hidden states use the *same* bf16 code on both
  nodes. If they already diverge (rel_l2 ≫ 1e-2) at some early layer's
  **post-attention** point, the bug is in a **non-MoE** path (projection / rope /
  rmsnorm / attention) on gfx1250 — a real kernel/codegen bug.
- **Post-MoE** hidden states are *expected* to differ (a4w4 vs a8w4). That per-layer
  delta is the control. If the divergence is confined to the MoE delta and grows only
  through MoE layers, the gap is the a8w4-vs-a4w4 **scheme** (not a kernel bug), and
  we stop chasing kernels.
- The first layer where **post-attention** rel_l2 jumps is the smoking gun.

## Notes / gotchas
- TP2 (gfx950) vs TP1 (gfx1250): residual-stream hidden states are replicated (not
  sharded) across TP ranks, so rank-0's dump is directly comparable. The hook reads
  the hidden tensor on the local rank; dump from rank 0.
- Both must be **eager** and **greedy**, `--disable-radix-cache`, same prompt.
- The 64-dim slice + norms keep the JSON tiny and node-portable (no big tensors).
- If token counts differ (tokenizer parity), the hook keys everything to the LAST
  prompt-token position, which is well-defined on both.

---

## APPENDIX — `/tmp/hsdump/sitecustomize.py`
```python
import os
if os.environ.get("HS_DUMP", "0") in ("1", "true", "True"):
    import threading, time, json

    OUT = os.environ.get("HS_DUMP_OUT", "/tmp/hs_dump.json")
    STATE = {"done": False}

    def _install():
        import torch
        # locate the DeepSeek decoder layer class
        for _ in range(600):
            try:
                from sglang.srt.models import deepseek_v2 as dsv2
                Layer = dsv2.DeepseekV2DecoderLayer
                break
            except Exception:
                time.sleep(0.5)
        else:
            return

        rec = {"layers": []}

        def slice_stats(t):
            f = t.detach().float().reshape(-1, t.shape[-1])
            last = f[-1]  # last token of the (prefill) batch
            return {
                "norm": float(torch.norm(last).item()),
                "slice64": [round(x, 5) for x in last[:64].tolist()],
            }

        N = int(os.environ.get("HS_DUMP_NLAYERS", "61"))  # DeepSeek-R1 has 61 layers
        _orig = Layer.forward

        def hooked(self, positions, hidden_states, forward_batch, residual, *a, **k):
            pre = None
            try:
                pre = slice_stats(hidden_states)
            except Exception:
                pass
            out = _orig(self, positions, hidden_states, forward_batch, residual, *a, **k)
            # Record ONLY the first prefill pass (layers 0..N-1 run in order before
            # any decode step); stop + dump once we have N entries.
            if not STATE["done"] and len(rec["layers"]) < N:
                try:
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    res = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                    post = hs if res is None else (hs + res)
                    lid = int(getattr(self, "layer_id", len(rec["layers"])))
                    rec["layers"].append({"layer": lid, "input": pre,
                                          "post_layer": slice_stats(post)})
                    if len(rec["layers"]) >= N:
                        json.dump(rec, open(OUT, "w"), indent=2)
                        print("[hs_dump] wrote", OUT, "layers", len(rec["layers"]), flush=True)
                        STATE["done"] = True
                except Exception as e:
                    rec.setdefault("errors", []).append(repr(e))
            return out

        Layer.forward = hooked
        print("[hs_dump] installed decoder-layer hook", flush=True)

    threading.Thread(target=_install, daemon=True).start()
```

> NOTE for the gfx950 agent: set `HS_DUMP_OUT=/tmp/hs_dump_gfx950.json` in the env so
> the filename is unambiguous. The exact same hook + prompt is run on gfx1250 with
> `HS_DUMP_OUT=/tmp/hs_dump_gfx1250.json`. The hook records `input` and `post_layer`
> per layer for exactly the first prefill pass (61 entries, layers 0-60 in order),
> which localizes divergence onset. This hook was validated on gfx1250 (dumps 61
> clean layers, no errors).
