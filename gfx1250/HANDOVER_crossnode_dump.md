# Cross-node hidden-state dump — gfx950 (reference) vs gfx1250 (a8w4)

## Why
`DeepSeek-R1-0528-MXFP4` GSM8K: **gfx950 (native a4w4) = 0.93 (stably reproducible)**
vs **gfx1250 (forced a8w4) = ~0.85**. On gfx1250 every component is numerically clean
in isolation (EXPERIMENT_LOG E20/E22: MoE a8w4 is *more* accurate than a4w4; triton MLA
decode + MHA prefill match a torch fp32 ref to ~0.16%; bf16 GEMMs are literally torch).
The gap is real and reproducible. The only way left to localize it is to **diff the
two nodes' per-layer residual stream on the exact same fixed prompt**.

The canonical dump hook is **`scripts/hsdump_sitecustomize.py`** in this dir. This doc
hands the **gfx950 side** off to the cursor agent on that node; I run the same hook on
gfx1250 and diff.

## IMPORTANT — what is a clean signal vs a confound
The two nodes differ by construction in **MoE** (gfx950 a4w4 vs gfx1250 a8w4), so:
- **`post_layer` / MoE-sublayer deltas WILL diverge cross-node — that is EXPECTED**
  (the control), not a bug. And that divergence **propagates downstream**: once layer 3
  (first MoE layer) differs, every later layer's input differs too.
- Therefore the clean signals are:
  1. **The DENSE layers `0 .. first_k_dense_replace-1`** (R1: layers **0-2**, no MoE).
     Their residual should match cross-node to the **TP/bf16 floor (~1e-2)**. If a dense
     layer's `input`/`post_attn`/`post_layer` diverges beyond that → a real **non-MoE**
     gfx1250 difference (projection / rope / rmsnorm / attention / dense-MLP).
  2. **`post_attn`** (attention-sublayer output, captured BEFORE MoE) at every layer.
     It is the same bf16 code on both nodes. Its *input* is inherited (so it drifts once
     MoE divergence starts), but a **sudden jump in `post_attn` relative to its own
     `input`** localizes an attention-side difference.
- TP2 (gfx950) vs TP1 (gfx1250): residual stream is **replicated** (all-reduced) so it's
  comparable; only caveat is a ~1e-3 reduction-order floor that compounds to ~1e-2 by
  late layers. Interpret everything against this floor. (gfx1250 currently cannot run
  TP2, so this floor is unavoidable; look for divergence clearly ABOVE it.)

## What to run on the gfx950 node
### 1. Drop in the hook (no repo edits)
Copy `scripts/hsdump_sitecustomize.py` (from this dir) to a fresh dir as
`sitecustomize.py`, e.g. `/tmp/hsdump/sitecustomize.py`. It monkeypatches the DeepSeek
decoder layer + attention module and, for the FIRST prefill pass, records per layer:
`input`, `post_attn` (pre-MoE), `post_layer` (post-MoE) as {norm, slice64}. Gated by
`HS_DUMP=1`; writes `$HS_DUMP_OUT` then stops.

### 2. Launch with the hook + the SAME recipe that gives 0.93
Use your working gfx950 launch script, but add:
- env: `PYTHONPATH=/tmp/hsdump:$PYTHONPATH HS_DUMP=1 HS_DUMP_OUT=/tmp/hs_dump_gfx950.json`
- `--disable-radix-cache` (fixed prompt always cold-prefills; no prefix reuse)
- `--disable-cuda-graph` (EAGER required — the Python hook does not run under cuda-graph replay)
- `--skip-server-warmup` (else the warmup prefill captures the wrong prompt)
- keep everything else identical to your 0.93 run (TP2, `--kv-cache-dtype auto`,
  `--attention-backend triton`, `AITER_FORCE_A8W4=1`, etc.)

### 3. Send the EXACT fixed prompt (greedy, 1 token) — byte-identical
```bash
curl -s http://localhost:PORT/generate -H 'Content-Type: application/json' -d '{
  "text": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nAnswer:",
  "sampling_params": {"temperature": 0, "max_new_tokens": 1}
}'
```

### 4. Hand back `/tmp/hs_dump_gfx950.json`
Copy it into this repo dir (next to `hs_dump_gfx1250.json`) or paste its contents.

## How I read the diff (localization logic)
Per layer, rel_l2 of the `slice64` vectors (and norm ratio) between the two dumps:
- **Dense layers 0-2**: expect match ≤ ~1e-2 (TP floor). A jump here = **non-MoE bug**.
- **`post_attn` vs `input`** growth pattern: a localized `post_attn` jump = attention-side.
- **`post_layer` at MoE layers (3-60)**: expected to diverge (a4w4 vs a8w4) — the control.
  If divergence is *confined* to this and only starts at layer 3, the gap is the
  **a8w4-vs-a4w4 scheme** (not a fixable kernel bug), and we stop chasing kernels.
- Smoking gun = the earliest layer/sublayer whose divergence exceeds the TP floor and
  is NOT explained by the expected MoE delta.

## Gotchas
- Both nodes: eager, greedy, `--disable-radix-cache`, `--skip-server-warmup`, same prompt.
- Dump from rank 0 (residual stream is replicated across TP ranks).
- `slice64` + norms keep the JSON tiny/portable (no big tensors).
- If `first_k_dense_replace` differs from 3, note it (config.json) — the dense window
  is the primary clean comparison zone.
- The hook records exactly the first prefill (61 layers). Validated on gfx1250 (dumps
  61 layers, post_attn captured for all).

> Set `HS_DUMP_OUT=/tmp/hs_dump_gfx950.json` on gfx950; I use
> `/tmp/hs_dump_gfx1250.json` on gfx1250. Same hook file (`scripts/hsdump_sitecustomize.py`),
> same prompt, both eager + no-radix + skip-warmup.
