# Cross-node hidden-state dump — gfx950 (reference) vs gfx1250 (a8w4)

## Why
`DeepSeek-R1-0528-MXFP4` GSM8K: **gfx950 (native a4w4) = 0.93 (stably reproducible)**
vs **gfx1250 (forced a8w4) = ~0.81-0.85**. On gfx1250 every component is numerically
clean in isolation (EXPERIMENT_LOG E20/E22: MoE a8w4 is *more* accurate than a4w4;
triton MLA decode + MHA prefill match a torch fp32 ref to ~0.16%; bf16 GEMMs are
literally torch). The gap is real and reproducible. The only way left to localize it is
to **diff the two nodes' per-layer residual stream on the exact same fixed prompt**.

---

## ⚠️ CRITICAL (2026-07-09): BOTH NODES MUST RUN THE SAME sglang COMMIT
`000a61a2662ba8a951a460b82c2dbb607febafce`

The dump hook monkeypatches `DeepseekV2DecoderLayer.forward` and reconstructs the
residual stream as `post = hidden_states + residual` from the layer's return. On
sglang `000a61a2` the layer returns a **3-tuple `(hidden_states, residual, topk_indices)`**
and manages the residual via `layer_communicator` (prepare_attn / prepare_mlp /
postprocess_layer). Against that convention the hook's `post_layer` reconstruction is
**not physically correct** — its absolute norm "explodes" (e.g. ~81000 at layer 60 vs a
healthy ~350). **This is a hook artifact, not a model bug** (verified: eager GSM8K on
gfx1250 @ 000a61a2 = 0.833, coherent generation).

Consequence for the diff:
- The **absolute** `post_layer` norms are meaningless, BUT
- if **both nodes run the exact same commit + same hook**, the reconstruction is done
  **identically on both sides**, so the **cross-node `rel_l2` comparison is still valid
  (apples-to-apples)**. The artifact cancels.
- Running the two nodes on **different** commits does NOT cancel and produces a garbage
  diff. This already bit us once:

### What went wrong with the OLD `hs_dump_gfx950.json` (do not reuse it)
The previous `hs_dump_gfx950.json` was produced on the **old `7aa6082` tree**
(`/sgl-workspace/sglang_gfx-1250`), where the layer returned a 2-tuple and the hook's
reconstruction happened to be correct (gentle norms 2.4 -> 358). Diffing it against the
gfx1250 `000a61a2` dump is invalid — proven two ways:
1. **layer-0 `input` mismatch** (rel_l2 = 1.22): the embedding output differs, i.e. the
   two runs tokenized the SAME prompt into DIFFERENT token ids (different tokenizer
   version — `000a61a2` applies "v5 tokenizer component mismatch" fixes + restores
   `add_bos_token=True`).
2. **gfx950 `post_layer` norms are gentle** (hook worked = old tree) while gfx1250's
   explode (hook artifact = 000a61a2). Same commit would make BOTH explode identically.

**=> The gfx950 side MUST be re-dumped on `000a61a2`. The environment being on 000a61a2
is not enough; the JSON file itself must be regenerated there.**

---

## STATUS of the two dumps
- **gfx1250 side: DONE.** `hs_dump_gfx1250.json` in this dir was re-dumped on
  `000a61a2` (2026-07-09): 61 layers, `post_attn` captured for all 61, rank-0 single
  file, from the fixed "Natalia" prompt (greedy 1 tok -> `' Natal'`). Working recipe
  (`AITER_FORCE_A8W4=1`, `SGLANG_MOE_SHUFFLE_GFX1250=1`, `AITER_GROUPED_FORCE_SPLIT_K1=1`,
  `--kv-cache-dtype auto`, triton attn), eager + no-radix + skip-warmup.
- **gfx950 side: TODO on the gfx950 node — this is what this doc is for.**

---

## What is a clean signal vs a confound (localization logic)
The two nodes differ **by construction** only in **MoE** (gfx950 a4w4 vs gfx1250 a8w4):
- **`post_layer` deltas at MoE layers WILL diverge cross-node — that is EXPECTED** (the
  control), and it **propagates**: once layer 3 (first MoE) differs, every later layer's
  input differs too. Not a bug.
- The **clean signals** are:
  1. **Dense layers 0-2** (`first_k_dense_replace = 3`, no MoE): their `input`,
     `post_attn`, `post_layer` should match cross-node to the **TP/bf16 floor (~1e-2)**.
     A jump here = a real **non-MoE** gfx1250 difference (proj / rope / rmsnorm / attn /
     dense-MLP).
  2. **`post_attn`** (attention-sublayer output, captured BEFORE MoE) at every layer —
     same bf16 code on both nodes. Its *input* drifts once MoE divergence starts, but a
     **sudden `post_attn` jump relative to its own `input`** localizes an attention-side
     difference.
- TP2 (gfx950) vs TP1 (gfx1250): the residual stream is replicated (all-reduced), so
  it's comparable; only a ~1e-3 reduction-order floor that compounds to ~1e-2 by late
  layers. gfx1250 currently cannot run TP2, so this floor is unavoidable — look for
  divergence clearly ABOVE it.

**Smoking gun** = the earliest layer/sublayer whose divergence exceeds the TP floor and
is NOT explained by the expected MoE delta. If divergence is *confined* to `post_layer`
and only starts at layer 3, the gap is the **a8w4-vs-a4w4 scheme** (not a fixable kernel
bug) and we stop chasing kernels.

---

## STEP-BY-STEP: re-dump gfx950 on 000a61a2

### 1. Put the gfx950 sglang tree on the exact commit
```bash
cd <your gfx950 sglang tree>          # e.g. /sgl-workspace/sglang_gfx-1250
git fetch --all
git checkout 000a61a2662ba8a951a460b82c2dbb607febafce
git rev-parse HEAD                     # MUST print 000a61a2662ba8a951a460b82c2dbb607febafce
```
Notes:
- gfx950 does **NOT** need `SGLANG_MOE_SHUFFLE_GFX1250` (it runs native a4w4; that flag is
  a gfx1250-only workaround). Do not set it.
- Keep whatever aiter you normally use on gfx950 (stock; the gfx1250 flydsl fixes are
  arch-gated and irrelevant here).

### 2. Drop in the dump hook (no repo edits)
Copy the canonical hook from this skill dir to a fresh dir **as `sitecustomize.py`**:
```bash
mkdir -p /tmp/hsdump
cp <skill dir>/scripts/hsdump_sitecustomize.py /tmp/hsdump/sitecustomize.py
```
(The file must be named `sitecustomize.py` so Python auto-imports it via `PYTHONPATH`.)

### 3. Launch with the hook + the SAME recipe that gives 0.93, in EAGER dump mode
Add to your gfx950 launch env:
```
PYTHONPATH=/tmp/hsdump:$PYTHONPATH
HS_DUMP=1
HS_DUMP_OUT=/tmp/hs_dump_gfx950.json
```
Add to the launch args (dump requires these):
```
--disable-cuda-graph      # EAGER — the Python hook does NOT run under cuda-graph replay
--disable-radix-cache     # fixed prompt always cold-prefills; no prefix reuse
--skip-server-warmup      # else the warmup prefill captures the wrong prompt
```
Keep everything else identical to your 0.93 run (TP2, `HIP_VISIBLE_DEVICES`,
`--kv-cache-dtype auto`, `--attention-backend triton`, `AITER_FORCE_A8W4=1`,
`--model-path <gfx950 R1-0528-MXFP4>`, etc.).

On startup you should see `[hs_dump] installed (attn_hook=yes) N=61`.

### 4. Send the EXACT fixed prompt — byte-identical, greedy, 1 token
```bash
curl -s http://localhost:PORT/generate -H 'Content-Type: application/json' -d '{
  "text": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nAnswer:",
  "sampling_params": {"temperature": 0, "max_new_tokens": 1}
}'
```
The hook fires on this first prefill and prints, e.g.:
```
[hs_dump] rank=0 wrote /tmp/hs_dump_gfx950.json layers=61 (post_attn captured=61)
```
(Under TP2 only rank 0 writes `/tmp/hs_dump_gfx950.json`; rank 1 writes a harmless
`.rank1` copy. Use the rank-0 file.)

### 5. Hand back the file
Copy the new `/tmp/hs_dump_gfx950.json` into this skill dir, overwriting the stale one:
```bash
cp /tmp/hs_dump_gfx950.json <skill dir>/hs_dump_gfx950.json
```

---

## After both dumps exist (run on the gfx1250 side)
```bash
cd <skill dir>
python3 scripts/plot_crossnode_dump.py     # writes crossnode_dump_compare.png + prints a table
```
The script already compares `input`, `post_attn`, and `post_layer` per layer (rel_l2 of
the last-token slice64, plus norms).

### Sanity checks that the diff is now VALID (must pass, else the dumps are mismatched)
- **layer-0 `input` rel_l2 ≈ 0** (identical embeddings = identical tokenization). If it's
  ~1, the prompt/tokenizer still differs — the dumps are NOT comparable, fix that first.
- **dense layers 0-2** `input` / `post_attn` / `post_layer` rel_l2 all **≤ ~1e-2**.
- both nodes' `post_layer` norms should now show the **same** (artifact) trajectory shape
  (both "explode" on 000a61a2) — confirming same commit + same hook.

Only once those pass do the MoE-layer (3-60) `post_attn` / `post_layer` divergences mean
anything. The earliest clean-signal jump above the floor is the localizer.

---

## Gotchas
- Both nodes: **000a61a2**, eager, greedy, `--disable-radix-cache`, `--skip-server-warmup`,
  same prompt.
- Dump from rank 0 (residual stream is replicated across TP ranks).
- `slice64` + norms keep the JSON tiny/portable (no big tensors).
- `first_k_dense_replace = 3` for R1-0528 (dense layers 0-2 = primary clean zone),
  `num_hidden_layers = 61`, `n_routed_experts = 256`. Verify in the checkpoint config.json.
- The hook records exactly the first prefill (61 layers). Validated on gfx1250 @ 000a61a2
  (61 layers, post_attn all captured).
- The `post_layer` absolute norms are a known hook artifact on 000a61a2 (see the CRITICAL
  section) — do not chase them; only the cross-node rel_l2 (same commit both sides) matters.
