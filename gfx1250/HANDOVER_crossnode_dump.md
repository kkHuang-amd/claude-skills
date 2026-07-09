# Cross-node hidden-state dump — gfx950 (reference) vs gfx1250 (a8w4)

> **MULTI-NODE WRITE WORKFLOW (read first).** Several machines share this dir and pushing to the
> same shared files (`EXPERIMENT_LOG.md`, `STATUS.md`) already caused git conflicts. So:
> **write your raw runs to your node's own file first** — `results_gfx950.md` (gfx950 ref),
> `results_gfx1250-j19-10.md`, `results_gfx1250-H21-18.md` — then promote only a short, node-tagged
> summary into `EXPERIMENT_LOG.md` / `STATUS.md`. Tag every entry with your node
> (e.g. `[node: gfx950]`). `git pull --rebase` before you push. This keeps concurrent pushes from
> clobbering each other.

## Why
`DeepSeek-R1-0528-MXFP4` GSM8K: **gfx950 (native a4w4) = 0.93 (stably reproducible)**
vs **gfx1250 (forced a8w4) = ~0.81-0.85**. On gfx1250 every component is numerically
clean in isolation (EXPERIMENT_LOG E20/E22: MoE a8w4 is *more* accurate than a4w4;
triton MLA decode + MHA prefill match a torch fp32 ref to ~0.16%; bf16 GEMMs are
literally torch). The gap is real and reproducible. The only way left to localize it is
to **diff the two nodes' per-layer residual stream on the exact same fixed prompt**.

---

## ⚠️ CRITICAL (2026-07-09): capture the REAL prefill, not the M:1 readiness forward

`--skip-server-warmup` does **NOT** stop sglang from running a spurious **1-token
(M:1) forward** right after `"fired up"` (a readiness/health path). The original hook
captured the **FIRST** forward through the decoder layers and stopped — so it grabbed
that **M:1 dummy instead of the real fixed prompt**. A 1-token degenerate pass produces
garbage per-layer stats:
- fake "residual explosion" (post_layer norm ~81000 at L60 vs a healthy ~350),
- layer-0 `input` that does NOT match the other node (looks like a tokenizer mismatch
  but is really just a different last token).

**Root cause = capturing the wrong forward. Neither a commit/hook incompatibility nor a
tokenizer-version mismatch was ever real.** Verified: once the correct ~41-token
"Natalia" prefill is captured, gfx1250 @ 000a61a2 layer-0 `input` matches gfx950
**exactly (rel_l2 = 0.0)** and post_layer norms are healthy (2.37 -> 319, tracking
gfx950's 2.38 -> 358). The two nodes do NOT need to be on the same commit — the
tokenizer output is identical across `7aa6082` and `000a61a2` (proven by rel_l2=0 at L0).

### THE FIX (already applied to `scripts/hsdump_sitecustomize.py`)
The hook now only records a forward whose token count is `>= HS_DUMP_MIN_TOKENS`
(default **8**, well under the ~41-token prompt), so the M:1 readiness forward is
skipped and the real prefill is captured. **Always copy the current
`scripts/hsdump_sitecustomize.py` fresh** (do not reuse an older sitecustomize.py).
Confirm from the server log that the aiter GEMM shapes right before
`[hs_dump] ... wrote` are `M:41` (the prompt), NOT `M:1`.

---

## STATUS of the two dumps
- **gfx1250 side: DONE & VERIFIED.** `hs_dump_gfx1250.json` in this dir, re-dumped on
  `000a61a2` (2026-07-09) with the fixed hook: 61 layers, `post_attn` all captured,
  rank-0, correct M:41 "Natalia" prefill (`' Natal'`), L0 matches gfx950 (rel_l2 0.0).
  Working recipe (`AITER_FORCE_A8W4=1`, `SGLANG_MOE_SHUFFLE_GFX1250=1`,
  `AITER_GROUPED_FORCE_SPLIT_K1=1`, `--kv-cache-dtype auto`, triton attn), eager +
  no-radix + skip-warmup.
- **gfx950 side: the EXISTING `hs_dump_gfx950.json` is VALID** (it captured Natalia
  correctly on 7aa6082; L0 matches, norms healthy). A re-dump is **optional** (only for
  extra confidence / matched commit) — if you do re-dump, use the fixed hook and the
  steps below and confirm the `M:41` shape.

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

### Sanity checks that the diff is VALID (must pass, else a dump captured the wrong forward)
- **layer-0 `input` rel_l2 ≈ 0** (identical embeddings). If it's ~1, that dump captured
  the M:1 readiness forward, not the prompt — re-dump with the fixed hook.
- **dense layers 0-2** `input` and `post_layer` rel_l2 **≤ ~1e-2** (TP/bf16 floor).
- **post_layer norms grow gently and track between nodes** (both ~2.4 -> ~320-360). A
  norm blowup (~10^4) = the M:1 artifact = wrong forward captured.
- NOTE on `post_attn`: under **TP2 (gfx950) vs TP1 (gfx1250)** the raw attention-sublayer
  output is captured at a point that may be **pre-all-reduce** on the TP2 node (rank 0
  sees only part of the heads), so `post_attn` rel_l2 reads high (~0.4-1.0) even at dense
  layers where `input`/`post_layer` match to the floor. Treat `post_attn` as unreliable
  in a TP2-vs-TP1 comparison; rely on `input` and `post_layer` (both post-all-reduce /
  replicated). `post_attn` becomes clean only if both nodes run the same TP.

Once those pass, the MoE-layer (3-60) `post_layer` divergence is the control (a4w4 vs
a8w4, expected). The earliest clean-signal (`input`/`post_layer` at a dense layer, or the
residual ENTERING the first MoE layer) jump above the floor is the localizer.

---

## STEP 2 (E33 localizer): per-layer dump UNDER identical bf16 MoE emul
Now that the gap is proven non-MoE (E33), the localizer is a cross-node per-layer dump where the
**MoE is made byte-identical** on both nodes via the bf16 emul (`SGLANG_MOE_EMUL=a8w4`). Then MoE no
longer diverges by construction, and the earliest above-floor rel_l2 localizes the **non-MoE** op.

**gfx1250 side: DONE** — `hs_dump_gfx1250_emul_a8w4.json` in this dir (Natalia prompt, L0 rel_l2
matches real dump, 61 layers, post_attn all). Produced with `run_ds-r1_emuldump.sh` (combines
`moe_emul_sitecustomize.py` + `hsdump_sitecustomize.py` via a single `/tmp/step2/sitecustomize.py`
that exec_module's both; `SGLANG_MOE_EMUL=a8w4`, eager, no-radix, skip-warmup, `--mem-fraction-static
0.83` [weights need >=0.821; the emul dequant scratch is OUT-of-pool HIP mem so keep mem-fraction just
above the weight minimum], `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`).

**gfx950 side: TODO** — reproduce with the SAME combo hook + `SGLANG_MOE_EMUL=a8w4`:
1. `/tmp/step2/sitecustomize.py` that exec_module's BOTH `moe_emul_sitecustomize.py` and
   `hsdump_sitecustomize.py` from this skill's `scripts/` (see this node's copy for the 15-line file).
2. env: `PYTHONPATH=/tmp/step2:$PYTHONPATH SGLANG_MOE_EMUL=a8w4 HS_DUMP=1
   HS_DUMP_OUT=/tmp/hs_dump_gfx950_emul_a8w4.json` + your gfx950 recipe (TP2 etc); launch eager +
   `--disable-radix-cache --skip-server-warmup`. (mem-fraction: emul needs out-of-pool scratch; on
   gfx950 8xMI355X TP2 the earlier E30 emul ran at 0.70 — keep whatever let the emul run there.)
3. Send the SAME Natalia prompt (greedy, 1 tok). Confirm `[moe_emul] MODE=a8w4` + `[hs_dump] wrote
   ... layers=61` and that the dump L0 `input` norm matches (~2.287 → same tokenization).
4. Hand back `hs_dump_gfx950_emul_a8w4.json` to this dir.

Then on this node:
```bash
python3 scripts/plot_crossnode_dump.py --g950 hs_dump_gfx950_emul_a8w4.json \
    --g1250 hs_dump_gfx1250_emul_a8w4.json --out crossnode_emul_compare.png
```
Interpretation: MoE is identical now, so `post_layer` no longer jumps at layer 3. Any layer whose
`input`/`post_attn`/`post_layer` rel_l2 climbs **above the ~1e-2 TP floor** = the non-MoE platform op
that carries the 0.95-vs-0.80 gap. (Same TP2-vs-TP1 floor caveat; a 15%-accuracy bias should exceed it.)

## Gotchas
- Both nodes: eager, greedy, `--disable-radix-cache`, `--skip-server-warmup`, same prompt.
  (Same commit is NOT required — tokenizer output matches across 7aa6082/000a61a2.)
- **Must capture the real prefill, not the M:1 readiness forward** — use the fixed hook
  (`HS_DUMP_MIN_TOKENS`) and confirm the `M:41` GEMM shape precedes `[hs_dump] wrote`.
- Send the prompt directly; avoid calling `/health` first (it can trigger extra forwards).
- Dump from rank 0 (residual stream is replicated across TP ranks).
- `slice64` + norms keep the JSON tiny/portable (no big tensors).
- `first_k_dense_replace = 3` for R1-0528 (dense layers 0-2 = primary clean zone),
  `num_hidden_layers = 61`, `n_routed_experts = 256`. Verify in the checkpoint config.json.
- gfx1250 weight md5 (`model.safetensors.index.json`) = `cbc58fe6f6a7c603a290af2361798c8f`
  — compare on the other node to confirm the same checkpoint.
