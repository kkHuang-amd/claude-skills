# Handover: gfx950 apples-to-apples MoE emul (confirm E31 with the CHUNKED emul)

## Why this run
E31 (node H21-18 gfx1250) ran the bf16 MoE emul DIRECTLY on gfx1250 and got:
- a8w4-emul = **0.800**, a16w4-emul (most ideal, zero MoE quant) = **0.825** (40Q)
- i.e. ~= gfx1250's real 0.81, NOT gfx950's 0.925 (E30).
=> making the MoE ideal does NOT close the gap => the gap is **NOT the MoE** (case B).

BUT there is one caveat to kill first: **the gfx1250 emul used a CHUNKED-dequant variant**
(the original E30 batched all-active dequant OOMs at gfx1250 TP1 with the ~353 GB bf16
model). E30's gfx950 0.925 was measured with the ORIGINAL BATCHED emul. To make E31's
conclusion airtight, run the **SAME chunked emul on gfx950** and confirm it still gives
~0.925 (i.e. the chunking is numerically equivalent, not a bug that depresses gfx1250).

Decision:
- **gfx950 chunked-emul ~= 0.925** (== E30 batched) => chunked math is fine => E31 stands:
  gfx1250's 0.80/0.825 is real => the gap is a **gfx1250 non-MoE effect**. Proceed to the
  whole-forward cross-node logit comparison.
- **gfx950 chunked-emul << 0.925** (drops like gfx1250) => the CHUNKED emul has a bug =>
  E31 is invalid; fix the emul and re-run on gfx1250.

## What to run on the gfx950 box (TP2)
The emul hook is now the **chunked** version in this skill dir:
`scripts/moe_emul_sitecustomize.py` (updated 2026-07-09; docstring notes the chunked change).
It is a drop-in replacement for the batched one (same env `SGLANG_MOE_EMUL`, same math).

```bash
mkdir -p /tmp/moeemul
cp <skill>/scripts/moe_emul_sitecustomize.py /tmp/moeemul/sitecustomize.py

# Start from the WORKING gfx950 launch (the modified tree /sgl-workspace/sglang_gfx-1250,
# TP2, GPU2,3, the one that natively serves a4w4 at ~0.93), and ADD:
#   PYTHONPATH=/tmp/moeemul:<sglang>/python:$PYTHONPATH
#   SGLANG_MOE_EMUL=a8w4            # (then also a16w4)
#   --disable-cuda-graph            # REQUIRED: the Python hook can't run under cuda-graph replay
#   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# keep the rest of your 0.93 recipe unchanged (AITER_FORCE_A8W4 etc. are harmless; the MoE
# is replaced by the emul anyway). gfx950 TP2 has more room, so mem-fraction can stay as in
# your normal script (no need for the TP1 0.88).
```
Confirm in the log: `[moe_emul] installed, MODE=a8w4 ...` and `[moe_emul] MODE=a8w4 active=... E=...`.
Sanity `curl /generate` (coherent, not token-0). Then, matching E30/E31 exactly:
```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 40 --parallel 40 --num-shots 5 --port <port>
```

## Runs (each ~10-11 min on gfx950 TP2 per E30 timing)
1. `SGLANG_MOE_EMUL=a8w4`   — expect ~0.925 (== E30 batched). THE apples-to-apples check.
2. `SGLANG_MOE_EMUL=a16w4`  — expect >=0.925 (most ideal). Confirms the a16w4 path too
   (gfx1250 got 0.825 here — this is the direct counterpart).
3. (optional) `SGLANG_MOE_EMUL=a4w4` — should also ~0.925 (E30 batched got 0.925).

## Report back (please TAG with the gfx950 node)
Append to EXPERIMENT_LOG.md a short entry tagged `[node: <gfx950-host> gfx950]` with the
a8w4 and a16w4 chunked-emul 40Q accuracies, and update STATUS.md's E31 banner:
- if gfx950 chunked a8w4/a16w4 ~= 0.925 -> "E31 confirmed: gap is non-MoE (chunked emul equiv)".
- else -> "chunked emul bug; E31 retracted".

## Notes
- The emul REPLACES the MoE with bf16 math on BOTH platforms; only non-MoE (attention/norm/
  rope/linear/sampling) runs natively per-platform. So if gfx950-emul=0.925 and
  gfx1250-emul=0.80-0.825 with the SAME emul, the difference is non-MoE + gfx1250 hardware.
- gfx950 runs a4w4 natively; the emul bypasses that, so this does not need the gfx950 fp4
  absorb / a4w4 kernel — it is pure bf16 MoE math (arch-independent).
- Keep everything else identical to E30/E31: eager, greedy 5-shot, 40Q parallel 40.
