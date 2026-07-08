# STATUS — gfx1250 DeepSeek-R1-0528-MXFP4 accuracy gap investigation

Single entry point / handover snapshot. Last updated **2026-07-09**.
Read this first, then EXPERIMENT_LOG.md (E17-E24) for detail, CHANGES.md for the exact
code edits, HANDOVER_crossnode_dump.md for the next experiment.

## The problem
Serve `DeepSeek-R1-0528-MXFP4` on **gfx1250** (SGLang+aiter, forced **a8w4** MoE).
GSM8K (5-shot completion greedy):
- **gfx1250 (a8w4) = ~0.81-0.85** (1319Q = 0.811)
- **gfx950 (a4w4)  = 0.93** — **stably reproducible** (user-confirmed), same HF
  checkpoint, same sglang model/attn/MoE code (only fp8.py/fp8_kernel.py differ),
  same eval. So the ~8-12% gap is **REAL and comparable**, not a measurement artifact.

## Working recipe (env + args), on a healthy gfx1250
`AITER_FORCE_A8W4=1`, `SGLANG_MOE_SHUFFLE_GFX1250=1`, `AITER_GROUPED_FORCE_SPLIT_K1=1`,
`--kv-cache-dtype auto` (bf16; fp8 KV breaks decode), `--attention-backend triton`,
cuda-graph ON. Model at `/dockerx/data/models/DeepSeek-R1-0528-MXFP4`. Launch:
`/sgl-workspace/sglang/run_ds-r1.sh`. Serves, decode coherent, GSM8K ~0.85.

## 3 code changes THIS environment needed (re-apply after any machine switch)
See CHANGES.md "2026-07-08 session edits" for exact diffs. Summary:
1. **aiter** `ops/flydsl/kernels/gemm_mxscale_gfx1250.py` ~L3010: bisect off-by-one
   `math.ceil(math.log2(batch_count))` -> `int(batch_count).bit_length()` (power-of-2
   expert count; R1=256). Real bug, op-test 3.2e-3->3.4e-6, but end-to-end neutral.
   After editing, `rm -rf /root/.flydsl/cache` so FlyDSL recompiles.
2. **sglang** `quark/schemes/quark_w4a4_mxfp4_moe.py`: enable weight `shuffle_weight
   (w,(16,16))` on gfx1250 when a8w4 (mirror DSv4 fp8.py). MANDATORY — without it
   GSM8K = 0.000 garbage (B-scale is already n32k4-shuffled; weight must match).
3. **aiter** `ops/flydsl/grouped_moe_gfx1250.py`: add `AITER_GROUPED_FORCE_SPLIT_K1`
   env (force split_k1=split_k2=1; CSV picks 2 for token=1 decode -> illegal-address)
   + `AITER_GROUPED_FORCE_TILE_M` investigation knob (default off).

## What is RULED OUT (do not re-chase) — EXPERIMENT_LOG E18-E23
- **MoE a8w4 kernel**: op-test == quant-ref @ 3e-6; and a8w4 activation quant (3.6%)
  is **4x more accurate** than gfx950's a4w4 (15%). MoE is cleaner on gfx1250, cannot
  be the gap. (`scripts/moe_quant_probe.py`)
- **Attention** (triton MLA decode + MHA prefill): both match torch fp32 to ~0.16-0.18%
  across all 61 layers, no outlier. Clean. (`scripts/attn_probe_sitecustomize.py`)
- **bf16 GEMMs** (attn proj / lm_head / dense-MLP): log `torch solution:0` = torch itself.
- cuda-graph vs eager (0.85 vs 0.84), `--disable-radix-cache` (0.83),
  `--triton-attention-reduce-in-fp32` (no-op), qk-rmsnorm torch-vs-triton — all neutral.
- `SGLANG_ROCM_FUSED_DECODE_MLA=1` **crashes** (keep =0).
- Failure modes (`%?%?` loops) are generic high-confidence greedy loops, not numeric collapse.

=> On gfx1250 **every component is numerically correct in isolation**. No single-kernel bug.

## THE remaining path: cross-node per-layer diff
The gap must be a cross-layer/hardware interaction only visible by comparing the two
nodes' **per-layer residual stream** on the SAME fixed prompt.
- **HANDOVER_crossnode_dump.md** — give to the gfx950 node's agent. Contains the
  validated dump hook + launch flags (eager, `--disable-radix-cache`,
  `--skip-server-warmup`) + the byte-identical fixed prompt.
- **scripts/hsdump_sitecustomize.py** — canonical dump hook (post-attention split:
  records per layer `input`, `post_attn` (pre-MoE), `post_layer` (post-MoE)).
- **hs_dump_gfx1250.json** — gfx1250 (TP1) dump, **STALE**: made with the OLD hook (no
  `post_attn` field). Must re-dump gfx1250 with the new hook when a machine is available.
- Diff logic: clean signals are the **dense layers 0-2 (no MoE)** and **`post_attn`** —
  they should match cross-node to the ~1e-2 TP/bf16 floor; a jump = non-MoE bug.
  `post_layer` at MoE layers 3-60 diverges by construction (a4w4 vs a8w4) = control.

### Confounds to remember
1. **MoE differs by construction** (a4w4 vs a8w4) → residual diverges from layer 3 and
   **propagates** downstream. Not a bug. (Why the hook splits `post_attn`.)
2. **expert-routing flips**: top-k argmax on gate logits can select different experts
   from tiny FP diffs → large localized residual change (not a bug).
3. **TP2(gfx950) vs TP1(gfx1250)**: residual replicated so comparable, but adds a ~1e-3
   reduction-order floor compounding to ~1e-2. gfx1250 currently **cannot run TP2**
   (user, 2026-07-09), so this floor is unavoidable — look for divergence clearly above it.
   (A TP2 dump on GPU 2,3 was launched 2026-07-08 but the machine shut down first.)

## Artifacts in this skill dir
- `SKILL.md`, `EXPERIMENT_LOG.md` (E0-E24), `CHANGES.md`, `gfx1250.md` — the playbook.
- `HANDOVER_crossnode_dump.md` — cross-node dump instructions + hook.
- `hs_dump_gfx1250.json` — gfx1250 TP1 per-layer dump (fixed prompt).
- `scripts/moe_quant_probe.py` — MoE a4w4-vs-a8w4 quant-error probe (E20).
- `scripts/attn_probe_sitecustomize.py` — attention decode/prefill torch-fp32 hook (E22).
- `scripts/hsdump_sitecustomize.py` — cross-node per-layer dump hook (post-attn split).

## Machine-switch checklist
1. Confirm GPUs: `ls /dev/kfd /dev/dri && python3 -c "import torch;print(torch.cuda.device_count())"`.
2. Re-apply the 3 code changes (CHANGES.md) if the new repo lacks them; `rm -rf /root/.flydsl/cache`.
3. Point `run_ds-r1.sh --model-path` at the checkpoint; verify it's W4A4 (E2) / matches HF fingerprint (E23).
4. Launch working recipe, sanity curl (coherent, not token-0), GSM8K ~0.85.
5. Next real work = the cross-node dump diff (HANDOVER_crossnode_dump.md).
