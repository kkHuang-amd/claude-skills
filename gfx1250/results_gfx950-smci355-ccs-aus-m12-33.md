# Per-node results — gfx950 node `smci355-ccs-aus-m12-33` (8x MI355X, gfx950)

Dedicated raw-run log for THIS gfx950 box (hostname `smci355-ccs-aus-m12-33`), to avoid
write conflicts on the shared `EXPERIMENT_LOG.md` / `STATUS.md` / `results_gfx950.md`.
Tree: `/sgl-workspace/sglang_gfx-1250` @ `000a61a2`. Model:
`/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4` (tokenizer.json md5 `60264268dbd8a72aa3c8d813faec0ed3`).
Native a4w4 GSM8K reference on this node ~0.93-0.95.

Promote short summaries into the shared logs only when safe; write new runs here first.

---

## 2026-07-10 — fp8 KV cache validation (fp8_e4m3) — WORKS on gfx950

Question: does fp8 KV cache break decode on gfx950 like it does on gfx1250 (SKILL §4.7:
gfx1250 fp8-KV -> every decode step emits token 0 -> GSM8K ~0)?

Setup: `run_ds-r1_gfx950.sh` with only `--kv-cache-dtype auto` changed to `--kv-cache-dtype
fp8_e4m3` (script: `/tmp/run_gfx950_fp8kv.sh`). TP2 (GPU0,1), cuda-graph ON (production
recipe), real native a4w4 MoE, `AITER_FORCE_A8W4=1` (linear bf16-dequant + Triton qk-rmsnorm),
triton attn backend. KV cache allocated as `torch.float8_e4m3fn`, 76.03 GB.

Sanity decode (greedy): coherent, NO token-0 degeneration —
- "The capital of France is" -> " Paris. Paris is located in northern France ..."
- "Question: What is 2+2? Answer:" -> " 4 ..."

GSM8K (5-shot, greedy):
| harness | Q | Accuracy | Invalid | latency |
|---|---|---|---|---|
| few_shot_gsm8k (parallel 32) | 200 | 0.950 | 0.000 | 33.8 s |
| benchmark/gsm8k/bench_sglang.py (parallel 1319) | 1319 | **0.941** | 0.000 | 214.9 s (627.9 tok/s) |

Reference (bf16 KV, same node): 200Q ~0.945-0.950, 1319Q 0.944.

**Conclusion: fp8 KV cache is accuracy-neutral on gfx950** (1319Q 0.941 vs bf16 0.944 = noise;
decode coherent). => the fp8-KV decode breakage (token-0 loop) recorded in SKILL §4.7 is
**gfx1250-specific** (its triton MLA decode fp8-KV read path), NOT a model/recipe issue. gfx950
can use fp8 KV to halve KV memory (76 GB fp8) with no accuracy loss.

---

## (append new smci355-ccs-aus-m12-33 runs below)

## 2026-07-10 — fp8 KV bug cross-check for gfx1250 (tests 1/2/3)  [node: smci355-ccs-aus-m12-33 gfx950, gcnArchName gfx950:sramecc+:xnack-]

Cross-control for the gfx1250 fp8-KV decode crash (gfx1250 fp8 KV = GSM8K 0.000; gfx950 = 0.941;
byte-identical triton MLA code, commit 000a61a2 == 3923a34d). Goal: is fp8 OVERFLOW handling or
fp8 DOT correctness different on gfx950 vs gfx1250?

### Test 1 — bf16->fp8 (e4m3fn) cast overflow  [HIP_VISIBLE_DEVICES=1 python3]
```
bf16   100.0 -> fp8 96.0
bf16   400.0 -> fp8 384.0
bf16   448.0 -> fp8 448.0
bf16   449.0 -> fp8 448.0
bf16   500.0 -> fp8 nan
bf16  1000.0 -> fp8 nan
bf16  5000.0 -> fp8 nan
arch: gfx950:sramecc+:xnack-
```
=> gfx950 is **NOT pure-saturate**: 449->448 (round to max) but 500/1000/5000 -> **NaN**.
**IDENTICAL to gfx1250** (449->448, 500+->NaN). Threshold ~464 (midpoint 448<->480=NaN encoding).
CONCLUSION: fp8 overflow handling is the SAME on both archs -> NOT the arch difference.

### Test 2 — minimal triton fp8xfp8 tl.dot (64x64x64, single tile), rel_l2 vs fp64 of same fp8 vals
A = randn*qscale (query-like), B = randn*0.4 (latent KV-like); /tmp/test2_fp8_dot.py
```
arch: gfx950:sramecc+:xnack-
qscale    1  |a|max(fp8)=    3.2  A_has_nan=False  rel_l2=1.263e-05  C_has_nan=False
qscale   30  |a|max(fp8)=   96.0  A_has_nan=False  rel_l2=1.376e-05  C_has_nan=False
qscale  100  |a|max(fp8)=  320.0  A_has_nan=False  rel_l2=1.324e-05  C_has_nan=False
qscale  300  |a|max(fp8)=    nan  A_has_nan=True   rel_l2=nan        C_has_nan=True
```
vs gfx1250: 1/30/100 -> rel_l2 1e-7~3e-7 (correct); 300 -> NaN.
=> gfx950 fp8 dot is CORRECT up to |a|max=320 (rel_l2 ~1.3e-5). qscale 300 -> NaN on BOTH archs,
but the NaN comes from the bf16->fp8 CAST (|a|~992>464 -> A_has_nan=True), NOT the dot itself.
Minor diff: gfx950 rel_l2 ~1.3e-5 vs gfx1250 ~3e-7 (gfx950 slightly LESS precise) — opposite
direction, both tiny; does not explain gfx1250's crash.

### Test 3 — real kernel: fp8 KV + radix ON + cuda-graph ON (unmodified q->fp8 code)
`run_ds-r1_gfx950.sh` with `--kv-cache-dtype fp8_e4m3` (radix ON + cuda-graph ON by default),
native a4w4 MoE, TP2. GSM8K 5-shot: 200Q = 0.950, **1319Q = 0.941** (invalid 0.000), decode
coherent. => the ORIGINAL q->fp8 downcast path is GOOD on gfx950. (raw run recorded in the
"fp8 KV cache validation" section above.)

### Net for the gfx1250 investigation
Tests 1 & 2 show gfx950 and gfx1250 have the SAME fp8 overflow handling (round-then-NaN) and the
SAME fp8-dot behavior (correct at |a|<=~320, cast-NaN at ~992). So the gfx1250 fp8-KV crash at
real query |max|~330 (< 448, no cast-NaN, dot correct at that magnitude per gfx1250's own test)
is NOT explained by raw fp8 cast/dot semantics — the divergence must be elsewhere in the gfx1250
path (e.g. how the triton MLA decode/extend kernel scales/accumulates fp8 KV, scale application,
or a gfx1250-specific codegen of the same kernel), not in the ISA-level fp8 saturate/dot.
