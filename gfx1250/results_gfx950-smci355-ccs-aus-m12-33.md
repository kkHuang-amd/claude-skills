# Per-node results — gfx950 node `smci355-ccs-aus-m12-33` (8x MI355X, gfx950)

Dedicated raw-run log for THIS gfx950 box (hostname `smci355-ccs-aus-m12-33`), to avoid
write conflicts on the shared `EXPERIMENT_LOG.md` / `STATUS.md` / `results_gfx950.md`.
Tree: `/sgl-workspace/sglang_gfx-1250` @ `000a61a2`. Model:
`/dockerx/data/amd/DeepSeek-R1-0528-MXFP4` (tokenizer.json md5 `60264268dbd8a72aa3c8d813faec0ed3`).
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
