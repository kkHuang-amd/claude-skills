# DeepSeek-V4-Pro serving perf on **B200** — 70k/300 sweep (2026-06-25)

B200 (NVIDIA) port of the 70k/300 low-concurrency study (companion to the MI35x
log `EXPERIMENT_LOG_2026-06-25.md`). Same workload regime: **70k input / 300
output, low concurrency {2,4,8,16,32}** — long-context, prefill-dominated. Two
goals: (1) confirm gsm8k correctness with DP-attention + the dp collectives on
B200; (2) find the best B200 serving config per concurrency, and check whether
**conc=32 even fits** given B200's smaller VRAM.

- **Date**: 2026-06-25
- **HW**: 8× **NVIDIA B200**, 183 GB/GPU (vs MI35x 288 GB) — host `dgx-021`.
- **Model**: `/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro` (fp8 ckpt,
  MoE experts are **mxfp4** → `is_fp4_experts=True`, served via
  `--moe-runner-backend flashinfer_mxfp4`). `max_position_embeddings=1048576`.
- **sglang**: `/sgl-workspace/sglang-upstream` (HEAD `ffb1afd5e`; PR #28216
  gatherv + #29103 reduce_scatter). **Pinned via `PYTHONPATH`** because the
  default `import sglang` on this box resolves to `/sgl-workspace/sglang`
  (HEAD `a17753e`, no DP work) — wrong tree.
- **Client**: `python3 -m sglang.bench_serving` (sglang-oai), ratio 1.0,
  request-rate inf, ignore-eos (default), num-prompts=conc×4, warmups=conc×1.
- **Scripts** (in `useful-scripts/benchmarking/dsv4/`):
  `run_sgl_dsv4_70k_b200.sh` (B200 70k launch; MODE/CHUNK/SWA/MEM/DELAYER knobs),
  `run_sgl_dsv4_b200.sh` (canonical B200 launch + dp toggle, used for gsm8k),
  `gsm8k_b200.sh` (gsm8k via in-tree few_shot_gsm8k, no lm_eval needed),
  reused `sweep_dsv4_sglang_client.sh`.
- Results under `/dockerx/raid/home/wunhuang/workspace/bench_results_dsv4_70k_b200/`,
  logs under `/dockerx/raid/home/wunhuang/workspace/dsv4_b200_logs/`.

> **Headline (best-of, B200, 70k/300):**
> | conc | mode | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) | retract | vs MI35x |
> |---:|:--|---:|---:|---:|---:|---:|---:|
> | 2  | tp8 | 15,969 | 5.03 | 12.6 | 8.8  | 0 | +15% |
> | 4  | dp8 | 21,885 | 7.54 | 17.7 | 12.8 | 0 | +24% |
> | 8  | dp8 | **34,234** | 10.85 | 18.6 | 16.4 | 0 | +17% |
> | 16 | dp8 | **43,576** | 16.07 | 32.5 | 25.8 | 0 | +18% |
> | 32 | dp8 | **47,065** | 20.01 | 91.4 | 47.4 | 0 | +12% |
>
> **conc=32 FITS on B200** (dp8, mem 0.80): 128/128 successful, **0 retract**.
> B200 beats MI35x at every concurrency. **Crossover is earlier than MI35x**:
> dp8 already wins at **conc 4** (MI35x preferred tp8 there). Only conc 2 → tp8.

---

## Best-of summary table (B200, 70k/300) — client medians

`TP,DP,EP` = `8,` is plain TP8 (conc 2); `8,8` is TP8+DP-attention; no EP.
`Interactivity = 1000 / Median ITL` (tok/s/user).

| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) |
|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 70000 | 300 | 8,  | 2  | 15,969.42 | 8,795.32  | 5,028.74  | 11.240 | 88.97 |
| 70000 | 300 | 8,8 | 4  | 21,885.06 | 12,822.97 | 7,682.28  | 16.725 | 59.79 |
| 70000 | 300 | 8,8 | 8  | 34,234.45 | 16,374.56 | 11,039.37 | 16.937 | 59.04 |
| 70000 | 300 | 8,8 | 16 | 43,575.60 | 25,811.36 | 15,929.52 | 19.467 | 51.37 |
| 70000 | 300 | 8,8 | 32 | 47,065.23 | 47,265.20 | 19,335.18 | 20.540 | 48.68 |

### Reproduction commands

All scripts in `useful-scripts/benchmarking/dsv4/`; codebase pinned to
`/sgl-workspace/sglang-upstream` (via the scripts' `PYTHONPATH`).

**Server (config A — conc 2 → TP8, table row `8,`):**
```bash
cd /dockerx/raid/home/wunhuang/workspace/useful-scripts/benchmarking/dsv4
MODE=tp8 CHUNK=32768 MEM=0.90 bash run_sgl_dsv4_70k_b200.sh
```
expands to:
```bash
sglang serve --model-path /dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 8000 --trust-remote-code --tp 8 \
  --disable-radix-cache --max-running-requests 64 --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 32768 --disable-flashinfer-autotune \
  --kv-cache-dtype fp8_e4m3 --context-length 73728 --cuda-graph-max-bs 64
```

**Server (config B — conc 4/8/16/32 → TP8+DP-attention, table rows `8,8`):**
```bash
cd /dockerx/raid/home/wunhuang/workspace/useful-scripts/benchmarking/dsv4
MODE=tp8dp8 CHUNK=16384 SWA=0.1 MEM=0.80 DELAYER=off bash run_sgl_dsv4_70k_b200.sh
```
expands to (note `--chunked-prefill-size 131072` = 16384×8; dp divides by dp_size):
```bash
sglang serve --model-path /dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 8000 --trust-remote-code --tp 8 --dp 8 --enable-dp-attention \
  --disable-radix-cache --max-running-requests 64 --mem-fraction-static 0.80 \
  --swa-full-tokens-ratio 0.1 --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 131072 --disable-flashinfer-autotune \
  --kv-cache-dtype fp8_e4m3 --context-length 73728 --cuda-graph-max-bs 64
```

B200 env applied by the wrapper (dp adds the last two):
```bash
export PYTHONPATH=/sgl-workspace/sglang-upstream/python:$PYTHONPATH
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
export SGLANG_DP_USE_REDUCE_SCATTER=1   # dp only
export SGLANG_DP_USE_GATHERV=1          # dp only
```

**Client (run after the server is ready):** num-prompts=conc×4, warmups=conc×1,
fixed lengths (ratio 1.0), ignore-eos (default).
```bash
cd /dockerx/raid/home/wunhuang/workspace/useful-scripts/benchmarking/dsv4
# against config A (conc 2):
PYTHONPATH=/sgl-workspace/sglang-upstream/python BENCH="python3 -m sglang.bench_serving" \
MODEL=/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro \
RESULT_DIR=/dockerx/raid/home/wunhuang/workspace/bench_results_dsv4_70k_b200 \
WORKLOADS="70000:300" CONCS="2" NP_MULT=4 WARM_MULT=1 RATIO=1.0 \
bash sweep_dsv4_sglang_client.sh

# against config B (conc 4 8 16 32):
PYTHONPATH=/sgl-workspace/sglang-upstream/python BENCH="python3 -m sglang.bench_serving" \
MODEL=/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro \
RESULT_DIR=/dockerx/raid/home/wunhuang/workspace/bench_results_dsv4_70k_b200/dp8 \
WORKLOADS="70000:300" CONCS="4 8 16 32" NP_MULT=4 WARM_MULT=1 RATIO=1.0 \
bash sweep_dsv4_sglang_client.sh
```
Each point expands to (conc 8 example):
```bash
python3 -m sglang.bench_serving --backend sglang-oai --base-url http://127.0.0.1:8000 \
  --model /dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 70000 --random-output-len 300 \
  --random-range-ratio 1.0 --num-prompts 32 --max-concurrency 8 \
  --request-rate inf --warmup-requests 8 \
  --output-file .../sglangClient_dsv4_isl70000_osl300_c8.jsonl
```

| Concurrency | Server config | Client CONCS |
|---:|:--|:--|
| 2  | A (tp8) | `CONCS="2"` |
| 4  | B (dp8) | `CONCS="4"` |
| 8  | B (dp8) | `CONCS="8"` |
| 16 | B (dp8) | `CONCS="16"` |
| 32 | B (dp8) | `CONCS="32"` |

---

## Env: B200 ≠ MI35x (the flags are NOT all portable)

The MI35x recipe uses ROCm-only flags (`SGLANG_USE_AITER`, `SGLANG_USE_ROCM700A`,
`SGLANG_HACK_FLASHMLA_BACKEND`, `SGLANG_OPT_USE_FUSED_COMPRESS*`,
`AITER_BF16_FP8_MOE_BOUND`, ...). None apply on B200. The B200 env set used
(verbatim from the canonical B200 command) is:

```
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
SGLANG_OPT_USE_JIT_NORM=1
SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
SGLANG_OPT_USE_TOPK_V2=1
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
```

Note `SGLANG_OPT_USE_JIT_INDEXER_METADATA`/`SGLANG_OPT_USE_TOPK_V2` are **=1 on
B200** but **=false on MI35x** — opposite settings. DP collectives
`SGLANG_DP_USE_REDUCE_SCATTER` / `SGLANG_DP_USE_GATHERV` default to `_default_hip`
/ off on CUDA, so they must be **explicitly set to 1** on B200 (they're on by
default on ROCm).

### gsm8k correctness with DP collectives (Task 1)
tp8dp8 + `SGLANG_DP_USE_REDUCE_SCATTER=1` + `SGLANG_DP_USE_GATHERV=1`:
**gsm8k acc = 0.970, invalid = 0.000** (400 q, 5-shot). The reduce_scatter
(decode) + gatherv (prefill) collectives are numerically correct on B200.

## Phase A — TP8 (no dp), chunk 32768/rank, mem 0.90

| conc | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) | retract |
|---:|---:|---:|---:|---:|---:|
| 2 | 15,969 | 5.03 | 12.6 | 8.8  | 0 |
| 4 | 19,856 | 7.88 | 21.0 | 14.2 | 0 |
| 8 | 22,092 | 9.67 | 52.0 | 25.2 | 0 |

- TP8 pool (mem 0.90): `max_total_num_tokens=2,457,600`, swa=245,760.
- Same saturation shape as MI35x: TP8 plateaus ~conc 8 (~22k) as a single
  request's attention is sharded over all 8 GPUs; extra conc just queues
  (TPOT 12.6→52ms, E2E 8.8→25.2s from conc 2→8).

## Phase B — TP8 + DP-attention, chunk 16384/rank (global 131072), swa 0.1, mem 0.80, delayer OFF

| conc | total tok/s | TTFT(s) | TPOT(ms) | E2E(s) | retract |
|---:|---:|---:|---:|---:|---:|
| 2  | 12,346 | 6.13 | —    | —    | 0 |
| 4  | 21,885 | 7.54 | 17.7 | 12.8 | 0 |
| 8  | 34,234 | 10.85| 18.6 | 16.4 | 0 |
| 16 | 43,576 | 16.07| 32.5 | 25.8 | 0 |
| 32 | 47,065 | 20.01| 91.4 | 47.4 | 0 |

- Per-DP-rank pool (mem 0.80): full=871,424, swa=87,040 tokens/rank. conc=32 in
  dp8 needs only 4 reqs×70k=280k tokens/rank → full pool ~1/3 used, **memory not
  binding** (0 retract). **mem 0.80 is the right choice** (matches the MI35x dp
  recipe; leaves room for the global MoE activation = chunk_global = 131072).
- DP scales where TP8 plateaus (34→44→47k). **dp8 vs tp8: conc4 +10%, conc8
  +55%.**
- **B200-specific crossover:** dp8 already beats tp8 at **conc 4** (21,885 vs
  19,856). On MI35x conc 4 preferred tp8. Only **conc 2** prefers tp8
  (15,969 tp8 vs 12,346 dp8 — dp idles 6/8 ranks at conc 2).

## DP collectives ON vs OFF (SGLANG_DP_USE_REDUCE_SCATTER / GATHERV)

Same dp8 config (chunk 16384/rank, swa 0.1, mem 0.80, delayer off); only the two
flags toggled (`DP_COLL=on|off` knob in `run_sgl_dsv4_70k_b200.sh`). All 0
retract, all requests successful. conc 2 is tp8 so unaffected.

| conc | TTT ON (tok/s) | TTT OFF (tok/s) | Δ tput | Med E2E ON→OFF (ms) | Med ITL ON→OFF (ms) |
|---:|---:|---:|---:|---:|---:|
| 4  | 21,885.06 | 21,254.30 | **−2.9%** | 12,822.97 → 13,191.80 | 16.725 → 16.024 |
| 8  | 34,234.45 | 32,666.53 | **−4.6%** | 16,374.56 → 17,194.87 | 16.937 → 16.268 |
| 16 | 43,575.60 | 40,630.80 | **−6.8%** | 25,811.36 → 27,627.26 | 19.467 → 18.871 |
| 32 | 47,065.23 | 45,005.40 | **−4.4%** | 47,265.20 → 49,119.22 | 20.540 → 20.167 |

- Turning the collectives OFF costs **−2.9% to −6.8% total throughput** (worst at
  conc 16) and raises Median E2E across the board.
- Median ITL (per-token decode latency) is marginally *better* with them OFF —
  reduce_scatter targets the decode all-reduce while gatherv targets prefill
  aggregation; since this regime is prefill-dominated (ISL≫OSL), gatherv's prefill
  win drives the net throughput gain even though the decode-side effect is tiny.
- **Keep both ON**: faster *and* numerically validated (gsm8k acc 0.970).
- Raw data: `bench_results_dsv4_70k_b200/dp8_nocoll/`.

### Best-of table — collectives OFF (client medians)

Same layout as the ON best-of table above; conc 2 is tp8 (unaffected by the flags).
`Interactivity = 1000 / Median ITL`.

| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) |
|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 70000 | 300 | 8,  | 2  | 15,969.42 | 8,795.32  | 5,028.74  | 11.240 | 88.97 |
| 70000 | 300 | 8,8 | 4  | 21,254.30 | 13,191.80 | 8,257.28  | 16.024 | 62.41 |
| 70000 | 300 | 8,8 | 8  | 32,666.53 | 17,194.87 | 12,066.80 | 16.268 | 61.47 |
| 70000 | 300 | 8,8 | 16 | 40,630.80 | 27,627.26 | 17,779.35 | 18.871 | 52.99 |
| 70000 | 300 | 8,8 | 32 | 45,005.40 | 49,119.22 | 21,293.89 | 20.167 | 49.59 |

## Not applicable / not tried
- **two-batch-overlap (TBO)**: rejected at arg-check — `ValueError: When enabling
  two batch overlap, moe_a2a_backend cannot be 'none'`. TBO needs an EP/all2all
  MoE backend (deepep), not `flashinfer_mxfp4`. EP path not explored (same as
  MI35x). This is the most promising remaining throughput lever if EP is set up.
- chunk-size sweep: kept 16384/rank (MI35x found it ~neutral and it bounds the
  global MoE batch; 32768/rank → global 262144 risks more MoE activation mem).
- prefill-delayer: kept **OFF** (MI35x proved OFF wins for prefill-dominated
  low-conc; needs decode occupancy to protect, which 70k/300 lacks).

## Recommended B200 settings (70k/300)
- **conc 2** → `MODE=tp8` (chunk 32768/rank, mem 0.90).
  `MODE=tp8 CHUNK=32768 MEM=0.90 bash run_sgl_dsv4_70k_b200.sh`
- **conc 4, 8, 16, 32** → `MODE=tp8dp8`, chunk **16384/rank**
  (`--chunked-prefill-size 131072`), swa 0.1, **mem 0.80**, delayer OFF,
  `SGLANG_DP_USE_REDUCE_SCATTER=1` + `SGLANG_DP_USE_GATHERV=1`.
  `MODE=tp8dp8 CHUNK=16384 MEM=0.80 DELAYER=off bash run_sgl_dsv4_70k_b200.sh`
- Common: `--moe-runner-backend flashinfer_mxfp4 --kv-cache-dtype fp8_e4m3
  --context-length 73728 --cuda-graph-max-bs 64 --max-running-requests 64
  --disable-radix-cache --disable-flashinfer-autotune` + the 6 B200 env flags.

## Open / next
- EP/deepep + TBO on B200 (only way to test TBO; could lift the dp numbers).
- conc>32 or longer context would start pressuring the dp swa pool (87k/rank).
- shared-expert-local PoC (`SE_LOCAL=on`) not benchmarked here (kept the
  validated-correct 2-flag dp set); worth a perf check.
