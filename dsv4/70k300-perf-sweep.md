# 70k/300 perf sweep — reproducible reference (2026-07-03)

Re-sweep of the **ISL 70000 / OSL 300, low-concurrency {2,4,8,16,32}** workload on
DeepSeek-V4-Pro (8×MI355X), SGLang-only. Reproduces the best-of baseline
(`TODO_70k300.md`) and the DP+TBO A/B (`TBO_RESEARCH.md §19`), and adds a **conc-4
DP / DP+TBO** datapoint. All numbers reproduced the prior references within noise.

## Environment / versions

| item | value |
|---|---|
| Date | 2026-07-03 |
| Hardware | 8× AMD Instinct MI355X (gfx950) |
| Model | `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro` (FP8, 64 shards) |
| sglang tree | `/sgl-workspace/sglang-upstream` (pinned via PYTHONPATH) |
| branch | `feat/dsv4-ep-tbo-prefill` |
| HEAD | `236bcab4e` (`fix(dsv4-tbo): avoid empty two-chunk TBO child on degenerate batches`) |
| key TBO commits | `52792aeec` non-EP DP TBO · `efbe53adf` record_stream→event+ref OOM fix · `d774760a0` drop `SGLANG_TBO_PAD_BUCKET`/`SGLANG_ENABLE_DP_TBO` |
| bench client | `python3 -m sglang.bench_serving` (`--backend sglang-oai`), `NP_MULT=4 WARM_MULT=1` |
| launch script | `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_70k.sh` |
| sweep script | `useful-scripts/benchmarking/dsv4/sweep_dsv4_sglang_client.sh` |

### Pre-flight fix required (recurring)
The Jul-2 `main`-merge into `sglang-upstream` re-introduced the `cohere2_moe.py`
`@strict` import crash (`StrictDataclassDefinitionError`, hf_hub≥1.x — see
`SKILL.md §2a`). Fix applied: make `strict` a no-op in
`python/sglang/srt/configs/cohere2_moe.py` (drop the `huggingface_hub.dataclasses`
import, keep the `def strict(cls): return cls` fallback). Working-tree edit; re-apply
after any future merge that pulls in `main`'s version of that file.

## Results (total tok/s)

Per-conc best config: **c2/c4 → TP8**, **c8/16/32 → DP + TBO**.

| conc | TP8 | DP baseline | DP + TBO | TBO Δ vs DP |
|---:|---:|---:|---:|---:|
| 2  | **13,922** | — | — | — |
| 4  | **17,653** | 16,968 | 17,509 | +3.2% |
| 8  | ~22k (sat.) | 29,138 | **30,736** | **+5.5%** |
| 16 | — | 36,704 | **40,558** | **+10.5%** |
| 32 | — | 42,042 | **46,606** | **+11.7%** |

Bold = best config at that concurrency.

### Full metrics (mean TTFT / median TPOT / median ITL, ms)

| conc | config | total tok/s | Mean TTFT | Median TPOT | Median ITL |
|---:|:--|---:|---:|---:|---:|
| 2  | TP8        | 13,922 | 4,936  | 17.25 | 14.41 |
| 4  | TP8        | 17,653 | 7,952  | 26.12 | 14.95 |
| 4  | DP         | 16,968 | 9,806  | 20.72 | 20.64 |
| 4  | DP+TBO     | 17,509 | 8,941  | 22.56 | 21.75 |
| 8  | DP         | 29,138 | 12,646 | 21.00 | 20.88 |
| 8  | DP+TBO     | 30,736 | 11,265 | 22.08 | 21.96 |
| 16 | DP         | 36,704 | 18,766 | 43.96 | 23.09 |
| 16 | DP+TBO     | 40,558 | 16,298 | 41.42 | 22.98 |
| 32 | DP         | 42,042 | 29,235 | 86.67 | 28.04 |
| 32 | DP+TBO     | 46,606 | 25,533 | 81.10 | 27.91 |

### Validation vs prior reference (within noise)

| conc | metric | reference | this run |
|---:|:--|---:|---:|
| 2 | TP8 | 13,937 | 13,922 |
| 4 | TP8 | 17,655 | 17,653 |
| 8 | DP / +TBO | 29,126 / 30,661 | 29,138 / 30,736 |
| 16 | DP / +TBO | 36,698 / 40,376 | 36,704 / 40,558 |
| 32 | DP / +TBO | 42,035 / 46,935 | 42,042 / 46,606 |

## Findings

- **TBO win grows with concurrency: +5.5% → +10.5% → +11.7%** (c8→c16→c32), consistent
  with the prefill-dominated 70k/300 profile. TTFT also drops 10–13%. **No HSA crash** at
  any conc — the `record_stream→event+ref` fix (`efbe53adf`) holds at mem0.80 with the
  3.88M-token KV.
- **c4 DP vs TP8 (new datapoint, closes an open item):** DP baseline **16,968 < TP8
  17,653** → confirms DP-attention loses to plain TP8 at c4 (idle ranks). **DP+TBO
  (17,509) nearly recovers the gap** but TP8 stays marginally best at c4.
- Recommended per-conc config is unchanged from `TODO_70k300.md`.

## Commands (exact, reproducible)

cwd for all: `/workspace/useful-scripts/benchmarking/dsv4/`.
Cold load ~9–10 min/launch (faster if weights cached); poll `http://127.0.0.1:8000/health`
(200) before benching. `run_sgl_dsv4_70k.sh` internally pins the sglang-upstream PYTHONPATH.

### Config 1 — TP8 baseline (conc 2, 4)
```bash
# server
MODE=tp8 CHUNK=32768 SWA=0.1 MEM=0.92 PORT=8000 bash run_sgl_dsv4_70k.sh
# client
PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter \
RESULT_DIR=/workspace/bench_results_dsv4_70k/tp8_chunk32768 \
BENCH="python3 -m sglang.bench_serving" BACKEND=sglang-oai \
WORKLOADS="70000:300" CONCS="2 4" NP_MULT=4 WARM_MULT=1 bash sweep_dsv4_sglang_client.sh
```

### Config 2 — DP-attention baseline (conc 4, 8, 16, 32)
```bash
# server
MODE=tp8dp8 CHUNK=16384 SWA=0.1 MEM=0.80 DELAYER=off PORT=8000 bash run_sgl_dsv4_70k.sh
# client
PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter \
RESULT_DIR=/workspace/bench_results_dsv4_70k/tp8dp8_chunk16384_nodelayer \
BENCH="python3 -m sglang.bench_serving" BACKEND=sglang-oai \
WORKLOADS="70000:300" CONCS="4 8 16 32" NP_MULT=4 WARM_MULT=1 bash sweep_dsv4_sglang_client.sh
```

### Config 3 — DP-attention + TBO (conc 4, 8, 16, 32)
```bash
# server (only diff vs Config 2 = TBO flag + HW-queue cap)
GPU_MAX_HW_QUEUES=5 SGL_EXTRA_ARGS="--enable-two-batch-overlap" \
MODE=tp8dp8 CHUNK=16384 SWA=0.1 MEM=0.80 DELAYER=off PORT=8000 bash run_sgl_dsv4_70k.sh
# client
PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter \
RESULT_DIR=/workspace/bench_results_dsv4_70k/tp8dp8_chunk16384_tbo \
BENCH="python3 -m sglang.bench_serving" BACKEND=sglang-oai \
WORKLOADS="70000:300" CONCS="4 8 16 32" NP_MULT=4 WARM_MULT=1 bash sweep_dsv4_sglang_client.sh
```

### Server kill between configs (frees all VRAM)
```bash
pkill -9 -f 'sglang serve'; pkill -9 -f 'sglang.launch_server'
kill -9 $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+ /{print $1}') 2>/dev/null
sleep 20; rocm-smi --showmeminfo vram | grep -i used | head -1   # confirm back to ~0.3 GB/GPU
```

## Raw results
`/workspace/bench_results_dsv4_70k/{tp8_chunk32768, tp8dp8_chunk16384_nodelayer, tp8dp8_chunk16384_tbo}/`
(`.log` + `.jsonl` per conc, plus `sglangClient_sweep_summary.txt`).
