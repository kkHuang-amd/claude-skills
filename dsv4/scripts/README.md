# DSV4 investigation scripts

Helper scripts created during the DSV4 MoE perf investigation (PR
sgl-project/sglang#27858). The **launch / sweep / summarize** scripts live in
`useful-scripts/benchmarking/dsv4/` (`run_sgl_dsv4_aligned.sh`,
`run_atom_dsv4_aligned.sh`, `summarize_sgl_dsv4.py`, etc.); the scripts here are
the ad-hoc analysis/benchmark tools built on top of them.

Paths below assume artifacts under `/workspace/`. Adjust as needed.

## Benchmarking

- **`sweep_driver.sh`** — drives a full concurrency sweep against an already-running
  OpenAI-compatible server via `bench_dsv4.py`.
  `bash sweep_driver.sh <RESULT_DIR> "<1k_conc_list>" "<8k_conc_list>"`
  Defaults: random ratio **0.8**, num-prompts = conc*8, warmups = conc*2,
  ignore-eos on (matches the ATOM Experiment-1 methodology). Edit the `run 1024
  1024` / `run 8192 1024` lines to change ISL/OSL or the np/warm rule.
  Summarize results with `summarize_sgl_dsv4.py <label>=<RESULT_DIR>`; render
  with the layout in `../RESULTS_TABLE_FORMAT.md`.

- **`compare_atom.py`** — SGLang-vs-ATOM comparison table (out tok/s, SGL/ATOM%,
  ITL, TTFT). Loads SGLang result jsonls from `/workspace/bench_r08/{tp8,tp8dp8}`;
  ATOM numbers are hard-coded from `../EXPERIMENT_LOG.md` Experiment 1 (update the
  `ATOM_TP8` / `ATOM_DP8` dicts if those change).

## Kernel microbench

- **`microbench_matrix.py`** — isolates moe1/moe2 kernel time by replaying a single
  captured decode step through `aiter.fused_moe`, varying weights-engine / routing
  / `intermediate_pad`. Needs dumps at `/workspace/moe1_dump_{sgl,atom}/full.pt`
  (produced by `moe_dump/moe1_dump.py`). Run with `AITER_BF16_FP8_MOE_BOUND=0` to
  select the flydsl kernels. This is what produced the "pad=0 vs pad=128" 127→104µs
  / 98→80µs numbers in the PR.

## Routing investigation (active-expert gap)

Goal: prove ATOM and SGLang route to the same #active experts given identical
input (the 103-vs-216 was a data artifact, not an engine difference).

- **`moe_dump/router_dump.py`** — sentinel-gated dump/count of the MoE router stage
  (hidden, router_logits, topk_ids/weights, correction_bias, config) per layer.
  Install to site-packages so worker procs can import it (`sglang serve` does NOT
  propagate inline env/PYTHONPATH to workers — gate on the sentinel files
  `/workspace/router_dump/.enable` for full dump, `.count` for lightweight
  per-step active-expert counting). Hook call sites:
    - SGLang: `TopK.forward_cuda` STANDARD branch + `MoERunnerConfig.layer_id`.
    - ATOM:   after `FusedMoE.select_experts` in `model_ops/moe.py` apply().
- **`moe_dump/moe1_dump.py`** — dumps full moe1 tensors (weights/scales/routing/
  kwargs) at the `fused_moe` call site, gated by `DUMP_MOE1_DIR`. Feeds
  `microbench_matrix.py`.
- **`router_probe.py`** — fires 64 fixed, diverse prompts concurrently at the
  server so a 64-wide decode batch forms (`python3 router_probe.py <port>`).
  Auto-detects model id from `/v1/models`. Needs long max_tokens so the batch
  persists for many steps.
- **`analyze_router.py`** — per-layer active-expert counts + hidden homogeneity for
  ATOM vs SGLang from the router dumps; validates a reference sqrtsoftplus+bias
  selector reproduces each engine's actual topk.
- **`cross_hidden.py`** — cross-engine match of hidden states / router_logits
  (cosine) on the same prompts.
- **`analyze_routing.py`** — quick concentration/collapse stats from the older
  `moe1_dump_{sgl,atom}/full.pt` dumps.

## Typical flow

1. Launch server: `DP_MODE=tp8 bash run_sgl_dsv4_aligned.sh` (or ATOM equivalent).
2. Perf sweep: `bash sweep_driver.sh /workspace/bench_r08/tp8 "2 4 8 16 32 64" "4 8 16 32 64"`.
3. Summarize + format per `../RESULTS_TABLE_FORMAT.md`; compare via `compare_atom.py`.
4. (Investigation) enable dump sentinels, run `router_probe.py`, then
   `analyze_router.py` / `cross_hidden.py`; kernel timing via `microbench_matrix.py`.
