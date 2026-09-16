# AgentX on B200 / CUDA — platform notes

Platform-isolated companion to `../SKILL.md`, which was written on an
**MI355X / ROCm** node. Everything here is the **NVIDIA B200** node: 8× B200
(driver 580.173.02), 224 cores, 2015 GB RAM, repo at
`/workspace/agentx/InferenceX`. Read this file first when working on B200; use
`../SKILL.md` for pipeline mechanics that are platform-independent (layout
requirements, trace resolution, monitoring traps), and ignore its ROCm-specific
results.

## CONTINUE HERE

**Goal (evolved): explain the B200-vs-MI355X ITL gap on AgentX c128.** It began
as "get a profiling reference at parity with CI"; that reference exists and the
work moved on to the cross-platform gap. Read §8 (parity/CI), the A/B sections,
and `../exchange/b200-decode-trace.md` for the current front.

**Status (2026-09-16 10:25): the apparent 2.5× ITL gap is down to a
config-matched 1.55×, and the open question is whether that 1.55× is in decode
kernels or in the prefill barrier around them.**

Settled so far:

- `prefill_decode_interval` was 10 on MI355X and hardcoded 24 on B200. Matching
  it moved the gap from an apparent 2.5× to a stable ~1.6× at *both* settings
  (33.9/20.7 at pdi=24, 50.6/32.1 at pdi=10), so the knob inflated the gap, it
  never was the cause.
- Multi-stream overlap (CUDA-only, no ROCm branch) is worth ~4-5 % of step time.
  It fills gaps rather than inflating kernels, so per-kernel times **are**
  comparable across platforms without normalising streams.
- Four controls now matched: pdi, `accept len` (3.770 vs 3.78), per-request KV
  working set (151,908 vs ~151,000 tokens), cuda-graph replay (100 % both).
  Queueing explains only ~6.5 % of the gap; KV pressure and lost graph replay
  are excluded.
- At matched batch the residual is **1.54-1.57× on log-implied step time**
  across batch 9/12/16.

**The discriminator, and the one number still missing:** log-implied step time
includes amortised prefill. B200's *pure decode* step wall is **15.9 ms** while
its log-implied step is ~72 ms at batch 9 — only ~22 % of wall time is inside
decode steps. So if MI355X's pure decode step wall is ~25 ms the gap is in
kernels; if it is ~16 ms the gap is in the prefill/waiting portion and the
kernel breakdown is the wrong place to look. **These have different fixes.**

**Next action:** `git pull` in `/workspace/claude-skills`, read
`agentx/exchange/` for a new `mi355x-*.md`, and compare against
`agentx/exchange/b200-decode-trace.md`. If MI355X has not reported yet, the
request is written out in `agentx/exchange/README.md` and the capture
instructions in `agentx/analysis/MI355X_CAPTURE_PROMPT.md`.

### The CI-parity reference arm, for the record

**Reference arm (pdi=24, multi-stream, 2026-09-15): COMPLETE, n=1.**

| metric | value |
|---|---|
| output tok/s/GPU | **403.6** |
| total tok/s/GPU | **46,170** |
| TTFT p50 | 4.59 s |
| ITL p90 | 20.7 ms |
| interactivity p90 | 48.31 |
| theoretical cache hit | 0.9635 |
| GPU / host cache hit | 0.523 / 0.429 |
| ISL mean | 114,401 tokens |
| OSL actual / expected mean | 1,009 / 1,241 |
| profiled requests | 11,612 (22 dropped `InvalidInferenceResultError`) |
| duration | 3,628 s |
| avg total GPU power | 5,723 W (~715 W/GPU), 1.78 J per output token |

The launcher's own gate passed: aiperf request error rate 1/11613 = 0.009 %
against a 10 % threshold, and the `sglang:` server-metric prefix was present.

Read it back with
`python3 /workspace/agentx/show_result.py /workspace/agentx/results/b200-tp8-ep8-dpatrue-c128/*.json`.

Two things to know about this number before comparing it to anything:

- **HiCache is doing real work**: 0.523 of hits came from HBM and 0.429 from the
  host DRAM tier. A comparison against any arm with a different
  `allocated_cpu_dram_gb` (1612 here) is not like-for-like.
- **OSL actual (1,009) is ~19 % below expected (1,241)**, so the replay is
  producing shorter responses than the traces ask for. Worth checking against
  CI's json — if CI matches expected more closely, the throughput numbers are
  not measuring the same work.

A replicate of this arm was never run; the spread bound came from the trace run
instead (output tok/s/GPU reproduced to −0.29 %, ITL p90 within ~9 % under
non-matched conditions). Treat ITL differences under ~10 % as unresolved unless
conditions are matched.

**Always run `/workspace/agentx/free_gpus.sh` before launching an arm** — the
launcher orphans a whole process family and holds the dist-init ports (§7).

### First run, for the record

Checkpoint complete (66/66 shards — the 0813 variant has 66, not the 64 of
plain V4-Pro). Launched as:

```bash
cd /workspace/agentx && ./wait_and_run.sh env CONC=128 ./agentx_run_b200.sh \
  > /workspace/agentx/logs/arm3_c128.log 2>&1 &
```

Confirmed from `sglang_command.txt` that this really is the dp8+ep8+megamoe
arm: `--tp 8 --dp 8 --ep-size 8 --enable-dp-attention --enable-dp-lm-head
--moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe
--enable-deepseek-v4-fp4-indexer --mem-fraction-static 0.88
--chunked-prefill-size 49152 --max-running-requests 256 --cuda-graph-max-bs 32
--speculative-algorithm DSPARK --speculative-dspark-block-size 6
--hicache-ratio 8`.

**Artefacts are split across two directories for this run** (fixed in the
scripts afterwards, see §6):

```
/workspace/agentx/InferenceX/agentx/results/b200-tp8-ep8-dpatrue-c128/   server.log, router.log, sglang_command.txt, aiperf artefacts
/workspace/agentx/results/b200-tp8-ep8-dpatrue-c128/                     final agg json
```

Decisions taken: **arm 3 at CONC=128** is the profiling reference, and **the CI
comparison is deferred** — report absolute numbers now, diff against CI later
when a CI json is available (§8).

**Monitoring:** `watch_arm.sh` (CUDA port of the parent skill's script) runs in
the background, log `/workspace/agentx/logs/watch_arm3_c128.log`; it emits only
on server-ready, phase change, ~20 min stall, server death and completion, plus
a 30 min heartbeat, and exits once the result json appears. Stop it with
`kill 32346`.

**PID gotcha on this node:** `$!` after `nohup ./script.sh &` returns the
*wrapper* shell, not the script. Always resolve the real one with
`pgrep -af '<script>.sh'` before quoting or killing a PID — `wait_and_run.sh`
and `watch_arm.sh` both hit this.

Ready: `utils/aiperf` submodule initialised at the pinned `754356e9`; isolated
AIPerf venv installed (CPython 3.11.16, 131 packages, aiperf 0.12.0, scenario
`inferencex-agentx-mvp` registered); trace corpus downloaded (1.8 GB, 393
traces); launcher flags verified against the image's SGLang.

Blocking: the B200 recipe needs **`DeepSeek-V4-Pro-0813`**, which is still
downloading into `/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813` (35/~64 shards,
509 GB at 2026-09-15 09:58, `hf download` PID 3764, cwd `/shared_nfs`).
`model.safetensors.index.json` has not landed yet.

**Next:** watch the queued run, one channel only:

```bash
R=/workspace/agentx/results/b200-tp8-ep8-dpatrue-c128
rg 'CKPT ready|ready to roll|Traceback|ERROR|Successful requests|Total token throughput' \
   /workspace/agentx/logs/arm3_c128.log "$R"/server.log 2>/dev/null | tail -20 | cut -c1-200
```

**Pass criteria:** server reaches `/health` 200, the router comes up on 8888
with the backend on 8889, aiperf replay completes, and `$R` holds
`$RESULT_FILENAME.json` plus `server.log`, `router.log`,
`sglang_command.txt`. Then read the score with
`python3 /workspace/agentx/show_result.py "$R"/*.json` and **run the same arm a
second time** for the noise floor.

## 1. The launcher

B200 runs exactly one script:

```
benchmarks/single_node/agentic/dsv4_fp4_b200_sglang_mtp.sh
```

Not the `mi355x` variants the parent skill drives. Two behaviours worth knowing
before reading its stdout:

- It **auto-corrects `INFMAX_CONTAINER_WORKSPACE`**: if
  `$INFMAX_CONTAINER_WORKSPACE/utils/aiperf` does not exist it falls back to the
  repo root computed from the script path, and rewrites a `/workspace/*`
  `RESULT_DIR` to sit under that root. It also builds a throwaway
  `/tmp/inferencex-agentic-venv` and puts it on `PATH` when the repo is not at
  `/workspace`, to keep AIPerf's Transformers-main dependency away from the
  image's pinned Transformers.
- It **re-downloads the model only when `MODEL_PATH` is missing or empty**
  (lines 35-38). A half-downloaded directory is accepted and then fails deep in
  weight loading. Hence `/workspace/agentx/check_ckpt.sh`, which both wrapper
  scripts call before launch.

## 2. Canonical arms (configs/nvidia-master.yaml:931)

Config key `dsv4-fp4-b200-sglang-agentic-hicache-mtp`, model
`deepseek-ai/DeepSeek-V4-Pro-0813`, `model-prefix: dsv4`,
`runner: cluster:b200-nscale`, `dram-utilization: 0.80`, image
`lmsysorg/sglang:nightly-dev-20260827-20621aa1`. Three search-space arms, all
`spec-decoding: draft_model`:

| arm | shape | conc-list |
|---|---|---|
| 1 | tp8, kv-offloading `none` | 1, 2, 3, 4, 5 |
| 2 | tp8, kv-offloading `dram` + hicache | 8, 10, 16 |
| 3 | tp8, ep8, dp-attn, `dram` + hicache, sglang-router 0.3.2 | 64, 96, 128, 160 |

`dram-utilization: 0.80` is what the matrix generator multiplies by node RAM to
get `TOTAL_CPU_DRAM_GB`; on this node that is 0.80 × 2015 ≈ **1612**.

Note the concurrency scale: arm 1 tops out at 5 and arm 3 starts at 64. AgentX
concurrency counts **live session trees**, not requests, and the launcher sets
`MAX_RUNNING_REQUESTS=2*CONC` to leave room for subagent fan-out.

## 3. What the launcher does differently per arm

With `DP_ATTENTION=true` it starts **`sglang_router` on `$PORT`** and moves the
backend to `$PORT+1` (`--dp-aware`, `cache_aware` policy, correlation-id
routing), adds `--dp $TP --enable-dp-attention --enable-dp-lm-head`,
`--moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe`, drops
`MEM_FRACTION_STATIC` to 0.88 for the FP4 indexer's workspace, sets
`CHUNKED_PREFILL_SIZE=6144*TP` (SGLang divides this by dp_size, so it is 6144
per rank) and pins `CUDA_GRAPH_MAX_BS=32`.

With `DP_ATTENTION=false` it is a plain TP8 server:
`--moe-runner-backend flashinfer_mxfp4`, `MEM_FRACTION_STATIC=0.90`,
`CHUNKED_PREFILL_SIZE=8192`, `CUDA_GRAPH_MAX_BS=2*CONC`, no router.

HiCache on DSv4 has **no `--hicache-size`**; capacity is a host/device token
ratio. The launcher caps it: 8 when DP-attention is on, 2.75 otherwise, and
exits if `HICACHE_RATIO` exceeds that cap.

Speculative decoding is DSPARK, not MTP: `--speculative-algorithm DSPARK
--speculative-dspark-block-size 6 --speculative-num-steps 1
--speculative-eagle-topk 1 --speculative-num-draft-tokens 7`. Unless
`EVAL_ONLY=true`, throughput runs also export `SGLANG_SIMULATE_ACC_LEN=3.77`
with `SGLANG_SIMULATE_ACC_METHOD=match-expected` — i.e. **the throughput arm
uses a committed golden synthetic acceptance length**, so acceptance is not a
measured quantity there. Real target verification only happens in the eval
path.

## 4. Two checkpoints on this node — do not mix them up

| path | state | used by |
|---|---|---|
| `/shared_nfs/deepseek-ai/DeepSeek-V4-Pro` | complete: 64/64 shards, 864,704,792,696 B | `dsv4-fp4-b200-vllm` config |
| `/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813` | downloading (35 shards @ 09:58) | **the sglang agentic recipe** |

`config.json` reports `quant_method: "fp8"` on the V4-Pro checkpoint: routed
experts are FP4, the shared expert is stored FP8. That is why
`--enforce-shared-experts-fusion` is a *precision* change on this family, not
just a throughput knob — the parent skill has the details.

## 5. Software actually present (differs from the config's pinned image)

The config pins `lmsysorg/sglang:nightly-dev-20260827-20621aa1`. We are **not**
in that image: SGLang here is **0.5.19**, editable at `/sgl-workspace/sglang`.
Its `--help` does accept `--speculative-algorithm DSPARK`,
`--enable-w4a4-mxfp4-megamoe` and `--enable-deepseek-v4-fp4-indexer`, so the
launcher's flags resolve — but any number produced here is not image-matched to
CI, and that belongs in any report.

System `python3` is 3.12.3. The AIPerf venv is a separate 3.11.16 on purpose:
`install_agentic_deps` refuses to share site-packages with the server, because
installing AIPerf into the server interpreter can upgrade
FastAPI/Starlette/transformers under a running server.

## 6. Environment layout

```
/workspace/agentx/
  InferenceX/            repo, main @ ca108e273 (clean), utils/aiperf @ 754356e9
  agentx_env.sh          source before any launcher; sets INFMAX_CONTAINER_WORKSPACE
  agentx_smoke_b200.sh   canonical arm 1, CONC=2, DURATION=300
  agentx_run_b200.sh     canonical arm 3 by default, all vars overridable
  check_ckpt.sh          shard-count / *.incomplete guard
  runtime/               AIPERF_RUNTIME_DIR: venv/ uv/ uv-cache/
  hf_cache/              HF_HOME (this node has no /shared_nfs/hf_cache)
  results/               RESULT_DIR root
  logs/                  install.log, submodule.log, trace_download.log
```

`AIPERF_RUNTIME_DIR` is pinned in `agentx_env.sh` because the library default
ends in `$$` (shell PID), which makes `install_agentic_deps` rebuild the whole
venv on every invocation.

`MODEL_PREFIX=dsv4` is load-bearing: it selects the uncapped
`semianalysisai/cc-traces-weka-062126` corpus. Unset, it silently falls through
to the 256k-capped variant, and `check_env_vars` does not check it.

**`RESULT_DIR` gets rewritten, `AGENTIC_OUTPUT_DIR` does not.** The launcher
rewrites any `/workspace/*` `RESULT_DIR` to `$INFMAX_CONTAINER_WORKSPACE/<rest>`
(lines 23-25) but leaves `AGENTIC_OUTPUT_DIR` untouched, so the naive setting
`AGENTIC_OUTPUT_DIR="$RESULT_DIR"` scatters `server.log` and the agg json into
two different trees. Both wrapper scripts now pre-apply the same rewrite to
`AGENTIC_OUTPUT_DIR` and echo the effective path at startup.

`RESULT_FILENAME` is parsed, not just used as a name: `benchmark_lib.sh:2088`
extracts precision and framework from the two fields before `_tp`, so keep the
`dsv4_fp4_sglang_tp...` prefix shape.

## 7. Traps

**The launcher orphans a whole process family, and freeing VRAM is not proof of
a clean node.** When the script exits after writing results it leaves behind
`launch_server`, `sglang::router`, `data_parallel_controller`, `detokenizer`,
one `tokenizer_worker` per rank, and the 8 GPU schedulers — 9+ processes.
Use `/workspace/agentx/free_gpus.sh`, which kills the family and then verifies
processes, VRAM **and** ports.

This bit once, expensively: killing only the PIDs `nvidia-smi` reports dropped
VRAM to 4 MiB while the parent kept holding the dist-init port, and the next run
died 30 s into startup with

```
ValueError: port_base at 10889 is not available in 30 seconds
```

`kill <launch_server pid>` alone is also not enough — the schedulers survive it.

**Verify the outcome, not the artefact.** `sglang_command.txt` is written
*before* the server starts, so its existence (and correct flags) says nothing
about whether the server came up. The failed trace run above had a perfect
`sglang_command.txt` and no server. Check rising VRAM, or `SERVER READY` from
the watcher, before believing a run is alive.

**`watch_arm.sh`: test each result tree separately.** The completion check has
now been broken twice, both times silently — the watcher just heartbeats forever
and the run looks unfinished. The second time was `ls "$A"/*.json "$B"/*.json`,
which returns non-zero when *either* path is missing, and the agg json only ever
lands in one of the two trees. Use `compgen -G` per directory. The general
lesson matches the parent skill's: a stall detector's completion path is the
part that breaks, and it breaks quietly.

**`hw`, `image` and `recipe_fingerprint` come out empty on a local run.** They
are populated from `RUNNER_TYPE`, `IMAGE` and `RECIPE_FINGERPRINT`, which only
the CI matrix sets, so the first run's json cannot be told apart from any other
run. `agentx_run_b200.sh` now fills all three (local sglang version + commit,
launcher + repo commit). The first reference run predates that and has them
blank — it was SGLang 0.5.19 at `/sgl-workspace/sglang` `0bcd822`, repo
`ca108e273`.

### Carried over from the parent skill

- **A dead server keeps returning 200.** If the schedulers die, the HTTP layer
  stays up and the agentic warmup barrier waits indefinitely with `errors=0`.
  `errors=0` plus a frozen `returned=` is a corpse, not health. Crashes land in
  `$RESULT_DIR/server.log`, not launcher stdout — watch both.
- Keep full logs on disk, grep markers only:
  `rg 'ready to roll|Initialization failed|Traceback|Successful requests|Total token throughput' <log> | tail -20 | cut -c1-200`.
- One channel per log: never tail and poll the same file.

**Not applicable here** (ROCm-only findings in `../SKILL.md`): the `EP_SIZE=1`
vs EP8 OOM trap, aiter / MegaMoEv2-on-ROCm work, the topk_v2 (#36684) patching
saga, and every measured tok/s number in its CONTINUE HERE section.

## 8. Parity check against CI

**The score.** `process_agentic_result` writes
`$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json`; the headline number is
`request_metrics.throughput.per_gpu.output_tput_tps`, with
`total_tput_tps`, `latency.ttft`, `latency.itl` p90, `latency.intvty` p90 and
`request_metrics.cache.theoretical_cache_hit_rate` as the secondaries that tell
you whether a throughput delta is real or a cache/ISL artefact. Extract them
with:

```bash
python3 /workspace/agentx/show_result.py <ci_agg.json> <our_agg.json>
```

which prints our run as percentages against the first file.

**Where CI's numbers live, and why they are not here.** Each single-node job
uploads `bmk_${RESULT_FILENAME}` (containing `agg_${RESULT_FILENAME}_*.json`);
`collect-results.yml` then downloads `bmk_*`, runs
`python3 -m infx.results.collect_results results/ bmk` and publishes
`results_bmk` (see `docs/recovery-results-procedures.md` §"Throughput
results"). Nothing of that is committed — `perf-changelog.yaml` logs *changes*,
not results. **`gh` is not installed on this node**, so the CI reference json
has to be supplied: either install/authenticate `gh` for
`SemiAnalysisAI/InferenceX`, or drop the artifact json somewhere under
`/workspace/agentx/` and point `show_result.py` at it.

**Comparability checklist** — get these wrong and the parity question is
unanswerable regardless of the number:

- Arm shape must be one of the three conc-list entries in §2, at a concurrency
  CI actually ran. Interpolated concurrencies have no counterpart to compare to.
- `MODEL_PREFIX=dsv4` (uncapped corpus) and the full 393-trace replay.
- `TOTAL_CPU_DRAM_GB` from the config's `dram-utilization: 0.80` — CI derives it
  from its own runner's RAM, so if CI's `b200-nscale` host has a different RAM
  size, the HiCache host tier is a different size and the dram arms are not
  strictly comparable. Worth checking against the CI json's metadata.
- Golden AL matches: `SGLANG_SIMULATE_ACC_LEN=3.77` equals
  `golden_al_distribution/dsv4-pro-0813-dspark.yaml` K=6 (3.77), so the
  simulated acceptance length is identical. Good — but it also means the
  throughput arm's acceptance is *assumed*, not measured, on both sides.
- **Biggest parity risk: the image.** CI pins
  `lmsysorg/sglang:nightly-dev-20260827-20621aa1`; this container is SGLang
  0.5.19 editable. A delta may be the image, not the hardware or the config.
  State the image on both sides in any comparison.

**Run it twice.** There is no measured noise floor for AgentX on this node yet
(the ROCm side saw ~5.7 % replicate spread on throughput and ~7 % on ITL p90).
A profiling baseline built on n=1 cannot tell a real regression from spread
later, so the reference arm should get one replicate before it is treated as
the baseline.

## 9. ITL / interactivity gap investigation vs MI355X

The question: at c128, MI355X shows ITL p90 50.6 ms (intvty 19.8) while B200
shows 20.7 ms (intvty 48.3). Interactivity here is just per-user token rate —
`1000 / ITL_ms` reproduces both numbers — so this is entirely an ITL question.

**ITL is not a kernel number.** From the scheduler log,

```
ITL       = step_time / accept_len
step_time = batch_per_rank * accept_len / gen_throughput_per_rank
```

so a platform can lose on ITL three different ways: bigger batch per step, loss
of cuda-graph replay, or genuinely slower kernels at matched batch. They need
different fixes. `decode_stats.py` separates them from `server.log` alone, with
no profiler:

```bash
python3 /workspace/agentx/decode_stats.py <b200 server.log> <mi355x server.log>
```

**B200 c128 reference** (steady-state middle half, 5,092 decode steps):

| quantity | p50 | p90 |
|---|---|---|
| running-req / DP rank | 9 | 12 |
| accept len | 3.77 | 3.85 |
| gen tput / rank | 436 tok/s | 711 |
| implied step time | 71.8 ms | 98.8 ms |
| implied ITL | 19.1 ms | 26.2 ms |
| cuda graph replay | **100 % of steps** | |
| KV pool usage | 0.62 | 0.95 |

The implied ITL reproduces the aiperf-reported 20.7 ms, so the decomposition is
sound and can be applied to the other platform's log directly.

### The scheduling confound found on 2026-09-15

`prefill_decode_interval` is "the number of decode rounds to run after a prefill
batch before scheduling the next prefill" (`server_args.py:718`). The two
launchers do **not** agree on it:

| launcher | value |
|---|---|
| `dsv4_fp4_b200_sglang_mtp.sh` | hardcoded **24** (20 only at `CONC=160`) |
| `dsv4_fp4_mi355x_sglang_mtp.sh` | `${PREFILL_DECODE_INTERVAL:-10}` → **10** |

So MI355X interrupts its decode stream with prefill **2.4× more often** than
B200 does, and every cross-platform ITL number on file carries that difference.
With ISL ~105-114k, a single prefill is many chunked iterations, each one a gap
in every running request's token stream — exactly where an ITL p90 tail comes
from. This has to be ruled out before any kernel-level claim.

### A/B RESULT: `prefill_decode_interval` 24 → 10 (B200 c128, 2026-09-15)

Single variable, everything else byte-identical (`--dp 8 --ep-size 8
--mem-fraction-static 0.88` verified in both `sglang_command.txt`).

| metric | pdi=24 (base) | pdi=10 | delta |
|---|---|---|---|
| ITL p90 | 20.7 ms | **32.1 ms** | **+55.2 %** |
| interactivity p90 | 48.31 | 31.13 | −35.6 % |
| TTFT p50 | 4.59 s | **2.66 s** | **−42.0 %** |
| output tok/s/GPU | 403.6 | 356.2 | −11.7 % |
| total tok/s/GPU | 46,170 | 40,400 | −12.5 % |
| cache hit | 0.9635 | 0.9617 | flat |
| ISL mean | 114,401 | 109,800 | −4.0 % |
| avg GPU power | 5,723 W | 5,150 W | −10.0 % |

Direction as predicted, and the magnitude is far outside anything noise could
explain. **`prefill_decode_interval` is a direct ITL ↔ TTFT trade knob**, not a
second-order tuning parameter.

**The mechanism, from the `step_time(batch)` curves** (`decode_stats.py`):

| running-req/rank | step ms @ pdi24 | step ms @ pdi10 | offset |
|---|---|---|---|
| 8 | 70.0 | 101.1 | +31.1 |
| 9 | 71.9 | 99.5 | +27.6 |
| 12 | 76.4 | 102.5 | +26.1 |
| 16 | 84.8 | 109.6 | +24.8 |

The two curves are **offset by a near-constant ~26-28 ms, not steeper** — the
signature of a fixed per-step cost (prefill stealing the scheduler slot),
not of more work per step and not of slower kernels. Batch composition also
shifted (`running-req/rank` p50 9 → 11, KV usage 0.62 → 0.78) because a shorter
interval admits requests faster, so the aggregate ITL delta mixes both effects;
the curve is what separates them.

**Cross-check on the amortised-prefill model.** If a prefill batch costs `P` and
is admitted every `N` decode rounds, the per-step surcharge is `P/N`. The
observed offset then implies `P × (1/10 − 1/24) ≈ 27 ms`, so `P ≈ 460 ms` — the
right order for one ~49k-token chunked prefill (6144 × 8 ranks) on this
hardware. The same model accounts for the baseline's own step: 460/24 ≈ 19 ms of
its 72 ms, and 460/10 ≈ 46 ms of pdi10's 100 ms.

**This also shows the decode step is not context-bound at these batches.** The
baseline curve runs 68 ms at batch=1 and only 85 ms at batch=16 — nearly flat,
where KV-read cost would scale with batch × context. A step dominated by a
batch-independent cost points at MoE weight traffic plus fixed per-step
overheads (DSPARK draft forwards, DP sync, graph replay), not at the 110k-token
KV read. Any kernel-level work on ITL should target that, and any "memory-bound"
claim needs the achieved-bandwidth number on *weights*, not on KV.

### Decode trace, B200 c128 at pdi=10 (matched to MI355X)

`traces/b200-tp8-ep8-dpatrue-c128-pdi10trace2/`, 8 files, one per DP rank, all
named `*-DECODE.trace.json.gz` — `profile_by_stage` worked, so prefill is
excluded. Captured 8 scheduler steps = 61 model forwards (≈7.6 per step, which
is DSPARK block-size 6 drafting plus verify). Summarise with
`analysis/trace_summary.py`.

**[RETRACTED] These files contain no decode steps at all.** `profile_by_stage`
named every file `-DECODE` while each one holds exactly **one `EXTEND` step**.
An earlier version of this section reported a "decode" kernel breakdown
(MoE 36 %, GEMM 21 %, attention 13.5 %) and a "63 ms/step decode" figure; both
described a chunked extend and are withdrawn. The filename is not evidence —
check the `step[...]` annotations.

What the 8 files actually captured, one step per rank:

| rank | own tokens | step GPU ms | `mega_moe` ms | `sparse_attn` ms | `gemm_1d1d` ms |
|---|---|---|---|---|---|
| TP-0 | 6144 | 446 | **163** | 60 | 93 |
| TP-6 | 5877 | 445 | **158** | 62 | 89 |
| TP-4 | 6144 | 730 | **188** | 142 | 93 |
| TP-5 | 2584 | 443 | **292** | 32 | 40 |
| TP-1 | 2385 | 426 | **311** | 13 | 37 |
| TP-7 | 1762 | 441 | **333** | 22 | 28 |
| TP-2 | 1601 | 438 | **336** | 18 | 27 |
| TP-3 | 541 | 442 | **391** | 8 | 12 |

### Finding 1 — a prefill anywhere freezes the whole DP group

Own token counts span 11× (541 → 6144) yet **every rank's step takes ~440 ms**.
DP-attention steps are group-synchronous, so each rank's step time is the
*slowest* rank's work. One rank's 6144-token chunk stalls all eight.

This measures the amortised-prefill model directly instead of inferring it:
`P ≈ 440-450 ms` observed, and

```
450 ms × (1/10 − 1/24) = 26.2 ms
```

against the 26-28 ms per-step offset measured from the logs. The earlier
back-derived `P ≈ 460 ms` was right, by a route that turned out to be wrong.

**Lever this exposes:** the barrier is sized by the slowest rank's chunk, so
per-rank prefill imbalance is paid by everyone. `CHUNKED_PREFILL_SIZE` (49152
global ÷ dp8 = 6144/rank here) trades barrier length against barrier count, and
balancing chunks across ranks is worth more than shaving the MoE kernel.

### Finding 2 — `mega_moe_impl` absorbs the wait, so single-rank kernel rankings lie

`sparse_attn` and `gemm_1d1d` rise monotonically with a rank's own tokens — real
compute. `mega_moe` moves the **opposite** way (391 ms at 541 tokens, 158 ms at
5877) and per-call it is 3.20 ms on the lightest rank against 1.29-1.33 ms on
the heaviest. The light ranks are blocking *inside* the fused megamoe a2a kernel
waiting for peers.

So "MoE is 90 % of the step" — true on TP-3 — is a **barrier, not a bottleneck**.
Reading one rank's trace here produces exactly the wrong optimisation target.
For cross-platform kernel comparison use `sparse_attn` + `gemm_1d1d` per token
on the *heaviest* rank, or compare ranks against each other; never a single
rank's top-kernel list.

Note also TP-4: same 6144 tokens as TP-0 but 142 ms of attention against 60 ms,
and a 730 ms step — the straggler that set the barrier for everyone. Worth
finding out what made that rank's attention 2.4× more expensive at equal token
count (likely context length of the sequences it held).

**Do not read the 114 ms of `Synchronize` as overhead either** — that is the CPU
blocking on the GPU, i.e. idle, not cost.

### Decode-side trace, 40 steps (`-pdi10trace40`, 68 MB, 8 ranks)

Step inventory per rank: `TARGET_VERIFY` × 737-2680, `IDLE` × 1-4,
`EXTEND` × 1-4. There is no `DECODE` annotation — with DSPARK the decode path
appears as **`TARGET_VERIFY`** (the target model verifying draft tokens).

`TARGET_VERIFY`, per rank (TP-0; all eight ranks agree):

| quantity | value |
|---|---|
| steps in window | **62** |
| step wall ms | p50 **17.2**, mean 15.9, p90 30.3, max 32.1 |
| all steps in the window are `bs=7` | |
| GPU kernel ms per step (summed, **overlapping**) | p50 ≈ 45 |

Typical (**p50**) verify step, per role — use these, not the means:

| role | p50 ms/step | mean | min | max |
|---|---|---|---|---|
| gemm | **21.1** | 11.2 | 0.67 | 22.1 |
| moe | **13.7** | 7.9 | 0.51 | 17.5 |
| attn | **5.2** | 2.8 | 0.19 | 5.7 |
| other | 3.4 | 2.2 | 0.77 | 3.8 |
| quant | 1.3 | 0.8 | 0.12 | 1.6 |
| comm | 0.6 | 0.4 | 0.03 | 0.70 |

~45 ms of overlapping kernel time inside a 17.2 ms wall step ⇒ ~2.6× stream
concurrency.

**[CORRECTED again] Report p50, not the mean.** The per-step distribution is
bimodal: most steps sit at p50 ≈ max, a minority are near-empty, and the mean is
dragged down by the latter. An earlier version of this table quoted the means
(moe 7.93 ms) which conflicted with a Perfetto measurement of 16.18 ms of
`mega_moe` in one step — that step was simply a normal one, between p50 13.70
and max 17.45. `trace_summary.py` now always prints mean/min/p50/max together.

Also retracted: the claim that a 34× per-step spread in `moe` at fixed `bs=7`
proved barrier waiting. `attn` spreads 40× and `gemm` 34× as well — every role
moves together, which is the bimodality above, not a MoE-specific wait.

**[CORRECTED] An earlier version of this table said 2077 steps of 2.89 ms and
"GPU busy ≈ 25 %". Both were wrong**, caught by the observation that a single
target-verify step visibly takes 26-28 ms:

- `gpu_user_annotation` `step[...]` events **nest and overlap**: one ~30 ms
  TARGET_VERIFY step also emits hundreds of ~0.4 ms sub-slices. Counting them
  all gave 2077 pseudo-steps of mean 2.89 ms for what were 62 real steps, and
  made kernel→step attribution arbitrary. `trace_common.load()` now keeps only
  the outermost annotations; the CPU-side `user_annotation` count independently
  agreed at 62.
- Summed kernel duration over wall time is **not** a busy fraction — kernels
  overlap across streams, and the honest figure here is 25.7 ms of kernel inside
  a 17 ms step, i.e. **161 %**. The GPU is heavily occupied during decode, the
  opposite of the retracted "overhead-limited" reading.

What survives: **decode is GEMM-dominated (43.6 %) plus MoE (30.8 %), with
attention only 11 %.** Long context stays cheap because the FP4 indexer's sparse
attention reads top-k blocks rather than the full KV. Role *percentages* were
unaffected by the step-count error, since they are ratios.

Also still standing: **decode is not barrier-limited.** `comm` is 1.4 % and all
eight ranks agree on step time, against the 11× imbalance seen during extend.
The DP synchronisation tax lives in prefill, not decode.

**This capture killed the arm.** Seconds after the traces were written the
server died with

```
Assertion failed: deep_gemm/include/deep_gemm/comm/barrier.cuh:45,
condition: false and "Grid sync timeout"
```

repeated on every rank. The fused megamoe kernel holds a **device-side grid sync
barrier across ranks**, and profiler perturbation at `NUM_STEPS=40` with
`record_shapes` pushed it past its timeout. The HTTP layer stayed up, so the
symptom was the familiar one: progress frozen at `done=295` with `errors=0`, and
the watcher's STALL was the only signal. No agg json was produced for this arm.

Two consequences:

- **`NUM_STEPS` must stay small — 8 is known good, 40 is fatal.** It is now the
  default again in `trace_arm_b200.sh`. For a longer window, cut instrumentation
  instead: `activities=["GPU"]` only and `record_shapes=false`.
- It **independently confirms** that `mega_moe_impl` contains a cross-rank
  barrier, which is exactly why its time inflates on lightly loaded ranks.
- **The decode numbers above need re-validation at `NUM_STEPS=8`.** They come
  from a capture whose run died moments later, so the last steps may have been
  taken while the grid sync was already degrading.

**Caveat that blocks absolute claims:** `TARGET_VERIFY` is 2.89 ms while the
log-implied decode step is ~100 ms, and the window holds 2077 of them — so one
annotation is *not* one scheduler step (likely per-request or per-TBO
microbatch). Until that granularity is pinned, only the **role mix** and the
**GPU busy fraction** are safe to compare across platforms; absolute per-step
times are not.

Tooling note: `trace_ranks.py` first reported `TARGET_VERIFY` compute as 0.0 ms.
The tool was right and the *display* was wrong — it printed medians, and with
2077 highly skewed steps (p50 0.4 ms, mean 2.9 ms) the median hid the whole
workload. It now reports totals and means only.

### Still outstanding

- **No decode-step trace exists yet.** Every decode-side claim in this document
  still rests on the scheduler logs, not on a trace. Re-capture with
  `NUM_STEPS=40` (now the default in `trace_arm_b200.sh`) so the window spans
  both extends and decodes, and let `analysis/trace_summary.py` split them by
  annotation.
- Achieved bandwidth/FLOPs vs peak for the real compute kernels. `record_shapes`
  is on; the one op that resolved was `sglang::deep_gemm_fp8_fp8_bf16_nt` with
  dims `[[6144,7168],[6144,14],[2048,7168],[2048,14],[6144,2048]]`. Until
  per-call bytes are pinned, no "X-bound" claim is measured.

### Rough spread, from the trace run as a second pdi=10 point

The trace run (`-pdi10trace2`, DURATION 1800, profiler armed) vs the full pdi=10
run: **output tok/s/GPU 355.2 vs 356.2, −0.29 %** — that metric reproduced to
three parts in a thousand despite half the duration and profiler perturbation.

It is *not* a clean replicate: ISL −17.6 % (a shorter run samples a different
trace mix), duration halved, and the cache tier split moved substantially
(GPU-tier hit 0.386 vs 0.505, host-tier 0.555 vs 0.446) because the prefix cache
had less time to warm. ITL p90 came in +9.15 %.

Useful bound rather than a noise floor: **run-to-run ITL p90 movement under
non-matched conditions is ~9 %**, and the pdi 24→10 effect was +55 %, six times
larger. The pdi conclusion needs no replicate to defend. Conversely, treat any
ITL difference under ~10 % as unresolved until conditions are matched.

### A/B RESULT: multi-stream overlap on vs off (B200 c128, pdi=10)

Second config asymmetry with MI355X, and a structural one rather than a tunable:

- `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` defaults to **True**, and
  `deepseek_v4.py:853` enables `alt_streams` only under
  `(_is_cuda and that flag)` or `(_is_npu and ...)` — **there is no ROCm
  branch**.
- `arg_groups/model_hook.py` explicitly does
  `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)` in the `is_hip()` branch for
  `DeepseekV4ForCausalLM`.
- A separate `SGLANG_ROCM_USE_MULTI_STREAM` (default **False**) exists but is
  used at different sites (`deepseek_v4.py:1804`, `deepseek_v2.py:2672`).

So B200 overlaps kernels across streams by default and MI355X cannot, which
makes per-kernel wall durations non-comparable across the two (overlapping
kernels contend for SMs: each takes longer, the step takes less).

Measured directly, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`, everything else
identical, both 3630 s:

| metric | multi-stream | single-stream | delta |
|---|---|---|---|
| output tok/s/GPU | 356.2 | 345.2 | **−3.07 %** |
| total tok/s/GPU | 40,400 | 39,210 | −2.96 % |
| ITL p90 | 32.1 ms | 34.0 ms | **+5.88 %** |
| interactivity p90 | 31.13 | 29.40 | −5.56 % |
| TTFT p50 | 2.66 s | 3.02 s | +13.4 % |
| cache hit / ISL | 0.9617 / 114k | 0.9612 / 108k | flat / −1.4 % |

At matched batch and matched KV usage (both p50 batch 11, KV 0.78), the
`step_time(batch)` curves are offset by a consistent **4-5 %**:

| batch/rank | multi | single |
|---|---|---|
| 6 | 98.4 | 102.1 |
| 8 | 101.1 | 105.4 |
| 10 | 100.8 | 105.9 |
| 12 | 102.5 | 107.7 |

**Verdict: multi-stream overlap is worth ~4-5 % of decode step time (~3 %
throughput, ~6 % ITL) — real, reproducible, and far too small to explain the
1.58× config-matched gap with MI355X.** Kernel-level comparison across the two
platforms can proceed with a ~5 % caveat instead of needing the streams
normalised away.

### Trace comparison, multi vs single stream (identical capture settings)

`traces/b200-tp8-ep8-dpatrue-c128-1stream_trace40/` (64 MB) against
`...-pdi10trace40/` (68 MB), both `num_steps=40`, `record_shapes=true`,
`activities=[CPU,GPU]` — capture settings were deliberately kept identical so
profiler perturbation cancels.

| `TARGET_VERIFY` per step | multi | single |
|---|---|---|
| compute (summed kernel) | 15.10-15.24 ms | 14.89-15.11 ms |
| barrier (moe+comm) | 8.02-8.71 ms | 7.79-8.80 ms |
| gemm | 10.97-11.20 ms | 9.40-10.98 ms |
| attn | 2.83-3.06 ms | 2.56-3.71 ms |
| step wall (mean) | 15.88-16.37 ms | 17.53-17.63 ms |
| streams active inside verify steps | **132** | **4** |
| summed kernel ÷ wall | 1.49 | 1.33 |

**The kernel work is unchanged; only the wall time moves.** compute is 15.1 ms
either way and barrier ~8.3 ms either way, while the step gets ~10 % longer and
the stream count collapses 132 → 4. Overlap fills gaps inside the step; it does
**not** inflate individual kernel durations through SM contention.

**This answers the comparability worry directly: per-kernel times on B200 ARE
comparable to MI355X without normalising streams away.** Only step wall time
needs the ~4-5 % caveat.

Three estimates of the effect size, in descending order of trust: **+4-5 %**
(logs, matched batch and matched KV usage, thousands of steps, no profiler),
+9.4 % (trace step means, 42-80 steps), +15 % (trace step p50). Quote the log
number; the trace establishes the mechanism, not the magnitude.

A prediction that was wrong, recorded because it was diagnostic: single-stream
was expected to show *lower* per-role kernel times (no SM contention). It did
not — `moe` p50 even rose slightly (13.70 → 15.05 ms). Kernel cost is not
contention-driven here.

Process note: a mid-flight read of this run showed single-stream *faster* by
10-26 % at matched batch. That was a run-phase artefact — early in the profiling
phase the KV pool is emptier and contexts are shorter, so steps are cheaper. It
was flagged as an unresolved confound at the time and disappeared on the
complete data. **Do not compare a partial run against a complete one**, even at
matched batch; match KV pool usage too, which `decode_stats.py` reports.

### CROSS-PLATFORM, both sides now measured at both pdi settings (2026-09-16)

MI355X was re-run at `prefill_decode_interval=24` to match B200.

| arm | pdi | tok/s/chip | intvty p90 | ITL p90 | TTFT avg | TTFT p50 | cache hit | ISL |
|---|---|---|---|---|---|---|---|---|
| MI355X | 10 | 34,713 | 19.8 | 50.6 ms | 5.64 s | 2.52 s | 95.1 % | 105,652 |
| MI355X | **24** | 36,995 | 29.5 | **33.9 ms** | 11.07 s | 4.01 s | 95.6 % | ? |
| B200 multi-stream | 10 | 40,401 | 31.1 | 32.1 ms | 9.87 s | 2.66 s | 96.2 % | 109,796 |
| B200 single-stream | 10 | 39,207 | 29.4 | 34.0 ms | 11.66 s | 3.02 s | 96.1 % | 108,273 |
| B200 multi-stream | **24** | 46,173 | 48.3 | **20.7 ms** | 12.33 s | 4.59 s | 96.4 % | 114,401 |

**The platform ITL gap is ~1.6× at both settings:** 33.9/20.7 = **1.64×** at
pdi=24, 50.6/32.1 = **1.58×** at pdi=10. So the `prefill_decode_interval`
asymmetry never was the *cause* of the gap — it inflated a real 1.6× into an
apparent 2.5× (50.6 vs 20.7). The residual 1.6× is what a trace has to explain.

**The pdi knob behaves the same on both platforms**, which validates the
amortised-prefill model as platform-independent rather than a CUDA artefact:
10→24 improves ITL by 33 % on MI355X and 35.5 % on B200, and costs TTFT p50
+59 % and +72.6 % respectively.

**Throughput gap widens at pdi=24**: B200 leads by 16.4 % at pdi=10 and 24.8 %
at pdi=24, because B200 gains more from the wider interval (+14.3 % vs +6.6 %).
That asymmetry is itself a lead — plausibly related to B200 having an active
HiCache host tier, so the cost structure of being interrupted differs.

ISL/OSL now recorded on both sides, so the throughput comparison is firm:

| arm | pdi | ISL | OSL | tok/s/chip | input | **output** |
|---|---|---|---|---|---|---|
| MI355X | 10 | 105,652 | 929 | 34,713 | 34,410 | **302** |
| MI355X | 24 | 108,371 | 967 | 36,995 | 36,667 | **327** |
| B200 | 10 | 109,796 | 977 | 40,401 | 40,045 | **356** |
| B200 | 24 | 114,401 | 1,009 | 46,173 | 45,769 | **404** |

The workload shape matches (ISL/OSL ≈ 112-113 on both) and **the ISL difference
runs against B200** — it carries 5.6 % more input and 4.3 % more output at
pdi=24 and still wins, so its advantage is understated, not inflated.

On output tok/s/chip, the metric least sensitive to cache-hit differences:
B200 leads **+23.4 %** at pdi=24 and **+17.9 %** at pdi=10.

**The tension worth chasing: ITL gap 1.64× but output-throughput gap only
1.23×**, both at pdi=24. Throughput is batch × per-step rate while ITL is
per-request latency, so the only way both hold is that MI355X runs a larger
in-flight batch to compensate. That predicts
`running-req/rank ≈ 9 × 1.64/1.23 ≈ 12` on MI355X against B200's measured 9 at
pdi=24 — falsifiable from the MI355X `server.log` alone with
`decode_stats.py`, no trace needed. If MI355X also reports ~9, the reasoning is
missing something and `accept len` or the step-granularity definition is the
next place to look.

### Consequence for the MI355X comparison

MI355X runs `prefill_decode_interval` = **10**; the B200 baseline ran **24**. The
honest, config-matched comparison is therefore:

| | ITL p90 |
|---|---|
| MI355X c128 (pdi 10) | 50.6 ms |
| B200 c128 **pdi 10** | 32.1 ms |
| B200 c128 pdi 24 | 20.7 ms |

**About 38 % of the apparent 2.5× gap was a launcher config difference, not
hardware.** The real, config-matched gap is **1.58×**, and that is the number to
investigate with a trace. Quoting 2.5× would have sent the kernel work chasing
30 % of a gap that a one-line config change explains.

**Untested prediction worth one run:** the model says ITL keeps improving as the
interval rises — pdi=48 would drop the surcharge to ~10 ms and put step time
near 62 ms (ITL ≈ 16.5 ms), at further TTFT cost. That is also a plausible way to
beat the CI reference on interactivity, if TTFT has slack.

The one-line change that made this testable (the launcher hardcoded the value):

```
-    PREFILL_DECODE_INTERVAL=24
+    PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-24}"
```

It mirrors what the MI355X launcher already does, and is the only local
modification to the repo. Revert with `git checkout --` on that file.

**What MI355X's log has to answer first**, in order:

1. **`#running-req` per rank.** If it is ~20 rather than ~9, the step is simply
   doing more work and worse ITL is queueing, not kernels. MI355X reports GPU
   pool 97 % against 0.62 KV usage here, which makes this the leading
   hypothesis. Falsify by comparing `step_time` at *matched* batch, not at
   matched concurrency.
2. **`cuda graph: True` fraction.** DSPARK runs several small draft/verify
   forwards per step; outside graph replay, launch overhead alone can account
   for tens of ms. B200 is at 100 %.
3. Only if batch and graph status match does "MI355X kernels are slower" become
   the live hypothesis — and then it needs a roofline number, not a trace
   ranking.

**Confound that runs the right way:** ISL is 114,401 on B200 vs 105,652 on
MI355X, i.e. B200 carries 8 % *more* context and is still 2.5× faster per
token. The confound opposes the conclusion, which strengthens it.

**Separately, TTFT is a different mechanism and should not be mixed in.** B200's
overall cache hit 0.9635 splits 0.523 HBM / 0.429 host DRAM, so ~43 % of its
hits are restored over C2C from the HiCache host tier; MI355X hits almost
entirely in HBM (pool 97 %). That explains B200's worse TTFT (4.59 s vs 2.49 s
p50) and has no bearing on ITL, which is a pure forward.

**Then, and only then, the trace.** `trace_arm_b200.sh` arms the torch profiler
over a decode window. Two things make it different from a wall-clock window:
`num_steps` lets the server stop itself after N scheduler steps (bounded trace
size), and `profile_by_stage` keeps a 114k-token prefill from swamping the
decode steps. `with_stack` is off by default — eight DP ranks with stacks
produce enormous traces and measurable overhead.

When reading the trace, the failure mode to avoid is ranking kernels by time and
naming the top one: a stall on a matmul is usually that matmul waiting for
operands. The falsification test is to make the suspected unit cheaper and see
whether step time moves. The number that would actually settle "MI355X decode is
at its memory ceiling" is achieved KV-read bandwidth vs peak HBM — both chips
are in the same HBM3e class, so a 2.5× gap at comparable peak is a software
(achieved-utilization) claim, not a hardware one.

**Better experiment than two points:** sweep batch-per-rank at fixed context on
each platform and compare `step_time(batch)` curves. Two curves separate a
constant per-step overhead (graph/launch/sync — curves offset) from a slope
difference (kernel efficiency), which two single points never can.

## 10. How this environment was built

```bash
cd /workspace/agentx/InferenceX
git submodule update --init --recursive utils/aiperf          # ~5 min
( source /workspace/agentx/agentx_env.sh && source benchmarks/benchmark_lib.sh \
  && install_agentic_deps ) > /workspace/agentx/logs/install.log 2>&1   # ~4.5 min
source /workspace/agentx/agentx_env.sh
"$AIPERF_VENV/bin/hf" download --repo-type dataset semianalysisai/cc-traces-weka-062126   # ~4 min
```

The clone is on `main` `ca108e273`, deliberately not the parent skill's pinned
`8fcfc6283`: the B200 agentic recipes only exist on newer main.
