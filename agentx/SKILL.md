---
name: agentx-inferencex-test-env
description: >-
  Build and run the InferenceX "AgentX" agentic trace-replay benchmark (aiperf
  --scenario inferencex-agentx-mvp) on a single node, using
  benchmarks/single_node/agentic/*.sh. Use when standing up AgentX from scratch,
  when a launcher aborts before the server starts (missing aiperf venv, empty
  utils/aiperf submodule, unresolved traces, INFMAX_CONTAINER_WORKSPACE wrong),
  or when wiring DeepSeek-V4-Pro FP4 + SGLang MTP on MI355X.
---

# AgentX single-node test environment

AgentX is InferenceX's *agentic coding* scenario: instead of a synthetic
fixed-ISL/OSL sweep, it replays recorded Claude-Code trajectories (multi-turn,
tool-heavy, with subagent fan-out) from a HuggingFace dataset against a live
OpenAI-compatible server, driven by a **forked AIPerf** running the
`inferencex-agentx-mvp` scenario.

The launcher scripts under `benchmarks/single_node/agentic/` are labelled
**MVP / experimental** by the repo itself — they are a reference implementation
of the plumbing, not a published benchmark.

## CONTINUE HERE

**Status (2026-08-28, 10:36 local):** a **topk_v2 run is IN FLIGHT** (started
10:35:55, PID under its own setsid session). Results ->
`/workspace/results/b200align-topkv2-c64-1200s/`. See "IN FLIGHT" below.

### The one number that matters: b200align is +6.2 % over DP8+TBO at full length

**b200align c64, 3600 s, valid** (`/workspace/results/b200align-tp8-c64-3600s/`):

| | TP8 c64 | DP8+TBO c64 | **b200align** |
|---|---|---|---|
| aggregate tok/s/GPU | 17,080 | 17,443 | **18,518** |
| curve-reconstruction full | 17,219 | 17,594 | **18,670** |
| vs TP8 | — | +2.1 % | **+8.4 %** |
| vs DP8+TBO | — | — | **+6.2 %** |

Per-cutoff, same method for all three (cumulative tokens / cutoff / 8):

| cutoff | TP8 | DP8+TBO | b200align |
|---|---|---|---|
| 300 s | 13,613 | 10,988 | 14,002 |
| 600 s | 15,166 | 16,306 | 16,652 |
| 900 s | 17,320 | 17,740 | 18,963 |
| 1200 s | 17,325 | 17,322 | 18,792 |
| 1800 s | 18,266 | 18,171 | 19,232 |
| 2400 s | 18,649 | 18,521 | 19,337 |
| 3000 s | 17,856 | 18,180 | 19,132 |
| 3600 s | 17,219 | — | **18,670** |

b200align leads at **every** cutoff with no crossover — stronger than the
endpoint alone, given §25 records the TBO-vs-TP8 ordering flipping three times.
Still **below §25's >=10 % bar**, so: direction clear, ordering stable, magnitude
moderate, *not* "proven" by this file's own standard. Note the 1200 s run read
+9.1 % — the short window **over**states it.

**Latency is the bigger story than throughput:**

| mode | conc | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | cache hit | dur |
|---|---|---|---|---|---|---|---|
| TP8 | 64 | 17,080 | 13.8 | 72.7 ms | 3.35 s | 0.689 | 3629 s |
| DP8+TBO | 64 | 17,443 | 13.5 | 73.9 ms | 4.28 s | 0.955 | 3629 s |
| **b200align** | 64 | **18,518** | **20.0** | **50.0 ms** | 7.21 s | 0.955 | 3630 s |
| b200align | 64 | 18,485 | 20.7 | 48.2 ms | 5.75 s | 0.956 | 1227 s |

Fields: `request_metrics.latency.{intvty.p90, itl.p90, ttft.mean}`,
`server_metrics.cache.gpu_cache_hit_rate`. **P90 interactivity +45 %, ITL p90
-32 %** — far outside noise and reproduced independently by the 1200 s run.
**TTFT is the price: 7.21 s vs 3.35/4.28 s.** Consistent with
`--prefill-decode-interval 10` forcing decode between prefills: smoother
inter-token, slower first token.

**Attribution is exact.** A full `sglang_command.txt` diff against
`tbo-tp8-c64` shows **four** differing flags and nothing else — no other
additions, removals or value changes:
`--enable-dp-attention-local-control-broadcast`, `--prefill-decode-interval 10`,
`--stream-interval 20`, `--tokenizer-worker-num 8`. The +6.2 % / +45 % belongs to
those four **as a set**; no per-flag attribution exists yet.
`--incremental-streaming-output` was dropped during bisection and is **not** in
this result. Comparability: duration 3630 vs 3629 s, `input.mean` 116,923 vs
116,248/115,357, `output_actual.mean` 965.3 vs 956.0/919.4, `errors=0`, zero
hard failures.

**UNRESOLVED — TP8 cache hit reads 0.689 here but §22/§23 record TP8 at 91.8 %
and DP at 83.2 %, i.e. the opposite ordering.** Both DP arms read 0.955. §15
warns the cache metric is device-tier only. Do not cite either number until this
is reconciled.

### IN FLIGHT — sglang#36684 (topk_v2) at c64 / 1200 s

Cherry-picked **PR 36684** ("[AMD] Enable deepseek-v4 topk_transform v2
kernel", merged 2026-08-28 05:36 UTC) onto the local tree — **not** a mainline
update, deliberately: mainline carries three days of unrelated change (local
HEAD is 2026-08-25 `a1f9508dd4`) plus 20 uncommitted local files, which would
destroy attribution against the 18,518 baseline. #35619 (§17's TBO blocker) is
still **OPEN upstream**, so a mainline update would not have reintroduced it —
that particular fear was unfounded.

Applied with `git apply --exclude='test/*'`; touches
`topk_v2.cuh`, `topk_impl.cuh`, and one line of `server_args.py`
(`SGLANG_OPT_USE_TOPK_V2.set(False)` -> `True`). **Verified the changed line is
on our path**: it sits in `elif model_arch in ["DeepseekV4ForCausalLM"]:` ->
`elif is_hip():`. The other `set(False)` nearby belongs to the DSA family
(V3.2 / GLM-5.x) branch and is irrelevant. **Setting the env var alone is not
equivalent to this PR** — it also patches the kernel.
**JIT cache was moved aside** (`/root/.cache/sglang/jit` ->
`jit.pre36684`) so the patched `.cuh` recompiles; without that the old objects
are reused and the change is a no-op. Rollback: that directory plus
`/tmp/server_args.pre36684.bak`.

**Watch for**: the kernel was disabled on ROCm because it needs
`<cooperative_groups.h>` and `cg::this_cluster()` (server_args.py comment near
the DSA branch), which ROCm lacks. If it cannot build on gfx950 the run should
fail loudly at JIT/cuda-graph capture, not silently fall back. The monitor greps
`cooperative_groups|this_cluster|JIT compilation failed|hipcc.*error`.

**Compare against** (same conc, same 1200 s setting, actual 1227 s):
tok/s/chip **18,485**, P90 intvty **20.7**, ITL p90 **48.2 ms**,
TTFT avg **5.75 s**.

### Traps learned 2026-08-28 — these cost most of a day

1. **`/metrics` under-reports when `--tokenizer-worker-num > 1`.**
   `sglang:generation_tokens_total` read ~1 token per returned request on runs
   actually generating 767-965 tok/req, and `sglang:num_running_reqs` read 0.0
   on all 8 DP ranks while requests were being served. The frontend is split
   across 8 tokenizer processes and the endpoint appears to report one.
   **This single artefact produced three wrong root causes in a row**
   (`--chat-template`, then the scratch-reclaim OOR, then "the four B200 flags
   break generation") and a healthy run was killed on the strength of it.
   Judge generation health by **warmup progress vs the reference arm at matched
   `elapsed`** — `tbo-tp8-c64/benchmark.log` carries the per-30 s series
   (13/21/31/38/40/44 at 30..180 s) — by `errors=` in the aiperf line, and
   afterwards by `request_metrics.tokens.output_actual.mean`.
2. **Early `Decode batch` counts prove nothing.** tbo logs **0** decode batches
   across its first 2147 prefill batches. A prefill-only early log is normal.
3. **The server survives the launcher.** Twice, a completed run left
   `sglang.launch_server` alive holding 92 % VRAM with no launcher process.
   `kill -TERM` on the process group did not work (a second launcher process
   even appeared); only per-PID `kill -9` matching `sglang.launch_server` **and**
   `sglang::` cleared it.
4. **KFD reclaim is bursty, not gradual.** After the kill, 5 of 8 GPUs sat at
   90 %+ for ~11 minutes with **no process holding memory**
   (`rocm-smi --showpids` showed only `gpuagent` at 0), then all 8 dropped to 0
   within 30 s. Wait for it; do not conclude a leak.
5. **`pgrep -f <pattern>` matches your own shell command.** It falsely reported
   vim open, aiperf running and stray servers, four separate times. Use
   `ps -eo comm` / `ps -eo args` with a bracketed first character instead.
6. **Monitors that dedupe with `comm` consume a token permanently.** A `FAILED`
   match on `AIPERF_FAILED_REQUEST_THRESHOLD=0.10` in the env dump meant a real
   later `FAILED` could never fire. Filter out `^[A-Z_]+=` / `AIPERF_` lines.
7. **Never let a curl's `--max-time` exceed the tool timeout**, and never probe a
   loaded server with a short timeout and read 0 bytes as "broken" — under
   AgentX load both router and backend return nothing within 60 s.

### RETRACTION OF A RETRACTION — the `--chat-template` story is unresolved

Earlier today this file (a) called for landing `--chat-template` as a
correctness fix, (b) landed it on all four MI355X launchers, then (c) retracted
it, blaming it for collapsing generation to ~1 token/request. **(c) is also
wrong.** The next run, with no template, showed the same "~1 tok/req" reading —
which turned out to be trap 1 above, a metrics artefact. There was very likely
no collapse at any point.

What is still solid: `--tool-call-parser deepseekv4` makes
`resolve_chat_encoding_spec()` return `"dsv4"`
(`entrypoints/openai/chat_encoding.py:112`), so `encoding_dsv4.py` — not a jinja
template — renders DSv4 prompts, and `--chat-template` overrides that native
path. The startup line `No chat template found, defaulting to 'string' content
format` is misleading log noise. So the flag is **unnecessary**; it was never
shown to be **harmful**. It is currently absent from all four launchers, which
matches every reference arm, and that is the right default. §24.1 in
`references/b200-alignment.md` still overstates the case and needs this
correction folded in.

**What genuinely went wrong in the 04:23 and 04:56 runs is still unexplained.**
Those two really were degraded — warmup 324/707 at 2430 s against tbo's 701/707
at 1200 s, and both ended in
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` (`Available Free mem : 318 MB`) — that part
is not an artefact. Between then and the first healthy run, **two** things
changed: the template was removed **and** `--enable-prefill-delayer` +
`--enable-two-batch-overlap` were added back. Both changed at once, so neither
is established. `tbo_mtp.sh:170` calls guarded TBO+delayer "the validated
combination", which makes the delayer/TBO explanation the more likely one, but
it is a hypothesis. The 06:41 run (no template, no delayer/TBO) was tracking tbo
normally at elapsed=150 s when it was killed, which weakly argues **against** the
template being the cause.

`HSA_NO_SCRATCH_RECLAIM=0` was added, then removed again, and is **not** in the
working config; the container default (=1) is what all reference arms use.
`benchmarks/multi_node/amd_utils/env.sh:312` does pin it to 0 for
DeepSeek-V4-Pro with the comment "resolve the OOR issue", so if OOR ever recurs
on a healthy workload, that is the lever — and *that* result would be real
evidence rather than a workaround validated on a broken run.

### Next actions

1. Read the topk_v2 result against the 1200 s baseline above. If it fails to
   build on gfx950, roll back (`jit.pre36684`, `server_args.pre36684.bak`).
2. **Per-flag bisect of the four B200 flags**, ~90 min each. Start by removing
   `--prefill-decode-interval 10` alone — it is the prime suspect for both the
   ITL win and the TTFT regression.
3. Reconcile the TP8 cache-hit contradiction (0.689 here vs §22/§23's 91.8 %).
4. Settle the 04:23/04:56 degradation: rerun the current healthy config with
   delayer+TBO **removed**, template still absent. If it degrades, delayer/TBO
   was the fix and the template is fully exonerated.
5. Fold the trap list into the permanent sections; §24.1 still needs the
   correction above.

**Still parked from before:** `--tokenizer-worker-num 8` + `--stream-interval 20`
as an isolated pair (§24) — now partly answered, they are inside the +6.2 %
four; the 64-group cache-aware routing re-run (§23); stock-MoE `dp8 + ep8` at
c64 (§20.3c); `--enable-deepseek-v4-fp4-indexer` behind an accuracy gate;
`--load-balance-method total_tokens` (§18); the two sglang#35619 bugs (§17).

**Do not redo:** conc 48 full arm (§19); §20 H1/H2; the B200-vs-MI355X recipe
comparison (§20.3c); the full §21/§22 ladder; §23's routing run in its 8-key
form.

**Traps to re-read before touching anything** — §14 (stale-router trap), §15
(cache metric is device-tier only), §17 (#35619 blocks TBO), the four in §22.6
and §23, plus the seven above. While a run is in flight, test for the aggregate
by its **exact** `RESULT_FILENAME`, never `ls "$D"/*.json`.

### Reproducing an arm

```bash
# 1200 s b200align (current working config)
EP_SIZE=1 CONC=64 DURATION=1200 RESULT_DIR=/workspace/results/<name> \
  bash /workspace/claude-skills/agentx/agentx_b200align.sh
```
Launch it under `setsid nohup ... < /dev/null &` — a session kill took down the
03:35 run through its process group. Keep `--warmup-requests-per-lane 10`; cut
only `--benchmark-duration` (§25).

### Results index — every arm on file

Headline is tok/s/GPU for agentic trace replay, req/s for fixed-seq-len.
**Never compare agentic arms across concurrency on headline** — the trace mix
is an outcome of concurrency, not a controlled variable
(`references/conc-and-trace-mix.md` §19.6).

| arm | conc | headline | note | detail |
|---|---|---|---|---|
| TP8 agentic | 48 | 19,171.9 tok/s/GPU | +2.33 % vs published; longer traces | `conc-and-trace-mix.md` §19 |
| TP8 agentic | 64 | 17,079.5 | more reqs, more computed tokens than c48 | §16 |
| DP8+TBO agentic | 64 | 17,443.3 | +2.1 % over TP8, TTFT p50 3.10 s | §16 |
| MegaMoE+EPLB agentic | 64 | 12,630.8 | −26 % — MTPR padding vs variable ISL | `megamoe.md` §13 |
| ladder A/B/C/D | 64 | see table | DP +35 %, TBO −0.4 %/−10.8 % | `dp-tbo.md` §22 |
| ladder B + router | 64 | 2.20-3.12 req/s | routing fixes cache, imbalance kills it | `dp-tbo.md` §23 |

### Where the detail lives

- `references/published-arms.md` — §10 reproducing the published points, §11
  smoke run, §12 verified reproduction.
- `references/megamoe.md` — §13 MegaMoE MTPR vs `--chunked-prefill-size`.
- `references/conc-and-trace-mix.md` — §16 three serving paths at conc 64,
  §19 the conc 48 result and why cross-concurrency headlines are invalid,
  §25 how short a window can be (the >=10 % rule).
- `references/dp-tbo.md` — §18 rank imbalance, §20 the open investigation,
  §21 the ladder design, §22 its 8-point result, §23 cache-aware routing.
- `references/b200-alignment.md` — §24 flags B200 sets that we do not.

**Scripts here:** `agentx_env.sh` (sourced by all), `agentx_run.sh` (published
arms), `agentx_tbo.sh`, `agentx_megamoe.sh`, `agentx_smoke.sh`,
`agentx_debug.sh` (fast loop, §14 — start here for any diagnosis),
`agentx_ladder.sh` (§22 fixed-seq-len ladder), `agentx_router.sh` (§23).
Long runs should point at a frozen copy (`.ladder_frozen.sh`,
`.router_frozen.sh`) kept **in this directory** — a copy in `/tmp` breaks
`SKILL_DIR` and cannot source `agentx_env.sh`.

**External state:** `/workspace/InferenceX` @ `8fcfc6283` (+launchers under
`benchmarks/single_node/agentic/`, NOT version-controlled),
`/workspace/agentx-runtime/venv`, `/shared_nfs/hf_cache`, `/workspace/results/*`
(all completed runs incl. `armB-tp8-c48` and `ladder/`, with Prometheus
exports).

## 1. Layout — what has to exist where

`benchmarks/benchmark_lib.sh` hardcodes a container layout:

```
INFMAX_CONTAINER_WORKSPACE   default /workspace        <- must be the REPO ROOT
  ├── utils/agentic-benchmark/   requirements.txt, scripts/, analysis/
  ├── utils/aiperf/              git submodule -> SemiAnalysisAI/aiperf
  └── utils/agentic/             aggregation + validation python packages
```

Two things follow, and both bite:

- `write_agentic_result_json` and the power/validation steps do
  `cd "$INFMAX_CONTAINER_WORKSPACE"` and then `python -m utils.agentic...`.
  If the repo is **not** cloned directly at `/workspace`, you must export
  `INFMAX_CONTAINER_WORKSPACE=<repo root>` or every post-run aggregation step
  fails with `No module named utils`.
- `utils/aiperf` is a **submodule and is empty after a plain `git clone`**.
  `install_agentic_deps` does `uv pip install -e "$AIPERF_DIR"`, which fails on
  an empty directory. Init it explicitly.

## 2. Build the environment

```bash
# 2.1 repo at the pinned commit + the aiperf submodule
git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/InferenceX
cd /workspace/InferenceX
git checkout 8fcfc62830f76848b7431d051794349cf4680cf7
git submodule update --init --recursive utils/aiperf   # -> aiperf @ 754356e9

# 2.2 env file — ships with this skill, no need to write it.
#     Check the two paths inside still match this machine:
#       INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX   (repo root)
#       HF_HOME=/shared_nfs/hf_cache                       (NOT /workspace: 98% full)
cat /workspace/claude-skills/agentx/agentx_env.sh

# 2.3 isolated AIPerf venv (uv is auto-downloaded if absent)
cd /workspace/InferenceX
( source /workspace/claude-skills/agentx/agentx_env.sh
  source benchmarks/benchmark_lib.sh
  install_agentic_deps ) > /tmp/agentx_install.log 2>&1

# 2.4 pre-fetch the trace corpus (1.8 GB, public, no HF token needed)
source /workspace/claude-skills/agentx/agentx_env.sh
"$AIPERF_VENV/bin/hf" download --repo-type dataset semianalysisai/cc-traces-weka-062126
```

### Why `AIPERF_RUNTIME_DIR` is pinned

The default is `${TMPDIR:-/tmp}/inferencex-agentic-$$` — **`$$` is the shell
PID**, so every invocation gets a fresh directory and
`install_agentic_deps` (which starts with `rm -rf "$AIPERF_VENV"`) rebuilds the
whole venv from scratch. Pinning it to a stable path keeps the uv cache warm;
the rebuild then takes seconds instead of minutes.

### Why the venv is separate from the server's Python

`install_agentic_deps` deliberately refuses to share site-packages with
SGLang/vLLM: installing AIPerf into the server's interpreter can upgrade
FastAPI/Starlette/transformers underneath a running server. It also pins
**Python 3.11** via `uv venv --python 3.11`, because aiperf's
`requires-python = ">=3.11,<3.14"` while the sglang-rocm / vllm-rocm images
still ship 3.10 as `python3`. uv downloads a standalone 3.11 if the image has
none. Verified here: system `python3` is 3.10.12, venv is 3.11.16.

## 3. Trace source

`resolve_trace_source` picks the loader from `MODEL_PREFIX`, not from the
script name:

| `MODEL_PREFIX` | loader | HF dataset |
|---|---|---|
| `dsv4*`, `glm5.2*`, `minimaxm3*`, `kimik3*` | `semianalysis_cc_traces_weka_062126` | `semianalysisai/cc-traces-weka-062126` |
| anything else | `..._062126_256k` | `semianalysisai/cc-traces-weka-062126-256k` |

`MODEL_PREFIX` is **not** in the `check_env_vars` list, so leaving it unset does
not error — it silently falls through to the 256k-capped corpus. Set it.
Override with `WEKA_LOADER_OVERRIDE=<loader name>` (14 accepted names, see the
`case` in `resolve_trace_source`).

The corpus holds 393 unique traces; `build_replay_cmd` passes
`--num-dataset-entries 393` so all of them load.

## 4. Running

```bash
cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro"   # 805 GiB, 64 shards
export TP=8 EP_SIZE=1 DP_ATTENTION="false"
export CONC=32
export IS_AGENTIC=1
export KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0
export DURATION=3600
export PORT=8888
export RESULT_DIR="/workspace/results/dsv4-tp8-c32"
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_agentic_c32"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
```

### Deltas from the "reference" command that circulates for this script

| Reference said | Reality |
|---|---|
| `cd /workspace` then `bash benchmarks/...` | only correct if the **repo itself** is at `/workspace`; otherwise `cd` into the repo and set `INFMAX_CONTAINER_WORKSPACE` |
| `MODEL_PATH=/models/DeepSeek-V4-Pro` | does not exist here; weights are `/shared_nfs/models/DeepSeek-V4-Pro` (symlink to `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`). If the path is missing **or empty** the script silently starts an 805 GiB `hf download` into it |
| `RESULT_DIR=/workspace/results` | fine, but `RESULT_FILENAME` is **also** required — `write_agentic_result_json` writes `$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json` and produces `.json` (a dotfile) when unset. It is set by the CI runners, not by the launcher |
| — | `AGENTIC_OUTPUT_DIR` defaults to `$INFMAX_CONTAINER_WORKSPACE`, i.e. the aggregate lands in the repo root. Point it at `$RESULT_DIR` |
| `DURATION=3600` | the scenario enforces a **900 s minimum**; below that the launcher adds `--unsafe-override` and flags `submission_valid=false` |

### Knobs for a short smoke run

- `DURATION=300` → auto `--unsafe-override`.
- `AIPERF_WARMUP_REQUESTS_PER_LANE=1` → skip the 10-request-per-lane warmup ramp.
- `AIPERF_EXPERIMENTAL_FAST=1` → forces `duration=1200` **and** warmup 1/lane
  (it overrides `DURATION`, so don't combine it with a shorter `DURATION`).
- `EVAL_ONLY=true` → skips replay entirely, runs `run_eval` instead, and
  disables the simulated-acceptance-length pin.

## 5. What the launcher actually does

1. `check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION` — anything else (`MODEL_PREFIX`, `RESULT_FILENAME`, `PORT`) is unvalidated.
2. **GPU drain gate**: polls `rocm-smi --showmemuse` for up to 15 min and refuses to start until *every* GPU is ≤10 % VRAM. An 805 GiB checkpoint takes minutes to reclaim after a previous run, and booting into a half-drained node fails RCCL init with a bogus HIP "unhandled cuda error".
3. `resolve_trace_source` → `install_agentic_deps` → `hf download` the dataset.
4. Launch SGLang, `wait_for_server_ready` on `/health` (no timeout — it waits until the PID dies).
5. `build_replay_cmd` → `run_agentic_replay_and_write_outputs`:
   AIPerf replay → `process_agentic_result` → GPU power adapter →
   `analyze_benchmark_distributions.py` → `validate_agentic_result`.

Note the ordering: **results are written before validation**, and the function
returns the replay exit code *after* writing artifacts. A non-zero exit still
leaves usable artifacts in `$RESULT_DIR`.

## 6. Recipe specifics — dsv4 FP4 + SGLang MTP on MI355X

- `deepseek-ai/DeepSeek-V4-Pro` is FP4/FP8 **mixed** (FP4 MoE expert weights,
  FP8 elsewhere); InferenceX labels it `precision: fp4`. The top-level
  `quantization_config` in `config.json` reads `fp8` — that is expected, not a
  wrong checkpoint.
- Spec decode is **EAGLE with `--speculative-eagle-topk 1`**, not NEXTN: the
  V3/R1 NEXTN loader crashes on the V4 architecture. Depth 3.
- Throughput runs pin acceptance length: `SGLANG_SIMULATE_ACC_LEN=2.49` from
  `golden_al_distribution/dsv4_mtp.yaml` (thinking_on, depth 3). `EVAL_ONLY=true`
  turns this off so accuracy stays real.
- **No `--chat-template`** on purpose: `deepseek_v4_thinking.jinja` drops tool
  definitions and `role: tool` messages, which would truncate the tool-heavy
  AgentX prompts and distort ISL.
- `MEM_FRACTION_STATIC=0.89`, `CHUNKED_PREFILL_SIZE=8192`, `--page-size 256`,
  `--kv-cache-dtype fp8_e4m3`, `--watchdog-timeout 3600`.
- `MAX_RUNNING_REQUESTS = min(2*CONC, 256)`, `CUDA_GRAPH_MAX_BS = min(that, 128)`
  — AgentX `CONC` counts **session trees**, and subagent fan-out pushes
  instantaneous request concurrency above it.
- `CONC >= 32` bumps `AGENTIC_WARMUP_GRACE_PERIOD` to 3600.
- The DP-attention branch (`DP_ATTENTION=true`) starts `sglang_router` with
  consistent hashing on the AIPerf correlation id, on `PORT`, with the engine
  moved to `PORT+1`. It is dormant — no dp-attn arm exists for this key in
  `configs/amd-master.yaml`.

## 7. Gotchas hit while building this

- **Empty `utils/aiperf`** — plain clone leaves the submodule empty; the venv
  install then fails late, and the failure surfaces as `aiperf: No such file or
  directory` rather than as an install error.
- **Python 3.10 default** — see §2; without the 3.11 pin the venv silently ends
  up without `aiperf`/`hf` binaries.
- **`INFMAX_CONTAINER_WORKSPACE`** — see §1. Post-run aggregation is the only
  thing that breaks, i.e. *after* an hour of benchmarking.
- **`SGLANG_ENABLE_UNIFIED_RADIX_TREE` is deprecated** in sglang
  `0.5.18.dev20260825`; the launcher still exports it and the server prints a
  `UserWarning` at startup. Harmless.
- **Disk**: `/workspace` here is a 10 T NFS volume at 98 % (≈260 G free). The
  venv (~1 GB) and results fit; model weights and the HF cache must not go
  there. `HF_HOME` is pointed at `/shared_nfs/hf_cache` in the env file.
- **`rocm-smi` low-power warning** (`AMD GPU device(s) is/are in a low-power
  state`) on an idle node is normal and does not affect the ≤10 % drain gate.

## 8. Where the results land

```
$RESULT_DIR/
  sglang_command.txt         exact server argv
  server.log                 SGLANG_* env dump + server stdout/stderr
  benchmark_command.txt      exact aiperf argv
  benchmark.log              replay stdout
  gpu_metrics.csv            per-GPU power/util samples (ENABLE_AGENTX_POWER=1 default)
  aiperf_artifacts/          profile_export.json / .jsonl, server_metrics_export.json
$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json     aggregate consumed by CI
```

Disable power collection with `ENABLE_AGENTX_POWER=0` if `amd-smi` sampling is
in the way.

## 9. Debugging a live run

Repo ships `.agents/skills/debug-agentx-runs/SKILL.md` for the cluster/Slurm
case. Single-node equivalent — one channel per log, filtered:

```bash
rg -n -i 'Phase |warmup|profiling|in_flight=|kv_usage=|prefix_cache_hit=|ERROR|Traceback|OOM|RCCL|timeout' \
   "$RESULT_DIR"/{server,benchmark}.log | tail -20 | cut -c1-200
```

Phase order to expect: GPU drain gate → deps install → dataset download →
weight load → `ready to roll` → AIPerf *Configure Profiling* (dataset
reconstruct + mmap, 4–14 min; timeout raised to 1800 s) → warmup → profiling.

## 14. Fast debug loop — `agentx_debug.sh`

A full launcher run is ~90 min at conc 64, and almost none of it is the thing
you are usually debugging: ~25 min weight load, ~40 min warmup (10 requests per
lane x CONC lanes), then 3600 s of profiling. For diagnosing a metric or a
crash you need none of that.

**First, though: check the artifacts you already have.** Every completed run
persists the full Prometheus scrape (`aiperf_artifacts/server_metrics_export.json`)
and the aggregate JSON. Most "why is this number strange" questions are
answerable at zero GPU cost — the §15 cache-tier finding below was.

```bash
agentx_debug.sh serve  /workspace/results/armB-tp8-c64      # once, ~25 min
agentx_debug.sh probe  /workspace/results/armB-tp8-c64 /tmp/p1 300 1   # ~5 min
agentx_debug.sh probe  /workspace/results/armB-tp8-c64 /tmp/p2 300 1   # ~5 min
agentx_debug.sh status
agentx_debug.sh stop
```

`serve` replays the argv from that run's `sglang_command.txt` plus the `SGLANG_*`
block persisted at the top of its `server.log`, so the server is identical to
the reference run without re-deriving anything. `probe` takes that run's
`benchmark_command.txt` and rewrites only `--benchmark-duration`,
`--warmup-requests-per-lane` and the artifact dir, so every other flag still
exercises the same code path.

Iteration drops from ~90 min to ~5 while the server stays resident.

**Probe numbers are for trends and bugs only.** `duration < 900` forces
`--unsafe-override` and stamps `submission_valid=false`; a 1-per-lane warmup
never reaches the steady state the published arms measure. Never quote a probe
against the leaderboard.

### The stale-router trap (worse than the stale server)

§11 notes the launcher leaves the SGLang server running. In **DP-attention
mode it also leaves `sglang::router`**, which binds `PORT` (8888) while the
engine sits on `PORT+1`. A router that outlives its backend still answers
`/health` — so the next run's `wait_for_server_ready` passes instantly, AIPerf
sends warmup traffic to a dead backend, and the run dies with
`Terminal warmup failure` while the real server is still loading weights.

VRAM and `pgrep sglang.launch_server` both miss it: the router holds no GPU
memory and does not match that pattern. Check the port:

```bash
ss -lntp | grep -E ':(8888|8889)\b'
```

`agentx_debug.sh status` / `stop` cover server, router, VRAM and ports together;
`serve` refuses to start unless all four are clear.

## 15. `gpu_cache_hit_rate` is device-tier only

With HiCache on, `server_metrics.cache.gpu_cache_hit_rate` counts **only the
device tier**. Hits served from the host DRAM tier land in
`cpu_cache_hit_rate`, and only `overall_cache_hit_rate` is comparable across
configurations.

Measured at conc 64 on identical traces (ISL mean within 1.3%):

| Run | device tokens | host tokens | gpu_hit | cpu_hit | overall_hit |
|---|---|---|---|---|---|
| TP8 + HiCache | 338,866,944 | 128,572,160 | 0.6888 | 0.2613 | **0.95008** |
| DP8 + TBO | 479,664,640 | — | 0.9549 | — | **0.95492** |

The TP8 arm looks 27 points worse on `gpu_cache_hit_rate` and is in fact within
0.5% on overall hit rate — 26% of its hits simply came from host DRAM. Reading
the device-only field as "the cache hit rate" makes a healthy HiCache
configuration look broken. Use `overall_cache_hit_rate`, and read
`cached_tokens_by_source` to see which tiers actually served.

## 17. sglang#35619 blocks TBO for every DP-attention config

With PR #35619 applied, any `--enable-dp-attention --enable-two-batch-overlap`
run dies at the first prefill:

```
Exception: 3 errors happen:
Field mega_moe_global_num_tokens_cpu has value, but is not yet supported
Field mega_moe_global_max_tokens has value, but is not yet supported
Field mega_moe_sync_tokens has value, but is not yet supported
```

`scheduler_components/dp_attn.py` `_update_gather_batch` sets
`batch.mega_moe_global_num_tokens` unconditionally on the DP gather path — it
checks neither `moe_a2a_backend` nor `SGLANG_AMD_USE_FLYDSL_MEGA_MOE` — and
`forward_batch_info.py` expands it into three `ForwardBatch` fields that the TBO
ubatch splitter's field whitelist rejects. **MegaMoE does not have to be enabled
for this to fire.**

Reverting the PR restores TBO, which is how the DP8+TBO row above was measured.
A fix needs either a MegaMoE gate on the assignment, or the three fields added
to the TBO whitelist with split semantics defined.

Reported upstream alongside the `SGLANG_AITER_MEGA_RANK_SYNC` IndexError
(§10 of `dsv4/megamoe/PR35619_UPSTREAM_REPRO.md`).
