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

**Status (2026-08-28, 04:05 local):** machine **idle** — 8 GPUs at 0 % VRAM, no
sglang processes, ports 8888/8889 clear. Environment unchanged; still no PR
patch on `/sgl-workspace/sglang` (reverted so TBO works — §17).

**What is settled.** DP attention **alone** is **+35 %** on fixed-seq-len at a
near-zero cache hit rate — the largest lever on file. TBO **alone never gains**
(−0.4 % at 8 % hit, −10.8 % at 83 % hit), so §16's "TP8 and DP8+TBO within
2.1 %" is DP winning and TBO plus the delayer giving it back. The mechanism is
**cache partitioning**: DP splits the radix cache per rank, so TP8 hits 91.8 %
where DP hits 83.2 % and computes 2x the prefill. Routing on a prefix-stable
key **does repair it** (91.3 %, computed prefill halved) but with only 8 keys
over 8 DP ranks the load collapsed onto half the ranks and cost more than it
won. Details: `references/dp-tbo.md` §22, §23.

**PENDING — user-deferred on 2026-08-28, do not start without asking:**
(1) land `--chat-template` (§24.1); (2) `--tokenizer-worker-num 8` +
`--stream-interval 20` as a pair (§24); (3) the 64-group cache-aware routing
re-run (§23). All three are still the right next steps; they are parked, not
dropped.

**TRAP (new, cost one arm): DSv4 needs `HSA_NO_SCRATCH_RECLAIM=0`.**
The 04:23 b200align run reached warmup and then died in prefill with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES Code: 0x1008, Available Free mem : 318 MB`
on DP0 (`Fatal Python error: Aborted` in `watchdog.py:147`, `scheduler_0`
exit -3); the other ranks then failed with gloo `Connection closed by peer`,
which is **secondary — do not chase it**. The container ships
`HSA_NO_SCRATCH_RECLAIM=1` as an environment default (it is in `env` but in no
shell init file), so every arm here inherited it.
`benchmarks/multi_node/amd_utils/env.sh:312` already pins it to **0** for
DeepSeek-V4-Pro with the comment "resolve the OOR issue"; the single-node
agentic launchers never picked that fix up. Now set in
`..._b200align_mtp.sh` only — the TP8 and TBO reference numbers were measured
with the container default, so setting it in those would break comparability.

Two things this crash was **not**, both settled by the tbo arm as a control:
*not* shared-experts fusion (`tbo-tp8-c64` completed a full 3600 s run with the
identical `--disable-shared-experts-fusion`, `mem-fraction-static 0.90` and
`--chunked-prefill-size 65536` on DP8), and *not* KV-pool sizing (b200align died
at **0.08** peak full-token-usage where tbo survived **0.50** — the exhausted
memory was scratch/activation outside the static pool, so lowering
`mem-fraction-static` would have been the wrong lever). Evidence archived at
`/workspace/results/b200align-oom-0424/`.

**IN FLIGHT (2026-08-28 04:56):** B200-aligned DP arm, **fusion OFF**, c64,
`DURATION=1200`, results in `/workspace/results/b200align-tp8-c64/`.
Scratch reclaim on. Launched detached with `setsid` so a session kill cannot
take it down again:

```bash
setsid nohup env EP_SIZE=1 CONC=64 DURATION=1200 \
  bash /workspace/claude-skills/agentx/agentx_b200align.sh \
  > /workspace/results/b200align-tp8-c64/launcher.log 2>&1 < /dev/null &
```

Two changes versus the 03:35 attempt, **both verified in the `sglang_command.txt`
the launcher wrote at startup, not assumed from the source**:
`--disable-shared-experts-fusion` (was `--enforce-`) and `--chat-template
.../deepseek_v4_thinking.jinja` (was absent). Dropping fusion removes §16
confound 1, so this arm now differs from the DP baseline only in the B200 DP
flags plus the template — two variables, not four. Fusion-with-DP moves to the
backlog as its own single-variable experiment.

**Its 1200 s references are no-template numbers** (TP8 c64 17,325, DP8+TBO c64
17,322), so the template is an uncontrolled variable in that comparison; a
templated TP8 reference arm is needed for a clean read. Per §25 a 1200 s window
resolves >=10 % only.

**PRIOR ATTEMPTS, ZERO DATA (both).** 03:35 launch died 03:42 when the shell
holding it was killed — `server.log` stopped at `Load weight end elapsed=274.5 s`,
before KV-cache init and server-ready, so aiperf never started; no traceback, no
OOM, no watchdog, i.e. an external SIGKILL, not a software fault. Evidence
archived at `/workspace/results/b200align-tp8-c64-aborted-0335/`. A 04:12
relaunch was killed deliberately 8 minutes in (weights still loading) to apply
the fusion change, because the script must never be edited while running
(§22.6).

The wrapper `agentx/agentx_b200align.sh` (03:35) was already on disk and is
parameterised; what was lost with the shell is the **override line** in front
of it. Reconstructed from `sglang_command.txt` against the wrapper's own
defaults (`TP=8 CONC=32 EP_SIZE=8 DURATION=3600 DP_ATTENTION=true`):

```bash
EP_SIZE=1 CONC=64 DURATION=1200 bash /workspace/claude-skills/agentx/agentx_b200align.sh
```

Derivation: `--max-running-requests 128` = 2*CONC -> **CONC=64** (not the
default 32); **no `--ep-size`** in the command while the launcher emits it for
`EP_SIZE>1` -> **EP_SIZE=1** (not the default 8); `--chunked-prefill-size
65536` = 8192*TP -> TP=8 (default); backend `--port 8889` = PORT+1 -> PORT=8888
with `DP_ATTENTION=true` (default). `DURATION=1200` is the one value not
recoverable from `sglang_command.txt` — it is a client-side aiperf flag and the
run died before `benchmark_command.txt` was written; 1200 is taken from the
03:37 CONTINUE HERE note. `RESULT_DIR` defaults to
`/workspace/results/b200align-tp8-c64`, which matches the directory on disk,
and `RESULT_FILENAME` carries the `agentic-b200align` infix.

The arm is DP8 + dp-attention with B200's DP flags
(`--enable-dp-attention-local-control-broadcast`, `--tokenizer-worker-num 8`,
`--stream-interval 20`, `--incremental-streaming-output`,
`--prefill-decode-interval 10`), **no TBO, no prefill-delayer**, and
shared-experts fusion ON (`--enforce-shared-experts-fusion`) — a deliberate
deviation from B200, and fusion has never been combined with DP here
(§16 confound 1). `DURATION=1200`, warmup 10/lane.
**Caveat before re-running it as-is:** this arm moves **three** things at once
(B200 DP flags, the §24 tokenizer/stream pair, fusion-on-with-DP), and it
carries **no chat template**, so by §24.1's own rule its number would be
re-measured anyway. Land §24.1 first.
**Compare it only against the reconstructed 1200 s values** — TP8 c64 **17,325**
and DP8+TBO c64 **17,322** — never against the 3600 s headlines
(`references/conc-and-trace-mix.md` §25).

**RETRACTED — `--chat-template` must NEVER be passed on a DSv4 AgentX arm.**
Earlier today this file argued the opposite, the flag was landed on all four
MI355X launchers, and it **broke the arm**: generation collapsed to **~1 output
token per request** (live `/metrics`: 62.7 M prompt tokens vs **331** generated
over 329 requests, `num_running_reqs` 0.0 on all 8 DP ranks) against **919-956
tok/req** on the untemplated reference arms. The flag has been removed again
from all four launchers; broken-run evidence is in
`/workspace/results/b200align-chattemplate-broken-0456/`.

Why: SGLang never used a jinja template for this model. `--tool-call-parser
deepseekv4` makes `resolve_chat_encoding_spec()` return `"dsv4"`
(`entrypoints/openai/chat_encoding.py:112`), so the **native encoder**
`entrypoints/openai/encoding_dsv4.py` owns thinking, tool calls, tool results,
EOS and reasoning history. `--chat-template` *overrides* that with a nine-line
system/user/assistant template. The startup line `No chat template found,
defaulting to 'string' content format` is misleading log noise, not a defect.
Full post-mortem and the standing rule: `references/b200-alignment.md` §24.1.

Secondary symptoms seen before the cause was found — 3-4x slower warmup
(324/707 at 2430 s vs tbo 701/707 at 1200 s) and a prefill-only profile (2147
prefill batches, 2 decode) — are **consequences of the collapse, not causes**:
at ~1 token per turn the replay still feeds the whole history back, so prompts
snowball to ~163 k tokens/request. Three other explanations were proposed and
each was killed by a control (shared-experts fusion: tbo ran the identical
setting fine; aiter untuned GEMMs: tbo has 5x more and is faster;
decode starvation: like-for-like over the same 2147 prefill batches, tbo logged
0 decode and b200align 2). **Get output tokens/request from `/metrics` early —
it would have caught this in minutes instead of 65.**

**sglang#36656 checked — does NOT affect AgentX, no need to merge first.**
The PR deletes an 8-line silent `mem_fraction_static *= 0.85` in
`server_args.py:6417-6423` that fires when
`resolved_view(self).attention_backend == "aiter"` and `context_len > 8192`.
AgentX runs `--attention-backend dsv4`, its own backend and **not an alias for
`aiter`** (`server_args.py:190`; only `"compressed"` aliases to `dsv4`), so the
branch never fires on our path. Confirmed empirically, not just by reading:
every arm's `server.log` shows `attention_backend='dsv4'` and the mem fraction
we passed, verbatim — b200align and tbo `0.9`, armB c64/c48 `0.89`. The reduced
values (0.765 / 0.7565) appear nowhere. The only other silent rewrite in
`server_args.py` is `adjust_mem_fraction_for_vlm` (line 10302), a vision path
DSv4 text serving never enters. This becomes live only if an arm switches to
`--attention-backend aiter`.

**Short-window rule (new, §25):** a short AgentX arm resolves **>=10 %** effects
and nothing smaller. The c48>c64 ordering is stable from 600 s; the 2.1 %
TBO-vs-TP8 ordering flips three times before 3000 s. Keep warmup at 10/lane and
cut only `--benchmark-duration`; cutting warmup biases against DP arms
specifically, because DP ranks each fill their own cache.

**NEXT ACTION — in this order, and the order matters.**

1. **Land `--chat-template`** (`references/b200-alignment.md` §24.1) — the
   tool-message objection is resolved above, the template is safe here. We have
   been running AgentX with **no chat template at all**: the model ships none,
   and every arm logged `No chat template found, defaulting to 'string' content
   format` while aiperf posts to `/v1/chat/completions`. Prompts carry no role
   markers and no trailing `<think>`, so `SGLANG_DEFAULT_THINKING=1` and
   `--reasoning-parser deepseek-v4` may never have been exercised. This changes
   the workload, so it must land **before** any further A/B or every baseline
   is re-measured afterwards. Correctness first, not a tuning knob.
2. **`--tokenizer-worker-num 8` + `--stream-interval 20`** as one pair, on
   ladder rung A, points 1 and 5 (§24). We run 1 tokenizer worker and flush
   every token; B200 runs 8 and flushes every 20, on ~950-token outputs x
   ~4,300 requests. Same shape as §20.3b's unexplained 17-19 % scheduler-level
   rank idle that never appears in `gfx_activity`. Split the pair only if it
   moves the number.
3. **Cache-aware routing, decisive re-run**: §23's experiment with
   `--gsp-num-groups 64 --gsp-prompts-per-group 8` (still 512 requests), so
   keys >> ranks. §23's negative result is a hash-collision artifact of 8 keys
   on 8 ranks and is **not** a verdict on routing.
4. Reproduce the TBO PR on an older SGLang (§21.4) — TBO shows no gain
   anywhere, so suspect a regression rather than a workload shape.

**Backlog after that:** stock-MoE `dp8 + ep8` at c64, never run here (§20.3c);
`--prefill-decode-interval 10` and `--enable-deepseek-v4-fp4-indexer` (the
latter needs an accuracy gate, §24); TBO @ conc 48; `tbo` with shared-experts
fusion **on** (§16 confound 1 — the §22 ladder held fusion off in all rungs);
`--load-balance-method total_tokens` (§18); the two sglang#35619 bugs (§17,
not started — the PR is reverted, not patched).

**Do not redo:** conc 48 full arm (§19); §20 H1/H2; the B200-vs-MI355X recipe
comparison (§20.3c); the full §21/§22 ladder, all 8 points; §23's routing run
(re-run it only in the 64-group form above).

**Traps to re-read before touching anything** — §14 (stale-router trap), §15
(cache metric is device-tier only), §17 (#35619 blocks TBO), plus the four that
cost real GPU time in §22.6 and §23: the Prometheus `cache_hit_rate` gauge
reads **0.0** on this path (use `server.log`'s `#cached-token`); `stop` must
match `sglang::` children and wait >200 s; never edit a shell script while it
is running; `python -m sglang.benchmark.serving` fails from `/sgl-workspace`
(the repo dir shadows the package). While a run is in flight, test for the
aggregate by its **exact** `RESULT_FILENAME`, never `ls "$D"/*.json` —
`gpu_metrics_identity.json` appears minutes in and a glob reports "finished"
far too early.

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
