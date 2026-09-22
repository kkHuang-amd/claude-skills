# DSv4-Pro FP4 + DSpark MTP — fixed ISL 128k / OSL 1k, MI355X vs B200

**Both nodes run the scripts in this folder. Do not re-derive a config on
either side — a config difference is exactly what this test is designed to
rule out.** If a knob has to change on one platform, change it in this folder,
write down why in §7, and re-run both sides.

- `dsv4_fp4_mi355x_sglang_mtp.sh` — AMD MI355X (ROCm, gfx950)
- `dsv4_fp4_b200_sglang_mtp.sh` — NVIDIA B200

---

## 1. What this measures and why it exists

The AgentX arms (`benchmarks/single_node/agentic/dsv4_fp4_{mi355x,b200}_sglang_mtp.sh`)
replay recorded multi-turn agent trajectories. Their input length is an
*outcome* of the trace mix, the concurrency and the prefix-cache hit rate, so an
MI355X-vs-B200 delta there mixes scheduling, DP load balance and cache effects
into whatever kernel difference exists.

This pair pins **ISL = 131072, OSL = 1024, range-ratio = 1.0**. Every request is
byte-identical in shape, so:

- DP ranks receive identical work — no imbalance, no need for cache-aware routing
- the prefix cache is off, so every request does a real 128k prefill
- the speculative acceptance length is pinned to the same number on both sides

What is left is the kernel-wise gap, plus the platform-path differences listed
in §6 that no flag can remove.

Server configuration is copied from each platform's own AgentX launcher, so
numbers here stay interpretable next to the AgentX results.

**Primary measurement point: `CONC=128`.**

---

## 2. What you need from the InferenceX repo

The scripts are portable (they can live anywhere) but they are **not
self-contained** — they need an InferenceX checkout for the harness and the
benchmark client. Point `INFERENCEX_ROOT` at it; the default is
`/workspace/InferenceX`. If the script sits inside the repo at
`benchmarks/single_node/<subdir>/`, it derives the root itself.

Required files inside that checkout:

| Path | Why |
|---|---|
| `benchmarks/benchmark_lib.sh` | sourced — provides `check_env_vars`, `wait_for_server_ready`, `wait_for_ready`, `wait_for_amd_gpu_clean`, `start_gpu_monitor` / `stop_gpu_monitor`, `write_command`, `run_benchmark_serving`, `run_server_client` |
| `infx/bench_serving/benchmark_serving.py` | the load generator (`--dataset-name random --ignore-eos`) |
| `infx/bench_serving/encoding_dsv4.py` | the `--dsv4` DeepSeek-V4 prompt framing, applied **client-side** |
| `infx/bench_serving/backend_request_func.py`, `benchmark_outcome.py`, `benchmark_utils.py` | imported by the client |
| `infx/bench_serving/server_watch.py` | used because the scripts pass `--server-pid` |

**Not needed** — do not go looking for them: the AIPerf venv, the agentic trace
corpus, `benchmarks/single_node/chat_templates/`, `golden_al_distribution/`,
`configs/`. Those belong to the AgentX replay path, which this test does not use.

Both scripts pass `--bench-serving-dir "$INFERENCEX_ROOT"`, so `PYTHONPATH` is
set for you and **you do not have to `cd` into the repo**.

Runtime prerequisites on the node: a working SGLang install (`python3 -m
sglang.launch_server`), the DSv4-Pro FP4 checkpoint, and `transformers` for the
client's tokenizer. On MI355X the script waits for prior-job HBM reclaim via
`rocm-smi` before launching.

If `/workspace/claude-skills` is not mounted on your node, copy this whole
folder over and run it from wherever it lands.

---

## 3. Run it

Identical for both platforms except the script name and the two `EP_SIZE` /
workspace notes below.

### MI355X

```bash
export INFERENCEX_ROOT=/workspace/InferenceX          # adjust if your checkout differs
export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro" # local weights; omit to hf download
export TP=8 EP_SIZE=1 DP_ATTENTION=true
export CONC=128 ISL=131072 OSL=1024 RANDOM_RANGE_RATIO=1.0
export PORT=8888
export RESULT_DIR=/workspace/results/dsv4-fixed-128k1k-mi355x-c128
export RESULT_FILENAME=dsv4_fixed128k1k_mi355x_dp8_c128
mkdir -p "$RESULT_DIR"

setsid nohup bash dsv4_fp4_mi355x_sglang_mtp.sh \
  > "$RESULT_DIR/launcher.log" 2>&1 < /dev/null &
```

### B200

```bash
export INFERENCEX_ROOT=/ix                             # repo mount on the B200 image
export INFMAX_CONTAINER_WORKSPACE="$INFERENCEX_ROOT"   # only if your runner does not set it
export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PATH=/path/to/DeepSeek-V4-Pro
export TP=8 EP_SIZE=8 DP_ATTENTION=true                # CONFIRM EP_SIZE — see §7
export CONC=128 ISL=131072 OSL=1024 RANDOM_RANGE_RATIO=1.0
export PORT=8888
export RESULT_DIR=/workspace/results/dsv4-fixed-128k1k-b200-c128
export RESULT_FILENAME=dsv4_fixed128k1k_b200_dp8_c128
mkdir -p "$RESULT_DIR"

setsid nohup bash dsv4_fp4_b200_sglang_mtp.sh \
  > "$RESULT_DIR/launcher.log" 2>&1 < /dev/null &
```

`setsid` matters: a terminal hang-up otherwise takes the whole process group
down mid-run.

### Watching it without drowning in log

```bash
grep -E "ready to roll|Initialization failed|Traceback|max_total_num_tokens" \
  "$RESULT_DIR/server.log" | tail -5 | cut -c1-200
grep -E "Successful requests|Total token throughput|Mean TTFT|Mean TPOT|Mean ITL" \
  "$RESULT_DIR/launcher.log" | cut -c1-200
```

### Teardown

The launcher has no cleanup trap. After the run the server, the eight
`sglang::scheduler_DP*` and the `sglang::tokenizer_worker` processes survive and
keep holding HBM. Kill the whole process group, then gate the next launch on
**both** VRAM and the port being free — VRAM at 0 % does not mean 8888 is free.
Never `pkill -f <pattern>`: the pattern matches your own command line.

---

## 4. Results, and how to read them

`run_benchmark_serving` writes `$RESULT_DIR/$RESULT_FILENAME.json`.

**Pass criteria before any number is quoted:**

1. the server reached ready and the result JSON exists
2. `errors == 0` / `Successful requests == NUM_PROMPTS` (384 at c128)
3. measured mean input length is within ~3 % of 131072 — the client-side DSv4
   framing adds a few tokens. If it moved more than that, the two sides did not
   run the same workload and the comparison is void.

**Read TTFT and TPOT separately.** At ISL 128k / OSL 1k the aggregate token
throughput is overwhelmingly prefill, so it will hide the decode picture
entirely:

- **Mean TTFT** → prefill kernels (chunked prefill GEMM + attention + MoE)
- **Mean TPOT / ITL** → decode kernels (paged attention over 128k KV, MoE,
  DSpark verify)

Report the three numbers side by side with the run's ISL mean. Do not quote a
single "throughput gap" figure for this test.

---

## 5. The KV-layout trap — read this before attributing anything to kernels

**`--kv-cache-dtype` is a no-op for DSv4.**
`srt/arg_groups/overrides.py:947` (`_deepseek_v4_kv_cache_dtype`) rewrites
`auto` → `fp8_e4m3` for `DeepseekV4ForCausalLM`. The MI355X AgentX launcher
passes it explicitly and the B200 one omits it; both resolve to the same value.
It is **not** the difference between the platforms.

The layout is chosen by two HIP-only gates in
`kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py`:

```
is_unified_kv_triton() = is_hip() and SGLANG_HACK_FLASHMLA_BACKEND == "unified_kv_triton"
is_unified_kv_fp8()    = is_unified_kv_triton() and SGLANG_DSV4_UNIFIED_KV_FP8 and gfx95
```

consumed at `model_executor/pool_configurator.py:998`:

```
unified     -> kv_bytes = dsv4_unified_row_bytes(qk_nope, qk_rope, fp8)   # kv_cache_dtype ignored
not unified -> kv_bytes = qk_nope + qk_rope*2 + 8                         # the CUDA path
```

DSv4 has `head_dim 512`, `qk_rope_head_dim 64`, so `qk_nope = 448`:

| arm | bytes / token | note |
|---|---|---|
| MI355X, `UNIFIED_KV_FP8=0` (AgentX default) | `(448+64)*2` = **1024** | unified BF16 |
| MI355X, `UNIFIED_KV_FP8=1` | **640** | two-pool fp8; needs gfx95; incompatible with HiCache |
| B200 — can never be unified (`is_hip()` is False) | `448+128+8` = **584** | nope fp8 + rope bf16 + scale |

**The default MI355X arm therefore moves 1.75x the KV bytes per token that B200
does.** At ISL 128k decode is KV-bandwidth bound, so running only the default
arm folds a quantisation difference into what you will read as a kernel gap.

**Required: run both MI355X arms.**

```bash
# arm A — AgentX-equivalent (default)
UNIFIED_KV_FP8=0 RESULT_DIR=.../mi355x-c128-kvbf16 RESULT_FILENAME=..._kvbf16 bash dsv4_fp4_mi355x_sglang_mtp.sh
# arm B — KV footprint closest to B200
UNIFIED_KV_FP8=1 RESULT_DIR=.../mi355x-c128-kvfp8  RESULT_FILENAME=..._kvfp8  bash dsv4_fp4_mi355x_sglang_mtp.sh
```

Arm B at 640 B is still 9.6 % above B200's 584 B. State that residual; there is
no flag that closes it. The fp8 layout's measured capacity ratio is 1.546x
(`../DSV4_UNIFIED_KV_FP8_PAIR.md`, table 2) against the 1.50x claimed by
sgl-project/sglang#37413.

---

## 6. Asymmetries no flag can remove — list these next to any reported gap

These are platform code paths, not tunables:

- **MoE.** MI355X DP8/EP1 runs aiter with `--enforce-shared-experts-fusion`.
  B200 DP runs `--moe-a2a-backend megamoe` + `--enable-w4a4-mxfp4-megamoe` +
  `--disable-shared-experts-fusion`. Different kernels **and** different expert
  quantisation.
- **Attention backend.** MI355X `--attention-backend dsv4` (HIP radix) with
  `--page-size 256`; B200 leaves the default and sets no page size.
- **SWA ratio.** 0.10 on MI355X vs 0.02 on the B200 DP branch — each platform's
  own AgentX value. Override `SWA_FULL_TOKENS_RATIO_DP` to align if you want
  that arm.
- **KV layout.** §5.

---

## 7. What was deliberately made identical, and what still needs confirming

Decided and hard-coded the same on both sides:

| Knob | Value | Reason |
|---|---|---|
| sglang-router | **not used** on either side | Fixed length makes every request identical, so cache-aware routing has nothing to route on. Non-PD `load_balance_method` resolves to `round_robin` (`srt/arg_groups/serving_hook.py:200`), so DP attention works with the client hitting the backend port directly. |
| Prefix cache | `--disable-radix-cache`, no HiCache | Random fixed-length prompts share no prefix; a cache would only add nondeterministic hits. |
| Chunked prefill | `CHUNK_PER_RANK=8192` → `8192*TP` | Chunk size sets the prefill GEMM/attention shape and must match across platforms. **The B200 AgentX launcher uses 6144/rank — this is a deliberate deviation on that side.** |
| Acceptance length | `SGLANG_SIMULATE_ACC_LEN=3.77`, `match-expected`, `real-draft-token` | Same as both AgentX launchers; removes per-platform AL drift so both verify the same tokens per step. |
| DSpark | gamma 6, num-steps 1, eagle-topk 1, draft-tokens 7 | Identical in both AgentX launchers. |
| `num-prompts` | `3*CONC` (384 at c128) | The other `fixed_seq_len` scripts use `CONC*10`; at 128k that is impractical. |
| `max-running-requests` | `CONC` | AgentX's `2*CONC` headroom exists for subagent fan-out inside a session tree; there is none here. |
| `--chat-template` | **not passed** on either side | The client posts to `/v1/completions` and `--dsv4` frames prompts client-side, so a server-side template is never consulted. The B200 AgentX launcher passes one and MI355X does not; dropping it removes that asymmetry. |
| `--enable-prefill-delayer` | **off** (set `ENABLE_PREFILL_DELAYER=1` to restore) | The MI355X AgentX DP branch enables it; the B200 launcher never does, in either branch. Enabling on one side only would be a one-sided scheduling change layered on top of the kernel comparison. |

**Still unconfirmed — B200 owners, please answer before the first comparison:**

1. **`EP_SIZE` at c128.** The B200 DP branch passes `--ep-size $EP_SIZE`
   unconditionally. Every MI355X AgentX arm on file runs `ep_size=1`. The §3
   B200 block guesses 8; correct it and tell the AMD side.
2. **`CHUNK_PER_RANK`.** We force 8192 for symmetry; your AgentX launcher uses
   6144. Confirm 8192 is acceptable, or we run both at 6144 instead.
3. **`--enable-prefill-delayer`.** Off here. Confirm your AgentX arms also do
   not use it.

Any answer that changes a knob: change it **in this folder**, note it here, and
re-run both sides.

---

## 8. Status

Both scripts are syntax-checked and **have not been run**. No numbers exist yet
on either platform. First action is the MI355X c128 arm A in §3/§5.

Related: `../SKILL.md` (AgentX environment and traps),
`../DSV4_UNIFIED_KV_FP8_PAIR.md` (the unified-KV bf16-vs-fp8 measurements).
