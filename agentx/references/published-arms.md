# Published-arm reproduction — how to run the leaderboard points and what we got

_Split out of `SKILL.md` on 2026-08-28. Section numbers are
preserved so every `§N` cross-reference in the skill still resolves._

## 10. Reproducing the published inferencex.semianalysis.com points

The leaderboard row set for **MI355X (SGLang) / DeepSeek-V4-Pro / FP4 /
agentic** is exactly the search space of config key
`dsv4-fp4-mi355x-sglang-agentic-mtp` in `configs/amd-master.yaml` — 8 points,
no others:

| tp | conc | tokens/USD | p90 interactivity (tok/s/user) | throughput/chip (tok/s) |
|---|---|---|---|---|
| 4 | 1  |  6,783,422 | 122  |  2,826.4 |
| 4 | 2  |  7,410,968 |  98.6 |  3,087.9 |
| 4 | 4  | 12,967,114 |  93.7 |  5,403.0 |
| 4 | 8  | 21,499,766 |  58.2 |  8,958.2 |
| 4 | 10 | 26,691,545 |  50.2 | 11,121.5 |
| 8 | 16 | 21,770,384 |  55.7 |  9,071.0 |
| 8 | 32 | 37,610,957 |  33.1 | 15,671.2 |
| 8 | 48 | 44,965,881 |  24.4 | 18,735.8 |

### Which env goes with which arm

The two arms differ by **more than TP** — the TP8 arm runs HiCache:

```bash
# arm A — TP4, no host KV tier, conc in {1,2,4,8,10}
export TP=4 EP_SIZE=1 DP_ATTENTION=false
export KV_OFFLOADING=none
export TOTAL_CPU_DRAM_GB=1199        # informational for this recipe, see below

# arm B — TP8 + HiCache DRAM tier, conc in {16,32,48}
export TP=8 EP_SIZE=1 DP_ATTENTION=false
export KV_OFFLOADING=dram
export KV_OFFLOAD_BACKEND=hicache
export TOTAL_CPU_DRAM_GB=2399

# both arms
export DURATION=3600                 # scenario minimum is 900 s
export MODEL_PREFIX=dsv4 SPEC_DECODING=mtp
```

`TOTAL_CPU_DRAM_GB` comes from the generator formula in
`benchmarks/single_node/agentic/README.md`:
`floor(min(available_MiB, 2861022) * 1048576 * utilization * tp / gpus_per_node / 1e9)`
with `dram-utilization: 0.8` and the `cluster:mi355x-amds` hardware entry in
`configs/runners.yaml` (`available-cpu-dram-mib: 3095781`, capped at the 3 TB
decimal limit) → **1199 GB at TP4, 2399 GB at TP8**. DSv4 SGLang is the
documented exception that ignores the byte budget and sizes its host tier from
`--hicache-ratio` (1.5) instead; `KV_OFFLOADING=dram` is what actually turns
HiCache on.

**`TP=8` with `KV_OFFLOADING=none` reproduces nothing.** PR #2713: *"add a TP4
arm at [1, 2, 4, 8, 10] with no host KV tier … and remove the TP8 no-offload
arm."*

### Where the leaderboard columns come from

Two of the three are read straight out of
`$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json`:

| Leaderboard column | JSON path |
|---|---|
| p90 interactivity (tok/s/user) | `request_metrics.latency.intvty.p90` |
| throughput/chip (tok/s) | `request_metrics.throughput.per_gpu.total_tput_tps` |
| total tokens per USD | **not in this repo** — derived on the site from throughput and chip pricing |

Note `per_gpu.total_tput_tps` is **(input + output) tokens / s / GPU**. With a
~90 K-token mean ISL the agentic corpus is prefill-dominated, which is why the
published per-chip numbers are in the thousands while output-only throughput is
two orders of magnitude lower (smoke run: input 9264 tok/s vs output 84 tok/s).

### Known non-reproducible deltas on this node

- **Image.** The key pins `lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260822`;
  this container is sglang `0.5.18.dev20260825+g0c7ff19e3b` — a different build,
  three days later. Numbers will not be bit-comparable.
- **Runner.** Official runs are on the `cluster:mi355x-amds` fleet.
- **Metadata fields are CI-supplied.** `hw`, `image`, `framework`, `precision`,
  `recipe_fingerprint` come out empty, and `spec_decoding` reads `none` even
  though EAGLE/MTP was active, unless you export `SPEC_DECODING=mtp` yourself.
  Measurement is unaffected; result labelling is not.

Working in our favour: throughput runs pin acceptance length with
`SGLANG_SIMULATE_ACC_LEN=2.49`, so speculative-decode acceptance is simulated
rather than measured, and does not vary between machines.

## 11. Verified smoke run (2026-08-26, this node)

`TP=8 CONC=2 KV_OFFLOADING=none DURATION=300 AIPERF_WARMUP_REQUESTS_PER_LANE=1`

Timeline — total ~40 min wall clock:

| Phase | Duration |
|---|---|
| GPU drain gate | 10 s (idle node) |
| deps install (warm uv cache) + dataset (cached) | ~40 s |
| weight load | 628–1213 s per rank, 111.22 GB/rank |
| fp8 dequant + aiter JIT + cuda graph → `ready to roll` | 1410 s cumulative |
| AIPerf configure + warmup | ~150 s (4 requests) |
| profiling | 330 s (32 requests, 0 errors) |
| aggregation + validation | ~30 s |

Server at readiness: `max_total_num_tokens=10826240`, `context_len=1048576`,
`max_running_requests=4` (= 2×CONC).

Result: `intvty.p90 = 134.37 tok/s/user`, `per_gpu.total_tput_tps = 1168.52`,
`gpu_cache_hit_rate = 0.839`, mean ISL 89,828 / mean OSL 814.
Not comparable to any published row (removed arm, 300 s instead of 3600 s,
warmup 1/lane instead of 10) — it validates the pipeline, not the numbers.

Artifacts produced, all present:
`server.log`, `sglang_command.txt`, `benchmark.log`, `benchmark_command.txt`,
`gpu_metrics.csv` + energy start/end + `gpu_metrics_identity.json`,
`power_validation.json`, `metrics_plots.png`,
`workload_distribution_plots.png` / `_summary.txt`, `aiperf_artifacts/`,
and the aggregate `<RESULT_FILENAME>.json`.

### Two operational traps confirmed here

- **The launcher does not stop the server.** After
  `run_agentic_replay_and_write_outputs` returns, the script exits and leaves
  `python3 -m sglang.launch_server` running with all 8 GPUs at ~97 % VRAM. CI
  gets away with it because the container is torn down. Interactively you must
  kill it, or the **next** run's drain gate spends its full 15 min and then
  aborts. Reclaim after `SIGTERM` took under a minute here.
- **`pgrep -f <script>` self-matches.** Checking liveness with
  `pgrep -f agentx_smoke.sh` from a shell whose own command line contains that
  string returns the checking shell's PID and makes a dead run look alive. Match
  on the PID you captured at launch, or confirm with `rocm-smi` VRAM instead.

## 12. Verified reproduction of a published point (2026-08-26)

Arm A, `TP=4 CONC=10 KV_OFFLOADING=none TOTAL_CPU_DRAM_GB=1199 DURATION=3600`
via `/workspace/claude-skills/agentx/agentx_run.sh`, against the published
`dsv4-fp4-mi355x-sglang-agentic-mtp` row for tp4/conc10:

| Metric | Published | Measured here | Delta |
|---|---|---|---|
| p90 interactivity (tok/s/user) | 50.2 | 51.88 | +3.3 % |
| throughput/chip (tok/s) | 11,121.5 | 11,628.6 | +4.6 % |

Run quality: 1236/1345 requests successful, **0 errors**, profiling 3640 s,
`validate_agentic_result` passed (`0/1236 = 0.000%`). Supporting numbers —
whole-node 46,514 tok/s ÷ 4 GPUs; input 11,540 + output 88 tok/s/GPU (confirming
`per_gpu.total_tput_tps` is input+output); ISL mean 134,980, OSL mean 1,032;
prefix cache hit rate 0.9585; TTFT p50 0.510 s / p90 1.440 s; TPOT p90 0.01928 s.

Both metrics land slightly high, in the same direction, on a build three days
newer than the pinned image and on a node outside the `cluster:mi355x-amds`
fleet. **Agreement within ~5 % is the expected reproduction quality here** —
treat a larger gap as a configuration problem, not machine variance.

This also settles the README contradiction: the "results are not published on
inferencex.com" disclaimer in `benchmarks/single_node/agentic/README.md` is
stale for this config key, which is changelog-governed (PR #2600, #2713) and
whose full 8-point search space matches the published leaderboard rows exactly.

Timings for planning a full sweep (TP4, weights warm in page cache):
server ready 855 s → warmup (10 req/lane, 109 requests) → 3640 s profiling →
aggregation. Wall clock 3940 s ≈ 66 min per point. A first run on a cold page
cache adds ~10 min (TP8 cold took 1410 s to ready). Budget ~9 h for all 8
points, and remember to kill the server between them (§11).

