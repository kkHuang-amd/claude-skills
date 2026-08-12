# DeepSeek-V4 A2A Expert Parallelism — HANDOVER (2026-07-24)

This document covers A2A EP backends for DeepSeek-V4:

```text
A2A EP
├─ mori-ep      separate mori dispatch/combine + aiter MoE
├─ flydsl-ep    separate FlyDSL dispatch/combine + aiter MoE
└─ mega-moe     fused FlyDSL dispatch/GEMM/combine
```

MegaMoE is one A2A EP implementation, not the parent category. Its kernel-specific
optimization record remains under `dsv4/megamoe/`.

## 1. Current recommended FlyDSL-EP state

Environment:

- model: DeepSeek-V4-Pro A8W4
- hardware: 8x MI355X gfx950
- topology: TP8/DP8/EP8, DP attention
- target workload: 8192 input / 1024 output, concurrency 256

Status:

- correctness PASS
- decode deadlock fixed
- padded decode-M fixed
- rank-synchronized dynamic recv cap PASS
- prefill TBO functional and positive after stream/grid tuning

Repositories:

- `/sgl-workspace/sglang-flydsl-a2a`
  - `HaiShaw/sglang`, branch `feat/flydsl-a2a`
  - commits:
    - `807a16187` integration/correctness
    - `15a76b3f9` vendored FlyDSL A2A kernels
    - `d5021a439` TBO comm stream
    - `373e4ac27` TBO comm-grid tuning
- `/sgl-workspace/aiter`, pinned commit `9127c94a1`
  - fused MoE only; no A2A patch required
- FlyDSL runtime 0.2.4:
  - `/sgl-workspace/flydsl-0.2.4-py`
- mori.shmem:
  - symmetric allocation and P2P transport

Vendored A2A implementation:

```text
python/sglang/kernels/third_party/flydsl_a2a/
  communication_ops_utils.py
  flydsl_dispatch_combine_intranode_kernel.py
  flydsl_dispatch_combine_intranode_op.py
  mega_moe_tuning_config/
    flydsl_gfx950_mi355x_IntraNode_ep8.json
```

## 2. Final validated performance

Common harness:

- fixed random 8192/1024
- concurrency 256
- 2048 prompts / 512 warmups
- `sglang-oai`
- mem fraction 0.90
- CUDA graph ON
- radix cache OFF
- shared-expert fusion OFF

Current guarded-delayer comparison matrix:

| Backend | TBO | Delayer | Total tok/s | TPOT | TTFT | GSM8K |
|---|:---:|:---:|---:|---:|---:|---:|
| DP | off | on | 30,895.54 | 55.80 ms | 19.00 s | 0.943 |
| DP | on | on | **33,076.53** | **52.93 ms** | **16.91 s** | — |
| FlyDSL-EP | off | on | 31,227.35 | 55.87 ms | 18.17 s | 0.931 |
| FlyDSL-EP | on | off | 32,087.20 | 53.91 ms | 18.27 s | 0.940 |
| FlyDSL-EP | on | on | **32,368.95** | 54.36 ms | **17.30 s** | 0.936 |

TBO effect with guarded delayer:

- DP: +7.1% throughput, -5.1% TPOT, -11.0% TTFT
- FlyDSL-EP: +3.7% throughput, -2.7% TPOT, -4.8% TTFT

### FlyDSL-EP, no TBO

Delayer OFF baseline:

| Run | Total tok/s | Output tok/s | TPOT | TTFT |
|---|---:|---:|---:|---:|
| 1 | 30,772.64 | 3,419.18 | 55.55 ms | 19.83 s |
| 2 | 30,928.78 | 3,436.53 | 55.27 ms | 19.94 s |
| Mean | **30,850.71** | **3,427.86** | **55.41 ms** | **19.88 s** |

Recommended delayer ON with mixed-slot guard:

| Run | Total tok/s | Output tok/s | TPOT | TTFT |
|---|---:|---:|---:|---:|
| 1 | 31,215.12 | 3,468.35 | 55.91 ms | 18.17 s |
| 2 | 31,239.58 | 3,471.06 | 55.83 ms | 18.17 s |
| Mean | **31,227.35** | **3,469.71** | **55.87 ms** | **18.17 s** |

Guarded delayer vs delayer OFF: +1.2% throughput and -8.6% TTFT. GSM8K
0.931/invalid0. The DSV4 launcher enables delayer for FlyDSL unless `DELAYER=off`.

Historical same-harness DP reference from the original branch:

- 30,425.55 total tok/s
- 3,380.62 output tok/s
- 60.15 ms TPOT

DP regression isolation (2026-07-24):

| Tree | Commit/state | Total tok/s | TPOT | TTFT |
|---|---|---:|---:|---:|
| old clean base | `02236fa38` | 30,593.82 | 59.01 ms | 16.38 s |
| old dirty MegaMoE worktree | same base + local edits | 30,530.99 | 59.94 ms | 15.32 s |
| new clean pre-A2A base | `11b0e5c5a` | 27,697.37 | 75.09 ms | 2.25 s |
| FlyDSL-A2A HEAD, DP mode | A2A commits present, backend=none | 27,722.32 | 74.78 ms | 2.26 s |

Conclusions:

- FlyDSL-A2A commits change DP by +0.09%: no DP regression from this feature.
- Dirty MegaMoE edits are not the 10% difference: clean and dirty old base agree.
- The regression entered in upstream history `02236fa38..11b0e5c5a`.
- Both branches report identical delayer args and `prefill_delayer.py` is unchanged;
  the TTFT/TPOT shift points to an upstream scheduler/config interaction.
- Delayer-OFF 512-prompt screening looked fast (30.8k), but the full 2048-prompt run
  degraded to 20.0k with wave-like stalls. Disabling delayer is not a sustained-load fix.
- Automated first-parent bisect used a validated 512-prompt proxy (good endpoint
  30.7k, bad endpoint 23.0k; threshold 27k) and found:
  - parent `927979e12`: 30,402.64 tok/s (GOOD)
  - first bad `d03c8cee8`: 23,086.85 tok/s (BAD)
  - commit: `Negotiate PrefillDelayer only after KV-budget admission checks (#31835)`
- That commit moves `PrefillDelayer.negotiate_should_allow_prefill()` from the start
  of `add_one_req()` to after all KV-budget gates. This changes the prefill/decode
  scheduling balance and matches the observed TTFT improvement / TPOT regression.
- Root mechanism: after KV admission, ranks commonly report `mixed`; the mixed branch
  times out after 30 passes and admits prefill while decode slots are still constrained.
  The old `all` branch kept slot-based decode protection until slots freed.
- Fix (current FlyDSL-A2A branch): preserve post-KV correctness, but gather the effective
  per-scheduler `max_running_requests` as a sixth DP field; when mixed ranks are under
  slot pressure, continue delaying instead of taking the generic mixed timeout. An env
  kill switch `SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD` defaults to true.
- Validation: original KV-reject unit tests PASS; new mixed-slot test PASS; full 2048
  prompts = 30,895.54 tok/s, TPOT 55.80ms, TTFT 19.00s; GSM8K 0.943/invalid0.
- Fixed timeout tuning is not a substitute: 100 passes gives 15.6k / 68.8s TTFT, and
  1000 passes over-delays. Simply disabling delayer also fails on sustained workloads.

### FlyDSL-EP + prefill TBO

Settings:

- `--enable-two-batch-overlap`
- `GPU_MAX_HW_QUEUES=5`
- `SGLANG_FLYDSL_TBO_BLOCK_NUM=64`
- delayer OFF for the original two-run tuning baseline; guarded delayer ON also validated

| Run | Total tok/s | TPOT | TTFT |
|---|---:|---:|---:|
| 1 | 32,133.00 | 53.96 ms | 18.16 s |
| 2 | 32,041.39 | 53.86 ms | 18.38 s |
| Mean | **32,087.20** | **53.91 ms** | **18.27 s** |

Guarded-delayer ON single validation:

- 32,368.95 total tok/s
- 54.36 ms TPOT
- 17.30 s TTFT
- GSM8K 0.936 / invalid 0

Relative to FlyDSL no-TBO:

- throughput +4.0%
- TPOT -2.7%
- TTFT -8.1%

These percentages use the delayer-OFF no-TBO baseline. Use the comparison matrix
above for the fully guarded-delayer A/B.

Correctness:

- FlyDSL no-TBO GSM8K: 0.936, invalid 0
- FlyDSL TBO block64 GSM8K: 0.940, invalid 0

Negative configurations:

- pre-fix (`d03c8cee8`) delayer ON: 26,534.68 tok/s, TPOT 76.99 ms
- TBO default grid: 30,405.56 tok/s after comm-stream fix
- TBO block32: 22,284.80 tok/s, under-parallelized

## 3. Dynamic recv-cap design

The physical symmetric buffers remain fully allocated. Decode CUDA graphs compile
dispatch and combine with a smaller shared logical cap:

```text
bound = max(dp_global_num_tokens) * dp_size
recv_cap = next_power_of_two(max(32, bound))
recv_cap = min(recv_cap, physical_cap)
```

Safety argument:

1. One source token is deduplicated by destination PE.
2. A destination receives at most one row per global source token.
3. Global tokens are bounded by `max(per-rank tokens) * dp_size`.
4. FlyDSL reuses SGLang's FlashInfer A2A DP-sync bookkeeping.
5. All ranks capture/replay the same graph tier and recv stride.
6. Dispatch encoding, returned views, and combine decoding use the same cap.

Eager/prefill retains the full cap. Dynamic sizing is CUDA-graph decode only.

## 4. TBO design

DSV4 TBO is prefill-only. Decode uses the normal CUDA-graph path.

Initial FlyDSL dispatcher accepted `async_finish` but did not use it. All A2A and
compute kernels ran on one stream; TBO doubled launches without hiding communication.

Final implementation:

- one shared comm stream for both TBO inner dispatchers
- ready/done events
- compute stream waits on comm completion
- event + Python-reference tensor lifetime
- no `record_stream(comm)` deferred-free pattern

Trace after stream fix:

- A2A stream: tid6
- compute stream: tid8
- most A2A events overlap compute

Grid tuning:

- default prefill geometry: dispatch 128 blocks, combine 256 blocks
- TBO optimum for this DSV4 shape: both 64 blocks
- block32 is too small

Matched trace, default vs block64:

- A2A: 674 -> 614 ms/forward
- overlap fraction: 29.1% -> 33.7%
- overlapped events: 1,786/1,952 -> 1,952/1,952
- other GEMM: 423 -> 338 ms/forward

FlyDSL A2A uses GPU CUs/cache/P2P traffic; it is not pure DMA. The standalone-optimal
grid was too large when overlapped with compute.

## 5. DP and mori controls

### DP TBO

Current branch, delayer ON in both arms:

- DP: 27,722 tok/s, TPOT 74.78 ms
- DP+TBO: 28,699 tok/s, TPOT 71.96 ms
- effect: +3.5% throughput

Trace:

- all 732 comm-stream NCCL events overlap compute
- ~49.5% NCCL time hidden
- NCCL work changes only 684 -> 730 ms/forward

NCCL also uses CUs, but its tuned channel/grid footprint interferes less than the
original FlyDSL A2A grid.

### Mori-EP

Mori-EP and FlyDSL-EP share the same high-level token-major contract:

- dispatch produces token-major recv rows
- aiter sorts and runs the expert GEMMs
- combine scatters/reduces to source tokens

Mori's decode-small-cap work identified the same padded-M problem. Fair mori serving
comparisons require the `feat/mori-ep-decode-small-cap` implementation; do not compare
against untuned fixed-cap mori.

Historical DSV4 mori-EP TBO was also negative before backend-specific tuning.

## 6. Condensed debug timeline

1. Microbench role
   - Initially benchmarked the wrong expert-major fixed-slot role.
   - Production SGLang uses token-major rows; corrected FlyDSL beat mori A2A 9–18%.
2. Decode hang
   - Kernel epochs/fences/geometry were investigated extensively.
   - Actual cause: FlyDSL missing from DeepSeek `_enable_a2a_moe`; wrong forward path
     executed a mismatched-size TP all-reduce.
3. Wrong output
   - Cross-run comparison incorrectly implicated combine.
   - In-run reference combine matched to BF16 noise.
   - Actual cause: FlyDSL missing from `_shared_expert_use_tp1`.
4. Decode performance
   - Initial FlyDSL serving was 14,934 tok/s / 139 ms TPOT.
   - Every decode MoE used padded M=32768.
   - Trace: input quant 47%, MoE GEMM 33%, sorting 2%, A2A only ~3%.
5. Sizing failures
   - local runner slice: OOB fault
   - shmem view slice: prefill write fault
   - local per-rank graph cap: rank-different stride crash
6. Final sizing
   - op-level cap + DP-synchronized graph tier
7. Ownership
   - A2A code vendored into SGLang; temporary aiter patch reverted
8. TBO
   - initial overlap=0
   - comm stream recovered ~5%
   - block64 CU tuning made TBO +4% over no-TBO

## 7. Main misjudgments and rules

1. Stall site is not root cause.
   - Log rank/model collective sequence before editing kernel waits.
2. Workaround sensitivity is not proof.
   - Delaying a hang with fences/geometry does not establish causality.
3. Cross-run numerical diffs are weak.
   - Use an in-run reference on identical tensors.
4. Backend allowlists are dangerous.
   - Audit every `is_mori()`/backend discriminator; prefer semantic helpers.
5. Do not infer architectural limits without a trace.
   - “Un-fused A2A is structurally too slow” was disproved; padded compute was 82%.
6. Collective bounds must use synchronized global metadata.
   - Local token counts cannot define shared encoding geometry.
7. Flags do not prove outcomes.
   - Verify TBO using stream IDs and measured temporal overlap.
8. Move one variable at a time.
   - Queue-cap and delayer controls were required for fair TBO attribution.
9. Do not transfer another backend's memory root cause without measurements.
   - FlyDSL shmem ownership differed from DP's old `record_stream` bug.
10. Less parallelism is not automatically better.
    - block64 won; block32 collapsed.

## 8. Reproduction

Launcher:

```bash
/dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
```

FlyDSL-EP:

```bash
MODE=flydsl PORT=8000 bash run_sgl_dsv4_unified.sh
```

FlyDSL-EP + TBO:

```bash
MODE=flydsl-tbo PORT=8000 bash run_sgl_dsv4_unified.sh
```

`flydsl-tbo` automatically sets `GPU_MAX_HW_QUEUES=5`, enables TBO, uses block64,
and enables the guarded delayer. `DELAYER=off` reproduces the original TBO tuning baseline.

Serving sweep:

```bash
WORKLOADS="8192:1024" CONCS="256" NP_MULT=8 WARM_MULT=2 \
BACKEND=sglang-oai \
bash sweep_dsv4_sglang_client.sh
```

Correctness:

```bash
python3 -m sglang.test.few_shot_gsm8k \
  --num-questions 1319 --parallel 256 --port 8000
```

Focused dynamic-cap test:

```text
test/manual/dsv4/test_flydsl_dynamic_recv_cap.py
```

## 9. Remaining work

1. block64 is validated only for DSV4-Pro, MI355X, EP8, 8k/1k c256.
2. TBO still pays split-MoE weight reload and disables cross-layer fused-mHC.
3. Other models/EP sizes/workloads need independent geometry sweeps.
4. Push/PR work remains pending.
5. The validation launcher and this handover are outside the SGLang repository.

## 10. Related documents

- `MORI_EP_DECODE_ROOTCAUSE.md`
- `TBO_RESEARCH.md`
- `PERF_REG_LOG.md`
- `megamoe/MEGAMOE_OPT_DIRECTIONS_HANDOVER.md`
- `megamoe/KERNEL_OWNER_DECODE_PLAN.md`
- `megamoe/GEMM_CODESIGN_WIP.md`
