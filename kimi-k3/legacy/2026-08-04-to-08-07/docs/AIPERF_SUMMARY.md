# Kimi-K3 cfg7 vs DCP8 aiperf reproduction

Date: 2026-08-05

## Scope

- Hardware: 8x MI355X
- SGLang target: `/sgl-workspace/sglang`, `e6311f7559` plus the local AMD
  Kimi-K3 DCP port
- Client: aiperf 0.11.0, transformers 5.12.1, huggingface_hub 1.24.0,
  tiktoken 0.13.0
- Workload: 8 prefixes, prefix length 63,240, synthetic input 4,760,
  output 350, warmup 32
- Concurrency/request pairs: 6/60, 12/72, 24/96, 32/128
- Both sweeps completed all four points with zero request errors.

This is a same-code A/B comparison on the current working tree. It is not an
exact checkout of the historical clean `0e756912` tree named in
`/sgl-workspace/REPRODUCE.md`.

## Current cfg7 baseline

- Concurrency 6: output 69.21 tok/s; TTFT p50/p90 10.09/28.61 s;
  ITL p50 38.74 ms; cache hit 73.4%.
- Concurrency 12: output 82.27 tok/s; TTFT p50/p90 24.32/47.99 s;
  ITL p50 68.14 ms; cache hit 75.0%.
- Concurrency 24: output 113.51 tok/s; TTFT p50/p90 27.04/90.23 s;
  ITL p50 96.17 ms; cache hit 80.9%.
- Concurrency 32: output 101.72 tok/s; TTFT p50/p90 44.73/109.75 s;
  ITL p50 138.96 ms; cache hit 80.0%.

Artifacts:
`/sgl-workspace/aiperf-results/cfg7-current-baseline/artifacts_run2`.

## DCP8 AITER

The cfg7 workload was unchanged. Server differences were `--dcp-size 8`,
`--attention-backend aiter`, and `--mem-fraction-static 0.93`. The requested
0.85 failed before startup because DCP reduced the resolved static fraction and
left negative memory for the hybrid Mamba cache.

- Concurrency 6: output 43.12 tok/s; TTFT p50/p90 18.96/62.94 s;
  ITL p50 29.72 ms; cache hit 77.9%; throughput delta -37.7%.
- Concurrency 12: output 50.66 tok/s; TTFT p50/p90 39.84/110.30 s;
  ITL p50 37.46 ms; cache hit 79.3%; throughput delta -38.4%.
- Concurrency 24: output 67.08 tok/s; TTFT p50/p90 61.50/220.69 s;
  ITL p50 59.62 ms; cache hit 83.3%; throughput delta -40.9%.
- Concurrency 32: output 72.46 tok/s; TTFT p50/p90 71.69/234.20 s;
  ITL p50 73.02 ms; cache hit 85.2%; throughput delta -28.8%.

Artifacts: `/sgl-workspace/aiperf-results/cfg7-current-dcp8/artifacts_run1`.

## Interpretation

DCP8 improves median decode ITL at every concurrency, but loses aggregate
output throughput because TTFT and queueing increase substantially. The
baseline reported 1,225,230 KV tokens. DCP8 reported 511,680 physical tokens
per rank; with eight DCP shards its aggregate virtual KV capacity is about
4,093,440 tokens, so it does not have less total KV capacity than the baseline.
The remaining constraints are slower DCP prefill and an effective
running-request cap of 29 from the Mamba state pool.

Peak output throughput was 113.51 tok/s for the baseline at concurrency 24 and
72.46 tok/s for DCP8 at concurrency 32, a 36.2% reduction.

## DCP page-size A/B at concurrency 24

A matched standalone A/B used the same random seed, 32 warmup requests, 96
profile requests, and otherwise identical DCP8 settings.

- Page size 32: output 63.88 tok/s; TTFT p50/p90 61.48/227.23 s;
  ITL p50 59.43 ms; cache hit 82.7%.
- Page size 64: output 59.86 tok/s; TTFT p50/p90 64.57/235.67 s;
  ITL p50 86.42 ms; cache hit 82.3%.
- Page size 64 changed output throughput by -6.3%, TTFT p50 by +5.0%,
  TTFT p90 by +3.7%, and ITL p50 by +45.4%.

Conclusion: page size 64 does not help this workload. Keep the DCP default at
32.

Artifacts:

- `/sgl-workspace/aiperf-results/cfg7-current-dcp8-page32-c24-matched`
- `/sgl-workspace/aiperf-results/cfg7-current-dcp8-page64-c24`

## Tuned DCP8 at concurrency 24

The tuned launch uses Triton prefill with AITER DCP decode, caps the Mamba pool
to the 120 slots needed by 24 requests, removes the INT8 Mamba checkpoint,
requests `mem-fraction-static=0.88` (resolved to 0.748 by the AITER long-context
safety factor), and captures decode graphs through batch 24.

The full OSL 350, warmup 32, request-count 96 confirmation used the same seed as
the matched current-DCP control:

- Current DCP: output 63.88 tok/s (7.98/GPU); TTFT p50/p90
  61.48/227.23 s; ITL p50 59.43 ms; cache hit 82.7%.
- Tuned DCP: output 101.67 tok/s (12.71/GPU); TTFT p50/p90
  26.17/145.87 s; ITL p50 67.29 ms; cache hit 81.9%.
- Tuned delta: output +59.2%, TTFT p50 -57.4%, TTFT p90 -35.8%, and
  ITL p50 +13.2%.

For reference, the current non-DCP cfg7 result at concurrency 24 was
113.51 tok/s, TTFT p50 27.04 s, and ITL p50 96.17 ms. The tuned DCP launch
therefore nearly matches baseline aggregate throughput, slightly improves
median TTFT, and retains lower median ITL.

Artifacts:
`/sgl-workspace/aiperf-results/dcp8-short-tuned-c24/full_artifacts`.

## Tuned DCP8 full sweep

To cover concurrency 32, the final sweep used requested
`mem-fraction-static=0.90` (resolved 0.765), `max-running-requests=32`,
`cuda-graph-max-bs=32`, and `max-mamba-cache-size=160`. Triton prefill,
AITER DCP decode, page size 32, and no INT8 Mamba checkpoint were retained.
All four points completed with zero errors.

- Concurrency 6: output 75.47 tok/s (9.43/GPU); TTFT p50/p90
  7.03/28.93 s; ITL p50 26.50 ms; cache hit 77.8%.
- Concurrency 12: output 97.38 tok/s (12.17/GPU); TTFT p50/p90
  15.95/51.25 s; ITL p50 43.38 ms; cache hit 80.8%.
- Concurrency 24: output 112.31 tok/s (14.04/GPU); TTFT p50/p90
  26.18/131.47 s; ITL p50 70.09 ms; cache hit 83.0%.
- Concurrency 32: output 114.23 tok/s (14.28/GPU); TTFT p50/p90
  34.35/114.38 s; ITL p50 117.31 ms; cache hit 85.3%.

The tuned DCP8 peak is 0.6% above the current non-DCP peak. Median TTFT and
median ITL are lower than non-DCP at every concurrency; TTFT p90 remains worse
at concurrency 12, 24, and 32, most notably at concurrency 24.

Artifacts:
`/sgl-workspace/aiperf-results/cfg7-current-dcp8-tuned-all/artifacts`.

## Custom 132k context / Mamba 320 full sweep

This sweep used the user-supplied launch profile: context length 132,000,
requested `mem-fraction-static=0.98` (resolved 0.833), Mamba cache 320,
decode graph max batch 64, Triton prefill, AITER DCP decode, and no INT8 Mamba
checkpoint. All four points completed with zero errors.

- Concurrency 6: output 75.33 tok/s (9.42/GPU); TTFT p50/p90
  7.72/26.28 s; ITL p50 33.13 ms; cache hit 78.8%.
- Concurrency 12: output 94.22 tok/s (11.78/GPU); TTFT p50/p90
  14.73/62.71 s; ITL p50 45.49 ms; cache hit 81.1%.
- Concurrency 24: output 120.02 tok/s (15.00/GPU); TTFT p50/p90
  22.51/112.68 s; ITL p50 80.41 ms; cache hit 84.1%.
- Concurrency 32: output 121.46 tok/s (15.18/GPU); TTFT p50/p90
  32.95/115.69 s; ITL p50 126.40 ms; cache hit 86.4%.

Peak throughput is 7.0% above the current non-DCP peak and 6.3% above the
previous tuned DCP peak. TTFT p50 is lower than non-DCP at every concurrency;
TTFT p90 is better at concurrency 6 but remains higher at 12, 24, and 32.

Artifacts:
`/sgl-workspace/aiperf-results/custom-ctx132k-mamba320-all/artifacts`.

## Custom 132k + PR 33599 full sweep

PR 33599 (`[AMD] Fuse Kimi-K3 attn-residual aggregation`) was semantically
ported onto the local Kimi-K3 branch without creating a commit. The new ROCm
kernel passed BF16 parity checks at `nvb=1,4,8`. The server and aiperf commands
were otherwise identical to the Custom 132k / Mamba 320 run. All four points
completed with zero errors.

- Concurrency 6: output 79.43 tok/s (9.93/GPU); TTFT p50/p90
  7.52/27.59 s; ITL p50 31.66 ms; cache hit 78.9%.
- Concurrency 12: output 100.69 tok/s (12.59/GPU); TTFT p50/p90
  14.00/52.83 s; ITL p50 43.04 ms; cache hit 81.2%.
- Concurrency 24: output 122.04 tok/s (15.25/GPU); TTFT p50/p90
  21.78/110.75 s; ITL p50 79.67 ms; cache hit 84.0%.
- Concurrency 32: output 142.41 tok/s (17.80/GPU); TTFT p50/p90
  31.14/111.70 s; ITL p50 89.39 ms; cache hit 86.3%.

Relative to the same Custom launch before the PR, throughput changed by +5.4%,
+6.9%, +1.7%, and +17.2% at concurrency 6/12/24/32. Concurrency-32 median ITL
improved by 29.3%. Peak throughput is 25.5% above the current non-DCP peak.

Artifacts:
`/sgl-workspace/aiperf-results/custom-ctx132k-mamba320-pr33599-all/artifacts`.

## Custom 132k + PR 33599 + FP8 KV full sweep

The same Custom + PR launch was tested with
`--kv-cache-dtype fp8_e4m3`. AITER MLA DCP rejects FP8 KV by default because
the path was previously unvalidated, so this run used the explicit local
experimental opt-in `SGLANG_EXPERIMENTAL_AITER_DCP_FP8=1`.

FP8 doubled per-rank token capacity from 454,528 to 909,088 while keeping KV
memory at 11.70 GB. Chat smoke passed, GSM8K 200 scored 0.990, and all four
aiperf points completed with zero errors.

- Concurrency 6: output 91.08 tok/s (11.39/GPU); TTFT p50/p90
  5.47/21.07 s; ITL p50 30.32 ms; cache hit 78.8%.
- Concurrency 12: output 120.70 tok/s (15.09/GPU); TTFT p50/p90
  10.53/35.41 s; ITL p50 39.64 ms; cache hit 81.2%.
- Concurrency 24: output 150.83 tok/s (18.85/GPU); TTFT p50/p90
  15.82/96.76 s; ITL p50 64.67 ms; cache hit 84.0%.
- Concurrency 32: output 172.68 tok/s (21.58/GPU); TTFT p50/p90
  24.53/90.90 s; ITL p50 80.42 ms; cache hit 86.0%.

Relative to BF16 KV, FP8 KV improved throughput by 14.7%, 19.9%, 23.6%, and
21.3% at concurrency 6/12/24/32. TTFT p50 improved by 21–27% and median ITL
improved by 4–19%. Peak throughput is 52.1% above the current non-DCP peak.

Artifacts:
`/sgl-workspace/aiperf-results/custom-ctx132k-mamba320-pr33599-fp8kv-all/artifacts`.

## FP8 KV long-context accuracy validation

A deterministic BF16-vs-FP8 comparison placed unique retrieval codes at 10%,
50%, and 90% of four context sizes. Actual server prompt lengths were
approximately 7.66k, 31.17k, 64.87k, and 114.61k tokens.

- BF16 KV exact retrieval: 12/12.
- FP8 KV exact retrieval: 12/12.
- BF16/FP8 response text identical: 12/12.
- Prompt token counts identical: 12/12.
- Maximum aligned output-token logprob difference: 0.002064.
- Mean absolute aligned logprob difference: 0.0000434.
- Total sequential latency: BF16 111.14 s; FP8 92.80 s (-16.5%).
- At the 114.6k prompt length, mean latency was BF16 20.15 s versus FP8
  15.34 s (-23.9%).

Together with GSM8K 200 = 0.990, these results show no detected accuracy
regression through 114.6k prompt tokens. The retrieval suite is synthetic and
does not replace a broad long-context reasoning or multimodal evaluation.

Artifacts and reusable harness:
`/sgl-workspace/aiperf-results/context_accuracy`.

## Custom 132k + PR 33599 + FP8 KV + DSPARK ReplaySSM

The speculative run added DSPARK block size 7 and ReplaySSM spec-verify. The
current CLI flag is `--enable-gdn-replayssm-spec`; the model README's
`--enable-linear-replayssm-spec` spelling is stale. DCP requires
`SGLANG_RAGGED_VERIFY_MODE=static`, and the run also used
`SGLANG_PREP_IN_CUDA_GRAPH=1`.

The server resolved to 48 effective running requests, FP32 Mamba state, and
1,127,168 aggregate KV tokens after allocating the ReplaySSM rings. Batch-8
smoke observed mean accept length 4.01; the full aiperf run ended at average
accept length 2.84. GSM8K 200 scored 0.985. All four sweep points completed
with zero errors.

- Concurrency 6: output 94.65 tok/s (11.83/GPU); TTFT p50/p90
  1.04/9.47 s; ITL p50 37.78 ms; cache hit 78.6%.
- Concurrency 12: output 115.36 tok/s (14.42/GPU); TTFT p50/p90
  1.34/35.59 s; ITL p50 60.61 ms; cache hit 80.9%.
- Concurrency 24: output 139.01 tok/s (17.38/GPU); TTFT p50/p90
  2.10/94.79 s; ITL p50 102.26 ms; cache hit 84.1%.
- Concurrency 32: output 160.92 tok/s (20.12/GPU); TTFT p50/p90
  2.94/84.96 s; ITL p50 128.11 ms; cache hit 86.4%.

Relative to the same FP8 KV launch without speculation, output throughput was
+3.9%, -4.4%, -7.8%, and -6.8% at concurrency 6/12/24/32. DSPARK substantially
reduced median TTFT, but average accept length 2.84 was insufficient to offset
verify overhead at concurrency 12 and above. AIPerf ITL is also affected by
speculative chunk streaming and should not be interpreted as a one-token decode
step in this comparison.

Artifacts:
`/sgl-workspace/aiperf-results/custom-ctx132k-pr33599-fp8kv-dspark-replayssm-all/artifacts`.

## DSPARK 8k/1k accept-length measurement

The non-DCP Triton launch supplied by the user was tested with Radix Cache
disabled, DSPARK block size auto-inferred as 7, 7,998 actual prompt tokens,
1,000 forced output tokens, concurrency 32, and 128 measured requests.

- Mean accept length: 5.6779.
- Median accept length: 7.6336.
- Minimum/maximum per-request accept length: 3.0675 / 7.8125.
- Total elapsed time: 347.91 s.
- Mean request latency: 82.60 s.
- Total generated tokens: 128,000.

This closely reproduces the checkpoint README's GSM8K accept-length figure
5.666 and confirms that the earlier 2.84 result was specific to the 68k
long-context workload, not a general DSPARK failure.

Artifacts:

- `/sgl-workspace/aiperf-results/dspark-8k1k-c32-exact.json`
- `/sgl-workspace/aiperf-results/dspark-8k1k-c32-exact.log`

## Standard SGLang 8192/1024 benchmark

The DSV4 methodology was then reproduced with
`python -m sglang.benchmark.serving`: fixed 8,192 input tokens, fixed 1,024
output tokens (`random-range-ratio=1.0`), concurrency 32, 64 warmups, and 256
measured requests. Radix Cache remained disabled.

- Successful requests: 256/256.
- Benchmark duration: 543.07 s.
- Total input tokens: 2,097,152.
- Total output tokens: 262,144.
- Input throughput: 3,861.65 tok/s.
- Output throughput: 482.71 tok/s.
- Total token throughput: 4,344.35 tok/s (543.04 tok/s/GPU).
- Median TTFT: 1,629.41 ms.
- Median TPOT: 59.90 ms.
- Median ITL: 42.97 ms.
- Scheduler average accept length after warmup + measured run: 7.0083.

Artifacts:
`/sgl-workspace/aiperf-results/dspark-bench-serving-8192-1024-c32`.
