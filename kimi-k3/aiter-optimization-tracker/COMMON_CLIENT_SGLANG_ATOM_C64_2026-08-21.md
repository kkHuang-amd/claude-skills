# Common-client SGLang versus ATOM C64 — 2026-08-21

## Decision

The apparent `>9%` ATOM throughput lead at C64 was primarily a native-client
benchmark artifact, not an engine throughput gap.

With one common streaming OpenAI client, one persisted prompt manifest, and
the same current AITER base, total-token throughput is effectively tied:

```text
SGLang: 9800.69 tok/s
ATOM:   9846.50 tok/s
ATOM delta: +0.47%
```

Both engines completed 512/512 requests with no client contract failures.
The remaining 0.47% is below normal endpoint noise and was not measured in an
interleaved ABBA order.

The engines do have a large latency-shape difference:

```text
median TTFT: SGLang 27579.66 ms, ATOM 14689.19 ms  (ATOM -46.74%)
median TPOT: SGLang    31.74 ms, ATOM    44.70 ms  (ATOM +40.82%)
median E2E:  SGLang 60029.54 ms, ATOM 59643.74 ms  (ATOM -0.64%)
```

SGLang batches/delays most first tokens and then decodes faster. ATOM staggers
first-token delivery across each C64 cohort and decodes more slowly. Their
approximately 60-second median E2E and makespan are nearly identical.

## Controlled variables

```text
SGLang: 455b744aa77b2078de7577619dc12d2775fc1091
ATOM:   27f8639bad0948755630236aa7796b4b21348668
AITER:  dc4bdf1c142181ad90b7f6948564126df4c05fde
AITER path for both:
  /sgl-workspace/aiter-atom-current/aiter/__init__.py
FlyDSL for both: 0.3.1
GPU: 8x AMD Instinct MI355X / gfx950
```

SGLang used AITER prefill, tuned Triton decode, the validated K3 profile, and
FULL decode graphs through BS256. ATOM used the Kimi-K3 recipe, AITER MLA,
FULL graphs through BS64, and single-stream MoE:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

The engine recipes are not numerically identical: ATOM uses its PTPC-FP8
online-quant recipe while SGLang uses the retained K3 MXFP4/A8W4 profile.
Using the same AITER removes library revision as a variable, not framework
scheduling, backend selection, or precision policy.

## Common client and workload

Client:

```text
/workspace/useful-scripts/benchmarking/common_oai_benchmark.py
```

The tool uses the same streaming `POST /v1/completions` request lifecycle,
prompt manifest, SSE validation, and TTFT/TPOT calculation for both engines.
Kimi-K3 required adding the general CLI option:

```text
--trust-remote-code
```

Validation:

```text
common-client unit tests: 2 passed
successful requests:      512/512 per engine
failed requests:          0
SSE [DONE]:               required
non-empty text:           required
server prompt tokens:     exactly 8192
completion tokens:        exactly 1024
```

Workload:

```text
input/output:       exact 8192/1024
concurrency:        64
warmup requests:    128
measured requests:  512
seed:               42
request rate:       saturated
```

Shared manifest:

```text
path:
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/prompt-manifest-c64-8k.jsonl.gz
count: 640
gzip SHA-256:
  a0ffe42a7a9bcaea5e0797172dc7522716feb172dbe4a3ebddab60205828d071
prompt-content SHA-256:
  e0cc6508eed7c019c221842d0fecf29e867a5e71ccca5021f253fca2174f4e8f
```

## Aggregate result

```text
metric                         SGLang        ATOM          ATOM delta
successful requests            512           512          equal
failed requests                  0             0          equal
duration                       481.46 s      479.22 s      -0.47%
request throughput               1.0634        1.0684      +0.47%
input throughput              8711.72       8752.44        +0.47%
output throughput             1088.97       1094.06        +0.47%
total-token throughput        9800.69       9846.50        +0.47%
median TTFT                  27579.66 ms   14689.19 ms    -46.74%
median TPOT                     31.74 ms      44.70 ms    +40.82%
median E2E                   60029.54 ms   59643.74 ms     -0.64%
```

## Why the native-client comparison looked different

Earlier native-client results were:

```text
SGLang native bench_serving: 9041.51 tok/s
ATOM native benchmark:       9878.91 tok/s
apparent ATOM lead:             +9.26%
```

Relative to those:

```text
SGLang common client vs native: +8.40%
ATOM common client vs native:   -0.33%
```

The common client raises SGLang to parity while leaving ATOM essentially
unchanged. Therefore the old cross-framework throughput comparison mixed
client prompt/request scheduling and accounting with engine performance.
Cross-engine claims must use the common client and persisted manifest.

## Request timeline analysis

Per-request records independently reproduce each saved summary. Median cohort
behavior for each consecutive group of 64 requests:

```text
SGLang first-token waves:
  27.72, 88.23, 148.42, 208.52, 268.39, 328.35, 389.39, 449.35 s
SGLang completion waves:
  60.66, 120.80, 180.88, 240.82, 300.75, 361.81, 421.78, 481.45 s

ATOM first-token waves:
  14.36, 73.98, 133.59, 193.16, 254.57, 314.12, 373.70, 433.78 s
ATOM completion waves:
  59.70, 119.32, 178.88, 240.25, 299.84, 359.42, 419.09, 479.21 s
```

ATOM TTFT is strongly ordered by slot within each 64-request cohort
(`Spearman 0.905`): it staggers first tokens from about 1.1 to 27.7 seconds.
SGLang emits at least 75% of first tokens close to 27.6 seconds, then catches
up through its lower TPOT. Completion waves are tight for both.

This is consistent with different server scheduling/batching policies, not a
common-client timestamp error. TTFT is based on the first observed non-empty
SSE text chunk, so transport chunking can affect small differences, but not
the observed 13-second median split.

## Implications

1. There is no demonstrated 10% C64 throughput gap after client and AITER
   control; current evidence is parity within 0.5%.
2. Use the common client for every future SGLang/ATOM performance claim.
3. Optimize latency policy according to product target:
   - ATOM favors earlier first tokens.
   - SGLang favors faster decode after a delayed first-token wave.
4. A scheduler/config A/B is more valuable than copying kernels if TTFT is the
   target.
5. For kernel attribution, capture matched traces using this exact manifest
   and a representative C64 cohort; do not compare native-client traces.

## Artifacts

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/
  prompt-manifest-c64-8k.jsonl.gz
  sglang-same-aiter/common-client/{summary.json,requests.jsonl}
  atom-same-aiter/common-client/{summary.json,requests.jsonl}
  sglang-same-aiter/server.log
  atom-same-aiter/server.log
  sglang-same-aiter-run-v2.log
  atom-same-aiter-run-v2.log
  run_engine.sh
```

The initial different-AITER attempt and the first tokenizer-trust failure are
retained as diagnostics. No server or client process remains; GPUs and
experiment caches were released.
