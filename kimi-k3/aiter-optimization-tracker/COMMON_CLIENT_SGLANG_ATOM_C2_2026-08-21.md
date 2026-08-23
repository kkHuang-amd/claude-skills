# Common-client SGLang versus ATOM C2 — 2026-08-21

## Decision

Unlike C64, SGLang has a real low-concurrency performance lead after client,
prompt, and AITER controls:

```text
five-run median throughput: SGLang 1138.02 vs ATOM 768.04 tok/s
ATOM delta: -32.51%

median TTFT: SGLang 960.14 vs ATOM 870.18 ms
ATOM delta: -9.37%

median TPOT: SGLang 14.90 vs ATOM 22.62 ms
ATOM delta: +51.77%

median E2E: SGLang 16192.43 vs ATOM 23997.45 ms
ATOM delta: +48.20%
```

ATOM returns the first token about 90 ms earlier, but SGLang's much faster
decode dominates the 1024-token completion. All 80 measured requests per
engine succeeded.

The C64 common-client result showed throughput parity within 0.47%. Together,
the results establish a concurrency-dependent framework gap:

```text
C2:  SGLang +48.17% throughput over ATOM
C64: ATOM   +0.47% throughput over SGLang (within noise)
```

Client methodology explained the old C64 discrepancy, but it does not explain
the C2 gap.

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

SGLang used the validated K3 profile with AITER prefill, tuned Triton decode,
optional KDA/B2/front/MLA fusions, up-only latent MXFP4, and FULL decode graphs.
ATOM used the Kimi-K3 recipe, AITER MLA, PTPC-FP8 online quantization, FULL
graphs, and single-stream MoE:

```text
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

The numerical/backend recipes are not identical. Same AITER and common client
remove two major variables, while quantization policy, scheduler, graph
structure, and framework-owned K3 kernels remain intentionally representative
of each engine's selected profile.

## Common workload

```text
client:             common_oai_benchmark.py
input/output:       exact 8192/1024
concurrency:        2
warmup requests:    4 per round
measured requests:  16 per round
rounds:             5 per engine
seed:               42
request rate:       saturated
SSE [DONE]:         required
server prompt len:  exactly 8192
completion len:     exactly 1024
```

Shared manifest:

```text
path:
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/prompt-manifest-c2-8192.jsonl.gz
count: 20
gzip SHA-256:
  16d8cd4665b1b5374ff5b974d95bcb1abbed1f1d7447f63c87103f53dccefc60
uncompressed JSONL SHA-256:
  18d3daf60341627764caea8b5e15ef257cac701056c5be63410d5b3df3860633
logical prompt SHA-256:
  185482ba78f0c9d45ba867f8c7714eb95db7b28a181bcfc6929dee2f8ee7a188
```

## Per-round results

Throughput is total tokens/second; latency values are milliseconds:

```text
round  SGLang throughput  TTFT    TPOT   E2E
1      1136.11            960.14  14.92  16210.65
2      1138.77            959.69  14.90  16192.43
3      1138.02            960.06  14.92  16185.33
4      1134.86            962.09  14.89  16219.55
5      1138.55            960.55  14.89  16189.84

round  ATOM throughput    TTFT    TPOT   E2E
1       767.76            871.41  22.62  24004.37
2       768.45            871.81  22.61  23984.27
3       768.06            867.93  22.62  23996.34
4       768.04            868.51  22.61  23997.45
5       767.74            870.18  22.63  24010.66
```

Round-to-round spread is small for both engines, so the median gap is not a
first-run/JIT outlier.

## C2 versus C64 interpretation

Common-client medians:

```text
                C2 total tok/s   C64 total tok/s
SGLang          1138.02          9800.69
ATOM             768.04          9846.50
ATOM delta       -32.51%           +0.47%
```

At C2, decode is the critical path and SGLang's `14.90 ms` TPOT is 34.1%
lower than ATOM's `22.62 ms`. ATOM's 9.4% TTFT advantage cannot compensate
over a 1024-token output.

At C64, SGLang delays first tokens but reaches `31.74 ms` TPOT versus ATOM's
`44.70 ms`; scheduler overlap and batching let both finish the full wave in
about 480 seconds. Thus throughput converges despite different per-request
latency shapes.

The C2 gap should be attributed with matched traces before proposing a port.
Likely categories to measure, not assume:

```text
KDA fused decode and f_b boundary
MLA Q/cache and tuned Triton decode
K3 B2/front projections
MoE route/sort/quant and stage kernels
TP collective implementation
graph replay launch count and elementwise boundaries
precision/online-quant differences
```

## Implications

1. Keep common-client methodology for all cross-engine results.
2. SGLang already has a major C2 advantage; multi-stream is not the first
   priority for closing a framework gap.
3. ATOM-style multi-stream improves ATOM C2 only modestly and regresses C64;
   it cannot explain the 32.5% single-stream C2 gap.
4. Capture compact C2 traces from both engines with the same manifest, then
   compare per-token decode kernels and collectives.
5. Preserve SGLang's current `_add3`, fused KDA, and tuned decode paths during
   any multi-stream experiment.

## Artifacts

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/
  prompt-manifest-c2-8192.jsonl.gz
  sglang-same-aiter/round{1..5}/{summary.json,requests.jsonl}
  atom-same-aiter/round{1..5}/{summary.json,requests.jsonl}
  sglang-same-aiter/server.log
  atom-same-aiter/server.log
  sglang-run.log
  atom-run.log
```

No server or client process remains. GPUs and experiment caches were released.
