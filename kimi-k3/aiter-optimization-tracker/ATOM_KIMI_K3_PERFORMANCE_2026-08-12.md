# ATOM Kimi-K3 TP8 performance — 2026-08-12

## Result

ROCm ATOM `main` was rebuilt after a container restart and benchmarked on the
same 8x MI355X machine as the SGLang reproduction.

```text
Machine: crsuse2-m2m-002.crusoe.amd.com
ATOM:    f782218a526611b62745f2c0bb5e94656e04da1e
AITER:   284a1eb401bb15f6368a68b34eb0cd693ee1fcd3

Torch:   2.9.1+rocm7.2.0.lw.git7e1940d4
Triton:  3.6.0+git42270451
FlyDSL:  0.3.0
GPU:     8x AMD Instinct MI355X
```

Torch, Triton and all existing runtime dependencies were preserved. ATOM and
AITER were installed editable with dependency resolution disabled.

## Recipe configuration

The launch followed `recipes/Kimi-K3.md`, mapping only the local model path:

```text
model:                    /shared_nfs/models/Kimi-K3
tensor parallel:          8
KV cache:                 FP8
max model length:         16384
max sequences:            64
max batched tokens:       16384
GPU memory utilization:   0.93
cache block size:         128
prefix caching:           disabled
online quantization:      PTPC FP8 with recipe exclusions
```

The model loaded in about 229 seconds. Cold graph compilation and startup then
completed successfully. Torch 2.9 emitted non-fatal warnings because its Triton
mutation analysis attempted to import `specialize_impl`; it fell back, saved
the computation graphs and served all benchmark requests.

## Matched endpoint methodology

```text
dataset:                  random
input / output:           fixed 8192 / 1024
random range ratio:       1.0
request rate:             infinite
measured requests:        8 x concurrency
warmup requests:          64 at each concurrency
seed:                     42
ignore EOS:               enabled
backend:                  ATOM OpenAI completions
```

All C2-C32 measured requests succeeded (496/496). A later matched C64 run also
completed 512/512 requests.

## Performance

The comparison uses the same-machine SGLang production reproduction, not the
older recorded baseline.

```text
Concurrency  ATOM total tok/s  SGLang total tok/s  ATOM delta  ATOM median TPOT
C2                    799.99               980.24      -18.39%             21.56 ms
C4                   1483.11              1764.84      -15.96%             22.42 ms
C8                   2521.14              2921.82      -13.71%             25.59 ms
C16                  4290.71              4492.89       -4.50%             28.54 ms
C32                  6117.82              6295.63       -2.82%             38.27 ms
C64                  8380.39              8014.22       +4.57%             52.66 ms
```

ATOM's gap is largest at low concurrency, narrows steadily toward C32, and
reverses at C64. ATOM's C64 median TPOT was 52.66 ms versus SGLang's 55.58 ms
(`-5.25%`).

These numbers compare framework recipes rather than identical numerical
precision: ATOM uses the recipe's FP8 KV cache and PTPC-FP8 online
quantization, while the reproduced SGLang production command used BF16 model
dtype and its selected AITER/SGLang Kimi paths.

## Artifacts

```text
/workspace/kimi-k3-runs/atom-rebuild/server.log
/workspace/kimi-k3-runs/atom-rebuild/smoke.json
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c2.log
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c4.log
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c8.log
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c16.log
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c32.log
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c2.json
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c4.json
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c8.json
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c16.json
/workspace/kimi-k3-runs/atom-rebuild/endpoint/c32.json
/workspace/kimi-k3-runs/c64/atom-c64.log
/workspace/kimi-k3-runs/c64/atom-c64.json
/workspace/kimi-k3-runs/c64/sglang-c64.log
/workspace/kimi-k3-runs/c64/sglang-c64.jsonl
```

No ATOM server or benchmark process was left running.
