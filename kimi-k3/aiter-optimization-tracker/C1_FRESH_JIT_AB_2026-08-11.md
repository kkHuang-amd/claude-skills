# Kimi-K3 fresh-JIT C1 A/B — 2026-08-11

## Method

```text
8× MI355X / TP8
Torch 2.9.1 / Triton 3.6
8192 input / 1024 output
concurrency 1
8 warmups + 8 measured requests
seed 42
radix cache disabled
```

All AITER runs shared a new JIT directory. Fresh-cache dispatch traces had
already confirmed the B1 candidate kernels execute.

The baseline was measured both before and after the candidates:

```text
baseline 1: output 60.63 tok/s, TPOT 16.00 ms
baseline 2: output 60.71 tok/s, TPOT 15.98 ms
mean:       output 60.67 tok/s, TPOT 15.99 ms
```

## Results

```text
profile     output tok/s  delta    median TPOT  delta
baseline       60.67        —        15.99 ms     —
#4503          62.12      +2.39%      15.61 ms   -2.38%
#4504          66.49      +9.59%      14.55 ms   -9.01%
#4572          60.63      -0.07%      15.99 ms    0.00%
```

Median E2E:

```text
baseline mean  16876.78 ms
#4503          16483.66 ms  (-2.33%)
#4504          15399.43 ms  (-8.75%)
#4572          16880.52 ms  (+0.02%)
```

TTFT stayed near 517–518 ms, confirming the gains and loss are in decode.

## Capacity cost

Server-reported `max_total_num_tokens`:

```text
baseline  933883
#4503     844681  (-9.55%)
#4504     808424  (-13.44%)
#4572     933883  (flat)
```

#4503/#4504 retain extra FP8-packed weights even when runtime batch is not 1.
The memory/capacity cost therefore applies to the whole deployment.

## Accuracy and dispatch evidence

Previous focused gates remain applicable:

- #4503: 7 focused tests; GSM8K 50 = 1.000.
- #4504: 6 focused tests; GSM8K 50 = 1.000.
- #4572: 64 gate tests; fresh-cache C1 trace observed 23 candidate launches.

Fresh-JIT B1 traces observed:

```text
#4503 latent tail:        11776 launches
#4504 tri projection:     11776 launches
#4504 shared down:        11776 launches
```

## Decision

### General production

Keep the current selected profile:

```text
SGLANG_K3_AITER_MOE_PREROUTE_FP8=0
SGLANG_K3_AITER_LATENT_TAIL_FP8=0
```

The extra memory is paid at all concurrencies, while the kernels cover B1 only.

### Optional latency-C1 profile

Enable #4504 when the deployment is explicitly optimized for single-request
latency and can accept about 13.4% less token capacity:

```text
SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
SGLANG_K3_AITER_LATENT_TAIL_FP8=0
```

#4503 is also positive at C1, but its 2.4% throughput gain is small relative to
the 9.6% capacity loss. Keep it opt-in off unless a combined #4503+#4504 C1
profile is evaluated.

Reject the #4572 selective dispatch: its isolated kernel gain does not survive
the C1 endpoint gate.

Artifacts:

```text
stage2-runs/2026-08-11-c1-fresh-jit-ab/
```
