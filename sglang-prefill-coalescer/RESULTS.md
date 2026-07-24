# Results — SGLang prefill coalescer vs ATOM

Env: 8xMI355X (gfx950), DeepSeek-V4-Pro (`/dockerx/data/deepseek-ai/DeepSeek-V4-Pro`),
TP8 + DP-attention (+ TBO), kv fp8, page 256, mem 0.90, max-num-batched-tokens
16384. Client `random`, `--request-rate inf`. All ON/OFF pairs are same-session,
same server config, only the coalescer toggled.

For the ATOM-vs-SGLang comparison the environment was upgraded (aiter→main,
flydsl→0.2.4) so ATOM's coalescer build could run in-place; see PROBLEMS.md.

## Throughput A/B — 1k/1k, concurrency 1024 (total token throughput, tok/s)

| engine | workload | OFF | ON (target_fill=0.7) | Δ | ON TTFT vs OFF |
| --- | --- | ---: | ---: | ---: | --- |
| SGLang port | uniform (range-ratio 1.0) | 25,120 | 25,148 | +0.1% | ~same |
| SGLang port | **fragmented (range-ratio 0.3)** | 15,747 | **20,571** | **+30.6%** | 4,959 vs 4,596 ms |
| ATOM native | uniform (range-ratio 1.0) | 29,524 | 29,256 | −0.9% | ~same |
| ATOM native | **fragmented (range-ratio 0.3)** | 15,236 | **20,588** | **+35.1%** | 4,767 vs 4,408 ms |

Notes
- SGLang port and ATOM native are different engines → absolute tok/s not
  directly comparable; what matters is the ON/OFF Δ per engine, and both agree:
  neutral on uniform, large gain on fragmented.
- OFF numbers between engines are close on the fragmented point (15.7k vs 15.2k),
  a good cross-check that the workload/setup match.
- The gain comes from the alignment gate + SUM fill target cutting wasted small
  prefill forwards / DP idle when ranks have uneven prefill token counts.

## Earlier non-TBO / uniform points (SGLang port, for reference)

| workload | OFF | ON | Δ |
| --- | ---: | ---: | ---: |
| non-TBO 1k/1k c256 | 12,353 | 11,688 | −5.4% |
| non-TBO 8k/1k c256 | 31,190 | 31,167 | −0.1% |
| non-TBO 1k/1k c1024 | 25,120 | 25,148 | +0.1% |
| TBO 8k/1k c256 | 33,115 | 33,278 | +0.5% |
| TBO 1k/1k c1024 | 25,079 | 25,428 | +1.4% |

All uniform (range-ratio 1.0) → all ~neutral. This is the "wrong workload"
trap: nothing to coalesce. Switching to range-ratio 0.3 is what surfaced the
+30.6%.

## gsm8k accuracy (full 1319, DeepSeek-V4-Pro)

| config | max_tokens | score |
| --- | ---: | ---: |
| baseline main (no delayer) | 8192 | 0.931 |
| coalescer ON | 8192 | 0.939 |
| coalescer ON (200-sample) | 2048 | 0.855 |
| baseline main (200-sample) | 2048 | 0.860 |

- ON ≈ OFF (0.939 vs 0.931, within noise) → no accuracy regression.
- The low 0.883/0.85x came from `max_tokens=2048` truncating DSV4's max-effort
  reasoning (the launch script sets `SGLANG_DSV4_REASONING_EFFORT=max`). Use
  `--max-tokens 8192` for a valid gsm8k number on this model.

## Component versions at validation time

- SGLang: `feat/prefill-coalescer-sum-fill-target` @ `aa9547b98`
- ATOM: repo `origin/main` `c5601741`; installed wheel `0.1.4.dev335+gc5601741f`
  (coalescer = PR #1611 `4b2b574b`)
- aiter: `origin/main` `874840aef` (upgraded from `d9b3e0d2e`)
- flydsl: `0.2.4`
- triton-custom: `3.6.0` (`4227045199`)
