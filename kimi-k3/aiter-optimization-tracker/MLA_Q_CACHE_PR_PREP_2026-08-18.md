# MLA Q/cache SGLang PR preparation — 2026-08-18

Local branch:

```text
/sgl-workspace/sglang-pr-mla-q-cache
perf/k3-mla-q-cache-fusion
base origin/main 0077f84d3
commit 9f9e2e2be
```

Draft PR created:
[sgl-project/sglang#35308](https://github.com/sgl-project/sglang/pull/35308).

## Scope

- Add a fail-closed gfx950 adapter for
  `aiter.fused_qk_rope_concat_and_cache_mla`.
- Fuse identity-RoPE Q materialization, Q concat and KV-cache write.
- Support same-dtype BF16/BF16 and FP8/FP8 plus BF16-Q/FP8-KV mixed output for
  Triton decode.
- Wire decode/idle only; DCP and unsupported contracts retain the existing
  split/cat/cache path.

Enable:

```bash
export SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1
```

## AITER dependency

The SGLang branch intentionally does not update the AITER pin. It depends on
AITER #4342 / commit `770790cd` or newer operator semantics, specifically the
`compute_all_q_rope` behavior needed for correct batch64 execution.

Main's current AITER pin `d9e5ef7` is insufficient:

```text
new argument rejected by old schema
14-argument compatibility call passes token1
token64 output incorrect
```

This dependency must be documented in the future PR or resolved by a separate
AITER pin update.

## Validation

Using the current compatible AITER checkout:

```text
focused operator tests: 9 passed
GSM8K 1319: 0.955, invalid 0.001
```

Server configuration used Triton prefill and Triton decode with FP8 KV. AITER
prefill was not used because 12-head AITER backend enablement belongs to the
separate independent integrations PR.

Artifacts:

```text
/workspace/kimi-k3-runs/mla-q-cache-pr-gsm8k-2026-08-18/
```

## Suggested PR title

```text
[AMD] Fuse Kimi-K3 MLA Q and cache preparation
```
