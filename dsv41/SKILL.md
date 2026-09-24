---
name: dsv41
description: >-
  DeepSeek-V4.1-Flash on AMD gfx950 (MI350X/MI355X) with SGLang, tracking PR sgl-project/sglang#39857
  ([AMD] Support DeepSeek-V4.1 on gfx950 with DSpark and fused kernels). Use when working in
  /sgl-workspace/sglang-dsv41 (branch dsv41-amd-main), launching or benchmarking DSV4.1-Flash,
  DSpark speculative decoding, HIP low-ratio attention / KV store / RoPE fusion, BF16 WO-A,
  mHC split-H, TP4 all-reduce+mHC fusion, sharded Engram, or AITER MoE on this branch.
---

# DSV4.1 on gfx950 (PR #39857)

## CONTINUE HERE

**Status:** runs & GSM8K OK (~0.90-0.91). With PR's measurement method (scripts/run_pr_style_c1.sh) bs1 is
DSpark off 147.7 (PR 155.7, -5%), on 621 (PR 642, -3%). Ruled out: aiter #5561/#5802, BOUND, sgl-kernel sort_output, env.
State: aiter has #5561+#5802 applied; sgl-kernel = rebuilt from branch (old backed up); servers up on :30000/:30001.
**Next:** RUNBOOK.md done. Next task: compare vLLM on MI355X for this model and trace vLLM optimizations
reusable in SGLang (vLLM not installed in this container).
**Files:** source `/sgl-workspace/sglang-dsv41` (fork `kevin-mii/sglang`, branch `dsv41-amd-main`,
HEAD `e2e824dc58`).
**Repro:** TBD (the PR gives no explicit launch command; see "Launch config" below).
**Pass criteria:** GSM8K 5-shot full 1319 TP4/EP4 ≈ 90.45% (DSpark off) / 90.22% (DSpark on).

## Folder rules (MUST follow -- keep this dir tidy)

```
dsv41/
  SKILL.md      index: CONTINUE HERE, recipe, gotchas, rules.
  VLLM_COMPARE.md  vLLM-vs-SGLang perf comparison + reusable-optimization inventory (own CONTINUE HERE).
  RUNBOOK.md    how to rebuild the validated environment from a fresh container (keep in sync with scripts/).
  NOTES.md      longer findings / investigation log, newest first.
  scripts/      ALL runnable scripts (*.sh, *.py). Nothing executable anywhere else.
  patches/      patches for OTHER repos (aiter, ...), named <repo>_<upstreamPR>_<what>.patch.
                Apply with `git -C /sgl-workspace/<repo> apply patches/<f>`; record applied state in CONTINUE HERE.
  results/      small curated result tables (*.md), one file per benchmark type
                (gsm8k.md, perf.md, ...). Append a row per run; never paste raw logs here.
```

- Raw logs, traces, dumps, JSONL outputs -> `/shared_nfs/kk/dsv41/`, never in this dir. Result rows link to them.
- New script: put in `scripts/`, header comment = purpose + env knobs + where its output goes.
  Prefer adding a knob to an existing script over a near-duplicate copy.
- Scripts resolve paths relative to themselves (`$(dirname "$0")/..`) -- no hardcoded cwd.
- Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line (it kills your shell).
  Stop servers by PID: `ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'`.
- Scripts must use `grep -E`, NOT `rg`: `rg` exists only in the agent shell, not in nohup/non-interactive bash.
- Do not add other subfolders without updating this section first.
- Update CONTINUE HERE whenever status changes.

## Scripts

- `scripts/setup_env.sh` -- idempotent env setup (branch, aiter patches, sgl-kernel rebuild). See RUNBOOK.md.
- `scripts/pipeline_eval.sh` -- wait for server, then gsm8k + throughput; one line per step to
  `/shared_nfs/kk/dsv41/summary.txt`. Needs `TAG PORT SERVER_LOG SERVER_PID`.
- `scripts/run_pr_style_c1.sh` -- PR's exact bs1 method (warm-up + 6-run median, flush, seed 42, 1000/TPOT);
  use this for PR comparisons. Appends to `results/perf_prstyle.md`.
- `scripts/run_throughput.sh` -- PR-matching sweep (ISL4096/OSL1024, conc 1/8/32, random-ids + random/ShareGPT);
  appends to `results/perf.md`.

- `scripts/launch_server.sh` -- start server (see recipe below). Run in background:
  `nohup bash scripts/launch_server.sh > /shared_nfs/kk/dsv41/server.log 2>&1 &`
- `scripts/run_gsm8k.sh` -- GSM8K few-shot vs running server; `TAG=... NQ=1319 SHOTS=5`.
  Appends to `results/gsm8k.md`.

## PR summary (as of 2026-09-24)

- Author kevin-mii, open/draft, 116 commits, head `kevin-mii:dsv41-amd-main` → `sgl-project:main`.
- Continues #39186. Still carries unmerged prerequisites #38798 and #39666 (these inflate the diff).
  Will be rebased to an AMD-only diff once they land.
- aiter fixes moved upstream: ROCm/aiter#5561, #5562, #5802 (Docker aiter patches + FMoE tuning CSV removed).
- Opt-in (mixed/slower results): Engram prefetch, shared-expert multistream, compact KV.

## Launch recipe (verified source: cookbook, not the PR body)

Official recipe: `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx` (MI350X cells, `verified: true`),
prose in `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`. Preview image `lmsysorg/sglang:dev-dsv41-mi35x`.
Wrapped in `scripts/launch_server.sh`: TP4+EP4, env `SGLANG_USE_AITER=1` (load-bearing: else fp4 experts
hit Triton runner and assert), `SGLANG_MOE_PADDING=1`, `AITER_FLYDSL_FORCE_REDUCE=1` (run-to-run determinism),
`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`; flags `--disable-radix-cache --cuda-graph-backend-prefill breakable
--cuda-graph-max-bs-prefill 4096`. `DSPARK=1` adds `--mem-fraction-static 0.8 --speculative-algorithm DSPARK
--speculative-dspark-block-size 5 --cuda-graph-max-bs-decode 64`. No hierarchical cache on ROCm (conflicts
with --disable-radix-cache). PD + speculative cannot be combined.

Env notes: installed editable sglang is `/sgl-workspace/sglang` (main), NOT the branch -> script uses
`PYTHONPATH=/sgl-workspace/sglang-dsv41/python`. Installed `sglang-kernel 0.4.7` lacks the branch's AOT
`deepseek_v4_topk_transform_512(sort_output=)`; `low_ratio_backend_hip.py` detects it and falls back with a
warning (perf only). Rebuild sgl-kernel from the branch for full perf.

## Reference numbers (4×MI350X, ISL 4096 / OSL 1024, output tok/s, DSpark off → on)

Random bs1 155.7 → 641.9 (4.12×); real text bs1 155.9 → 335.1 (2.15×), bs8 951.7 → 1446.5 (1.52×),
bs32 2182.2 → 2241.9 (1.03×). Real acceptance, not simulated (older 548–556 figures were simulated).

## Known issues from PR

- Intermittent RCCL graph-capture abort: not proven fixed.
- AITER tuning tolerances still need independent validation.
- Not run: CUDA/ROCm 10 validation, fresh Docker build, full CP/EAGLE configs.

## Gotchas

- **Cookbook V4.1 MI350X recipe is missing `AITER_BF16_FP8_MOE_BOUND=0`** (PR body, commit 86ab3ad1bc and the
  DSV4 cookbook all set it). Without it, aiter (default bound 256, `aiter/fused_moe.py` ~L1044) routes M<256
  MoE to bf16 activations -> CK stage1 `RuntimeError: Unsupported kernel config for moe heuristic dispatch`
  during prefill graph capture (num_tokens=240). Fix for that route is ROCm/aiter#5802 (unmerged).
  `scripts/launch_server.sh` now exports it plus `TRITON_HIP_USE_ASYNC_COPY=0`.
- aiter PRs the branch relies on, all **unmerged on aiter main as of 2026-09-24** (local aiter `acf8fdf93`):
  #5561 FlyDSL stage-1 LDS-DMA drain (commit says GSM8K 0.885 unpatched vs 0.905 patched on 200 q),
  #5562 tuned FMoE CSV for dsv41 ep4 a8w4 (commit says no e2e gain), #5802 bf16 SiLU route.
  The dropped patches are saved in `patches/` (both `git apply --check` clean on `acf8fdf93`); the Dockerfile's
  other two aiter cherry-picks (#5283, #5279) are already in local aiter. Local aiter state: see CONTINUE HERE. Pre-existing local aiter edits (unrelated files) backed up at
  `/shared_nfs/kk/dsv41/aiter_preexisting_local.diff`.

- The fork's `origin/main` is stale -- never diff against it (~5.7k files). Remote `upstream`
  (sgl-project/sglang) is added; real PR diff:
  `git diff --stat $(git merge-base HEAD upstream/main) HEAD` → 106 files, +16358/-666
  (as of 2026-09-24: merge-base 81b81664a6, 116 ahead / 3 behind upstream/main).
  Hot dirs: `python/sglang/kernels/ops/attention/dsv4/`, `kernels/jit/csrc/deepseek_v4/`,
  `srt/models/deepseek_common/amd/`, `test/registered/kernels/ops/attention/dsv4/`.
- `gh` is not installed in this container; use WebFetch for PR pages.
- Related skills: `dsv4` (DSV4-Pro benchmarking), `aiter-custom-allreduce-nan-crash`,
  `sglang-prefill-coalescer`, `perf-bottleneck-attribution`.
