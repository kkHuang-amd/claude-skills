---
name: kimi-k3-deployment-benchmark
description: Deploy and accuracy-benchmark open-weight Kimi-K3 with SGLang on 8 GPUs. Use when launching Kimi-K3, switching inference Docker images, running KVV OCRBench/MMMU-Pro/Tool-call/BEAM 1M, diagnosing startup warmup or long-context failures, or handing the Kimi-K3 benchmark environment to another engineer.
---

> ## Entry points — read ONE of these, never this directory
>
> `kimi-k3/` holds 115 markdown files (~634 KB, ~158k tokens). Do not read it
> in bulk, and do not read `SUMMARY.md` in full (36 KB).
>
> | Your task | Start at |
> |---|---|
> | Resume optimization work (kernels, traces, candidates) | `CONTINUE_HERE.md` |
> | Any K3 work — rules that always apply | `DEV_RULES.md` |
> | Fresh container: restore worktrees / canvases | `start_prompt.md` §2 and §4 |
> | Deploy + accuracy benchmark | this document, below |
>
> For `SUMMARY.md`: `rg -n '^## ' SUMMARY.md` then `sed -n 'A,Bp' SUMMARY.md`.

# Kimi-K3 deployment and benchmark

Use the scripts in:

```text
/dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3
```

Read [CONTINUE_HERE.md](CONTINUE_HERE.md) before launching — it records the
current status, next action and worktree state. For the original deployment
handover (paths, validated parameters, prior scores, failure modes) see
[k3-handover-2026-08-12/HANDOVER.md](k3-handover-2026-08-12/HANDOVER.md);
note it is dated 2026-08-12 and superseded by `HANDOFF_2026-08-21.md`.

## Workflow

1. Verify the Docker image exposes 8 GPUs and contains the intended SGLang,
   AITER, FlyDSL, and ROCm versions.
2. Mount `/dockerx/data/models/Kimi-K3` and the KVV checkout into the container.
3. Run `launch_server.sh` in the foreground and save its output.
4. Do not send requests at `Application startup complete`. Wait for:

   ```text
   The server is fired up and ready to roll!
   ```

5. Run `smoke_test.sh`.
6. Run `run_kvv.sh ocrbench` first. Continue with `mmmu` and `toolcall` only
   after OCRBench succeeds.
7. Run BEAM separately with `run_beam.sh`. Keep Radix Cache enabled and use
   the Kimi-K3 tokenizer, not BEAM's bundled Kimi-K2.6 tokenizer.
8. Record the Docker image digest, SGLang commit/version, command, result log,
   and score. Do not mix outputs from different weights or images.

## Invariants

- Model weights: `/dockerx/data/models/Kimi-K3` (96 shards, about 1.6 TB).
- KVV checkout: `/sgl-workspace/kvv-bench/kvv-k3-0727-update`.
- API endpoint: `http://localhost:8000/v1`.
- K3 accuracy settings: effort `max`, temperature `1.0`, top-p `0.95`.
- OCRBench max output: `16384`.
- MMMU-Pro max output: `98304`.
- Tool-call max output: `32768`.
- BEAM max output: `32768`, concurrency `16`, model tokenizer required.
- SGLang serves the local model path as its model ID. Confirm with
  `/v1/models`; do not assume it.

## Stop conditions

- Stop if the server exits, requests return connection errors, or the active
  model ID differs from the intended weight path.
- Stop BEAM if any request reports context-length overflow. Fix tokenizer
  truncation and remove failed output rows before resuming.
- If startup warmup times out after 600 seconds, requests were probably sent
  before full warmup completed. Restart the server and wait for `ready to roll`.
- If AITER reports untuned GEMM shapes and falls back to torch, accuracy may
  still run, but record the warning because performance comparisons are invalid.

