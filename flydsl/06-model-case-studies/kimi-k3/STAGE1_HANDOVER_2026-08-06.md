# Kimi-K3 Stage 1 optimization handover

Date: 2026-08-06

## Start here in the new chat

Stage 1 is complete. The retained deliverable is the MoE copy cleanup at
`e9d8cb947285fd5c2db85aedb28ac80816786bb9`. KDA and MLA fusion experiments
were tested individually and together, but were removed because the matched
combined gain was only 0.23%.

Before doing any new optimization:

1. Confirm `/sgl-workspace/sglang` is the intended checkout.
2. On the original checkout, confirm `git rev-parse HEAD` is
   `e9d8cb947285fd5c2db85aedb28ac80816786bb9`. On a newer image checkout,
   apply the exported Stage 1 patch below and record the new resulting commit
   hash; it will normally differ because the parent commit changed.
3. Confirm Python imports SGLang from this checkout, not only from the new
   image's preinstalled package.
4. Record the new Docker image name/digest, SGLang commit, AITER commit, ROCm
   version, and GPU architecture.
5. Run the retained smoke/correctness tests before comparing performance.

No server, benchmark, or GPU process was active at handover time.

## Critical Docker transition warning

The current persistent `run_docker.sh` mounts the model, KVV checkout, and
benchmark scripts, but it does **not** mount `/sgl-workspace/sglang`.

The three local commits below have not been pushed to a remote. The new image
is expected to contain the prerequisite DCP code. The retained Stage 1 commit
has therefore been exported as:

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/e9d8cb9472-remove-repeated-moe-copies.patch
SHA-256: c8aaf2c536b2901e3a8bf5ca52b79a8b5d7ef21ea181d6fef544bcdb489f10ae
```

Apply it from the new image's SGLang checkout with:

```bash
git am /dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/e9d8cb9472-remove-repeated-moe-copies.patch
```

If the new image already contains equivalent changes, inspect the patch and
skip it rather than applying a duplicate. If `git am` reports conflicts, abort
with `git am --abort`, port the four-file semantic change onto the new tree,
and rerun the focused tests.

For Stage 1 code testing, preserve/mount at least:

```text
/sgl-workspace/sglang
/dockerx/data/models/Kimi-K3
/dockerx/data/models/Kimi-K3-DSpark
/dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3
```

After selecting the new checkout, make sure it wins Python package resolution.
For example, use the image's normal editable-install workflow or verify:

```bash
cd /sgl-workspace/sglang
python - <<'PY'
import sglang
print(sglang.__file__)
PY
git rev-parse HEAD
```

Do not benchmark until the import path points at the intended checkout.

## Repository state

Repository:

```text
/sgl-workspace/sglang
branch: kimi-k3
HEAD: e9d8cb947285fd5c2db85aedb28ac80816786bb9
remote relation: ahead of origin/kimi-k3 by 3 commits
```

Local commits:

```text
e9d8cb9472 perf(kimi-k3): remove repeated MoE copies
064a868801 fix(dcp): sync latest Kimi-K3 PR corrections
1c3ef401df feat: support Kimi-K3 AITER DCP on ROCm
```

There are pre-existing uncommitted AOT/HIP packaging changes that are unrelated
to Stage 1:

- modified/deleted `python/pyproject*.toml`
- modified/deleted `python/sglang/kernels/aot/pyproject*.toml`
- untracked HIP sources under `python/sglang/kernels/aot/`

Do not discard, commit, or mix those files into Kimi-K3 optimization commits
without first identifying their owner and purpose.

## Retained Stage 1 change

Commit `e9d8cb9472`:

- caches the AITER top-k correction-bias cast;
- makes routed MoE honor the supplied zero-copy output buffer;
- adds focused tests for correction-bias caching and MXFP4 output aliasing.

Files in the retained commit:

```text
python/sglang/srt/layers/moe/topk.py
python/sglang/srt/layers/quantization/mxfp4.py
test/registered/unit/layers/moe/test_topk_correction_bias_cache.py
test/registered/unit/layers/quantization/test_mxfp4_situ_output.py
```

Measured effect:

- PyTorch-native launches: 622 -> 530 (-92).
- PyTorch-native GPU time: 2.86 -> 2.45 ms/step (-0.40 ms).
- Initial single-wave output throughput screen: +0.64%.
- Median TPOT: -0.54%.
- Median ITL: -0.47%.

Focused retained tests were rerun at handover: 7 passed.

## Experiments not retained

### ROCm KDA packed decode + gated RMSNorm

- Kernel parity passed for batch 1/8/32 and both gate variants.
- Initial short screen reported +0.82% output throughput.
- The short screen used a different launch/sample length, so this percentage
  was provisional.
- Code and experiment tests were removed.

### Triton MLA preparation fusion

- Reused AITER fused RoPE/QK concat/cache write for the Triton backend.
- BF16/FP8 cache-bit, batch 1/32, fallback-gate, and CUDA Graph address tests
  passed.
- Matched 64-warmup/256-request result: +0.14% output throughput.
- Code and experiment tests were removed.

### Combined KDA + MLA on top of retained MoE

Correctness:

- 18 tests passed.
- 10 subtests passed.
- GSM8K 200: 0.980 (196/200); prior experimental run was 0.990.

Authoritative matched serving comparison:

```text
Workload: TP8, DCP off, DSPARK off, Triton attention
Lengths: fixed 8192 input / 1024 output
Concurrency: 32
Warmups/measured: 64 / 256
Radix Cache: disabled
```

```text
MoE-only retained baseline:
  output throughput 523.4068 tok/s
  median TPOT       47.0694 ms
  median ITL        37.9876 ms

MoE + KDA + MLA:
  output throughput 524.6119 tok/s
  median TPOT       47.0419 ms
  median ITL        37.9384 ms

Delta:
  output throughput +0.2302%
  median TPOT       -0.0583%
  median ITL        -0.1293%
```

Conclusion: the standalone percentages were not additive. KDA and MLA stayed
within serving noise under a controlled full comparison and were removed.

## Earlier DCP/long-context status

The AMD Kimi-K3 DCP port is contained in the two commits before Stage 1.
Important validated results include:

- TP8 non-DCP GSM8K: 0.985.
- DCP8 GSM8K: 0.985.
- DCP8 page size 32 is better than 64 for the tested workload.
- Tuned DCP8 recovered the original long-context throughput gap.
- Custom 132k + PR 33599 + FP8 KV peaked at 172.68 output tok/s at
  concurrency 32.
- BF16/FP8 retrieval responses matched 12/12 through approximately 114.6k
  prompt tokens.
- DSPARK long-context accept length was workload-sensitive: 2.84 on the 68k
  shared-prefix workload versus 5.678 on 8k/1k.

Use the persistent reports below for exact conditions and metrics.

## Persistent reports

All of these survive Docker replacement:

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/AIPERF_SUMMARY.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/TP8_ROCM_TRACE_ANALYSIS.md
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/TP8_ROCM_structured_analysis.txt
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/TP8_ROCM_trace_summary.json
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/context-accuracy-comparison.json
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/kimi-k3-dcp-aiperf.canvas.tsx
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/kimi-k3-tp8-rocm-trace.canvas.tsx
```

The raw AIPerf artifacts and ROCm traces remain under `/sgl-workspace` and may
not survive a container replacement unless that workspace is preserved.

## Persistent scripts

Directory:

```text
/dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3
```

Most relevant scripts:

```text
launch_server.sh
wait_server.sh
smoke_test.sh
stop_server.sh
run_gsm8k.sh
run_aiperf_68k_sweep.sh
run_standard_8k1k.sh
run_context_accuracy.py
run_docker.sh
```

The script README documents the common sequence and environment overrides.

## New-image validation sequence

1. Record image and software versions.
2. Verify the model and local SGLang checkout are mounted.
3. Verify `sglang.__file__`, record `git rev-parse HEAD`, and apply the exported
   Stage 1 patch if the new tree does not already contain its changes.
4. Confirm all eight GPUs are idle and visible.
5. Run retained focused tests:

   ```bash
   cd /sgl-workspace/sglang
   python -m pytest -q \
     test/registered/unit/layers/moe/test_topk_correction_bias_cache.py \
     test/registered/unit/layers/quantization/test_mxfp4_situ_output.py
   ```

6. Launch TP8 non-DCP Triton:

   ```bash
   cd /dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3
   DCP_SIZE=1 ATTENTION_BACKEND=triton \
   MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 \
   RADIX_CACHE=0 ENABLE_INT8_MAMBA_CHECKPOINT=1 \
   ENABLE_CACHE_REPORT=1 ./launch_server.sh
   ```

7. Wait for `The server is fired up and ready to roll!`.
8. Run smoke and GSM8K.
9. Run `run_standard_8k1k.sh` with the same 64-warmup/256-request methodology.
10. Compare against the retained 523.4068 output tok/s baseline before
    attributing any difference to the new image.
11. Stop the server before changing configurations.

## Suggested Stage 2 direction

Do not restart the reverted KDA/MLA preparation experiments unless the kernel or
workload changes materially. The production trace shows larger opportunities:

1. MoE route plus stage 1/2: approximately 34.9% of the decode step.
2. Dense GEMMs: approximately 21.1%.
3. Triton grouped MLA decode: approximately 15.8%.
4. Custom all-reduce: approximately 5.8%, with rank-dependent spin included.
5. Cache the fallback BF16-scaled `w_kc` only if the path is active in the new
   image and a production trace confirms repeated materialization.

For every Stage 2 change:

- use a correctness oracle first;
- run a matched baseline/optimized comparison from the same image;
- use fixed 8192/1024, concurrency 32, 64 warmups, and 256 measured requests;
- avoid adding percentages from unmatched short screens;
- retain only changes with a repeatable production-level gain.
