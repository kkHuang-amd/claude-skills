# K3 SGLang rebase onto upstream main — 2026-08-25

## CONTINUE HERE

**Status:** Rebase done; GSM8K-verified and perf A/B'd against the pre-rebase
tree. Costs ~0.2-0.5% throughput (see 5b). **Pushed 2026-08-25.**
`origin/perf/k3_opts_0812` = `bf087c33f` (force-pushed over `455b744aa`):
the 18 rebased commits plus one review-fix commit carrying the revised §4a
and §4b resolutions. The pre-rebase tip is preserved on the remote as
`backup/k3_opts_0812-pre-rebase-20260825` = `455b744aa`.
Rebase base was `e567d3c80`, 18 commits on `sgl-project/sglang` main
`67853c58` (2026-08-25). PR #35499 (K3 dspark draft attn) is now included.

**Next:** All three §4 resolutions are reviewed (4c closed, 4b and 4a revised).
Two things still need a machine this host cannot provide: §4b's block-FP8
branch needs an FP8_PB_WO checkpoint, and §4a needs a CUDA host with
flashinfer installed (`is_flashinfer_available()` is False here, so that whole
block is dead code on ROCm). `test_mxfp4_situ_output.py` has been broken since
it was added and needs retargeting before it can guard anything.

**Files:** `python/sglang/srt/models/kimi_k3.py`,
`python/sglang/srt/layers/quantization/mxfp4.py`,
`python/sglang/srt/environ.py`, `python/sglang/__init__.py`

**Repro:**

```bash
PYTHONPATH=/sgl-workspace/sglang-k3-triton37/python:/sgl-workspace/aiter-k3-triton37 \
  python3 -c "import sglang, aiter, os; print(os.path.dirname(sglang.__file__)); print(os.path.dirname(aiter.__file__))"
```

**Pass criteria:** resolves to `sglang-k3-triton37/python/sglang` +
`aiter-k3-triton37/aiter`, no ImportError.

**Rollback:** `git -C /sgl-workspace/sglang-k3-triton37 reset --hard backup/k3_opts_0812-pre-rebase-20260825`
(the same ref exists on `origin`, so the remote can be restored with
`git push --force origin backup/k3_opts_0812-pre-rebase-20260825:perf/k3_opts_0812`).
A second worktree at `/sgl-workspace/sglang-k3-prerebase` is checked out on
that branch and was kept for further A/B runs.

---

## 1. Environment (verified 2026-08-25)

`/workspace` now exists — it is an NFS mount of the user's home volume
(`172.27.255.2:/volumes/.../wunhuang`). This supersedes the §0 warning in
`CONTINUE_HERE.md` that `/workspace` was absent.

Both K3 worktrees are present:

- sglang `/sgl-workspace/sglang-k3-triton37`, branch `perf/k3_opts_0812`,
  fork `HaiShaw/sglang`
- aiter `/sgl-workspace/aiter-k3-triton37`, branch `integration/k3-core-only`,
  fork `kkHuang-amd/aiter`, HEAD `b56d27be` (untouched by this work)

**These are not the pip-installed trees.** The editable installs in `/opt/venv`
point at `/sgl-workspace/sglang` and `/sgl-workspace/aiter`, so a bare
`import sglang` silently loads the wrong code. `/sgl-workspace/sglang` is a
different HEAD (`63d783bbe0`) with ~20 uncommitted changes — a separate working
copy, not a stale mirror. Always launch with the `PYTHONPATH` above and confirm
the resolved paths.

`aiter-k3-triton37` is a trimmed branch: 21 prebuilt `.so` under `aiter/jit/`
versus 118 in `/sgl-workspace/aiter`. With 575 new upstream sglang commits now
in play, a code path reaching a kernel outside those 21 will JIT on first call.
An unexpected compile pause at startup is most likely this, not a rebase defect.

## 2. What the rebase did

23 branch commits → **18**. Base moved from merge-base `65d62109` (2026-08-14)
to upstream main `67853c58`, i.e. 575 upstream commits absorbed.

Old branch tip `455b744aa` is preserved as
`backup/k3_opts_0812-pre-rebase-20260825`. `origin/perf/k3_opts_0812` is
untouched at `455b744aa` — **nothing was pushed**.

Final shape vs upstream main: 51 files changed, +9644/-75.

## 3. The 5 commits that did not survive

One merge commit (`1e01da2ab`) dropped by rebase, as normal. The other four were
**already upstreamed** — upstream's versions are supersets, so keeping the local
copies would have regressed them:

- `89df50593` opt-in Radix-4 router → upstream **#34490** (`edd675cec`).
  Upstream's kernel is 527 lines vs 401, its test 307 vs 119, and its wiring uses
  the registered `envs.SGLANG_K3_RADIX4_TOPK` plus a
  `bias = correction_bias.to(dtype=gating_output.dtype)` cast the local version
  lacked.
- `bcf1fbc2b` tune MLA decode stage-1 geometry → upstream **#34580** (`d01812d89`).
- `e36fae0e3` pin MLA decode split budget (tests) → same upstream PR, which
  shipped implementation and tests together.
- `3467e68e9` concat_and_cast_mha_k_pad_kernel → upstream **#34837**
  (`4d0c5a89a`); rebase detected it as empty and dropped it automatically.

**Worth knowing about `bcf1fbc2b`:** the local version set `block_n=64` for the
small-batch bucket. Upstream deliberately did not, and says why in a comment —
64 was measured 3-5% *slower* at batch 1-3, noise at 4-5 — so it keeps a single
`_MLA_BLOCK_N = 32`. Taking upstream means accepting that measurement. If you
have contrary numbers on gfx950, that is an upstream conversation, not a local
revert.

## 4. Conflict resolutions that need review

Everything else was mechanical. These three involved judgment:

### 4a. `mxfp4.py` — non-deferred finalize return

Merge-base returned `StandardCombineInput(hidden_states=result)`. Upstream then
added `result = result[0]` to unwrap the FFI return; the local commit instead
returned `symm_output`, the explicit output buffer, on the grounds that some
SiTU runner versions hand back a distinct wrapper even though `symm_output`
holds the published result.

Two fixes for one problem. The original resolution kept the local one outright.

**Reviewed and revised 2026-08-25 — the original resolution was too broad.**

- The block is **not** HIP-gated: `use_flashinfer` comes from
  `get_moe_runner_backend().is_flashinfer_mxfp4()` (`:390`) and `_fi_kernel`
  covers `trtllm_sm100` / `cutlass_sm120` / `cutlass_sm90`. It is reachable on
  CUDA. Only the two SiTU returns had been changed; the generic public path
  (`:1718`) still carries upstream's `[0]` unwrap.
- Returning `symm_output` was applied even when `get_moe_output_spec()` returned
  None and the code fell back to its own `torch.empty` — i.e. when no zero-copy
  publication happened at all — silently extending an AITER-SiTU-runner
  assumption to every caller.
- **The regression test does not back this up.** `test_mxfp4_situ_output.py`
  patches `sglang.kernels.ops.moe.trtllm_gen_moe.*`, which exists in neither the
  pre-rebase nor the post-rebase tree (both import
  `trtllm_fp4_block_scale_moe` from the external `flashinfer` package). It has
  failed at mock-target resolution since `8f38dc988` added it. This was not
  caused by the rebase, and the original decision had no executable backing.

Revised resolution: return the destination only when zero-copy actually
published a buffer, otherwise keep upstream's unwrap.
`zero_copy_published = symm_output is not None` is recorded before the fallback
allocation; both SiTU returns branch on it, and the discarded return of the
bypassed-topk call is now captured as `situ_result`. New helper
`_unwrap_trtllm_moe_output()` holds upstream's 1-tuple unwrap.

**Unverified and unverifiable on this host:** `is_flashinfer_available()` is
False here (no flashinfer installed), so the whole `use_flashinfer` block is
dead code on ROCm — the GSM8K runs never touched it. Needs a CUDA host with
flashinfer. The stale test needs retargeting before it can guard any of this.

### 4b. `kimi_k3.py` — ROCm KDA in-proj fusion × ModelOpt block-FP8

**A guard was added here that neither side had.** `_merge_kda_inproj_weights_hip()`
early-returns and never sets `_bfa_f_b_w`, but upstream #35077's block-FP8 path
requires it — chaining them as written leaves it `None` and crashes in the
following `gemm()`. `_may_fuse_kda_inproj()`'s `type(weight.data) is torch.Tensor`
check does not exclude a block-FP8 checkpoint, so the collision is reachable.

Resolution — the two features are now mutually exclusive:

```python
if (
    _is_hip
    and not self._bfa_uses_block_fp8
    and self._merge_kda_inproj_weights_hip()
):
    self._bfa_f_b_w = self.f_b_proj.weight
    return
```

Block-FP8 checkpoints take upstream's dequantizing path unchanged; everything
else on HIP takes the ROCm fusion.

**Reviewed and hardened 2026-08-25.** The mutual exclusion is correct, but the
justification above was wrong and the guard was incomplete:

- The stated crash ("fused path never sets `_bfa_f_b_w` → `None` → crash in the
  following `gemm()`") is not the real mechanism. The resolution already assigns
  `_bfa_f_b_w = self.f_b_proj.weight` right after the fused call, and the fused
  HIP branch reads `self.f_b_proj.weight` directly anyway.
- The real incompatibility is the quant method. The fused path hands
  `SimpleNamespace(weight=merged)` — which carries only `.weight` — to
  `fused_qkvg_proj.quant_method.apply()`; a block-FP8 method raises
  `AttributeError` on the missing `weight_scale_inv`. And `_merge_weights_as_views`
  applies no scale at all, so even past that the GEMMs would consume raw FP8 bytes.
- `_bfa_uses_block_fp8` probes only `b_proj`'s algo string, while the fusion spans
  `fused_qkvg_proj` / `f_a_proj` / `b_proj`. A checkpoint whose three in-proj
  weights are uniformly FP8 under any other algo name slipped past both the flag
  and `_may_fuse_kda_inproj()`'s `type(weight.data) is torch.Tensor` check (an FP8
  weight's `.data` *is* a plain Tensor). Mixed cases were only being caught by the
  uniform-dtype check as a side effect.

Hardening applied: `_may_fuse_kda_inproj()` now rejects quantized modules
structurally — any module carrying `weight_scale_inv` / `weight_scale`, and any
weight whose dtype is not bf16/fp16. `_bfa_uses_block_fp8` remains as the second
line of defence. Not yet numerically tested.

### 4c. Two env vars upstream deleted

`SGLANG_K3_SHARED_EXPERTS_ATTN_TP` and `SGLANG_K3_DENSE_MLP_ATTN_TP` came from
#33465 and were migrated by upstream `c439e7787` (#34715) into the parallel
config: `get_parallel().enable_shared_experts_attn_tp` /
`.enable_dense_mlp_attn_tp`. The local branch read them at module scope; after
rebase every consumer is upstream's, so those two lines were dead **and** would
raise `AttributeError` at import. Removed.

**Reviewed 2026-08-25 — closed, no action needed.** Both names have 0 references
left anywhere in the tree, and all three consumers in `kimi_k3.py` read the
upstream replacement through `get_parallel()`: `:331` dense-MLP attn-tp, `:588`
and `:594` shared-experts attn-tp. The functionality was handed over, not lost.
The successful server boot confirms the import-time `AttributeError` is gone.

Of the branch's six K3 env vars, only `SGLANG_K3_AITER_MLA_Q_CACHE_FUSION` was
genuinely new — upstream already defines the other five. The final commit's five
AITER-tuned-MoE-front / latent-MXFP4 vars are all new and were kept.

## 5. Verification actually performed

Import-level only. Every changed `.py` byte-compiles; with the real `PYTHONPATH`,
`sglang`, `aiter`, `kimi_k3`, `verify_mla`, `triton_backend` and
`is_dspark_draft` all import, and the new env vars read back. PR #35499's markers
are present (`is_dspark_draft`, `KV_GROUP_NUM`, `HAS_KV_HEADS`, `IS_CAUSAL`,
the `64: (4, 256, 4)` GQA block config) and the `k_extend.shape[1] != 1` guard it
removes is gone.

**Updated 2026-08-25 — GPU verification done.** Server boots on 8x MI355X with
the full K3 env set (`/shared_nfs/kk/k3_rebase_20260825/launch.sh`, log +
gsm8k logs alongside it), loading the correct trees. Accuracy against the
recorded baselines:

| Run | Result | Recorded baseline |
|-----|--------|-------------------|
| GSM8K 200 | 0.980 acc / 0.005 invalid | 0.985-0.990 (`start_prompt.md`) |
| GSM8K 1319 | **0.952 acc / 0.001 invalid** | 0.947-0.958, typically 0.950-0.955 |

GSM8K-1319 sits mid-band — no detectable regression from the 575 absorbed
upstream commits. The 200-question run is 1-2 questions below its band, which is
inside 1 sigma (~0.009) at that sample size and not meaningful on its own.
Throughput was 598 tok/s over 1319 questions.

Note the launch command must carry the `PYTHONPATH` from the CONTINUE HERE
block. A bare `python -m sglang.launch_server` silently loads
`/sgl-workspace/sglang` and tests nothing.

**Still untested:** the block-FP8 branch of §4b (the checkpoint at
`/shared_nfs/models/Kimi-K3` has no `quantization_config`, so only the
pass-through branch of `_may_fuse_kda_inproj()` was exercised), and any
performance comparison against the pre-rebase tip.

## 5b. Performance: same-day A/B vs the pre-rebase tree (2026-08-25)

Both trees measured on the same host, same `launch.sh` server config, same
client binary (post-rebase `sglang.benchmark.serving`), same parameters:
ISL/OSL 8192/1024, range ratio 1.0, `num_prompts = 8xC`, `warmup = 2xC`,
seed 0, ignore-EOS, request rate inf. Pre-rebase tree is
`backup/k3_opts_0812-pre-rebase-20260825` checked out as a second worktree at
`/sgl-workspace/sglang-k3-prerebase`.

Artifacts in `/shared_nfs/kk/k3_rebase_20260825/`: `sweep.sh` /
`sweep_prerebase.sh`, per-case `c<N>.log` and `pre_c<N>.log`,
`baseline_pre_rebase.tsv` (the older stored baseline).

| C | TTT pre | post | delta | TTFT pre | post | delta | TPOT pre | post | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1149.39 | 1143.83 | -0.48% | 822.27 | 821.70 | -0.07% | 14.87 | 14.95 | +0.54% |
| 4 | 1985.41 | 1978.31 | -0.36% | 1481.84 | 1478.39 | -0.23% | 16.70 | 16.75 | +0.30% |
| 8 | 3155.65 | 3141.73 | -0.44% | 2617.56 | 2609.50 | -0.31% | 20.43 | 20.56 | +0.64% |
| 16 | 4974.41 | 4966.07 | -0.17% | 3894.73 | 3885.13 | -0.25% | 25.16 | 25.22 | +0.24% |
| 32 | 7022.21 | 7010.09 | -0.17% | 7313.52 | 7184.53 | -1.76% | 34.09 | 34.20 | +0.32% |
| 64 | 9042.88 | 9037.54 | -0.06% | 14000.19 | 13841.92 | -1.13% | 50.32 | 50.36 | +0.08% |

Aggregate throughput delta **-0.192%**.

### Conclusion

The rebase carries a **small but real systematic throughput cost of roughly
0.2-0.5%**, largest at low concurrency (C2 -0.48%) and vanishing by C64
(-0.06%). Throughput is down at all six points and TPOT is up at all six —
both signs consistent. TTFT is slightly better at high concurrency
(C32 -1.76%, C64 -1.13%). This is not large enough to block the integration,
but it is not zero.

**Why this is above noise, not drift.** Run-to-run variance was measured, not
assumed: post-rebase C8 gave 3141.73 and 3139.58 tok/s (0.07% apart),
pre-rebase C8 gave 3154.04 and 3155.65 (0.05% apart). Noise is ~0.05-0.07%,
so the -0.44% at C8 is roughly six times it. C8 has two measurements per side
and is the firmest point; the other five have one per side, consistent in
direction but single-shot.

### Correction to the earlier stored-baseline comparison

An initial pass compared the post-rebase sweep against
`baseline_pre_rebase.tsv` and reported C8 TTFT +15.71% plus an aggregate
-0.50%. Both readings were wrong in attribution:

- **C8 TTFT is not a rebase regression.** The pre-rebase tree produces
  2612.71 / 2617.56 ms today, matching post-rebase 2609.50 / 2611.05. The
  stored baseline's 2255.18 ms is not reproducible on either tree.
- **Only that one cell drifted.** Stored-baseline throughput is within
  -0.59%..+0.23% of the pre-rebase tree measured today, and stored TTFT is
  within +/-0.11% at C2/C16/C32. The stored table remains usable for
  throughput and TPOT; its C8 TTFT value does not.
- The aggregate cost against the same-day pre-rebase tree is **-0.192%**, not
  -0.50%; roughly half of the original figure was baseline staleness.

## 5c. vLLM-ATOM recipe workload replayed on SGLang (2026-08-25)

Replicates the 68K-context GSM8K workload from
`/workspace/Kimi-K3-vLLM-ATOM-recipe.md` against the rebased SGLang tree.
Artifacts in `/shared_nfs/kk/k3_aiperf/`.

### How it was set up

`run_aiperf_workload_shapes.sh` could not be obtained - it is not in the public
`ROCm/ATOM` repo (`tools/`, `scripts/`, `scripts/performance/`, `recipes/` all
checked), is not indexed publicly, and this host is inside a container with no
docker CLI to open `rocm/atom-dev:vllm-kimi-k3-20260807`. **It is not needed:**
recipe section 9 documents a driver-free path that calls `aiperf profile`
directly with a pre-built prompt file, and the recipe author certifies the two
paths equivalent (measured ISL 68,088 vs 68,082).

Built from the recipe's own appendices: `make_gsm8k_corpus.py` produced
`gsm8k_corpus.txt` (8,312 problems, 4,272,924 chars) and `make_prompt_file.py`
produced `prompts.jsonl` (300 entries, prefix 63,240 x8 + fresh 4,760 x300,
round-trip ISL 68,000). aiperf 0.11.0 installed; sglang/aiter still import.

Server: `launch_baseline.sh` - the validated K3 env set (all 28 exports
identical to `launch.sh`), `--prefill-attention-backend aiter`,
`--decode-attention-backend triton`, `--kv-cache-dtype fp8_e4m3`, radix cache
ON, `--max-running-requests 64`, `--chunked-prefill-size 16384`,
`--context-length 70000`, kimi_k3 tool/reasoning parsers.
**Speculative decoding OFF** - this run isolates the engine difference.

### Results (735 requests, 0 errors, cache-hit instrumented)

Metrics sampled during the run by `sample_metrics_sglang.py` (see below).

| C | Requests | Cache Hit % | TTFT P50 | TTFT P90 | ITL P50 | In tok/s/gpu | Out tok/s/gpu |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.09 | 3954 | 4103 | 14.61 | 931.2 | 4.79 |
| 2 | 10 | 55.58 | 1310 | 5103 | 16.17 | 1975.0 | 10.15 |
| 4 | 20 | 91.23 | 1361 | 3317 | 18.07 | 4134.6 | 21.25 |
| 8 | 40 | 94.98 | 1446 | 2687 | 21.86 | 7247.3 | 37.26 |
| 12 | 60 | 94.57 | 1705 | 3718 | 25.96 | 8899.1 | 45.75 |
| 16 | 80 | 92.93 | 3716 | 5044 | 29.95 | 9605.9 | 49.38 |
| 24 | 120 | 92.99 | 4761 | 7258 | 37.88 | 11357.2 | 58.38 |
| 32 | 160 | 92.93 | 5818 | 9998 | 51.50 | 11416.3 | 58.69 |
| 48 | 240 | 92.93 | 8428 | 14203 | 67.50 | 12559.1 | 64.56 |

Hit rate converges to ~93%, the `--cache 93` design target, confirming the
prompt construction and prefix caching behave as intended.

Against the recipe's two published points (vLLM-ATOM 2026-08-07, **with**
DSpark speculative decoding on; SGLang **without**):

| | vLLM-ATOM C16 | SGLang C16 | vLLM-ATOM C24 | SGLang C24 |
|---|---:|---:|---:|---:|
| Cache hit % | 79.99 | **92.93** | 83.71 | **92.99** |
| TTFT P50 | 1039 | 3716 (+258%) | 1223 | 4761 (+289%) |
| TTFT P90 | 18419 | 5044 (-73%) | 19770 | 7258 (-63%) |
| ITL P50 | 32.04 | 29.95 (-6.5%) | 42.83 | 37.88 (-11.6%) |
| In tok/s/gpu | 7297 | 9605.9 (+31.6%) | 9111.4 | 11357.2 (+24.6%) |
| Out tok/s/gpu | 37.52 | 49.38 (+31.6%) | 46.84 | 58.38 (+24.6%) |

SGLang delivers 25-32% more throughput and 6-12% better ITL **without**
speculative decoding, against a vLLM-ATOM run that had it on.

**The TTFT gap is real and is not a cache artifact.** The working hypothesis
had been that a lower SGLang hit rate could explain the median TTFT
difference. The instrumented run refutes it: SGLang's hit rate is *higher* by
13 and 9 points at C16 / C24 and the median TTFT is still 3-4x worse. What
remains is a genuine difference in prefill scheduling, and the shape is
consistent across the sweep - vLLM-ATOM has the lower median with a very long
tail (P90 ~18-20 s), SGLang the higher median with a much tighter distribution.
Attributing it further needs a scheduler-level look, not another sweep.

The 13-point hit-rate advantage is plausibly the page-size difference: SGLang
ran its default paging against vLLM's `--block-size 128`, and finer pages track
prefix boundaries more closely. That turns "page size not aligned" from a
caveat into a variable with an observed consequence - worth a `--page-size 128`
run to confirm.

### DSpark speculative decoding (RadixArk draft)

Same server config plus `--speculative-algorithm DSPARK`,
`--speculative-draft-model-path .../RadixArk/Kimi-K3-DSpark`,
`--speculative-attention-mode decode`. Draft window is **not** free: RadixArk's
`config.json` carries `block_size: 7`, so gamma=7 and
`speculative_num_draft_tokens` resolves to 8. Launch script `launch_spec.sh`,
artifacts `artifacts_spec/`, per-point acceptance via `accept_by_window.py`.

| C | Cache Hit % | TTFT P50 | TTFT P90 | ITL P50 | In tok/s/gpu | Out tok/s/gpu | Accept % | AcceptLen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09 | 4005 | 4025 | 8.00 | 1263.6 | 6.50 | 38.71 | 3.71 |
| 2 | 39.19 | 974 | 4037 | 12.77 | 2612.5 | 13.43 | 38.14 | 3.67 |
| 4 | 33.92 | 4051 | 4454 | 23.64 | 2692.0 | 13.84 | 36.58 | 3.56 |
| 8 | 46.96 | 490 | 2645 | 25.47 | 6169.1 | 31.71 | 38.28 | 3.68 |
| 12 | 47.60 | 601 | 3503 | 33.10 | 7516.9 | 38.64 | 36.70 | 3.57 |
| 16 | 47.30 | 579 | 11804 | 40.19 | 7681.9 | 39.49 | 37.03 | 3.59 |
| 24 | 47.54 | 7668 | 10739 | 43.82 | 8134.6 | 41.82 | 39.64 | 3.77 |
| 32 | 46.59 | 16953 | 27421 | 46.55 | 7185.0 | 36.94 | 35.32 | 3.47 |
| 48 | 47.09 | 30500 | 51795 | 44.58 | 7614.5 | 39.14 | 35.62 | 3.49 |

Against the no-spec run on the same host:

| C | in/s/gpu delta | ITL P50 delta | TTFT P50 |
|---:|---:|---:|---|
| 1 | +35.7% | -45.2% | 3.95 -> 4.00 s |
| 2 | +32.3% | -21.0% | 1.31 -> 0.97 s |
| 4 | -34.9% | +30.8% | 1.36 -> 4.05 s |
| 8 | -14.9% | +16.5% | 1.45 -> 0.49 s |
| 12 | -15.5% | +27.5% | 1.71 -> 0.60 s |
| 16 | -20.0% | +34.2% | 3.72 -> 0.58 s |
| 24 | -28.4% | +15.7% | 4.76 -> 7.67 s |
| 32 | -37.1% | -9.6% | 5.82 -> 16.95 s |
| 48 | -39.4% | -34.0% | 8.43 -> 30.50 s |

**Speculation works; it just does not pay above C2 on this workload.** Acceptance
is stable at 35-40% (accept length 3.5-3.8 of 8 draft tokens) at every
concurrency, so the losses are not a broken draft. At C1-C2 the GPU has spare
compute and speculation converts it into throughput and latency; from C4 up the
batch already saturates and the draft + verify work displaces real work. TTFT
blows up at high concurrency (8.4 s -> 30.5 s at C48) because the draft model
must prefill the same 68K prompt, close to doubling prefill cost.

Two traps worth recording:

- **A single end-of-run metrics sample says nothing.** The last sample of the
  run reads `spec_accept_rate=0.0143`, `spec_accept_length=1.1` because traffic
  had drained. Across all 1,462 samples the mean is 0.351 / 3.45, and 1,389 of
  them exceed 5%. Read the distribution, not the tail.
- **The 47% cache-hit column here is an artifact, not a regression.** The draft
  model's prefill increments the same `prefill_effective_tokens_total` under
  `mode="input"` with no matching hit (there is only one `model_name` label),
  roughly doubling the denominator. The target's own hit rate is unchanged from
  the ~93% of the no-spec run.

**Getting it to start at all took three fixes**, each worth knowing:

1. `--speculative-num-draft-tokens 3` (mapped from the recipe's
   `num_speculative_tokens: 3`) is rejected — it must equal gamma+1, and gamma
   comes from the draft checkpoint's `block_size`, not from the flag.
2. With the default `--speculative-attention-mode prefill`, target verify enters
   as an extend with `qseqlen=8` and hits
   `asm_mla.cu:193 get_heuristic_kernel_mla: cannot get heuristic kernel!
   q_type:bf16 kv_type:fp8 gqa:16 ps:1 qseqlen:8`. AITER registers only
   `"bf16" "fp8" 16 1 4 0 1` — qSeqLen 4, not 8. Setting the mode to `decode`
   routes verify to Triton; real prefill still uses AITER.
3. `SGLANG_K3_AITER_MLA_GATE` is **not** involved — it controls the MLA gate
   projection (`mla_gate_aiter_hip`), not the attention kernel. Setting it to 0
   changed nothing.

**Kernel path confirmed against PR #35499** (statically, from the merged diff
plus local code and configs — not a runtime trace). DSpark has no
`draft_extend` phase: `dspark_draft.py` builds its propose batch with
`ForwardMode.TARGET_VERIFY`, so both paths go through `verify_mla`, and both
land on tuned `_BLOCK_CONFIG` entries keyed by head_dim — draft (RadixArk GQA)
on `64: (4, 256, 4)`, target (K3 MLA, 512+64) on `576: (4, 64, 8)`. All gates
hold: `is_gfx95_supported()` true, `SGLANG_ENABLE_SPLITKV_VERIFY` defaults true,
`speculative_eagle_topk=1`, and `KV_GROUP_NUM % BLOCK_H == 0` (4 % 4).

### Report page

`/shared_nfs/kk/k3_aiperf/report.html` — self-contained page with all three
tables, throughput and ITL charts, and the caveat list. Not published as an
Artifact: this session authenticates with `ANTHROPIC_API_KEY`, which blocks the
claude.ai login Artifacts require.

### Metrics instrumentation

`sample_metrics.py` (Appendix B) scrapes vLLM counter names and yields nothing
against SGLang, whose `/metrics` also needs `--enable-metrics` (it 404s
otherwise). `sample_metrics_sglang.py` emits the same short keys, so
`build_perf_table.py` runs unchanged:

- `cache_hits` <- `sglang:prefill_effective_tokens_total` modes
  `device_hit + host_hit + storage_hit`
- `cache_queries` <- that metric summed over **all** modes

The metric is per-tp_rank; every rank sees the same prefill, so summing ranks
scales both operands and leaves the ratio unchanged. Speculative slots stay 0
when spec decode is off - SGLang exposes `sglang:spec_accept_rate` /
`spec_accept_length` as gauges rather than monotonic counters, so they are
recorded under `spec_gauges` instead of being forced into counter slots, which
would produce wrong acceptance figures once spec decode is enabled.

**Mapping bug, caught and fixed:** the first version used `mode="input"` alone
as the denominator and produced hit rates of 1300-1700%. `mode="input"` is the
*miss* portion (tokens actually prefilled); the metric's own docstring defines
the windowed rate as `rate(sum of *_hit) / rate(sum of all modes)`. Both
operands were preserved in the samples file, so the fix was arithmetic on the
existing `metrics_samples.jsonl` - no re-run needed.

### Memory-mapping trap (cost one full sweep)

The first attempt mapped vLLM's `--gpu-memory-utilization 0.93` straight onto
SGLang's `--mem-fraction-static 0.93`. **These are not the same quantity.**
SGLang's figure statically reserves weights + KV pool and leaves activations to
the remainder; the KV pool came out at 2,142,496 tokens - roughly 3x this
workload's ~750K need - leaving ~17 GB, and all eight ranks hit
`torch.OutOfMemoryError` in `forward_extend` during C32. C32 returned 96/160
valid, C48 returned 0/240. At `--mem-fraction-static 0.85` the pool is
1,234,418 tokens, still ample, with zero OOM.

## 6. Related: DSpark draft models

Downloaded 2026-08-25 to `/shared_nfs/huggingface_models/`:

- `RadixArk/Kimi-K3-DSpark` — 4.50 GB, `DSparkDraftModel`, `model_type: qwen3`,
  GQA (64 heads / 16 KV, `head_dim=64`), ships `dspark.py` + `dflash.py`.
- `Inferact/Kimi-K3-DSpark` — 7.12 GB, `K3DSparkModel`, `model_type: k3_dspark`,
  MLA (`q_lora_rank=1536`, `kv_lora_rank=512`).

Same name, different architectures, **not interchangeable**. PR #35499 gates on
`_hf_arch(config) == "DSparkDraftModel"`, so its tuned draft-attention kernel
fires only for **RadixArk**; Inferact's MLA config takes the existing `576:`
path. The vLLM-ATOM recipe at `/workspace/Kimi-K3-vLLM-ATOM-recipe.md` specifies
Inferact and predates RadixArk's upload — reproduce that recipe with Inferact.

Base model: `/shared_nfs/huggingface_models/moonshotai/Kimi-K3`. The recipe's
`/models/Kimi-K3` paths do not exist on this host.

## 7. Also done this session

`/workspace/claude-skills/` had 89 files referencing the dead `/dockerx` mount.
Rewritten: the three home aliases (`/dockerx/home/wunhuang{,/tmp}`,
`/dockerx/var/amdsgl/kk/workspace`) all collapse to `/workspace`; model paths go
to `/shared_nfs/huggingface_models/<org>/`, except DeepSeek-V4-Flash which lives
at `/shared_nfs/hyperloom/models/`. Three rewritten targets do not exist and are
flagged in place: `moonshotai/Kimi-K3-DSpark`, `amd/DeepSeek-R1-MXFP4`, and
`kmd` (whose weights the docs already record as deleted).
