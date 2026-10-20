Owner node: mi355-4

# Proxy MoE 24L TP8/EP8 — profile & optimization on mi355-4

## CONTINUE HERE

**Handoff 2026-10-20 14:5x (UTC+8):** work moves to another node → use `SESSION_PROMPT_1020.md`;
the new node creates its own `OPT_<hostname>.md`. mi355-4 state at handoff: no server/tuner
running, GPUs idle; AITER local edits exported to `patches/aiter_2bb549102_proxymoe_tuning.patch`,
tuned CSV to `configs/`.

**Status:** first warm profile analysed (2026-10-20); no optimization tried yet.
**Next:** T2/T2b done (small gains). Decide EP1 (902 tok/s, TTFT 359, TPOT 16.4) vs EP8
(813 tok/s, TTFT 308, TPOT 18.5); then T3 (prefill comm overlap, 29% of EP8 prefill) / T4 (FP32
router GEMM 11.5%). Decode MoE needs a kernel beyond tuner candidates (production
`mfma_moe1_silu_mul_*_pm1_async_v33` beats all tuned rows at M≤1024). Earlier note — T2 (user: tune BOTH EP1 and EP8 shapes). Running on mi355-4 since 02:3x UTC: default
tuner then `--mxfp4-flydsl`, `--mp 8`, input `/shared_nfs/kk/proxy_moe/tune/proxymoe_fp4_untuned_fmoe.csv`
(2 keys x M in {1..8192}, 28 rows) → `tune/tuned_default.csv`, `tune/tuned_flydsl.csv`, logs
`tune/tune_*.log`. Then merge best-per-row into
`/sgl-workspace/aiter/aiter/configs/model_configs/proxymoe_fp4_tuned_fmoe.csv` (auto-merged by AITER
`get_config_file`) and re-run T1+EP1 and T1+EP8.
Tuner fix (local, uncommitted, AITER tree): `csrc/ck_gemm_moe_2stages_codegen/mxfp4_v2_tune_utils.py`
`v2_stage1_sorted_ref` had a per-row Python loop with 3 `.item()` syncs per sorted row (~131k rows
at M=8192 top16) → first attempt stalled >30 min in it. Vectorized (same output, checked
bit-exact vs old on random data); restarted 2026-10-20 ~03:0x UTC with `python3 -u`.
- `--mxfp4-flydsl` tuner: UNUSABLE for this model — every G1 candidate rejected by
  `mxfp4_gemm1_kernels._assert_supported`: "B-scale addressing covers at most 32 K tiles, got
  D_HIDDEN/BK=64 (H=16384)" (same >8192 class of limit as known issue 1). Possible kernel fix item.
- Default tuner IS the right one: it adds FlyDSL `gen_flydsl_2stages_task` / `_v2_` candidates
  (the in-use `mfma_moe1_silu_mul_*` family is `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`)
  plus CK: 10528 task groups for 28 rows. Restarted ~03:3x UTC with `--verbose` (progress =
  count of `[Done]` in `tune/tune_default.log`).
- That run (42m50s, 10528 tasks) produced 0 tuned rows: all FlyDSL v2 stage2 candidates failed
  `mxmoe_dispatcher.mxfp4_moe_gemm2`: "D_HIDDEN (16384) exceeds compile cap HIDDEN_MAX (8192)"
  (11200 hits). HIDDEN_MAX is only a host check + compile-cache bucket (N is runtime;
  comm_fused_moe passes HIDDEN_MAX=model_dim). Local fix (uncommitted, AITER tree): grow the bucket
  to ceil(D_HIDDEN/8192)*8192 instead of raising. Probe: 2 rows (EP1 M=8192, EP8 M=16) →
  `tune/probe_out.csv`, log `tune/probe.log`.
- Probe result (10m42s): EP1 M=8192 tuned `flydsl_moe1_…t128x64x256_w4_bnt0_fp4` 1496 µs +
  `flydsl_moe2_layout_…t64x256x128_reduce_persist_sbm128` 2647 µs = 4.14 ms vs production
  (t8 trace) 1.87 + 3.59 = 5.46 ms → −24% MoE prefill. EP8 M=16 tuned 1029 µs vs production
  442 µs — NOT comparable: tuner routes all 256 pairs into the 64 local experts (~64 experts read,
  ~6.2 TB/s) while real EP8 decode only sees ~1/8 of pairs locally (~25 experts). Tuner has no
  EP masking → EP8 rows from it would regress; do not ship EP8 tuned rows.
- Structural: EP1 moe2 writes topk(16) x hidden(16384) partials per token = 8x EP8's output
  traffic (4.3 GB per 8192-token chunk) → EP1 prefill stays slower than EP8 even after tuning
  (est. 533 → ~470 ms vs EP8 425 ms).
- Running: EP1 only, remaining 13 M → `tune/tuned_ep1.csv`, log `tune/tune_ep1.log`.
- EP1 tune done (63m28s, all 14 rows, err ~1%), `tune/tuned_ep1.csv`. Tuned vs production
  (TP0 traces): M=8192 4.14 vs 5.46 ms (better); M=16 521 vs 451 µs (WORSE — production's
  `mfma_moe1_silu_mul_*_pm1_async_v33` family apparently not in tuner candidates). Installed only
  rows M ∈ {2048, 4096, 8192} → `/sgl-workspace/aiter/aiter/configs/model_configs/proxymoe_fp4_tuned_fmoe.csv`
  (uncommitted). Validation run: `t2_ep1_tunedprefill_*` (T1 + EP1).
- EP8-aware tuning plan (T2b, after EP1 tune): runtime EP passes global topk_ids (0..511) +
  `expert_mask` (len 512, 64 local) and forces stage2 accumulate (`moe_sorting` `accumulate or
  expert_mask`); `get_2stage_cfgs(is_ep=True)` discards MXMOE / inline-sort tuned cfgs. Tuner has no
  mask support (`--e2e_tune` is gfx942-only). Emulation: tune with topk' = topk/ep = 2 over the 64
  local experts (same expected local pairs M*16/8 and same per-expert multinomial), stage2
  restricted to `atomic` kn2, then rewrite topk→16 in the CSV. Validate each row with the real op:
  `aiter.fused_moe(..., expert_mask=mask)` tuned vs default-heuristic output + timing.
- T2b started 2026-10-20 ~06:05 UTC: added env `TUNE_MOE_STAGE2_ATOMIC_ONLY` to
  `gemm_moe_tune.py` (skips non-`_atomic` kn2 in FlyDSL v1/v2 stage2 loops; uncommitted).
  Input `tune/ep8eq_untuned.csv` (topk=2, 64 experts, inter 4096, 14 M) → `tune/tuned_ep8eq.csv`,
  log `tune/tune_ep8eq.log`, 3808 tasks.
- T2b result (26m56s): tuner M=8192 2.28 ms vs production 2.80 ms; M=16 518 vs 442 µs (worse →
  decode not installed). Op-level check `scripts/ep_moe_check.py` (real `fused_moe` + expert_mask,
  production weight layout, random bytes): tuned vs default M=2048/4096/8192 = 1581/2093/3250 vs
  1645/2154/3414 µs (only −4..−5% e2e op); cos 0.99936, rel_l2 3.6% (systematic, fp4 requant path
  difference; default vs default rerun rel_l2 ≤2e-4). Installed EP8 rows M≥2048 (topk rewritten
  to 16) into `model_configs/proxymoe_fp4_tuned_fmoe.csv` (now 3 EP1 + 3 EP8 rows).
  Serving: prefill 425→410 ms/16k tok (moe1 `…fp4q_sort…` fuses act-quant; stage2 `gemm2_a4w4_port…atomic`). Current best for bench (1024/256, c16): T1 + EP1 = 893 tok/s.
One variable per experiment; record each run in Results and update the item's Status.
**Files:** model `sglang/srt/models/laguna.py`; MoE → quark MoE method → `aiter.fused_moe`
(log: `no tuned FlyDSL config ... using 2stage default`); analyser `scripts/trace_kernels.py`.
**Repro (profile + bench, ~5 min):**
```bash
PKG=/workspace/models/proxy_moe_sglang/proxy_moe_sglang; W=/shared_nfs/kk/proxy_moe
ART=/shared_nfs/kk/results/proxy_24l/<tag>; cd /tmp && NUM_LAYERS=24 TP=8 EP=8 SKIP_GEN=1 RUN_PROFILE=1 \
  AITER_LARGE_HIDDEN_WORKAROUND=1 TOKENIZER_DIR=$W/tokenizer MODEL_DIR=$W/proxy_24l ART=$ART \
  SERVER_EXTRA_ARGS="<flags under test>" bash $PKG/scripts/run_stage.sh > $ART.stage.log 2>&1
python3 $PKG/tools/summarize_run.py $ART --out $ART/summary.json
python3 /workspace/claude-skills/proxy-moe/scripts/trace_kernels.py \
  $(find $ART/profile -name '*TP-0-EP-0-DECODE*') --steps 5
```
**Pass criteria:** smoke passed, bench 64/64; compare median TPOT (19.1 ms) / TTFT / out tok/s
against the warm baseline row.

## JIT finding (cold vs warm, same container)

Cold run: long 6000-token smoke = 28.8 s; warm rerun = 0.48 s (short 0.92→0.33 s, conc 2.4→1.4 s).
The 29 s gap sits on the first M=8192 MoE call (only `fused_moe ... 2stage default` lines in the log)
→ first-use kernel JIT/compile. Always benchmark warm (or discard the first run per container).

## Warm profile — `/shared_nfs/kk/results/proxy_24l/warm_profile_20261020_0151` (TP0)

Profile batch: bs=16, input 1024 (prefill = 2 chunks of 8192 tok), decode 5 steps.

**Decode: 19.7 ms/step GPU** (matches TPOT 19.1 ms → GPU-bound, little CPU/launch gap).
- MoE 58.8% — `mfma_moe1_silu_mul_afp4_wfp4…t32x128x256` 0.30 ms/layer, `mfma_moe2…t32x128x256`
  0.16 ms/layer.
- All-reduce 16.2% — `aiter::cross_device_reduce_2stage`, 49 calls/step (2/layer), ~65 µs each for
  16x16384 bf16 = 512 KB.
- rmsnorm 3.9%, attention (`_unified_attention_gluon_kernel`) 2.6%, dense BF16 GEMMs ~7%,
  shared expert `_gemm_afp4wfp4` ~2%, `mxfp4_moe_sort` (AITER patch split path) 0.7%.

**Prefill: 425 ms for 16384 tokens (212 ms per 8192 chunk).**
- All-reduce 27.5% — `quickreduce::allreduce_prototype_twoshot<…CodecQ8>`, 1.19 ms per 256 MiB call.
- MoE 32.6% — moe1 `t128x128x256` 1.62 ms/call, moe2 `persist_cu256` 1.18 ms/call.
- Dense GEMM 21.6% — biggest `Cijk_…_S_B_…MT128x128x64` 0.97 ms/call (hipBLASLt; `S` output type
  suggests the FP32 router GEMM — UNVERIFIED, check shapes), triton `_gemm_a16_w16` 0.52 ms,
  `MT288x288x64` 0.42 ms.
- `dynamic_per_group_scaled_quant` (fp4) 3.5%, rmsnorm 4.6%, attention 1.3%.

### Roofline estimates (hypotheses — derived, not counter-measured)

Peaks used (MI355X): HBM 8 TB/s, MXFP4 dense ~10 PF, BF16 dense ~2.5 PF.
- Decode moe1: 256 token-expert pairs/layer over 512 experts; ~25 distinct local experts/rank
  expected (uniform routing) x 67 MB w13 = ~1.7 GB in 0.30 ms → ~5.6 TB/s (~70% HBM). moe2 ~5.2 TB/s
  (~65%). → decode MoE is near memory roofline; headroom ~30% at best.
- Prefill moe1: 2x16384x16384x8192 = 4.4 TFLOP in 1.62 ms → ~2.7 PF (~27% FP4 peak); moe2 ~1.9 PF
  (~19%). → prefill MoE far from compute roofline; untuned config is a likely cause.
- Decode all-reduce: 512 KB in 65 µs is latency-dominated, not bandwidth.
Falsify before acting: rocprof counters (FETCH_SIZE / MFMA util) on moe1/moe2.

## All-reduce path in use (verified 2026-10-20, warm_profile run)

No RCCL kernels in either trace. Server log: `[AR] Using AiterCustomAllreduce (AMD default)`;
`disable_custom_all_reduce=False`, `enable_aiter_allreduce_fusion=False`.
- Decode (16x16384 bf16 = 512 KB): AITER custom all-reduce, `aiter::cross_device_reduce_2stage`.
- Prefill (8192x16384 bf16 = 256 MiB): **QuickReduce INT8** (`quickreduce::…CodecQ8`, bf16 cast to
  fp16), because the container env sets `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` and the message is above
  the QR min size (`quick_all_reduce.py` `_QR_MIN_SIZE`). This is lossy and differs from the
  reference image's default → see T0.
- Other env set by the image that may matter: `NCCL_MIN_NCHANNELS=112`, `SGLANG_MOE_PADDING=1`,
  `HIP_FORCE_DEV_KERNARG=1`, `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

## AR wait attribution (decode, 2026-10-20)

`scripts/rank_skew.py`: across 8 ranks MoE time is flat (56–59 ms / 5 steps) but comm ranges
13.5–29 ms → comm kernels absorb waiting. `scripts/ar_wait_attrib.py`: AR time = 8.7 µs +
0.97 × (max_rank_segment − own_segment), r = 0.99; min-over-ranks AR p50 = 8.1 µs.
Split by segment: AR after attention 9.8 µs (skew 2.2 µs); AR after MoE 103.5 µs (skew 96.7 µs).
→ decode "comm 16%" is really per-layer EP load imbalance in MoE, not all-reduce cost.

## Task items

Status: TODO / WIP / DONE / DROP. Share = % of TP0 GPU time in the warm profile.

| ID | Item | Phase / share | Status | Done (UTC+8) | Result / notes |
|----|------|---------------|--------|--------------|----------------|
| T0 | Decide `ROCM_QUICK_REDUCE_QUANTIZATION` (env INT8; lossy) | prefill comm 27.5% | DROP | 2026-10-20 | user allows INT8 prefill AR; keep env as is |
| T1 | Decode AR: `--enable-aiter-allreduce-fusion` (AR+rmsnorm) | decode comm 16.2% + norm 3.9% | DONE (keep on) | 2026-10-20 | GPU 19.7→18.7 ms/step, TPOT 19.1→18.6, 789→808 tok/s; real AR cost ~8–10 µs, rest = wait after MoE (see "AR wait attribution") |
| T8 | MoE EP imbalance in decode: `--ep-size 1` (MoE TP8) vs EP8 | ~2.3 ms/step wait after MoE | DONE (trade-off) | 2026-10-20 | EP1: TPOT 18.6→16.5, 808→893 tok/s; prefill 425→533 ms/16k tok, TTFT 316→371 (moe2 output traffic 8x, structural) |
| T2 | Tune FlyDSL fused_moe, EP1 key (inter 512, 512 experts), M 1..8192 | EP1 prefill MoE 51% | DONE (prefill rows only) | 2026-10-20 | prefill 533→510 ms, 902 tok/s, TTFT 359; tuned decode rows slower than production → not installed |
| T2b | EP8-aware tuning (topk=2 equivalent, atomic-only stage2) + op-level check with expert_mask | EP8 prefill MoE 32.6% | DONE (prefill rows only) | 2026-10-20 | op −4..−5%, cos 0.99936 vs default; serving prefill 425→410 ms, 813 tok/s, TTFT 308 |
| T3 | Prefill comm overlap (TBO / comm-compute overlap) or DP-attention | EP8 prefill comm 28.9% | TODO | — | check Laguna supports TBO |
| T4 | Identify FP32 hipBLASLt `Cijk_…_S_B_MT128x128x64` GEMM (router?) and options | prefill GEMM ~11.5% | TODO | — | keep router FP32 unless agreed |
| T5 | rocprof counter check (FETCH_SIZE / MFMA util) on moe1/moe2 | — | TODO | — | roofline numbers are derived, not measured |
| T6 | Small: bf16→fp32 router-input copy, shared-expert add, split-quant | ~2–5% | TODO | — | |
| T7 | Upstream AITER fused MX-quant >8192 cols fix (replace local patch) | 0.7% decode | TODO | — | local patch in `aiter/ops/quant.py` |
| T9 | Decode MoE kernel: beat `mfma_moe1_silu_mul_*_pm1_async_v33` at M≤1024 (no tuner candidate does) | decode MoE ~60–64% | TODO | — | ~65–70% HBM BW today; needs kernel work |
| T10 | Upstream AITER fixes found during T2: tuner `v2_stage1_sorted_ref` vectorize; `mxfp4_moe_gemm2` HIDDEN_MAX bucket; `TUNE_MOE_STAGE2_ATOMIC_ONLY`; `--mxfp4-flydsl` gemm1 32-K-tile limit (H≤8192) | tooling | TODO | — | first three applied locally, uncommitted |

## Summary vs baseline (bench 1024 in / 256 out, 64 req, c16; Δ vs warm baseline)

| Config | Out tok/s | TTFT med ms | TPOT med ms | Prefill ms / 16k tok (TP0 GPU) |
|--------|-----------|-------------|-------------|--------------------------------|
| MI350X reference (pinned image) | 753 | 461 | 19.3 | — |
| Warm baseline EP8, defaults | 789 | 316 | 19.1 | 425 |
| EP8 + AR fusion (T1) | 808 (+2.4%) | 316 (0%) | 18.6 (−2.6%) | 424 |
| EP8 + AR fusion + tuned (T2b) | 813 (+3.0%) | 308 (−2.5%) | 18.5 (−3.1%) | 410 |
| EP1 + AR fusion (T8) | 893 (+13.2%) | 371 (+17.4%) | 16.5 (−13.6%) | 533 |
| EP1 + AR fusion + tuned (T2) | 902 (+14.3%) | 359 (+13.6%) | 16.4 (−14.1%) | 510 |

## Results

Append-only; every row carries node + date. GPU ms/step from TP0 decode trace.

| Date | Node | Run tag | Change | Out tok/s | TTFT med ms | TPOT med ms | Decode GPU ms/step | Notes |
|------|------|---------|--------|-----------|-------------|-------------|--------------------|-------|
| 2026-10-20 | mi355-4 | `warm_profile_20261020_0151` | none (warm baseline) | 789 | 316 | 19.1 | 19.7 | 2nd run in container |
| 2026-10-20 | mi355-4 | `t1_arfusion_20261020_0217` | `--enable-aiter-allreduce-fusion` | 808 | 316 | 18.6 | 18.7 | T1; decode AR → `reduce_scatter_cross_device_store` + `local_device_load_rmsnorm_naive` |
| 2026-10-20 | mi355-4 | `t8_ep1_arfusion_20261020_0223` | T1 + `--ep-size 1` (MoE TP8) | 893 | 371 | 16.5 | 16.4 | T8; prefill 533 ms/16k tok (moe2 173 ms); long smoke 24.7 s = cold JIT for new shapes |
| 2026-10-20 | mi355-4 | `t2_ep1_tunedprefill_20261020_0528` | T1 + EP1 + tuned MoE rows M≥2048 | 902 | 359 | 16.4 | 17.6* | T2 partial; prefill 533→510 ms/16k tok; MoE 262→235 ms (moe1 99.5 + `gemm2_a4w4_port…reduce` 91.2 + `moe_reduction` 44.7) — less than tuner's 199 ms (separate reduction kernel). *decode trace has 2 extra non-fused `cross_device_reduce_2stage` (1.39 ms) — likely outlier, tok/s unchanged |
| 2026-10-20 | mi355-4 | `t2b_ep8_tunedprefill_20261020_0634` | T1 + EP8 + tuned MoE rows M≥2048 (EP-eq tuning) | 813 | 308 | 18.5 | — | T2b; prefill 425→410 ms/16k tok; MoE 134→130 ms, quant 31→20 ms |
