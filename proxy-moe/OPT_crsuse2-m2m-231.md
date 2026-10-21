Owner node: crsuse2-m2m-231

# Proxy MoE 24L TP8 — optimization continued on crsuse2-m2m-231

Continues `OPT_mi355-4.md` (read-only here; owned by mi355-4). Task IDs (T3..T10) keep that doc's
numbering.

## CONTINUE HERE

**Status (2026-10-20 07:25 UTC):** env up; warm repro of EP8/EP1 best configs matches mi355-4
within ~1% (serving) / 0.5% (TP0 prefill GPU time); tuned rows confirmed hit in traces.
**T3 step 1 done:** `--enable-dp-attention --dp 8` on EP8 = large regression (229 tok/s, TTFT 9.8 s,
TPOT 24.4) → DROP DP-attn. See "T3 results".
**T4 DONE (2026-10-20):** bf16x3 router in prefill = EP8 TTFT 314→294 (−6.4%), EP1 365→343
(−6.0%); TP0 prefill GPU −9.6% / −7.7%; decode unchanged. Keep on (env
`SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1`, SGLang tree edit uncommitted: `environ.py`, `models/laguna.py`).
Current best: EP1 + AR fusion + tuned + T4 = 902 tok/s / TTFT 343 / TPOT 16.47.
**Status (2026-10-21):** TP8 best = EP1 + AR fusion + tuned + T4 + stage2 b_nt fix: 933.7 / 344.5 /
15.83 (new base). T11 (TP4) done: TP8 levers carry over; TP4-tuned prefill rows pushed; TP4 best
EP1 1090.3 / 256.0 / 13.73, EP4 1002.6 / 234.0 / 15.10 (see "T11"). Items A, B, D closed.
**fixed_seq_len sweep DONE 2026-10-21 07:50 UTC** (50/50, `fixed_seq_len/RESULTS.md`). Before that, user said: sweep first (TP8 + TP4; `proxy-moe/fixed_seq_len/README.md`,
session prompt `fixed_seq_len/SESSION_PROMPT.md`); optimizations after the sweep. Candidates then: F (prefill AR/compute overlap, TTFT up to ~-15%, large), G (server flags, cheap
A/B), I (upstream PRs; b_nt fix first). Doc edits: in worktree `/data/proxy_moe/wt/skills` (the main
checkout has 2 unrelated unmerged files), commit + push from there.
**Pushed 2026-10-20 (on top of the remote branch tips = newer upstream main; NOT yet re-tested on
those bases):** AITER `kkHuang-amd/aiter` `feat/proxy-model-1020` cfc37f9a1..39c03665a (5 commits:
quant >8192 split path, tuning fixes, mega_moe_tp NCK32, stage2 b_nt + D2 env switches, proxy
tuned CSV); SGLang `HaiShaw/sglang` `feat/proxy-model-1020` eae134d61f..effe332427 (Laguna bf16x3
router). Later pushed AITER cfae83274: load_hip_runtime reuses the already-loaded HIP runtime (new-base
CustomAllreduce hung: rank 0 never published its IPC key because hipIpcGetMemHandle failed in the
2nd, _rocm_sdk_devel copy). New-base run env: `source /data/proxy_moe/newbase_env.sh` (FlyDSL 0.4.0
in `/data/proxy_moe/pyenv_new`, required by new AITER; venv untouched); worktree needs
`git submodule update --init --reference /sgl-workspace/aiter/3rdparty/composable_kernel`.
Worktrees: `/data/proxy_moe/wt/{aiter,sglang}`; per-commit patches `/data/proxy_moe/patches/push/`.
Not pushed: pre-existing unrelated AITER edits (pa_mqa_logits_fp4_prefill.py, csrc/cpp_itfs/torch_utils.py,
mla .co) and SGLang hipify artifacts.
Local AITER edits: `/data/proxy_moe/patches/aiter_moe2_bnt_and_d2_experiments.patch` (b_nt fix +
default-off env experiments AITER_D2_STREAM_K / AITER_D2_DEFER_TID), `aiter_mega_moe_tp_nck32.patch`.
**Files:** model `sglang/srt/models/laguna.py`; analysers `/workspace/claude-skills/proxy-moe/scripts/`.
**Repro (warm, run from /tmp):**
```bash
PKG=/workspace/proxy_moe_sglang/proxy_moe_sglang; W=/data/proxy_moe
ART=/shared_nfs/kk/results/proxy_24l/<tag>; cd /tmp && NUM_LAYERS=24 TP=8 EP=8 SKIP_GEN=1 RUN_PROFILE=1 \
  AITER_LARGE_HIDDEN_WORKAROUND=1 TOKENIZER_DIR=$W/tokenizer MODEL_DIR=$W/proxy_24l ART=$ART \
  SERVER_EXTRA_ARGS="--enable-aiter-allreduce-fusion" bash $PKG/scripts/run_stage.sh > $ART.stage.log 2>&1
python3 $PKG/tools/summarize_run.py $ART --out $ART/summary.json
```
**Pass criteria (mi355-4 numbers, bench 1024/256, 64 req, c16):** EP8 813 tok/s / TTFT 308 /
TPOT 18.5; EP1 902 / 359 / 16.4.

## Environment on this node (2026-10-20)

- 8x MI355X (gfx950); `rg` not installed (use grep), `jq` present.
- Package at `/workspace/proxy_moe_sglang/proxy_moe_sglang` (not `/workspace/models/...`).
- SGLang `ed0b8f433f` (same as mi355-4; dirty files = image hipify outputs only).
- AITER `2bb549102` (same). Pre-existing unrelated local edits in AITER tree, left alone:
  `pa_mqa_logits_fp4_prefill.py`, `csrc/cpp_itfs/torch_utils.py`, `hsa/gfx950/mla_v4/...nm.co`.
- Applied: `aiter_mx_quant_large_hidden.py` (patched `aiter/ops/quant.py`),
  `patches/aiter_2bb549102_proxymoe_tuning.patch` (clean), tuned CSV copied to
  `aiter/configs/model_configs/proxymoe_fp4_tuned_fmoe.csv`.
- Env: `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (set in container, as on mi355-4), `SGLANG_USE_AITER=1`.
- Work dir `/data/proxy_moe` (local overlay, 28 TB free; `/shared_nfs` here is real NFS at 99%):
  tokenizer from HF pinned rev; `proxy_24l/config.json` sha256 `9b60ba1c…bd7bd` (matches);
  ckpt `gen_dummy_ckpt.py --seed 0 --workers 64`, log `/data/proxy_moe/gen_ckpt.log`.
- Results: `/shared_nfs/kk/results/proxy_24l/<run-tag>/`.

## T11 — TP4 (12 layers, 4 GPUs): do the TP8 optimizations carry over? + TP4 MoE tuning (user, 2026-10-20)

Reference (package, MI350X): TP4 916 tok/s / TTFT 368 / TPOT 15.8. Checkpoint `/data/proxy_moe/proxy_12l`
(config sha 38bbb1c5…43e2 matches; seed 0). Unoptimized baseline uses a PRISTINE env
(`/data/proxy_moe/basebase_env.sh`: worktrees `wt/aiter_base` @2bb549102 + `wt/sglang_base` @ed0b8f433f,
venv FlyDSL 0.3.4.1, package large-hidden patch) because the new base and /sgl-workspace both
carry our code changes. Optimized runs use `newbase_env.sh`. `repro.sh` now takes NL / TPN / GPUS
(ROCR_VISIBLE_DEVICES) / PORTN / BASE_ARGS; A/B pairs run concurrently on GPUs 0-3 and 4-7.
Note: tuned-CSV key (cu_num, token, model_dim, inter, expert, topk, ...) of our EP8 rows (inter 4096,
512 experts) equals the EP4 key → new-base EP4 silently uses EP8-tuned prefill rows.
Plan (rows): (a) pristine EP4 defaults; (b) new base EP4 defaults; (c) + AR fusion; (d) + T4;
(e) EP1 + AR fusion + T4; then TP4 MoE tuning (EP1 inter 1024; EP4 topk-equivalent 4 over 128
local experts, atomic stage2) → prefill rows only, validate, serving A/B.

### T11 results (2026-10-20, GPUs 0-3, sequential; Δ vs (a))

| Row | Config | Out tok/s | TTFT med | TPOT med | TP0 prefill ms/step |
|-----|--------|-----------|----------|----------|---------------------|
| ref | MI350X reference (package) | 916 | 368 | 15.8 | — |
| (a) | pristine base, EP4 defaults | **965.0** | **247.1** | **15.68** | 66.14 |
| (b) | new base, EP4 defaults (code changes only) | 981.2 (+1.7%) | 250.3 (+1.3%) | 15.39 (-1.8%) | 66.21 |
| (c) | + AR fusion | 995.6 (+3.2%) | 251.3 (+1.7%) | 15.14 (-3.4%) | — |
| (d) | + T4 bf16x3 router | 1000.4 (+3.7%) | 239.8 (-3.0%) | 15.11 (-3.6%) | 62.79 |
| (f) | EP1 + AR fusion + T4 | **1089.8 (+12.9%)** | 260.9 (+5.6%) | **13.71 (-12.6%)** | 71.64 |

Runs c231_tp4_{a_base_ep4def,b_new_ep4def,c_new_ep4_arf,d_new_ep4_arf_t4,f_new_ep1_arf_t4}_1020
(cold EP1 c231_tp4_e_new_ep1_cold_1020 1089.7/262.3/13.72). All TP8 levers carry over with the
same sign. EP4 prefill on the new base still runs the DEFAULT kernels (t128x128x256 v1 persist),
not the EP8-tuned rows (trace) → TP4 needs its own tuning. EP1-TP4 (inter 1024) has no tuned rows.
EP1 prefill moe2 21.4 ms/step (vs 11.4 EP4): EP1 output traffic again.
GOTCHA: concurrent lanes are impossible with the package run_stage.sh — `stop_server` does
`pkill -f sglang.launch_server` (line 136) and kills every server on the host; first concurrent
attempt (tp4_base_ep4_*, tp4_new_ep4def_*) is invalid.
TP4 tuning DONE (2026-10-21). First attempt OOM'd: tuner reference `dequant_w` built a full fp32 copy
of all experts (512 x 2048 x 16384 = 64 GiB at EP1-TP4); fixed in worktree
`csrc/ck_gemm_moe_2stages_codegen/mxfp4_v2_tune_utils.py` (row chunks, bf16 output; bit-identical on a
check). Key fact: the tuned-CSV `expert` column is the LOCAL expert count (EP8 rows have expert=64),
so EP4 (128 local) never matched EP8 rows. Op-level validation (`tune_tp4/validate.sh`, 1 GPU,
whole MoE us, default → tuned): EP1-TP4 M=2048 3258→3510 (+7.7%, worse), 4096 4777→4763, 8192
8389→7858 (-6.3%); EP4 2048 2929→3097 (+5.7%, worse), 4096 3858→3677 (-4.7%), 8192 6122→5736
(-6.3%); rel_l2 tuned vs default 3.6% (same systematic fp4 requant-path difference as TP8 rows).
Tuner-reported us disagree with the measurement at M=2048 → trust the measurement.
Installed only faster rows (`tune_tp4/install_tp4.csv` appended to worktree
`model_configs/proxymoe_fp4_tuned_fmoe.csv`): EP1-TP4 M=8192 (inter 1024, expert 512); EP4 M=4096,
8192 (inter 4096, expert 128, topk rewritten 4→16). Pushed AITER fd537aaa4 (rows) + 44a1da311 (tuner fix).

| Row | Config (TP4, new base, + TP4-tuned rows) | Out tok/s | TTFT med | TPOT med | TP0 prefill ms/step |
|-----|------------------------------------------|-----------|----------|----------|---------------------|
| (g) | EP1 + AR fusion + T4 + tuned (vs f) | 1090.3 (+13.0% vs a) | 256.0 (-1.9% vs f) | 13.73 | 71.64 → 68.98 |
| (h) | EP4 + AR fusion + T4 + tuned (vs d) | 1002.6 (+3.9% vs a) | 234.0 (-2.4% vs d) | 15.10 | 62.79 → 60.84 |

T11 conclusion: every TP8 lever carries over to TP4 with the same sign. Best TP4 throughput/TPOT =
(g) EP1: 1090.3 / 256.0 / 13.73 (+13.0% / +3.6% / -12.4% vs pristine EP4 defaults); best TP4 TTFT =
(h) EP4: 1002.6 / 234.0 / 15.10 (+3.9% / -5.3% / -3.7%), all three better than pristine.
Inputs: `/data/proxy_moe/tune_tp4/run_tune.sh` → `ep1_tp4_tuned.csv`, `ep4eq_tuned.csv`
(M 2048/4096/8192; EP4 equivalent = topk 4 over 128 experts, TUNE_MOE_STAGE2_ATOMIC_ONLY=1).

## TTT/GPU + P90 interactivity (dsv41 / InferenceX style), all runs on crsuse2-m2m-231

TTT/GPU = total_throughput (input + output tok/s) / num GPUs; P90 interactivity = 1000 / p90_itl_ms
(tok/s/user; definition matched to AgentX logs, e.g. ITL p90 128.92 ms → 7.76). Bench 1024/256,
64 req, conc 16. Δ vs the unoptimized pristine-base row of each group (both re-run on this node:
TP8 c231_tp8_base_ep8def_warm_1021 788.6 / 324.1 / 19.05, matches mi355-4's 789 / 316 / 19.1).
Generator: `/data/proxy_moe/tools/serving_table.py`.
### TP8 (24 layers, 8 GPUs)

| Config | TTT/GPU (tok/s) | P90 interactivity (tok/s/user) | Output tok/s | TTFT med (ms) | TPOT med (ms) | Run |
|---|---|---|---|---|---|---|
| Unoptimized: EP8, package defaults (pristine base) | 492.9 | 53.2 | 788.6 | 324.1 | 19.05 | c231_tp8_base_ep8def_warm_1021 |
| EP8 + AR fusion + tuned MoE | 504.4 (+2.3%) | 54.5 (+2.4%) | 807.0 (+2.3%) | 314.3 (-3.0%) | 18.64 (-2.2%) | c231_ep8_warm_1020 |
| EP8 + AR fusion + tuned + T4 | 508.7 (+3.2%) | 54.5 (+2.4%) | 814.0 (+3.2%) | 294.0 (-9.3%) | 18.52 (-2.8%) | c231_ep8_t4bf16x3_1020 |
| EP8 + AR fusion + tuned + T4 + b_nt fix | 521.1 (+5.7%) | 56.0 (+5.2%) | 833.8 (+5.7%) | 296.8 (-8.4%) | 18.06 (-5.2%) | c231_ep8_t4_bnt2fix_1020 |
| EP1 + AR fusion + tuned MoE | 558.4 (+13.3%) | 62.9 (+18.1%) | 893.4 (+13.3%) | 364.8 (+12.5%) | 16.53 (-13.2%) | c231_ep1_warm_1020 |
| EP1 + AR fusion + tuned + T4 | 563.5 (+14.3%) | 62.9 (+18.3%) | 901.6 (+14.3%) | 342.9 (+5.8%) | 16.47 (-13.6%) | c231_ep1_t4bf16x3_1020 |
| EP1 + AR fusion + tuned + T4 + b_nt fix | 582.0 (+18.1%) | 65.3 (+22.7%) | 931.2 (+18.1%) | 342.4 (+5.6%) | 15.89 (-16.6%) | c231_ep1_t4_bnt2fix_1020 |
| Same, pushed branches (new base) | 583.6 (+18.4%) | 65.5 (+23.1%) | 933.7 (+18.4%) | 344.5 (+6.3%) | 15.83 (-16.9%) | c231_newbase_ep1_warm_1020 |

### TP4 (12 layers, 4 GPUs)

| Config | TTT/GPU (tok/s) | P90 interactivity (tok/s/user) | Output tok/s | TTFT med (ms) | TPOT med (ms) | Run |
|---|---|---|---|---|---|---|
| (a) Unoptimized: EP4, package defaults (pristine base) | 1,206.3 | 64.8 | 965.0 | 247.1 | 15.68 | c231_tp4_a_base_ep4def_1020 |
| (b) New base, EP4 defaults (code changes only) | 1,226.6 (+1.7%) | 66.0 (+1.9%) | 981.2 (+1.7%) | 250.3 (+1.3%) | 15.39 (-1.9%) | c231_tp4_b_new_ep4def_1020 |
| (c) + AR fusion | 1,244.5 (+3.2%) | 67.1 (+3.7%) | 995.6 (+3.2%) | 251.3 (+1.7%) | 15.14 (-3.4%) | c231_tp4_c_new_ep4_arf_1020 |
| (d) + T4 router | 1,250.5 (+3.7%) | 67.1 (+3.7%) | 1000.4 (+3.7%) | 239.8 (-3.0%) | 15.11 (-3.7%) | c231_tp4_d_new_ep4_arf_t4_1020 |
| (h) + TP4-tuned MoE rows | 1,253.2 (+3.9%) | 67.0 (+3.5%) | 1002.6 (+3.9%) | 234.0 (-5.3%) | 15.10 (-3.7%) | c231_tp4_h_new_ep4_arf_t4_tuned_1021 |
| (f) EP1 + AR fusion + T4 | 1,362.2 (+12.9%) | 75.4 (+16.5%) | 1089.8 (+12.9%) | 260.9 (+5.6%) | 13.71 (-12.5%) | c231_tp4_f_new_ep1_arf_t4_1020 |
| (g) + TP4-tuned MoE rows | 1,362.9 (+13.0%) | 75.3 (+16.2%) | 1090.3 (+13.0%) | 256.0 (+3.6%) | 13.73 (-12.4%) | c231_tp4_g_new_ep1_arf_t4_tuned_1021 |

## Before vs after (bench 1024 in / 256 out, 64 req, c16; Δ vs unoptimized EP8 defaults)

| Config | Node / base | Out tok/s | TTFT med ms | TPOT med ms |
|--------|-------------|-----------|-------------|-------------|
| MI350X reference (pinned image) | package reference | 753 (-4.6%) | 461 (+45.9%) | 19.3 (+1.0%) |
| **Unoptimized**: EP8, package defaults (warm) | mi355-4, old base | **789** | **316** | **19.1** |
| EP8 + AR fusion + tuned MoE | this node, old base | 807.0 (+2.3%) | 314.3 (-0.5%) | 18.64 (-2.4%) |
| EP1 + AR fusion + tuned MoE | this node, old base | 893.4 (+13.2%) | 364.8 (+15.4%) | 16.53 (-13.5%) |
| + T4 bf16x3 router (prefill) | this node, old base | 901.6 (+14.3%) | 342.9 (+8.5%) | 16.47 (-13.8%) |
| + stage2 b_nt fix (decode moe2) | this node, old base | 931.2 (+18.0%) | 342.4 (+8.4%) | 15.89 (-16.8%) |
| **Same, pushed branches (new base)** | this node, new base | **933.7 (+18.3%)** | **344.5 (+9.0%)** | **15.83 (-17.1%)** |
| EP8 + AR fusion + tuned + T4 + b_nt fix | this node, old base | 833.8 (+5.7%) | 296.8 (-6.1%) | 18.06 (-5.4%) |

New base = AITER `kkHuang-amd/aiter@cfae83274` + SGLang `HaiShaw/sglang@effe332427` + FlyDSL 0.4.0,
run c231_newbase_ep1_warm_1020 (cold run 933.5 / 347.7 / 15.84; smoke pass, 64/64). Server log
confirms worktree paths; `versions.txt` shows the old pip metadata (run_stage reads
importlib.metadata) — ignore it for new-base runs. New vs old base final: +0.3% / +0.6% / -0.4%
(noise). EP1's TTFT is higher than EP8 by design (moe2 output traffic 8x in prefill, T8).

## T3 notes (static, 2026-10-20)

- Laguna (`models/laguna.py`) uses the layer_boundary stage API (`declare_attn`/`declare_ffn`/
  `append_stages`) and sizes attention by `attn_tp_size` → DP-attention is plumbed.
- TBO: `batch_overlap/operations_strategy.py` knows only DeepseekV2 / Qwen3Moe / MiMoV2 / DSV4;
  Laguna has no `op_*` decomposition, and SGLang TBO overlaps a2a dispatch/combine (DeepEP/mori),
  not TP all-reduce. → TBO for Laguna = port op_* + a2a MoE backend; non-trivial.
- Open question before any T3 work: with top-16 over 8 EP ranks almost every token hits every
  rank, so a2a volume ≈ all-gather volume; DP-attn alone mainly removes the attention-side AR.
  Measure (one variable): `--enable-dp-attention --dp 8` vs current, prefill comm ms per step.

## T3 results — DP-attention (2026-10-20, crsuse2-m2m-231)

One variable vs `c231_ep8_warm_1020`: `+ --enable-dp-attention --dp 8` (trace confirms DP0..7).
TP0 GPU ms/step (5 steps; profile batch 16 req is split 2 per DP rank, MoE still sees all tokens):

| Phase | Category | EP8 base | EP8 + DP-attn |
|-------|----------|----------|---------------|
| prefill | total | 82.46 | 75.69 (−8%) |
| prefill | comm | 23.82 (QR twoshot x19) | 21.35 = NCCL generic 10.95 x9 + QR twoshot 10.38 x9 (−10%) |
| prefill | norm | 3.95 | 0.55 |
| decode | total | 19.02 | 31.82 (+67%) |
| decode | comm | 3.07 | 11.18 (`cross_device_reduce_2stage` 7.70 x25 + NCCL 3.48 x24) |
| decode | gemm | 1.35 | 5.66 (new `MT256x16x128` 5.19 x49: attention weights now unsharded per rank, 8x weight read) |
| decode | moe | 11.46 | 11.08 |

- Prefill comm only −10%: as predicted, top-16 over 8 ranks → gather/scatter volume ≈ all-reduce.
- Decode loses on both replicated attention GEMMs and slower collectives (AR fusion path lost).
- KV capacity 9.57M → 0.48M tokens/rank (fine for this bench, bad in general).
- Serving TTFT 9.8 s with `#queue-req: 0` everywhere: NOT root-caused. Server log shows 1-seq
  prefills per DP rank in lockstep (DP ranks step together for MoE collectives); hypothesis only.
- Verdict: DROP for this model at TP8. Remaining T3 lever = overlap TP all-reduce with compute
  (e.g. split prefill chunk into 2 micro-batches inside Laguna) — no SGLang support, needs code.

## T4 — router FP32 GEMM (2026-10-20, crsuse2-m2m-231)

- Attribution (`/data/proxy_moe/tools/kernel_origin.py`, EP8 warm TP0 EXTEND trace): all 48
  `Cijk_Alik_Bljk_S_B_Bias_…MT128x128x64` kernels (46.8 ms / 5 steps) come from
  `laguna.py:128` `LagunaMoEGate.forward` → `aten::linear` fp32 `[8192,16384] x [512,16384]^T`.
  ~0.98 ms/call = 2·8192·16384·512 / 0.98 ms ≈ 140 TFLOP/s → at FP32-matrix rate (compute-bound).
- Input is bf16 in the model (cast to fp32 exact) → only W needs >bf16 precision. Split
  W = Σ_k bf16 parts, GEMM with fp32 accumulate. Offline bench (`tools/router_split_bench.py`,
  1 GPU, M=8192, random x/W, error vs fp64, top-16 on sigmoid scores):

| Variant | µs | rel_l2 vs fp64 | top-16 rows identical |
|---------|----|----------------|-----------------------|
| fp32 F.linear (current) | 1050 | 2.29e-6 | 100% |
| bf16 k=1 | 139 | 1.66e-3 | 96.89% |
| bf16 k=2 concat-K | 267 | 2.52e-6 | 99.988% |
| bf16 k=3 concat-K | 334 | 7.65e-7 | 100% |
| fp8 e4m3 (W per-channel, x per-token, `_scaled_mm` rowwise, bf16 out) GEMM only | 82 | 3.74e-2 | 43.7% |
| fp8 e4m3 incl. unfused torch x-quant | 702 | 3.74e-2 | 43.7% |

- concat-K needs `cat([x]*3)` per call: 852 us incl. the copy (bench above excluded it) → use
  **concat-N** instead: one GEMM `x @ [P0;P1;P2]^T` → [M,1536] fp32, sum 3 slices: 322 us,
  rel 8.2e-7, top-16 100%.
- FP8 router rejected on numerics: 56% of tokens get a different top-16 set (random data).
  hipBLASLt rowwise `_scaled_mm` only allows bf16 output. Decode caveat for k=3: small-M router is
  weight-read-bound and k=3 reads 48 MB/layer vs fp32 32 MB → use k=3 only for large M.

- k=3 concat-K (`mm([x,x,x], [P0|P1|P2]^T, out_dtype=fp32)`) is 3.1x faster AND more accurate
  than current fp32. Est. prefill saving ≈ 0.72 ms x 9.6 calls/step ≈ 6.9 ms/step of 82.5 (~8%),
  plus the bf16→fp32 input copy (1.37 ms/step, T6) disappears. Estimate only — must A/B in serving.

### T4 serving A/B (one variable: env on), crsuse2-m2m-231

| | EP8 base → T4 | EP1 base → T4 |
|-|---------------|---------------|
| out tok/s | 807.0 → 814.0 | 893.4 → 901.6 |
| TTFT med ms | 314.3 → 294.0 | 364.8 → 342.9 |
| TPOT med ms | 18.64 → 18.52 | 16.53 → 16.47 |
| TP0 prefill GPU ms/step | 82.46 → 74.58 | 101.79 → 93.99 |
| router GEMM ms/step | 9.36 (fp32 `S_B` MT128x128x64) → 2.94 (`BSS` MT192x256x64) + 0.17 sum | same |

bf16→fp32 input copy (1.37 ms/step) gone in prefill. Decode keeps fp32 router (`S_B` MT16x16x256,
0.26 ms/step); decode GPU 19.02→19.45 on EP8 is in comm (wait absorbs skew), MoE 11.46→11.26;
TPOT unchanged. Single run per config.

## T9 — decode MoE kernel (2026-10-20, crsuse2-m2m-231)

Ceiling: Triton streaming read (`/data/proxy_moe/tools/hbm_read_bw.py`) = **7.13 TB/s** achievable
(89% of 8 TB/s spec); `torch.sum` only 5.68 (bad probe). Op-level EP8 decode (`tools/decode_moe_roofline.py`,
real `aiter.fused_moe` + expert_mask, random routing, 1 GPU; bytes = unique local experts x
(w1 71.3 MB + w2 35.7 MB incl. e8m0 scales)):

| M | uniq experts | total us | moe1 us (TB/s, % of 7.13) | moe2 us (TB/s, %) |
|---|--------------|----------|---------------------------|-------------------|
| 1 | 1 | 74.8 | 45.4 (1.57, 22%) | 10.1 (3.53, 50%) |
| 8 | 12 | 252.7 | 149.9 (5.71, 80%) | 78.4 (5.45, 76%) |
| 16 | 30 | 561.8 | 347.2 (6.16, 86%) | 189.7 (5.64, 79%) |
| 32 | 43 | 789.3 | 498.1 (6.16, 86%) | 265.0 (5.78, 81%) |
| 64 | 55 | 979.6 | 617.3 (6.35, 89%) | 335.7 (5.84, 82%) |

Serving (TP0 trace, ~25 experts): moe1 298 us ~6.0 TB/s, moe2 160 us ~5.6 TB/s per layer.
Prize if both hit ~95% of achievable (6.8 TB/s): ~393 vs 458 us/layer → ~1.6 ms/step (EP8 TPOT
~-8%). Bytes assumed = ideal (not yet measured: FETCH_SIZE pending). Hypothesis for moe1 loss:
wave quantization (25 experts x 64 N-tiles = 1600 tiles / 256 CU = 6.25 waves) — unmeasured.

### T9 FETCH_SIZE (EP1 M=16, 190 experts, `rocprofv3_core.sh --pmc FETCH_SIZE --kernel-trace`)

| kernel | FETCH | ideal | ratio | dur | fetched BW | grid (WGs) | VGPR+AGPR | LDS |
|--------|-------|-------|-------|-----|-----------|------------|-----------|-----|
| moe1 t32x128x256 async | 1.709 GB | 1.693 | 1.009 | 262.7 us | 6.50 TB/s (91%) | 4x256 | 28+148 | 54.8 KB |
| moe2 t32x128x256 cshuffle pm1 | 0.849 GB | 0.847 | 1.002 | 146.1 us | 5.81 TB/s (81%) | 128x256 | 68+4 | 16.9 KB |

No over-fetch → moe2 gap is not traffic. Hypothesis (unmeasured): moe2 WG = 32x128 tile, K=512 →
2 K-iterations / ~32 KB weights per WG, no async copy → per-WG fixed cost + latency exposed.
ATT (2026-10-20, 1 CU, EP1 M=16, `/data/proxy_moe/t9_att/` input.yaml + listing.txt; FlyDSL
skills capture-kernel-trace / kernel-trace-analysis; source map collapsed to one line → ISA only):
stalls = 93% of cycles; VMEM-wait 70.7%, VMEM-load 12.3%, barrier 11%.
- Prologue serial metadata chain: pc8 load dword → pc21 vmcnt(0) **12.4%** → pc29/30 dependent
  loads → pc32 vmcnt(1) **12.6%** → branch pc40: 2072 waves enter, 1528 continue (**26% of waves
  are empty M-blocks**, grid y=256 vs ~190 valid) → ~25% of all stall = 2 serialized HBM round
  trips before the first weight load.
- Main body (only 2 K-iterations, 8 dwordx4 per wave = 32 KB weights/WG): waits pc110 6.2%,
  pc145 12.7% (A → LDS), pc167 12.0%, pc201 10.8%; barriers 5.5+2.5+1.5%.
- Reading: per-WG data time ~1.2 us (32 KB at 7.13 TB/s / 256 CU) is comparable to the 2
  metadata round trips → too few bytes in flight. Candidate fixes (each one experiment):
  (1) issue weight loads as soon as expert id is known (overlap token-id / A loads);
  (2) bigger N per WG (tile_n 256) to amortize the prologue; (3) size grid y by valid blocks.
- Exp (2) done 2026-10-20: `AITER_CONFIG_FMOE=/data/proxy_moe/t9_tn/{A,B}.csv` (one token=16 EP1
  row; A = default names explicitly, B = kn2 `..._t32x256x256_atomic_bnt2`). A reproduces default
  kernel (143.1 us); B 142.3 us (-0.6%, noise). Kernel trace: B halves WGs (64x256), 16 KB/wave,
  BUT VGPR 72 -> 104 → occupancy 7 -> 4 waves/SIMD; in-flight est. 7x32 KB vs 4x64 KB per CU ≈
  equal. CONFOUNDED (pattern 2) → hypothesis not falsified. Isolating it needs kernel change:
  tile_n 256 at <= 72 VGPR, or fix (1) weight prefetch before the metadata chain.
- Existing `_persist` moe2 (C.csv): 144.0 us (no change); grid still 128x256 (persist_cu256 =
  256 WGs looping over M-blocks; only ~190 valid blocks at M=16 → no effect), VGPR 92+4.
- Design review (user asked, 2026-10-20). Layer budget EP1 M=16: 436 us vs ideal 356 us
  (2.54 GB / 7.13 TB/s): moe2 gap ~24, moe1 ~27, sort ~13, rest gaps. Structural causes: 3
  kernels; 1 tile per WG (serial metadata prologue each); worst-case grid; K too short to pipeline.
  Options: D1 moe2 prologue fix (<= ~20 us/layer); D2 decode persistent streaming kernel with exact
  (expert, n-tile) work list + cross-item software pipeline, no sort kernels (~50-80 us/layer);
  D3 fused persistent sort+moe1+moe2+AR = mega_moe_tp (exists). CORRECTION: earlier "mega_moe_tp
  11-14% slower" compared 1-GPU fused_moe + ideal AR (8-10 us) vs 8-GPU measured fused; serving
  EP1 is moe 474 + comm 64 = ~538 us/layer → unfair, redo in one 8-GPU harness with CUDA graph.
  Plan: (1) fair 8-GPU compare; (2) ATT mega_moe_tp at h16384; (3) choose D3 / D2 / D1.
- (1) DONE: `/data/proxy_moe/tools/t9_fair8.py` (imports the test copy for weights/routing/ref,
  CUDA graph of 24 back-to-back calls, slowest rank; `T9_GPC=4 torchrun --nproc_per_node=8
  t9_fair8.py --tokens 16 64`). NOTE: the test's `T.call` causes a rank-0-only extra collective
  (DETAIL debug: mismatch) → harness checks errors locally. Results (us/layer):
  M=16 base fused_moe+CustomAllreduce 494.1 (moe only 481.7, AR ~12) vs mega bf16 507.7 (+2.8%);
  M=64 base 1000.1 (moe 981.5, AR ~19) vs mega 1186.9 (+18.7%). rel_l2 mega 0.0031, base 0.0060.
  → corrected: mega is +2.8% (not 11-14%) at M=16, clearly worse at M=64. Real AR is only
  12-19 us/layer, so AR fusion cannot buy much at decode; serving's 64 us "comm" is skew wait.
  8-GPU fused_moe (481.7) is ~10% slower than 1-GPU (436) — all GPUs busy (unexplained, not
  measured: clocks/power?).
- (2) ATT of mega_moe_tp (M=16, rank 0 only via `torchrun --no-python t9_att_mega/wrap.sh`,
  `sys_trace: false` — with sys_trace rocprofv3 SIGSEGVs in rocpd SQLite output and no ui_output).
  Tool `/data/proxy_moe/tools/att_wave_timeline.py` (per-wave role/begin/end/first-last MFMA).
  Stall summary: 47% `s_sleep` in comm waves (communication.py:960 comm_wave, :791 poll_loop) —
  a consequence (comm waves wait for GEMM), not the cause. Timeline: per compute CU 4 compute
  waves (1/SIMD, VGPR 256) MFMA from ~4% to ~87% of kernel; last ~13% (~60-65 us of ~507) is
  comm-only tail (base AR is ~12 us). 14 sampled CUs (att_target_cu 0,1,2,5): 11 have compute
  waves, 3 have none (~21% idle) — contradicts my reading of `_dyn_units` (P=4, 760 units / 256
  CTAs); unexplained. Hypothesis: tail inflated by GPC=4 (NCK 16 coarse chunks → less GEMM2/comm
  overlap). If tail ~= base AR: ~456 us vs base 494 (-8%) — estimate only.
  Next: understand NCK <= 31 limit to test GPC=2 (NCK=32) = one-variable test of the tail.
- NCK <= 31 root cause (2026-10-20): per-chunk ready/claim bits live in one i32 LDS word
  (C_PRDY/C_PBITS/C_FRDY/C_FBITS, C_YAGM); `VMASK = (1 << NV) - 1` (NV = NCK) is used as an
  i32 constant (communication.py `_claim_bit`, `_comm_pending`) and `fin_owned` builds
  `i32(1 << c)` — both out of signed-i32 range at NV=32. Shifts/tests on bit 31 are fine in
  two's complement. Local fix (AITER tree, uncommitted; `/data/proxy_moe/patches/aiter_mega_moe_tp_nck32.patch`):
  `VMASK32` (-1 at NV=32) for i32 uses, bit-31 constant as -2^31, limits `<= 31` → `<= 32`
  (config.py:98,196). NV<32 unchanged. All proxy functional cases PASS at native GPC=2 (NCK=32).
  Upstream candidate (T10) — lets mega_moe_tp accept H=16384.
- GPC=2 vs GPC=4 (one variable, fair8): M=16 503.9 vs 507.7 us (-0.8%, noise); M=64 1188.9 vs
  1186.9. → tail hypothesis (coarse chunks) FALSIFIED. mega stays +1.8% (M=16) / +18.5% (M=64)
  vs base (495.1 / 1003.3). Tail cause unknown.

## D2 design (decode persistent streaming moe2, FlyDSL) — 2026-10-20

Target: EP1 decode (TP8 MoE): E=512 local experts, D_INTER=512/rank (K of moe2), H=16384 (N),
top-16, M <= 64 tokens. Baseline moe2 (EP1 M=16, 1 GPU): 143 us, 0.847 GB, 5.9 TB/s (83% of
7.13 achievable); ideal ~119 us at 7.13, goal <= 125 us (>= 95%).
Problems to remove (from ATT): per-WG serial metadata chain (2 HBM round trips, ~25% stall),
worst-case grid (26% empty waves), 2 K-iterations per WG (no intra-WG pipelining).
Design:
- Grid = n_CU x k (persistent). Work item = (expert block j, N-tile n); items enumerated from
  the existing sorted metadata (sorted_expert_ids / num_valid) — no empty items.
- Each WG walks a contiguous range of items; loop flattened over (item, k-tile) so the next
  item's metadata + B tiles are prefetched while the current item computes (multi-stage LDS or
  register buffers). Weight (B) loads depend only on (expert, n) → issued before token-id / A.
- Keep the production layouts (shuffle_weight(16,16) B, e8m0_shuffle scales, moe1 fp4 a2 +
  scales, sorted token ids) and the same epilogue (atomic bf16 add to out[token, n]) so it
  drops into fused_moe's stage 2 via kernelName2 / env flag.
- Reuse MXFP4 MFMA / scale / B-load helpers from `mixed_moe_gemm_2stage_common.py`
  (`compile_mixed_moe_gemm2_common`).
Steps: (1) study gemm2 body + FlyDSL authoring skill; (2) standalone kernel + op test vs
fused_moe stage2 output; (3) perf vs 143 us + FETCH_SIZE + ATT; (4) fused_moe hook + 8-GPU
fair8 + serving A/B; (5) extend to moe1 (+ drop sort kernels).
- Implementation lives in AITER `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py`
  (uncommitted; pristine copy `/data/proxy_moe/patches/mixed_moe_gemm_2stage_common.py.orig`),
  env `AITER_D2_STREAM_K=k` on the `_persist` stage-2 variant (C.csv) → grid (cu*k,1), items =
  m_blocks x n_tiles, contiguous range per WG, per-item bx/by in `_persist_setup` (by rebound).
- S1 (stream only) EP1 M=16 moe2: default 139.4 us; k=1 271.2, k=2 187.9, k=4 148.2 → worse,
  approaches baseline as WGs grow: items inside a WG are fully serialized (metadata RTT + B RTT +
  barrier per item), latency hidden only by co-resident WGs. Output vs default rel_l2 0.0065
  (stream vs stream 0.0021, default vs default 0.0021; no element diff > 0.5) → bf16 atomic
  order, not missing items; still to check against an fp32 reference before shipping.
- S2 (next-item metadata as loop state, `_stream2`): no change (k=1 281.8, k=2 187.0, k=4 148.1,
  k=7 164.5 us). Reason (verified in code): body's own sorted-token-id LDS prologue
  (`emit_tid_lds_prologue`, vmcnt(0)) runs before the first B load.
- D1 = `AITER_D2_DEFER_TID=1` (forces existing `r216_defer_tid`: B/scale loads before the tid
  prologue; A rows are sorted-order, token ids only used by the epilogue) on the DEFAULT kernel:
  139.3/139.0 → 137.7/137.4 us (-1.2%, 2 runs), rel_l2 within noise. Stream + defer: 150.3.
  → ATT's 25% prologue stall cycles were mostly overlapped by other waves (pattern 1).
- Ceilings for moe2's bytes (0.847 GB), measured (Triton, 1 GPU):
  contiguous persistent read 7.18 TB/s (size-independent 0.85-4.3 GB);
  moe2 decomposition as short WGs (24k WGs x 32 KB, `tools/moe2_pattern_read.py`) 5.6-6.0 TB/s
  (64/128 KB per WG: <= 6.02) — moe2 itself (6.15 with defer) is AT this ceiling;
  persistent streaming over the real scattered pattern (190 experts x 4.46 MB,
  `tools/moe2_pattern_stream.py`) **6.84 TB/s = 116.5 us** (BLOCK 1024, 512 programs, 4 warps).
  → the limit is "many short WGs", not compute; D2 headroom moe2 ~137 → ~117-120 us (-15%);
  same idea for moe1 (1.69 GB, 264 us at 6.41) → ~247 us. Total ~35-40 us/layer ≈ 0.9 ms/step
  (~5% EP1 TPOT), + sort kernels (13 us) if removed.
- S3a: FlyDSL persistent streaming read (`/data/proxy_moe/d2/s3a_stream_read.py`): default cache
  policy 5.8-6.0 TB/s; **cache_modifier=2 (streaming) → 6.6 TB/s (120 us)**; depth 1 vs 2 no diff.
- ROOT CAUSE FOUND: stage-2 compile `compile_mixed_moe_gemm2_common` does `del b_nt` (also before
  FHMoE commit 00cbe979f) → `_bnt2` in stage-2 names was a no-op; B loads used the default
  (L2-allocating) policy. stage-1 honors it (`cache_modifier=b_nt`), which is why moe1 is 6.4-6.6.
  FIX (AITER tree, uncommitted, `/data/proxy_moe/patches/aiter_moe2_bnt_and_d2_experiments.patch`):
  keep b_nt, `cache_modifier=b_nt` in stage-2 `load_cell`, name tag `_bnt{b_nt}`. Default decode
  tier name has `_bnt2` → streaming; no-suffix names (bnt 0) unchanged (prefill L2 reuse).
  EP1 1-GPU default path: M=16 moe2 139.1 → **120.4 us (7.03 TB/s)**, layer 431.8 → 403.3 (-6.6%);
  M=64 layer 979.6 → 962.9 (-1.7%); rel_l2 vs pre-fix 0.002 (noise). Upstream candidate (T10).
- S3 (deferred — fix above already reaches ~99% of contiguous ceiling for moe2 at M=16):
  make the FlyDSL moe2 a real stream: flattened (item, k-tile) loop with B tiles of
  the next 1-2 iterations in flight (prefer direct-to-LDS `buffer_load ... lds` like mega_moe_tp
  to avoid VGPR/occupancy loss), no per-item barrier on the B path.

## Item A — decode "2 x 1 ms all-reduce" (2026-10-20): profiler artifact, DROP

- EP1 decode TP0: `cross_device_reduce_2stage` x2/step = 2.09 ms (c231_ep1_t4_bnt2fix) vs 0.65 ms
  (c231_newbase_ep1_warm) — same kernel, 3x run-to-run spread → wait, not cost.
- `rank_skew.py` (both runs): MoE 54.8 ms and rest ~21 ms over 5 steps identical on all ranks;
  comm 3.5-20 ms; rank 1 lowest both times (~0.7 ms/step = real collective cost).
- `/data/proxy_moe/tools/launch_skew.py` (per-step step-start / graph-launch / first-kernel skew
  across ranks): step 1 up to 17 ms, step 2 ~3 ms (profiler start-up), steady-state steps 4/6
  <= 0.18 ms start skew, <= 0.47 ms first-kernel skew.
- Cross-check: min-comm rank GPU busy ~15.95 ms/step vs serving TPOT 15.83 ms → decode is
  GPU-bound in serving; no measurable skew/CPU loss. The "-9% TPOT" estimate was based on
  non-steady profile steps. Lesson: drop the first 2 profiled steps before skew attribution.

## Items B (moe1 streaming) and D (fused MX quant >8192) — scoping (2026-10-20)

- New-base EP1 decode, rank 1, per step: moe1 7.26 ms (303 us/layer), moe2 3.29 ms (137 us/layer).
  D-related kernels only: `dynamic_per_group_scaled_quant` 0.106 + `mxfp4_moe_sort` 0.156 =
  0.26 ms/step (the "quant" category ~1 ms/step is mostly attention fp4 GEMMs `_gemm_afp4wfp4`,
  misfiled by trace_kernels.py) → D cap ~0.15 ms/step (~1% TPOT), not 0.3-0.5.
- B: moe1 1-GPU 257-264 us (6.4-6.6 TB/s) vs moe2 7.03 TB/s on real weights → B cap ~20 us/layer
  1-GPU (~0.5 ms/step in serving, ~3% TPOT). moe1 already uses b_nt=2 + async.
- Serving moe1/moe2 are both ~15% slower than 1-GPU (303/260, 137/120) — hypothesis: more unique
  experts with the real router than random top-k (more bytes), unmeasured.
- HBM ceiling measurement is NOT settled: same Triton read kernel/config (2048x256 w4, 1.69 GB)
  = 7.0-7.27 TB/s in `hbm_read_bw.py` (sweep) but 6.0 TB/s in `hbm_read_fixed.py` (fixed,
  sustained). Not clocks/power (mclk 2000, sclk 2396, 1064 W under load), not allocation size,
  not time decay, all-8-GPU concurrency = alone (6.0). Unexplained; use real-kernel numbers
  (moe2 7.03 TB/s) as the reference until resolved.

### D implemented (2026-10-20, new base worktree, commit in `feat/proxy-model-1020`, not pushed yet)
- `csrc/kernels/quant_kernels.cu` FUSED_MX_QUANT_MOE_SORT_KERNEL_DISPATCH: new tier cols <= 64*BlockSize
  → THREAD_DATA 32, block 2*BlockSize (kernel has no LDS / block reduction); `aiter/ops/quant.py`
  threshold N <= 16384. Check `/data/proxy_moe/tools/d_fused_quant_check.py`: out bit-exact vs split
  path (M 1/16/64/256); scale equal on every tile of real rows; differences only in tiles with pad
  rows, and the same pre-existing difference exists at H=8192 (456/456 tiles) → pad rows are
  unspecified in both paths. M=16: fused 16.6 us vs split 36.2 us (M=1 16.1 vs 25.7).
- Gotcha: run_stage's AITER_LARGE_HIDDEN_WORKAROUND=1 patch script aborts on the new quant.py →
  `newbase_env.sh` sets it to 0; `repro.sh` now honors the env var.

### D result in serving (c231_newbase_ep1_D_1020): 933.6 / 344.8 / 15.85 (no change). Decode GPU:
fused `<512,32>` 0.261 ms/step vs split 0.156 + 0.106 = 0.262. The 1-GPU "36.2 → 16.6 us" was a
Python-loop eager timing (allocs + extra launch), not kernel time — measure kernels under CUDA
graph / profiler before claiming. The fused kernel quantizes every sorted (token, expert) row, so
each token is quantized 16x; a quantize-once + scatter-scales kernel could reach ~3-4 us/layer
(~0.17 ms/step, ~1%) — not pursued. Commit message corrected before push (89fed3572).

### B result (2026-10-20): stopped
moe1-only swaps (EP1 M=16, 1 GPU, profiler kernel time): w2 257.0 us (6.59 TB/s), tile_n 64
256.5, w3 255.0, w3_kb2 254.9, w3_kb4 256.5, 64_w3_kb2 262.7, 64 wpe1 320.6, bnt0 284.5 →
WG count x2/x4 does not move it → wave-quantization hypothesis FALSIFIED. ATT moe1
(`/data/proxy_moe/t9_att_moe1/`): stalls 87% of cycles, VMEM-load 67% (issue blocked, queue full),
VMEM-wait 16%, VGPR 152 → 3 waves/SIMD. Memory-side limit for this access pattern (gate + up
regions per WG) vs moe2 6.9-7.0 TB/s; gap ~10-15 us/layer, <2% TPOT.

### B hypothesis (static, 2026-10-20)
moe1 EP1 M=16: grid 4 x 256 (n-tiles x M-blocks), ~190 valid M-blocks → ~760 working WGs; LDS
54.8 KB + 176 VGPR → 2 WGs/CU → 512 slots → 1.48 waves (2nd wave half empty). Test by swapping
kernelName1 only (CSV `/data/proxy_moe/t9_tn/B_<variant>.csv`): tile_n 64, split-K kb2/kb4 (w3),
w3, bnt0.

## Results (append-only; node + date on every row)

| Date | Node | Run tag | Config | Out tok/s | TTFT med | TPOT med | Notes |
|------|------|---------|--------|-----------|----------|----------|-------|
| 2026-10-20 | crsuse2-m2m-231 | c231_ep8_cold_1020 | EP8 + AR fusion + tuned (cold JIT, discard) | 799.4 | 457.0 | 18.57 | first run in container |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep8_warm_1020 | EP8 + AR fusion + tuned | 807.0 | 314.3 | 18.64 | prefill TP0 82.46 ms/step (412/16k tok); comm 28.9% |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep1_cold_1020 | EP1 + AR fusion + tuned (first EP1 run) | 893.0 | 363.1 | 16.55 | |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep1_warm_1020 | EP1 + AR fusion + tuned | 893.4 | 364.8 | 16.53 | prefill TP0 101.79 ms/step (509/16k tok); comm 20.9% |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep8_dpattn_1020 | EP8 + AR fusion + tuned + DP-attn dp8 | 229.1 | 9839.1 | 24.41 | regression; see T3 results |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep8_t4bf16x3_1020 | EP8 + AR fusion + tuned + T4 router bf16x3 | 814.0 | 294.0 | 18.52 | prefill 74.58 ms/step |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep1_t4bf16x3_1020 | EP1 + AR fusion + tuned + T4 router bf16x3 | 901.6 | 342.9 | 16.47 | prefill 93.99 ms/step |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep1_t4_bnt2fix_1020 | EP1 + AR fusion + tuned + T4 + stage2 b_nt fix | 931.2 | 342.4 | 15.89 | decode MoE 10.97 ms/step (was 11.38) |
| 2026-10-20 | crsuse2-m2m-231 | c231_ep8_t4_bnt2fix_1020 | EP8 + AR fusion + tuned + T4 + stage2 b_nt fix | 833.8 | 296.8 | 18.06 | moe2 3.58 ms/step (was 3.84) |
| 2026-10-20 | crsuse2-m2m-231 | c231_newbase_ep1_cold_1020 | NEW BASE: EP1 + AR fusion + tuned + T4 + b_nt fix (cold) | 933.5 | 347.7 | 15.84 | first run, JIT |
| 2026-10-21 | crsuse2-m2m-231 | c231_tp4_g_new_ep1_arf_t4_tuned_1021 | TP4 EP1 + AR fusion + T4 + TP4-tuned rows | 1090.3 | 256.0 | 13.73 | new base |
| 2026-10-21 | crsuse2-m2m-231 | c231_tp4_h_new_ep4_arf_t4_tuned_1021 | TP4 EP4 + AR fusion + T4 + TP4-tuned rows | 1002.6 | 234.0 | 15.10 | new base |
| 2026-10-20 | crsuse2-m2m-231 | c231_newbase_ep1_warm_1020 | NEW BASE: EP1 + AR fusion + tuned + T4 + b_nt fix | 933.7 | 344.5 | 15.83 | aiter cfae83274, sglang effe332427, flydsl 0.4.0 |
| 2026-10-21 | crsuse2-m2m-231 | c231_tp8_base_ep8def_warm_1021 | TP8 EP8 package defaults, PRISTINE base (unoptimized) | 788.6 | 324.1 | 19.05 | cold 788.1 / 325.3 / 19.07 |
