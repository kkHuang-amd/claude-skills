Owner node: crsuse2-m2m-231

# Proxy MoE 24L TP8 — optimization continued on crsuse2-m2m-231

Continues `OPT_mi355-4.md` (read-only here; owned by mi355-4). Task IDs (T3..T10) keep that doc's
numbering.

## CONTINUE HERE

**Status (2026-10-21, after the fixed_seq_len sweep):**
- Best configs (new base = pushed branches): TP8 EP1 933.7 tok/s / TTFT 344.5 / TPOT 15.83 (TTT/GPU
  583.6, P90 interactivity 65.5; +18.4% / +23.1% vs pristine EP8 defaults); TP4 EP1 1090.3 / 256.0 /
  13.73; TP4 EP4 1002.6 / 234.0 / 15.10. Tables: "TTT/GPU + P90 interactivity", "T11".
- fixed_seq_len sweep (TP8 + TP4, 1k/1k, 8k/1k, 70k/300, 50 points) DONE: `fixed_seq_len/RESULTS.md`
  (+ chunk 32768 comparison, GB300 comparability notes, geomean MI355X/GB300 tp4ep1 = 0.923 / 0.885 /
  0.843, all 0.891, CONC<=512 0.874).
- Closed: T1, T2/T2b, T4, T8, T9 (b_nt fix), T11, items A (profiler artifact), B (moe1 memory-side,
  <2% left), D (fused quant 16384: correct, no serving gain). Dropped: DP-attention, mega_moe_tp.

**Next (priority order, evidence in brackets):**
1. **F / T3 — prefill comm/compute overlap: M0 + M1 DONE → side-stream overlap NO-GO** (section
   "F M0 results"): AR concurrent with MoE slows both (AR 1.7-1.8x, moe1 1.3-1.4x); best CU split hides
   12.5% of one AR (<3% prefill). Prefill comm is still ~22% (ISL 1024 and 70k). Remaining comm levers
   without overlap (not started, ask the user first): cheaper AR (QR INT4/INT6, needs a logits check),
   AR+norm fusion at prefill sizes. P1 code not started. QR level measured (op level, section
   "F follow-up"): INT6 -11% / INT4 -17% AR time at 4x / 16x error → ~2.5% / ~3.8% prefill; not adopted.
   **F2 prefill SP P1 DONE (code local, not pushed)**: FFN-only (attention linears are bf16).
   Serving vs off: 70000/300 c8 TTT/GPU +5.3% / TTFT -6.1% (hip quant for shared expert) or +3.3% /
   -3.7% numerically equal to today (`..._SP_EXACT_SHARED=1`); 8192/1024 c64 +2.1% / +1.2%. Decode
   unchanged. USER DECISION: which variant; whether to commit/push the SGLang branch (strip debug
   `_SP_CHECK` first). Details "F2 P1".
2. **CONC=1024 decode ITL jump** — measured (section "Item 2"): median jump = decode step 43.9 →
   69.6 ms (bs512 → bs1024, 1.59x for 2x tokens), not prefill interleaving (that is the tail). Decode
   kernel breakdown bs1024 vs bs512 in progress.
3. **G — server flags, cheap A/B**: chunk per ISL (70k chunk 32768 = TTT +1..9% but TTFT worse at
   CONC 2-4 → repeat CONC 2/4 with more requests before deciding), other scheduler knobs.
4. **Iso-comparison data vs GB300**: tp4ep1 sweep with `ROCM_QUICK_REDUCE_QUANTIZATION` unset (bf16
   prefill AR) to size the INT8 effect; optionally MI355X W4A8 routed experts.
5. **I — upstream AITER PRs** (b_nt fix first; then load_hip_runtime, NCK32, quant 16384, tuner
   chunked dequant; strip the default-off D2 env switches). Ask the user before opening PRs.
6. Parked (small): D quantize-once kernel (~1% TPOT), B moe1 W1 read layout (<2%).

**Rules:** one variable per experiment; measure kernels under CUDA graph / profiler (not eager
Python loops); `run_stage.sh` pkills every sglang server → sequential runs only; new base via
`source /data/proxy_moe/newbase_env.sh`; results under `/shared_nfs/kk/results/proxy_24l/<tag>/`;
docs edited and pushed from worktree `/data/proxy_moe/wt/skills` (main checkout has 2 unrelated
unmerged files); no AI attribution lines in commits; ask before pushing code branches.
**Repro (best TP8):** `NL=24 TPN=8 SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1 ONE="<tag> 1 1" bash /data/proxy_moe/repro.sh`
after `source /data/proxy_moe/newbase_env.sh` (TP4: `NL=12 TPN=4 GPUS=0,1,2,3`).

## History (previous CONTINUE HERE contents, kept verbatim)

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

## F / T3 — prefill AR/compute overlap: design + measurement plan (2026-10-21, crsuse2-m2m-231)

Status: design only, no code yet. Static numbers below are from the existing trace
`c231_newbase_ep1_warm_1020` (TP0 EXTEND, `trace_kernels.py --steps 5`, ISL 1024, chunk 8192).

**What there is to hide (EP1 TP8, prefill TP0, 93.84 ms/step GPU):** comm 20.40 ms (21.7%, QR INT8
twoshot x19, AR not fused with norm in prefill: separate rmsnorm / add_rmsnorm_quant x19); MoE
49.3 ms (52.5% = moe1 19.97 + moe2 `gemm2_a4w4..imax512` 18.40 + moe_reduction 9.51 + sort 1.43;
trace_kernels files moe2 under "gemm"); dense GEMMs ~12 ms; attention core 1.6 ms (1.7% at ISL 1024;
much larger at 70k context, so the comm share there is lower: unmeasured, see M1).
Ceiling = 21.7% of prefill GPU time if every AR were hidden for free. Hypothesis, not a bound.

**Why not SGLang TBO as-is:** TBO interleaves two micro-batches' ops on ONE stream and gets overlap
from DeepEP/mori a2a running on NIC/copy engines. Our TP all-reduce is a CU kernel (QR twoshot) on the
compute stream, so interleaving alone overlaps nothing. Overlap needs the AR on a second stream.
Reusable from TBO: `batch_overlap/two_batch_overlap.py` two-chunk split (`compute_split_seq_index`,
`compute_split_token_index`, one sequence may straddle the split), only if attention is split too.

**Design S (proposed first prototype): split only the per-token part of each layer.**
Attention core (`self.attn(q,k,v)` in `LagunaAttention.forward`) runs on the full chunk, so no
attention-metadata split. Everything else is per token and splits at token index t = N/2:
- refactor `LagunaAttention.forward` into `attn_pre` (qkv_proj, qk norm, rope), `attn_core`,
  `attn_post` (g_proj gate, o_proj); `LagunaDecoderLayer.forward` gets a split path; the loop in
  `LagunaModel.forward` stays.
- per layer: core(full) → o_proj(A) → [comm] AR1(A) → o_proj(B) → [comm] AR1(B) → add+norm+MoE(A)
  (waits AR1(A)) → [comm] AR2(A) → MoE(B) → [comm] AR2(B) → add+norm+attn_pre(L+1, A) →
  attn_pre(L+1, B) → core(L+1) waits AR2(B). Hidden: AR1(B) behind MoE(A), AR2(A) behind MoE(B)
  (fully); AR1(A) behind o_proj(B), AR2(B) behind attn_pre(A) (partially) → ~50-75% of comm.
- comm stream: one dedicated stream, events for the 4 dependencies; all ARs stay in order on that
  one stream (QR/custom-AR IPC buffers are not safe for two concurrent ARs).
- gate: env `SGLANG_OPT_LAGUNA_PREFILL_SPLIT=<min tokens>` (default off), extend mode only, eager
  only (prefill is not graphed), off for DP-attention/PP.
- risk 1: the layer_boundary residual state (`residual_batch.start(forward_batch)`,
  `attn_boundary.prepare/finish`, `layers/layer_boundary/residual/stream.py`) is one object per
  batch; the split needs two (slice views of hidden/residual per half). Read that code first.
- risk 2: QR twoshot uses CUs; concurrent with MoE both kernels may stretch (contention). Measured
  outcome, not assumed.
- risk 3 (largest): MoE at M/2 twice is slower than M once (weight read per pass is fixed: EP1 TP8
  ~6.4 GB/rank/layer of fp4 experts, estimate). TP4 EP1 measured (T11 validate.sh): 2 x M4096 =
  9526 us vs M8192 7858 us = +21%. Applied to the 52.5% MoE share that is ~+11% prefill, i.e. it can
  cancel the 11-16% expected hiding at chunk 8192. Mitigation: split only when each half keeps
  M >= 8192 (chunk 16384), compared against chunk 16384 unsplit (one variable).
- Design T (later, only if S works): full TBO-style split including attention (two-chunk split +
  per-child attention metadata), hides AR2(B)/AR1(A) behind the other half's attention core too;
  worth it mainly at 70k where the core is large.

**Measurement plan (sequential, one variable each; record every run in Results):**
- M0 (1 GPU, no code): MoE op time, TP8-EP1 shape (inter 512, 512 experts, top-16) at M = 2048 /
  4096 / 8192 / 16384, profiler kernel time (`scripts/ep_moe_check.py` or `tune_tp4/validate.sh`
  adapted), plus QR INT8 AR time on 8 GPUs for [M,16384] bf16 at the same M (fixed latency vs
  linear). Gives the split penalty and the per-half AR cost.
- M1 (serving, no code): 70000/300 CONC 8 profile, TP0 EXTEND steady chunk: comm / MoE / attention
  shares at long context (`trace_kernels.py`).
- Go/no-go: projected = hidden comm (from the schedule above, using M0 AR times) - MoE penalty (M0)
  >= 5% of prefill GPU time at chunk 8192 or 16384. Otherwise stop and record why.
- P1 (code, ask first): design S behind the env gate. Correctness: greedy outputs identical
  split vs unsplit on a few prompts (AR order identical per token, so expect bit-exact) + GSM8K-style
  sanity not needed for a dummy ckpt; compare logits rel_l2.
- Overlap verification in the trace (outcome, not the enabling flag): (a) AR kernels appear on a
  second stream/tid; (b) exposed comm = time of AR intervals not covered by any compute kernel on
  another stream (new tool `tools/overlap_exposed.py`), target < 50% of the unsplit comm ms;
  (c) AR and MoE kernel durations vs unsplit (contention inflation); (d) step GPU ms (TP0, min-comm
  rank per `rank_skew.py`) and finally serving TTFT/TTT at 70000/300 CONC 8 and 8192/1024 CONC 256.

### F M0 results (2026-10-21, crsuse2-m2m-231) — side-stream overlap: NO-GO
Artifacts `/shared_nfs/kk/results/proxy_24l/c231_F_M0_1021/`; tools `/data/proxy_moe/tools/`
(`decode_moe_roofline.py` T9_EP1=1, `ar_m_scaling.py`, `ar_moe_overlap.py`). Random top-16 routing,
TP8-EP1 MoE shape, profiler kernel time / event wall time.

| M | MoE 1 GPU (us) | AR [M,16384] bf16, 8 GPU (us, path) |
|---|---|---|
| 2048 | 2147 | 290 (custom AR 2stage) |
| 4096 | 3014 | 569 (QR INT8 twoshot) |
| 8192 | 5128 (matches serving trace per layer: moe1 2.2 / moe2 2.0 / reduce 1.06 ms) | 1031 (serving trace 1.07) |
| 16384 | 10599 (no tuned row: default `moe2_..cshuffle` 6.9 ms) | 1939 |

- Split penalty: 2 x M4096 = 6028 vs M8192 5128 (+17.5%); 2 x M8192 = 10256 vs M16384 10599 (-3%,
  only because M16384 is untuned). AR is linear in bytes from 4096 up (no per-call overhead to lose).
- Direct overlap test (8 GPUs, MoE on main stream + AR on a side stream, launched together; trace
  confirms kernels overlap in time: 1.0 ms at M4096, 1.76 ms at M8192):

| M | AR CUs | MoE alone | AR alone | both | vs serial unmasked (moe+ar) | AR kernel in "both" | moe1 in "both" |
|---|---|---|---|---|---|---|---|
| 4096 | shared | 2951 | 556 | 3738 | +231 us (worse) | 1015 (1.8x) | 1798 (1.38x) |
| 8192 | shared | 5173 | 1037 | 6620 | +410 us (worse) | 1773 (1.7x) | 2737 (1.28x) |
| 4096 | 32 own / MoE 224 | 3079 | 569 | 3545 | +38 us (worse) | 2295 | 1612 |
| 8192 | 32 own / MoE 224 | 5339 | 1044 | 6080 | -130 us (12.5% of AR hidden) | 3659 | 2686 |
| 4096 | 64 own / MoE 192 | 3279 | 612 | 3715 | +208 us (worse) | 2421 | 1474 |
| 8192 | 64 own / MoE 192 | 5669 | 1160 | 6492 | +282 us (worse) | 2828 | 2937 |

  CU partition = `hipExtStreamCreateWithCUMask` (GOTCHA: load torch's own
  `_rocm_sdk_core/lib/libamdhip64.so.7` via ctypes; the system libamdhip64 is a different runtime →
  "invalid argument" on any op in that stream).
- Verdict: running QR AR concurrently with the MoE makes both slower; best case (AR on 32 private
  CUs, M8192) hides 12.5% of one AR ≈ 0.13 ms/layer, < 3% prefill before the split penalty. Design S
  (and TBO-style split with a side stream) is NO-GO with these kernels. Why the contention costs this
  much is NOT measured (hypotheses: QR twoshot is CU/issue-throughput bound — it stretches 3.5x on 32
  CUs; both kernels compete for HBM/fabric; persistent `cu256` MoE kernels lose a wave slot). Falsifier
  if revisited: AR alone on 32 masked CUs vs AR concurrent on the same 32 CUs (CU vs memory contention).
- Prefill comm (21.7%) remains; levers that do not need overlap: cheaper AR (QR INT4/INT6, accuracy
  check needed), fewer AR bytes, AR+norm fusion at prefill sizes. Not started.

### F M1 results (2026-10-21, crsuse2-m2m-231) — 70000/300 CONC 8 prefill shares
Driver `/data/proxy_moe/F_M1.sh` (sweep launch flags via `serve_proxy.sh`, TP8 EP1, chunk 8192).
- M1 (`c231_F_M1_70k_c8_1021`): profile started mid-run → only DECODE captured (by-stage profiling
  ends the session after the decode capture if decode comes first; `profile_stages` is unused in v1).
  Serving numbers reproduce the sweep point: TTT/GPU 3,566.9 / TTFT med 8,749 / ITL med 10.75
  (sweep 3,573.8 / 8,745 / 10.75).
- M1b (`c231_F_M1b_70k_c8_1021`): 2-request warm-up, then `start_profile` (by-stage, 9 steps =
  one 70k request, context 0→70k) before the bench → EXTEND captured. TP0 EXTEND shares:
  comm 22.2% (QR twoshot 20.5% + reduce_scatter 1.6%), MoE 48.8% (moe1 / moe2 / reduce / sort at
  2.04 / 1.96 / 0.99 / 0.15 ms per layer = the M8192 numbers of M0), dense GEMM ~12%, attention 6.1%
  (fmha full layers + unified_attention sliding layers), norm 3.9%, quant 3.9%. Serving 3,640.2 /
  8,692 / 10.76 (warmup 0 instead of 16, profiler on; not a sweep-comparable point).
- So the long-context prefill looks like ISL 1024 (comm 21.7%, MoE 52.5%): attention is only 6% at
  70k because 3 of 4 layers are sliding-window. Prefill comm ≈ 22% everywhere; the M0 verdict
  (overlap does not pay with these kernels) applies to 70k too.
- Trace caveat: kernel counts imply ~7 of the 9 steps were recorded (moe1 171 calls / 24 layers);
  shares and per-call times are unaffected, absolute ms/step from `--steps 9` are too low.

### F follow-up: cheaper prefill AR via QR quantization level (2026-10-21, crsuse2-m2m-231, op level)
`ar_m_scaling.py` (now also prints rel_l2 vs the exact fp32 sum of all ranks' inputs; inputs randn
per rank, seeded), 8 GPUs, [M,16384] bf16 (QR casts to fp16), one variable =
`ROCM_QUICK_REDUCE_QUANTIZATION`. Artifacts `/shared_nfs/kk/results/proxy_24l/c231_qr_quant_op_1021/`.

| Level | M4096 us | M8192 us | vs INT8 (M8192) | rel_l2 per AR |
|---|---|---|---|---|
| FP (no quant, ~GB300 bf16 AR) | 664 | 1301 | +25.9% | 1.7e-3 |
| INT8 (current) | 566 | 1034 | — | 7.7e-3 |
| INT6 | 484 | 918 | -11.2% | 3.0e-2 (3.9x INT8) |
| INT4 | 397 | 857 | -17.1% | 1.2e-1 (16x INT8) |

- Projected prefill gain (comm = 22% of prefill, all of it QR at chunk 8192): INT6 ~2.5%, INT4 ~3.8%,
  bought with 4x / 16x the per-AR error (48 ARs per forward, randn data; real activations with
  outliers can only be worse). INT4 at 12% error per AR is not a usable trade; INT6 is a small gain
  at a large error increase. Not run in serving; recommend not adopting (user decision).
- Bytes vs time: INT4 moves 1/4 of FP's bytes but is only 34% faster → the quantized modes are not
  link-bandwidth bound. Estimate only: twoshot moves ~1.75x payload per GPU → FP ~360 GB/s, INT8
  ~230, INT4 ~140 GB/s per GPU; xGMI peak for this node NOT measured, so no utilization claim.
  Consistent with M0 (QR stretches 3.5x on 32 CUs): QR time is dominated by on-GPU work
  (codec/reduction), not the links. Hypothesis; falsifier = QR time vs CU count at fixed level.
- Iso-comparison data point (Next item 4): bf16-like (FP) prefill AR would be +26% AR time ≈ +5.7%
  prefill GPU time vs the INT8 we run; size of the INT8 advantage in the GB300 comparison.
- AR+norm fusion at prefill sizes: norm + residual add ≈ 5% of prefill; the aiter fused AR path is
  custom-AR based (2stage, used only up to M2048 here), which would give up QR INT8 at M8192 (custom
  AR 2stage vs QR at M8192 not measured). Upper bound ~5% minus that loss; not started.

### F2 idea (2026-10-21): replace prefill AR by reduce-scatter + FP4 all-gather (sequence-parallel residual)
[CORRECTION 2026-10-21] quark `exclude` = `re:.*self_attn.*`, `mlp.gate$`, lm_head, embed_tokens:
ATTENTION LINEARS ARE BF16 (qkv_proj is UnquantizedLinearMethod; the first P1 run crashed on it).
Only the FFN-side AR feeds fp4 consumers (shared expert + routed MoE). The text below originally
assumed both ARs did; the P1 section has the corrected (FFN-only) design and estimate.
Original text: every linear except router/lm_head/embed is MXFP4 W4A4 → the consumers of both ARs
(next qkv_proj; shared expert + routed MoE) quantize their input to fp4 per 1x32 group per TOKEN. AR = RS + AG; the AG half can move the already-quantized fp4.
- Per layer (prefill only; decode keeps AR): o_proj partial → RS [M,H] → add + RMSNorm + MXFP4 quant
  on M/8 tokens → router on M/8 tokens (router weight replicated) + AG of topk ids/weights (tiny) →
  AG fp4 + e8m0 scales (0.53 B/elt vs 2 B bf16 / ~1 B QR INT8) → shared + routed MoE on M tokens with
  pre-quantized input (the existing split path `dynamic_per_group_scaled_quant` + `mxfp4_moe_sort`
  already takes fp4 tokens) → partial → RS → add + norm + quant on M/8 → AG fp4 → next qkv_proj.
- Equivalence: quantization is token-local, so quantizing the shard then gathering = gathering bf16
  then quantizing (bit-identical). With RS in bf16 the result is MORE exact than today's INT8 AR;
  with an INT8 RS it keeps half of today's error.
- Side wins: norm / residual add / quant / router run on M/8 tokens (~4-5% of prefill today); AG is a
  pure byte copy (no codec) → candidate for SDMA/copy engines, i.e. no CU contention (the M0 failure).
- Existing SGLang `--enable-attn-tp-input-scattered` is NOT usable: MLA-only (needs q_lora_rank,
  gathers the qkv latent) and gated to CUDA/NPU. GQA + fp4 gather needs new Laguna code.
- Estimate (hypothesis, from bytes, and QR was shown NOT byte-bound → must be measured): per layer
  today 2 x 1.03 ms; new = 2 x RS + 2 x AG(fp4). If RS ~0.5-0.65 ms and AG(fp4) ~0.15-0.25 ms →
  ~1.3-1.8 ms → comm -15..-37% (≈3-8% prefill) + ~4% from M/8 norm/quant/router.
- M2 (op level, 8 GPUs, no model code): at M=8192, H=16384: (a) RS bf16 via RCCL / aiter custom RS
  (`reduce_scatter_cross_device_store`) ; (b) AG of [8192, 8704 B] via RCCL / custom / SDMA peer copy;
  (c) AG on a side stream concurrent with MoE (M0-style). Go if projected >= 5% prefill.

### F2 M2 results (2026-10-21, crsuse2-m2m-231, op level, 8 GPUs) — GO (projected ~9% prefill)
`/data/proxy_moe/tools/rs_ag_bench.py`; artifacts `/shared_nfs/kk/results/proxy_24l/c231_F2_M2_1021/`
(`rs_ag_cap512_overlap.log`; aiter custom RS/AR need `AITER_CUSTOM_AR_MAX_SIZE=536870912`, default
cap 64 MiB). Event wall per call, rank 0. AG payload = [M, 8704 B] (fp4 + e8m0 scales).

| Op | M4096 us | M8192 us | note |
|---|---|---|---|
| AR QR INT8 (today, from M0) | 566 | 1034 | lossy, rel_l2 7.7e-3 |
| AR QR FP | 664 | 1301 | |
| AR aiter custom 2stage bf16 | 599 | 1197 | slower than QR INT8 → AR+norm fusion lever is weak |
| RS RCCL bf16 | 345 | **653** | exact sum |
| RS aiter custom bf16 | 440 | 861 | incl. copy-in memcpy |
| AG RCCL fp4 bytes | 112 | 194 | |
| AG aiter custom fp4 bytes | 85 | **158** | incl. copy-in memcpy; peer-copy check bit-exact |
| AG peer `copy_` x8 (IPC, DtoD memcpy) | 1452 | 1976 | slow; dropped |

- RS + AG(fp4) = 653 + 158 = **811 us vs 1034 us INT8 AR (-21.6%)** at M8192 while being exact (bf16
  RS); -0.45 ms/layer ≈ **4.5% of prefill GPU time** (layer ≈ 9.5 ms from M1b: 1.05 ms/AR x2 = 22.2%).
- Plus norm / residual add / attention-input quant / router on M/8 tokens: those are ~6% of prefill
  today (rmsnorm 3.9, add 1.2, quant ~1, router 0.3) → ~-5%. Total projection ~9% prefill GPU time
  (hypothesis until P1 measures it). Cost not yet measured: routed MoE must take pre-quantized fp4
  (today moe1 fuses the quant: `..._fp4q_sort_..`), i.e. the split quant + `mxfp4_moe_sort` path at
  prefill M; and AG of topk (tiny).
- AG on a side stream concurrent with the MoE hides 74% (aiter) / 41% (RCCL) at M8192 — unlike QR
  AR (M0, worse than serial). Not needed for the base design (AG is on the critical path); keeps a
  later chunk-pipelined variant open.
- P1 scope (code, ask the user first): Laguna prefill path behind an env gate: RS after o_proj and
  after MoE; sharded residual + add/norm/quant (+ router + topk AG) on M/8; AG fp4+scales; qkv_proj /
  shared expert / routed MoE consume pre-quantized input; decode and M % 8 != 0 keep AR. Correctness:
  greedy tokens and logits vs exact-AR (QR FP) reference.

### F2 P1 — implementation + correctness (2026-10-21, crsuse2-m2m-231; code local, NOT pushed)
SGLang worktree `/data/proxy_moe/wt/sglang` (`feat/proxy-model-1020`, uncommitted): `environ.py`
(`SGLANG_OPT_LAGUNA_PREFILL_SP`, default off; `..._SP_MIN_TOKENS` 4096), `moe/moe_runner/aiter.py`
(standard dispatch: fp4x2 `hidden_states` + `hidden_states_scale` → `a1_scale`, bf16 output),
`models/laguna.py` (`_sp_gather_mxfp4`, `LagunaMoE.forward_sp`, `LagunaModel._sp_eligible/_sp_forward`).
- Design (FFN-only, attention is bf16): per layer o_proj partial → RS bf16 (exact) → post-attn
  add+norm on the M/8 shard → router + top-k on the shard → AG top-k (int32 pack) → hip MXFP4 quant of
  the shard → AG fp4 + e8m0 → shared expert `(q, s)` + routed MoE prequant → partial; the attention
  shard is added into the partial's own rows → ONE AR (QR INT8) = attn + ffn for all tokens →
  next input_layernorm(summed, residual) on full tokens (residual stays replicated, as today).
  Eligible: extend, tokens >= 4096 and % tp == 0, EP1, no PP/DP-attn/aux capture; else old path.
- Comm per layer: RS 653 + AG 158 + AR 1034 = 1845 us vs 2 x 1034 = 2068 (-223 us ≈ -2.3% prefill)
  + post-attn norm / router / shared-expert quant on M/8 (~3%) → projected ~5% prefill (was ~9%
  under the wrong all-fp4 assumption).
- aiter `fused_moe` with hip-prequantized fp4 input + `a1_scale` (needs `dtype=bf16`) is BIT-EXACT vs
  bf16 input at M8192 and 62 us faster (`/data/proxy_moe/f2/moe_prequant_check.py`).
- Correctness (`/data/proxy_moe/f2/p1_check.sh`, `p1_compare.py`, results
  `/shared_nfs/kk/results/proxy_24l/c231_F2_P1_check_1021/`): end-to-end logprob / greedy comparisons
  are useless on the dummy ckpt (QR FP vs INT8, both correct: mean |Δ input logprob| 0.30, greedy
  0/32 match); fallback path (4100 tokens) identical to today (Δ 0). Per-layer self-check
  (temporary `SGLANG_LAGUNA_SP_CHECK=1`, reference = same inputs on the old path with exact RS+AG):
  SP vs same-math-no-SP rel_l2 8.5-8.7e-3 every layer (= INT8 AR error) → plumbing correct.
  SP vs today 0.10-0.13: entirely the shared expert's activation quantizer (today triton
  `dynamic_mxfp4_quant`, SP hip `per_1x32` so routed stays bit-exact); hip vs triton quant are not
  bit-equal (each ~11.5% rel error vs bf16 x). NOT equivalent to today for the shared expert →
  options: triton quant + 2nd AG (+158 us, eats most of the gain) or one quantizer for both. Open.
- Serving A/B (2026-10-21 13:47-14:10, crsuse2-m2m-231, one variable `SGLANG_OPT_LAGUNA_PREFILL_SP=1`,
  sweep launch flags + client via `sweep_proxy.sh`, tp8ep1, one run per point; driver
  `/data/proxy_moe/f2/p1_ab.sh`, results `/shared_nfs/kk/results/proxy_24l/c231_F2_P1_ab_1021/{off,on}/`).
  SP path verified active: "Laguna prefill SP active (first batch: 8192 tokens)" on all 8 ranks.

| Point | TTT/GPU off → on | TTFT med (ms) | ITL med (ms) | P90 interactivity |
|---|---|---|---|---|
| 70000/300 c8 | 3,574.6 → 3,765.2 (+5.3%) | 8,742 → 8,210 (-6.1%) | 10.75 → 10.75 | 92.6 → 92.5 |
| 8192/1024 c64 | 1,624.8 → 1,658.9 (+2.1%) | 7,828 → 7,347 (-6.1%) | 29.40 → 29.43 | 33.8 → 33.8 |

  Off reproduces the sweep point (70k c8 3,573.8). Gain matches the ~5% prefill projection;
  decode unchanged (SP is prefill-only). Shared expert here uses the hip quantizer (see above).
- Exact-shared variant `SGLANG_OPT_LAGUNA_PREFILL_SP_EXACT_SHARED=1` (default off): the shard is
  also quantized with the triton quantizer of the dense W4A4 linears and gathered a 2nd time for the
  shared expert. Self-check: SP vs today rel_l2 8.5-8.7e-3 every layer = the INT8 AR error alone →
  numerically equivalent to today (attention AR now exact; routed and shared bit-exact inputs).
  Serving (same day/node/flags, vs the off rows above): 70000/300 c8 TTT/GPU 3,691.2 (+3.3%), TTFT
  8,415 (-3.7%), ITL 10.74; 8192/1024 c64 1,644.9 (+1.2%), TTFT 7,539 (-3.7%), ITL 29.43. The 2nd AG
  costs ~2 points. Decision for the user: hip quantizer for the shared expert (+5.3%, shared-expert
  activations quantized by a different but equally valid MXFP4 rounding) vs exact (+3.3%).
- TODO before any push: remove `_SP_CHECK` / `_sp_reference` debug code and the one-time info log.

### Item 2 — CONC=1024 decode ITL jump: measured (2026-10-21, crsuse2-m2m-231, offline from the sweep)
`/data/proxy_moe/f2/itl_shape.py` on `fixed_seq_len/tp8ep1/isl1024-osl1024-c{512,1024}` (client
`--output-details` per-token ITLs + TP0 server decode/prefill log lines).

| CONC | ITL p10 / p50 / p75 / p90 / p99 (ms) | steps > 2x median: % steps / % time | server decode-only ms/step (running >= 98%) |
|---|---|---|---|
| 512 | 40.7 / 43.9 / 46.1 / 51.9 / 119.5 | 1.6% / 17.1% | 43.9 (197 intervals) |
| 1024 | 58.7 / 74.4 / 96.0 / 113.5 / 319.5 | 3.9% / 26.0% | 69.6 (197 intervals) |

- Decode-only intervals (no Prefill batch line between two decode log lines): bs512 43.9 ms/step,
  bs1024 69.6 ms/step (1.59x for 2x tokens; per token 0.086 → 0.068 ms). So the MEDIAN ITL jump is the
  decode step itself, not prefill interleaving (hypothesis falsified for the median). Prefill
  interleaving drives the tail (p75+; 26% vs 17% of time in > 2x-median steps; intervals that contain
  prefill: 464 ms vs 1,311 ms median).
- Decode profile at bs512/bs1024 NOT obtained: torch profiler SEGFAULTs at "Stop profiling" (both
  plain and by-stage) with `--cuda-graph-max-bs-decode 512/1024` and kills the server
  (`c231_item2_decode_c{512,1024}_1021/server.log`); bs8/bs16 decode profiles worked earlier. Not
  root-caused. Used op-level timing instead (`c231_item2_ops_1021/`, 1 GPU MoE EP1/TP8 random routing,
  8-GPU AR):

| Op per layer | M=512 | M=1024 | Δ x 24 layers |
|---|---|---|---|
| MoE (default kernels, t32 tiles) | 1,220 us (5.6 TB/s on 6.85 GB unique) | 1,789 us (3.8 TB/s) | +13.7 ms |
| AR x 2 (custom 2stage) | 2 x 104.5 | 2 x 174.9 | +3.4 ms |
| measured step (server) | 43.9 ms | 69.6 ms | +25.7 ms (remaining ~8.7 ms attention / dense GEMM / norm, not split) |

- MoE M=1024: same unique weight bytes as M=512 but 1.47x time; EP1/TP8 tuned rows exist only for
  M 2048/4096/8192, so M<=1024 uses default t32 tiles (avg 32 rows/expert at M1024 → many experts
  need 2 M-blocks → weights re-read; hypothesis). Probe, one variable (`AITER_CONFIG_FMOE` = tuned
  CSV + a token=1024 copy of the M2048 row, `/data/proxy_moe/f2/item2/with1024.csv`): M1024
  1,807 → 1,430 us (-21%, moe1 t64x128x256 at 5.97 TB/s); M512 with the same row 1,232 → 1,348 (worse,
  keep default). Output rel_l2 vs default 3.7% (same systematic fp4 requant-path difference as the
  accepted T11 tuned rows, 3.6%). Projected decode step bs1024: -9 ms (~-13%).

**Item 2 plan (CONC=1024 decode ITL jump), measurement only, runs after F's M0/M1:** one run
1024/1024 CONC 1024 (TP8 EP1) with the server's decode-batch log lines kept; compare (a) server
decode step time at running-req ~1024 (log `gen throughput` / per-step timing) vs client median ITL,
(b) fraction of steps that are mixed/prefill chunks during steady state, (c) ITL distribution shape
(bimodal = prefill interleaving; shifted = slower decode step, e.g. graph bs or attention at 1024).
Next variable picked from that result only.

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
| 2026-10-21 | crsuse2-m2m-231 | c231_F_M0_1021 | op-level: MoE M-scaling (1 GPU), QR AR M-scaling + AR/MoE concurrency (8 GPU) | — | — | — | side-stream overlap NO-GO, see F M0 results |
| 2026-10-21 | crsuse2-m2m-231 | c231_F_M1_70k_c8_1021 | TP8 EP1 sweep flags, 70000/300 CONC 8, profile mid-run | TTT/GPU 3566.9 | 8749 | ITL 10.75 | only DECODE trace captured |
| 2026-10-21 | crsuse2-m2m-231 | c231_F_M1b_70k_c8_1021 | same, warm-up + profile from bench start | TTT/GPU 3640.2 | 8692 | ITL 10.76 | EXTEND comm 22.2%, MoE 48.8%, attn 6.1% |
| 2026-10-21 | crsuse2-m2m-231 | c231_qr_quant_op_1021 | op-level QR AR FP/INT8/INT6/INT4, [8192,16384] | — | — | — | 1301/1034/918/857 us; rel_l2 1.7e-3/7.7e-3/3.0e-2/1.2e-1 |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_M2_1021 | op-level RS bf16 + AG fp4 vs AR, [8192,16384] | — | — | — | RS 653 + AG 158 = 811 us vs QR INT8 AR 1034 us |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/off | 70000/300 c8, tp8ep1 sweep flags | TTT/GPU 3574.6 | 8742 | ITL 10.75 | baseline |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/on | same + SGLANG_OPT_LAGUNA_PREFILL_SP=1 | TTT/GPU 3765.2 | 8210 | ITL 10.75 | +5.3% / -6.1% |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/off | 8192/1024 c64, tp8ep1 | TTT/GPU 1624.8 | 7828 | ITL 29.40 | baseline |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/on | same + SP | TTT/GPU 1658.9 | 7347 | ITL 29.43 | +2.1% / -6.1% |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/onx | 70000/300 c8 + SP + EXACT_SHARED | TTT/GPU 3691.2 | 8415 | ITL 10.74 | +3.3% / -3.7%, numerically = today |
| 2026-10-21 | crsuse2-m2m-231 | c231_F2_P1_ab_1021/onx | 8192/1024 c64 + SP + EXACT_SHARED | TTT/GPU 1644.9 | 7539 | ITL 29.43 | +1.2% / -3.7% |
