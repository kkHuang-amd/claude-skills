# Kimi-K3.1 bring-up + perf plan (gfx950, 8x MI355X)

Owner node: mi355-4. Created 2026-10-30.

Model: `/shared_nfs/Kimi-K3.1` (1.5 TB, 96 shards). `KimiK31ForConditionalGeneration`, 93 layers,
hidden 7168; KDA linear-attn layers + 24 full-attn layers (every 4th + 93) running MLA with
DSA sparse top-2048 indexer; MoE 896 experts top-16 + 2 shared, routed hidden 3584, sigmoid
router, SiTU act; MXFP4 (group 32) on routed experts only — attn/shared/dense/lm_head bf16;
vision tower present. Draft: `/shared_nfs/Kimi-K3-DSpark` (DSPARK, 7 draft tokens).

Stack: sglang `kevin-mii/sglang-k31-amd@dev_amd` (14603f485), aiter `main` (f0b96f49e,
branch `k31/main-20261030` in worktree `/sgl-workspace/aiter-k31`), flydsl 0.4.1.

Gates (from DEV_AMD.md): precision floor = reference recipe (bf16 dense/attn, fp8_ds_mla DSA
cache, bf16 MoE partials, bf16 all-reduce, `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`).
Accuracy = gsm8k200 with the real draft + mean accept len from decode log (accept len ~1.0
with a passing score = broken verify). Speed = A/B vs dev_amd head, same flags, simulated
acceptance `SGLANG_SIMULATE_ACC_LEN=3.84 SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`.
AgentX compared at P90 interactivity. Never run GPU microbenches next to a live server.

## T0 — environment (done 2026-10-30, mi355-4)
- [x] clone sglang-k31-amd, `pip install -e python --no-deps`
- [x] aiter main worktree, gfx950-only editable install, fresh JIT dir (0 old .so); flydsl 0.4.1
- [x] `env.sh` overrides `/etc/bash.bashrc` PYTHONPATH that shadows aiter with the old tree
- [x] aiter symbols imported by dev_amd's AMD commits all resolve on main (31 checked)
- [x] aiter PR scan -> `AITER_PRS.md`
- [x] DSpark draft downloaded to `/shared_nfs/Kimi-K3-DSpark` (4.2 GB, from HF RadixArk/Kimi-K3-DSpark)

## T1 — bring-up, reference recipe
1. Launch cookbook DCP8 recipe (TP8/DCP8, DSPARK, HiCache, breakable prefill graphs) with
   local model paths; log to `/shared_nfs/kimi-k31/logs/`. Expect long first-run JIT.
2. If it fails: bisect to a minimal config (TP8, no spec, no HiCache, no BCG) and climb back.
3. Run branch kernel tests on 1 GPU (server down): `test_kda_verify_gfx950.py`,
   `test_sparse_mla_decode_opus.py`, `test_triton_sparse_mla_hip.py`, `test_attn_res_hip.py`,
   `test/registered/amd/test_k3_a8w4_moe_gfx950.py`.
Pass: server ready, chat request coherent, all kernel tests pass on aiter main.

## T2 — accuracy baseline
gsm8k200 (sgl-eval) with real draft; record score + accept len. Also a no-spec run to
separate target accuracy from verify bugs. Results -> `results/<tag>/`, table in CONTINUE_HERE.

## T3 — perf baseline
- Decode sweep c1/4/8/16/32/48 and prefill (ISL sweep) with simulated acceptance; TP8 for low
  c, DCP8 for high c (per DEV_AMD "where it stands").
- One trace per rank following `../kimi-k3/aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md`
  (late prefill -> short decode, <=500 MiB/rank) -> per-kernel breakdown by op class
  (KDA, DSA indexer, sparse MLA, MoE stage1/2, GEMM, all-reduce/all-gather, misc).
- Compare to dev_amd's quoted numbers (vs B300 at P90: c1 -0.5%, c4 -5.7..-14.4%, c8 -2.2%).

## T4 — aiter PR evaluation (one PR at a time, gated, revertible)
Priority A in `AITER_PRS.md`. For each: cherry-pick onto a branch off `k31/main-20261030`,
kernel test, then endpoint A/B. Start with zero-risk ones (GEMM tables 7169/6959 vs sglang's
own table; fixes 6965/7104/7172/7161), then KDA (6026, 7028, 7034, 6912), MoE sort (6484+6485),
all-reduce fusion (6270 vs 6960 — overlapping, pick one).

## T5 — optimization targets (refine after T3 trace)
- c4 gap: decode steps at 3-7 requests on standard TP8 path (no monokernel here).
- c8: prefill turns stalling decode (chunked prefill / BCG / scheduling).
- MoE 896-expert routing/sort/quant handoff; KDA launch boundaries; DSA indexer + top-k.
- dev_amd is 6 commits behind `origin/dev` — check whether those matter before rebasing.

## Open questions for the user
- `/etc/bash.bashrc` hardcodes old aiter on PYTHONPATH — leave it (current: env.sh override) or edit?
- Low-concurrency TP8 numbers upstream used an out-of-repo decode monokernel — in scope?
  Answer (user, 2026-10-31): yes, as `OPT_TRACKER.md` row 13 (mono kernel design, new session).
