# gfx1250 MXFP4 a4w4->a8w4 — Experiment Log (2026-07-07)

> MULTI-NODE: several machines share this file. **Tag each new entry with its node**, e.g.
> `[node: H21-18 gfx1250]` / `[node: <host> gfx950]`. APPEND, don't rewrite. See STATUS.md
> "MULTI-NODE CONVENTION". (E0-E25 predate this and are the H21-17/H21-18 gfx1250 + gfx950
> cross-val work; E26/E27/E28/E30 were the gfx950 box; E31 = H21-18 gfx1250.)

Chronological record of what was run and observed. Repos (base commits at time
of work):
- `/sgl-workspace/sglang` @ `8d30387cd671a3bc8eae178988f7c119544d08b7` (2026-07-02)
- `/sgl-workspace/aiter`  @ `8815f4b56dbaf416a3370659b777839c70ff3bf9` (2026-06-30, fork akao-amd/aiter)
- Model: `/dockerx/models/amd/DeepSeek-R1-0528-MXFP4` (Quark W4A4 MXFP4)
- HW: 4x gfx1250, VRAM ~432 GB each. Shared with another user's `rccl-tests`.

> **MULTI-NODE CONVENTION (2026-07-09):** several machines are now validating this
> accuracy question in parallel (gfx1250 boxes + gfx950 reference). To avoid confusion
> when more than one node appends here, **every new experiment entry MUST start with a
> `node:` line** identifying the machine, arch, and repo commits, e.g.
> `node: gfx1250 / host ctheliosr-rck-g02-j19-10 | sglang 000a61a2 | aiter 8815f4b5`.
> Known nodes so far:
> - `gfx1250 / host ctheliosr-rck-g02-j19-10` — 4x gfx1250 (this log's primary node).
> - `gfx950` — 8x MI355X reference node (a4w4; E26/E27 dumps + ablations).
>
> If two nodes might edit simultaneously, prefer appending to your own node-suffixed
> section or coordinate; the `node:` tag makes every entry attributable regardless.

---

## E0. Baseline run of run_ds-r1.sh (unmodified) — hang, then a4w4 root cause

- `bash run_ds-r1.sh` (HIP_VISIBLE_DEVICES=0). Loaded aiter, built module_aiter_core,
  got to `Init torch distributed begin` and **hung ~20 min**.
- `py-spy dump` on the scheduler: stuck at the **first GPU op**
  `torch.ones(..., device='cuda')` in `distributed/parallel_state.py:302`
  (`init_world_group`), spinning ~104% CPU. Not weights, not a4w4 kernel yet.
- `dmesg`: earlier fatal GPU page fault on GPU0 from a **prior** dead process
  (pid 1567070): `no-retry page fault ... Faulty UTCL2 client ID: SQC (inst)`,
  `PERMISSION_FAULTS: 0x3`. => GPU0 was **wedged** by a prior a4w4 crash; the new
  process hung on HIP context init. Also stray `rccl-tests` (another user) on GPUs 0-3.
- Killed the hung process.

Takeaway: `SQC (inst)` instruction-cache fault == executing the unimplemented
fp4 scaled-WMMA. A crash wedges the GPU for subsequent processes (transient).

## E1. GPU health + VRAM probe

```bash
rocm-smi --showmeminfo vram   # ~432 GB total each, ~170 MB used
for g in 1 2 3; do timeout 25 python3 -c "import torch; x=torch.ones(8,device='cuda:$g'); torch.cuda.synchronize('cuda:$g'); print('OK')"; done
```
- GPU0 `torch.ones` **hung 20s** (wedged); GPU1/2/3 **OK**. Chose GPU1 (later GPU3).

## E2. Model quant config confirmed W4A4

```bash
python3 -c "import json;c=json.load(open('/dockerx/models/amd/DeepSeek-R1-0528-MXFP4/config.json'));print(c['quantization_config']['global_quant_config'])"
```
- weight: fp4, per_group, group_size 32, static, e8m0.
- input_tensors: fp4, per_group, 32, **dynamic**, e8m0.  => W4A4 (a4w4).
- `exclude`: all `model.layers.*.self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj`,
  `lm_head`, `re:model.layers.61.*`. => attention is bf16; fp4 = MoE experts + dense-MLP linears.

## E3. gfx1250 triton fp4 gemm kernels ALL fault (key finding)

On a **healthy** GPU (probed OK immediately before), first call faults:

```bash
# a8w4 (fp8 act x fp4 weight) triton
python3 -c "... from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4; gemm_a8wfp4(x_fp8,w,y,xs,ws,torch.bfloat16)"
# => Memory access fault by GPU node-3 ... Page not present

# a4w4 triton
python3 -c "... from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4; gemm_afp4wfp4(xq,wq,xs,ws,torch.bfloat16,y)"   # => fault

# a16wfp4 (bf16 act x fp4 weight) triton
python3 -c "... from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4; gemm_a16wfp4(x,wq,ws,dtype=torch.bfloat16,y=y)"  # => fault

# bf16 control
python3 -c "... from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16; gemm_a16w16(a,b)"  # NOTE: faulted too, but only AFTER GPU was
# just wedged by the prior fp4 fault; on a fresh GPU bf16 works. i.e. faults are fp4-kernel-specific.
```

- Conclusion: aiter triton `gemm.basic` fp4 kernels target gfx950, **fault on gfx1250**.
- Each fault **wedged** the GPU for the next process (observed via subsequent
  `torch.ones` hang), but recovered after the process died (all 4 GPUs later
  passed `torch.ones` again).

Quant helpers DO work on gfx1250 (safe, no gemm):
`dynamic_mxfp4_quant`, `dynamic_per_token_quant_fp8_i8`, `dynamic_mxfp8_quant` all OK.

## E4. aiter mainline check — no single (dense) a8w4 gemm

- `ROCm/aiter@main` flydsl dir + `gemm_kernels.py`: only bf16 hgemm + a8w8; the
  only a8w4/mxscale is the **grouped MoE** path. Confirmed no dense a8w4 gemm to reuse.
- `aiter/fused_moe.py` (~L435): on gfx1250, `AITER_FORCE_A8W4=1` sets
  `q_dtype_a=fp8` and routes MoE per_1x32 to the grouped a8w4 kernel; else fp4x2 (a4w4).

## E5. Dense a8w4 gemm wrapper + unit test — prep OK, launch faults

- Wrote `aiter/aiter/ops/flydsl/gemm_a8w4_gfx1250.py` + `op_tests/test_gemm_a8w4_gfx1250.py`.
- Recipe: `dynamic_mxfp8_quant` A; A-scale identity at `tile_m=16,m_warp=1`;
  weight raw `(N,K//2)`; B-scale n32k4 via `shuffle_scale_n32k4`; dense
  `compile_a8w4_gemm(grouped_masked_m=False, batch_count=1)`.
- Isolated prep test (no gemm): shapes all correct
  (`a_scale_shuf (32,16)`, `b_scale_shuf (8,512) == (N//32,(K//32)*32)`). OK.
- Full run: **Memory access fault** in the gemm launch.
- Grep confirmed: the **dense** `compile_a8w4_gemm`/`launch_mxscale_gemm` is
  **not called anywhere** in aiter (all callers `grouped_masked_m=True`) => dense
  path unvalidated/buggy. PARKED.

## E6. Linear -> bf16 dequant; full server launch (GPU3)

- Edited `quark_w4a4_mxfp4.py`: `AITER_FORCE_A8W4` -> dequant fp4 linear weights
  to bf16 at load, `F.linear` in apply. (diff in artifacts/)
- `run_ds-r1.sh` with `AITER_FORCE_A8W4=1`, GPU3, cuda graph ON:
  - **Load weight end. elapsed=119.89 s ... mem usage=353.42 GB** (bf16 dequant OK).
  - KV cache 34.85 GB, memory pool OK.
  - **Crashed during decode CUDA-graph capture** (bs=32): `Memory access fault ...
    Reason: Unknown`. Then coredump attempt -> **disk full** (260 GB `gpucore`).
- Re-run with `--disable-cuda-graph` (eager):
  - **`Application startup complete. Uvicorn running on 0.0.0.0:8000`**,
    `/model_info` 200 OK. Server actually served.
  - First forward (warmup/prefill) -> **Memory access fault** (surfaced at
    `alloc_for_extend`, async). Coredump -> disk full again.
- => baseline: linear bf16 works, server starts; the **forward-pass fault
  remains** (most likely the MoE a8w4 grouped kernel — the only novel forward path).

## E7. Coredump disk trap

- ROCr GPU coredump falls back to `gpucore.<pid>.gpu` in cwd (systemd-coredump
  absent); one was **260 GB**, filled `/` (overlay) to 100%.
- `HSA_ENABLE_COREDUMP=0` did NOT stop it. **`ulimit -c 0` does** (file fallback
  respects RLIMIT_CORE). Added to run_ds-r1.sh + `HSA_COREDUMP_PATTERN=/dev/null`,
  `AMD_COREDUMP=0`. Cleanup: `rm -f /sgl-workspace/*/gpucore.*.gpu`.

## E8. GPU access lost (host reboot)

- Attempted relaunch (eager + `AMD_SERIALIZE_KERNEL=3` to pinpoint the fault) ->
  `RuntimeError: No accelerator available`.
- `rocm-smi`: "Driver not initialized (amdgpu not found)". `torch.cuda.device_count()==0`.
  `/dev/kfd` and `/dev/dri` **gone**. `dmesg` shows a fresh boot ~01:24.
- Container came back after a host reboot **without GPU passthrough**. Not fixable
  from inside the container -> switching docker image.

---

## E9. New docker: GPUs back, re-applied changes, hit aiter fused_qk_rmsnorm build fail

- `torch.cuda.device_count()==4`; sglang @ `8d30387` / aiter @ `8815f4b` (same base
  commits as the diffs) — `git apply` of the linear bf16-dequant diff clean.
- Launched (eager + `AMD_SERIALIZE_KERNEL=3`). Weights load fine, then **scheduler
  dies during init**: aiter JIT build of `module_fused_qk_norm_rope_cache_quant_shuffle`
  fails: `rope_common.h:4458 no type named 'fp16_t' in namespace 'ck_tile'`,
  `:7401 no member 'type_convert'`, and `ck_tile/vec_convert.h:8-28` (`vec_t`,
  `thread_buffer`, `vector_traits`, `fp8_t/fp16_t/bf16_t/fp32_t`). No prebuilt `.so`.
- Wiped `aiter/jit` and rebuilt clean -> **same compile errors** (source/CK mismatch,
  not stale cache). Trigger: `forward_mla.py` imports `fused_qk_rmsnorm` under
  `_use_aiter`. Dispatch: triton backend -> prefill=MHA, decode=MLA -> qk-rmsnorm is
  decode-only.

## E10. Triton qk-rmsnorm drop-in -> reaches MoE forward; FlyDSL IndexError

- Wrote `triton_qk_rmsnorm.py` (bit-parity with torch RMSNorm: 0 diff small M, 1 bf16
  ULP at M=4096); bound it in `forward_mla.py` under `AITER_FORCE_A8W4`.
- Relaunch: **server starts** (`Uvicorn`), first MoE forward now runs and faults with
  a clean Python `IndexError: tuple index out of range` at `<flydsl-dispatch>:65`,
  via `fused_moe.py:451 _maybe_grouped_gfx1250_a8w4_moe -> stage1 ->
  moe_grouped_gemm_mxscale_gfx1250.py:1467 (dense) launch -> _run_compiled`.
- Forcing `AITER_GROUPED_DEEPGEMM_CONTIGUOUS=1` moved it to the contiguous branch
  (:1389) — still IndexError => both 12-arg branches mismatch the kernel's 13-param
  signature.

## E11. Root-cause + fix the IndexError (stage1 raw missing swiglu arg)

- `launch_mxscale_gemm_masked[_bias]` (gemm_mxscale_gfx1250.py) has 13 runtime params
  (trailing `swiglu_limit_f=inf`); stage1 raw `_run_compiled` calls passed 12.
  **stage2 raw calls pass `_no_act_swiglu_lim`** (correct reference). Fast-dispatch
  (2nd call) indexes the missing arg -> IndexError.
- Fix: append `_swiglu_lim_rt` to the 6 stage1 raw calls. Relaunch: IndexError gone.

## E12. split_k=2 raw+finalize GPU illegal-address -> force split_k=1

- Next fault: GPU `hipErrorIllegalAddress` in
  `moe_stage1_finalize_act_silu_bf16_e257_m16_i2048_gguu_v4_sk2`. Tuned CSV picks
  `split_k1=2` for DeepSeek dims -> `_get_fused_base()=None` -> raw+finalize path
  (unvalidated) instead of the op-test-validated fused path.
- Added `AITER_GROUPED_FORCE_SPLIT_K1` env (force split_k1=split_k2=1). Relaunch with
  it: **server fully up, `ready to roll`, /generate 200 OK, prefill correct**
  (" Paris"). GPU3 never wedged (all faults were clean or reaped).

## E13. Decode emits token 0 -> fp8 KV cache is the culprit

- `output_ids=[<good first token>,0,0,...]` on every prompt; GSM8K 40Q = **0.000 /
  invalid 1.000**. Prefill (MHA) correct, decode (MLA-absorb) degenerate.
- Bisect: `SGLANG_QK_RMSNORM_TORCH=1` (torch qk-rmsnorm) — no change => not the kernel.
  Decode-only difference is the MLA-absorb read of the **fp8_e4m3 KV cache**.
- Switch `--kv-cache-dtype fp8_e4m3 -> auto` (bf16): decode coherent (" Paris. Paris
  is located in the northern-central part of the country ..."). GSM8K 40Q =
  **0.800 / invalid 0.000** (eager). fp8-KV decode read on gfx1250 is broken.

## E14. CUDA graph works; final perf

- Dropped `--disable-cuda-graph` and `AMD_SERIALIZE_KERNEL=3`. Decode CUDA-graph
  capture (bs=32..1) **succeeds, no fault** (this was the E6 crash point). Decode
  `cuda graph: True`, coherent. GSM8K 40Q = **0.850 / invalid 0.000**, latency 57 s
  vs 359 s eager (~70 vs 16 tok/s). Under capture mode the qk-norm uses the
  alt-stream torch layernorm branch (not Triton) — also correct.
- bf16 gemms log `not found tuned config ... using torch solution:0` -> perf headroom.

## E15. GSM8K verification matrix (qk-kernel x cuda-graph)

All runs: DeepSeek-R1-0528-MXFP4, `AITER_FORCE_A8W4=1`,
`AITER_GROUPED_FORCE_SPLIT_K1=1`, `--kv-cache-dtype auto`, triton attn backend,
5-shot, greedy, `--max-new-tokens 512`.

| config                       | Q   | Accuracy | Invalid | tok/s |
|------------------------------|-----|----------|---------|-------|
| eager + torch qk             | 40  | 0.800    | 0.000   | 16.5  |
| cuda-graph + torch qk        | 40  | 0.850    | 0.000   | 70.8  |
| cuda-graph + torch qk        | 200 | 0.870    | 0.000   | 101.3 |
| eager + Triton qk            | 100 | 0.890    | 0.000   | 77.2  |
| cuda-graph + Triton qk       | 200 | 0.840    | 0.000   | 126.1 |
| cuda-graph + Triton qk       |1319 | 0.822    | 0.000   | 158.0 |

Full-set number (all 1319 GSM8K test questions, `benchmark/gsm8k/bench_sglang.py
--num-questions 1319 --parallel 1319 --num-shots 5`): **Accuracy 0.822, invalid
0.000**, 914 s end-to-end, 158 tok/s (server max_running_requests=32, so ~32
concurrent). This is the stable headline number for the working recipe.

Notes:
- All configs land at **0.82-0.89** with **invalid 0.000** => fully working; Triton
  qk-rmsnorm and torch are equivalent, eager and cuda-graph are equivalent. The
  0.84 vs 0.87 spread between the two cuda-graph/200Q runs is batch-composition
  run-noise (parallel=32), not a real qk-kernel difference (both use the same torch
  layernorm capture branch in decode).
- The **Triton qk-rmsnorm kernel is exercised only in eager** decode. Under
  cuda-graph, decode replays a graph captured in capture-mode, which uses the
  alt-stream + torch `q_a_layernorm` branch (not `fused_qk_rmsnorm`) — so
  "cuda-graph + Triton qk" and "cuda-graph + torch qk" run the same decode math
  (hence both 0.870). Row (a) `eager + Triton qk` is the one that actually validates
  the Triton kernel end-to-end.
- Stable score ~**0.87** (200 Q); ~100 tok/s under cuda-graph.

## E16. Accuracy gap investigation — a8w4 kernel numerics vs qdq reference

Question: GSM8K ~0.82 (completion) is below DeepSeek-R1's expected ~0.93-0.94.
Investigated whether it's a bug, quantization, kernel numerics, or eval setup.

- **split_k is NOT the lever (C ruled out).** The tuned CSV
  (`aiter/configs/tuned_grouped_fmoe.csv`) sets `split_k1=2` **only for token=1**
  (the single-token decode that hit the E12 `..._sk2` finalize fault); tokens>=2
  already use `split_k1=1`. So `AITER_GROUPED_FORCE_SPLIT_K1=1` did NOT cost
  accuracy — it matches the tuned config for all token>1.
- **a8w4 grouped MoE numerics (op-test verify, DeepSeek dims 7168/2048/256/topk8,
  quant-matched torch ref):** logits_diff grows with token count but stays under the
  kernel's own gate (<0.01): tokens=8 -> 3.4e-6 (rel_l2 0.26%); 64 -> 1.7e-3 (5.9%);
  128 -> 3.2e-3 (8.0%); 256 -> 2.0e-3; 512 -> 5.4e-3 (10.4%); 1024 -> 5.2e-3. The
  6-10% "rel_l2" is just sqrt(2*logits_diff); by the kernel's own tolerance it
  passes, but the token/max_m-dependent growth is a small systematic error.
- **aiter has NO working bf16 (a16w16) MoE on gfx1250:** `fused_moe` only routes
  a4w4/a8w4 to the flydsl grouped kernel; the bf16 `aiter.fmoe`/ck path is
  `module_moe_asm` (CK) which **fails to build on gfx1250** (ENABLE_CK=0 required;
  CK unsupported). A full bf16 MoE ceiling is also infeasible on VRAM (~650B params
  -> ~1.3 TB bf16 vs 432 GB) — that's why the model is MXFP4.

### Eval methodology (from the HF model card)
`amd/DeepSeek-R1-0528-MXFP4` reports AIME24/GPQA/MATH-500 (not GSM8K) via lighteval
with **chat template, temperature 0.6, top_p 0.95, max_new_tokens 65536, 10 seeds**,
and crucially **qdq emulation** (gfx950 docker, offline dequant -> bf16 compute) —
i.e. the published recovery (96-101%) reflects *ideal quantization*, not a real
MXFP4/a8w4 kernel.

GSM8K under matched methodology on our gfx1250 real-a8w4 server (0-shot chat +
--enable-thinking, temp 0.6, top_p 0.95, max_new_tokens 32768):
| method                                    | Q    | Accuracy | Invalid |
|-------------------------------------------|------|----------|---------|
| completion, greedy (5-shot)               | 1319 | 0.822    | 0.000   |
| thinking, max **8192** (TRUNCATED, wrong) | 40   | 0.675    | 0.000   |
| thinking, temp0.6 top_p0.95 max **32768** | 40   | 0.875    | 0.000   |
| thinking, temp0.6 top_p0.95 max **32768** | 200  | **0.855**| 0.000   |
| gfx950 reference (user-measured)          | -    | 0.93-0.94| -       |

Corrections/notes:
- The earlier 0.675 "thinking is worse" was a **max_new_tokens=8192 truncation
  artifact** (R1 reasoning needs up to 65536), NOT a quality signal. Fixed methodology
  gives 0.855-0.875.
- A stable, methodology-matched **0.855 (200 Q)** still trails gfx950's 0.93-0.94 by
  ~7-8%, invalid 0.000 (not a parsing issue). Several reasoning chains **ran away to
  the 32768 cap without converging** (single chain dominated the 4980 s latency) — a
  classic symptom of per-layer numerical error accumulating and pushing long chains
  into repetition loops.
- => The residual ~8% gap is most consistent with the **gfx1250 real a8w4 MoE kernel
  numerics** (vs the bf16 qdq-emulated reference), not a serving-integration bug. The
  serving baseline itself is correct and stable.

---

## Next steps (as of 2026-07-07; SUPERSEDED — see 2026-07-08 session below)

1. ~~Close the ~8% accuracy gap: improve gfx1250 a8w4 grouped MoE kernel numerics~~
   **DISPROVEN 2026-07-08 (E20): the a8w4 MoE is NOT the gap.**
2. Perf: tune the bf16 gemms (populate `bf16_tuned_gemm.csv`) to beat `torch solution:0`.
3. Fix fp8-KV decode read (triton MLA decode) on gfx1250 to halve KV memory (E13).
4. Fix the split_k>1 (token=1) raw+finalize MoE path (E12) instead of forcing split_k=1.
5. Optional: build a real dense a8w4 linear kernel to drop the bf16-dequant. NOTE:
   the earlier parked wrapper (gemm_a8w4_gfx1250.py + its test) was **removed on
   2026-07-07** (faulty, never on any code path; linear uses bf16 dequant). Recreate
   from git history / SKILL §5 if pursued.

---

# 2026-07-08 session — new docker, MoE exoneration (root-cause overturned)

Environment differs from the 2026-07-07 log:
- Model path moved to `/dockerx/data/models/DeepSeek-R1-0528-MXFP4` (Quark W4A4,
  fp4/fp4 confirmed; **n_routed_experts=256**, topk=8, model_dim=7168, inter=2048).
- `sglang` HEAD `000a61a2` ("Enable DSv4 a8w4 MoE: shuffle FP4 expert weights and
  bring-up env") — newer than the 8d30387 base. Already carries the gfx1250 MoE
  **B-scale n32k4 preshuffle** (`quark_w4a4_mxfp4_moe.py` `moe_shuffle_scale`,
  unconditional for gfx1250) but the **weight** shuffle was still gated to gfx95.
- `aiter` fork `8815f4b5` already has the §4.5 stage1 `_swiglu_lim_rt` fix baked in
  and defaults `split_k1=1`; but it does NOT implement `AITER_GROUPED_FORCE_SPLIT_K1`.

## E17. Re-applied the (lost) morning edits + brought the recipe up on this docker
Three code changes (the morning window's work was never synced here; kernel was
still the buggy version, no env switches, no E18):
1. **aiter bisect off-by-one** (`kernels/gemm_mxscale_gfx1250.py` ~L3010):
   `_bisect_iters = max(1, math.ceil(math.log2(batch_count)))`
   -> `max(1, int(batch_count).bit_length())`. The upper_bound over `m_tile_map`
   has answer space `[0, batch_count]` (needs `bit_length()` iters); `ceil(log2)`
   under-counts by one for **power-of-two** batch_count — R1 has **256 experts =
   2^8**, so it triggers; expert 1's tile resolved to expert 0's weights.
2. **sglang quark MoE weight shuffle on gfx1250** (`quark_w4a4_mxfp4_moe.py`),
   mirroring the DSv4 `fp8.py` commit (`is_shuffled = _is_shuffle_moe_mxfp4 or
   _use_aiter_a8w4`, gu-interleave off for a8w4 == plain `shuffle_weight(w,(16,16))`,
   matches op_tests/test_flydsl_grouped_gemm_gfx1250.py). New flag
   `_shuffle_moe_gfx1250 = _is_gfx1250 and AITER_FORCE_A8W4 and
   SGLANG_MOE_SHUFFLE_GFX1250(default true)`.
3. **aiter `AITER_GROUPED_FORCE_SPLIT_K1` env re-added** (`grouped_moe_gfx1250.py`):
   the tuned CSV row for the token=1 R1 decode dims sets split_k1=split_k2=2 (the
   E12 raw+finalize illegal-address path); force split_k=1 -> fused path. Also added
   `AITER_GROUPED_FORCE_TILE_M` investigation knob (default off).
Cleared `/root/.flydsl/cache` to force recompile with the bisect fix. Server boots,
cuda-graph capture clean, decode coherent (" Paris. Paris is located ...").

## E18. bisect fix WORKS at kernel level, but is END-TO-END NEUTRAL
- op-test (`--scenario verify --data-format a8w4 --experts 256 --model-dim 7168
  --inter-dim 2048 --no-check-aot-cache`), tokens 8 / 128 / 256:
  logits_diff = **3.40e-6 / 3.40e-6 / 3.42e-6** (contiguous now == non-contiguous;
  was 3.2e-3 on the 128/256 contiguous rows before the fix). Kernel bug is real & fixed.
- GSM8K on the server (5-shot, greedy, bf16 KV, cuda-graph, shuffle on, bisect fixed):
  | harness | Q | Accuracy | Invalid |
  |---|---|---|---|
  | few_shot_gsm8k | 200 | 0.835 | 0.000 |
  | few_shot_gsm8k | 1319 | **0.811** | 0.001 |
  | bench_sglang.py | 100 | 0.850 | 0.000 |
  0.811 (1319Q) ~= the 2026-07-07 baseline 0.822 (E15) => **the bisect fix does not
  move end-to-end accuracy.** The morning's "contiguous-M bug causes the ~11% gap"
  conclusion is WRONG: the bug is real, but its accuracy impact is ~0.

## E19. Phase 0 — cheap bounds
- **shuffle OFF** (`SGLANG_MOE_SHUFFLE_GFX1250=0`, bisect still fixed): GSM8K 100Q =
  **0.000 / invalid 1.000** (garbage). => on THIS code the weight shuffle is
  **mandatory** (B-scale is already n32k4-shuffled for gfx1250; an unshuffled weight
  mismatches it -> garbage). shuffle on = 0.85. Keep shuffle ON.
  (NUANCE — see E25b/E25c: shuffle is mandatory ON THIS STACK because THIS docker's aiter
  grouped kernel requires (16,16)-shuffled weight. It is NOT that "raw weight is inherently
  broken": commit 7aa6082 got 0.82 on gfx1250 with raw weight on the OLD docker's aiter.
  Verified 2026-07-09: reverting the bisect fix does NOT restore raw-weight (still 0.000),
  so the break is the aiter kernel VERSION, not sglang / bisect / split_k1.)
- **eager** (`--disable-cuda-graph`): 100Q = 0.840 ~= cuda-graph 0.850. cuda-graph
  capture branch is NOT the gap.

## E20. Phase 1 — MoE is DECISIVELY EXONERATED (root-cause overturned)
Pure-numeric probe on REAL layer-3 expert weights (`/tmp/moe_quant_probe.py`):
weights are natively fp4 in the checkpoint, so the *weight* is identical on both
paths; the ONLY difference between a4w4 (gfx950) and a8w4 (gfx1250) is the
**activation** quant (fp4 vs fp8). Using aiter `dynamic_mxfp4_quant` /
`dynamic_mxfp8_quant`, FFN output vs bf16 (fp4-weight, full-precision-act) ref:

| path | activation | FFN rel_l2 vs bf16 |
|---|---|---|
| a4w4 (gfx950) | fp4 | **0.151 (~15%)** |
| a8w4 (gfx1250) | fp8 | **0.036 (~3.6%)** |
| ratio a8w4/a4w4 | | **0.24** |

=> a8w4 (gfx1250) activation quant is **4x more accurate** than a4w4 (gfx950), as
basic fp8-vs-fp4 mantissa math demands. Combined with E18 (a8w4 kernel == its quant
ref at 3e-6) and E19 (unshuffled = garbage, so the layout is correct, not subtly
wrong), the a8w4 MoE **cannot** be the source of gfx1250's lower GSM8K — if anything
it should make gfx1250 *better*.

Corollary: gfx950 reaches 0.932 while running a4w4 (~15% per-layer FFN activation
error), so the network is **robust to MoE FFN error** (residual paths absorb it).
Chasing MoE numerics further is pointless.

## E21. More cheap probes (all negative)
- **`--triton-attention-reduce-in-fp32`**: 100Q = 0.850 (unchanged). This flag is
  not wired into the MLA decode kernel path anyway; no effect.
- **`SGLANG_ROCM_FUSED_DECODE_MLA=1`** (fused decode MLA): server **crashes** at
  cuda-graph capture — `TypeError: cannot unpack non-iterable ForwardMetadata object`
  (fused-rope decode MLA incompatible with this triton backend/version). This is why
  the recipe keeps it =0; it cannot be used for an A/B.
- **`--disable-radix-cache`**: 100Q = 0.830 ~= 0.850 (radix on). The few-shot GSM8K
  prefix-reuse (MLA extend-with-prefix path) is NOT the gap; forcing cold prefill
  every request does not help.
- **Failure-mode inspection** (greedy, per-token top-k logprobs on a degenerate case):
  the `%?%?%?...` repetition loop is a **high-confidence greedy loop** (chosen-token
  logprob ~= 0.0, i.e. prob ~= 1.0), i.e. generic greedy degeneration, NOT a numerical
  collapse (which would show flat/uniform logits). Other errors are ordinary arithmetic
  slips. So the failure modes do NOT indicate a decode numerical bug.

## E22. Phase 2b / Option A — attention kernels are ALSO numerically clean
Monkeypatched `TritonAttnBackend.forward_decode` and `forward_extend` (via a
`sitecustomize.py` on `PYTHONPATH`, env-gated `SGLANG_ATTN_PROBE=1`, eager server so
the Python hook actually runs — cuda-graph replay bypasses it). For each layer,
recomputed attention in **torch fp32** from the same q + paged KV and compared to the
triton kernel output `o` (rel_l2):
- **decode (MLA-absorb)**: all 61 layers rel_l2 **0.0015-0.0019** (overall 0.00175),
  no outlier layer (worst L35 = 0.0019).
- **prefill (MHA cold-prefill)**: all 61 layers rel_l2 **0.0013-0.0020**, no outlier.
Both are pure bf16-rounding noise => the **gfx1250 triton MLA decode AND MHA prefill
kernels are numerically correct**. Attention is exonerated.
- Also: the bf16 GEMMs (attention q/k/v/o projections, lm_head, dense-MLP) log
  `using torch solution:0` — they are literally torch gemm => correct by construction.

**Net after Option A:** on gfx1250, EVERY component checked in isolation is numerically
correct to bf16-noise level (MoE a8w4 is *more* accurate than a4w4; attention decode +
prefill match torch fp32; bf16 gemm == torch). No single-kernel bug found.

## E23. D — reference baseline validated as comparable (gap is REAL)
User confirmed: gfx950 0.93 is **stably reproducible** (not noisy/optimistic), and
both nodes pull the **same HF checkpoint** `amd/DeepSeek-R1-0528-MXFP4`.
- Checkpoint fingerprint on this node matches the HF card: `w_mxfp4_a_mxfp4` group32,
  82 shards / 376G, 256 experts / 61 layers, exclude list = all `self_attn.*` proj +
  `mlp.gate` + `lm_head` + layer 61 (so **attention projections are bf16 on both nodes**).
- Code: `git diff 7aa6082(gfx950) .. 000a61a2(gfx1250)` = only **3 commits**, touching
  `fp8.py` / `fp8_kernel.py` / run script / a new test — **model / attention / MoE /
  quark code identical** between the two nodes.
- Eval: both completion 5-shot greedy GSM8K (confirmed same on two harnesses:
  few_shot_gsm8k and benchmark/gsm8k/bench_sglang.py, both ~0.81-0.85).
=> checkpoint, model code, and eval are all comparable. The 0.85(gfx1250-a8w4) vs
0.93(gfx950-a4w4) gap is **real and reproducible**, and the ONLY forced difference is
a8w4-vs-a4w4 MoE + gfx1250-vs-gfx950 hardware/triton-codegen. Since a8w4 is *more*
accurate and every gfx1250 kernel is clean in isolation, the remaining explanation is
either (a) a cross-layer interaction not visible in per-op probes, or (b) the
a8w4-vs-a4w4 *scheme* interacting with the calibrated weights. Requires cross-node
per-layer comparison to localize.

## E24. Cross-node hidden-state dump (handover) + TP2 note
- Wrote `HANDOVER_crossnode_dump.md` + a validated dump hook (`/tmp/hsdump/
  sitecustomize.py`, env `HS_DUMP=1`): monkeypatches `DeepseekV2DecoderLayer.forward`,
  records per-layer input & post-layer residual-stream (norm + 64-dim slice) for
  exactly the first prefill pass (61 layers) on a **fixed prompt** (the "Natalia" GSM8K
  Q), then dumps JSON and stops. Must launch **eager + `--disable-radix-cache` +
  `--skip-server-warmup`** (else the warmup prefill captures the wrong prompt).
- gfx1250 (TP1) dump produced & validated: `hs_dump_gfx1250.json` (61 unique layers,
  norms grow 2.3 -> 30 -> 319, no errors) — saved in this skill dir.
- **TP2 vs TP1 for the comparison**: the residual stream is **replicated** (all-reduced)
  under TP, so layer input/output hidden states are mathematically identical on TP1 vs
  TP2; slice[:64] is global coords (hidden=7168 not sharded). Only caveat: all-reduce /
  GEMM reduction-order differs => a ~1e-3 bf16 floor. Look for divergence that GROWS
  well beyond that floor at a specific layer. To remove the confound entirely, plan was
  to re-dump gfx1250 in **TP2** (this box has 4x gfx1250) so it's TP2-vs-TP2.
  **STATUS: TP2 dump run was launched (GPU 2,3) but the machine shut down before it
  completed; not captured.** NOTE (2026-07-09): user says this machine cannot run TP2
  reliably — so either dump gfx1250 in TP1 and tolerate the ~1e-3 TP floor, or use a
  box that supports TP2 on both sides.

## Next steps (updated 2026-07-09)
1. **Cross-node per-layer diff is the only remaining localizer.** Run the
   `HANDOVER_crossnode_dump.md` hook on gfx950 (TP2) for the fixed prompt, diff vs
   `hs_dump_gfx1250.json`. First layer whose **post-layer** residual diverges beyond the
   TP/bf16 floor (~1e-2) and keeps growing = the smoking gun. If divergence is confined
   to MoE-sublayer deltas (expected a4w4 vs a8w4), the gap is the scheme, not a bug.
2. If TP2-vs-TP2 is wanted and this box can't do TP2, get a gfx1250 box that can, or
   compare TP1(gfx1250) vs TP2(gfx950) accepting the ~1e-3 floor.
3. Keep all E17 code changes (bisect / shuffle / split_k1) — correct & required, just
   not the accuracy gap.
4. Everything else (MoE, attention decode+prefill, bf16 gemm, cuda-graph, radix,
   fp32-reduce) is already ruled out — do NOT re-chase them.

## E25b. Clarification: "unshuffled = 0.000" is a scale/weight MISMATCH, not universal
Question (user): before yesterday's edits (no gfx1250 weight shuffle, no bisect fix)
GSM8K was still ~0.8x; but yesterday `SGLANG_MOE_SHUFFLE_GFX1250=0` gave 0.000 garbage —
how did the earlier code get 0.8x with no weight shuffle?
Anchor: the CURRENT `quark_w4a4_mxfp4_moe.py` `process_weights_after_loading` shuffles the
MoE **B-scale to n32k4 UNCONDITIONALLY for gfx1250** (`if _is_gfx1250: moe_shuffle_scale`),
while the **weight** shuffle was gated to gfx95 only (`_is_shuffle_moe_mxfp4 =
is_gfx95_supported()`). So:
| state | weight | B-scale | consistent? | GSM8K |
|---|---|---|---|---|
| current code, `SGLANG_MOE_SHUFFLE_GFX1250=0` | raw | n32k4-shuffled | NO (mismatch) | 0.000 |
| current code + fix (E17) | (16,16)-shuffled | n32k4-shuffled | yes | 0.85 |
| earlier 0.8x runs | consistent (see below) | consistent | yes | 0.8x |
=> **0.000 is caused by weight-raw WHILE scale-shuffled (layout mismatch), NOT by "no
shuffle" per se.** The earlier 0.8x code was internally CONSISTENT — either (a) it did not
n32k4-shuffle the gfx1250 B-scale either (both raw), or (b) its weight-shuffle gate also
covered gfx1250 (both shuffled). Someone later added the gfx1250 n32k4 B-scale shuffle
WITHOUT adding the matching weight shuffle -> silent mismatch -> 0.000. The E17 weight-
shuffle fix simply realigns the weight to the already-shuffled scale.
Corrections: E19 / SKILL / STATUS said "unshuffled = garbage" too absolutely — it should
read "weight-unshuffled while scale-shuffled = garbage".
CONFIRM when a box is up: (1) `git log -p` on quark_w4a4_mxfp4_moe.py to date the gfx1250
`moe_shuffle_scale` vs weight-shuffle gate; (2) on current code, ALSO skip the gfx1250
`moe_shuffle_scale` (both raw, consistent) and expect GSM8K to return to ~0.82 — proving
the mismatch theory. Tool for (2): `scripts/gfx1250_disable_moe_scale_shuffle_sitecustomize.py`
(env `SGLANG_DISABLE_MOE_SCALE_SHUFFLE_GFX1250=1` + `SGLANG_MOE_SHUFFLE_GFX1250=0`).
  - both-raw -> ~0.82 => confirms mismatch theory (old code was both-raw).
  - both-raw -> still 0.000 => the aiter grouped kernel REQUIRES shuffled B/B-scale, so the
    old 0.82 must have shuffled BOTH (old weight-shuffle gate covered gfx1250).

## E25c. CORRECTED: 7aa6082 IS the gfx1250 0.82 recipe commit; raw-weight break is an aiter-VERSION difference
(My earlier claim here — "7aa6082 was the gfx950 tree, never run on gfx1250" — was WRONG;
user corrected it, git confirms.)
- `git show --no-patch 7aa6082b57`: authored by me (Cursor co-author) 2026-07-07, message
  "Enable DeepSeek-R1-0528-MXFP4 serving on a single **gfx1250** ... GSM8K **~0.82**
  completion". So **7aa6082 is the gfx1250 working-recipe commit, 0.82 ON gfx1250.** The
  gfx950 cross-val (gfx1250.md) simply COPIED that same tree (rev 7aa6082) onto gfx950 and
  got 0.932 there — same commit used on both nodes; both numbers are real.
- `git show 7aa6082b57:...quark_w4a4_mxfp4_moe.py` is IDENTICAL to the current pre-fix
  file: gfx1250 = n32k4 B-scale shuffle + weight NOT shuffled ("raw-weight" state).
  => At 7aa6082 this raw-weight state gave **0.82 on gfx1250**.

### VERIFIED ablation (2026-07-09, machine back up)
Reproduced the raw-weight state (`SGLANG_MOE_SHUFFLE_GFX1250=0`) on the CURRENT stack:
| stack | weight | bisect | GSM8K 100Q |
|---|---|---|---|
| 7aa6082 (original docker aiter) | raw | buggy (ceil log2) | **0.82** (historical) |
| current, raw | raw | fixed (bit_length) | **0.000** / invalid 0.99 |
| current, raw + bisect REVERTED to buggy | raw | buggy | **0.000** / invalid 1.0 |
| current, shuffled (E17 fix) | (16,16) | fixed | 0.85 |
=> Reverting the bisect fix did NOT restore raw-weight (still 0.000). So the bisect fix is
NOT the cause, and split_k1 is identical on both. **The raw-weight 0.82->0.000 difference
is the aiter grouped-a8w4 kernel VERSION/build itself** (this docker's aiter requires
(16,16)-shuffled weight; the original docker's aiter tolerated/handled raw weight). The
sglang quark file is byte-identical between 7aa6082 and current, so it's not sglang.
=> **Reconciliation (no contradiction):** 7aa6082 got 0.82 with raw weight on the OLD
docker's aiter; on THIS docker's aiter, raw weight = 0.000 and (16,16)-shuffled = 0.85.
The E17 weight-shuffle fix correctly aligns sglang to THIS aiter's kernel expectation.
Corrections applied: E19 "shuffle mandatory" is true ON THIS STACK but NOT because raw is
inherently broken; it is because this aiter build requires shuffled weight.
- The gfx1250 n32k4 scale-shuffle block was added by `6f1c92907 "Squash gfx1250
  development"` (before it: unconditional e8m0_shuffle). The 8d30387 docker (the even
  earlier 0.822 runs) is a different fork not in this repo — can't inspect.

## E25. gfx95-gated path audit — MLA absorb BMM is fp4 on gfx950, bf16 on gfx1250
Question (user): are there MANY `_use_aiter_gfx95`-gated paths that gfx950 takes and
gfx1250 does not, causing the gap? Audited all `is_gfx95_supported` / `_use_aiter_gfx95`
branches in the DeepSeek path. For **R1** (attention q/k/v/o proj are bf16 = excluded
from quant), most gfx95 branches are ALSO gated on `weight.dtype == fp8/uint8` and so
**do not fire on either platform**. The branches that DO fire for R1:
- `quark_post_load_weights` (`quark/utils.py`, called in `deepseek_weight_loader.py:632`
  under `_use_aiter_gfx95 and quark and DeepseekV3`): on gfx950 it dynamically quantizes
  the bf16 `kv_b_proj` weight into **mxfp4** w_kc/w_vc (+e8m0 scales). Its bf16 split is
  **identical** to the generic path (`deepseek_weight_loader.py:628-630`); the ONLY
  difference is the mxfp4 quantization. On gfx1250 it is **skipped** -> w_kc/w_vc stay bf16.
- Consequently `forward_mla.py:479` (`_use_aiter_gfx95 and w_kc.dtype==uint8`, and :874
  for w_vc): the **MLA absorb BMM** (`q_nope @ w_kc`, `attn_out @ w_vc`) runs **fp4×fp4**
  (`batched_gemm_afp4wfp4_pre_quant`) on gfx950, but **bf16** on gfx1250.
- `deepseek_v2.py:2421` (`_use_aiter_gfx95 and n_routed_experts==256`): only sizes a
  `gemm_output_zero_allocator` buffer — **no numerical effect**.
- All the fused-RMSNorm-fp8 / fp8-bmm gfx95 branches require fp8 weights -> don't fire
  for R1's bf16 attention.

**Key result:** the confirmed R1 non-MoE gfx95 divergence is the **MLA absorb BMM
(gfx950 fp4 vs gfx1250 bf16)**. Note the DIRECTION: gfx1250 is again the *more precise*
one (bf16 > fp4). So across every audited path (absorb BMM, MoE) gfx1250 is equal-or-
more-precise than gfx950, yet scores lower. This makes a **quantization-MATCHING effect**
the leading thesis: the Quark PTQ model may perform best under its own quant error
pattern (a4w4 MoE + fp4 absorb), and gfx1250's more-precise bf16/a8w4 substitutions are a
distribution mismatch.

### DECISIVE single-node experiment (no cross-node confound) — TODO on gfx950
Ablate gfx950 to gfx1250's MORE-precise absorb path and re-measure:
- Tool: `scripts/gfx950_disable_absorb_fp4_sitecustomize.py` (env
  `SGLANG_DISABLE_QUARK_ABSORB_FP4=1`) monkeypatches `quark_post_load_weights` to return
  **bf16** w_kc/w_vc (mimicking gfx1250), so gfx950's absorb BMM runs bf16.
- Run GSM8K on gfx950 with it. **If accuracy drops 0.93 -> ~0.85**, the fp4 absorb (i.e.
  quant-matching) IS the gap. **If it stays 0.93**, the absorb isn't it -> the gap is the
  a8w4-vs-a4w4 MoE scheme.
- **IMPORTANT (user, 2026-07-09): gfx1250 A0 does NOT support a4w4 (fp4-activation)
  gemm** — it lacks `V_WMMA_SCALE_F32_32X16X128_F4`, which is the whole reason a8w4 is
  used. So the gfx950 fp4 absorb (`batched_gemm_afp4wfp4_pre_quant`) **cannot be ported
  to gfx1250** (it would crash, SQC inst fault). Therefore:
  - The complementary "force fp4 absorb on gfx1250" test is **infeasible** — do NOT try it.
  - If the gfx950 ablation shows absorb precision matters, the only feasible gfx1250
    absorb-quant direction is **a8w4 absorb** (fp8-act x fp4-weight, via the supported fp8
    scaled-WMMA, mirroring the MoE workaround). CAVEAT: a8w4 (fp8 act) is still MORE
    precise than gfx950's fp4 act, so if the effect is pure quant-matching, a8w4 absorb may
    only PARTIALLY close the gap, not fully.
- See `HANDOVER_gfx950_ablation.md`.

## E26. gfx950 absorb-BMM ablation DONE — absorb precision is NOT the gap (2026-07-09)
Ran the E25 decisive single-node A/B on gfx950 (8x MI355X, 309 GB). Same recipe on
both, only the absorb weight prep differs; two servers in parallel (TP2 each):
- **baseline** (GPU0,1 :8000): normal gfx950 = **fp4 absorb** (`quark_post_load_weights`
  mxfp4 w_kc/w_vc).
- **ablation** (GPU2,3 :8001): `PYTHONPATH=/tmp/abl SGLANG_DISABLE_QUARK_ABSORB_FP4=1`
  -> `scripts/gfx950_disable_absorb_fp4_sitecustomize.py` monkeypatches
  `quark_post_load_weights` to return **bf16** w_kc/w_vc (mimics gfx1250). Confirmed the
  hook fired on every TP rank (`[gfx950_abl] quark_post_load_weights -> bf16 absorb`).
  MoE stays a4w4 on both (only the absorb changes).

Both run the modified tree at `/sgl-workspace/sglang_gfx-1250` (via PYTHONPATH; the
pip-editable `/sgl-workspace/sglang` is the OLD tree and must NOT be used). Model
`/dockerx/data/amd/DeepSeek-R1-0528-MXFP4`. No source edits — the ablation is a pure
runtime monkeypatch.

| absorb path            | GSM8K 200Q | GSM8K 1319Q | Invalid |
|------------------------|------------|-------------|---------|
| fp4 (gfx950 native)    | 0.950      | **0.944**   | 0.000   |
| bf16 (mimic gfx1250)   | 0.950      | **0.942**   | 0.000   |

**Result: 0.944 vs 0.942 (1319Q) = ~2-3 questions = pure noise; 200Q identical (0.950).**
Forcing the absorb to gfx1250's more-precise bf16 does **NOT** drop accuracy toward 0.85.
=> The **MLA absorb BMM precision (fp4 vs bf16) is NOT the gfx1250 gap.** The E25
quantization-matching thesis is FALSE for the absorb path. (Both baselines also confirm
the gfx950 reference ~0.94 reproduces on the modified tree, consistent with gfx1250.md.)

Remaining suspects (unchanged from E23/E24): the **a8w4-vs-a4w4 MoE scheme interaction**
with the calibrated weights, and/or a cross-layer interaction only visible via the
**cross-node per-layer dump** (`HANDOVER_crossnode_dump.md`) — now the single remaining
localizer. Note: E20 already showed a8w4 MoE is *more* accurate than a4w4 per-op, so the
scheme effect (if any) is a distribution/quant-matching interaction, not a kernel bug.

Do NOT re-run the gfx1250 absorb-quant EMULATION for the WEIGHT side expecting a big
swing: the gfx950 ablation shows absorb weight precision (fp4 vs bf16) is accuracy-
neutral, so `SGLANG_ABSORB_QUANT_EMUL=w_fp4` on gfx1250 is expected neutral too. The
activation-side a8w4/a4w4 emulation is the only part of that handover still worth doing,
and only if the cross-node dump points back at the absorb.

## E27. gfx950 cross-node per-layer dump PRODUCED (2026-07-09)
Produced the gfx950 side of the cross-node residual-stream diff (HANDOVER_crossnode_dump.md).
- **Config = the ablation, not the baseline** (user request): `SGLANG_DISABLE_QUARK_ABSORB_FP4=1`
  (bf16 absorb, mimic gfx1250) so the non-MoE numerics are as close to gfx1250 as possible.
  The only remaining forced non-MoE divergence is the MLA KV-buffer store/load layout
  (`_is_cuda or _use_aiter_gfx95`, forward_mha L488/L513) — numerically equivalent (same
  bf16 kv_a/k_pe values), plus per-arch triton codegen (~1e-3 bf16 floor). MoE stays a4w4.
- Launch: eager (`--disable-cuda-graph`) + `--disable-radix-cache` + `--skip-server-warmup`,
  TP2 (model 353 GB > single 309 GB card, so TP1 impossible), fixed "Natalia" GSM8K prompt,
  greedy 1 tok (-> " First"). Both hooks combined in one `/tmp/dump/sitecustomize.py`
  (`import _abl; import _hsdump`) since Python auto-imports only one sitecustomize.
- **Hook bug found & fixed**: with TP2 both ranks wrote the same `$HS_DUMP_OUT` -> corrupted
  JSON (parse error). Fixed `scripts/hsdump_sitecustomize.py` to be **rank-gated** (rank 0 ->
  canonical file, others -> `.rank{N}`). Re-ran clean.
- **Result: `hs_dump_gfx950.json`** (saved in skill dir): 61 layers, post_attn all 61
  captured, norms input 2.3->78 / post_layer 2.4->358, rank0==rank1 (post_layer norm diff
  0.0 -> residual fully replicated under TP, rank0 representative).
- **BLOCKER for the actual diff:** `hs_dump_gfx1250.json` is STALE (old hook, no post_attn).
  Must re-dump gfx1250 with the updated rank-gated hook on a gfx1250 box, then diff dense
  layers 0-2 + post_attn (clean signals) vs post_layer at MoE layers 3-60 (a4w4-vs-a8w4
  control). gfx950 side is DONE and ready.

Op notes (traps hit): (1) `pkill -f "sglang"` also matches the shell running it -> kills
your own command; use `pkill -9 -f "[l]aunch_server"` (regex bracket) so the pattern
doesn't match itself. (2) The two sitecustomize hooks can't both be named sitecustomize.py
on PYTHONPATH — combine via a single importer.

## E28. PARTIAL cross-node diff (gfx950 new dump vs gfx1250 STALE dump) — divergence is 100% MoE
Compared `hs_dump_gfx950.json` (E27, a4w4 MoE, bf16 absorb) vs `hs_dump_gfx1250.json`
(a8w4 MoE, bf16 absorb, STALE — only `input`/`post_layer`, no `post_attn`). Both 61 layers,
same layer ids, slice64[64] of the last token, same "Natalia" prompt. Metric: rel_l2 of the
slice64 (`||g950 - g1250|| / ||g1250||`). Chart: `crossnode_dump_compare.png` (script
`scripts/plot_crossnode_dump.py`).

| layer | zone  | input rel_l2 | post_layer rel_l2 |
|-------|-------|--------------|-------------------|
| 0     | dense | 0.0000       | 0.0025            |
| 1     | dense | 0.0031       | 0.0038            |
| 2     | dense | 0.0066       | 0.0057            |
| **3** | moe   | **0.0096**   | **0.2100**        |
| 4     | moe   | 1.38         | 0.28              |
| 10    | moe   | 2.37         | 0.64              |
| 30    | moe   | 1.30         | 0.87              |
| 60    | moe   | 0.69         | 0.71              |

**Findings:**
1. **Dense layers 0-2 (no MoE) match to the bf16 floor** (rel_l2 0.003-0.007, below the
   ~1e-2 TP floor). => the non-MoE residual path (attention, rmsnorm, rope, dense-MLP, and
   the now-bf16-on-both absorb) is **cross-node consistent** — end-to-end residual-level
   confirmation of E22's per-op result. No non-MoE bug.
2. **Divergence starts EXACTLY at layer 3** (the first MoE layer): its `input` is still
   matched (0.0096) but `post_layer` jumps to **0.21**, i.e. the divergence is injected by
   the MoE sublayer, then propagates (layer 4+ `input` inherits it, rel_l2 > 1) and compounds
   through depth. This is the expected a4w4-vs-a8w4 control, localized to its origin.
3. **Systematic magnitude difference:** gfx950 (a4w4) residual norm grows LARGER at depth
   (L50 input 20.5 vs 9.75; L60 post_layer 358 vs 319) — a4w4's bigger activation-quant error
   (E20: 15% vs a8w4 3.6%) injects more energy — yet a4w4 scores HIGHER (0.93 vs 0.85).
   Consistent with the quantization-matching thesis (the PTQ net's residual dynamics expect
   the a4w4 perturbation it was calibrated under).

**Conclusion:** the ONLY cross-node numerical divergence is the MoE, and it originates
strictly at the first MoE layer; everything non-MoE is consistent. This does NOT by itself
prove the MoE *scheme* causes the accuracy gap (E20 shows a8w4 is per-op *more* accurate) —
it proves the MoE is the only thing that makes the two nodes' states differ, and the gap must
therefore be a property of the a4w4-vs-a8w4 *scheme x calibrated-weights* interaction, not a
non-MoE bug.

**Limitation:** the gfx1250 dump is STALE (no `post_attn`), so at MoE layers 3-60 we cannot
check whether the attention sublayer diverges *independently* of the inherited MoE drift. The
dense-layer match strongly implies attention is clean, but a strict deep-layer check needs a
gfx1250 **re-dump with the updated rank-gated hook** (`scripts/hsdump_sitecustomize.py`), then
re-run `scripts/plot_crossnode_dump.py`.

## E29. Tried to run higher-precision-activation MoE on gfx950 — BLOCKED (no compatible kernel)
Goal: test the quant-matching thesis end-to-end by running the MoE with **higher activation
precision than a4w4** on gfx950 (a16w4 = bf16 act x fp4 weight, and/or a8w4 = fp8 act) and
comparing GSM8K vs the a4w4 0.93. Idea: a4w4 vs a8w4 differ ONLY in activation precision; the
fp4 weight matmul is the same.
- **Why not a pure-bf16 emulation:** dequantizing all 256 experts to bf16 = the full bf16
  model (~1.3 TB, infeasible; E16). And any path through the fp4-activation kernel re-quantizes
  the activation to fp4, destroying the >fp4 precision. So the only memory-feasible way to get
  >fp4 activation is a real kernel that keeps weights fp4 (a16w4 or a8w4).
- **Added an aiter experiment knob** (env-gated, reversible, default off): `fused_moe.py` after
  the gfx1250 block — `AITER_MOE_FORCE_ADTYPE` in {bf16,fp8,fp4} overrides `q_dtype_a`. With
  `=bf16` the MoE dtype signature correctly became `(bf16 act, bf16, fp4x2 weight)` = a16w4.
- **Result: FAULTS on gfx950** with `RuntimeError: Unsupported scales/output dtype!` from the
  aiter fused_moe op, in BOTH cuda-graph capture AND eager (first forward). => gfx950's MXFP4
  MoE only has a working kernel for the **a4w4 (fp4x2 activation)** path; the bf16/fp8-activation
  grouped kernels are not wired for R1's weight layout (weight scales are e8m0-shuffled for the
  fp4 path, incompatible with the a16w4/a8w4 kernel's expected scale/output dtype). Forcing fp8
  (a8w4) would hit the same wall (a8w4 grouped is gfx1250-flydsl-only).

**Implication:** running a8w4 (or a16w4) MoE on gfx950 for a clean A/B is **not possible with
existing kernels** — it genuinely requires writing/porting a kernel (the user's original idea).
BUT the motivation to write it is weak: E18 (a8w4 op-test == quant-ref @ 3e-6) and E20 (a8w4
per-op MORE accurate than a4w4) already show the gfx1250 a8w4 kernel is numerically correct, so
the gap is most likely the **scheme / cross-layer interaction**, not a fixable kernel bug — and
a universal a8w4 kernel would NOT help gfx1250 (it's already forced onto a8w4; the only more-
accurate option is a4w4, which gfx1250 A0 cannot run). A universal kernel's only value would be
diagnostic (run a8w4 on gfx950), which is a lot of work for a confirmation.

**Cheaper remaining diagnostics** (preferred over the kernel): (1) a **slow per-forward bf16
emulation** on a small Q count — dequant only the *active* experts per forward into a ~30 GB
bf16 scratch (fits 309 GB) and do a bf16 grouped GEMM with mxfp8-qdq (a8w4) vs mxfp4-qdq (a4w4,
validation) activations; hours for ~100 Q but decisive on the ~8-11% effect. (2) the **cross-node
dump** (E28) once gfx1250 is re-dumped with the new hook. The `AITER_MOE_FORCE_ADTYPE` knob is
left in aiter (harmless, default off) for a future box that has the a16w4/a8w4 MoE kernel wired.

## E30. bf16 MoE emulation on gfx950 — a8w4 SCHEME is accuracy-NEUTRAL (flips the leaning)
Ran diagnostic (1) above. Tool: `scripts/moe_emul_sitecustomize.py` (env `SGLANG_MOE_EMUL` in
{a4w4,a8w4,a16w4}). Mechanism (no repo edits): neuter the MoE weight/scale shuffles so weights
keep the clean fp4 layout, then replace `AiterRunnerCore.run` with a bf16 grouped FFN — per
active expert: dequant fp4->bf16, q-dq the activation per mode (mxfp4=a4w4, mxfp8=a8w4, none=
a16w4) on BOTH stages, silu(gate)*up, topk-weighted sum. Weights stay fp4 (dequanted per-forward;
batched over active experts + all-bf16 to fit memory). gfx950, TP2, **eager** (Python hook can't
run under cuda-graph), `--mem-fraction-static 0.70` + `PYTORCH_HIP_ALLOC_CONF=expandable_segments`
(the bf16 dequant scratch OOMs at 0.90). Slow (~4 tok/s batched); GSM8K 40 Q, parallel 40.

| MoE path (emulated, bf16 matmul, fp4 weight) | GSM8K 40Q | Invalid |
|----------------------------------------------|-----------|---------|
| a4w4 emul (mxfp4 activation) = VALIDATION     | **0.925** | 0.000   |
| a8w4 emul (mxfp8 activation) = the TEST       | **0.925** | 0.000   |
| (native gfx950 a4w4 kernel, ref)             | 0.93-0.945| 0.000   |
| (gfx1250 real a8w4 flydsl kernel, ref)       | ~0.85     | 0.000   |

**Findings:**
1. **Validation passed:** a4w4 emul (0.925) reproduces the native gfx950 a4w4 kernel (~0.93) =>
   the bf16 emulation + fp4 dequant + FFN structure are faithful; the framework is trustworthy.
2. **a8w4 emul == a4w4 emul == 0.925** (identical, same 40Q set, only the activation quant
   differs). => The **a8w4-vs-a4w4 MoE SCHEME / ideal numerics are accuracy-neutral.** The
   quantization-matching thesis (E25) is **FALSE for the MoE too**: making the MoE activation
   more precise (fp8) does NOT hurt vs a4w4.
3. **Therefore the gfx1250 gap is NOT the a8w4 scheme** — an *idealized* a8w4 (fp8 act x fp4
   weight, bf16 accumulate) scores 0.925, but the **gfx1250 real flydsl a8w4 kernel scores
   ~0.85** for the same scheme. So the shortfall is in the **gfx1250 real-kernel execution
   numerics**, not the scheme, attention, absorb, or serving integration.

**This FLIPS the E29 leaning.** Per E29's own decision rule ("gfx950 a8w4 stays ~0.93 => gfx1250
drop is a fixable kernel issue => writing/fixing the kernel is worthwhile"), the a8w4 kernel path
IS now the actionable target. Reconciles with E16 (the gfx1250 grouped a8w4 kernel's logits_diff
GROWS with token/max_m: 8 tok -> 3.4e-6 but 512 tok -> 5.4e-3 / 10% rel_l2) — a token-count-
dependent error that small-token op-tests (E18, 3e-6) miss but that accumulates over real long
GSM8K generations. The ideal a8w4 emul (E30) has no such growth, hence 0.925.

**Actionable next:** fix the gfx1250 grouped a8w4 kernel's large-token / max_m numerics (FlyDSL
`moe_grouped_gemm_mxscale_gfx1250.py` / `gemm_mxscale_gfx1250.py`) — the accumulation/tiling that
degrades at high token counts — OR write a correct a8w4 MoE kernel for gfx1250; either should
recover toward ~0.92. Validate any new kernel end-to-end (not just small-token op-test) against
this E30 emul (0.925) and the token-swept logits_diff, since the op-test at small tokens is blind
to the growth. (Caveat: 40Q is a spot-check; the internal a4w4-vs-a8w4 A/B is clean and both match
native, and the gap to gfx1250's 0.85 is far larger than 40Q noise. A 200Q emul rerun ~3 h each
would tighten it if desired.)

## E32-H21 (H21-18 node's own writeup). bf16 MoE emul RUN ON gfx1250 — the gap is NOT the MoE (case B)  [node: H21-18 gfx1250]
> NOTE (numbering dedup 2026-07-09): this is the H21-18 node's full writeup of the "bf16 MoE emul on
> gfx1250" run. It is the SAME experiment as **E32** (this-node summary) below; renamed from its
> original "E31" to remove the collision with the token-sweep E31. "E31" now uniquely = the
> token-sweep entry immediately below.
E30 inferred "gfx1250 real a8w4 kernel is the gap" from a gfx950 emul (0.925) vs gfx1250
real (0.85). E31 runs the SAME emulation directly ON gfx1250 (bypasses the real flydsl
kernel with ideal bf16 MoE) to test that inference. Setup: `scripts/moe_emul_sitecustomize.py`
(chunked per-32-expert dequant — see note), `--disable-cuda-graph`, mem-fraction 0.88,
GPU3 TP1, few_shot_gsm8k 40Q parallel 40.

| platform | real kernel | a8w4 emul (40Q) | a16w4 emul (40Q, no act quant, most ideal) |
|---|---|---|---|
| gfx1250 | ~0.85 (40Q) / 0.811 (1319Q) | **0.800** | **0.825** |
| gfx950 (E30) | ~0.93 | 0.925 | (not run; expect >=0.925) |

**Result: on gfx1250 even the MOST ideal MoE (a16w4 = bf16 weight x bf16 act, zero MoE
quant) scores only 0.825 ~= gfx1250 real 0.81 — NOT gfx950's 0.925.** a8w4 vs a16w4
(0.80 vs 0.825) is within 40Q noise. => **bypassing the real MoE kernel does NOT close the
gap** => the gap is **NOT the MoE** (kernel, scheme, AND activation-quant all excluded;
E30's "real kernel is the gap" is WRONG for gfx1250). This is CASE B (HANDOVER_gfx1250_moe_
emul.md): even ideal bf16 a8w4 degrades on gfx1250 => a new/fixed MoE kernel will NOT help.
=> The gap is a **gfx1250-specific NON-MoE effect** (attention/norm/rope/linear/sampling
run natively per-platform; the emul only swaps the MoE). Consistent with the emul faithfully
reproducing each platform's real score (gfx1250 emul~0.81==real; gfx950 emul~0.925==real).

CAVEATS (do not over-conclude):
1. 40Q noise (0.825 vs 0.925 = 4 questions). But the 1319Q real gap (0.811 vs 0.932) is the
   same ~0.12 and the emul tracks each platform's real number, so the direction is real.
2. The gfx1250 emul is the CHUNKED variant (edited from the E30 batched one for TP1 memory:
   the batched all-active dequant OOMs at ~25 GB with the ~353 GB bf16-dequant model). The
   math should be identical, but for a strict apples-to-apples the SAME chunked emul should be
   run on gfx950 (expect ~0.925). If gfx950-chunked also = 0.925, the platform (non-MoE) gap
   is confirmed; if gfx950-chunked drops too, suspect a chunked-emul bug.
3. This contradicts neither E22 (attention matches torch fp32 per-op) nor E28 (cross-node
   dump divergence starts at MoE layer 3) directly — those compared REAL a4w4-vs-a8w4 MoE.
   E31 says: with MoE made identical/ideal, gfx1250 still trails => the non-MoE residual
   difference (small per-op, but the whole gfx1250 forward vs gfx950) is what remains.

Next: (a) run the same chunked emul on gfx950 (apples-to-apples confirm); (b) if confirmed
non-MoE, bisect the non-MoE gfx1250 path more aggressively (whole-forward gfx1250-vs-gfx950
logit compare, not just per-op) — the per-op attention checks (E22) were vs torch, not vs
gfx950, so a systematic small gfx1250 attention/norm bias would pass E22 yet accumulate.

## E31. Token-swept a8w4 kernel probe — bisect off-by-one found & fixed, but END-TO-END NEUTRAL
node: gfx1250 / host ctheliosr-rck-g02-j19-10. (The H21-18 node's former "E31" above was renamed to
**E32-H21** to remove the collision; all cross-refs to "E31" in this file/STATUS mean THIS token-sweep.)
node: gfx1250 / host ctheliosr-rck-g02-j19-10 | sglang 000a61a2 | aiter 8815f4b5 | 2026-07-09

Goal (per E30's actionable): characterize the gfx1250 grouped a8w4 kernel's token/max_m-dependent
error and localize it, then fix and re-measure GSM8K.

**Setup / prerequisites re-applied on this fresh docker** (STATUS "3 code changes"; the tree
started clean at 000a61a2, aiter clean at 8815f4b5):
- sglang: `SGLANG_MOE_SHUFFLE_GFX1250` env-gated (16,16) weight shuffle in
  `quark_w4a4_mxfp4_moe.py` (MANDATORY; without it GSM8K=0.000 garbage — reconfirmed:
  flag off -> decode `爲爲爲...`, GSM8K 0.000/invalid 1.000; flag on -> " Paris...", 0.80).
- aiter: `AITER_GROUPED_FORCE_SPLIT_K1` env in `grouped_moe_gfx1250.py` (this fresh aiter
  lacked it -> warmup crashed with the §4.5 `IndexError` because CSV picks split_k=2 -> raw path).
- GSM8K baseline on this node (both cuda-graph): flag-on = **0.80 (40Q)** / **0.833 (eager 30Q)**.

**Token-sweep probe** (`op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario verify
--data-format a8w4 --experts 256 --topk 8 --model-dim 7168 --inter-dim 2048 --no-check-aot-cache
--tokens 8..1024`, `AITER_GROUPED_FORCE_SPLIT_K1=1` to match serving; needs `--no-check-aot-cache`
else it forces `FLYDSL_RUNTIME_RUN_ONLY=1` and misses the AOT cache):

| tokens | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|--------|---|----|----|----|-----|-----|-----|------|
| logits_diff (buggy) | 3.4e-6 | 3.4e-6 | **1.6e-3** | 1.7e-3 | 3.2e-3 | 2.0e-3 | **5.4e-3** | 5.2e-3 |
| rel_l2 (buggy)      | 0.26%  | 0.26%  | **5.6%**   | 5.9%   | 8.0%   | 6.3%   | **10.4%** | 10.2% |

**Localization (AITER_GROUPED_DEBUG=1):** the jump is at tokens 16->32 and coincides EXACTLY with
`grouped_contiguous_m` flipping **False->True** (routing picks contiguous-M when max_m crosses a
threshold: tok16 max_m=64 contiguous=False exact; tok32 max_m=256 contiguous=True error). tile
config is identical across all token counts (default tile_m=64/tile_n=256/tile_k=256/split_k=1,
CSV miss cfg_row=None). So it is NOT a config change — it is a bug in the **contiguous-M path**.

**Root cause + fix:** `kernels/gemm_mxscale_gfx1250.py:3010` per-tile expert-attribution binary
search over the `[0, batch_count]` inclusive layout-offset range (batch_count+1 values) used
`_bisect_iters = max(1, math.ceil(math.log2(batch_count)))`. For power-of-two batch_count (R1 =
**256 experts**) `ceil(log2(256))=8` under-counts by 1 -> search un-converged -> wrong expert on the
contiguous path. Fixed to `int(batch_count).bit_length()` (=9). (This IS STATUS "change #1"; it was
NOT present in this fresh aiter.) `rm -rf /root/.flydsl/cache` after editing.

**After fix:** op-test flat **3.4e-6 at ALL token counts** (32:3.40e-6, 128:3.41e-6, 512:3.46e-6,
1024:3.43e-6). The token-growth is entirely gone -> the E16/E30 "large-token kernel numerics" =
this bisect bug.

**END-TO-END RESULT: NEUTRAL.** GSM8K on this node WITH the bisect fix (split_k1 + shuffle +
cuda-graph, kernels recompiled after cache clear): **100Q = 0.830** — unchanged vs the 0.80/0.83
baseline. So **fixing the large-token kernel error does NOT move GSM8K.** This:
- **reconfirms E18** ("bisect fix real but end-to-end neutral"), and
- **overturns E30's actionable conclusion** (E30 said the real-kernel large-token numerics were the
  gap and fixing them should recover ~0.92 — they were fixed here and GSM8K did not move).

**New clean contradiction to resolve:** op-test now says the real fused MoE path (route+quant+
scatter+gemm) == the a8w4 quant-ref to 3e-6 at all token counts, AND E30 says an ideal a8w4 bf16
emul = 0.925 (40Q) — yet the real gfx1250 kernel = 0.83 (100Q). If the real kernel truly equals
ideal a8w4 to 3e-6, it "should" score ~0.925. It does not. => the 0.83<->0.925 gap is **NOT in the
MoE gemm numerics** (op-test clean) and is either (a) E30's 0.925@40Q being an optimistic/noisy
spot-check vs a proper large-Q number, or (b) a difference in the real serving path not covered by
either the op-test verify or the E30 bf16-emul.

**Decisive next (proposed):** on THIS gfx1250 node, same 100Q, run `scripts/moe_emul_sitecustomize.py`
(`SGLANG_MOE_EMUL=a8w4`, the E30 bf16 emul) vs the real flydsl kernel — same HW/everything, only
MoE impl differs. emul>>real => non-gemm real-kernel issue; emul~=real(~0.83) => the "to 0.93" gap
is scheme/measurement (E30's 0.925 was gfx950/40Q-optimistic) and gfx1250-vs-gfx950 baseline
comparability must be re-checked.

**Kept:** the bisect fix is a genuine correctness fix (10% -> 3e-6 on random inputs) and is left
applied in aiter (`gemm_mxscale_gfx1250.py:3010`).

## E32. bf16 MoE emul ON gfx1250 — even IDEAL MoE doesn't close the gap => gap is NON-MoE
node: gfx1250 (H21-18 SECOND node, GPU3, TP1) | 40Q
> This is the this-node SUMMARY of the same run that **E32-H21** (above) writes up in full detail.
> Kept both: E32-H21 = H21-18's original text, E32 = the summary + cross-node reconciliation.

Ran the E31 "decisive next" (E30-style `moe_emul_sitecustomize.py`) but ON a gfx1250 box: swap the
MoE for a bf16 FFN (weights dequant fp4->bf16, activation quant per mode), measure GSM8K 40Q:

| gfx1250 (GPU3, TP1, 40Q)                         | Accuracy        |
|--------------------------------------------------|-----------------|
| a8w4 real flydsl kernel                          | ~0.85 / 0.811 (1319Q) |
| a8w4 bf16 emul (mxfp8 activation)                | 0.800           |
| **a16w4 bf16 emul (bf16 activation = IDEAL, zero MoE quant)** | **0.825** |
| gfx950 bf16 emul (E30, batched)                  | 0.925           |

**Finding:** on gfx1250, even the fully ideal MoE (a16w4, no activation quant at all) = **0.825 ≈
real 0.81**, far below gfx950's 0.925. Making the MoE ideal does NOT recover the gap =>
**the gap is NOT the MoE** (real kernel [E31], scheme, or activation quant — all excluded) but
**gfx1250's NON-MoE platform execution** (attention / norm / rope / linear / sampling — the parts
the emul does NOT replace). Consistent with E31 (fixing MoE kernel numerics was end-to-end neutral).

**Reconciles the E31 contradiction and the cross-node dump:** the dump's dense-layer match was only
to the ~1e-2 TP2-vs-TP1 rel_l2 floor over 3 dense layers; a SMALL systematic non-MoE bias below that
floor, accumulated over 61 layers + long generation, is invisible per-layer but is the gap. E22's
attention check was "vs torch fp32", NOT "vs gfx950" — so a per-op-small but systematic gfx1250
attention/norm/rope bias was never excluded cross-node.

**CAVEAT (must resolve before trusting):** the gfx1250 emul here is the **chunked** variant (TP1
memory forces it; the batched E30 emul OOMs on one card), while gfx950's 0.925 used the **batched**
emul. So 0.825(chunked)-vs-0.925(batched) is not yet a clean apples-to-apples — the chunked emul
could itself be lossy. **Control needed: run the SAME chunked emul on gfx950 and confirm it is still
~0.925.** Only then is "gap = non-MoE" trustworthy.

**Next (agreed):**
1. [priority] gfx950 chunked-emul control -> expect 0.925 (rules out a chunked-emul bug).
2. whole-forward cross-node logits comparison (final logits, not per-op / not per-layer residual) to
   localize the systematic non-MoE gfx1250 bias (attention/norm/rope/linear/sampling).

## E33. gfx950 chunked-emul CONTROL — passes; gap is DECISIVELY non-MoE
node: gfx950 (reference node) | 40Q

Ran the SAME chunked `moe_emul_sitecustomize.py` (a8w4) on gfx950 that gfx1250 used in E32, to rule
out the chunked-vs-batched confound:
- **gfx950 chunked a8w4 emul = 0.950** (40Q, latency ~562 s ≈ 9.4 min) == E30 batched 0.925 (40Q
  noise) == native a4w4 ~0.93-0.945. So the chunked emul is faithful (NOT lossy).

**Clean apples-to-apples (identical chunked a8w4 bf16-MoE emul, only the platform differs):**

| chunked a8w4 bf16-MoE emul | GSM8K 40Q |
|----------------------------|-----------|
| gfx950                     | **0.950** |
| gfx1250                    | **0.800** |

The MoE math is now byte-identical bf16 on both nodes, so the **0.15 gap is 100% gfx1250's non-MoE
platform execution** (attention / norm / rope / linear / sampling). MoE (real kernel, scheme,
activation quant) is **fully excluded**. This closes E31/E32: the gap is NOT the MoE.

**Next = localize WHICH non-MoE op.** Refined localizer (better than final-logits): re-run the
cross-node per-layer residual dump (`hsdump_sitecustomize.py`) **with `SGLANG_MOE_EMUL=a8w4` on BOTH
nodes** — MoE identical (bf16) so it no longer diverges by construction, and every layer (incl. the
attention/norm of MoE layers 3-60) becomes comparable. The earliest layer/sublayer whose rel_l2
climbs above the TP floor localizes the offending non-MoE op. (Caveat unchanged: gfx950 TP2 vs
gfx1250 TP1 ~1e-2 floor; a 15%-accuracy non-MoE bias should exceed it somewhere.)

## E34. non-MoE per-op vs torch-fp32 probe (same-node, no TP floor) — built + validated
node: gfx1250 / host ctheliosr-rck-g02-j19-10 | sglang 000a61a2 | aiter 8815f4b5 | 2026-07-09

Backup localizer for E33 ("gap is non-MoE"): extend E22 (attention-only) to EVERY non-MoE op vs a
pure-torch fp32 reference, SAME-node (no TP2-vs-TP1 floor). Tool:
`scripts/nonmoe_probe_sitecustomize.py` (env `SGLANG_NONMOE_PROBE=1`, launch
`run_ds-r1_nonmoeprobe.sh`, EAGER). Each hook recomputes the fp32 ref from the SAME inputs
(snapshotted BEFORE the op — rmsnorm/rope mutate in place), records rel_l2 by op+shape:
rmsnorm->`forward_native`, rope->`forward_native`, linear->`x.float()@W.float().t()(+bias)`,
lm_head->`LogitsProcessor._get_logits` vs `h.float()@lm_head.weight.float().t()`, sampling->greedy
argmax(stored) vs argmax(fp32) disagreement + top1-top2 logit margin.

**Validated end-to-end** (Natalia prompt, 32 new tokens, coherent "...Half of 48 is 24. So she
sold"); all hooks fired (2700 records):

| op                                    | mean rel_l2 | note |
|---------------------------------------|-------------|------|
| rope[q], rope[k]                      | **0.0**     | bit-exact vs torch native |
| rmsnorm[d=512] (kv_a) / [d=1536] (q_a)| 1e-6 / 2e-6 | ~exact |
| rmsnorm[d=7168] (main)                | 0.0024      | bf16 noise |
| linear[* : o_proj/q/kv/gate/up/down]  | 0.0015-0.0017 | bf16 gemm noise |
| lm_head[vocab~129280]                 | 0.0017      | bf16 gemm noise |
| sampling argmax_disagree              | **0.0**     | greedy bf16==fp32, no flipped ties |
| sampling top1-top2 margin             | 4.4 (logits)| healthy, not tie-prone |

**Preliminary result:** every non-MoE op on gfx1250 matches its own torch-fp32 ref to bf16 noise
(~1e-3) or better; rope bit-exact; greedy stable. => **no non-MoE op is grossly WRONG on gfx1250**
(mirrors E22, now across rmsnorm/rope/linear/lm_head/sampling).

**Scope:** this is gfx1250-op-vs-gfx1250-torch — it catches a *broken* gfx1250 op (as the bisect bug
was for MoE) and finds none. It CANNOT catch a systematic gfx1250-vs-gfx950 platform bias where each
side is individually "correct vs torch" but differs cross-node (accumulation/rounding order). So the
E33 non-MoE gap is more likely such a systematic cross-node effect => the **E33 step-2 cross-node
emul-dump is the primary localizer**; this probe (ready, validated) is the backup that already rules
out a broken non-MoE op. Files: `scripts/nonmoe_probe_sitecustomize.py`, `run_ds-r1_nonmoeprobe.sh`.

## E36. DSv4-Flash GSM8K on gfx1250 = 0.925 — hardware + a8w4 MoE fine; R1 gap is R1-specific non-MoE
node: gfx1250 / host ctheliosr-rck-g02-j19-10 | sglang(/opt/venv serve) | aiter 8815f4b5(+edits) | 2026-07-09

User's sharp sanity check: "if the gap were a gfx1250 HARDWARE problem, DSv4 couldn't score well on
gfx1250." Measured it directly. Model `/dockerx/models/DeepSeek-V4-Flash` (DeepseekV4ForCausalLM, 43
layers, **fp8 w8a8** dynamic e4m3; `is_fp4_experts=True` -> experts are fp4, so with
`AITER_FORCE_A8W4=1` DSv4 uses the SAME a8w4 grouped MoE kernel as R1). Launch `run_v4_gfx1250.sh`
style on GPU3, `--attention-backend dsv4`, fp8 KV. Same GSM8K 5-shot completion bench as R1.

Traps hit + fixed:
- The stock `run_v4_gfx1250.sh` has **no coredump guard**; first launch faulted during cuda-graph
  capture (bs=200) and wrote a **95 GB `gpucore.*.gpu`** (deleted; disk restored). Added
  `ulimit -c 0` + `HSA_COREDUMP_PATTERN=/dev/null` + `AMD_COREDUMP=0`.
- DSv4 experts=fp4 -> a8w4 MoE hits the SAME split_k>1 raw-path fault as R1 at large bs; needed
  `AITER_GROUPED_FORCE_SPLIT_K1=1` (not in the stock v4 script).
- Even so, batched cuda-graph decode threw `hipErrorIllegalAddress` mid-GSM8K (DSv4-specific, likely
  dsv4-attn / fp8-KV at batch decode — unrelated to R1's gap). Worked around with **eager**
  (`--disable-cuda-graph`) + small batch (`--max-running-requests 16`, `--parallel 8`).

**Result: DSv4 GSM8K 40Q = 0.925, invalid 0.000** (coherent, correct: "...The answer is 72."). And
the other container has served DSv4 on gfx1250 GPU0 for hours. => **DSv4 scores well on gfx1250.**

**Implications (clean same-machine contrast, same bench):** DSv4=0.925 vs R1=0.80-0.83 on gfx1250.
- gfx1250 **hardware is fine** (a model scores 0.925). Confirms E34 / the user's argument.
- The **a8w4 grouped MoE kernel is fine** — DSv4 uses the exact same kernel and scores 0.925
  (independent confirmation of E30/E31/E33: MoE not the gap).
- DSv4 and R1 differ in the **non-MoE path**: DSv4 = `dsv4` attention backend + fp8 w8a8 linears +
  fp8 KV, calibrated fp8; R1 = triton MLA + mxfp4->bf16-dequant linears + bf16 KV, calibrated mxfp4.
  => the R1 gap lives in R1's **specific non-MoE path** (triton MLA attention and/or the mxfp4
  scheme running as bf16/a8w4), which DSv4 does not share. Consistent with E33 (gap is non-MoE) and
  narrows it to R1/MLA-specific execution, most plausibly a cross-platform (wave32-vs-wave64)
  accumulation difference in R1's triton MLA that DSv4's dsv4-backend attention avoids, OR a
  quant-scheme-matching effect specific to the mxfp4 checkpoint. Cross-node emul-dump (step 2)
  remains the localizer; focus attention-sublayer.

## E37. DSv4 cuda-graph batched-decode crash on gfx1250 — REPRODUCIBLE (DSv4-specific, not the R1 gap)
node: gfx1250 / host ctheliosr-rck-g02-j19-10 | 2026-07-09

Confirmed reproducible (per user request): DSv4-Flash launched WITH cuda-graph
(`run_v4_gpu3_safe.sh`: `--cuda-graph-max-bs 64`, `AITER_GROUPED_FORCE_SPLIT_K1=1`, coredump guards)
+ the crashing GSM8K command (`--parallel 32`, 40Q) -> **server dies with
`torch.AcceleratorError: CUDA error: an illegal memory access` (hipErrorIllegalAddress)**, from
`ExchangeDevice`. Second identical repro (first was E36's initial attempt).
- Trigger: `#running-req` climbs to **31-32** with **cuda-graph decode active (`cuda graph: True`,
  bs=32)** + interleaved prefills (swa mixed batch). 
- Control: eager + `--max-running-requests 16` + `--parallel 8` does NOT crash (that config gave the
  0.925 number in E36). => fault is specific to **DSv4 cuda-graph decode at batch >= ~32 with swa**,
  i.e. dsv4-attention/swa/graph — **DSv4-specific, unrelated to R1's accuracy gap**.
- Coredump guards held (no `gpucore.*.gpu`; disk stable). GPU3 freed cleanly after.
- Repro files: `run_v4_gpu3_safe.sh` (graph on) + `bench_sglang.py --parallel 32 --port 8100`;
  workaround for a usable DSv4 serve = eager + small batch (E36).

## E38. The DSv4 crash is CAUSED BY the E31 bisect "fix" — reverted (net-negative)
node: gfx1250 / host ctheliosr-rck-g02-j19-10 | aiter 8815f4b5 | 2026-07-09

Tested (user idea): revert the E31 edit in `gemm_mxscale_gfx1250.py:3010`
(`int(batch_count).bit_length()` -> back to original `math.ceil(math.log2(batch_count))`),
`rm -rf /root/.flydsl/cache`, and re-run the E37 crash repro (DSv4, cuda-graph ON, GSM8K
`--parallel 32`).

**Result: NO CRASH. DSv4 = 0.925, invalid 0.000, server survived** (same config that crashed 2x
with the fix present). => **the DSv4 `hipErrorIllegalAddress` was caused by the bisect "fix", not by
DSv4.** Mechanism (very likely): `bit_length(256)=9` iterations vs `ceil(log2 256)=8` — the extra
bisection step drives `mid` to the inclusive upper bound `batch_count` and does
`buffer_load(layout_rsrc, batch_count)` = an **out-of-bounds read** of the layout-offset buffer ->
illegal address. It faults under DSv4's high-concurrency contiguous-M prefill (bs~32); R1 tolerated
it (no crash in E31) but it was a latent OOB.

**Verdict on the bisect "fix" (STATUS "code change #1"): NET-NEGATIVE, keep it REVERTED.**
- It fixed only the contiguous-M op-test error (10% -> 3e-6) which E31 proved **end-to-end NEUTRAL**
  (R1 GSM8K unchanged 0.83 vs 0.80).
- It INTRODUCES an OOB read that crashes DSv4 (E37) and is a latent crash for R1 at high concurrency.
- => No accuracy benefit, real crash cost. Left reverted (`ceil(log2)`, original). A correct fix
  would need an OOB-safe bisect (clamp `mid`/`hi` to `batch_count-1`, or bound the buffer), but since
  the numeric error is end-to-end neutral it is not worth the risk right now.
- **Action item for STATUS/CHANGES:** demote change #1 from "recommended" to "do NOT apply as-is
  (causes DSv4 OOB crash); only the shuffle (#2) + split_k1 (#3) are needed.**
- **PARTLY SUPERSEDED by E39:** the MoE fix that #1 targets IS actually part of the real gap fix
  (its benefit was masked by the attention error, so E31's "neutral" was the masking confound). So
  the fix is *needed* — but the `bit_length` **implementation** has the OOB crash and must be
  reimplemented OOB-safe (clamp `mid`/`hi` to `batch_count-1`) before use. Keep reverted until then.

## E39. BREAKTHROUGH — the gap is TWO gfx1250 real kernels (MoE + attention) that MUTUALLY MASK
node: gfx1250 (SECOND node) | reported to primary | 40Q

The other gfx1250 box ran the decisive combo: idealize the MoE (bf16 emul) AND the attention
(torch-naive softmax over paged KV, i.e. torch SDPA reference) **independently and together**:

| gfx1250 config                                  | GSM8K |
|-------------------------------------------------|-------|
| real MoE + real attention (baseline)            | ~0.80 |
| idealize MoE ONLY (bf16 emul)                    | ~0.825 |
| idealize ATTENTION ONLY (torch softmax/SDPA)     | ~0.825 |
| idealize BOTH                                    | **0.925 (== gfx950)** |

**Conclusion: the ~0.10-0.12 gap is caused by TWO real gfx1250 kernels EACH injecting error — the
FlyDSL a8w4 grouped MoE (large-token, E16) AND the Triton MLA softmax attention (decode/extend) —
and they MUTUALLY MASK: fixing only one leaves the other, so accuracy barely moves (0.80->0.825).
Fixing BOTH recovers 0.925.**

This resolves every prior "innocent" verdict as the masking confound:
- **E31** ("MoE fix end-to-end neutral") — neutral only because the attention error still dominated.
  The MoE fix IS real and needed; it just can't show alone.
- **E33/E36** (idealize MoE only -> 0.825) — same masking (attention still real).
- **E22/E34** ("attention matches torch to 0.16%, clean") — WRONG as an end-to-end verdict: that
  0.16%/per-op **accumulates** over the long CoT and, once the MoE is also idealized, is worth ~0.10.
  Per-op-vs-torch (same-node) genuinely cannot see an accumulating systematic bias — exactly the E34
  scope caveat.
- Combo still used the Triton **qk-rmsnorm** and still hit 0.925 => **qk-norm is fine**; the
  attention problem is specifically in the **softmax decode/extend kernel** (triton MLA), not qk-norm.

**Actionable (fix BOTH gfx1250 real kernels; MUST validate end-to-end — small-token op-tests are blind):**
1. **Triton MLA softmax attention (decode + extend) numerics** — reference = torch SDPA over paged KV
   (torch-naive recovers it). Likely a softmax/reduction precision or wave32 accumulation issue.
2. **FlyDSL a8w4 grouped MoE large-token numerics** (E16) — bf16 emul recovers it; needs a correct
   real-kernel fix (the `bit_length` bisect targets this but is OOB-crashy per E38 — reimplement
   OOB-safe, or find the true large-token accumulation issue).
- Fix one -> ~0.825; fix both -> ~0.925.
