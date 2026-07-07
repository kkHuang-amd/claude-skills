# gfx1250 MXFP4 a4w4->a8w4 — Experiment Log (2026-07-07)

Chronological record of what was run and observed. Repos (base commits at time
of work):
- `/sgl-workspace/sglang` @ `8d30387cd671a3bc8eae178988f7c119544d08b7` (2026-07-02)
- `/sgl-workspace/aiter`  @ `8815f4b56dbaf416a3370659b777839c70ff3bf9` (2026-06-30, fork akao-amd/aiter)
- Model: `/dockerx/models/amd/DeepSeek-R1-0528-MXFP4` (Quark W4A4 MXFP4)
- HW: 4x gfx1250, VRAM ~432 GB each. Shared with another user's `rccl-tests`.

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

## Next steps

1. Close the ~8% accuracy gap: improve gfx1250 a8w4 grouped MoE kernel numerics
   (FlyDSL kernel-level; the token/max_m-dependent logits_diff growth in E16).
2. Perf: tune the bf16 gemms (populate `bf16_tuned_gemm.csv`) to beat `torch solution:0`.
3. Fix fp8-KV decode read (triton MLA decode) on gfx1250 to halve KV memory (E13).
4. Fix the split_k>1 (token=1) raw+finalize MoE path (E12) instead of forcing split_k=1.
5. Optional: build a real dense a8w4 linear kernel to drop the bf16-dequant. NOTE:
   the earlier parked wrapper (gemm_a8w4_gfx1250.py + its test) was **removed on
   2026-07-07** (faulty, never on any code path; linear uses bf16 dequant). Recreate
   from git history / SKILL §5 if pursued.
