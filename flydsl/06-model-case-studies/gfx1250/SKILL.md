---
name: gfx1250-mxfp4-a4w4-to-a8w4
description: Run an MXFP4 (Quark W4A4) checkpoint such as DeepSeek-R1-0528-MXFP4 on AMD gfx1250, which lacks the fp4-activation WMMA scale instruction V_WMMA_SCALE_F32_32X16X128_F4. Use when a4w4 (fp4-activation x fp4-weight) GEMMs cause "Memory access fault ... SQC (inst)" GPU page faults on gfx1250, when converting a4w4 compute to a8w4 (fp8-activation x fp4-weight), when SGLang + aiter MXFP4 MoE / linear layers crash on gfx1250, or when the aiter triton gemm.basic fp4 kernels fault on this arch. Captures which code paths use a4w4, the AITER_FORCE_A8W4 MoE switch, the bf16-dequant linear workaround, the (unfinished) dense flydsl a8w4 gemm, and operational traps (GPU wedging, 260GB GPU coredumps filling the disk). ALSO covers the full working serving recipe reached on 2026-07-07: the aiter fused_qk_rmsnorm JIT build failure (rope_common.h / ck_tile vec_convert.h) worked around with a Triton RMSNorm; the FlyDSL grouped a8w4 "IndexError: tuple index out of range" fixed by adding the missing swiglu_limit arg to the stage1 raw gemm launch; the split_k>1 raw+finalize path GPU illegal-address fixed by forcing split_k=1 (AITER_GROUPED_FORCE_SPLIT_K1); the decode-only "every token is 0" degeneration root-caused to the fp8_e4m3 KV cache (use bf16 KV / --kv-cache-dtype auto); and CUDA graph capture (previously crashing) now working, giving GSM8K ~0.85.
---

# gfx1250: a4w4 -> a8w4 for MXFP4 checkpoints (DeepSeek-R1-0528-MXFP4 on SGLang+aiter)

Investigation playbook. Target: serve `DeepSeek-R1-0528-MXFP4` (Quark **W4A4**
MXFP4) with SGLang + aiter on a single **gfx1250** GPU. This card does **not**
implement `V_WMMA_SCALE_F32_32X16X128_F4` (the fp4-activation scaled-WMMA), so
any **a4w4** (fp4 act x fp4 weight) GEMM crashes. The goal is to convert a4w4
compute to **a8w4** (fp8 act x fp4 weight), which uses the fp8 scaled-WMMA
(`...16X128_F8F6F4`) that the hardware does support.

Environment at time of writing:
- Machine: 4x gfx1250 (VRAM ~432 GB each), ROCm, `WAVE_SIZE=32`.
- Repos (local, editable): `/sgl-workspace/sglang`, `/sgl-workspace/aiter`
  (fork `akao-amd/aiter`, ~2026-06-30).
- Launch script: `/sgl-workspace/sglang/run_ds-r1.sh` (HIP_VISIBLE_DEVICES, TP=1,
  triton attention backend, fp8 kv cache).
- Shared box: another user runs `rccl-tests` on GPUs 0-3 concurrently — be
  careful, GPU faults are disruptive (see §7).

---

## 1. Crash signature (a4w4 hitting the missing instruction)

- `dmesg`: `amdgpu ... no-retry page fault ... Faulty UTCL2 client ID: SQC (inst)`,
  `PERMISSION_FAULTS: 0x3`, from the python process. `SQC (inst)` = the
  **instruction cache** faulting = the shader tried to execute an instruction the
  HW does not implement (the fp4 scaled-WMMA).
- Process-level: `Memory access fault by GPU node-N ... Reason: Page not present`,
  then `Fatal Python error: Aborted` / exit code -6.
- A prior a4w4 crash can leave the GPU **wedged**: a fresh process then **hangs**
  at the very first HIP op (`torch.ones(device='cuda')` inside
  `distributed/parallel_state.py:__init__`, py-spy shows it stuck spinning at
  ~100% CPU on the first tensor allocation). The wedge is transient — it clears
  after the dead process is reaped, but can look like an init hang for many
  minutes.

Confirm arch + that the model is W4A4:
```bash
python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"  # gfx1250
python3 -c "import json;c=json.load(open('/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4/config.json'));print(c['quantization_config']['global_quant_config'])"
# weight dtype=fp4 (static, per_group/32/e8m0), input_tensors dtype=fp4 (dynamic, per_group/32/e8m0) => W4A4
```

---

## 2. Where a4w4 is used (SGLang Quark MXFP4 dispatch)

`python/sglang/srt/layers/quantization/quark/quark.py` selects schemes:
- `_is_mx_fp4(weight, input)` true (both fp4 per_group/32/e8m0) ->
  - **Linear**: `QuarkW4A4MXFP4` (`schemes/quark_w4a4_mxfp4.py`)
  - **MoE**:    `QuarkW4A4MXFp4MoE` (`schemes/quark_w4a4_mxfp4_moe.py`)
- `_is_mx_w4a8(...)` true (fp4 weight + **static per-tensor fp8** input) ->
  `QuarkW4A8MXFp4MoE` — the existing a8w4 MoE scheme, but it needs a checkpoint
  that already carries static fp8 input scales. Our checkpoint is fp4-dynamic, so
  this scheme is **not** selected.

The DeepSeek-R1-0528-MXFP4 `exclude` list excludes **all** `self_attn.*` proj
(q_a/q_b/kv_a/kv_b/o_proj) and `lm_head` and layer 61 -> those run bf16. So the
fp4 (a4w4) surface is:
- **MoE experts** (layers 3-60) via `QuarkW4A4MXFp4MoE` -> aiter `fused_moe(quant_type=PER_1X32)`.
- **Dense-MLP linears** (layers 0-2 gate/up/down) + any non-excluded linear via
  `QuarkW4A4MXFP4.apply_weights` -> `dynamic_mxfp4_quant(x)` + `gemm_afp4wfp4` (fp4xfp4).

Linear `apply_weights` also has fused tuple-input paths
(`gemm_afp4wfp4_pre_quant`, `fused_gemm_afp4wfp4_split_cat`) — these are only used
by MLA attention (`deepseek_common/.../forward_mha.py`, gated on
`kv_b_proj.weight.dtype==uint8` and `SGLANG_AITER_FP8_PREFILL_ATTN`), which are
excluded/bf16 here, so only the plain-tensor path matters for this checkpoint.

---

## 3. gfx1250 kernel reality (measured, important)

All of aiter's **triton `gemm.basic` fp4 kernels memory-fault on gfx1250** even
with correct inputs on a healthy GPU (they target gfx950):
- `aiter.ops.triton.gemm_afp4wfp4` (a4w4) — faults
- `aiter.ops.triton.gemm.basic.gemm_a8wfp4` (a8w4) — faults
- `aiter.ops.triton.gemm.basic.gemm_a16wfp4` (bf16xfp4) — faults
- (`gemm_a16w16` bf16 works fine — the faults are fp4-kernel specific, not the GPU)

The **working** fp4/a8w4 path on gfx1250 is **flydsl / gluon**, and only the
**grouped (MoE)** form is exercised/tested upstream:
- `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py::_maybe_grouped_gfx1250_a8w4_moe`
- `aiter/aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py`
- kernel: `aiter/aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py`
  (`compile_mxscale_gemm` / `compile_a8w4_gemm`, WMMA 16x16x128).

The **dense (non-grouped)** launcher of the same kernel exists
(`launch_mxscale_gemm(c,a,b,a_scale,b_scale,M,N,stream)`, `grouped_masked_m=False`,
`batch_count=1`) but is **never called anywhere in aiter/mainline** — every real
caller sets `grouped_masked_m=True`. This dense path is unvalidated and **faults**
(see §6).

Quant/activation dtype note: on gfx1250, `aiter.fused_moe` maps `per_1x32` weight
to either fp4x2 or fp8 activation via `AITER_FORCE_A8W4`:
```python
# aiter/aiter/ops/flydsl -> aiter/fused_moe.py (~line 435)
if get_gfx() == "gfx1250":
    if os.environ.get("AITER_FORCE_A8W4","0") in ("1"):
        q_dtype_a = dtypes.fp8     # a8w4 (MXFP8 act x MXFP4 weight) -> grouped a8w4 kernel
    else:
        q_dtype_a = dtypes.fp4x2   # a4w4 -> needs the missing instruction -> crash
```

---

## 4. THE FIXES

### 4.1 MoE -> a8w4 (env only, no code)

Set `AITER_FORCE_A8W4=1`. aiter's `fused_moe` then quantizes MoE activations to
MXFP8 and routes to the flydsl grouped a8w4 kernel (`_maybe_grouped_gfx1250_a8w4_moe`).
Already added to `run_ds-r1.sh`. NOTE: still needs runtime validation — a forward
fault was observed during the first decode/warmup pass and was not yet root-caused
before GPUs went away (see §8).

### 4.2 Linear -> bf16 dequant (code, chosen interim)

There is no working dense a8w4 gemm on gfx1250 yet (§3, §6). Because the a4w4
linear surface here is tiny (dense-MLP layers 0-2; shared experts fuse into the
MoE) and VRAM is ~432 GB, the pragmatic fix is to **dequantize the FP4 linear
weights to bf16 at load** and run a plain `F.linear`.

Implemented in `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`,
gated by the same `AITER_FORCE_A8W4` flag:
- module flag `_dequant_linear_to_bf16 = _is_hip and get_bool_env_var("AITER_FORCE_A8W4","false")`
- helper `_dequant_mxfp4_to_bf16(weight_u8[N,K//2], scale_e8m0[N,K//32]) -> bf16[N,K]`
  (unpack 2 fp4/byte via the e2m1 LUT, multiply by `2^(e8m0-127)` per 32-group).
- `process_weights_after_loading`: if flag, replace `layer.weight` with the bf16
  dense weight and drop `weight_scale`, set `layer.dequantized_bf16=True`.
- `apply_weights`: if `layer.dequantized_bf16`, `return F.linear(x, layer.weight, bias)`
  (handles the plain-tensor path; tuple/fused paths are MLA-only and excluded here).

Result: model **loads fully** (353 GB) and the server **starts** (Uvicorn up,
`/model_info` 200) in eager mode. This is the current known-good baseline for the
linear side.

### 4.3 Run script knobs

`/sgl-workspace/sglang/run_ds-r1.sh`:
- `HIP_VISIBLE_DEVICES=<healthy gpu>` (GPU 0 was wedged from a prior crash; use another)
- `AITER_FORCE_A8W4=1`   (MoE a8w4 + triggers the linear bf16-dequant + the Triton qk-rmsnorm, §4.4)
- `AITER_GROUPED_FORCE_SPLIT_K1=1`  (§4.6 — avoid the split_k>1 raw+finalize MoE path)
- `--kv-cache-dtype auto`  (§4.7 — bf16 KV; fp8_e4m3 KV breaks decode on gfx1250)
- `ulimit -c 0` + `HSA_COREDUMP_PATTERN=/dev/null` + `AMD_COREDUMP=0`  (see §7)

### 4.4 aiter fused_qk_rmsnorm won't build -> Triton RMSNorm (code)

With `SGLANG_USE_AITER=1`, `deepseek_common/.../forward_mla.py` imports aiter's
`fused_qk_rmsnorm` at module load (bound to the MLA q/k RMSNorm call at
`forward_mla.py` ~L317, `elif _use_aiter:`). On gfx1250 that triggers a JIT build
of `module_fused_qk_norm_rope_cache_quant_shuffle` which **fails to compile**:
`csrc/kernels/rope/rope_common.h` and `csrc/include/ck_tile/vec_convert.h` use
`ck_tile::fp16_t` / `ck_tile::type_convert` / `vec_t` / `vector_traits` that this
docker's composable_kernel (submodule `af7118e34`, therock-7.10-704) does not
expose in scope. There is no prebuilt `.so`, so the on-demand build fails and
kills the scheduler during init (wiping `aiter/jit` and rebuilding does NOT help —
it's a source/CK mismatch, not a stale cache).

IMPORTANT dispatch fact (triton backend, `attention_backend_handler.py`):
- **prefill** (extend, no prefix) -> `AttnForwardMethod.MHA` (`forward_normal`,
  does NOT call fused_qk_rmsnorm).
- **decode** -> `AttnForwardMethod.MLA` (`forward_absorb_prepare` -> our qk-rmsnorm).
So this norm path (and the fix below) is **decode-only**; a correct prefill does
not validate it.

FIX: a self-contained Triton RMSNorm drop-in, gated by `AITER_FORCE_A8W4`:
- new file `python/sglang/srt/models/deepseek_common/attention_forward_methods/triton_qk_rmsnorm.py`
  (`fused_qk_rmsnorm_triton(q,q_w,q_eps,k,k_w,k_eps)->(q_out,k_out)`, matches
  `RMSNorm.forward_native`: fp32 var, `rsqrt(var+eps)`, `*weight`, cast back; plus
  a `fused_qk_rmsnorm_torch` pure-torch reference used for bisection).
- `forward_mla.py`: under `_use_aiter and get_bool_env_var("AITER_FORCE_A8W4")`,
  bind `fused_qk_rmsnorm_bf16` to the Triton version instead of the aiter import
  (env `SGLANG_QK_RMSNORM_TORCH=1` selects the torch reference — used only to prove
  the kernel is not the cause of any bug; the Triton path is the default).
Triton output is bit-parity with torch RMSNorm (0 diff small M, 1 bf16 ULP large M).

### 4.5 FlyDSL grouped a8w4 "IndexError: tuple index out of range" (aiter code)

Once the server boots and the first MoE forward runs, `AITER_FORCE_A8W4=1` routes
MoE through `aiter/ops/flydsl/grouped_moe_gfx1250.py::_maybe_grouped_gfx1250_a8w4_moe`
-> `moe_grouped_gemm_mxscale_gfx1250.py::launch` (stage1). On the **raw (non-fused)**
gemm path it crashed with `IndexError: tuple index out of range` at
`<flydsl-dispatch>:65` (flydsl `jit_executor` fast-dispatch), on the *second*
forward (first call compiles+runs via the slow path applying Python defaults;
second call uses the strict positional fast-dispatch).

Root cause: the compiled masked host entry `launch_mxscale_gemm_masked[_bias]`
(`gemm_mxscale_gfx1250.py`) has **13** runtime params (trailing
`swiglu_limit_f=inf`), but stage1's raw `_run_compiled(_get_raw_base()/...bias(), ...)`
call sites passed only **12** (omitting the swiglu arg). **stage2's** raw calls
correctly pass `_no_act_swiglu_lim` — it is the reference pattern.

FIX (aiter `moe_grouped_gemm_mxscale_gfx1250.py`): append `_swiglu_lim_rt` to the 6
stage1 raw `_run_compiled` calls (persistent/contiguous/dense x bias/no-bias). All
6 end with `2 * cfg.inter_dim,\n stream,\n )`; add `_swiglu_lim_rt,` before the `)`.

### 4.6 split_k>1 raw+finalize GPU illegal-address -> force split_k=1 (aiter code/env)

After 4.5, the next fault was a real GPU `hipErrorIllegalAddress` in kernel
`moe_stage1_finalize_act_silu_bf16_e257_m16_i2048_gguu_v4_**sk2**`. The tuned CSV
(`_find_grouped_config`) picks `split_k1=2` for the DeepSeek dims; `split_k>1` makes
`_get_fused_base()` return None -> `use_fused_gemm=False` -> the **raw gemm +
separate finalize/split-k-reduce** path runs (the one with 4.5's bug and this
finalize fault). The op-tests use small dims with default `split_k1=1` -> the
**fused** stage1 (activation in epilogue, no finalize kernel) -> validated. So the
raw+finalize+split-k path is simply unvalidated/broken on DeepSeek shapes.

FIX: added env `AITER_GROUPED_FORCE_SPLIT_K1` in `grouped_moe_gfx1250.py` (right
after the CSV override) that forces `split_k1=split_k2=1`, routing stage1/stage2 to
the fused path. Set `AITER_GROUPED_FORCE_SPLIT_K1=1`. (Do NOT bother forcing
`AITER_GROUPED_DEEPGEMM_CONTIGUOUS`; both the contiguous and dense raw branches hit
4.5, and the real fix is split_k=1 -> fused.)

### 4.7 decode emits token 0 every step -> fp8 KV cache is the culprit (config)

Symptom after 4.4-4.6: server serves, **prefill is numerically correct** (first
token good, e.g. "The capital of France is" -> " Paris"), but **every decode step
emits token 0** (`output_ids=[<good>,0,0,...]`), so GSM8K = 0.000 / invalid 1.000.
Bisection: swapping the qk-rmsnorm to the torch reference did NOT change it (rules
out 4.4). The decode-only difference vs the working MHA prefill is the MLA-absorb
attention **reading the accumulated KV cache**, which was **fp8_e4m3**
(`--kv-cache-dtype fp8_e4m3`).

FIX (interim, 2026-07-07): use bf16 KV cache -> `--kv-cache-dtype auto`. Decode becomes
coherent and GSM8K jumps to ~0.80 (eager) with invalid 0.000.

**UPDATE 2026-07-10 (E43, node H21-18): fp8 KV cache is now FIXED and usable — this §4.7
"fp8 KV is broken, use bf16" is SUPERSEDED.** Root cause (pinned): on gfx1250 triton
`tl.dot(fp8, fp8)` returns GARBAGE (~1e34) for contraction dim **K >= 128** (K=64 fine;
bf16 fine at all K) — a gfx1250-specific fp8-MFMA codegen bug (gfx950 fp8 dot at K=512 is
correct, scores 0.941). The MLA nope QK dot is K=512, and the triton MLA kernels downcast
q to fp8 to match an fp8 KV cache (`q.to(K_Buffer.dtype)`) -> fp8xfp8 K=512 -> garbage ->
softmax(inf) -> NaN -> degenerate decode. Two spots: `decode_attention.py::_fwd_grouped_
kernel_stage1` (decode) and `extend_attention.py` prefix loops (fires when radix cache
reuses a prefix — the blind spot behind "prefill never reads the fp8 cache"). FIX: keep q
in bf16 and upcast the fp8 K to bf16 in the dot (`tl.dot(q, k.to(q.dtype))`); no-op for
bf16 KV. Validated: fp8 KV GSM8K 1319Q **0.949** == bf16 0.951, full speed, full recipe
(radix + cuda-graph, no workaround flags). Halves KV memory. See EXPERIMENT_LOG E43 +
results_gfx1250-H21-18.md; upstream repro in artifacts/triton_fp8_dot_largek_gfx1250_repro.py.
**gfx1250 DEV RULE: never `tl.dot(a_fp8, b_fp8)` with contraction K>=128 — upcast to bf16,
tile K<=64, or use a validated scaled-fp8 gemm. bf16 tl.dot is unaffected.**

### 4.8 CUDA graph now works (previously crashed)

EXPERIMENT_LOG E6 recorded a crash during **decode CUDA-graph capture (bs=32)**.
With 4.4-4.7 applied, capture of all decode batch sizes (bs=32..1) **succeeds with
no fault**, decode under cuda-graph is correct (`cuda graph: True`), and GSM8K holds
(~0.85) while running ~4-6x faster than eager (40 Q: 57 s vs 359 s; 70 vs 16 tok/s).
=> drop `--disable-cuda-graph` and `AMD_SERIALIZE_KERNEL=3` for real serving; keep
them only when diagnosing a new async GPU fault. NOTE: under capture mode
`get_is_capture_mode()` is True, so `forward_absorb_prepare` takes the alt-stream +
torch `q_a_layernorm`/`kv_a_layernorm` branch (not the Triton qk-rmsnorm) — both are
verified good.

### 4.9 Two triton-dtype issues in the MLA attention, and what class each is (E36/E43)

Two separate fixes live in `decode_attention.py` / `extend_attention.py`. They look similar
(both about a dtype in a `tl.dot`) but are FUNDAMENTALLY different classes — do not conflate:

- **fp8 KV read = a gfx1250 CODEGEN BUG (a broken instruction).** `tl.dot(fp8, fp8)` on gfx1250
  returns ~1e34 GARBAGE for contraction dim **K >= 128** (K=64 fine; bf16 fine at all K). The MLA
  nope dot is K=512; the kernels downcast q to fp8 to match an fp8 KV cache -> garbage -> NaN ->
  degenerate decode (§4.7 UPDATE). FIX = keep q bf16, upcast the fp8 K to bf16 (`tl.dot(q,
  k.to(q.dtype))`). gfx1250 DEV RULE: never `tl.dot(a_fp8,b_fp8)` with K>=128. Repro/report in
  `artifacts/triton_fp8_dot_largek_gfx1250_repro.py`.

- **FIX A (fp32 P·V) = a PRECISION choice, NOT codegen.** The original code downcast the softmax
  weights `p` to bf16 before `tl.dot(p, v)`. The gfx1250 bf16 P·V `tl.dot` is numerically normal
  (rel_l2 ~2.4e-3, stable across K=16..512, no garbage — E43), so this is not a broken instruction;
  it is ordinary bf16 rounding of `p` (~0.2-0.4%/weight) that accumulates over long CoT. Keeping p
  fp32 (`tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)`) was credited with ~0.85 -> ~0.925.
  **⚠️ BUT on THIS image FIX A is accuracy-neutral — it is NOT the lever (E43 A/B).** Single-var
  A/B on gfx1250 (toggle only P·V p dtype, else identical, bf16 KV, prod recipe): p-fp32 (on) =
  1319Q 0.951; p-bf16 (off) = 1319Q **0.948** (200Q 0.965 vs 0.960) — reverting FIX A does NOT drop
  accuracy. So the historical 0.85->0.925 attribution (E36/E37/E39) was a CONFOUND (that saga's
  masking/emul) or was fixed by the newer sglang(3923a34d)/aiter(9af05b91). This also dissolves the
  gfx950 puzzle (why gfx950 "didn't need" FIX A): nobody needs it on this stack. FIX A is left in
  place (neutral + fp32 P·V is safe; droppable for a small perf gain). The fp8-KV upcast fixes above
  are the ones that actually matter. See STATUS.md + results_gfx1250-H21-18.md.
  NOTE — the A/B was run on **bf16 KV** (the correct carrier: `v` is bf16 so FIX-A-off = `p.to(bf16)`
  cleanly isolates the p fp32-vs-bf16 precision variable). **On fp8 KV the P·V dot MUST stay fp32:**
  there `v = tl.trans(k)` is fp8, so an FIX-A-off `p.to(v.dtype)` would downcast p to fp8 and, since
  the P·V contraction (seq) can be >=128, hit the fp8 `tl.dot` K>=128 garbage bug (§4.7/§4.9 above).
  So the fp32 P·V form is REQUIRED on fp8 KV — not for accuracy, but to avoid the fp8 codegen bug.

---

## 5. a8w4 mxscale GEMM recipe (for a real dense kernel / reference)

MXFP8 activation x MXFP4 weight, e8m0 block scales, via `compile_a8w4_gemm`
(`kernels/gemm_mxscale_gfx1250.py`, `scale_mode="mxscale"`):
- **A (activation)**: FP8 e4m3 `(M,K)` + e8m0 per-1x32 scale `(M,K//32)` from
  `aiter.ops.triton.quant.dynamic_mxfp8_quant(x, quant_dtype=dtypes.fp8)`.
  A-scale must be preshuffled to the warp-tile layout
  (`grouped_moe_gfx1250._grouped_a8w4_preshuffle_e8m0_scale`), **but is identity
  when `tile_m//m_warp == 16`** (wmma_rep=1) — so pick `tile_m=16, m_warp=1` to
  skip A-scale reshuffling entirely.
- **B (weight)**: FP4 packed `(N,K//2)` uint8 passed **raw** (the kernel TDM
  descriptor handles WMMA tiling — no weight preshuffle). B-scale e8m0
  `(N,K//32)` -> n32k4 via `aiter.ops.shuffle.shuffle_scale_n32k4(s.view(1,N,K//32))`
  -> `(N//32,(K//32)*32)`.
- ptpc mode (`scale_mode="ptpc"`, per-token fp32 A + per-channel fp32 B) is
  simpler but needs per-channel weight scales; our checkpoint is per-group mxfp4,
  so mxscale is required.
- Dense launcher: `launch = compile_a8w4_gemm(N=N,K=K,tile_m=16,tile_n=128,tile_k=128,
  m_warp=1,n_warp=2,num_buffers=2,out_dtype='bf16',grouped_masked_m=False,batch_count=1)`;
  then `_run_compiled(launch, y, a_fp8, w_packed_u8, a_scale, b_scale_n32k4, M, N, stream)`.

Constraints: `K%tile_k==0`, `N%tile_n==0`, `tile_k%128==0`, `K//tile_k >= num_buffers`.

---

## 6. Dense a8w4 gemm attempt — status: DROPPED (superseded by bf16 dequant)

Originally implemented `aiter/aiter/ops/flydsl/gemm_a8w4_gfx1250.py`
(`run_gemm_a8w4_gfx1250`, `preshuffle_a8w4_weight_scale`) + unit test
`aiter/op_tests/test_gemm_a8w4_gfx1250.py`, following §5, to give a dense
(non-grouped) a8w4 GEMM for the plain `nn.Linear` layers.
- All the Python prep was **verified correct** (mxfp8 quant, A-scale identity,
  B-scale n32k4 shapes all matched — tested without launching the gemm).
- The **gemm launch itself memory-faulted** on gfx1250: the dense
  (`grouped_masked_m=False`) codepath of `gemm_mxscale_gfx1250` is unexercised
  upstream and buggy.
- **DROPPED (2026-07-07):** the linear layers use the bf16-dequant path (§4.2,
  `F.linear` = a16w16) instead, which is correct and cheap for the tiny linear
  surface. The dense a8w4 wrapper was never on any code path (only its own test
  imported it), so `gemm_a8w4_gfx1250.py` and `test_gemm_a8w4_gfx1250.py` were
  **removed** from the aiter tree/commit to avoid dead, faulting code. Recreate
  from git history / §5 if a real dense a8w4 kernel is ever needed.

FlyDSL skills for kernel work: https://github.com/ROCm/FlyDSL/tree/main/.claude/skills
(`gemm-optimization`, `flydsl-kernel-authoring`, `debug-flydsl-kernel`).

---

## 7. Operational traps (cost real time here)

- **260 GB GPU coredumps fill the disk.** On any GPU fault, ROCr tries to core
  dump; `/proc/sys/kernel/core_pattern` pipes to `systemd-coredump` which is
  absent in the container, so it falls back to writing `gpucore.<pid>.gpu` in the
  process cwd — **hundreds of GB** (one was 260 GB), filling `/` (overlay) to
  100% and then cascading failures ("No space left on device", GPU core dump
  failed -> Aborted). FIX: `ulimit -c 0` in the launch shell (the file-based
  fallback respects `RLIMIT_CORE`); `HSA_ENABLE_COREDUMP=0` alone did NOT work.
  Clean up: `rm -f /sgl-workspace/*/gpucore.*.gpu`.
- **GPU wedging.** A faulting fp4 kernel can leave that GPU unusable until the
  dead process is reaped; new processes then hang at first HIP op. Use a
  different, verified-healthy GPU (`timeout 20 python3 -c "import torch;
  torch.ones(8,device='cuda:N').sum()"`) and avoid re-running known-faulting
  kernels.
- **Pinpoint an async GPU fault**: run eager (`--disable-cuda-graph`) with
  `AMD_SERIALIZE_KERNEL=3` so the fault surfaces synchronously at the faulting
  kernel instead of at a later sync point.
- **GPU access can vanish after a host reboot**: `/dev/kfd` and `/dev/dri`
  missing, `rocm-smi` -> "Driver not initialized (amdgpu not found)",
  `torch.cuda.device_count()==0`. Needs host/container-level re-attach of the GPU
  devices; not fixable from inside the container.

---

## 8. Current status / next steps

**WORKING END-TO-END (2026-07-07).** DeepSeek-R1-0528-MXFP4 serves on a single
gfx1250 with SGLang+aiter, CUDA graph ON, GSM8K ~0.85 (40 Q, 5-shot, greedy),
invalid 0.000, ~70 tok/s.

Done:
- Linear a4w4 -> bf16 dequant (§4.2); MoE a8w4 via `AITER_FORCE_A8W4=1` (§4.1).
- Triton qk-rmsnorm replacing the unbuildable aiter fused_qk_rmsnorm (§4.4).
- FlyDSL stage1 raw gemm swiglu-arg fix (§4.5).
- `AITER_GROUPED_FORCE_SPLIT_K1=1` -> fused stage1/stage2 (§4.6).
- bf16 KV cache (`--kv-cache-dtype auto`) — fp8 KV breaks decode (§4.7).
- CUDA graph capture works; drop `--disable-cuda-graph`/`AMD_SERIALIZE_KERNEL` for perf (§4.8).
- Coredump disk trap neutralized (§7).

The full working recipe (env + args): `AITER_FORCE_A8W4=1`,
`AITER_GROUPED_FORCE_SPLIT_K1=1`, `--kv-cache-dtype auto`, `--attention-backend triton`,
cuda-graph ON, on a healthy GPU. See CHANGES.md for the exact code diffs.

**UPDATE 2026-07-08 (EXPERIMENT_LOG E17-E20): the accuracy-gap root-cause below is
OVERTURNED.** On a newer docker (model at `/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4`,
sglang `000a61a2`, aiter `8815f4b5`) the a8w4 MoE was **decisively exonerated**:
- a real FlyDSL contiguous-M bisect off-by-one bug was found+fixed (power-of-two
  expert count; R1=256 experts) — op-test contiguous 3.2e-3 -> 3.4e-6 — but it is
  **end-to-end neutral** (GSM8K 1319Q 0.811 ~= the 0.822 baseline).
- pure-numeric probe on real weights: a8w4 (fp8 act) FFN error 3.6% is **4x smaller**
  than a4w4 (fp4 act, gfx950) 15%; so a8w4 cannot be why gfx1250 (0.85) < gfx950 (0.93).
- weight shuffle on gfx1250 is now **mandatory** (unshuffled = GSM8K 0.000 garbage,
  because the B-scale is already n32k4-shuffled); mirror the DSv4 `fp8.py` shuffle.
The gap is now attributed to a **non-MoE gfx1250 path**, not the MoE kernel numerics.
See EXPERIMENT_LOG "2026-07-08 session".

**FURTHER UPDATE 2026-07-08/09 (EXPERIMENT_LOG E21-E24): attention is ALSO exonerated.**
A torch-fp32 hook on `TritonAttnBackend.forward_decode`/`forward_extend` shows both the
MLA decode and MHA prefill kernels match to ~0.16-0.18% (bf16 noise) across all 61
layers; bf16 gemms are literally torch. Cheap A/B knobs all negative
(reduce-in-fp32 no-op, fused-decode-MLA crashes, disable-radix-cache 0.83, cuda-graph
== eager). D validated the baseline is comparable (same HF checkpoint, model/attn/MoE
code identical between the gfx950 commit 7aa6082 and gfx1250 000a61a2, same eval) and
the 0.85-vs-0.93 gap is **real & reproducible**. On gfx1250 **every component is
numerically correct in isolation**, so localizing the gap now requires a **cross-node
per-layer residual-stream diff** — see `HANDOVER_crossnode_dump.md` + the produced
`hs_dump_gfx1250.json`. Run the same hook on gfx950 and diff.

Open / next:
- **Cross-node per-layer diff** (gfx950 vs gfx1250, fixed prompt) — the only remaining
  localizer. See `HANDOVER_crossnode_dump.md`. Everything on-node is already ruled out.
- **Perf**: the bf16 gemms fall back to `torch solution:0` ("not found tuned config
  in bf16_tuned_gemm.csv") — tune/replace for higher throughput.
- **fp8 KV cache** decode read is broken on gfx1250 (§4.7) — would halve KV memory
  if fixed (triton MLA decode + fp8 kv path).
- **split_k>1 raw+finalize** MoE path (§4.5/§4.6) is still broken (we route around it
  via split_k=1); the raw finalize-act kernel faults on DeepSeek shapes.
- **Dense a8w4 linear kernel** (§6): still parked; linear stays bf16-dequant.
- Larger GSM8K (200+ Q) for a stable accuracy number.

Key files:
- `sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` (linear bf16-dequant)
- `sglang/python/sglang/srt/layers/quantization/quark/quark.py` (scheme dispatch)
- `sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` (MoE a4w4 scheme)
- `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/triton_qk_rmsnorm.py` (NEW, §4.4)
- `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` (qk-rmsnorm binding §4.4; MLA-absorb = decode path)
- `sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py` (triton: prefill=MHA, decode=MLA)
- `sglang/python/sglang/srt/layers/moe/moe_runner/aiter.py` (AiterMoeQuantInfo / PER_1X32)
- `aiter/aiter/fused_moe.py` (~L451 gfx1250 grouped a8w4 dispatch; ~L435 AITER_FORCE_A8W4 branch)
- `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py` (AITER_GROUPED_FORCE_SPLIT_K1 §4.6)
- `aiter/aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py` (stage1 raw swiglu-arg fix §4.5)
- `sglang/run_ds-r1.sh` (launch)

(Removed 2026-07-07: `aiter/aiter/ops/flydsl/gemm_a8w4_gfx1250.py` and
`aiter/op_tests/test_gemm_a8w4_gfx1250.py` — the parked dense a8w4 wrapper, dropped
in favor of the bf16-dequant linear path; see §6.)
