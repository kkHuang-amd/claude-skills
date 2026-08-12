# Method: bf16 EMULATION of the MXFP4 MoE with a selectable activation-quant scheme

Self-contained implementation guide so an agent on ANY node (esp. gfx1250) can reconstruct the
E30 emulation even if the exact hook file / code commit differs. Ground-truth implementation:
`scripts/moe_emul_sitecustomize.py` in this skill dir — read it too; this doc explains WHY/HOW so
you can re-derive it against the live tree.

## 0. Goal
Run the DeepSeek-R1-0528-MXFP4 MoE with a chosen activation-quant precision, computed in an
*idealized* bf16 path (no fp4/fp8 hardware kernel), to separate the quant **scheme** from the
real **kernel**:
- `a4w4` : activation mxfp4 q-dq  (VALIDATION — must reproduce the platform's native a4w4 ~0.93)
- `a8w4` : activation mxfp8 q-dq  (the TEST — the scheme gfx1250 is forced onto)
- `a16w4`: no activation quant     (upper precision bound)
Weights are ALWAYS the checkpoint's real fp4, dequantized to bf16 (no extra weight error). Only
the activation precision changes. The matmul is bf16 (fp32 accumulate). This runs on gfx950 AND
gfx1250 because it uses only bf16 GEMM (fp4/fp8 kernels fault on gfx1250; bf16 does not).

E30 result (gfx950, 40Q): a4w4-emul = a8w4-emul = 0.925 == native a4w4 => the a8w4 SCHEME is
accuracy-neutral. Running this on gfx1250 tells you if its ~0.85 is the flydsl real kernel
(emul≈0.925) or a deeper issue (emul≈0.85). See `HANDOVER_gfx1250_moe_emul.md` for the run/read.

## 1. Where to intercept (verify against the live tree)
DeepSeek MXFP4 MoE dispatch (sglang):
`QuarkW4A4MXFp4MoE.apply_weights` (`layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`)
-> `MoeRunner(AITER)` -> `AiterRunnerCore.run` (`layers/moe/moe_runner/aiter.py`)
-> `aiter.fused_moe(...)` -> (gfx1250) flydsl grouped a8w4 kernel / (gfx950) a4w4 kernel.

**Replace `AiterRunnerCore.run`** with a bf16 reimplementation. This is the single clean seam:
it receives everything needed and its output feeds the normal combine/all-reduce. Signature
(confirm in the live file):
```
def run(self, runner_input, quant_info, running_state, hooks=None) -> AiterRunnerOutput
```
Inputs used:
- `runner_input.hidden_states`  [T, model_dim] bf16   (T tokens for this forward)
- `runner_input.topk_ids`       [T, topk] int          (global expert ids; may include a "sink"
                                                         id == num_experts for masked slots)
- `runner_input.topk_weights`   [T, topk] float32
- `quant_info.w13_weight`, `w2_weight`  (fp4x2-viewed uint8 packed weights)
- `quant_info.w13_scale`, `w2_scale`    (e8m0 uint8 block scales)
- `self.config.activation` == "silu"
Return `AiterRunnerOutput(hidden_states=<[T, model_dim] bf16>)`.

## 2. Get the weights into a CLEAN fp4 layout (neuter the shuffles)
`process_weights_after_loading` shuffles weights+scales into a kernel-specific layout that is NOT
trivially dequantable:
- gfx950 branch: `e8m0_shuffle` on scales + `shuffle_weight(w,(16,16))` on weights.
- gfx1250 branch (`_is_gfx1250`): `moe_shuffle_scale` (n32k4) on scales + `shuffle_weight`.
Since we BYPASS the real kernel, we don't want either shuffle. **Monkeypatch these three names to
identity in the `quark_w4a4_mxfp4_moe` module namespace BEFORE weights load** (they are imported
there at module top-level):
```
QM.shuffle_weight   = lambda w, *a, **k: w
QM.e8m0_shuffle     = lambda w, *a, **k: w
QM.moe_shuffle_scale= lambda w, *a, **k: w
```
Then the weights keep the `create_weights` layout:
- `w13_weight` [E, 2*inter_local, model_dim//2] uint8 (fp4x2; gate then up along dim1)
- `w13_scale`  [E, 2*inter_local, model_dim//32] uint8 (e8m0)
- `w2_weight`  [E, model_dim, inter_local//2] uint8
- `w2_scale`   [E, model_dim, inter_local//32] uint8
where `E` = num_experts (+1 sink, e.g. 257), `inter_local` = inter_dim / TP (experts are
replicated across TP, the intermediate dim is sharded).

Install the patch early (a `sitecustomize.py` on PYTHONPATH, or a background thread that polls
until the module imports, then patches — see the ground-truth file). Env-gate the whole thing on
`SGLANG_MOE_EMUL`.

## 3. Dequantize MXFP4 -> bf16 (LUT + e8m0 scale)
e2m1 value table (index = 4-bit code), and per-32 e8m0 scale (code 255 => 0):
```
_MXFP4_VALUES = [0,0.5,1,1.5,2,3,4,6, -0,-0.5,-1,-1.5,-2,-3,-4,-6]   # bf16
def dequant_mxfp4_2d(w_u8 [M,K//2], s_e8m0 [M,K//32]) -> bf16 [M,K]:
    lo = w_u8 & 0xF ; hi = w_u8 >> 4                     # two fp4 per byte, low nibble first
    vals[:,0::2] = LUT[lo] ; vals[:,1::2] = LUT[hi]      # build in bf16 (fp32 [M,K] OOMs)
    scale = 2**(s_e8m0.float() - 127) ; scale[s_e8m0==255] = 0
    return (vals.view(M,K//32,32) * scale.view(M,K//32,1)).view(M,K).bf16
```
Batch over experts by reshaping [A, N, K//2] -> [A*N, K//2], dequant, -> [A, N, K] (one call, not
A calls — the per-expert dequant is the main cost). **Keep everything bf16** (an fp32 [A*N,K]
scratch for all experts OOMs, ~15 GB).

## 4. Activation quant-dequant (the ONLY knob that changes between schemes)
```
a4w4:  from aiter.ops.triton.quant import dynamic_mxfp4_quant
       xq, xs = dynamic_mxfp4_quant(x);  x' = dequant_mxfp4_2d(xq.view(uint8), xs)
a8w4:  from aiter.ops.triton.quant import dynamic_mxfp8_quant ; from aiter import dtypes
       xq, xs = dynamic_mxfp8_quant(x, quant_dtype=dtypes.fp8)
       x' = (xq.float().view(N,K//32,32) * 2**(xs.float()-127).view(N,K//32,1)).view(N,K).bf16
a16w4: x' = x.bf16   (no quant)
```
Apply to BOTH stages for faithfulness: stage1 = the MoE input hidden_states; stage2 = the
intermediate `h` (post-SwiGLU, before down_proj). (dynamic_mxfp*_quant are per-row/1x32; safe on
2D [N, K] with K % 32 == 0; model_dim=7168 and inter_local both divisible by 32.)

## 5. The bf16 grouped FFN (per active expert)
```
xq_all = act_q(hidden_states.bf16())            # stage1 quant ONCE for all tokens
active = unique(topk_ids); active = active[(active>=0) & (active<E)]   # skip sink/invalid
w13d = dequant_batched(w13_weight[active], w13_scale[active])   # [A, 2*inter, model] bf16
w2d  = dequant_batched(w2_weight[active],  w2_scale[active])    # [A, model, inter]  bf16
out = zeros(T, model, float32)
for ai, e in enumerate(active):
    tok, slot = (topk_ids == e).nonzero(...)     # tokens routed to e, and which topk slot
    x_e  = xq_all[tok]                           # [n, model]
    gu   = x_e @ w13d[ai].T                      # [n, 2*inter]
    gate, up = gu.chunk(2, -1)                   # gate = FIRST half, up = SECOND half
    h    = silu(gate) * up                       # [n, inter]   (activation == "silu"/G1U1 SwiGLU)
    h    = act_q(h)                              # stage2 quant
    y    = h @ w2d[ai].T                         # [n, model]
    out.index_add_(0, tok, y.float() * topk_weights[tok, slot].unsqueeze(-1))
return out.to(bf16)
```
Notes / gotchas:
- **gate/up order**: w13 = [gate; up] concatenated along dim1 -> `chunk(2)` gives (gate, up).
  If accuracy is garbage, try swapping (some builds interleave/di­ffer); the a4w4 VALIDATION run
  catches this.
- **doweight_stage1 / apply_router_weight_on_input**: default path applies topk_weights at
  combine (stage2 output) — that's the `* topk_weights` above. If the live pre_permute pre-scales
  hidden by the router weight (topk==1 case), don't double-apply.
- **TP**: experts replicated, inter sharded => this per-rank partial `out` (over local inter) is
  summed across ranks by the framework's all-reduce. Match the local layout; don't all-reduce here.
- **empty batch**: if hidden_states.shape[0]==0 return it unchanged.
- **sink expert** (id == num_experts): skip via `active < E_real`, or just let it contribute ~0.

## 6. Run constraints (MANDATORY)
- **EAGER** (`--disable-cuda-graph`): the Python hook cannot run under cuda-graph replay.
- **Memory**: the bf16 dequant scratch is ~11-22 GB (all active experts). Lower
  `--mem-fraction-static` (gfx950 TP2: ~0.70; gfx1250 TP1: ~0.88 — model ~353 GB must still fit)
  and `export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Reduce KV / max-running-requests
  if OOM.
- **Slow** (~4 tok/s batched). Use 40 Q (parallel 40) for a spot-check; 200 Q (~3 h) to firm up.
- Keep the platform's normal recipe env (AITER_FORCE_A8W4=1, split_k1, Triton qk-rmsnorm, bf16 KV,
  triton attn). The MoE is replaced by the emul regardless; the rest of the model is unchanged.

## 7. Validation protocol (do NOT skip)
1. Run `SGLANG_MOE_EMUL=a4w4` FIRST. It MUST reproduce the platform's native a4w4 GSM8K (~0.93,
   40Q ~0.925). If it doesn't, the emulation is wrong (layout/gate-up/scale/dequant) — fix before
   trusting a8w4. (On gfx1250, native a4w4 can't run, but a4w4-emul should still ~0.925 and match
   gfx950's a4w4-emul; use gfx950's 0.925 as the cross-node anchor.)
2. Then `SGLANG_MOE_EMUL=a8w4`. Compare to the real a8w4 kernel's score on that platform.

## 8. Interpretation
| a8w4-emul vs real a8w4 kernel | conclusion |
|-------------------------------|------------|
| emul ~0.925, real ~0.85 (gfx1250) | the a8w4 SCHEME is fine; the **real kernel** is the gap (fixable) |
| emul == real | the kernel matches the scheme; look elsewhere (non-MoE / deeper execution) |

## 9. Provenance
Method introduced in E30 (EXPERIMENT_LOG.md). Reuses the E20 dequant (`scripts/moe_quant_probe.py`).
Ground-truth hook: `scripts/moe_emul_sitecustomize.py`. Run/interpret on gfx1250:
`HANDOVER_gfx1250_moe_emul.md`. Validated on gfx950: a4w4-emul=a8w4-emul=0.925 (40Q)==native a4w4.
