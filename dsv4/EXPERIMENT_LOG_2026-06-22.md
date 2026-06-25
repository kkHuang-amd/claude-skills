# DeepSeek-V4-Pro serving perf — experiment log (2026-06-22)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 57 — flat-row RoPE kernel: attention-output rope 168us -> 59us, BELOW ATOM (2026-06-22)

### Trigger
In a c512/ISL8192/OSL32 production trace, `apply_rotary_emb_contig_kernel` (our
attention-output inverse rope) measured **p50 168us** vs ATOM's `inverse_rope_gptj`
**83us** (~2x slower). Investigated.

### Root cause (NOT BLOCK_M; my earlier tuning used the wrong shape)
- Trace grid = `[1024, 128, 1]` → **n_heads = 128** (DP-attention computes ALL heads on
  each rank's local tokens). The real rope shape is **[8192, 128, 64]**, 8x bigger than
  the [8192, **16**, 64] I used when tuning BLOCK_M for PR #28783 (so those 11us numbers
  were at the wrong shape; BLOCK_M=8 happened to stay optimal so the PR is still fine).
- microbench at the REAL shape: contig standalone 68us / 3.94 TB/s (warm). But **cold**
  (rotate 24 distinct tensors, mimicking production where each step's `o` is freshly
  produced) = **124us / 2.17 TB/s**. A pure streaming rw of the same 268MB hits
  **5.37 TB/s cold** → the kernel left ~2.5x on the table.
- The contig kernel's program reads BLOCK_M tokens for ONE head, strided by
  n_heads*head_dim (24576) → very scattered 128B chunks → poor cold/HBM efficiency.
  Strided-slice vs dense made no difference; BLOCK_M/num_warps re-sweep cold only got
  2.17→2.56 TB/s. So it's the access pattern, not config.

### Fix — `apply_rotary_emb_flat_kernel`
Iterate (token, head) pairs flattened as `row = token*n_heads + head`, BLOCK_ROWS
**consecutive** rows per program. Consecutive rows are only `head_dim` apart (not
`n_heads*head_dim`) → far less scattered. Math identical (bit-exact, max_abs_err 0.0 vs
contig). Cold microbench (BLOCK_ROWS=16, num_warps=1): **4.5-4.6 TB/s** (~2x), 85% of the
5.37 ceiling, on both dense and strided slices.

### Validation (dev clone, flat rope default-on via set_batched_rope)
- gsm8k 5-shot: **flex 0.9560 / strict 0.9568** (correct).
- c512/ISL8192/OSL32 production trace, rope p50: **168us (old contig) → 59us (flat)** =
  **~2.85x, and BELOW ATOM's 83us** (1.4x faster than ATOM).

### Net
SGLang's attention-output rope now beats ATOM. Removed the now-unused
`apply_rotary_emb_contig_kernel`. Committed `d6e817b0f4` (dev clone).
Artifacts: `dsv4_bench/trace_c512_i8192_o32/sgl_8192_flatrope/` (trace),
`.../gsm8k_flatrope.log`. Servers stopped, VRAM ~0.3 GB/GPU.

---

## Exp 58 — gatherv MoE-gather: 2 redundant 940MB DtoD copies removed (2026-06-22)

### Trigger
Trace observation: between the gather `ncclDevKernel` and `topk_softplus` (the MoE
gate region), SGLang had **Memcpy DtoD 358us + Memcpy DtoD 356us + gate GEMM 297us
= 1011us**, while ATOM had only 2 elementwise (~354us). (NOTE: the DtoD memcpys are
cat="gpu_memcpy", not "kernel" — easy to miss when filtering kernels only.)

### Root cause
`_dp_gather_via_all_gatherv` (dp_attention.py):
```
gathered = all_gatherv(local)      # NCCL allocates its OWN output buffer
gathered = torch.cat([gathered])   # 1-elem list -> still a full-buffer DtoD copy
global_tokens.copy_(gathered)      # copy into the pre-alloc dp buffer
```
Each copy is the whole gathered hidden = sum(sizes)*hidden = **939.5MB** at
c512/ISL8192 (~370us, ~2.5 TB/s) -> ~700us/layer of pure overhead. ATOM gathers
straight into its destination.

### Fix
Add optional `output=` to `GroupCoordinator.all_gatherv`; when given (single-tensor
input), NCCL gathers directly into it (skip the internal `torch.empty` alloc). In
`_dp_gather_via_all_gatherv` pass `output=global_tokens` (sum(sizes)==buffer rows is
already guaranteed by the caller, else it falls back to all_reduce) and drop the
`cat` + `copy_`. `output=` defaults None so all other all_gatherv callers unchanged.

### Validation
- gsm8k 5-shot: **flex 0.9538 / strict 0.9545** (gather is correctness-critical -> OK).
- c512/ISL8192/OSL32 trace, gate region (nccl->topk): **1011us -> 332us (~3x)**, now
  just the gate GEMM, below ATOM's ~354us. Big (>100us) DtoD memcpys in the trace
  dropped from many to 3.

### Net
~680us/layer saved in the MoE-gather path (× compress/MoE layers × prefill steps).
Committed `e6cec15e18` (dev clone). Artifacts:
`dsv4_bench/trace_c512_i8192_o32/sgl_8192_nocopy/`, `.../gsm8k_nocopy.log`.
Servers stopped, VRAM ~0.3 GB/GPU.
