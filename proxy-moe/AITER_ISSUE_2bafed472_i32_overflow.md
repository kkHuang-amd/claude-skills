Owner node: ctheliosp-zuse-f19-14

# DRAFT (not filed): grouped a8w4 launch overflows int32 after 2bafed472 (cached FlyDSL launchers)

Status: draft for the user to review and file against ROCm/aiter. Nothing below has been posted anywhere.

## Summary

After `2bafed472` "[FlyDSL] Cache compiled FlyDSL launchers on host paths (#6946)", the gfx1250 grouped MoE a8w4 TDM
GEMM (`flydsl_grouped_gemm_a8w4_masked`, `aiter/ops/flydsl/grouped_gemm_mxfp4.py`) fails on large prefills:

```
File "aiter/ops/flydsl/grouped_gemm_mxfp4.py", line 251, in flydsl_grouped_gemm_a8w4_masked
File "aiter/ops/flydsl/kernels/tensor_shim.py", line 324, in _run_compiled
    cf(*args)
File "<flydsl-cabi-fill>", line 4, in fill
struct.error: 'i' format requires -2147483648 <= number <= 2147483647
```

## Root cause

The commit switched the launch to `_run_compiled(launch_gemm_a8w4_tdm, out.view(-1), ..., a_scales.view(torch.int32).view(-1),
w_scales.view(torch.int32).view(-1), ..., quant_scale_tensor, ..., ep_row_map_tensor, ...)` with rank-1 views
("normalize their ranks without specializing the compiled callable on buffer shapes"). The FlyDSL C ABI fill packs a
tensor argument's size as int32. `quant_scale_tensor` and `ep_row_map_tensor` default to `out` as well.

`out` is (1, contiguous_m, N). In a proxy-MoE prefill (model_dim 16384, 256 local experts, top-16, 8192-token chunks +
a 6000-token request): contiguous_m = 139264, N = 16384 -> 139264 x 16384 = 2,281,701,376 elements > 2**31 - 1.
Every Python int argument of that launch was within int32 (largest: contiguous_m = 139264), so the failing field is a
tensor size. Before the commit `out` was passed un-flattened, so no single dimension came close to 2**31.

The launcher only uses these tensors through their base pointer (`fx.get_iter(arg_c)` etc.; the inner kernel takes
`fx.Pointer`), so the packed size is never read by the kernel.

## Reproducer

SGLang proxy-MoE server on gfx1250, TP2/EP2, 4 layers, aiter gfx1250 grouped MoE (default AITER flags), chunked prefill 8192:
run a request of ~6000 tokens, then a 16-way concurrent 1k-token batch (our smoke test + bench_serving). Failed 2/2 at
`2bafed472`-containing aiter; passes after the fix below. (Not yet reduced to a standalone script.)

## Fix used on our branch

`kkHuang-amd/aiter@feat/proxy-model-gfx1250-1023` commit `7ea6f2f46`: in `grouped_gemm_mxfp4.py`, pass each flattened
tensor through

```python
_I32_MAX = 2**31 - 1

def _flat_base(t):
    flat = t.view(-1)
    return flat[:_I32_MAX] if flat.numel() > _I32_MAX else flat
```

applied to `out`, `quant_scale_tensor`, `ep_row_map_tensor`, `a_scales`, `w_scales`. Same base address, same kernel, rank
and specialization key unchanged (no recompile). Validated: 2/2 passes, TTFT/TPOT unchanged (215-216 ms / 8.84-8.85 ms
for 1k1k c16 TP2).

## Suggestions for upstream

1. Same pattern elsewhere: the commit touched 14 files; `moe_kernels.py`, `mla_kernels.py`, `mega_moe/quant.py`,
   `hc_mix.py`, `qsa_flydsl_layout.py`, ... contain ~57 `.view(-1)` calls. Any rank-1 tensor argument that can exceed
   2**31 elements and reaches the cached launcher has the same failure. (`grouped_moe_gfx1250.py` wraps its flattened args
   in `ptr_arg`, which has no size field, so it is not affected.)
2. Cleaner options than clamping: declare base-address-only parameters as `fx.Pointer` and pass `ptr_arg(t)` (as already
   done for A/B), or have the FlyDSL C ABI carry tensor sizes as int64.
3. A unit test with a > 2**31-element `out` (e.g. contiguous_m x N with N = 16384, contiguous_m >= 131073) would catch it.
