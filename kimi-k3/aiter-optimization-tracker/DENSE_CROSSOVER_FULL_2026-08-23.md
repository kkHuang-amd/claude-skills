# Current-AITER dense crossover — full graph matrix

PASS with explicit skips. Regenerated from the existing `dense-crossover.json` without a GPU rerun. MI355X/gfx950; AITER `dc4bdf1c142181ad90b7f6948564126df4c05fde`; SGLang `455b744aa77b2078de7577619dc12d2775fc1091`; graph-primary, eager=false, warmup=10, iterations=100, seed=20260823.

## Aggregate gates

- Matrix: 132 rows; 125 supported/OK; 7 explicit skips.
- Supported rows: 125/125 graph timings finite, outputs finite, numerical comparisons present, input-change freshness passed with distinct hashes, and dispatch API/layout metadata present.
- Numerical overall: max rel-L2 `0.182990`, min cosine `0.983201`; alternate-input max rel-L2 `0.182990`, min cosine `0.983201`.
- PTPC dispatch: 19 tuned FlyDSL / observed-ATOM-equivalent-family rows, 5 tuned CK rows, 12 generic null-config fallback rows, and 6 unsupported skips. By projection: tuned FlyDSL throughout for `shared_down` and `kda_mla_output`; mixed FlyDSL/CK for `mla_qkv_a` and `mla_gate`; generic fallback throughout for `latent_up` and `merged_front`; unsupported for `kda_inproj`.

## Numerical by mode

| Mode | Rows | max rel-L2 | min cosine | max alternate rel-L2 | min alternate cosine |
|---|---:|---:|---:|---:|---:|
| bf16 | 41 | 0.004587 | 0.999990 | 0.004554 | 0.999990 |
| mxfp4 | 42 | 0.166022 | 0.986141 | 0.173342 | 0.984888 |
| ptpc_fp8 | 36 | 0.037986 | 0.999278 | 0.037986 | 0.999278 |
| rmsnorm_mxfp4 | 6 | 0.182990 | 0.983201 | 0.182990 | 0.983201 |

## Complete-chain crossover versus BF16

| Projection | Mode | winning tested M buckets | speedup at M=2/4/8/16/32/64 | representative dual storage MiB (one shape) |
|---|---|---|---|---:|
| kda_inproj | mxfp4 | 32-64 | 0.943x / 0.929x / 0.972x / 0.977x / 1.044x / 1.455x | 108.83 |
| kda_mla_output | mxfp4 | none | 0.893x / 0.870x / 0.839x / 0.870x / 0.864x / 0.853x | 26.58 |
| kda_mla_output | ptpc_fp8 | none | 0.978x / 0.981x / 0.953x / 0.945x / 0.955x / 0.931x | 31.53 |
| latent_up | mxfp4 | 2-64 | 1.115x / 1.084x / 1.074x / 1.014x / 1.295x / 1.076x | 62.02 |
| latent_up | ptpc_fp8 | 2-8,32-64 | 1.122x / 1.076x / 1.067x / 0.969x / 1.302x / 1.032x | 73.53 |
| latent_up | rmsnorm_mxfp4 | 2-64 | 1.085x / 1.090x / 1.076x / 1.002x / 1.308x / 1.074x | 62.02 |
| merged_front | mxfp4 | 64 | 0.793x / n/a / 0.791x / 0.828x / 0.884x / 1.009x | 104.12 |
| merged_front | ptpc_fp8 | 16-64 | 0.934x / n/a / 0.946x / 1.040x / 1.084x / 1.060x | 123.40 |
| mla_gate | mxfp4 | none | 0.840x / 0.829x / 0.805x / 0.832x / 0.895x / 0.974x | 26.58 |
| mla_gate | ptpc_fp8 | none | 0.840x / 0.809x / 0.797x / 0.864x / 0.937x / 0.972x | 31.51 |
| mla_qkv_a | mxfp4 | none | 0.625x / 0.660x / 0.838x / 0.624x / 0.691x / 0.739x | 36.59 |
| mla_qkv_a | ptpc_fp8 | 64 | 0.868x / 0.855x / 0.822x / 0.900x / 0.981x / 1.049x | 43.32 |
| shared_down | mxfp4 | none | 0.917x / 0.858x / 0.934x / 0.898x / 0.910x / 0.900x | 13.29 |
| shared_down | ptpc_fp8 | 8 | 0.979x / 0.948x / 1.007x / 0.944x / 0.965x / 0.980x | 15.78 |

## Fastest observed candidate policy

This is a microbenchmark candidate only; endpoint and accuracy gates remain required.

| Projection | fastest mode by M=2/4/8/16/32/64 | prepared modes required | representative retained MiB (one shape) |
|---|---|---|---:|
| kda_inproj | bf16 / bf16 / bf16 / bf16 / mxfp4 / mxfp4 | bf16, mxfp4 | 108.83 |
| kda_mla_output | bf16 / bf16 / bf16 / bf16 / bf16 / bf16 | bf16 | 21.00 |
| latent_up | ptpc_fp8 / rmsnorm_mxfp4 / rmsnorm_mxfp4 / mxfp4 / rmsnorm_mxfp4 / mxfp4 | MXFP4-shared, ptpc_fp8 | 86.52 |
| merged_front | bf16 / ptpc_fp8 / bf16 / ptpc_fp8 / ptpc_fp8 / ptpc_fp8 | bf16, ptpc_fp8 | 123.40 |
| mla_gate | bf16 / bf16 / bf16 / bf16 / bf16 / bf16 | bf16 | 21.00 |
| mla_qkv_a | bf16 / bf16 / bf16 / bf16 / bf16 / ptpc_fp8 | bf16, ptpc_fp8 | 43.32 |
| shared_down | bf16 / bf16 / ptpc_fp8 / bf16 / bf16 / bf16 | bf16, ptpc_fp8 | 15.78 |

## Explicit skips

- `merged_front:m4:bf16`: eager API smoke failed: ValueError: Invalid tile combination: tile_m * tile_k must be divisible by ldg_vec_size * block_threads = 2048; got 1024
- `kda_inproj:m2:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=2, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!
- `kda_inproj:m4:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=4, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!
- `kda_inproj:m8:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=8, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!
- `kda_inproj:m16:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=16, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!
- `kda_inproj:m32:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=32, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!
- `kda_inproj:m64:ptpc_fp8`: eager API smoke failed: RuntimeError: gemm_a8w8_bpreshuffle failed for shape M=64, N=6288, K=7168, dtype=torch.bfloat16, config=None: This GEMM is not supported!

## Storage scope and layer assumptions

The timing rows report one representative TP8-local shape. They are not model totals. Actual Kimi-K3 multiplicities used by `layer-weighted-storage.json` and `.csv` are:

- `latent_up`, `shared_down`, and SGLang-only `merged_front`: 92 MoE layers (layer 0 is dense).
- SGLang-only `kda_inproj`: 69 KDA layers.
- `mla_qkv_a` and `mla_gate`: 24 MLA layers.
- `kda_mla_output`: 93 output projections. This combined count is valid because the 69 KDA and 24 MLA paths both retain the same TP8-local `[7168,1536]` BF16 `o_proj` representation and use the same prepared formats.

## Recommended-policy model-layer-weighted storage

BF16 remains retained. Thresholds select a runtime mode but do not create one prepared copy per M bucket. MXFP4 and RMSNorm+MXFP4 share one representation and are not double-counted.

| Contribution | Layers | prepared B/layer/GPU | incremental B/GPU | incremental GiB/GPU |
|---|---:|---:|---:|---:|
| `latent_up` MXFP4, all tested M | 92 | 13,647,872 | 1,255,604,224 | 1.169373 |
| `kda_inproj` MXFP4, M32+ | 69 | 23,969,792 | 1,653,915,648 | 1.540329 |
| `mla_qkv_a` PTPC FP8, M64 | 24 | 15,147,264 | 363,534,336 | 0.338568 |
| **Total incremental prepared** | | | **3,273,054,208** | **3.048269** |

For these three projection families, retained BF16 is `11,673,632,768 B` (`10.871918 GiB/GPU`); BF16 plus prepared copies is `14,946,686,976 B` (`13.920187 GiB/GPU`).

The former `+52,764,928 B` (`50.320557 MiB`) number is withdrawn as a model cost. It is only `representative_shape_storage`: one prepared copy of each of the three selected shapes without layer multiplicity.

## Estimated token-capacity impact

Estimate only, not a server measurement: `MOE_LATENT_SPLIT_2026-08-20.md` measured a `142,753`-token loss for `1,255,604,224 B/GPU` of latent-up packed weights (92 × `13,647,872 B`). Applying that documented linear bytes/token calibration to `3,273,054,208 B/GPU` estimates a `372,122`-token loss from the `1,519,705` baseline, leaving about `1,147,583` tokens. Allocator behavior and bytes/token slope may differ for the other prepared tensors.

## Interpretation

- Tuned PTPC FlyDSL rows for `shared_down`, `kda_mla_output`, `mla_qkv_a`, and `mla_gate` match the ATOM-observed API/backend family; exact kernel identity is not claimed. Larger-M tuned CK rows are valid tuned dispatch but not ATOM backend-family-equivalent.
- `latent_up` and `merged_front` PTPC use null-config generic fallback for all M and are kept separate from tuned PTPC evidence. `kda_inproj` PTPC is unsupported and skipped.
- Candidate hybrid policy: `latent_up` MXFP4 for all tested M; `kda_inproj` BF16 through M16 then MXFP4; `mla_qkv_a` BF16 through M32 then tuned PTPC at M64; BF16 elsewhere. Treat the 0.7% `shared_down` PTPC M8 win as noise-sized, and defer generic-fallback `merged_front` PTPC despite its M16+ wins until backend attribution. Endpoint and paired accuracy evidence are still required.
- `merged_front` M4 BF16 is a runtime-path skip from an invalid tuned tile combination; quant rows remain valid, but no BF16-relative M4 speedup is claimed.
