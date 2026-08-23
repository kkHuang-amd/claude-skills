# Current-AITER dense crossover smoke — 2026-08-23

## Result

PASS on gfx950 (MI355X), AITER `dc4bdf1c142181ad90b7f6948564126df4c05fde`, SGLang `455b744aa77b2078de7577619dc12d2775fc1091`. All 15 requested M=2 rows completed HIP graph replay, were finite, and produced numerical comparisons. Across all modes, maximum relative L2 was `0.1829895675` and minimum cosine was `0.9832006097`. The benchmark does not implement an input-change replay gate, so none is claimed.

Artifacts: `/workspace/kimi-k3-runs/dense-crossover-2026-08-23/current-smoke/`. Full command logs are `dense-m2.log` and `latent-m2.log`; concise machine-readable evidence is `filtered-summary.json`. The initial unpinned process imported `/sgl-workspace/aiter` despite reporting the default current-root revision; it is retained as `unpinned-invalid-dense-m2.*` and excluded. Valid runs pinned `PYTHONPATH=/sgl-workspace/aiter-atom-current:/sgl-workspace/sglang-k3-triton37/python` without overriding the benchmark's default `--aiter-root`.

## PTPC dispatch and ATOM match

All four PTPC cases selected `aiter.gemm_a8w8_bpreshuffle` with non-null tuned `libtype=flydsl` configs:

- `shared_down` M2/N7168/K768: kernelId 1404, `flydsl_bpreshuflle_16x64x256_F8_F8_B16_0x1x0x1_default`.
- `kda_mla_output` M2/N3584/K7168: kernelId 497, `flydsl_bpreshuflle_16x64x512_F8_F8_B16_0x2x0x2_default`.
- `mla_qkv_a` M2/N18432/K7168: kernelId 1059, `flydsl_bpreshuflle_16x64x512_F8_F8_B16_1x4x0x2_default`.
- `mla_gate` M2/N4096/K7168: kernelId 617, `flydsl_bpreshuflle_16x64x512_F8_F8_B16_1x2x0x2_default`.

The ATOM BS2 graph trace observes caller `aiter::gemm_a8w8_bpreshuffle` and a generic `kernel_gemm_0` under that caller, so the API and tuned FlyDSL family match the observed ATOM path. Trace symbols do not retain the tuned kernel name, so exact per-shape config identity is not claimed. No null/CK fallback occurred; all four are production-path-equivalent at the observable API/backend-family level.

`get_hip_quant(QuantType.per_Token)` was probed on the actual current import. Activation `[2,3584]` produced scale `[2,1]`; weight rows `[7,3584]` produced scale `[7,1]`, both contiguous/finite. This is the row-wise orientation consumed by `gemm_a8w8_bpreshuffle(XQ,WQ,x_scale,w_scale,...)`. Evidence: `ptpc-scale-orientation.log`.

## Exact commands

```bash
env HIP_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH="/sgl-workspace/aiter-atom-current:/sgl-workspace/sglang-k3-triton37/python" \
  python /workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_dense_crossover.py \
  --shapes shared_down,kda_mla_output,mla_qkv_a,mla_gate \
  --modes bf16,ptpc_fp8,mxfp4 --m-values 2 --warmup 3 --iterations 10 \
  --output-json /workspace/kimi-k3-runs/dense-crossover-2026-08-23/current-smoke/dense-m2.json \
  --output-csv /workspace/kimi-k3-runs/dense-crossover-2026-08-23/current-smoke/dense-m2.csv

env HIP_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH="/sgl-workspace/aiter-atom-current:/sgl-workspace/sglang-k3-triton37/python" \
  python /workspace/useful-scripts/benchmarking/kimi-k3/micro/benchmark_dense_crossover.py \
  --shapes latent_up --modes bf16,mxfp4,rmsnorm_mxfp4 \
  --m-values 2 --warmup 3 --iterations 10 \
  --output-json /workspace/kimi-k3-runs/dense-crossover-2026-08-23/current-smoke/latent-m2.json \
  --output-csv /workspace/kimi-k3-runs/dense-crossover-2026-08-23/current-smoke/latent-m2.csv
```

## Cleanup

All eight GPUs returned to 0% use and baseline `297,766,912 B` VRAM. ROCm SMI still lists stale PID 2327953 as `UNKNOWN` with zero VRAM/SDMA/CU occupancy. No `/tmp/aiter-jit-*` or `/tmp/sglang-cache-*` paths remained.
