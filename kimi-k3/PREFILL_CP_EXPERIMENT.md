# Kimi-K3 Prefill CP experiment

## CONTINUE HERE

- Status: complete on 2026-08-04.
- Result: true 8-way Prefill CP is functionally correct but is not viable for
  this Kimi-K3 68k/350 workload on 8x MI355X.
- Recommendation: use `ENABLE_PREFILL_CP=0`. Enable Radix Cache separately when
  the workload has reusable prefixes.
- Final artifacts:
  `/dockerx/home/wunhuang/tmp/benchmark-results/kimi-k3-prefill-cp/20260804T043116Z-cp8-mem093-bs8-chunk8k`.
- Exact next step, if investigating CP further: determine why
  `--attn-cp-size 8` increases per-GPU weights from 194.38 GB to 253.20 GB
  before running another performance sweep.

## Environment

- Hardware: 8x AMD Instinct MI355X (`gfx950`)
- Model: `/dockerx/data/Kimi-K3`, 96 shards, approximately 1.6 TB
- SGLang: `0.5.16.dev20260803+ge6311f7559`
- Source commit: `e6311f7559dfab9ae6bc5ba36a34872ed738cd6f`
- Workload: random fixed-length ISL 68000 / OSL 350
- Concurrency: 1, 2, 4, 8
- Samples: `4 * concurrency`; warmups: `1 * concurrency`
- Both sides: Radix Cache disabled, TP8, memory fraction 0.93,
  max-running-requests 8, decode CUDA Graph max batch 8, prefill chunk 8192

Experimental side:

```text
--enable-prefill-cp --cp-strategy zigzag --attn-cp-size 8
```

Control side omitted all three flags. `/server_info` confirmed
`enable_prefill_cp=true, attn_cp_size=8` for the experiment and
`enable_prefill_cp=false, attn_cp_size=1` for the control.

## Results

| Concurrency | Baseline total tok/s | CP total tok/s | CP throughput | Baseline median TTFT | CP median TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 3897.14 | 1001.44 | -74.30% | 10.08 s | 41.05 s |
| 2 | 4781.20 | 1003.53 | -79.01% | 15.65 s | 108.75 s |
| 4 | 5273.87 | 1002.87 | -80.98% | 27.02 s | 245.13 s |
| 8 | 5495.68 | 1001.27 | -81.78% | 48.22 s | 518.62 s |

All measured requests completed successfully. CP throughput remains flat near
1000 tok/s because its 92,211-token cache capacity admits only one 68k request
at a time. Higher client concurrency only increases queueing.

## Memory explanation

Final `/server_info` memory profiles:

```text
baseline:   weights=194.38 GB, KV=62.90 GB, graph=0.95 GB, capacity=2,442,661
prefill CP: weights=253.20 GB, KV= 2.37 GB, graph=1.21 GB, capacity=   92,211
```

Setting only `--enable-prefill-cp` is not sufficient for Kimi-K3: the initial
run reported the flag as enabled but retained `attn_cp_size=1`, producing a
no-op result. Explicit `--attn-cp-size 8` activated ranks `ATTN_CP0..7`, but
also changed the weight layout and consumed about 58.8 GB more HBM per GPU.

The 32k prefill chunk OOMed in KDA transient tensors. Reducing the chunk to
8192 was required for the 68k request to complete. Memory fraction 0.93 was the
only tested stable point: 0.85 could not create a KV pool, 0.98 over-allocated
KV before the unaccounted 7.6 GB INT8 Mamba checkpoint pool, and 0.95 left
insufficient KDA transient headroom.

## Implementation fixes made during validation

- Added explicit `--attn-cp-size`.
- Made server readiness fail fast on generic startup tracebacks.
- Guarded a CUDA-only `cuda_fp8.h` include in the Kimi-K3 SiTU JIT header on
  ROCm. A standalone SiTU compile/run test passed afterward.
- Added reusable long-context benchmark, A/B orchestration, and summarization
  scripts under
  `/dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/kimi-k3`.

The server was stopped after the run. All eight GPUs returned to 0% allocated
VRAM and no KFD processes remained.
