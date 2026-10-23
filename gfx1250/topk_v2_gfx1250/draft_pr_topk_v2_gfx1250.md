# Draft PR: Enable TopK v2 on AMD gfx1250

> **Revalidation status (2026-08-31): do not submit the cluster changes in this
> draft.** The proposed persistent cluster path dereferences
> `cooperative_groups::map_shared_rank()`, but the installed gfx1250 ROCm
> compiler does not provide `__builtin_amdgcn_map_shared_rank`; HIP's fallback
> returns `nullptr`, causing a GPU memory fault. An exact global-workspace
> replacement passed all 278 TopK v2 tests but was 90x--1,000x slower than v2
> streaming across batch 31/64/128 and C4 lengths 16K--262K, so it was rejected
> and removed. The existing register/streaming v2 path remains the supported
> gfx1250 route; its TP1 8k1k microbenchmark is 1.477x--1.672x faster than v1.
>
> A TDM multicast follow-up also passed AITER's 26-check gfx1250 TDM feature
> suite and made the isolated histogram phase faster for batch 1 at long
> context. However, the full correctness-checked TopK candidate remained slower
> at the model's maximum C4 length of 262,144: streaming was 56.0 us versus
> 87.3 us for a 2-workgroup TDM cluster and 92.1 us for four workgroups.
> Batch 2 and 4 regressed further. There is therefore no production TDM-cluster
> threshold within DeepSeek-V4-Flash's 1,048,576-token context limit.

## Title

Enable DeepSeek-V4 TopK v2 on AMD gfx1250

## Summary

This PR enables the DeepSeek-V4 TopK v2 JIT path on ROCm only for AMD gfx1250. The CUDA dispatch remains on the existing code path, and non-gfx1250 ROCm devices keep the conservative behavior.

The gfx1250 implementation uses:

- ROCm architecture detection from `gcnArchName` to add `SGL_ROCM_ARCH_GFX1250`.
- HIP cooperative groups headers for ROCm builds.
- HIP cluster launch attributes through `hipLaunchKernelEx` only when compiling for gfx1250.
- A gfx1250-specific dispatch guard that avoids the unstable fused small-batch cluster path and routes small batches through the safe v2 register/streaming path.
- Persistent cluster TopK v2 for larger cluster-routed batches.

## Notes for Reviewers

The fused small-batch cluster kernel is intentionally disabled for gfx1250. Local bring-up showed memory faults when routing `batch <= kNumPersistentClusters` through the fused DSMEM cluster path. Persistent cluster launch is stable on the tested larger batch shapes, so gfx1250 uses persistent cluster only when `batch > kNumPersistentClusters`.

The server launch script also now defaults to:

- `SGLANG_HACK_FLASHMLA_BACKEND=triton`, because `unified_kv_triton` hits a gfx1250 Triton/Gluon compile failure in `tdm.async_gather`.
- `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`, because decode CUDA graph capture hits the same `tdm.async_gather` compile path.

Both defaults are overrideable through environment variables.

## Validation Commands and Results

### Unit Correctness

```bash
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py::test_topk_v2_raw_indices -s
```

Result:

```text
10 passed, 5 warnings in 6.75s
```

### TopK v1 vs v2 Kernel Microbenchmark

Command:

```bash
python3 - <<'PY'
import torch
from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.dsv4.topk import plan_topk_v2, topk_transform_512, topk_transform_512_v2

PAGE_SIZE = 64

def make_inputs(batch, seq, k):
    torch.manual_seed(42)
    scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
    out1 = torch.empty(batch, k, dtype=torch.int32, device="cuda")
    out2 = torch.empty(batch, k, dtype=torch.int32, device="cuda")
    metadata = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    return scores, seq_lens, page_table, out1, out2, metadata

def same_sets(a, b):
    return all(set(x) == set(y) for x, y in zip(a.cpu().tolist(), b.cpu().tolist()))

def bench(batch, seq, k=512):
    scores, seq_lens, page_table, out1, out2, metadata = make_inputs(batch, seq, k)

    def v1(scores, seq_lens, page_table):
        topk_transform_512(scores, seq_lens, page_table, out1, PAGE_SIZE)
        return out1

    def v2(scores, seq_lens, page_table):
        topk_transform_512_v2(scores, seq_lens, page_table, out2, PAGE_SIZE, metadata)
        return out2

    v1(scores, seq_lens, page_table)
    v2(scores, seq_lens, page_table)
    torch.cuda.synchronize()
    if not same_sets(out1, out2):
        raise RuntimeError(f"v1/v2 set mismatch for batch={batch} seq={seq}")

    r1 = marker.do_bench(v1, input_args=(scores, seq_lens, page_table), use_cuda_graph=False, warmup_iters=20, replay_iters=100, metrics=(0.5, "avg"), memory_args=(scores, seq_lens)).times
    r2 = marker.do_bench(v2, input_args=(scores, seq_lens, page_table), use_cuda_graph=False, warmup_iters=20, replay_iters=100, metrics=(0.5, "avg"), memory_args=(scores, seq_lens)).times
    print(f"batch={batch:<4} seq={seq:<7} k={k}: v1 median {r1[0]*1e6:.2f} us avg {r1[1]*1e6:.2f} us | v2 median {r2[0]*1e6:.2f} us avg {r2[1]*1e6:.2f} us | speedup {r1[0]/r2[0]:.2f}x")

for shape in [(1, 131072), (4, 131072), (64, 131072), (64, 262144)]:
    bench(*shape)
PY
```

Result:

```text
batch=1    seq=131072  k=512: v1 median 371.24 us avg 371.25 us | v2 median 71.99 us avg 72.28 us | speedup 5.16x
batch=4    seq=131072  k=512: v1 median 375.32 us avg 400.57 us | v2 median 71.63 us avg 71.89 us | speedup 5.24x
batch=64   seq=131072  k=512: v1 median 367.43 us avg 367.79 us | v2 median 82.40 us avg 82.68 us | speedup 4.46x
batch=64   seq=262144  k=512: v1 median 710.26 us avg 721.71 us | v2 median 136.52 us avg 136.69 us | speedup 5.20x
```

### Server Bring-Up Without TopK v2

Command:

```bash
PORT=8100 SGLANG_OPT_USE_TOPK_V2=false ./run_v4_gfx1250.sh
python3 benchmark/gsm8k/bench_sglang.py --num-questions 10 --parallel 10 --port 8100
```

Result:

```text
Accuracy: 1.000
Invalid: 0.000
Latency: 317.107 s
Output throughput: 2.826 token/s
```

### Server Bring-Up With TopK v2

Command:

```bash
PORT=8100 SGLANG_OPT_USE_TOPK_V2=true ./run_v4_gfx1250.sh
python3 benchmark/gsm8k/bench_sglang.py --num-questions 10 --parallel 10 --port 8100
```

Result:

```text
Accuracy: 0.900
Invalid: 0.000
Latency: 34.932 s
Output throughput: 26.881 token/s
```

Server-level smoke result on this small 10-question run:

- Latency improved from `317.107 s` to `34.932 s`.
- Output throughput improved from `2.826 token/s` to `26.881 token/s`.
- Throughput ratio: about `9.5x`.

Use a larger GSM8K run for final accuracy reporting; this 10-question run is a bring-up smoke comparison.

## Bring-Up Issues Resolved

1. `unified_kv_triton` failed on gfx1250 with:

   ```text
   'constexpr' object has no attribute 'shape'
   gl.amd.gfx1250.tdm.async_gather(...)
   ```

   Workaround: default the script to `SGLANG_HACK_FLASHMLA_BACKEND=triton`.

2. Decode CUDA graph capture hit the same async-gather compile path.

   Workaround: default the script to `--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`.

3. A stale server process held GPU memory after an interrupted run, causing:

   ```text
   Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.6
   ```

   Resolution: terminate leftover `sglang serve` and `sglang::scheduler` processes before retrying.

## Residual Risk

- Fused small-batch cluster remains disabled on gfx1250 because local DSMEM cluster attempts faulted.
- The server A/B run used a small GSM8K sample and disabled CUDA graph; it is suitable for bring-up signal, not final accuracy/perf signoff.
- `unified_kv_triton` and gfx1250 decode graph capture need separate Triton/Gluon investigation.

