# Cluster Launch Examples

## gfx1250 Capability Probe

Use this to confirm the runtime device and normal shared-memory limits:

```bash
python3 - <<'PY'
import torch

print("hip", torch.version.hip)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device", torch.cuda.get_device_name(0))
    print("gcnArchName", getattr(props, "gcnArchName", None))
    print("shared_memory_per_block", props.shared_memory_per_block)
    print("shared_memory_per_multiprocessor", props.shared_memory_per_multiprocessor)
PY
```

Expected local gfx1250 output:

```text
hip 7.15.0
cuda available True
device AMD Radeon Graphics
gcnArchName gfx1250
shared_memory_per_block 327680
shared_memory_per_multiprocessor 327680
```

Header evidence to inspect:

```bash
rg "clusterLaunch|hipLaunchAttributeClusterDimension|map_shared_rank|cluster_workgroup" \
  /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/include/hip
```

## TopK v2 Correctness Commands

Focused raw-index test that covers trivial, register, streaming fallback, persistent cluster, and non-cluster streaming:

```bash
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py::test_topk_v2_raw_indices -s
```

Known useful cluster boundary tests:

```bash
# Fused small-batch cluster boundary; currently faults on gfx1250 if routed to fused cluster.
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py -k 'test_topk_v2_output_indices and 30-131072' -s

# Persistent pool boundary; should pass on gfx1250.
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py -k 'test_topk_v2_output_indices and 31-131072' -s

# Longer persistent cluster-routed case; should pass on gfx1250.
python3 -m pytest -q test/registered/jit/deepseek_v4/test_topk_v2.py -k 'test_topk_v2_output_indices and 40-262144' -s
```

## v1 vs v2 Microbenchmark Snippet

This mirrors `bench_topk.py` but keeps the shape list short and verifies set equality.

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

Local gfx1250 result from the TopK v2 bring-up:

```text
batch=1    seq=131072  k=512: v1 median 372.12 us avg 372.74 us | v2 median 72.43 us avg 72.85 us | speedup 5.14x
batch=4    seq=131072  k=512: v1 median 370.64 us avg 384.31 us | v2 median 71.75 us avg 71.98 us | speedup 5.17x
batch=64   seq=131072  k=512: v1 median 369.55 us avg 374.12 us | v2 median 81.92 us avg 82.41 us | speedup 4.51x
batch=64   seq=262144  k=512: v1 median 709.75 us avg 719.58 us | v2 median 136.29 us avg 136.51 us | speedup 5.21x
```

## Server A/B Launch Pattern

Run separate ports to compare without and with TopK v2:

```bash
# Without TopK v2
PORT=8100 SGLANG_OPT_USE_TOPK_V2=false ./run_v4_gfx1250.sh

# With TopK v2
PORT=8101 SGLANG_OPT_USE_TOPK_V2=true ./run_v4_gfx1250.sh
```

Then run the same benchmark against each port, for example:

```bash
python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 200 --port 8100
python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 200 --port 8101
```

Use the larger TODO target only after the smoke comparison passes:

```bash
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --port <port>
```

