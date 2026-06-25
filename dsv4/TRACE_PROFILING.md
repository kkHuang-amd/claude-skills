# DSV4 profiling & trace techniques (sglang + ATOM, MI300/MI355X)

Reusable how-to for the trace/profiling work done 2026-06-23/24 (decode vs prefill
collective analysis, reduce_scatter PoC verification). Pairs with `SKILL.md`.

---

## 0. TL;DR cheat-sheet

- **Decode-heavy trace** = fire a wave with a LONG output len, let prefill drain,
  then profile a short window during decode.
- **CUDA-graph caveat**: decode runs inside a hipGraph → torch-profiler per-kernel
  durations are GARBAGE (ns-scale). Kernel **names/counts are valid** (qualitative),
  but for **timing** use a collective microbench, not the decode trace.
- **GPU cleanup**: `for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}'); do kill -9 $p; done`
  — never `pkill -f "<pattern>"` if `<pattern>` appears in your own shell command
  (it self-matches the wrapper and kills your shell mid-command → "0ms empty" exits).

---

## 1. Capturing a decode-heavy (wave-tail) trace

Decode kernels are output-length-independent, so use a long OSL to make the decode
tail long and the capture window forgiving.

### sglang
Launch with the profiler dir env, then drive `/start_profile` + `/stop_profile`:

```bash
# server: add SGLANG_TORCH_PROFILER_DIR=<dir> to the launch env
SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/dsv4_bench/trace_x ... bash run_sgl_dsv4_aligned.sh

# fire a wave (512 reqs, 8k in / long OSL so decode tail is long)
python3 -m sglang.bench_serving --backend sglang-oai --base-url http://127.0.0.1:8000 \
  --model <model> --dataset-name random --random-input-len 8192 --random-output-len 3000 \
  --random-range-ratio 1.0 --num-prompts 512 --max-concurrency 512 \
  --request-rate inf --warmup-requests 0 &

# wait for prefill to drain (~95s @ c512/8k), then profile 12s of decode
sleep 95
python3 - <<'PY'
import requests, time
b="http://localhost:8000"
requests.post(f"{b}/start_profile", timeout=30)
time.sleep(12)
requests.post(f"{b}/stop_profile", timeout=180)
time.sleep(40)   # let the trace flush to disk
PY
ls -t <dir>/*.trace.json.gz | head
```

Confirm decode is actually running (sglang batch logs):
```bash
rg -n "Decode batch" <srv.log> | tail   # look for "cuda graph: True", gen throughput
```

### ATOM
Same idea, but ATOM has **no batch logs**, so rely on a long OSL (e.g. 3000) for a
wide, forgiving window:
```bash
ATOM_DISABLE_SIDE_STREAMS=1 ATOM_TORCH_PROFILER_DIR=<dir> bash run_atom_dsv4_aligned.sh
# fire wave (same bench_serving, --backend sglang-oai against ATOM's OAI port),
# sleep ~95s, then /start_profile + /stop_profile as above.
```
ATOM trace files land under `<dir>/dp0_tp0/*.pt.trace.json.gz`.

---

## 2. ⚠️ The CUDA-graph profiling limitation (critical)

Decode runs inside a replayed **hipGraph**. In the torch/kineto trace:
- individual GPU kernels appear with **near-zero / nonsense durations** (e.g.
  `mfma_moe` 161ns, `index_elementwise` 5ns) — these are NOT real.
- the real time is hidden inside `hipGraphLaunch` events (one per decode step).

So you **cannot** read per-kernel GPU times from a decode trace. What you CAN do:
- **Qualitative**: kernel **names and counts** are valid. e.g. count `sample`
  (= decode steps), count a per-layer kernel (≈ 2×n_layers under graph dedup), or
  detect which collective is present (`cross_device_reduce_2stage` vs
  `ncclDevKernel_Generic` vs aiter `reduce_scatter_first_dim`).
- For **timing**: use a collective microbench (§4) or capture in a non-graph phase.

Prefill, by contrast, runs **eager** for DSV4 (prefill CUDA graph disabled), so
prefill kernel durations in the trace ARE real.

---

## 3. Analyzing a trace (Python)

```python
import gzip, json, collections
ev = json.load(gzip.open("<trace>.json.gz", "rt"))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and "name" in e]

# decode steps
print("decode steps:", sum(1 for e in k if "sample" in e["name"].lower()))

# which combine collective fired (qualitative; counts are reliable, durations are not)
def c(sub): return sum(1 for e in k if sub.lower() in e["name"].lower())
for key in ["cross_device_reduce", "ncclDevKernel", "reduce_scatter", "allgather_vec"]:
    print(key, c(key))

# top kernels by total dur (ONLY meaningful for eager / prefill traces)
tot = collections.Counter()
for e in k:
    if "dur" in e: tot[e["name"][:70]] += e["dur"]
for n, t in tot.most_common(15): print(f"{t/1000:9.1f}us  {n}")
```

Kernel-name decoder ring (ROCm DSV4 DP-MoE):
| kernel | meaning |
|---|---|
| `_ZN5aiter13allgather_vec...` | aiter custom **all_gather** (equal-len, MAX_LEN gather) |
| `_ZN5aiter26cross_device_reduce_2stage...` | aiter custom **all_reduce** (quickreduce; MAX_LEN combine) |
| `_ZN5aiter24reduce_scatter_first_dim...` | aiter custom **reduce_scatter** (equal-chunk MAX_LEN combine) |
| `ncclDevKernel_Generic_1(...)` | RCCL generic collective (all_gatherv / reduce_scatterv, SUM_LEN) |
| `_paged_decode_*`, `mfma_moe1/2`, `qk_norm_rope` | decode attn / MoE GEMM / rope (per-layer; graph-dedup ≈ 2 steps) |

---

## 4. Collective microbench (real timing when graph blocks the trace)

When you need real GPU times for a collective at the decode size, measure it in
isolation with torchrun (8 GPUs), not from the graphed decode trace.
`/sgl-workspace/dsv4_bench/comm_microbench.py` (RCCL all_reduce vs reduce_scatter):

```python
# torchrun --nproc_per_node=8 --master_port=29555 comm_microbench.py
import os, torch, torch.distributed as dist
lr = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(lr); dist.init_process_group("nccl")
dev = torch.device(f"cuda:{lr}"); hidden = 7168; dtype = torch.bfloat16
def bench(fn, it=200, wu=50):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it*1000  # us
for M in [256, 512, 1024, 2048]:           # M = running_per_rank * world
    ar = torch.randn(M, hidden, device=dev, dtype=dtype)
    ri = torch.randn(M, hidden, device=dev, dtype=dtype)
    ro = torch.empty(M//world, hidden, device=dev, dtype=dtype)
    t_ar = bench(lambda: dist.all_reduce(ar))
    t_rs = bench(lambda: dist.reduce_scatter_tensor(ro, ri))
    if lr == 0: print(M, round(t_ar,1), round(t_rs,1))
```

Caveat: `dist.all_reduce`/`reduce_scatter_tensor` measure **RCCL**. The actual MAX_LEN
decode combine uses aiter's custom quickreduce / reduce_scatter (faster than RCCL),
so RCCL-vs-RCCL numbers don't directly predict the aiter-vs-aiter delta — they only
bound the traffic ratio (all_reduce ≈ 2× reduce_scatter bytes).

DSV4 collective size at c512 decode: global M = running_per_rank(~64) × dp(8) = ~512,
hidden 7168, bf16 → global buffer ~7.3 MB, per-rank gather input ~0.9 MB.

---

## 5. Attributing TTFT: prefill compute vs scheduling (from batch logs)

To tell whether a TTFT change is real prefill slowdown or just admission/scheduling
re-balance, parse the sglang server batch logs (no profiling needed):

```python
import re, statistics as st
pf, dec = [], []
for ln in open("<srv.log>", errors="ignore"):
    if "DP0 TP0]" not in ln: continue            # one rank, avoid 8x dup
    if "Prefill batch" in ln and "#new-token: 8192" in ln:   # full chunks only
        m = re.search(r"input throughput \(token/s\): ([\d.]+)", ln)
        if m: pf.append(float(m.group(1)))
    elif "Decode batch" in ln:
        rr = re.search(r"#running-req: (\d+)", ln)
        g = re.search(r"gen throughput \(token/s\): ([\d.]+)", ln)
        if rr and g: dec.append((int(rr.group(1)), float(g.group(1))))
print("prefill tok/s/rank median:", st.median(pf))           # compute speed
# decode per-step TPOT at matched batch: TPOT_ms = running_req / gen_thr * 1000
```

- `Prefill batch ... input throughput` at fixed `#new-token` = per-rank prefill
  **compute** speed (scheduling-independent). Equal OFF vs ON ⇒ prefill not slower.
- `Decode batch ... gen throughput` bucketed by `#running-req` ⇒ per-step decode
  speed. (Raw gen-throughput medians mix batch sizes — always bucket by running-req.)
- Example finding (06-23): reduce_scatter PoC TTFT rose but prefill input-throughput
  was identical (6948 vs 6976) ⇒ the TTFT change was admission re-balance from faster
  decode, NOT a prefill regression.

---

## 6. Gotchas learned

- **pkill self-match**: `pkill -f "sglang serve"` / `-f model-path` / `-f run_sgl...`
  matches the cursor shell wrapper's own argv → kills the wrapper → command returns
  in ~70ms with empty output. Use the rocm-smi showpids recipe (§0) or kill by
  explicit numeric PID.
- **VRAM lag**: after killing, `rocm-smi --showmeminfo vram` may still show GBs for a
  few seconds; re-check until it drops to ~0.3 GB (297 MB) = clean.
- **`cohere2_moe.py` `@strict` crash**: on hf_hub>=1.x the server crashes at import
  (`StrictDataclassDefinitionError`). Keep the local no-op `strict` workaround
  (SKILL §2a); it is NOT committed and gets reverted by repo sync, so re-apply
  before launching if the server dies on import.
- **profiler flush**: sleep ~30-40s after `/stop_profile` before listing files; the
  `.trace.json.gz` is written asynchronously.
- **decode confirmation**: only trust a "decode trace" once batch logs show
  `Decode batch ... cuda graph: True` during the capture window.
```
