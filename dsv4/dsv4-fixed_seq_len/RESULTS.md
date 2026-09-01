# DSV4 fixed-seq-len results — 8x MI355X, 2026-08-31

Sweep run 09:28–12:35, **20/20 points, no server failures**. Box verified clean
afterwards (0 sglang procs, VRAM 0 % on all 8 GPUs).

`Interactivity = 1000 / median_ITL_ms`. `TTT = total_throughput` (input+output
tok/s), so at 1k/1k it is 2x output throughput and at 8k/1k ~9x — the two tables
are not comparable on that column.

## Config

| | |
|---|---|
| model | `/dockerx/data/models/DeepSeek-V4-Pro` — **fp8**, 64 shards, 806G |
| sglang | `0.5.18.dev20260829+g4d53767b09`, editable at `/sgl-workspace/sglang` (main `cdbfe90b4a`) |
| launcher | `serve_mi355x.sh` (extracted from InferenceX `8fcfc62830` `dsv4_fp4_mi355x_sglang.sh`) |
| client | `python3 -m sglang.bench_serving --backend sglang-oai` |
| recipes | conc<=32 -> tp4; 64,128 -> tp4+dp4; 256+ -> tp8+dp8; EP=1 throughout |
| load | range-ratio 1.0 (exact ISL), np=CONC*8, warmup=CONC*2, request-rate inf, ignore_eos on |
| context | 16384 |

## Tables

```
# 1024/1024
Input_len	output_len	TP,DP,EP	Concurrency	TTT (tok/s)	Median E2EL (ms)	Median TTFT (ms)	Median ITL (ms)	Interactivity (tok/s/user)
1024	1024	4,1,1	2	286.287791	14231.384185	267.442263	13.576519	73.656585
1024	1024	4,1,1	4	523.275399	15596.740074	450.623781	14.725339	67.910151
1024	1024	4,1,1	8	901.520292	18108.340701	776.618958	16.568487	60.355541
1024	1024	4,1,1	16	1478.203870	22064.822396	1287.933834	19.542202	51.171306
1024	1024	4,1,1	32	2225.910442	29328.050333	2351.537868	24.623441	40.611708
1024	1024	4,4,1	64	3200.761081	40813.337479	1830.312445	34.668890	28.844304
1024	1024	4,4,1	128	5324.552833	49053.601994	4166.270141	38.902068	25.705574
1024	1024	8,8,1	256	11979.963208	43342.389083	4301.897717	32.755969	30.528787
1024	1024	8,8,1	512	17008.667709	60574.094794	8482.632734	40.768105	24.528979
1024	1024	8,8,1	1024	22697.761214	90956.323494	17121.372827	52.885539	18.908761

# 8192/1024
Input_len	output_len	TP,DP,EP	Concurrency	TTT (tok/s)	Median E2EL (ms)	Median TTFT (ms)	Median ITL (ms)	Interactivity (tok/s/user)
8192	1024	4,1,1	2	1221.820080	15047.365241	869.536619	13.790332	72.514570
8192	1024	4,1,1	4	2165.224943	16977.760827	1207.125309	14.910773	67.065606
8192	1024	4,1,1	8	3610.560817	20365.009671	2070.790410	16.506493	60.582221
8192	1024	4,1,1	16	5543.967871	26582.603551	3976.133305	19.166172	52.175260
8192	1024	4,1,1	32	7647.689590	38576.894494	7419.510068	24.066920	41.550809
8192	1024	4,4,1	64	11607.493795	50368.183473	4969.687776	33.877389	29.518213
8192	1024	4,4,1	128	16851.840895	68958.567545	11339.780725	37.067752	26.977627
8192	1024	8,8,1	256	35247.493680	66145.126226	11213.110224	31.994310	31.255558
8192	1024	8,8,1	512	43974.080837	106350.283192	27715.547660	40.708943	24.564627
8192	1024	8,8,1	1024	51238.571997	182935.069945	58608.463585	54.000832	18.518233
```

## Caveats

1. **FP8, not FP4.** No FP4 DSV4 exists on this box, so these are NOT comparable
   to the `dsv4-fp4-mi355x-sglang` CI dashboard entry, whose name and config say
   fp4. The launch flags do not force precision — it comes from the checkpoint.
2. **c128 -> c256 is a hardware change, not scaling.** TTT more than doubles
   (1k: 5325 -> 11980; 8k: 16852 -> 35247) while TTFT and ITL improve, because
   the recipe goes tp4+dp4 -> tp8+dp8. Same reason c64 TTFT beats c32: c64 is
   the first DP-attention point. Do not read either as a load-scaling effect.
3. **One bad prompt (index 2985)** cost exactly one request in `isl8192 c512`
   (4095/4096) and `isl8192 c1024` (8191/8192). sglang's random dataset is
   seeded, so the same prompt recurs. The server rejected it: "maximum context
   length of 16384 ... a total of 17407 tokens: 16383 from the input messages
   and 1024 for the completion", while the client counted it as 8192 input
   tokens. Cause: the dataset generates token ids, decodes to text, and the
   server re-tokenizes; the round-trip is not length-preserving and this prompt
   doubled. Only bites when np > 2985 AND ISL is already large. Impact <0.03 %
   of requests. Fix: MAX_MODEL_LEN > 16384, or `--dataset-name random-ids`.
4. **Client differs from CI.** CI's `run_benchmark_serving` calls its own
   vendored `utils/bench_serving/benchmark_serving.py`, not sglang's. Flags line
   up (warmup 2xCONC, ignore-eos, request-rate inf) but CI uses np=CONC*10.

All other points passed every fidelity check: exact ISL inputs, exact 1024-token
outputs, full CONC*8 completions, zero client errors.

## Regenerating

`python3 report.py` rebuilds both tables from `runs/*/result.jsonl`. Changing the
Interactivity definition or switching TTT to output-only is a script edit, not a
re-run of the benchmark.

---

# 70000/300 — added 2026-08-31 13:00–13:14

Different scripts and different server configs from the 1k/8k sweep above:
`useful-scripts/benchmarking/dsv4/{run_sgl_dsv4_70k,sweep_dsv4_sglang_client}.sh`,
NP_MULT=4, WARM_MULT=1 (vs CONC*8 / CONC*2 above). Same model, same sglang,
same client and same ratio 1.0. Regenerate with `python3 report_70k.py`.

| phase | server config | conc |
|---|---|---|
| A | `MODE=tp8 CHUNK=32768 SWA=0.1 MEM=0.92` | 2, 4 |
| B | `MODE=tp8dp8 CHUNK=16384 SWA=0.1 MEM=0.80 DELAYER=off` (eff chunk 131072) | 8, 16, 32 |

```
# 70000/300
Input_len	output_len	TP,DP,EP	Concurrency	TTT (tok/s)	Median E2EL (ms)	Median TTFT (ms)	Median ITL (ms)	Interactivity (tok/s/user)
70000	300	8,1,1	2	14908.005357	9425.768855	4631.951437	13.372957	74.777777
70000	300	8,1,1	4	18893.360055	14877.002811	7615.009724	13.864572	72.126282
70000	300	8,8,1	8	30436.764876	18454.020049	12553.882132	19.375751	51.610903
70000	300	8,8,1	16	38803.741658	28983.279886	16785.797714	20.738632	48.219187
70000	300	8,8,1	32	44538.717465	50424.603203	26222.513499	24.248799	41.239157

# integrity
  c2   completed=8/8 errors=0  [tp8_chunk32768]
  c4   completed=16/16 errors=0  [tp8_chunk32768]
  c8   completed=32/32 errors=0  [tp8dp8_chunk16384_nodelayer]
  c16  completed=64/64 errors=0  [tp8dp8_chunk16384_nodelayer]
  c32  completed=128/128 errors=0  [tp8dp8_chunk16384_nodelayer]
```

5/5 points, zero errors, every request completed, no OOM. Box clean afterwards.

**Caveats**
1. TTT here is ~234x output throughput (70300 tokens/request), so the column is
   NOT comparable to the 1k/1k or 8k/1k tables. The workload is prefill-bound:
   TTFT is 49-58 % of E2EL at every point.
2. **conc 4 -> 8 is a config change, not scaling** — TTT 18893 -> 30437 while
   ITL worsens 13.9 -> 19.4 ms, because phase B turns DP-attention on and drops
   mem-fraction 0.92 -> 0.80 at the same time.
3. **Lengths not independently verified.** The 70k client script does not pass
   `--output-details`, so no per-request input_lens/output_lens were saved.
   Completion counts and error counts are verified; exact 70000/300 token
   lengths are not. This matters more here than elsewhere: 70000 is where the
   random-dataset tokenizer round-trip would overflow CTXLEN=73728. Nothing
   errored, so it probably did not happen — but the data cannot prove it.
4. **MODEL and PYTHONPATH deviated from the given commands**, both forced:
   the scripts' `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro` does not exist here
   (used `/dockerx/data/models/DeepSeek-V4-Pro`), and `/sgl-workspace/sglang-upstream`
   does not exist either — Python skips the missing PYTHONPATH entry and resolves
   to the editable tree at `/sgl-workspace/sglang/python`, i.e. upstream **main**.
