# DSV4 fixed-seq-len refresh — live notes (started 2026-08-31 09:28)

Sweep driver: `sweep_mi355x.sh`, log `sweep.log`, results under `runs/`,
rolling summary `runs/summary.csv`. Each point writes `runs/<tag>/DONE` on
success, so re-running the driver resumes instead of redoing work.

## Config actually used
- model `/dockerx/data/models/DeepSeek-V4-Pro` — **fp8**, 64 shards, 806G.
  **There is no FP4 DSV4 on this box**, so these are NOT comparable to the
  fp4 CI dashboard entry for `dsv4-fp4-mi355x-sglang`.
- ISL/OSL 1024/1024 and 8192/1024; CONC 2..1024.
- recipes: conc<=32 -> tp4 no-DP; 64,128 -> tp4+dp4; 256+ -> tp8+dp8. EP=1
  throughout (upstream only passes --ep-size when EP_SIZE>1).
- client `python3 -m sglang.bench_serving --backend sglang-oai`,
  range-ratio 1.0 (= exact ISL), np=CONC*8, warmup=CONC*2, request-rate inf,
  ignore_eos on by default.
- sglang 0.5.18.dev20260829+g4d53767b09, editable at /sgl-workspace/sglang
  (main cdbfe90b4a, 20 pre-existing dirty files belonging to other work —
  untouched).

## Traps hit / avoided here
- `sglang bench` subcommand does not exist in this build; entry point is
  `python3 -m sglang.bench_serving`.
- **Must run the client from outside `/sgl-workspace`** — the `sglang/` repo
  dir there shadows the installed package and `-m sglang.bench_serving` dies
  with `ModuleNotFoundError: sglang.benchmark.serving`. Driver cds to /tmp.
- mi355x binds `--cuda-graph-max-bs` and `--max-running-requests` to CONC, so
  every concurrency point needs its own server: 20 launches, no reuse.
- Teardown kills by PID and verifies VRAM drains; `pkill -f 'sglang::'` does
  not work (setproctitle rewrites argv).

## Observed
- server ready in ~3 min at tp4 (09:28:37 launch -> decoding by 09:32).
  TP4 occupies GPUs 0-3 at 89% VRAM, GPUs 4-7 idle.

## Trap: silent Monitor
A `tail -f <log> | grep --line-buffered ... | awk '{print substr(...); fflush()}'`
Monitor produced **zero events across 10 completed points** while the sweep was
demonstrably writing matching lines. Silence read exactly like "still running".
Verified progress out-of-band instead (log mtime, `grep -c '^==='`, DONE count,
rocm-smi). Re-armed as plain `tail -n0 -F | grep --line-buffered -E ...` with no
awk stage. **Do not trust a quiet monitor as evidence a run is alive** — check
the log mtime and the DONE count.

## Sweep complete 2026-08-31 ~12:35 — 20/20 points, box clean afterwards

Tables: `python3 report.py` (reads runs/*/result.jsonl, so the table can be
redefined without re-running). `runs/summary.csv` has been REGENERATED from the per-point
JSON with correct keys (20 rows, adds output/request throughput, tpot, p99 ttft,
completed-vs-expected, duration). The driver's own rolling CSV is kept as
`runs/summary.csv.driver-raw` — its total-throughput column is blank because the
driver wrote a key name that does not exist (`total_token_throughput`; the real
key is `total_throughput`). Fix that line in `sweep_mi355x.sh` before the next
sweep. Final tables live in `RESULTS.md`.

### Only defect: 1 bad prompt, index 2985
`isl8192 c512` completed 4095/4096 and `isl8192 c1024` completed 8191/8192.
Same failing request index (2985) in both — sglang's random dataset is
seeded, so the same prompt recurs. Server rejected it:
"maximum context length of 16384 ... a total of 17407 tokens: **16383 from the
input messages** and 1024 for the completion", while the client counted that
same prompt as 8192 input tokens.
**Cause: sglang's random dataset generates token ids, decodes to text, and the
server re-tokenizes — the round-trip is not length-preserving, and for this one
prompt it doubled.** Only bites when np > 2985 AND ISL is already large:
isl8192 c512/c1024 hit it; isl8192 c256 (np=2048) does not reach index 2985;
no isl1024 point expands far enough to exceed 16384.
Impact 1/4096 and 1/8192 requests (<0.03 %), throughput effect negligible.
Fix if it ever matters: raise MAX_MODEL_LEN above 16384, or use
`--dataset-name random-ids`.
