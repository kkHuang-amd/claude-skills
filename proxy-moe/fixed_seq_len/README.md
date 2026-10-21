Owner node: crsuse2-m2m-231

# Proxy MoE fixed_seq_len sweep (InferenceX-style)

Proxy MoE (24 layers, 8x MI355X) performance at three fixed shapes, across concurrency, with
the current best server configurations. Modeled on `dsv4/dsv4-fixed_seq_len/` (same client
settings, one server per point, kill-by-PID teardown, resumable DONE markers, same report
columns plus TTT/GPU and P90 interactivity).

## CONTINUE HERE

**Status (2026-10-21 01:18 UTC):** FULL SWEEP RUNNING on crsuse2-m2m-231 (user: run it in this session),
sweep pid in `/shared_nfs/kk/results/proxy_24l/fixed_seq_len/sweep.pid` (404161), log `sweep.log`.
Survival: sweep pid has PPID 1, own process group, nohup → survives the Claude session ending.
It does NOT survive the container/node being reclaimed (no scheduler visible inside the container;
check the reservation). Results are on /shared_nfs and DONE markers make a restart resume; `/data`
(checkpoints, worktrees, FlyDSL 0.4.0) is container-local: regenerate ckpt (seed 0, minutes), re-create
worktrees from the pushed branches. Node-local scripts backed up to `/shared_nfs/kk/proxy_moe_node_backup_c231/`.
Earlier: scripts written; 2-point smoke (tp8ep1 1024/1024 c2 and 70000/300 c2; tp4ep1 1024/1024 c2)
run on crsuse2-m2m-231 → passed, see "Smoke". Full sweep NOT started (user: start it in the next session,
look at optimizations only after the sweep).
**Next:** start the full sweep in the background, monitor by DONE count, then `report.py --md`
→ `RESULTS.md`, commit + push.
**Repro (full sweep, ~4-6 h, 50 points):**
```bash
nohup bash /data/proxy_moe/wt/skills/proxy-moe/fixed_seq_len/sweep_proxy.sh \
  > /shared_nfs/kk/results/proxy_24l/fixed_seq_len/sweep.log 2>&1 &
echo $! > /shared_nfs/kk/results/proxy_24l/fixed_seq_len/sweep.pid
```
**Progress:** `find /shared_nfs/kk/results/proxy_24l/fixed_seq_len -name DONE | wc -l` (of 50),
`grep -E '^===|FAILED' .../sweep.log | tail`. Stop: `kill $(cat .../sweep.pid)` then kill the
server PID in the current point's `server.pid`.
**Pass criteria:** every point DONE; report.py prints no WARNINGS (completed = CONC*8, or CONC*4
for 70k; exact input/output lengths; no client errors).

## Shapes, concurrency, recipes

| Shape (ISL/OSL) | CONC | np | warmup |
|---|---|---|---|
| 1024/1024 | 2 4 8 16 32 64 128 256 512 1024 | CONC*8 | CONC*2 |
| 8192/1024 | 2 4 8 16 32 64 128 256 512 1024 | CONC*8 | CONC*2 |
| 70000/300 | 2 4 8 16 32 | CONC*4 | CONC*2 |

| Recipe | TP, DP, EP | Why |
|---|---|---|
| `tp8ep1` | 8,1,1 | 24 layers on 8 GPUs; best throughput / TPOT at TP8 |
| `tp4ep1` | 4,1,1 | 12 layers on GPUs 0-3 (proxy spec for 4 GPUs); best throughput / TPOT at TP4 |

User decision 2026-10-21: TP8 and TP4 only for now (EP1 = InferenceX default, no --ep-size).
`tp8ep8` / `tp4ep4` (best TTFT) are wired in `recipe_env` for a later run.

Both recipes: pushed branches (AITER `kkHuang-amd/aiter@feat/proxy-model-1020`, SGLang
`HaiShaw/sglang@feat/proxy-model-1020`, FlyDSL 0.4.0) via `/data/proxy_moe/newbase_env.sh`,
`--enable-aiter-allreduce-fusion`, `SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1`, tuned MoE rows in AITER's
`model_configs/proxymoe_fp4_tuned_fmoe.csv`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (container env).
Server: `--max-running-requests CONC --cuda-graph-max-bs-decode CONC` (this SGLang split `--cuda-graph-max-bs` into -decode / -prefill; the old name is now ambiguous) (so one server per point, as on
the dsv4 MI355X sweep), `--context-length 73728` (70000 + 300 fits; model max 131072),
`--chunked-prefill-size 8192` and `--mem-fraction-static 0.88` (package defaults), fp8 KV.
`recipe_env` sets the model (24L / 12L) and GPUs per recipe (`MODEL_24L` / `MODEL_12L` override).

Client: `python3 -m sglang.bench_serving --backend sglang --dataset-name random-ids
--tokenize-prompt --random-range-ratio 1.0 --request-rate inf`, ignore_eos on (default), run from
/tmp. `random-ids` sends token ids, so lengths are exact (the dsv4 sweep's `random` dataset
detokenizes and re-tokenizes, which inflated one 8k prompt to 16383 tokens).

## Files

| File | Role |
|---|---|
| `serve_proxy.sh` | launch one server (CONC required), wait for `/health_generate`, verify the port belongs to our PID, write `server.pid`, exit 0 |
| `sweep_proxy.sh` | driver: recipes x shapes x CONC, server per point, client, DONE marker, teardown by PID + VRAM drain |
| `report.py` | tables per recipe/shape from `<OUT>/<recipe>/isl*-osl*-c*/result.jsonl`; `--md` for markdown; integrity warnings on stderr |

Results: `/shared_nfs/kk/results/proxy_24l/fixed_seq_len/<recipe>/isl<ISL>-osl<OSL>-c<CONC>/`
(`server.log`, `launch.log`, `client.log`, `result.jsonl`, `DONE`).

## Report columns

`Input_len, output_len, TP,DP,EP, Concurrency, TTT (tok/s), TTT/GPU, Median E2EL, Median TTFT,
Median ITL, Interactivity (= 1000 / median ITL), P90 interactivity (= 1000 / p90 ITL)`.
TTT = total_throughput (input + output tok/s). The first nine match dsv4's `report.py`.

## Traps (from dsv4 and this project)

- Do not reuse the package `run_stage.sh` for sweeps: its `stop_server` runs
  `pkill -f sglang.launch_server` and kills every server on the host.
- `pkill -f 'sglang::'` alone is not enough to find the launcher; teardown kills the launcher
  PID, then leftover `sglang::*` children, then waits until VRAM drains.
- Run clients from outside `/sgl-workspace` (the repo dir shadows `sglang.benchmark`).
- The first launch after a code or shape change JIT-compiles; CUDA-graph capture up to CONC=1024
  adds launch time. Neither affects the measured numbers (warmup requests run first).
- A silent monitor is not proof of progress: check the DONE count and the log mtime.
- 8192/1024 at CONC=1024 needs ~9.4M KV tokens; capacity is ~9.5M per GPU at 0.88 → expect
  little headroom (watch for retractions in server.log).

## Smoke (2026-10-21, crsuse2-m2m-231, NP_MULT=2 so "completed" warnings are expected)

Output: `/shared_nfs/kk/results/proxy_24l/fixed_seq_len_smoke/`. First attempt failed at argument
parsing: this SGLang made `--cuda-graph-max-bs` ambiguous (`-decode` / `-prefill`) → fixed.

| Recipe | Shape | CONC | TTT/GPU | Median TTFT (ms) | Median ITL (ms) | P90 interactivity |
|---|---|---|---|---|---|---|
| tp8ep1 | 1024/1024 | 2 | 78.5 | 112.7 | 6.25 | 156.6 |
| tp8ep1 | 70000/300 | 2 | 3,383.2 | 2,246.4 | 6.49 | 153.4 |
| tp4ep1 | 1024/1024 | 2 | 217.9 | 86.6 | 4.49 | 219.2 |
