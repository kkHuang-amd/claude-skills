# DSV4.1-Flash regression check: sglang release image (2026-10-04) vs main 58f0d250ec + #42055

Owner node: mi355-4 (PIDs / GPU ids / ports below refer to mi355-4).

## CONTINUE HERE

**Status: CLOSED 2026-10-05 14:46 +08 (user).** TP2 c64 and c8 = no regression (table below). TP4 c64 stopped by user
before it finished (no data). Open: TTFT p50 at TP2 c64 1369 -> 1600 ms and slower load/warmup on the new code.
Scripts now default to this best config (SKILL.md "Scripts"); no EXTRA_ARGS_rep needed any more.
Was: single lane, serial, RESTARTED 2026-10-05 12:12 +08: pgid 1833, port 8888, TP2 GPUs 0,1
rel1004_tp2_c64 -> rel1004_tp2_c8, then TP4 GPUs 0-3 rel1004_tp4_c64 (~4 h, ETA ~16:00 +08). Kill: `kill -- -1833`, then sglang::.
First serial attempt (pgid 56217, 11:38 +08) was killed by a CONTAINER RESTART at 2026-10-05 04:03:36 UTC (PID 1 restarted;
no OOM/dmesg entry; env unchanged) after only 2:26 of profiling -> no data. Partial logs: agentx/killed_restart_1005/.
Warmup note: new code returned 34/707 warmup requests at 210 s vs 48/707 baseline (tp2r_c64); ~13 min ready->profiling.
**Parallel attempt FAILED 2026-10-05 11:36 +08** (3 lanes, below): all 3 servers died at load, before any benchmark --
higher ranks (TP1 at TP2, TP2/TP3 at TP4) hung after "engram host table: dropped the page cache of 48 checkpoint files
(475 GiB) before pre-faulting the per-rank shard"; gloo monitoredBarrier 480 s timeout. Finished ranks took ~220 s to
load vs 108 s single-lane -> 8 ranks re-reading the checkpoint from NFS at once. Logs: agentx/failed_parallel_1005/.
Do not run >2 DSV4.1 servers concurrently on this node.
**Next:** wait for the 3600 s runs (~75 min incl. warmup), compare final JSON TTT / P90 against the baselines below;
if a point is worse than its baseline at all (user, 2026-10-05), rerun it alone on one lane with the node otherwise
idle (the baselines were single-lane runs), and judge regression only on the single-lane number.
**Files:** sglang `/sgl-workspace/sglang` affa261e3d (gateway-v0.3.1-10642, user-updated release image; dirty only from
image build: pyproject + aot .hip). aiter `/sgl-workspace/aiter` e7d2453f2 + same local edits as MAIN_REGRESS_1002.md.
Launcher `scripts/agentx_lane.sh` -> `agentx_colleague_run.sh`, Replay variant (prefill graph disabled + bounded replay).
Lanes: pid 45688 GPUs 0,1:8888 rel1004_tp2_c8; pid 45689 GPUs 2,3:8889 rel1004_tp2_c64; pid 45690 GPUs 4-7:8890
rel1004_tp4_c64. Kill: `kill -- -<pid>`, then sglang:: children. Progress /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/lane_<port>.txt.
**Repro (one lane; set TP/GPUS/PORT/POINTS per point):**
```bash
cd /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx && PYTHONPATH=/sgl-workspace/mori SRC=/sgl-workspace/sglang/python \
  SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0 EP_SIZE=1 DURATION=3600 \
  EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" \
  EXTRA_ARGS_rep="--fp8-gemm-backend aiter --enforce-shared-experts-fusion --enable-decoder-swa-bounded-replay --cuda-graph-backend-prefill disabled" \
  TP=2 GPUS=2,3 PORT=8889 POINTS="rel1004_tp2_c64:64:4:4096:0.85:rep" \
  setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_lane.sh > lane8889_rel1004.nohup 2>&1 < /dev/null &
```
**Pass:** TTT within ~1% and P90 within ~5% of the baseline (same noise band as MAIN_REGRESS_1002.md).

Caveat: these 3 run in parallel on one node (host engram tables share CPU DRAM / PCIe); baselines ran one lane at a time.

## Baselines (main 58f0d250ec + #42055, Replay, from results/agentx.md)

| point | conc / pdi / chunk / mem | baseline TTT / P90 | dir |
|---|---|---|---|
| TP2 c8 | 8 / 16 / 16384 / 0.70 | 29,760.8 / 237.2 | tp2r_c8 |
| TP2 c64 | 64 / 4 / 4096 / 0.85 | 142,243.2 / 52.0 | tp2r_c64 |
| TP4 c64 | 64 / 4 / 4096 / 0.85 | 88,391.8 / 66.5 | tp4s_c64_rep |

## Results (append-only; node + date per row)

| date (node) | point | TTT / P90 | vs baseline | notes |
|---|---|---|---|---|
| 2026-10-05 (mi355-4) | TP2 c64 rel1004_tp2_c64 | 140,810.0 / 50.5 | -1.0% / -2.9% | single lane, 7863/8570 ok, 0 err; live 59:00 141.6k->140.8k, intvty p50 67=67, TTFT p50 1369->1600 ms; within noise band |
| 2026-10-05 (mi355-4) | TP2 c8 rel1004_tp2_c8 | 29,751.0 / 239.6 | -0.03% / +1.0% | single lane, 1376/1463 ok (same as baseline), 0 err; TTFT p50 337->347 ms; no regression |
