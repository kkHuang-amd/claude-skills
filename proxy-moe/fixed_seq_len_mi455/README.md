Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Proxy MoE fixed_seq_len sweep on MI455 (gfx1250), TP8 EP1 vs TP8 EP8

MI455 counterpart of `../fixed_seq_len/` (MI355X, owned by crsuse2-m2m-231): same client, one server per
point, kill-by-PID teardown, resumable DONE markers, same `report.py` columns (TTT/GPU, P90
interactivity = 1000 / p90 ITL).

## CONTINUE HERE

**Status (2026-10-28 18:00, h13-17): OVERNIGHT SWEEP 1028 COMPLETE -- new 34/34 + old 34/34 points, 0 failures, no reruns needed.**
Full tables: `RESULTS_1028.md` (`python3 cmp_1028.py > RESULTS_1028.md`; per-point new / old / 10-23, geomeans, cross-platform).
Geomean TTT/GPU gains (new vs same-night old config; old vs 10-23 = 0.99-1.01x, so node state matches 10-23):
| shape | EP1 new/old | EP8 new/old | MI455 best vs MI355X (10-23 -> 10-28) | vs GB300 (10-23 -> 10-28) |
|---|---|---|---|---|
| 1024/1024 | 1.05x | 1.03x | 0.41x -> 0.43x | 0.37x -> 0.38x |
| 8192/1024 | 1.16x | 1.16x | 0.36x -> 0.42x | 0.31x -> 0.37x |
| 70000/300 | 1.90x | 2.12x | 0.21x -> 0.44x | 0.17x -> 0.37x |
(MI455 = TP8, MI355X / GB300 = TP4, all 12 layers, per-GPU throughput.) Next (user, 10-29): try `--enable-aiter-allreduce-fusion`.

**History: (2026-10-28 13:56, h13-17) OVERNIGHT SWEEP 1028 RUNNING** (runner `run_overnight_1028.sh`, nohup setsid, pid 906249,
log `/shared_nfs/kk/results/proxy_12l/overnight_1028.log`; user: "不要讓它中斷，早上希望可以有結果"; the agent monitors and reruns failures).
Preflight (MODE=new, 4 points, all 8/8 or 512/512 completed, already DONE in the new OUT): TTT/GPU vs 10-23 -- EP1 70k c2
1789 -> 3206, EP8 70k c2 1815 -> 3522, EP1 1k c64 475 -> 510, EP8 1k c64 486 -> 521; median TTFT 70k c2 4710 -> 2285 (EP1),
4485 -> 1859 (EP8). Gather-reduce EP compact = default auto (`AITER_GR_EP_COMPACT` unset -> 3) in MODE=new, 0 in MODE=old.
Plan (confirmed by the user 2026-10-28): baseline = the 10-23 sweep; MODE=new = current best config; MODE=old = 10-23 config rerun
the same night on the same node (same-condition baseline); `--enable-aiter-allreduce-fusion` NOT in this sweep (try 10-29).
- Driver `sweep_mi455_1028.sh` (MODE=new|old, OUT=...): same 34 points / client as 10-23, client timeout 3600 s, up to 3 tries per
  point, failures -> `$OUT/FAILED.txt`, resumable (DONE). Runner `run_overnight_1028.sh`: new -> old -> `REPORT.md` for each.
- OUT new: `/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_allon_h13-17_1028`; old:
  `/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_oldcfg_h13-17_1028`.
- MODE=new: `/workspace/proxy_work/allon_tp8.env` (aiter env, `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`, router FlyDSL + bf16x3,
  fused QK norm/RoPE/KV, aiter LM head, FlyDSL PA decode + extend attn, bf16 prefill GEMM CSV, custom AR <= 1 MiB, CUMEM, DDA on)
  + `--attention-backend aiter --page-size 64`; EP1 = `ep1_grouped_fmoe_gfx1250_v3.csv` + prefill tile rule; EP8 = prefill tile
  rule only (no CSV); local aiter (uncommitted, h13-17): tile rule, gap-tile skip, LDS race fix, gather-reduce EP compact (auto).
- MODE=old: 10-23 env (CUMEM, DDA off, router FlyDSL, fused QK, AR 1 MiB), triton attention, v3 / ep8 v1 CSVs, with
  `AITER_TDM_PREFILL_TILE_RULE=0 AITER_GR_EP_COMPACT=0` (today's code otherwise; race fix cannot be switched off).
- Monitor: `grep -E '^===|->|FAILED|WARNING' $OUT/sweep.log | tail`; `cat $OUT/FAILED.txt`; progress `find $OUT -name DONE | wc -l` /34.
- Resume after a crash: `nohup setsid bash run_overnight_1028.sh > /shared_nfs/kk/results/proxy_12l/overnight_1028.log 2>&1 &`.
- Morning: `REPORT.md` in both OUT dirs; compare new vs old vs 10-23 vs MI355X / GB300 (`cmp_platforms.py`, RESULTS.md tables).

**MODE=new finished 2026-10-28 15:37: 34/34 points, 0 failures** (MODE=old started right after). TTT/GPU 10-23 -> 10-28 new:
| shape | EP1 range (x) | EP8 range (x) | med TTFT |
|---|---|---|---|
| 1024/1024 c2..64 | 1.00-1.07 | 0.96 (c2) - 1.07 | -28..-36% |
| 8192/1024 c2..64 | 1.07-1.28 (c16 741.5 -> 926.4, c64 1264 -> 1622) | 1.03-1.33 (c64 1315 -> 1755) | -38..-49% |
| 70000/300 c2..32 | 1.79-1.99 (c32 1860 -> 3696) | 1.94-2.26 (c32 1962 -> 4442) | -51..-64% |
P90 interactivity equal or better at every point except EP8 1k c2 (157.5 -> 153.3) and EP8 8k c2 (151.1 -> 148.8).
Same-night old-config rerun pending (separates code/config gains from node state).

**Previous status (2026-10-23 16:10):** 10-23 sweep complete, below.

**Status (2026-10-23 16:10):** SWEEP COMPLETE, 34/34 points, no warnings. Tables and observations in
`RESULTS.md`. Next: EP1 prefill CSV rows (token 1024/8192), EP8 decode rows; then re-run affected points
(delete their DONE).
**Progress:** `find $OUT -name DONE | wc -l` (of 34); `grep -E '^===|->|FAILED' $OUT/sweep.log | tail`.
**Report:** `python3 ../fixed_seq_len/report.py $OUT --md` -> `RESULTS.md`.
**Stop:** `kill $(cat $OUT/sweep.pid)`, then kill the PID in the current point's `server.pid`.
**Restart / resume:** `nohup setsid bash sweep_mi455.sh > $OUT/sweep.log 2>&1 &` (DONE points are skipped).
`OUT=/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_h13-17`.

## Points

| Shape (ISL/OSL) | CONC | np | warmup |
|---|---|---|---|
| 1024/1024 | 2 4 8 16 32 64 | CONC*8 | CONC*2 |
| 8192/1024 | 2 4 8 16 32 64 | CONC*8 | CONC*2 |
| 70000/300 | 2 4 8 16 32 | CONC*4 | CONC*2 |

Recipes `tp8ep1`, `tp8ep8`, both on the 12-layer proxy (`/workspace/proxy_work/proxy_12l`; MI455 has no
24-layer checkpoint, so these are NOT comparable 1:1 with the MI355X TP8 = 24-layer tables).

## Configuration (all MI455 optimizations as of 2026-10-23)

- sglang `HaiShaw/sglang@feat/proxy-model-gfx1250-1023` @ `204430a4f0`; aiter
  `kkHuang-amd/aiter@feat/proxy-model-gfx1250-1023` @ `dae49be2b` (C++ JIT rebuilt from source);
  RCCL 2.32.3 swapped into the container (node-local).
- Both recipes: `NCCL_CUMEM_ENABLE=1 RCCL_DDA_ENABLE=0` (CUMEM transport, DDA off: DDA hangs),
  `SGLANG_OPT_LAGUNA_ROUTER_FLYDSL=1` (fp32 router GEMV/WMMA), `AITER_CUSTOM_AR_MAX_SIZE=1048576`
  (aiter gfx1250 custom AR up to 1 MiB, RCCL above), `SGLANG_OPT_LAGUNA_FUSED_QK_NORM_ROPE_KV=1`
  (fused decode q/k norm + RoPE + fp8 KV store; bit-identical outputs).
- `tp8ep1`: `AITER_CONFIG_GROUPED_FMOE=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv`
  (tiles, n_warp=2, gemm2 persistent 384 for tokens <= 8).
- `tp8ep8`: `AITER_CONFIG_GROUPED_FMOE=/workspace/proxy_work/ep8_grouped_fmoe_gfx1250_v1.csv`
  (gemm1 persistent 384 for tokens <= 16; neutral in the c=2 A/B).
- Server: `--attention-backend triton --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.80
  --chunked-prefill-size 8192 --context-length 73728 --max-running-requests CONC
  --cuda-graph-max-bs-decode CONC`. Not used: `--enable-aiter-allreduce-fusion` and the bf16x3 router
  (MI355X recipe, unverified on gfx1250).
- Client: `sglang.bench_serving --backend sglang --dataset-name random-ids --tokenize-prompt
  --random-range-ratio 1.0 --request-rate inf`, run from /tmp.

## Files

`serve_mi455.sh` (one server; logs the effective env into server.log), `sweep_mi455.sh` (driver;
teardown waits for VRAM < 5% via `amd-smi metric --mem-usage`, no rocm-smi in this image).
Details and the optimization history: `../MI455_RUN_h13-17.md`.
