Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# Proxy MoE fixed_seq_len sweep on MI455 (gfx1250), TP8 EP1 vs TP8 EP8

MI455 counterpart of `../fixed_seq_len/` (MI355X, owned by crsuse2-m2m-231): same client, one server per
point, kill-by-PID teardown, resumable DONE markers, same `report.py` columns (TTT/GPU, P90
interactivity = 1000 / p90 ITL).

## CONTINUE HERE

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
