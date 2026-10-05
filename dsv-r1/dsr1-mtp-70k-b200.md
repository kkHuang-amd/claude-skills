Owner node: hungry-hippo-fin-03-1

# DeepSeek-R1-0528 MTP on 8xB200 — ISL 70k / OSL 300, conc 4/8/16/32/64

Counterpart of the MI355X run in `dsr1-mtp-70k.md` (owner mi355-4), for a B200 vs MI355X comparison.
Do not edit that doc; MI355X numbers live there.

## CONTINUE HERE

**Status (update 16:40):** run3 COMPLETE (chunk 65536, mem 0.82), all 0 failed. Total tok/s +3.8..4.4% vs
run1; B200/MI355X total tok/s 0.767 / 0.753 / 0.744 / 0.736 / 0.750 (c4..c64). Nothing running.
**Status (update 15:30):** run2 (`CHUNK=196608`, mem 0.82) FAILED: OOM on first prefill (needs +22.7 GB,
~4 GB free per GPU); stopped after c4. Fitting 196608 would need mem ~0.68 (KV ~15 x 70k seqs). User chose
chunk 65536 at mem 0.82 instead: run3 (`CHUNK=65536 RUN=3`, out `sweep_run3.out`) started 15:30.
**Status (update 15:10):** run1 COMPLETE, c4-c64 all 0 failed. Total tok/s B200/MI355X: 0.737 / 0.724 /
0.713 / 0.709 / 0.723 (c4..c64). Next (optional): `CHUNK=196608 RUN=2` like-for-like run.
**Status (update 14:22):** c4-c32 done. c64 of the first sweep died at 109/512 (13:19, container/agent
backend outage; partial logs in `run1/aborted/`). c64 re-run alone started 14:22 (`CONCS=64 RUN=1`,
out `sweep_run1_c64.out`). After it: optional `CHUNK=196608 RUN=2` for like-for-like prefill chunking.

B200 / MI355X (run2) at same settings: out tok/s 0.74 / 0.72 / 0.71 / 0.71 (c4/8/16/32); TTFT 1.85x / 1.60x /
1.28x / 2.48x; TPOT 1.08x / 1.30x / 1.43x / 1.22x. Confounders: prefill chunk 32768 vs 196608, KV capacity
~26 vs ~45 x 70k seqs (c32 queues on B200), accept len ~2.8 vs ~2.97.

**Status (update 11:50+):** run1 sweep RUNNING on hungry-hippo-fin-03-1 with weights at
`/shared_nfs/deepseek-ai/DeepSeek-R1-0528`; out `/shared_nfs/kk/dsr1-mtp-70k-b200/sweep_run1.out`, logs `run1/`.
Server startup ~11 min (sweep waits max 15 min). KV max_total_num_tokens=1889664 (~26 x 70k seqs at mem 0.82).
**Earlier status:** blocked on weights (2026-10-05, hungry-hippo-fin-03-1). Env OK: 8xB200 visible, all flags
exist in local sglang 0.5.21. The FP8 R1-0528 copies on /shared_nfs were incomplete (HF cache: 12 blobs,
157/163 shard symlinks broken; `/shared_nfs/models--DeepSeek-R1-0528`: 6 shards). No sweep has run yet.
User is downloading the weights themselves (agent's unauthenticated `hf download` stopped; it left
`*.incomplete` blobs in the HF cache blobs dir, which a resumed `hf download` reuses).
**Next:** when 163 shards resolve (`find $M -xtype l | wc -l` == 0), start the sweep with
`MODEL=$M D=/shared_nfs/kk/dsr1-mtp-70k-b200 RUN=1`, where
`M=/shared_nfs/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52`.
Container: needs `--gpus all`, and host `WORKSPACE=/SFS-aGqda6ct/amd/kk` so skills land at /workspace/claude-skills.
**Files:** `dsv-r1/scripts/launch_b200.sh`, `dsv-r1/scripts/sweep2_b200.sh`.
Env knobs: `MODEL` (local checkpoint path), `D` (log dir, default `/shared_nfs/kk/dsr1-mtp-70k-b200`),
`RUN` (run number → `$D/run<N>/`), `CONCS`, `CHUNK` (prefill chunk, default 32768), `PORT`.
**Repro:**
```bash
MODEL=/path/to/DeepSeek-R1-0528 nohup /workspace/claude-skills/dsv-r1/scripts/sweep2_b200.sh > /tmp/b200_sweep.out 2>&1 &
# progress / results:
cat /tmp/b200_sweep.out
rg "Output token throughput|Total token throughput|Mean TTFT|Mean TPOT|Accept length" $D/run1/c*.log | cut -c1-160
```
**Pass criteria:** all 5 concurrencies complete with 0 failed requests (c64, maybe c32, will not be fully
KV-resident — record `max_total_num_tokens` from the server log).

## Command choice (from SGLang cookbook DeepSeek-R1 page)

Cookbook has no B200-tuned MTP recipe: all B200 Pareto configs are non-MTP, and MTP only appears in the
generic command generator as EAGLE 3/1/4 (DeepSeek-V3 page also lists 3/1/4 as the default). Same method as
MI355X: platform fp8 8-GPU low-latency Pareto config + MTP flags.
- Env: `SGLANG_ENABLE_JIT_DEEPGEMM=false`
- `--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.82 --chunked-prefill-size 32768
  --max-prefill-tokens 32768 --cuda-graph-max-bs-decode 128 --scheduler-recv-interval 10
  --fp8-gemm-backend flashinfer_trtllm`
- Aligned with MI355X: added `--disable-radix-cache`; dropped cookbook `--stream-interval 30` (default 1, as
  MI355X); no `--enable-symm-mem`; `--max-running-requests` = conc per server restart.
- Model: FP8 `deepseek-ai/DeepSeek-R1-0528` (same weights as MI355X; not the FP4 checkpoint).

Known difference vs MI355X: prefill chunk 32768 (3 chunks per 70k prompt) vs 196608 on MI355X (1 chunk).
Affects TTFT. Optional extra run with `CHUNK=196608 RUN=2` for a like-for-like comparison.

## Results (append-only; node + date per row)

| node | date | conc | prompts | out tok/s | total tok/s | mean TTFT ms | mean TPOT ms | accept len |
|------|------|------|---------|-----------|-------------|--------------|--------------|------------|
| hungry-hippo-fin-03-1 | 2026-10-05 | 4 (run1, MRR=4) | 32 | 77.52 | 18166 | 7615 | 25.75 | 2.83 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 8 (run1, MRR=8) | 64 | 80.86 | 18949 | 9298 | 67.44 | 2.81 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 16 (run1, MRR=16) | 128 | 82.19 | 19259 | 10647 | 158.89 | 2.73 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 32 (run1, MRR=32) | 256 | 82.91 | 19428 | 30029 | 284.95 | 2.82 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 64 (run1 re-run alone, MRR=64) | 512 | 82.85 | 19415 | 138712 | 292.56 | 2.80 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 4 (run3, CHUNK=65536, MRR=4) | 32 | 80.64 | 18897 | 7380 | 24.48 | 2.90 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 8 (run3, CHUNK=65536, MRR=8) | 64 | 84.19 | 19728 | 8596 | 65.88 | 2.83 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 16 (run3, CHUNK=65536, MRR=16) | 128 | 85.81 | 20109 | 10604 | 151.18 | 2.80 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 32 (run3, CHUNK=65536, MRR=32) | 256 | 86.06 | 20168 | 29363 | 273.07 | 2.77 |
| hungry-hippo-fin-03-1 | 2026-10-05 | 64 (run3, CHUNK=65536, MRR=64) | 512 | 86.01 | 20155 | 133938 | 280.89 | 2.79 |
