Owner node: mi355-4

# DeepSeek-R1-0528 MTP — fixed-length ISL 70k / OSL 300, conc 4/8/16/32/64

## CONTINUE HERE

**Status:** done — run2 complete, 0 failed requests, results in table below (2026-10-05). Run1 (np=conc*4, default MRR) aborted after c4/c8.
**Next:** none requested. Workload is prefill-bound (~27k total tok/s plateau from c8); TPOT grows ~linearly with conc due to 70k prefills interleaving with decode.
**Run2 settings:** server restarted per conc with `--max-running-requests=conc`, `--num-prompts=conc*8`,
`--warmup-requests 2` (warmup = first 70k prompt, output capped at 32, not counted). Script: `sweep2.sh`.
Note: KV holds max_total_num_tokens=3217291 at mem 0.8 (~45 x 70.3k seqs), so c64 cannot be fully resident.
**Files:** scripts in `dsv-r1/scripts/` (copies; the live run executes from
`/shared_nfs/kk/dsr1-mtp-70k/`). Logs stay on NFS: run1 `/shared_nfs/kk/dsr1-mtp-70k/c<N>.log`,
run2 `/shared_nfs/kk/dsr1-mtp-70k/run2/{server_c<N>,c<N>}.log` + `c<N>.jsonl`.
Cookbook snapshot used for the command choice: `dsv-r1/cookbook-DeepSeek-R1.md`.
**Repro:**
```bash
cd /shared_nfs/kk/dsr1-mtp-70k && nohup ./launch.sh > server.log 2>&1 &
# after ready:
./sweep.sh; rg "Output token throughput|Total token throughput|Mean TTFT|Mean TPOT|Accept length" c*.log | cut -c1-160
```
**Pass criteria:** all 5 concurrencies complete with 0 failed requests.

## Command choice (from SGLang cookbook DeepSeek-R1 page)

The cookbook has no MI355X-tuned MTP recipe: its MI355X Pareto configs (fp8/fp4,
low-latency/high-throughput) are all non-MTP, and MTP appears only in the generic
command generator as `--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.
Used: MI355X fp8 low-latency config + those MTP flags.
- Env: `SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4`
- `--attention-backend aiter --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8
  --disable-radix-cache --chunked-prefill-size 196608 --max-prefill-tokens 196608
  --cuda-graph-max-bs-decode 128` (plain `--cuda-graph-max-bs` is ambiguous in this build)
- Dropped `--num-continuous-decode-steps 4`: defined but unused in this sglang
  build (0.5.21.dev20261004, affa261e3d). Dropped `--enable-symm-mem` (cookbook marks it unstable).
- Model: `/shared_nfs/deepseek-ai/DeepSeek-R1-0528` (FP8, nextn layers = 1).

## Results (append-only; node + date per row)

| node | date | conc | prompts | out tok/s | total tok/s | mean TTFT ms | mean TPOT ms | accept len |
|------|------|------|---------|-----------|-------------|--------------|--------------|------------|
| mi355-4 | 2026-10-05 | 4 (run1, MRR default=48) | 16 | 98.95 | 23186 | 4978 | 22.68 | 2.97 |
| mi355-4 | 2026-10-05 | 8 (run1, MRR default=48) | 32 | 70.67 | 16560 | 19352 | 48.38 | 3.02 |
| mi355-4 | 2026-10-05 | 4 (run2, MRR=4) | 32 | 105.16 | 24644 | 4119 | 23.95 | 3.00 |
| mi355-4 | 2026-10-05 | 8 (run2, MRR=8) | 64 | 111.74 | 26186 | 5829 | 52.05 | 2.99 |
| mi355-4 | 2026-10-05 | 16 (run2, MRR=16) | 128 | 115.35 | 27030 | 8320 | 110.85 | 2.97 |
| mi355-4 | 2026-10-05 | 32 (run2, MRR=32) | 256 | 116.99 | 27414 | 12127 | 233.00 | 2.95 |
| mi355-4 | 2026-10-05 | 64 (run2, MRR=64) | 512 | 114.64 | 26864 | 59037 | 358.39 | 2.95 |
