Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# RCCL DDA warmup hang: TP4 repro + test of the fabric-init memset fix

Background and the TP8 repro: `RCCL_DDA_HANG_REPRO.md`. Since the docker restart this container sees only 4 GPUs (0-3), so this first tries TP4.

## CONTINUE HERE

**Status (2026-10-26, h13-17):** DONE at 4 ranks: comm-init stress hangs 5/5 on `4ca8623`, 0/5 (750 iters) on fix branch `9c52e34`. **The SDK librccl is now `9c52e34` (node-local, user decision 2026-10-26).** Copies: `/workspace/proxy_work/rccl_9c52e34/librccl.so.1.0`, `/workspace/proxy_work/rccl_4ca8623_librccl.so.1.0`; install either with `rccl_swap.sh install <lib>`.
**Next:** TP8 server 6/6 clean starts with DDA on (no `RCCL_DDA_ENABLE=0`) when 8 GPUs return; if clean, drop `RCCL_DDA_ENABLE=0` from the best config. Parent-commit control build skipped (user, 2026-10-26).
**Files:** `scripts/mi455_hang_dump.sh` (now takes `TP=`, `KEEP=1` for a hang tally, `ROCGDB=1`); `/workspace/proxy_work/rccl_swap.sh` (restore stock 2.30.7).
**Repro:**
```bash
cd /tmp && TP=4 KEEP=1 ROCGDB=1 TRIES=8 bash /workspace/claude-skills/proxy-moe/scripts/mi455_hang_dump.sh <run-tag>
```
**Pass criteria:** baseline shows hangs (in `ddaAllReduceFlatLL`); with the fix, 0 hangs.

## Fix under test

- Branch `users/wenkaidu/dda-fabric-init-memset-race` of ROCm/rocm-systems, tip `9c52e34` (2026-10-25), RCCL 2.32.3. Parent `5512a98` is main.
- Change: `src/algorithms/dda/fabric/fabric_init.cu` `ncclDdaFabricCommInit` adds `cudaStreamSynchronize(nullptr)` plus `bootstrapBarrier` after zeroing the LL scratch and epoch cells. Cause given in the commit: the null-stream memset is not ordered against non-blocking collective streams. A late epoch reset gives one rank LL flag 1 while its peers use 2, so the peers spin. That matches our rocgdb picture: all ranks in `ddaAllReduceFlatLL`, and one rank (TP1) partly retired.
- Build (node-local): `/tmp/rs-dda/projects/rccl` -> `/tmp/rccl_fix_install`, log `/workspace/proxy_work/rccl_fix_9c52e34_build.log`.

## Results (append-only; node + date on every row)

| Node | Date | RCCL | TP | Env | Hung starts | Run dir |
|---|---|---|---|---|---|---|
| h13-17 | 2026-10-26 | 2.32.3 `4ca8623` | 4 | `NCCL_CUMEM_ENABLE=1` (server, MAX_BS 64) | **0 of 8** | `/shared_nfs/kk/results/proxy_12l/dda_hang_tp4_rccl2323_h13-17_1026/` |

TP4 server starts do not reproduce the hang (TP8 was 4 of 7). Next: `scripts/rccl_comminit_stress.sh` (fresh communicator + 2x 192 KB AR per iteration, variants plain / busy null stream / host jitter).

### Comm-init stress test reproduces it at 4 ranks (2026-10-26, h13-17)

- `scripts/rccl_comminit_stress.py`: per iteration a fresh group (full init, NOT `ncclCommSplit`: torch splits when `device_id` is bound, and a split comm does not run `ncclDdaFabricCommInit`), then 2x AllReduce bf16 98304 on a side stream. Verified 1 `ncclDdaFabricCommInit` per rank per iteration.
- RCCL 2.32.3 `4ca8623`, `NCCL_CUMEM_ENABLE=1`, `--jitter-ms 50`: **HUNG at iter 40** (all 4 ranks "init done", no AR completion). rocgdb: every rank 768 waves in `dda::common::ddaAllReduceFlatLL<__hip_bfloat16, 4>`, the same signature as the TP8 server hang. Run dir `/shared_nfs/kk/results/proxy_12l/comminit_stress_rccl2323_h13-17_1026b/jitter/`.
- (An earlier run with `device_id`, i.e. split comms, passed 150/150; it did not exercise the init. Discard it.)
- A/B driver (node-local): `/workspace/proxy_work/dda_fix_ab.sh` -> `comminit_ab_{base,fix}_h13-17_1026/`. It swaps the fix lib in and restores `4ca8623` at the end.

### A/B result: the fix resolves the hang (2026-10-26, h13-17)

4 ranks, `NCCL_CUMEM_ENABLE=1`, 150 iters per run, TMO 420 s (exit 124 = hang). Run dirs `/shared_nfs/kk/results/proxy_12l/comminit_ab_{base,fix}_h13-17_1026/`.

| Node | Date | RCCL | plain | busy 20 ms | jitter 50 ms x3 | Hung runs |
|---|---|---|---|---|---|---|
| h13-17 | 2026-10-26 | 2.32.3 `4ca8623` (base) | hang @56 | hang @111 | hang @20 / @108 / @86 | **5 of 5** (mean ~76 iters to hang) |
| h13-17 | 2026-10-26 | 2.32.3 `9c52e34` (fix branch) | 150/150 | 150/150 | 150/150 x3 | **0 of 5** (750 iters, all results correct) |

- At the base rate (~1 hang per 77 iters), 750 clean iters by chance is about e^-9.7 < 0.01%.
- Caveat: the fix build is the branch tip (main as of 2026-10-25 + the 6-line fix), not `4ca8623` + the fix only. Other main commits since `4ca8623` (e.g. #13503 DDA LL reduce-scatter) are in it too. A clean attribution would build the parent `5512a98`; not done.
- RCCL restored to `4ca8623` after the A/B (node-local).
