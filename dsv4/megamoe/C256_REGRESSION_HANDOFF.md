# c256 regression check on the PR codebase after merging latest mainline

## CONTINUE HERE

**Status (2026-09-15 09:15 UTC): DONE -- the merge did NOT cause a regression.**
`/sgl-workspace/sglang-MegaMoE` @ `ff7f522abd` (merge commit `8fb290c7cb`,
"Merge branch 'main' into feat/aiter-megamoe-v2") measures **54,919 tok/s/chip**
at c256, **+1.7 % over the 53,991 baseline** -- inside the ~5 % replicate spread,
i.e. a null result, and on the favourable side of it. All three pass criteria met:
`errors=0` (0/13,603), intvty p90 = 15.9 (target ~16.0), and tok/s far above the
~51,300 no-regression floor.

```
RESULT_DIR  /workspace/results/megamoe-eplb-c256-postmerge
log         /shared_nfs/kk/pr35619/postmerge_c256.log
```

| arm | tok/s/chip | intvty p90 | ITL p90 | TTFT avg | TTFT p50 | cache hit | KV pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (pre-merge) | 53,991 | 16.0 | 62.5 ms | 17.16 s | 5.61 s | 95.3 % | 11,906,048 |
| **post-merge** | **54,919** | 15.9 | 63.0 ms | 14.99 s | 5.47 s | 95.3 % | 11,906,048 |
| delta | **+1.7 %** | -0.1 | +0.8 % | -12.6 % | -2.5 % | 0 | identical |

Comparability checks all pass: **KV pool identical** (11,906,048 -- the
memory-matching gate in `summary_table.py`'s legend), weights 130.74 GB identical,
EPLB `rebalance start` count identical (128 in both), ISL mean 116,372 vs 116,822.
Only ITL p90 is nominally worse (+0.5 ms, 0.8 %), which is far under the replicate
spread and is contradicted by TTFT avg improving 12.6 %; do not read either as a
real effect.

### PENDING (2026-09-15 20:2x UTC+8): B200-aligned c256 re-measure, BLOCKED on GPUs

`dsv4_fp4_mi355x_sglang_mtp.sh` has been edited to adopt three B200 values.
A new c256 baseline must be measured with these in place; the 53,991 / 54,919
numbers above do NOT apply to the aligned config.

| setting | was (MI355X) | now (B200-aligned) |
|---|---|---|
| `--prefill-decode-interval` | 10 | 24, or 20 when `CONC >= 160` |
| router `--policy` | `consistent_hashing` | `cache_aware` |
| `--load-balance-method` | `round_robin` | `total_requests` |
| `--balance-abs-threshold` | (absent) | 32, only when `CONC >= 160` |

At c256 this resolves to PDI=20 + `--balance-abs-threshold 32`. B200 gates the
20 on `CONC -eq 160` exactly; this port uses `>=` so c256 inherits it. An
explicit `PREFILL_DECODE_INTERVAL` env value still overrides. `bash -n` passes
and the branch logic was verified at CONC 64/128/160/256.

Backup of the pre-edit file: `/shared_nfs/kk/pr35619/mi355x_mtp.sh.bak.1228`.
NOTE the file was ALREADY dirty before this edit (+81 lines of MegaMoE support,
the PR body) -- do not `git checkout` it.

### RESULT (2026-09-16): B200-aligned c128 -- the same trade reproduces

`completed=9,825, cancelled=35, errors=0`, EPLB rebalanced 80 times,
`policy cache_aware` confirmed. KV pool 12,077,312 matches the pre-align c128 row
exactly, weights identical, ISL mean +2.6 %.

| arm | tok/s/chip | intvty p90 | ITL p90 | TTFT avg | TTFT p50 | cache hit |
|---|---:|---:|---:|---:|---:|---:|
| c128 pre-align | 34,713 | 19.8 | 50.6 ms | 5.64 s | 2.52 s | 95.1 % |
| **c128 B200-aligned** | **36,995** | 29.5 | 33.9 ms | 11.07 s | 4.01 s | 95.6 % |
| delta | +6.6 % | +49 % | **-33 %** | **+96 %** | **+59 %** | +0.5 pp |

**The headline: the TTFT-for-ITL trade is reproducible, not a c256 artifact.**
Both concurrencies move the same way and by a similar magnitude (ITL -30/-33 %,
TTFT p50 +187/+59 %), which is much stronger evidence than either arm alone.

On throughput, +6.6 % here and +4.7 % at c256 both sit at or inside the ~5 %
replicate spread. Neither is conclusive alone; that both are positive is
suggestive but they are **not replicates** -- the two aligned arms run different
configs (PDI 24 no-threshold vs PDI 20 + threshold). A real throughput claim
needs repeats of one config.

**Do not conclude "alignment reverses the c128 MegaMoE deficit."** It looks that
way (aligned MegaMoE 36,995 now beats DPA's 35,406, where pre-align it lost
34,713 vs 35,406), but the DPA row is **un-aligned**, so that comparison confounds
the MoE path with the launcher config. It needs a B200-aligned DPA c128 arm.

The absolute damage is much milder at c128: TTFT p50 4.01 s vs c256's 15.69 s.

### (completed) IN FLIGHT (2026-09-16 07:39 UTC+8): B200-aligned **c128**

Pairs with the pre-align c128 MegaMoE row (34,713) to test whether the
TTFT-for-ITL trade seen at c256 also appears at half the concurrency. At c128 the
`CONC >= 160` branch does NOT fire, so this arm runs
`--prefill-decode-interval 24` with **no** `--balance-abs-threshold` -- both
confirmed in the launch, along with `--load-balance-method total_requests` and a
clean tree (28 `sglang-MegaMoE/python` refs, 0 wrong-tree).

```
RESULT_DIR  /workspace/results/megamoe-eplb-c128-b200aligned
log         /shared_nfs/kk/pr35619/b200aligned_c128.log
```

ETA ~09:40 UTC+8. Register a `ROWS` entry before reading. Note this is a
*different* aligned config from the c256 arm (PDI 24 vs 20, no threshold), so the
two aligned arms are not a clean concurrency sweep of one configuration.

### RESULT (2026-09-16): B200-aligned c256 -- a latency-profile trade, not a speedup

Finished clean: `completed=14,013, cancelled=173, errors=0`, EPLB rebalanced 104
times, `--policy cache_aware` confirmed in `router.log`. KV pool 11,906,048 and
weights 130.74 GB are identical to the rows above, so it is comparable.

| arm | tok/s/chip | intvty p90 | ITL p90 | TTFT avg | TTFT p50 | cache hit |
|---|---:|---:|---:|---:|---:|---:|
| post-merge (pre-align) | 54,919 | 15.9 | 63.0 ms | 14.99 s | 5.47 s | 95.3 % |
| **B200-aligned** | **57,516** | 22.8 | 43.9 ms | 26.22 s | 15.69 s | 95.9 % |
| delta | +4.7 % | +43 % | **-30 %** | **+75 %** | **+187 %** | +0.6 pp |

How to read it, in order of confidence:

1. **TTFT got much worse and ITL got much better. Both are real** -- +187 % on
   TTFT p50 and -30 % on ITL p90 are far outside the ~5 % replicate spread.
2. **The throughput gain is NOT established.** +4.7 % sits inside that same ~5 %
   spread. Do not report this as "aligning made it faster".
3. **intvty p90 is not a second witness for the ITL win** -- it is 1/p90(ITL) by
   construction (same note as in `summary_table.py`). Counting 22.8 as
   independent corroboration double-counts one measurement.

**Attribution is blocked by design.** Three values moved at once
(prefill-decode-interval 10->20, load-balance-method, router policy +
balance-abs-threshold), so none of the deltas can be assigned to a single one.
`--prefill-decode-interval` is the prime suspect for the TTFT/ITL swap since it
directly sets how long decode runs before prefill is admitted, but that is a
hypothesis, not a measurement. To de-confound, re-run changing only PDI 10->20
and leaving the router/load-balance values at their pre-align settings.

**Whether this is an improvement depends on the SLO, and is your call:** it buys
decode smoothness and throughput headroom at the cost of a 15.7 s median
time-to-first-token, up from 5.5 s.

Leftover server after this run held 291 GB/GPU (trap 3 again) and was cleaned up.

### Reading the result

Its `ROWS` entry is registered, so reading it is one command:

```bash
python3 /workspace/claude-skills/agentx/summary_table.py    # row "MegaMoE+EPLB EP8 B200-aligned"
rg -o 'completed=[0-9,]+, cancelled=[0-9]+, errors=[0-9]+' /shared_nfs/kk/pr35619/b200aligned_c256.log | tail -1
```

If that row prints as partial/blank, the run did not finish -- check the log for
`Traceback`, and check whether the server outlived the launcher (trap 3) and is
still holding VRAM. Expect to clean up leftover `sglang::` processes; the node
was idle-claimed from another tenant, so leaving 300 GB pinned overnight is
antisocial.

Judge the result as a **new operating point, not a regression check**: the three
aligned values changed the configuration, so 53,991 and 54,919 do not apply. The
useful question is how this compares to 54,919 as a *config delta*, and whether
`errors=0` and intvty p90 stayed healthy.

**IN FLIGHT (as of 22:00 UTC+8):** launched 12:47:09 UTC (20:47 UTC+8), ETA ~14:37 UTC. Flags
confirmed in `sglang_command.txt`: `--prefill-decode-interval 20`,
`--load-balance-method total_requests`, and `--balance-abs-threshold 32` in the
router args. Log `/shared_nfs/kk/pr35619/b200aligned_c256.log`.

Earlier this evening the node was held by **another container** (PIDs
927xxx-939xxx, absent from our PID namespace, 244-289 GB each at 82-98 %
utilisation) -- never kill those. When it released, VRAM sat at ~53 GB/GPU with
**zero** KFD holders and crept down only ~125 MB/min, which extrapolates to 7 h.
That extrapolation is wrong: this reclaim is **plateau-then-cliff**, not linear,
and it dropped all 8 GPUs to the 284 MB baseline in one step. Do not linearly
extrapolate it and do not reach for a GPU reset -- per
`kimi-k3/aiter-optimization-tracker/EXECUTION_2026-08-10.md` per-GPU reset is
**unsupported on this system** (the documented fallback was a node reboot), even
though `/sys/class/drm/card0/device/reset` is writable and the container does
hold `cap_sys_admin`.

`wait_and_launch_c256.sh` automates the claim: it demands 3 consecutive idle
samples 180 s apart (guarding against the other tenant's between-phase dips),
then re-verifies preconditions before launching. Reusable for the next arm.

To run manually instead:

```bash
export PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python${PYTHONPATH:+:$PYTHONPATH}
cd /shared_nfs/kk/pr35619
CONC=256 ENABLE_EPLB=1 MEM_FRACTION_STATIC_DP_MEGAMOE=0.85 \
MORI_SHMEM_HEAP_SIZE=17179869184 DURATION=3600 \
RESULT_DIR=/workspace/results/megamoe-eplb-c256-b200aligned \
nohup bash agentx_dp8_c128_megamoe.sh > /shared_nfs/kk/pr35619/b200aligned_c256.log 2>&1 &
```

**Next (after the c256 question):** nothing required for the c256 question -- it is answered. The obvious
follow-up, if wanted, is the **c128** post-merge arm, since MegaMoE's sign flips
with concurrency (it *lost* 2.0 % at c128 pre-merge) and only c256 was re-measured.
The plain-DP c256 control was deliberately NOT re-run: the doc gates it on "only if
the MegaMoE arm moved", and it did not.

All three preconditions were verified before launch:
1. PYTHONPATH -- confirmed the trap is real: unset resolves to
   `/sgl-workspace/sglang/python`. Run log has 28 `sglang-MegaMoE/python`
   references and **0** wrong-tree references, so this run measures the PR tree.
2. aiter @ `ffa945f93` still carries both uncommitted fixes (`@cache` = 1,
   `torch.Stream` = 1).
3. GPUs were **not** idle at session start -- see "GPU reclaim" note below.

**Next:** when the run finishes, read `completed=/errors=` from the log, add a
`ROWS` entry to `agentx/summary_table.py`, and compare against 53,991.
**Pass criteria:** MegaMoE+EPLB c256 within noise of **53,991 tok/s/chip** (the
replicate spread on this node is ~5 %, so treat anything above ~51,300 as no
regression), `errors=0`, and intvty p90 near 16.0.

## The numbers to compare against

All at tp8 + dp8, `DeepSeek-V4-Pro-0813`, DSPARK spec decode, hicache ratio 1.5,
chunk/rank 8192, `DURATION=3600`, AgentX `semianalysis_cc_traces_weka_062126`.

| arm | conc | mem-frac | tok/s/chip | intvty p90 | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU pool | ISL mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MegaMoE+EPLB EP8 **post-merge** | 256 | 0.85 | **54,919** | 15.9 | 63.0 ms | 14.99 s | 5.47 s | 95.3 % | 81 % | 116,372 |
| MegaMoE+EPLB EP8 | 256 | 0.85 | **53,991** | 16.0 | 62.5 ms | 17.16 s | 5.61 s | 95.3 % | 78 % | 116,822 |
| DPA (no EP) | 256 | 0.92 | 50,600 | 14.5 | 69.2 ms | 18.36 s | 7.47 s | 95.3 % | 78 % | 116,035 |
| MegaMoE+EPLB EP8 | 128 | 0.85 | 34,713 | 19.8 | 50.6 ms | 5.64 s | 2.52 s | 95.1 % | 97 % | 105,652 |
| DPA (no EP) | 128 | 0.92 | 35,406 | 19.1 | 52.2 ms | 5.56 s | 2.49 s | 95.2 % | 97 % | 106,222 |

Regenerate the table with `python3 /workspace/claude-skills/agentx/summary_table.py`
(all four rows are registered in its `ROWS`). Raw results live in
`/workspace/results/{megamoe-eplb-dp8-ep8-c256-mf085,dp8-noep-c256-d3600,megamoe-eplb-dp8-ep8-c128-mf085,dp8-noep-c128-d3600}/`.

**Do not compare against `megamoe-eplb-c256-slotguard-probe2` (44,501).** That was a
`DURATION=900` probe; AgentX is duration-based, so a 900 s profile is not comparable
to a 3600 s one.

## Environment the numbers belong to

```
sglang   /sgl-workspace/sglang-MegaMoE @ cd70422d82   (PR #35619 head + 3 local commits)
aiter    /sgl-workspace/aiter @ ffa945f93, rebuilt from source, + 2 uncommitted fixes
model    /shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813
hw       8x MI355X (gfx950), ROCm
```

The three local sglang commits on top of the PR head:

| commit | what |
|---|---|
| `ceb1d7b3c2` | MegaMoE EPLB accuracy + startup fixes (already on the PR; gsm8k 0.816 -> 0.951) |
| `291afec535` | the three `base-a-test-cpu` CI failures |
| `cd70422d82` | drop the ungated prefill-delayer mixed-slot guard; gate SWA eviction on `holds_kv` |

## Two things that will bite right after the merge

**1. `/sgl-workspace/sglang-MegaMoE` is NOT the installed sglang.** The editable
install resolves to `/sgl-workspace/sglang/python`. Every runner must export
PYTHONPATH, or it will silently measure the wrong tree:

```bash
export PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python${PYTHONPATH:+:$PYTHONPATH}
python3 -c "import sglang; print(sglang.__file__)"   # must print sglang-MegaMoE
```

`benchmarks/benchmark_lib.sh` prepends and preserves `$PYTHONPATH`, so exporting it
in the wrapper is enough.

**2. aiter carries two uncommitted fixes that are NOT upstream.** Both are still
absent from `aiter@ffa945f93`:

- `aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py`:
  `@lru_cache(maxsize=32)` -> `@cache`. **Without this, every AgentX run dies
  55-70 s into warmup** with `Memory access fault by GPU node-N ... Reason: Unknown`.
  AgentX is variable-length, so it blows past 32 compile configs; fixed-length
  benchmarks and gsm8k stay under the ceiling and look fine, which is why this hides.
- `csrc/cpp_itfs/torch_utils.py`: `torch.cuda.Stream` -> `torch.Stream`.

Verify before every run, and never `git stash` them to update aiter:

```bash
rg -c '@cache' /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py  # 1
# to restore: git apply -3 /workspace/claude-skills/agentx/aiter_fp4_prefill_cache_fix.patch
```

If aiter itself is rebuilt/updated as part of this work, read §2b of
`PR35619_METHOD_20260914.md` first: `aiter/jit/core.py` only rebuilds a module when
its `.so` is **missing**, so a `git pull` leaves stale device binaries and you
measure new Python over old kernels.

**3. GPU reclaim after a leftover server takes ~12 minutes.** At the start of the
post-merge session the GPUs were still held at ~300 GB each by the *completed*
`megamoe-eplb-c256-slotguard-probe2` run -- trap 3 in `agentx/SKILL.md`, "the
server survives the launcher". After per-PID `kill -9` on `sglang.launch_server`
and `sglang::`, VRAM sat at a flat ~29 GB/GPU for ~12 min with
`rocm-smi --showpids` and `amd-smi process` both reporting **no** holders, then
dropped to the 284 MB idle baseline within one 2-min poll. Do not interpret the
29 GB plateau as a leak and do not launch on top of it -- `mem-fraction 0.85`
(245 GiB) plus the 16 GiB mori heap needs 261 of the 288 GiB, so a 29 GB
squatter would OOM the run.

Worth noting: the probe2 scheduler's environ had **no** `PYTHONPATH` at all, so
that 44,501 probe may have measured `/sgl-workspace/sglang` rather than the PR
tree. Another reason not to compare against it.

## Exact commands

```bash
export PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python${PYTHONPATH:+:$PYTHONPATH}
cd /shared_nfs/kk/pr35619

# MegaMoE + EPLB, c256  (the arm under test)
CONC=256 ENABLE_EPLB=1 MEM_FRACTION_STATIC_DP_MEGAMOE=0.85 \
MORI_SHMEM_HEAP_SIZE=17179869184 DURATION=3600 \
RESULT_DIR=/workspace/results/megamoe-eplb-c256-postmerge \
nohup bash agentx_dp8_c128_megamoe.sh > /shared_nfs/kk/pr35619/postmerge_c256.log 2>&1 &

# plain DP c256 control, only if the MegaMoE arm moved
CONC=256 DURATION=3600 RESULT_DIR=/workspace/results/dp8-noep-c256-postmerge \
nohup bash agentx_dp8_c128.sh > /shared_nfs/kk/pr35619/postmerge_c256_dp.log 2>&1 &
```

Timing: ~14 min load + graph capture, ~2,250 s warmup (2,845 requests at c256), then
3600 s profiling. About 1h50 per arm. `MORI_SHMEM_HEAP_SIZE` must stay 16 GiB —
`mem-fraction 0.85` plus the default 40 GiB heap does not fit (the heap is charged
outside that budget; see the mori sizing section in `agentx/SKILL.md`).

## Reading the result

```bash
D=/workspace/results/megamoe-eplb-c256-postmerge
rg -o 'completed=[0-9,]+, cancelled=[0-9]+, errors=[0-9]+' /shared_nfs/kk/pr35619/postmerge_c256.log | tail -1
rg -c 'rebalance start' $D/server.log
python3 /workspace/claude-skills/agentx/summary_table.py   # after adding a ROWS entry
```

## If it looks stuck, check for the dead-scheduler pattern first

An intermittent EPLB rebalance deadlock (seen 1 of 3 runs at c256) presents as a
*slow* run, not a crash: aiperf `in_flight` frozen with `errors=0`, `/metrics` still
answering 200, last batch line tens of minutes stale, and **VRAM back at the ~298 MB
idle baseline** with `sglang::schedul <defunct>`. Nothing is logged. Full detection
recipe and cleanup in `agentx/SKILL.md`, section "Intermittent: EPLB rebalance
deadlocks and the run looks merely slow". A wedged run leaves 8888/8889 bound, which
makes the next launch die with `[Errno 98] Address already in use`.

Also worth knowing before declaring a regression: the first MegaMoE run after any
aiter rebuild spends 15-30 min compiling FlyDSL kernels and looks hung (log stops
after the NCCL line, VRAM flat, schedulers `R` with `stime` ~7x `utime`).

## Possible merge-induced breakages to expect

Both bit us on the 2026-09-14 merge and may recur:

- `--cuda-graph-max-bs` was split into `--cuda-graph-max-bs-decode` / `-prefill`.
  The InferenceX launcher already emits the new name; `run_sgl_dsv4_unified.sh`
  (fixed-length) still emits the old one and dies with `ambiguous option`.
- `MLPSyncBatchInfo` gained `prefill_cuda_graph_max_prefix_len`, which the PR's
  `test_aiter_megamoe_rank_sync.py` fixture had to add. Expect more of this class:
  the PR's tests build `SimpleNamespace` fixtures that go stale whenever main widens
  an interface.
