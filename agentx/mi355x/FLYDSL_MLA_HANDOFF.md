# FlyDSL MLA decode kernel — handoff

Everything below is measured on this node unless it says otherwise. The point
of this doc is that the next session can pick the kernel up without re-deriving
any of it.

**Read "CONTINUE HERE" and nothing else unless it sends you there.** It is
self-contained as of session 6: the live task, the config that ships, the
repro commands, the geometry sweep, the profile and the ablation ladder are all
in it.

**Everything from "Gate 1 result (2026-09-20)" down is history, and much of it is
now WRONG.** It was written when the kernel folded the verify window into M, an
assumption production metadata never satisfied (see the root cause in CONTINUE
HERE), so **every ratio and every ladder below is void** — they were measured on
a kernel reading the wrong KV for six of every seven tokens, and on a harness
that built its own too-friendly metadata. What survives from the history is the
*mechanisms*: "Step 9" (an LDS bank conflict worth 50 µs), "Step 12" (two
baseline-measurement errors), "How to pick the next step", "FlyDSL authoring
traps" and "Method traps" — those are the parts worth reading, and they are why
this doc exists.

## CONTINUE HERE

**Status (end of session 9): GATE 2 IS PASSED AND BOTH GATES ARE NOW CLOSED.
On the real AgentX c128 recipe, matched-bs reads `dstep = -0.91 ms`
(agg p50 114.29 -> 112.94, -1.2 %) over 24 cells at weight 6121 — and the
production band is consistently signed: bs 9-15, the heaviest cells, run
-0.9 %, -1.6 %, -2.0 %, -2.4 %, -2.0 %, -2.9 %, with 17 of 24 cells on-faster
(bs=8 at +2.7 % is the one notable outlier). That magnitude is what the kernel
arithmetic predicts — ~12 % off a slice that is ~7 % of the step — so the win
is real and cashable, just small.**

**The AgentX headline cannot see it, and that is a property of the instrument,
not of the kernel.** tok/s/chip reads +0.98 % and ITL p90 -4.0 %, both inside
their own measured noise floors (5.67 % and 7.01 %, `agentx` skill, "the noise
floor is now known per metric"). **Do not run another AgentX A/B to resolve a
kernel of this size** — quote matched-bs Δstep, and use the headline only to
show nothing regressed.

| mode | conc | chunk/rank | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier | GPU pool | ISL mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flydsl OFF | 128 | 8192 | 40,240 | 30.6 | 32.7 ms | 9.11 s | 3.77 s | 95.72% | 95.71% | 100% | 109,766 |
| **flydsl ON** | 128 | 8192 | **40,635** | **31.9** | **31.4 ms** | 9.54 s | 3.72 s | 95.74% | 95.72% | 81% | 111,216 |
| ref hcasplit4 | 128 | 8192 | 40,346 | 31.4 | 31.8 ms | 9.31 s | 3.75 s | 95.72% | 95.70% | 84% | 110,700 |

**The OFF arm reproduces the historical reference to 0.26 % on tok/s/chip and to
two decimals on cache hit**, which is the evidence that the recipe was matched
rather than approximated — see "Run the real launcher, not a reconstruction".

**Tooling, all in `/shared_nfs/kk` and all reusable as-is:**

- `agentx_flydsl_ab.sh` — the two-arm driver. It calls
  `/workspace/claude-skills/agentx/agentx_c128_hcasplit.sh`, i.e. the script
  that produced the historical arm, so env and aiperf flags are the reference's.
  Only `SGLANG_MLA_FLYDSL` differs between arms.
- `agentx_table.py "label=<result_dir>" ...` — prints the comparison table
  above. Column-to-JSON-field mapping is in its docstring.
- `step_stats.py` + `matched_bs.py` — the Δstep path.
- `stop_all.sh` — teardown that cannot kill its own caller.

**Run the real launcher, not a reconstruction.** Session 9 first ran a
recovered recipe whose *server flags* matched the historical c128 command
exactly (diffed: only `--schedule-conservativeness 0.3` and
`--triton-attention-num-kv-splits 16` extra). It was still wrong: **14
environment variables differed**, two of them on the attention path
(`SGLANG_MLA_HCA_KV_SPLITS=4`, `SGLANG_ROCM_FUSED_DECODE_MLA=1`) and one that
pins the MTP accept length (`SGLANG_SIMULATE_ACC_LEN=3.77`, plus
`_METHOD=match-expected` and `_TOKEN_MODE=real-draft-token`). A flag diff does
not prove recipe parity; the historical server.log's first ~33 lines are an env
dump, so diff that too. The fix was to stop reconstructing and call the arm's
own script.

**Splits, for anyone reading the sweep:** `SGLANG_MLA_HCA_KV_SPLITS=4` overrides
HCA (ratio 128) only. SWA (ratio 0) and CSA (ratio 4) keep
`_kv_splits_heuristic`, which returns 2 at c128 — so since the gate was lifted
the kernel serves both 4 and 2, which is why `12/1192/2` is in the shape table.
`kv_splits=1` never reaches the hook.

**Session 9 changes, none committed.** `gsm8k_flydsl_ab.sh` gained a
`CLIENT_CMD` hook (swap the load generator without touching the server recipe)
and a `STEP_BLOCKS` hook (pull `step_time_dict` before teardown). New files:
`agentx_flydsl_ab.sh`, `agentx_table.py`, `stop_all.sh`, `c128_client.sh`.
`step_stats.py` was fixed to read `GET /server_info` (`internal_states`) —
`/get_internal_state` is the scheduler-side request name and has no HTTP route.

**Three operational traps session 9 paid for:**

- **`pkill -f <pattern>` matches the shell that runs it** whenever the pattern
  appears literally in that command line. It killed this session's shell twice.
  Use `stop_all.sh`, which keeps patterns out of the caller's argv.
- **Leftover `port_base` 9123 kills the next arm silently.** The first c128
  launch lost its whole off arm this way because the previous teardown had not
  finished. `stop_all.sh` waits on the ports.
- **`step_time_dict` lives only in the server's memory** and the AgentX launcher
  tears the server down when the replay ends, so `agentx_flydsl_ab.sh` polls a
  snapshot every 60 s and keeps the last one.

**Still open, and now the only things:**

1. **Nothing from sessions 5-9 is committed.** That is the largest outstanding
   risk — the working tree is the only copy of the win.
2. ~~The kv_len tail is unmeasured.~~ **DONE — the tail wins too, so the whole
   production distribution (134 to 1901) is covered and no shape loses.** The
   probe puts kv_len at mean 1192 / p50 1148 / p99 1492 / max 1901; measured
   at the tail, Triton against FlyDSL: `12:1492:4` 160.4 → 141.9 (0.88x),
   `24:1492:4` 310.8 → 260.5 (0.84x), `12:1901:4` 180.2 → 160.8 (0.89x),
   `24:1901:4` 358.7 → 310.0 (0.86x), relL2 9.7e-04 to 1.04e-03 — lower than at
   the short shapes, as expected since P's bf16 error dilutes over more KV.

---

**Status (end of session 8): the stream gate is LIFTED and it cost nothing.
`SGLANG_MLA_FLYDSL_RATIO` is now a set (`"0,4,128"`, or `"all"`), the kernel
serves all three decode streams, and gsm8k 1319 reads **0.936** — identical to
the off arm, above the narrow-gate 0.933 — with `calls=2868 hits=2868 skips={}`,
i.e. not one fallback. Every CSA/SWA shape measured offline is a win too
(0.84-0.93x). So item 1 of session 7's list is DONE and coverage is no longer
the open question.**

**The only thing still open is gate 2 (matched-bs Δstep), and its missing
tooling is now written: `/shared_nfs/kk/step_stats.py`.** It pulls
`step_time_dict` off a live server's `GET /server_info` — the
`/get_internal_state` this first tried is the scheduler-side request name and
has no HTTP route; the per-rank dicts are under `internal_states` (populated when
`SGLANG_RECORD_STEP_TIME=1`), merges every DP rank, and prints exactly the
block format `matched_bs.py` parses. **It has not been run against a server
yet** — the units conversion (`gap_latency / decode_log_interval` is seconds, so
the script multiplies by 1e3) is the one thing to sanity-check on first use:
the p50 cells should land near the `Decode batch` lines' implied step time.

**Session 9, in one command each:**

```bash
# gate 2, both arms, with step recording on
cd /shared_nfs/kk
for arm in off on; do
  FLY=$([ $arm = on ] && echo 1 || echo 0)
  SGLANG_RECORD_STEP_TIME=1 SGLANG_MLA_FLYDSL_QLEN=1 SGLANG_MLA_FLYDSL_HW=64 \
    SGLANG_MLA_FLYDSL_KVPAD=16 SGLANG_MLA_FLYDSL_RATIO=0,4,128 \
    ARMS=$arm OUT=/shared_nfs/kk/flydsl_gate2 ./gsm8k_flydsl_ab.sh
  # NOTE: step_stats.py must run while that arm's server is still alive, so
  # either call it from inside the script before kill_server, or run the arms
  # by hand. This is the one wiring step left.
  python3 step_stats.py $arm --port 8889 >> /shared_nfs/kk/gate2_blocks.txt
done
python3 matched_bs.py /shared_nfs/kk/gate2_blocks.txt gate2=off,on
```

Read the matched-bs cells, not the wall clock — "Barrier wait is not cashable"
means a real 16 µs MLA win can still read ~0 end to end. **If gate 2 comes back
at ~0, that is the answer to "should we keep optimising this kernel."**

**Session 8 changes, none committed.** `flydsl_decode_hook.py`: `_RATIO` (a
single `int`) became `_RATIOS` (a set, via `_parse_ratios`, accepting
comma-separated values, `none`, or `all`), and the gate is now
`_RATIOS is not None and compress_ratio not in _RATIOS`. Its old comment
justified the gate on the prefix assumption, which per-token mode discarded;
the gate now exists only to keep out streams whose shapes are not a win.
New file `/shared_nfs/kk/step_stats.py`. Nothing else was touched, and the
production-shape number is unchanged (119.9-123.1 µs against Triton 138-140).

**Two operational notes session 8 paid for:**

- **`verify_real.py` cannot be run with `RATIO=all`.** Its Triton reference arm
  is the call with `compress_ratio=None`; admitting that ratio makes the harness
  score FlyDSL against itself. Sweep CSA/SWA by shape (`VR_CASES`) instead — the
  ratio argument there is only the gate selector, and per-token mode is
  layout-agnostic, so `kv_len` is the whole story.
- **Do not leave `HIP_VISIBLE_DEVICES=0` exported in the shell that launches the
  server.** The single-GPU repro commands set it; the 8-GPU launch then dies in
  init with `HIP error: invalid device ordinal` on ranks 1-7. Launch with
  `env -u HIP_VISIBLE_DEVICES`.

**Coverage measured offline, per-token mode, `verify_real.py`** (CSA is the
512/1024 columns, SWA the 134 one; run in one sweep, same session as the golden
block below):

| bs, kv_len, splits | Triton | FlyDSL | ratio |
|---|---:|---:|---:|
| 8, 134, 4 | 54.7 | 48.7 | 0.89x |
| 12, 134, 4 | 54.6 | 50.6 | 0.93x |
| 24, 134, 4 | 99.0 | 91.9 | 0.93x |
| 12, 512, 4 | 87.5 | 77.8 | 0.89x |
| 24, 512, 4 | 169.6 | 142.4 | 0.84x |
| 12, 1024, 4 | 123.7 | 109.8 | 0.89x |
| 24, 1024, 4 | 241.0 | 205.3 | 0.85x |
| 32, 1024, 4 | 307.6 | 271.9 | 0.88x |

relL2 1.15e-03 to 1.98e-03 throughout, i.e. the expected bf16-P magnitude.

---

**Status (end of session 7): the kernel now BEATS Triton at every shape measured
— 0.87-0.88x at the production p50 (122.7 vs 139.6 µs), 0.82-0.91x elsewhere —
and gsm8k 1319 reads 0.933 against the off arm's 0.936. Session 7 took the
production shape from 137 µs to 121 µs in two steps, both now default:
`COOP=8` (cooperative online-softmax row stats, 10.3 µs) and `XLANE=1 MERGEP=1`
(one cross-lane reduction feeding one merged exp2 pass, 8.4 µs). Both came out of
one ablation that priced the old one-thread-per-row scan at 21.3 µs.**

**The softmax+PV block is now largely spent: of its original 53.8 µs, 18.7 µs
has been collected and the rest is PV MFMA plus the P publish, which "Why full
register residency is still closed at D=512" shows cannot be removed at this
shape.** So the next moves are not kernel micro-optimisation — they are
**coverage** (item 5, lifting the stream gate, roughly doubles the call sites) and
**gate 2** (matched-bs Δstep, the only gate still open and the only thing that
says whether 16 µs of kernel time is cashable end to end).

---

**Status (end of session 6): the accuracy bug is FIXED and GATE 1 IS
PASSED. The cause was not segmentation — it was the q-folding prefix assumption
itself, which production metadata never satisfied. The kernel now runs per-token
(`SGLANG_MLA_FLYDSL_QLEN=1`), and gsm8k 1319 reads 0.933 against the off arm's
0.936 on the same recipe, i.e. inside one sigma (0.007). The previous on arm was
0.903, so the 3.3-point gap closed to 0.3.**

---

### Start here (session 8)

**Do not start another kernel micro-optimisation.** The kernel now beats Triton
at every shape measured and the block those wins came from is largely spent: of
the softmax+PV block's 53.8 µs, 18.7 µs is collected, and what is left is the PV
MFMA plus the P publish — and "…and why full register residency is still closed
at D=512" *proves* the P publish cannot be removed at this shape. **There is no
named double-digit kernel item left.** Three separate routes into that block
(two-phase M=112, the operand swap, the cross-lane butterfly) are now built,
correct and measured nulls, so the prior probability on a fourth is low.

**The remaining value is in coverage and in proving the win is cashable, in this
order.**

**(1) ~~Lift the stream gate.~~ DONE IN SESSION 8 AND PASSED — gsm8k 0.936 with
`skips={}`. See the session-8 block at the top; the rest of this item is kept
only for the reasoning that motivated it.** The
kernel serves one of three decode streams; `eligible` refuses the other two on
stream identity alone (`flydsl_decode_hook.py:152`, `compress_ratio != _RATIO`).
Session 5 counted `hits=3211` against `skips={'stream ratio 4': 2477}`, so
admitting CSA alone is close to doubling coverage. **Per-token mode makes no
assumption about index layout, so this is free correctness-wise** — the gate
predates it and was set during an accuracy hunt, when stream identity was the
one-variable change available.

  * `_RATIO` is a single `int`, so it admits exactly one stream. **Make it a set
    (comma-separated env) before anything else**, or you cannot run HCA and CSA
    together at all.
  * Verify shapes offline first: CSA clamps per token (`index_topk` 512/1024) and
    SWA is short-kv, where the bs=1/kv=134 column already reads 0.89-0.91x. Run
    `verify_real.py` on those shapes before touching a server.
  * Then `SGLANG_MLA_FLYDSL_RATIO=4 SGLANG_MLA_FLYDSL_CHECK=1` reports the
    violation rate directly — but **read "the audit and check tooling still has
    the capture blind spot"** below first: at
    `--cuda-graph-max-bs-decode 128` the check only ever sees warmup, which is
    exactly how session 4 cleared a kernel that was wrong. Force eager steps by
    setting `PARALLEL` above it, and read the hit counter before believing any
    relL2.
  * Pass criteria: gsm8k ≥ ~0.93 with the wider gate, and the per-stream ratios
    from `verify_real.py` at or below 1.0x. A stream that is slower than Triton
    should stay gated out — coverage is only a win where the ratio is.

**(2) Gate 2 — matched-bs Δstep. The only gate still open, and the only thing
that says whether 16 µs of kernel time is cashable.** Details and the tooling
that has to be rebuilt are in item 2 of "Next, in order" below. **Remember
"Barrier wait is not cashable": an MLA win can read as ~0 end to end**, so read
the matched-bs cells, not the wall clock. If gate 2 comes back at ~0, that is the
answer to "should we keep optimising this kernel", and it is worth more than any
further µs.

**Also outstanding, and not measurement:** nothing from sessions 5-7 is
committed. The changed files are listed under "Session 7 changes" and "Session 6
changes" below.

---

**The config that ships** (per-token, two-phase off, cooperative row stats plus
the merged exp2 pass). All three session-7 knobs are now **defaults in
`flydsl_decode_hook.py`**, so this is the same as setting nothing:

```bash
SGLANG_MLA_FLYDSL=1 SGLANG_MLA_FLYDSL_QLEN=1 \
  SGLANG_MLA_FLYDSL_HW=64 SGLANG_MLA_FLYDSL_KVPAD=16 \
  SGLANG_MLA_FLYDSL_COOP=8 SGLANG_MLA_FLYDSL_XLANE=1 SGLANG_MLA_FLYDSL_MERGEP=1
```

**Repro, ~10 s each, no server:**

```bash
cd /shared_nfs/kk
export SGLANG_MLA_FLYDSL=1 SGLANG_MLA_FLYDSL_QLEN=1 \
  SGLANG_MLA_FLYDSL_HW=64 SGLANG_MLA_FLYDSL_KVPAD=16 \
  PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0
python3 verify_real.py        # correctness + time on PRODUCTION-layout metadata
python3 repro_bad.py          # the real dumped failing call, now passing
SGLANG_MLA_FLYDSL_ABL=stage python3 verify_real.py   # ladder rung; also empty|gather|qk
VR_CASES="12:1192:4" python3 verify_real.py           # one shape only
python3 flydsl_relayout_probe.py --dump-map            # the operand swap, ~6 s
python3 flydsl_relayout_probe.py --abl noswap          # its broken control arm
```

**Golden output of the first command, on the session-7 defaults** — if a fresh
session does not see roughly this, something has rotted before you changed
anything:

```
   8      134   4 |     52.9 |     48.2 | 0.91x | 1.97e-03
  12     1192   4 |    139.6 |    122.7 | 0.88x | 1.58e-03
  12     1192   2 |    151.8 |    135.9 | 0.90x | 9.88e-04
  24     1192   4 |    273.7 |    225.2 | 0.82x | 1.57e-03
```

Run-to-run spread is ±2 µs on both columns, so read a change of 5 µs or more.

**The four-arm control chain, verified to build and to be correct in one run**
(bs=12, kv=1192, splits=4; all four at relL2 1.57e-03). Any future change should
be judged against the arm below it, not against Triton:

| `COOP XLANE MERGEP` | µs | ratio | what it is |
|---|---:|---:|---|
| `0 0 0` | 141.6 | 1.00x | the session-6 kernel |
| `8 0 0` | 127.3 | 0.92x | cooperative row stats, LDS combine |
| `8 1 0` | 127.6 | 0.92x | … DPP combine instead (the null) |
| `8 1 1` | **122.2** | **0.88x** | … merged exp2 pass (the default) |

**Keep all four arms building.** Twice in session 7 a later edit silently broke an
earlier arm, both times through the AST-rewriter name collision described under
"Session 7 changes" — and an optimisation whose control arm no longer compiles
cannot be re-measured.

**Pass criteria, in order:**

1. `verify_real.py` relL2 **1.5-2.0e-03** at every shape — that is the bf16-P
   magnitude, not an error. Anything at 1e-02 or above is a real bug; scoring
   against `ref_torch()` rather than Triton is explained in "The correctness
   gate is 'vs torch', not 'vs shipped'".
2. Beat **121 µs** at bs=12, kv=1192, splits=4 on the default config (Triton
   138-141 in the same run; the pre-session-7 bar was 137). Run-to-run spread on this shape
   is ±2 µs on both sides, so read a change of 5 µs or more, not 2.
3. gsm8k 1319 ≥ ~0.93 against the off arm's 0.936, via the command below.
4. Gate 2 (matched-bs Δstep) is still open and needs its tooling rebuilt — item 2
   under "Next, in order".

**Two rules this session paid for, apply them before trusting any arm:**

- **Every ablation knob must reach `build_kernel` as an argument.** FlyDSL's JIT
  cache key covers the kernel source and its closure scalars but **not module
  globals**, so a global flag silently reuses the first-compiled arm. Four
  control arms read identical off one cached binary before this was found.
- **Every ablation set needs one arm whose expected result is obviously broken**
  (e.g. `tp_abl="allmask"` must produce relL2 1.0). That arm is the only reason
  the cache collision was caught, and without it this session would have shipped
  a kernel whose mask did nothing.

**Nothing is committed.** Changed files are listed at the end of this section.

---

| arm | gsm8k 1319 |
|---|---:|
| hook off (same recipe) | 0.936 |
| hook on, q-folded (session 5) | 0.903 |
| hook on, per-token (session 6) | 0.933 |
| hook on, per-token + `COOP=8` (session 7) | 0.940 |
| **hook on, + `XLANE=1 MERGEP=1` (session 7)** | **0.933** |

Both session-7 arms ran `Invalid: 0.000` with `calls=100 hits=100 skips={}`, i.e.
the kernel served every eligible call and nothing fell back. 0.933, 0.936 and
0.940 are all inside one sigma (0.007) of each other, and relL2 is identical at
1.57e-03 across all three, so **`_COOP=8`, `_XLANE=1` and `_MERGE_P=1` are now the
defaults in `flydsl_decode_hook.py`** and the full 15.8 µs is on by default.

```bash
# the passing arm, reproduced in ~12 min (2 min startup + 1 min bench)
SGLANG_MLA_FLYDSL_QLEN=1 SGLANG_MLA_FLYDSL_HW=64 SGLANG_MLA_FLYDSL_KVPAD=16 \
  ARMS=on /shared_nfs/kk/gsm8k_flydsl_ab.sh
```

**Note on the hit counter in that run:** it stops at `calls=100` per rank, which
is not a fallback — CUDA-graph *replay* does not re-enter the Python hook, so
only capture and eager steps are counted. The graphs were captured with the
per-token config, so the kernel is what runs inside them.

**The root cause, and it kills the design's premise.** Every stream's
`kv_indices` slice is `[SWA ring window] + [committed compressed indices]`, built
by `paged_decode_indices.py`: `n = min(positions[t] + 1, win)` entries at ring
offsets `(pos - n + 1 + w) % ring_stride`, then the stream's own compressed
tail. So once the window is saturated (`pos + 1 >= win`, i.e. after the first
128 tokens of any sequence — essentially always) the window segment **slides by
one per draft token and wraps**, while the compressed tail is **identical**
across the window. Measured on the dumped case: all 7 tokens have length 134,
token `q` starts at `3464 + q`, there is a ring wrap inside the first 128
entries, and the last 6 entries are the same for every token. **What the seven
tokens share is a suffix, not a prefix** — the assumption was backwards, so one
tile load could never have served the window. The per-token error confirms it:
`q=6` (the token whose slice the kernel actually read) was correct at 2.1e-03
and `q=0..5` were wrong by 0.09-0.38, monotonically worse the further from
`q=6`.

**The `compress_ratio == 128` gate never protected anything**, because the
sliding window segment is shared by all three streams. Bug 1 in session 5 was
real (it bought 3.2 points) but its stated mechanism was wrong: SWA and CSA are
not special, they were just the shapes where the violation was largest.

**Two of session 5's own signals were false, both from silent fallbacks:**

- **`splits=1` was not "bit-identical", it was a skip.** `kv_splits=1` never
  reaches the hook at all (a different, non-split kernel serves it), so both
  arms ran Triton and printed `relL2=0.000e+00`. The "error does not scale with
  split count, therefore segmentation" reasoning was built on that zero. With
  `SGLANG_MLA_FLYDSL_LOG=1` the hit counter shows calls 3-8 only, i.e. splits
  2/4/8. **A comparison harness must prove the path under test actually ran**
  — this doc has now been fooled by a degenerate zero three times.
- **The prefix check's "0 violations out of 124,992" was the capture gate**,
  exactly the trap that made the session-4 audit clear the kernel. The check is
  gated on `not is_current_stream_capturing()`, and at
  `--cuda-graph-max-bs-decode 128` it only ever saw warmup. Run on the dumped
  case it fires immediately: `PREFIX VIOLATION token 0/7 of request 0`.

**The fix, and what it costs.** Per-token mode drops the folding of the verify
window into M and folds **heads** instead — `Q_LEN=1, h_per_wg=64` gives the
same M=64 the shipped kernel uses, the KV tile is still amortised across the 64
heads of one token, and no assumption about index slices is needed. The freed
LDS (~50 KB) pays for `kv_pad=16`, which the hw=8 sweep had already measured at
~3%. Measured with `verify_integration.py 12`:

| shape | Triton | FlyDSL per-token | ratio |
|---|---:|---:|---:|
| bs=1 spl 4 | 50.8 µs | 49.8 | 0.98x |
| bs=1 spl 16 | 51.5 | 48.2 | 0.94x |
| bs=4 | 60.8 | 58.6 | 0.96x |
| bs=8 | 100.3 | 93.8 | 0.94x |
| bs=12 spl 4 (production p50) | 141.3 | **134.1** | **0.95x** |
| bs=24 | 268.0 | 259.5 | 0.97x |
| bs=32 | 366.3 | 357.5 | 0.98x |

relL2 1.2-1.4e-03 throughout, i.e. the expected bf16-P magnitude. The dumped bad
case goes from **2.127e-01 to 1.982e-03**.

**The geometry was re-swept in per-token mode, because every ranking in this doc
below was measured in the q-folded one.** On `verify_real.py` at bs=12, kv=1192,
Triton 138-140 µs:

- **`h_per_wg` is a cliff, not a curve: 64 is 137 µs and 32 and 16 are
  200-212 µs.** With `Q_LEN=1` the KV tile amortises over `h_per_wg` rows only,
  so at 16 rows staging dominates and the grid inflates to 2688 CTAs. M=128
  does not fit LDS (Q alone is 128 KB), so **64 is both the best and the
  largest** — the same `BLOCK_H=64` the shipped kernel uses.
- **`kv_pad=16` beats 8 and 32** at every `h_per_wg` (136.9 vs 143.3 vs 145.2 at
  hw=64). The q-folded geometry could not afford 16 at all.
- **`KV_SPLITS=4` is the best absolute**, which is what production already pins
  for HCA: 2 → 159.8, 4 → 137.1, 8 → 160.9, 16 → 231.3. Higher splits improve
  the *ratio* (0.89x at 16) while making both sides slower, because the 88 MB of
  fp32 partials doubles with the split count. Read the absolute column.

**And the profile in this geometry closed the double-buffering item without
building it** (`rocprofv3 --pmc OccupancyPercent MfmaUtil LDSBankConflict
MemUnitStalled`, both kernels in one process):

| counter | FlyDSL per-token | shipped |
|---|---:|---:|
| MemUnitStalled | **0.14 %** | 0.26 % |
| LDSBankConflict | **9.98 %** | 10.7 % |
| MfmaUtil | 8.65 % | 13.51 % |
| OccupancyPercent | 19.4 % | 18.76 % |

- **Double-buffering the KV tile has no prize here.** Its entire purpose is to
  hide global-load latency under compute and `MemUnitStalled` is ~0. It had been
  the live Tier-2 item, blocked at hw=16 for 32 KB of LDS that per-token mode
  *does* have — and the 32 KB turns out not to be worth spending.
- **Bank conflicts are done.** 9.98 % is now *below* the control; in the q-folded
  geometry `kv_pad=8` still read 11.28 % against 9.41 %.
- **Occupancy is not a differentiator** (19.4 vs 18.8), which retires the last
  reason to revisit the LDS footprint — see the two measured regressions in
  Steps 6 and 7.
- **The one visible gap is MfmaUtil, 8.65 % against 13.51 %**: the same
  arithmetic at two thirds of the control's MFMA utilisation, and still slightly
  faster overall. The ladder below confirms where it lives.

**The ablation ladder now runs in the geometry that ships, and this is the first
one in this doc that does.** All the ladders below were measured at q-folded
geometries that are now void. It runs through the hook — `SGLANG_MLA_FLYDSL_ABL=
empty|gather|stage|qk` on `verify_real.py` — which took a one-line fix: the
mfma_pv epilogue now indexes `a_f[... % N_ACC]`, the same idiom the non-mfma_pv
epilogue already used, so the ablation arms' fixed 4-register tail survives the
real epilogue instead of raising `list index out of range`.

Per-token, hw=64, kv_pad=16, bs=12, kv=1192, splits=4, total **139.1 µs**. Note
these rungs include the shared tail (the reduce plus the 88 MB of partial
allocations), which this doc measured at ~30 µs — so the kernel's own share is
against ~109 µs:

| rung | µs | delta | share of the kernel |
|---|---:|---:|---:|
| launch + epilogue + shared tail | 48.6 | | ~18 µs is ours |
| + tile loop | 56.1 | +7.5 | 7 % |
| + Q/KV → LDS staging | 69.8 | **+13.7** | 13 % |
| + QK MFMA | 85.3 | +15.5 | 14 % |
| **+ fast softmax + PV** | 139.1 | **+53.8** | **49 %** |

- **Softmax + PV is 49 %, the same share the q-folded ladder reported (72.7 µs of
  147)**, so that finding survives the geometry change intact and it is where the
  MfmaUtil gap lives. It is the only remaining item with a double-digit prize.
- **Staging is 13.7 µs, which is double-buffering's ceiling** — a second,
  independent refutation alongside `MemUnitStalled` ≈ 0.
- **QK is down to 15.5 µs**, confirming the `kv_pad` work is finished.

So the standing q-folded numbers (0.90x stage-1, 0.78-0.92x at bs>=8) were
measured on a kernel that was reading the wrong KV for six of every seven
tokens. **Treat every ratio above "Standing" as void.** Per-token mode is the
real baseline and it is ~0.95x.

**The harness this project was missing now exists: `/shared_nfs/kk/real_meta.py`
+ `verify_real.py`.** `real_meta.build(R, kv_len, q_len)` writes metadata the way
`paged_decode_indices.py` does — sliding ring window plus a shared committed
tail — instead of the growing prefixes `mla_qfold_proto.py` invents, and
`real_meta.check()` returns all three invariants so a harness can prove it is
testing the layout it claims to. Validated against the dump: it reports the same
48-of-56 prefix violations. **Run `verify_real.py` before believing any ratio.**

**And it settles a claim this doc got wrong: q-folding was never a short-kv
bug.** On production-layout metadata at kv=1192 the q-folded kernel reads
**5.0e-02**, 40x the expected 1.3e-03, at every batch size tried. The audit only
ever reported short-kv failures because the eager (non-captured) steps it could
see were the early ones. Per-token mode on the same metadata is 1.6e-03.

| arm, `verify_real.py` | kv=134 | kv=1192 bs=12 | kv=1192 bs=24 |
|---|---:|---:|---:|
| q-folded relL2 | 1.99e-01 | 5.04e-02 | 5.25e-02 |
| per-token relL2 | 1.97e-03 | 1.58e-03 | 1.57e-03 |

Part of q-folding's measured speed (0.77-0.95x here) was bought by loading a
seventh of the KV it owed.

**Next, in order:**

1. ~~gsm8k with per-token mode.~~ **DONE, PASSED** — see the table above.
2. **matched-bs Δstep, and its tooling has to be rebuilt first.**
   `matched_bs.py` consumes blocks of `batch= n= step_ms p50=` lines produced by
   a `decode_stats.py` that **no longer exists on this node** (same loss as
   `gsm8k_segplan.sh`), and the gate server logs carry no per-step timing —
   only ~98 coarse `Decode batch` lines. The instrumentation itself is still
   there: `SGLANG_RECORD_STEP_TIME` in
   `scheduler_components/metrics_reporter.py`. So gate 2 is: two server runs
   with that flag on, plus a ~40-line aggregator emitting the block format
   `matched_bs.py` parses. Remember "Barrier wait is not cashable" — an MLA win
   can read as ~0 end to end, so read the matched-bs cells, not the wall.
3. ~~The two-phase kernel — the route back to M=112.~~ **BUILT, CORRECT, AND
   CLOSED AS A MEASURED NULL (session 6).** It is
   `build_kernel(..., two_phase=True, win=128)`, reached from the hook with
   `SGLANG_MLA_FLYDSL_WIN=128`, correct on production-layout metadata at
   **2.06e-03** — and **slower than per-token at every geometry tried**:

   | arm at bs=12, kv=1192 (`verify_real.py`) | µs | ratio |
   |---|---:|---:|
   | Triton | 138 | — |
   | **per-token, hw=64, kv_pad=16** | **140** | **0.99x** |
   | two-phase, hw=8, kv_pad=16 | 153 | 1.12x |
   | two-phase, hw=16, kv_pad=8 | 174 | 1.24x |

   **The mechanism, and it is not the structure.** With the interval mask
   replaced by the old single upper bound (`tp_abl="oldmask"`, wrong results,
   readable time) the two-phase kernel runs at **135 µs, 0.89x** — so the phase
   split, the union axis and the index fetch are *free*. The whole cost is the
   per-row **lower** bound, and it is register pressure rather than arithmetic:
   it costs 38 µs at hw=16 and only 7 µs at hw=8, where the accumulator leaves
   headroom. hw=16 also cannot afford `kv_pad=16`, which is where per-token's
   win comes from.

   **Why the premise was weaker than it looked.** The prize was amortising KV
   loads across the verify window, and this doc's own "Closed" section had
   already priced that at **3%** (117 MB of working set fits the 256 MB
   Infinity Cache). Per-token also gets 7x more CTAs (672 against 384), which
   is worth more here than the loads it duplicates. A design whose prize is 3%
   cannot pay for a mask that costs 25%.

   The code is kept behind the flag because it is correct and cheap to re-open
   if KV traffic ever becomes the binding constraint — the two invariants it
   rests on are checked by `real_meta.check()` and both hold, 0 violations of 56
   slices, on the real dump and on generated metadata:

   - **(a) tail.** Entries at and after `WIN` are shared by the whole verify
     window (a prefix of the last token's tail when a commit boundary falls
     inside the window). At kv=1192 that is ~33 of 38 tiles, foldable at M=112
     on an invariant *stronger* than the one that just failed — equality, not
     prefix.
   - **(b) union.** Window segment `q` is window segment `0` shifted by `q`, so
     all seven windows live in one ring-contiguous run of `WIN + q_len - 1` =
     134 entries — **5 tiles, not 7x4**. And no shifting is needed in the
     kernel: scores and PV are already masked per row, so a row just needs an
     interval `q <= u < q + win_q` instead of today's single upper bound
     `u < row_len`.

   As built: `row_len` became a per-row `(lo, hi)` (`row_bounds`, one select
   chain — everything per-row is recovered from `row_len` plus `win_0`, since
   `u0_q + win_q = q + win_0`); the index fetch picks tail vs union
   branchlessly; and the reduce mirrors the tile total through a new `FOLD_WIN`
   constexpr, since its `act_num_segments` has to agree with a tile axis that is
   no longer one span of `kv_len`.
4. **If more speed is wanted, the only direction the profile leaves is
   MfmaUtil / the softmax+PV block.** Everything cheaper has now been measured
   out: padding is done, the occupancy routes are three-times refuted, and
   double-buffering has no prize.
   ~~The prerequisite is an ablation ladder that runs in per-token geometry.~~
   **DONE — `SGLANG_MLA_FLYDSL_ABL` on `verify_real.py`, and it puts the block at
   53.8 µs / 49 %.**
   ~~Blocked on the QK-C-fragment to PV-A-fragment relayout.~~ **SOLVED AND
   MEASURED (session 7): there is no relayout — swap the operands of both GEMMs
   and the correction is 8 `permlane32_swap`. See "The operand swap" above;
   `flydsl_relayout_probe.py` passes at 3.1e-08 and its `noswap` control fails at
   1.037.** But that section also establishes that the swap **cannot collect the
   prize any more cheaply than the thread mapping already did**: keeping P in
   registers is closed at `D = 512` (the d-split across waves forces P through
   LDS), and register-resident row stats need `QK_DSPLIT = 1`, which costs 6.9 µs.
   ~~The block is 53.8 µs and undifferentiated.~~ **ATTRIBUTED AND LARGELY SPENT
   (session 7): 21.3 µs of it was the row-stats scan and 3.0 µs is `_combine`;
   `COOP=8` took 10.3 µs of the 21.3 and `XLANE=1 MERGEP=1` took another 8.4, so
   18.7 µs of the 53.8 is collected and all three are now defaults.** The
   cross-lane butterfly on its own was a **measured null** (~1.2 µs) — its value
   is that it enables the merge. What is left in the block is the PV MFMA and the
   P publish, and the P publish is the one "Why full register residency is still
   closed at `D = 512`" proves cannot go. **There is no named double-digit kernel
   item left; items 2 and 5 (gate 2 and coverage) are where the remaining value
   is.**
5. **Lifting the stream gate is free correctness-wise and triples the call
   sites.** Per-token mode makes no assumption about index layout, so SWA
   (ratio 0) and CSA (ratio 4) are eligible too; the hook still refuses them
   through `SGLANG_MLA_FLYDSL_RATIO`. Verify each stream's shapes on
   `verify_real.py` first — CSA clamps per token and SWA is short-kv, where the
   bs=1/kv=134 column reads 0.89-0.91x.
6. The audit and check tooling still has the capture blind spot. If anything
   has to be verified inside the server again, force eager steps by setting
   `PARALLEL` above `--cuda-graph-max-bs-decode`, and **read the hit counter
   before believing any relL2**.

**Diagnostics written this session, all ~7 s, all in `/shared_nfs/kk`:**
`repro_bad.py` (the original, now passing), `diag_seg.py` (per-token relL2 —
this is what showed only `q=6` was right), `diag_prefix.py` (CPU-only prefix
audit of the dump: 48 violations of 56 slices), `diag_layout.py` (prints the
window/tail structure of a slice).

`diag_rows.py <kv_len>` prints per-draft-token relL2 on generated metadata and is
the probe that separates a working interval mask from a dead one: with the lower
bound dropped it reads `q0=1.9e-03 ... q6=2.1e-01`, monotone in q, while the
aggregate relL2 barely moves.

## Appendix: session 7's measurement record

How the softmax+PV block was attributed, what was built, and the nulls. You need
this if you are going back into the kernel or reusing the mechanisms; **you do not
need it to do (1) or (2) in "Start here".**

**Session 7 split that block into its parts, and the answer was not the part this
doc had been aiming at.** Three arms, all on `verify_real.py` at bs=12, kv=1192,
splits=4, all wrong-by-design so the time is readable (the relL2 column is how
you know the arm took effect and did not read a stale JIT binary):

| arm | µs | vs control | relL2 | what it removes |
|---|---:|---:|---:|---|
| control | 140.1 | — | 1.57e-03 | |
| `SGLANG_MLA_FLYDSL_QK_ABL=nostats` | **118.8** | **−21.3** | 3.17e+01 | the 64-thread row-stats scan |
| `SGLANG_MLA_FLYDSL_QK_ABL=nocombine` | 137.0 | −3.0 | 8.98e-01 | the whole `_combine` cross-wave reduce |
| `SGLANG_MLA_FLYDSL_QKDSPLIT=1` | 145.3 | **+6.9** | 1.57e-03 | `_combine`, by idling 4 of 8 waves |

**The prize is the row-stats scan: 21.3 µs, ~19 % of the 109 µs kernel and ~40 %
of the softmax+PV block.** `_fast_softmax_pv` has `M_ROWS = 64` of 512 threads
walk `BLOCK_K = 32` scores out of `lds_s` twice (once for the max, once for `l`)
while the other 448 threads wait at the barrier. **The score round trip is
*not* the prize — `_combine` is 3.0 µs, and buying it by forcing `QK_DSPLIT = 1`
costs 6.9 µs, so that trade is net negative.**

**That was built, and it works: `coop_stats` collects 10.3 of the 21.3 µs and is
correct.** `SGLANG_MLA_FLYDSL_COOP=8` gives every thread `STAT_CH = BLOCK_K /
STAT_SPLIT = 4` scores of one row, reduced against its **own local max**, which
is what makes the combine associative (`m = max(m_c)`,
`l = Σ l_c * exp2(m_c - m)`). All 512 threads work, the dependent chain is
`4 + 8` deep instead of `2 x 32`, and it costs one barrier plus
`2 x BLOCK_TH` f32 of LDS scratch. No atom change, no accumulator pressure, no
`BLOCK_K` change.

| `SGLANG_MLA_FLYDSL_COOP` | µs at bs=12, kv=1192, spl 4 | relL2 |
|---|---:|---:|
| 0 (the old scan) | 137.1 | 1.57e-03 |
| 2 | 139.1 | 1.57e-03 |
| 4 | 134.0 | 1.57e-03 |
| **8** | **126.8** | 1.57e-03 |
| 8 + `SGLANG_MLA_FLYDSL_SPAD=1` | 132.3 | 1.57e-03 |
| 8 + `SPAD=4` | 131.0 | 1.57e-03 |

**relL2 is unchanged at every setting**, so unlike the arms above these are
correct configurations, not ablations. `COOP=8` is the whole prize available this
way: `STAT_SPLIT` is capped at `BLOCK_TH // M_ROWS = 8`.

Across shapes (`verify_real.py`, full sweep, `COOP=8`):

| shape | Triton | FlyDSL COOP=8 | ratio | session 6 ratio |
|---|---:|---:|---:|---:|
| bs=8 kv=134 spl 4 | 54.2 | 48.8 | **0.90x** | 0.89x |
| bs=12 kv=1192 spl 4 (production p50) | 139.1 | **129.9** | **0.93x** | 1.00x |
| bs=12 spl 2 | 151.3 | 145.0 | 0.96x | 1.06x |
| bs=24 kv=1192 spl 4 | 267.0 | 242.5 | **0.91x** | 0.95x |

**`s_pad` is now refuted a second time, and in the geometry that was supposed to
rescue it.** The prediction was that the cooperative `(row, chunk)` mapping makes
`s_pad=1` conflict-free — with `S_STRIDE = 33` the bank is
`(row + chunk*4 + kk) % 32`, and for a half wave `row%4 + 4*chunk` covers all 32
banks exactly once. It is still a 5 µs regression, so the cost the original note
identified (the extra index arithmetic, and a non-compact tile dropping the
P-pass copy off its wide vectorised path) dominates the conflict it removes.
**Do not reopen `s_pad` a third time without a profile showing the conflict, not
an argument that it exists.**

**The combine stage was predicted to be the next 11 µs. IT IS NOT — the
cross-lane butterfly is a measured null.** `SGLANG_MLA_FLYDSL_XLANE=1`
(`coop_xlane`) replaces the LDS scratch, its barrier and the 64-thread 8-deep
loop with a DPP xor ladder: reduce the **max** across the group first
(`quad_perm 0xB1` = lane^1, `quad_perm 0x4E` = lane^2, `row_half_mirror 0x141` =
lane^7), so the sum then needs no per-thread rescale and the whole reduction is
`STAT_CH` exp2, 6 DPP movs, no barrier and no scratch. Three repeats each:

| arm | µs | relL2 |
|---|---|---:|
| `XLANE=0` (LDS scratch) | 130.0 / 130.3 / 128.8 | 1.57e-03 |
| `XLANE=1` (DPP butterfly) | 128.2 / 128.6 / 128.6 | 1.57e-03 |

**~1.2 µs, against a spread of ~1.5 µs.** Left at default 0 because that is the
arm gsm8k was run on; the code stays behind the flag because it is correct and
strictly less machinery (one barrier and 4 KB of LDS lighter).

**The mechanism, and it corrects this doc's reading of the 21.3 µs.** `nostats`
collapsed the scan from 32 scores to 1, so it removed the *work* as well as the
serialisation. `coop_stats` removes only the serialisation — 512 threads still do
the same 2048 LDS reads and 2048 exp2 per tile that 64 threads used to. So the
split is **~10.3 µs of serialisation (collected) and ~11 µs of irreducible
volume**, and no amount of reduction plumbing touches the second number. **A rung
measured by deleting its work prices work + serialisation together; only an arm
that deletes one of them tells you which you are buying.**

### Merging the duplicated exp2 pass: −8.4 µs, and the butterfly is its enabler

Every element's `exp2` was being computed **twice** — once in the stats pass for
`l`, once again in the P pass as `exp2(s - m_new)`. `SGLANG_MLA_FLYDSL_MERGEP=1`
(`merge_p`, implies `coop_xlane`) folds them into one. **Three repeats each, at
bs=12, kv=1192, splits=4:**

| arm | µs | mean | ratio |
|---|---|---:|---:|
| `COOP=8` (session 7's first default) | 128.3 / 130.2 / 130.7 | 129.7 | 0.93x |
| `+ XLANE=1` | 128.3 / 127.9 / 127.5 | 127.9 | 0.91x |
| `+ MERGEP=1` | 123.3 / 120.9 / 119.8 | **121.3** | **0.87x** |

**The butterfly is what makes it possible, which is why its own null was not a
dead end.** After the DPP reduction *every* lane holds `m_blk`, and `m_old` is in
`lds_m`, so every lane derives `m_new = max(m_old, m_blk)` itself with **no
broadcast**. Then `P = exp2(sv - m_blk) * exp2(m_blk - m_new)` reuses the
`STAT_CH` terms the `l` sum already built, for one scalar `exp2`. What goes away:
the P pass's `M_ROWS * BLOCK_K` exp2, its `lds_s` re-reads, and one of the three
barriers (`alias_p`'s second barrier is only needed when the P pass re-reads
`lds_s`, which it no longer does). The P pass's flat pair index is replaced by the
stats pass's `(row, chunk)` mapping, which is what puts the values in the right
registers to begin with.

**One ordering subtlety, and it is load bearing:** all `STAT_SPLIT` lanes must
read `lds_m` *before* the `c_ch == 0` lane overwrites it. That holds only because
the read is a separate, earlier instruction and the group is one wave in lockstep
— if the group ever spanned waves this would be a race.

**The pre-registered rung prediction held, so the mechanism is believed** (rule 2).
The ladder at the same shape, `XLANE=0/MERGEP=0` against `XLANE=1/MERGEP=1`:

| rung | before | after |
|---|---:|---:|
| + Q/KV → LDS staging | 70.5 | 70.7 |
| + QK MFMA | 87.9 | 86.8 |
| full kernel | 129.7 | 121.3 |

Neither the staging rung nor the QK rung moved; the whole 8.4 µs came out of the
softmax+PV block, which is what was predicted and is why this is not luck.

Across shapes (`verify_real.py`, `COOP=8 XLANE=1 MERGEP=1`):

| shape | Triton | FlyDSL | ratio | session 6 |
|---|---:|---:|---:|---:|
| bs=8 kv=134 spl 4 | 55.2 | 49.3 | 0.89x | 0.89x |
| bs=12 kv=1192 spl 4 (production p50) | 141.5 | **123.0** | **0.87x** | 1.00x |
| bs=12 spl 2 | 151.8 | 137.2 | **0.90x** | 1.06x |
| bs=24 kv=1192 spl 4 | 269.7 | **225.7** | **0.84x** | 0.95x |

**The operand swap is banked, not next.** Session 7 also solved the relayout that
this doc called the blocker — it is 8 `permlane32_swap`, proved by probe — and
then found that it cannot collect the 21.3 µs any more cheaply, because
register-resident row stats need `QK_DSPLIT = 1` and that is the 6.9 µs above.
Read "The operand swap" below before spending anything on it.

**Session 7 changes, none committed.** `flydsl_mla_decode.py`: new build args
`coop_stats`, `coop_xlane` and `merge_p` (the wins), `qk_dsplit` and `qk_abl`
(the measurement arms), and
`STAT_SPLIT`/`STAT_CH`/`STAT_ELEMS` plus an `lds.stat` scratch array. The kernel
name now carries `_qd{QK_DSPLIT}{qk_abl}_cs{STAT_SPLIT}` so none of these can
collide in the JIT cache — the trap that cost session 6 four arms.
`flydsl_decode_hook.py`: `SGLANG_MLA_FLYDSL_COOP`, `_QKDSPLIT`, `_QK_ABL`,
`_SPAD`, `_XLANE`, `_MERGEP`, all reaching `build_kernel` as arguments and all in
`_kernel`'s `lru_cache` key; `_COOP=8`, `_XLANE=1`, `_MERGE_P=1` are the defaults. New file `/shared_nfs/kk/flydsl_relayout_probe.py`.

**A FlyDSL authoring trap this cost three runs, and it bit twice.** The stats
block now has three arms selected by `if const_expr(...)`, but they are
*textually* in the same function and **the AST rewriter analyses assignments
syntactically, before `const_expr` is evaluated**. So a name assigned in *any*
arm counts as live before a dynamic `if` that assigns it in another, and the
untaken arm still poisons the taken one: sharing `m_blk` raised
`Variable(s) ['m_blk'] initialized as None before a dynamic if/else`. It bit once
when `coop_xlane` landed and again when `merge_p` hoisted `m_new`/`alpha` out of
`if c_ch == 0`, that second time breaking the `COOP=0` control arm which nobody
would have run.

**Rules that follow:** give every `const_expr` arm its own local names; after
adding an arm, **rebuild every other arm** (the four-arm chain above is there for
this); and when renaming, use word boundaries — a blind `alpha` → `alp1` also
rewrites `lds_alpha`.

**Session 6 changes, none committed.** `flydsl_decode_hook.py`: `_Q_LEN == 1`
means per-token mode and bypasses the caller-width check; `kv_pad`, the
two-phase `win` and the two-phase ablation are env-tunable
(`SGLANG_MLA_FLYDSL_KVPAD`, `_WIN`, `_TP_ABL`) and all reach `build_kernel` as
arguments. `flydsl_mla_decode.py`: `two_phase`, `win`, `tp_abl` build flags,
`dyn_pick`, `row_bounds`, `row_ok`. `paged_decode.py`: the reduce takes
`FOLD_WIN` and derives the two-phase tile total; `run_split` now returns
`(fold_q_len, fold_win)`.

**Recommended production config:** `SGLANG_MLA_FLYDSL=1
SGLANG_MLA_FLYDSL_QLEN=1 SGLANG_MLA_FLYDSL_HW=64 SGLANG_MLA_FLYDSL_KVPAD=16`,
i.e. per-token, two-phase off.

---

### The operand swap (session 7): the relayout is 8 instructions

**`/shared_nfs/kk/flydsl_relayout_probe.py`, one wave, ~6 s, no server.** It
builds the whole QK → row-reduce → PV chain in registers and scores it against
torch:

| arm | qk relL2 | rowmax relL2 | pv relL2 |
|---|---:|---:|---:|
| default (swap) | 1.95e-08 | 1.48e-08 | **3.08e-08** |
| `--abl noswap` | 1.95e-08 | 1.48e-08 | **1.037** |

The `noswap` arm is the mandated obviously-broken control: it drops the
`permlane32_swap` and feeds raw QK registers as the PV operand, and it fails at
exactly 1.0 while QK and the row max stay clean. So the swap is load bearing and
the chain is correct, not accidentally aligned.

**There is no relayout. Swap the operands of both GEMMs instead**, on a
`MFMA(32, 32, 16, BFloat16)` atom:

```
QK: A=KV[kv, dk], B=Q[q, dk]  ->  C1[kv, q]     (today: C[q, kv])
PV: A=V[d, kv],   B=P[q, kv]  ->  C2[d, q]      (today: C[q, d])
```

Both operands stay `(rows, K)` row-major, so this is *only* a change of which
view goes to `make_fragment_A` and which to `make_fragment_B` — the existing
`sAdonor`/`aRow`/`aCol` and `sBdonor`/`bRow`/`bCol` machinery already covers it,
and `kv_single`'s element-wise fragment build works unchanged (PV's A wants
`V[d, kv]`, read out of the same `(kv, d)` tile).

**Why this removes the transpose.** `q` becomes the atom's N axis in *both*
fragments, so a lane owns one `q` row from QK all the way through PV. Maps
dumped by the probe, not derived: C is `col = lane % 32`,
`row = 4*(lane//32) + i%4 + 8*(i//4)`; B is `n = lane % 32`,
`k = 8*(lane//32) + j`. Three consequences:

- **The row reduction is 16 in-lane register steps plus one
  `permlane32_swap`** — no `lds_s`, no barrier, no softmax replicated across 512
  threads. This is aiter's `_score_pair_max` (`fmha_gfx950/pipeline.py`), and
  `rocdl.permlane32_swap(struct<(i32,i32)>, a, b, False, False)` returns
  `([a.lo, b.lo], [a.hi, b.hi])`, i.e. passing the same register twice gives a
  two-way cross-half reduction.
- **C1 → PV-B is 4 `permlane32_swap` per 16-wide k-atom, 8 for a 32-kv tile.**
  C1 hands lane half `g` the kv set `{4g+i, 8+4g+i}`; B wants `{8g+j}`. Swapping
  register `i` with register `i+4` across the halves converts one into the other
  exactly. aiter needs zero swaps only because at 32x32x64 fp8 its K and its
  q-block both happen to be 32.
- **`alpha` becomes a per-lane scalar** (it depends only on `q = lane % 32`), so
  the accumulator rescale is one multiply over all 64 registers instead of the
  per-register `lds_alpha` lookup `_pv_atoms` does now.

Register arithmetic is neutral: `C2` is 2 d-tiles/wave x 2 q-tiles x 16 = **64
registers, the same `N_ACC` the kernel has today**, and 32x32x16 halves the QK
instruction count against 16x16x32 at equal MACs/cycle.

### …and why full register residency is still closed at D=512

**PV's output is 512 wide.** Holding it in registers for a 64-q tile costs
16 d-tiles x 2 q-tiles x 16 = **512 registers per lane**, so `d` *must* be split
across the 8 waves (2 d-tiles each, 64 registers). But **every wave's PV needs
the whole P tile**, and the only cross-wave channel is LDS. So P's LDS publish
is not a layout artifact — it is what the d-split costs, and the swap cannot
remove it.

Both escapes are priced and both fail:

- **Split kv across waves instead of d** → every wave carries the full-d
  accumulator, back to 512 registers.
- **Let every wave recompute S** → no split, but the QK arithmetic is multiplied
  by `NWAVE = 8`, turning the 15.5 µs QK rung into ~124 µs, more than the whole
  kernel.
- **`NWAVE = 1` at `h_per_wg = 32`** is genuinely register resident (256
  registers, all AGPR) but runs at 1 wave/SIMD, and `h_per_wg = 32` is already
  measured at 200-212 µs.

**aiter's fmha escapes this only because `head_dim = 128`**: 4 d-tiles x 16 = 64
registers for a 32-q block in a single wave, so no d-split and no P broadcast.
The design does not transfer to MLA's 512-wide V. This is a property of the
shape, not of FlyDSL, and it is why "keep P in registers" was never going to
collect all 53.8 µs.

### Why the swap does not collect the 21.3 µs either

The swap's whole value here would be register-resident *row stats*: a lane owns
one q row, so `m` and `l` are 16 in-lane steps plus one `permlane32_swap` in
every wave at once, instead of 64 threads scanning LDS while 448 wait. **But
registers do not cross waves, so the scores in a lane must be final, not
partials — and that means `QK_DSPLIT = 1`, which the table above prices at
+6.9 µs.** With the swap, `QK_ITEMS = (BLOCK_K/32) x (h_per_wg/32)`, so at
today's `BLOCK_K = 32` it is 2, leaving **6** of 8 waves idle during QK — worse
than the 4-of-8 that already cost 6.9 µs. Reaching `QK_ITEMS = 4` needs
`BLOCK_K = 64` (LDS 64 x 528 x 2 B = 68 KB, up from 34 KB — affordable) and
`QK_ITEMS = 8` needs `BLOCK_K = 128` (135 KB + 66 KB of Q ≈ 200 KB — over the
~160 KB budget). So the best the swap can do is ~−21.3 + 6.9 ≈ −14 µs for an
atom change, a `BLOCK_K` change, a transposed PV A operand and a rewritten
epilogue, against ~−21 µs for a thread-mapping change that touches one function.
**Build the cheap one first; if it lands, re-price this against what is left.**

For the record, the geometry arithmetic if it ever is worth building. Today
(shipped config, `qk_reuse=True`) `QK_ITEMS = M_TILES = 4` and
`QK_DSPLIT = min(D//MMA_K, NWAVE//QK_ITEMS) = 2`, so every score is the sum of
two wave partials, combined through `lds_s4` by `_combine`; then 64 of 512
threads scan rows out of `lds_s` while 448 idle.

S can only stay in registers when **`QK_DSPLIT == 1`**, because two partials of
the same score sit in two different waves and registers cannot cross waves.
That needs `QK_ITEMS >= NWAVE`, and with the swap `QK_ITEMS = (BLOCK_K/32) x
(h_per_wg/32)`. Two ways to reach 8, one of which is dead on LDS:

| route | QK_ITEMS | KV tile LDS (`KV_STRIDE = D + 16 = 528`) | verdict |
|---|---:|---|---|
| `BLOCK_K=128`, `NWAVE=8` | 8 | 128 x 528 x 2 B = **135 KB**, + 66 KB of Q = 200 KB | **over the ~160 KB budget** |
| `BLOCK_K=64`, `BLOCK_TH=256` (`NWAVE=4`) | 4 = NWAVE | 64 x 528 x 2 B = 68 KB, + 66 KB Q + 8 KB P ≈ **141 KB** | **fits, tight** |

A fourth route, if the d-split itself is ever the thing to attack:
**`BLOCK_TH = 256` / `NWAVE = 4` / `BLOCK_K = 64`**. Each
wave then owns one complete `(kv-slice, q-tile)` of S in registers, does its own
softmax (16 `exp2` per lane, all 256 threads working), converts to bf16 and
publishes only P; only `m`/`l` need a cross-wave reduce (64 q x 4 slices of f32,
tiny). Its cost is that PV's d-split is now 4-way: **4 d-tiles x 2 q-tiles x 16 =
128 accumulator registers per lane**, double today's 64.

**Pre-registered prediction (rule 2), as rungs and not as a total:** the f32 S
store and `_combine` disappear (32 KB of `lds_s` traffic per tile), one barrier
disappears, and the `exp2` work spreads from 64 threads to 256. **The staging
rung (13.7 µs) and the PV MFMA must not move.** If the total moves and those two
rungs move with it, it is luck and it will mislead the next three decisions.

**Two risks to price first.** `BLOCK_TH = 256` changes the staging geometry
(`BLOCK_TH = D` today, "one thread per output column", so `ROWS_PER_ROUND`,
`TH_PER_ROW` and `DOT_SPLIT` all move with it) — that is a second variable, so it
needs its own arm. And 128 accumulator registers is exactly the pressure rule 3
warns the ablation ladder hides behind its fixed `N_ACC = 4` tail: **judge this
on a full arm, never on a rung.**

---

**Gate 1 result (2026-09-20): the kernel FAILED it.
gsm8k 1319, same recipe, same day: hook off **0.936**, hook on **0.903**.
Two real integration bugs were found and fixed getting from 0.864 to 0.903,
and a ~3.3-point gap survives both and is unexplained. The kernel is correct
at every shape the offline sweep builds and the offline sweep still reports
0.78-0.92x at bs>=8 after both fixes, so what is left is a property of the
server's metadata, not of the kernel's arithmetic.**

| arm | gsm8k 1319 |
|---|---:|
| hook off (matches the historical 0.937) | **0.936** |
| hook on, as inherited from session 4 | 0.864 |
| + stream gate (HCA only) | 0.897 |
| + real verify-window width | **0.903** |

**Next, in order:**

1. ~~Make the prefix check capture-safe and run it.~~ **DONE, and it cleared
   the prefix assumption: 0 violations out of 124,992 checked token slices on
   real server metadata.** The check is gated on
   `not torch.cuda.is_current_stream_capturing()` (the syncs abort capture, and
   `--disable-cuda-graph` is a dead end on this recipe — it dies in an
   unrelated `gemm_a8w8_blockscale_bpreshuffle ... M=0`), and the run forced
   eager decode steps by setting `PARALLEL=200` above
   `--cuda-graph-max-bs-decode 128`. The same run also showed the stream gate
   doing real work — `hits=1860, skips={'stream ratio 4': 900, 'stream ratio
   0': 40}` — and **no `window` skips at all**, so bug 2 was rare on this
   workload and the 0.897 → 0.903 move was mostly noise.
   **So the kernel's core assumption is not what is wrong.**
2. ~~Audit the kernel's output against Triton's inside the server.~~ **DONE,
   and it cleared the kernel on every step it can see.**
   `SGLANG_MLA_FLYDSL_AUDIT=1` (`flydsl_decode_hook.audit`) re-runs the Triton
   stage-1 on the same real inputs — the recursion passes `compress_ratio=None`
   to force the fallback, which is why the skip counter shows `stream ratio
   None` — and compares the **merged** output. Over ~1,240 audited calls per
   rank the worst relL2 is **3.2e-04**, with most calls bit-identical, i.e.
   *tighter* than the 1.2e-03 the offline sweep reports. **On real server
   metadata, outside capture, this kernel's output is right.**
3. ~~Test CUDA-graph replay.~~ **DONE, and replay is NOT the cause.** With
   `--cuda-graph-max-bs-decode 1` (almost every decode step eager — the clean
   substitute for the broken `--disable-cuda-graph`), the gap survives: off
   **0.943**, on **0.920** at NQ=300.
4. ~~Re-audit with that eager config, so the audit sees real large batches.~~
   **DONE, and this is what found the bug.** The earlier audit had cleared the
   kernel only because at `--cuda-graph-max-bs-decode 128` every real decode
   step was captured and therefore unauditable; all it had actually seen was
   warmup and a few tiny batches. Re-run eager, the worst relL2 went to
   **6.1e-01**. See the new CONTINUE HERE at the top.
   **The audit's own trap, worth keeping:** the non-captured steps include the
   pre-capture warmup, whose inputs are dummy, so both paths produce all-zero
   output and the ratio reads as exactly `0.000e+00`. Those are counted
   separately as "degenerate" now — **a comparison harness needs a non-zero
   reference norm before its zero means anything**, and this session was
   briefly fooled by it twice (once here, once by a missing env var in
   `repro_bad.py`).
5. Fix the segmentation, then re-run gate 1, then gate 2.

**Bug 1 — no stream gate. Worth 3.2 points, and the most reusable finding.**
Three decode streams share one call site in
`deepseek_v4_backend_hip_radix.py` and the hook had no stream gate. SWA's
128-entry window slides token by token and CSA clamps per token, so in neither
is a draft token's index slice a prefix of the window's last token's — the
exact assumption q-folding rests on. `compress_ratio` is now threaded
backend → `runtime.decode` → `sparse_attn_v4_paged_decode` →
`_sparse_attn_v4_paged_decode_triton` → `eligible`, and the hook takes only
ratio 128 (`SGLANG_MLA_FLYDSL_RATIO`). The doc had predicted this fix was
needed ("a per-stream dispatch that keeps SWA on Triton") and it had not been
written; the offline sweep could never catch it because it builds the ragged
layout for every case, including the ones it labels "SWA-like" and "CSA-like".
**An offline harness that constructs its own metadata cannot falsify an
assumption about metadata.**

**CSA is gated out conservatively, not permanently — re-open it.** The gate is
on stream identity because that was the one-variable change available during an
accuracy hunt, but the real condition is *is this call prefix-shaped*, and for
CSA that is a **length** condition, not a stream property. CSA's per-token
index list is `clamp(compressed_history, max=c4_sparse_topk)` with
`index_topk` 512 or 1024: **below the cap the whole history is selected and the
prefix holds; only once the cap binds does each token pick its own top-k set.**
HCA is ragged with no cap, which is why it is prefix-shaped by construction,
and SWA's 128-entry sliding window never is. So CSA can be admitted whenever
the longest token in the batch is under `c4_sparse_topk`. It is worth the work:
the eager arm counted `hits=3211` against `skips={'stream ratio 4': 2477}`, so
admitting CSA is close to doubling the kernel's coverage. Cheap test first —
`SGLANG_MLA_FLYDSL_RATIO=4` with `SGLANG_MLA_FLYDSL_CHECK=1` reports the
violation rate directly.

**And do not over-read the HCA prefix result.** The 0/124,992 was measured on
gsm8k, i.e. short contexts, with the gate already restricted to HCA. HCA is
prefix-shaped by construction so this is expected, but nothing has been
measured on a long-context agentic workload.

**Bug 2 — the verify window is not always 7.** The hook hardcoded `_Q_LEN=7`
and admitted any batch with `T % 7 == 0`. But
`target_verify_num_draft_tokens` is `speculative_num_draft_tokens - 1` on a
**DSpark draft worker** (6, not 7), and a plain decode step carries one token
per request. At 6 tokens/request, `T % 7 == 0` whenever the batch size is a
multiple of 7 — roughly one step in seven — and the kernel then regrouped the
batch into 7-token windows that do not exist. The width is now passed from the
backend (`q_len=target_verify_num_draft_tokens if verify_as_decode else 1`) and
anything other than 7 is skipped by name. **A divisibility test is not a
substitute for being told the shape**; it silently succeeds on a fraction of
batches, which is exactly the signature of a several-point accuracy loss with
no crash.

**The gate recipe was lost and is now rebuilt**:
`/shared_nfs/kk/gsm8k_flydsl_ab.sh` (`ARMS="off on"`, results and logs in
`/shared_nfs/kk/flydsl_gate/`). `gsm8k_segplan.sh` no longer exists on this
node; the CLI was recovered from
`/workspace/results/megamoe-eplb-c128-hcasplit4/sglang_command.txt` and the env
from the launcher's MegaMoE+DP+EPLB branches, deliberately **without**
`SGLANG_SIMULATE_ACC_*`, which fakes the MTP accept length and would invalidate
an accuracy run. The off arm reproducing 0.936 against the historical 0.937 is
what validates the reconstruction.

**Two operational traps this cost an hour to:**

- **`pkill -f "sglang::"` kills the shell running it**, because that shell's
  own command line contains the pattern. It returns instantly with no output
  and the teardown silently does nothing, so the next arm dies on
  `port_base at 9123 is not available`. Use `pkill -f 'sglang[:][:]'`.
- **`--disable-cuda-graph` is not a usable debug mode on this recipe.** It
  fails in an unrelated place (`gemm_a8w8_blockscale_bpreshuffle ... M=0`), so
  any diagnostic that needs eager execution has to find another route —
  `torch.cuda.is_current_stream_capturing()` and the pre-capture warmup.
- **Startup time is not stable**: 3 min on a clean node, and up to ~13 min
  after a `kill -9` teardown, with the log sitting silently on
  `sglang is using nccl==2.27.7` the whole time. Two launches were written off
  as hangs that were only slow. Wait on the "ready to roll" marker with a
  generous timeout; do not infer a hang from a quiet log.

**Hook visibility is now in the log.** `eligible` tallies and prints
`[flydsl_mla] calls=N hits=N skips={...}` at calls 1, 100 and every
`SGLANG_MLA_FLYDSL_LOG` (default 5000) thereafter. This exists because the
first arm ran with no way to tell a silent fallback from a real null.

---

## CONTINUE HERE (session 4, superseded above)

**Status (2026-09-20, end of session 4): the kernel is written, faster than
the shipped one, and wired into the production decode path behind
`SGLANG_MLA_FLYDSL=1`. Nothing is committed. The next step is the deployment
gates, which need a server.**

**Next, in order:**

1. **gsm8k 1319 with the hook on**, threshold 0.937 (today's baseline). Launch
   the usual DSV4 recipe with `SGLANG_MLA_FLYDSL=1` in the server env, run
   `benchmark/gsm8k`. This is the first gate the kernel has ever been good
   enough to be worth running.
2. **matched-bs Δstep**: two server runs (hook off, hook on), then
   `python3 /shared_nfs/kk/matched_bs.py <logs> name=off_substr,on_substr`.
   Remember "Barrier wait is not cashable" under "Method traps" — an MLA win
   can show up as ~0 end to end, so read the matched-bs cells, not the wall.
3. Only then more kernel work. The remaining block is softmax+PV at ~49 % of
   the kernel; see "The plan from here", whose Tier 0/1 are done and whose
   ranking has moved twice since it was written.

**Watch for at the server:** the hook refuses shapes it cannot handle and
counts them in `flydsl_decode_hook._skips` — a silent fallback to Triton looks
like "no regression and no win". It requires bf16 KV (not the fp8 two-pool
path), D=512, a whole number of 7-token verify windows, and H divisible by 16.
`SGLANG_MLA_FLYDSL_CHECK=1` additionally asserts the prefix assumption the
q-folding rests on, at ~5 ms/call — use it once, not in a timed run.

**VOID (session 6): every number in this section was measured with q-folding on,
i.e. with the kernel reading the wrong KV for six of every seven tokens. The
live baseline is the per-token table at the top, ~0.95x.**

**Standing, measured inside the production launcher against production's own
`_kernel_config` (bs=12, KV_SPLITS=4):**

| | Triton | FlyDSL | ratio |
|---|---:|---:|---:|
| stage-1 split kernel | 117.6 µs | **105.5 µs** | **0.90x** |
| shared tail (reduce + partial allocs) | 30.5 | 29.5 | — |
| end to end | 148.2 | **135.0** | **0.91x** |

Across shapes, same launcher, each side with its own best `KV_SPLITS`:
**0.78-0.92x at bs >= 8** (bs=8 0.79, bs=12 0.92, bs=24 0.80, bs=32 0.88, the
splits=2 CSA-like point 0.82) and **1.18-1.27x at bs <= 4 with 4 splits**,
which the split knob fixes (bs=1 at 16 splits is 0.92x). Output is correct
everywhere: relL2 1.1-1.5e-03 against the Triton path, the expected magnitude
for two independent bf16 P roundings.

**Two measurement corrections got it here, both mine, both worth remembering:**

1. **The harness under-launches the shipped kernel.**
   `mla_qfold_proto.run_shipped` uses `block_h=64, BLOCK_K=16, num_stages=2`;
   production picks its config from `_kernel_config` and runs the same kernel
   at **117.6 µs**, where the harness measures 138-147. Every ratio in the
   sections below is against the handicapped baseline and is optimistic by
   ~20 %.
2. **The harness also adds a shared constant to both sides.** Its timed
   closure allocates the 88 MB of partials on every call, which inflates both
   kernels by the same ~30-40 µs and therefore **compresses the ratio toward
   1.0**. That is why the harness reported parity (145 vs 145) where the
   production launcher reports 0.90x. A shared additive overhead is not
   harmless just because it is shared.

**Shape verification is done and it is better than the headline** — correct
at every shape tried, and with each kernel given its own best `KV_SPLITS`,
**faster at seven of eight shapes by 3-17 %** (bs=1 0.83x, bs=8 0.84x, bs=24
0.89x, bs=32 0.95x, SWA 0.86x, CSA 0.87x, bs=4 0.97x) and **level at bs=12**,
which is the production p50. See "Step 11"; two fixes found by that sweep
(dropping a dynamic `if` around the kernel body, and the split-count knob at
low batch) are what turned the two losses into wins.

**The two deployment gates are blocked on plumbing, not on the kernel**: both
read server logs, so the kernel has to be wired into the decode path first,
and the index layout it wants is not what the decode path hands down. See
"The deployment gates are blocked on plumbing". It
was 28x four steps ago. Every arm from B2 onward is **bit-identical** in output, so all of
this has been pure throughput work.

**The gap was attributed by ablation, and it was not where this doc guessed.**
81 % of it was the softmax and the P publication — both stage-A shortcuts that
became pure waste the moment PV moved onto the atom — and fixing them took
4,103 µs to 1,019 µs with bit-identical output. Global→LDS staging, the
hypothesis this doc carried into the measurement, was **4 %**; do not spend a
day on async 128-bit copies. See "Where the time actually goes" for the ladder
and for what B3's remaining time is made of.

**Next: stage C, in this order.**

1. ~~Make the softmax one thread per (q, kv) pair.~~ **DONE** — `B3`,
   `build_kernel(..., fast_softmax=True)`. 4,103 → **1,019 µs**, output
   bit-identical to B2. Details under "Where the time actually goes".
2. ~~Fold heads into M.~~ **DONE up to 4 heads per workgroup** —
   `build_kernel(..., h_per_wg=N)`. 1,020 → **323 µs**, still bit-identical.
   See "Head folding: measured, and how far it actually fits".
3. ~~Replace the D-split QK with a per-wave `(m_tile, n_tile)` split.~~ **DONE**
   — it removed the LDS wall and `h_per_wg=8` now runs at **269 µs**.
4. ~~Give QK operand reuse.~~ **DONE** — `B6`,
   `build_kernel(..., qk_reuse=True)`. 269 → **249 µs**, output unchanged
   (relL2 1.184e-03). See "Where B6's time goes".
5. ~~Get `h_per_wg=16` to fit.~~ **DONE** — `B7`,
   `build_kernel(..., kv_single=True, alias_p=True)`. M=112 now launches and is
   correct, at **242-245 µs** against B6 hw=8's 248-253. **The M=112 the whole
   design was written around buys ~2 %, and the reason is structural** — see
   "Step 5: M=112 fits, and why it barely pays".
6. ~~Hoist the loop-invariant QK A fragments into registers.~~ **DONE** —
   `B8`, `build_kernel(..., q_resident=True)`. 254 → **227 µs at hw=8**, and it
   moved the best geometry back from hw=16 to hw=8. See "Step 6".
7. ~~Get a second CTA resident per CU by aliasing the dead Q region.~~
   **CLOSED, and the reason is worth reading before any other LDS work** —
   `B10`, `build_kernel(..., lds_pool=True)`. The pool itself is free and the
   *overlap* costs **+133 µs** (227 → 357), independent of the LDS size it
   saves. Occupancy was ruled out by a padding arm, not assumed. See "Step 7".
   Both routes to a smaller LDS footprint are now measured regressions
   (`q_global` in step 6, `lds_pool` here), and **no occupancy win has ever
   been demonstrated for this kernel** — do not spend on a third route without
   first measuring that 2 CTAs/CU is worth anything here.
8. ~~Vectorise the KV staging.~~ **DONE** — `B11`,
   `build_kernel(..., kv_vec=8)`. 227 → **195 µs**, and it revived `h_per_wg=16`
   (330 → 195) because the scalar path's register cost was part of what made
   that arm spill. See "Step 8". **This item had been closed by this doc as
   "4 %, do not spend a day on it"** — true at 4,103 µs, stale at 227.
9. ~~Profile before touching QK.~~ **DONE, and it was the whole ballgame** —
   `rocprofv3` said LDS bank conflicts 26.8 % (shipped: 9.3 %), and one line
   of row-stride padding (`kv_pad=8`) took 195 → **145 µs**. See "Step 9".
10. **Now: run the deployment gates** (see the top of this section). After
   that, the remaining kernel ideas are in "The plan from here", re-ranked by
   what the profile found.

## The plan from here

Attribution of the **winning** arm (B11 hw=16, 195 µs,
`--abl --hw=16 --reuse --kv1 --qres --vec8`), which is what this plan is
ordered by — note it is *not* the hw=8 ladder the previous steps used:

| rung | hw=16 | hw=8 | share of 195 |
|---|---:|---:|---:|
| launch + epilogue floor | ~28 µs | 29 µs | 14 % |
| + tile loop + staging | ~20 µs | 28 µs | 10 % |
| + QK MFMA | **80 µs** | 68 µs | **41 %** |
| + fast softmax + PV | 68 µs | 77 µs | 35 % |

> **Tier 0 is done and it invalidated most of what follows** — the 80 µs QK
> rung was LDS bank conflicts, fixed by `kv_pad=8` (see "Step 9"). The ladder
> below is the *pre-fix* attribution; re-run `--abl --hw=16 --reuse --kv1
> --qres --vec8` before using it. Of the tiers below, item 3 (re-open the QK
> split) is now much less attractive — its premise was the unexplained
> traffic — while the swizzle (to get padding hw=16 cannot afford) and item 6
> (overlap) are the live ones.

### Tier 0 — profile before touching QK (do this first)

QK is the biggest rung and **its cost has no attributed mechanism**. Counting
what it issues at hw=16: per wave per tile, 16 k-chunks x 2 n-tiles = 32
`ds_read_b128` + 32 MFMAs, and the whole 32 KB KV tile is read once per wave
(8x amplification per CTA, 256 KB/tile). At CDNA's 128 B/clk/CU that is ~14 µs
of LDS traffic across the call, and the MFMA issue cost is smaller still —
against **80 µs measured**. A 5x unexplained factor means the next structural
QK change is a guess, and this kernel has already paid for two of those
(`q_global`, `lds_pool`).

Get `rocprofv3` on the B11 hw=16 arm and read: `OccupancyPercent`, `MfmaUtil`,
LDS bank-conflict and `MemUnitStalled`-class counters, and if possible an ISA
dump to count `s_waitcnt` / `s_barrier` in the tile loop. That distinguishes
the three live hypotheses — bank conflicts on the B-fragment reads, barrier
skew across 8 waves, and plain latency with 1 CTA/CU and nothing to overlap —
which imply three *different* fixes. Cost: hours. Everything in Tier 2 depends
on the answer.

### Tier 1 — mechanical, cheap, aimed at named rungs (do regardless)

1. **Hoist the `alpha` LDS loads in `_pv_atoms`** (targets the 68 µs rung).
   `fx.ptr_load(lds_alpha + row)` sits in the `t` loop although `row` depends
   only on `(mt, i)`: 112 loads per lane per tile at hw=16 where 28 would do.
   LLVM may already CSE them — which is exactly why this is a 20-minute
   experiment and not a design. Falsified if the rung does not move.
2. **128-bit LDS store to match the 128-bit load** (targets the 20 µs rung).
   Staging still issues 8 `ds_write_b16` per thread per round. Then **re-test
   `q_vec`**, whose null was attributed to exactly this (bank-conflicting
   scalar stores) — the two only pay together.

### Tier 2 — structural, and only with Tier 0's answer in hand

3. **Re-open the QK work split.** Each wave reading the whole KV tile is a
   consequence of "one m-tile per wave", chosen back when A came from LDS.
   Now that A is resident, the trade has changed: splitting k across waves
   would cut B traffic 8x at the price of the score reduction B6 removed.
   Re-measure it rather than assuming B6's verdict still holds.
4. **Give wave 7 work** — 7 m-tiles over 8 waves idles 12.5 % at hw=16. A
   2-way D-split on one m-tile, not `qk_reuse=False` (14 items over 8 waves is
   the same 87.5 %).
5. **Double-buffer the KV tile** so the next tile's global load issues under
   the current tile's MFMAs. Prize is bounded by the staging rung, **≤20 µs**
   at hw=16 — it got much smaller when `kv_vec` landed, so it is no longer the
   headline item it looked like.

### Tier 3 — the only routes with a sub-146 ceiling

6. **Overlap the phases instead of queueing them.** Pipelined, the steady
   state is `max(rung)` not `sum(rung)`: **28 + ~80 ≈ 108 µs**. This is the
   first credible sub-146 target the project has had, and it is the reason to
   keep going at all. CDNA4 caveat, already priced: there is no WGMMA-from-LDS,
   MFMA sources from VGPRs, so an FA3-style producer/consumer split cannot
   remove the LDS→VGPR hop — what ports is the *pipelining*, not the role
   split. The async global→LDS atom (`BufferLoadAsyncLDS128b`) and a worked
   dual-wave example are in the in-tree `fmha_gfx950` pipeline.
7. **fp8 KV.** B200's 40.2 µs kernel is `flash_fwd_splitkv_mla_fp8` — **fp8**,
   against our bf16, so a large part of that 4x is dtype, not structure. Our
   fp8 path is blocked on HiCache x two-pool integration (see "Closed").

### Not on the plan, with reasons

- **AGPR accumulators.** No longer justified: they were to pay for the hw=16
  spill, and `kv_vec=8` removed the spill.
- **`KV_SPLITS` 4 → 8.** Not free: partials are `[N, SPLITS, H, D]` fp32
  ≈ 88 MB/call ≈ 25 µs of writes, so it buys grid parallelism at ~25 µs, and
  `SPLITS` must stop being module-level in `mla_qfold_proto.py` first. Revisit
  only if Tier 0 says the kernel is CTA-parallelism starved.
- **A smaller LDS footprint for occupancy.** Two measured regressions
  (`q_global`, `lds_pool`) and the prize was never measured. If Tier 0 reports
  occupancy as the binding constraint, that changes — and Tier 0 is how we
  would find out.
- **Adopting `pa_decode_gluon`'s SHUFFLE 5D KV layout.** It is an MHA kernel;
  MLA's 512+64 latent does not map onto it, and changing the pool layout is a
  far larger change than anything above.

## How to pick the next step (this project's own scoring rules)

Four of this kernel's wrong turns and one of its wins came from the same small
set of habits. These are not generic advice; each line is something that
already happened here.

1. **Attribute before optimising, and only spend on a rung whose share you
   measured.** The staging hypothesis was argued from code structure for a day
   and turned out to be 4 %; the softmax was 47 % and nobody had guessed it.
2. **Pre-register the prediction as a *rung*, not as a total.** B8 predicted
   "the QK rung falls"; it fell 90 → 64 µs and the total moved by the same
   amount, so the mechanism is believed. A total that moves while the predicted
   rung does not is luck, and it will mislead the next three decisions.
3. **The ablation ladder systematically hides register pressure** — its tail
   uses a fixed `N_ACC = 4`, so no rung carries the real accumulator. Any
   change that adds live registers (`q_resident`, `q_global`, AGPR work) has to
   be judged on a full arm. B9's ladder looked fine and the full arm was 397 µs.
4. **A data-movement change needs a prediction about the access *pattern*, not
   just the traffic.** B9 removed a 39.7 µs staging rung exactly as predicted
   and lost 160 µs to uncoalesced reads that no one had priced.
5. **One variable per arm, every arm a flag on the same kernel, control in the
   same process.** This is why B5-vs-B6-vs-B7-vs-B8 is readable at all.
6. **Correctness from the full `--mfma` run, time from a `--only=` run of two
   or three arms.** The full run inflates its late arms by up to 25 %.
7. **Record nulls with their mechanism, in this doc, the same day.** `QK_ACCS`,
   `block_k`, the length-aware plan, `q_global` and `lds_pool` are all here so
   that the next session's "obvious idea" is already priced.
8. **A change that alters *nothing* semantically can still cost 57 %, so give
   it control arms that separate its parts.** `lds_pool` bundles three things:
   one allocation, computed offsets, and reuse of dead space. `pool_split`
   (same pool, no reuse) and `pool_pad` (reuse, same LDS size class) cost one
   build flag each and turned "the pool is slow" into "the overlap is slow, and
   it is not occupancy". Without them the natural next move would have been a
   day inside the register allocator, aimed at the wrong thing.
9. **Verify the baseline is the production configuration before trusting any
   ratio.** This doc spent nine steps closing on 145 µs, which was the
   harness's under-tuned launch of the shipped kernel; production's own config
   runs the same kernel at 117 µs. A micro-harness is a control arm only if it
   launches the control the way production does.
10. **When a rung's arithmetic does not close, profile it — do not redesign
   it.** The QK rung was 80 µs against ~15 µs of countable LDS traffic and
   MFMA issue. Three structural attempts were aimed at that gap (`q_global`,
   `lds_pool`, and a planned QK re-split); the answer was a bank conflict,
   found in one `rocprofv3` run and fixed by adding 8 to a row stride. A 5x
   unexplained factor is a *measurement* task, not a design task.
11. **A closed "it's only N %" item re-opens when the total moves.** Staging was
   4 % of 4,103 µs and 28 % of 227 µs without anyone touching it; fixing it
   was then worth 14 %. Record what the percentage was *of*, and re-check the
   cheap items after any 2x.
12. **Never buy a prize you have not measured.** Steps 6 and 7 both spent on a
   smaller LDS footprint to get a second resident CTA, and nobody had ever
   measured what a second resident CTA is worth here. Measure the prize before
   the third attempt at winning it.

The gsm8k and matched-bs Δstep gates in "Pass criteria" are still not worth
spending — they gate a *candidate replacement*, and at 1.70x slower than the
kernel it would replace this is not one yet. They become the right next move
the moment a variant beats 145 µs.

**Files:** the kernel is `/shared_nfs/kk/flydsl_mla_decode.py`. Everything is
one `build_kernel(D, Q_LEN, BLOCK_K, SPLITS, mfma_qk, mfma_pv, fast_softmax,
h_per_wg, qk_reuse, kv_single, alias_p, q_resident, q_global, lds_pool,
pool_split, pool_pad, kv_vec, q_vec, kv_pad, s_pad, ablate, dbg_scores)`, so every arm from
stage A to B11 is a
flag on the same kernel and they can all be gated against each other in one
process. The current best is `BLOCK_K=32, mfma_qk=True, mfma_pv=True,
fast_softmax=True, h_per_wg=16, qk_reuse=True, kv_single=True, alias_p=True,
q_resident=True, kv_vec=8, kv_pad=8`.
The atom probe is
`/shared_nfs/kk/flydsl_mfma_probe.py`, the harness (`build`, `merge`,
`run_shipped`, `timeit`) is `/shared_nfs/kk/mla_qfold_proto.py`, and the kernel
to beat is `_paged_decode_split_kernel` in
`sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py`.

**Repro (the number to beat), one command:**

```bash
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/mla_qfold_proto.py 12
# shipped (1 token/CTA, M=64, acc[64,512]) = 145.8 us at the production shape
```

**Repro (the integrated path, correctness + time against production's own
Triton config, no server, ~10 s):**

```bash
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/verify_integration.py      # shape sweep, hook off vs on
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/verify_shapes.py           # kernel only, harness baseline
```

**Repro (every stage, correctness + time, one command, ~12 s):**

```bash
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/flydsl_mla_decode.py 12 --mfma
```

It builds twenty-one variants of the same kernel (stage A at BLOCK_K 32, B1 at
16 and 32, B2/B3 at 32, B5 and B6 at `h_per_wg` 4 and 8, B7/B8/B11 at 8 and 16,
B9 at 4 and 8, B10 at 4 and 8 plus its two control arms), times the shipped
kernel in the same process, and scores everything
against `ref_torch()`. Drop `--mfma` for stage A alone. Swap `--mfma` for
`--abl [--hw=N] [--reuse] [--kv1] [--qres] [--qg] [--vec8]` to re-run the attribution
ladder, or `--dbg` for the FMA-vs-MFMA score diff.

**Add `--only=B6,B7` (substring match on the arm tags) whenever you are reading
a time**, not just a relL2. Timing eleven kernels in one process inflates the
late arms: B7 hw=16 reports **308 µs** in the full run and **242-245 µs**,
reproducibly, when only two arms are built. The full run is the correctness
gate; a 2-3 arm run is the timing gate.

**Pass criteria:** merged-output relL2 against `ref_torch()` — NOT against the
shipped kernel, which is itself 1.2e-3 off (see "The correctness gate"), then
gsm8k 1319 ≥ 0.937 (today's baseline, `gsm8k_segplan.sh` is the working
recipe), then matched-bs Δstep via `/shared_nfs/kk/matched_bs.py`.

## Stage B results (2026-09-18)

At bs=12, one process, shipped kernel timed alongside. relL2 is against
`ref_torch()`; the shipped kernel scores 1.177e-03 on the same run.

| variant | relL2 vs torch | time | note |
|---|---:|---:|---|
| shipped Triton | 1.177e-03 | **145.8 µs** | reproduces the doc's baseline exactly |
| stage A, BLOCK_K 32 | 4.170e-07 | 8,504 µs | control, no MFMA |
| B1 (QK on MFMA), BLOCK_K 16 | 4.138e-07 | 3,307 µs | |
| B1, BLOCK_K 32 | 4.148e-07 | 4,140 µs | |
| B2 (QK+PV on MFMA), BLOCK_K 32 | 1.184e-03 | 4,063 µs | P rounded to bf16 |
| B3 (= B2 + `fast_softmax`) | 1.172e-03 | **1,019 µs** | bit-identical to B2, 4.0x faster |
| B4 `h_per_wg=2` | 1.180e-03 | 564 µs | bit-identical again |
| B4 `h_per_wg=4` | 1.180e-03 | 323 µs | old D-split QK |
| B5 `h_per_wg=8` | 1.181e-03 | 269 µs | per-wave `(m,n)` QK, 1.85x off shipped |
| B6 `h_per_wg=8` | 1.184e-03 | **249 µs** | + QK operand reuse, 1.70x off shipped |
| B7 `h_per_wg=16` | 1.174e-03 | 242 µs | + `kv_single`/`alias_p`, M=112, 1.66x off shipped |
| B8 `h_per_wg=8` | 1.183e-03 | 227 µs | + resident QK A fragments, 1.55x off shipped |
| B11 `h_per_wg=16` | 1.186e-03 | 195 µs | + 128-bit KV staging, 1.33x off shipped |
| B12 `h_per_wg=16` | 1.184e-03 | **145 µs** | + `kv_pad=8`, **level with shipped** |

B4, B5 and B6 differ only in how QK is split across waves; see "Head folding"
for that table in full.

**The P-in-bf16 decision, made deliberately.** B2 feeds the PV atom a bf16 P,
which moves accuracy from 4.1e-07 to **1.184e-03** — i.e. onto exactly the same
footing as the shipped Triton kernel (1.177e-03), which rounds P the same way.
B2 vs shipped is 6.7e-04, *smaller* than either kernel's distance from torch,
which is what two independent roundings of the same quantity should look like.
Keeping P in fp32 would mean an fp32 MFMA (`mfma_f32_16x16x4_f32`) and a ~8x
slower atom for accuracy the production kernel does not have today. Recorded
choice: **round P to bf16**; revisit only if an accuracy regression is traced
to it.

Geometry as built: one workgroup owns (request, head, kv-split) and all 7 draft
tokens; 512 threads = 8 waves; the QK contraction over D=512 is split evenly
across the 8 waves (2 k-chunks each) and reduced through LDS; PV gives each
wave 4 output d-tiles of 16 columns. BLOCK_K must be 32 for B2 because PV
contracts over BLOCK_K and the atom's K is 32.

## Where the time actually goes (ablation, bs=12)

`python3 /shared_nfs/kk/flydsl_mla_decode.py 12 --abl` builds a cumulative
ladder at B2's geometry: each rung adds one stage of the per-tile pipeline and
ends in a tail that consumes just enough of the result that nothing can be
dead-code-eliminated. Outputs are wrong by construction; only time is read.

| rung | time | delta | share of B2 | share of B3 |
|---|---:|---:|---:|---:|
| tile loop + index gather | 224 µs | | 6 % | **22 %** |
| + Q/KV/KVT → LDS | 401 µs | +177 | **4 %** | 17 % |
| + QK MFMA + LDS reduce | 727 µs | +326 | 8 % | **32 %** |
| + online softmax | 2,596 µs | **+1,869** | **47 %** | — |
| + PV MFMA (= B2) | 3,955 µs | **+1,359** | **34 %** | — |
| + fast softmax (= B3) | **1,019 µs** | −2,936 | — | 29 % (softmax+PV) |

The ladder reconstructs B2's standalone time (3,955 vs 4,063 µs measured
separately), so it is not missing a stage.

**The staging hypothesis was wrong.** This doc argued from code structure that
the 16-bit synchronous global→LDS staging had to be the problem, against the
in-tree template's async 128-bit copies. It is 172 µs — 4 %. The structural
argument was correct and the conclusion was not, which is the whole reason the
number had to be measured before optimising.

**What the two big rungs actually are.** Neither is the MFMA. Both are stage-A
shortcuts that only became waste once PV moved onto the atom:

- *Softmax, 1,871 µs.* All 512 threads compute the identical
  `Q_LEN x BLOCK_K` = 224-element `exp2`/max/select block every tile. In stage
  A that was necessary — each thread needed every `p[q, kk]` in registers to
  accumulate its own output column. In B2, P is read from LDS by the atom, so
  511 of every 512 copies are dead work.
- *PV rung, 1,354 µs.* This is not the atom, which is four `16x16x32` MFMAs per
  wave per tile. It is dominated by publishing P: because every thread holds
  all of `pvals` at compile-time indices, row `q` is written by the single
  thread with `d == q`, so 7 threads issue 32 LDS stores each behind a barrier
  while the other 505 wait.

Fixing the softmax redundancy fixes the publish too — when thread `d` computes
only its own `(q, kk)` pair, it also stores only its own `p`.

**Done, and it paid exactly as predicted (B3, `fast_softmax=True`).** Those two
rungs went from 3,225 µs to **292 µs, an 11x cut**, taking the whole kernel from
4,103 to **1,019 µs** — 4.0x, with **bit-identical output** (relL2 1.172e-03,
and the same 6.636e-04 against the shipped kernel). The gap to shipped is now
**7.0x, down from 28x**. What changed:

- Row stats (max, rescale factor, running sum) are computed by `Q_LEN` threads,
  each scanning its own row of `lds_s`; every other thread does exactly one
  `exp2`. Per tile that is ~2 x `N_PAIRS` exp2 instead of `BLOCK_TH` x
  `N_PAIRS` — 448 instead of 114,688.
- `m` and `l` moved out of per-thread carried state into two `Q_LEN`-element
  LDS arrays that persist across the tile loop, so the epilogue reads them from
  LDS. The carried state is now just the accumulator fragments.
- P publication is now one bf16 store per thread, no longer 7 threads x
  `BLOCK_K` stores behind a barrier.

**B3's remaining 1,019 µs is roughly balanced**, which is why the next step was
structural rather than another micro-fix: 22 % tile loop + epilogue, 17 %
staging, 32 % QK atom + LDS reduce, 29 % softmax + PV.

## Head folding: measured, and how far it actually fits

`build_kernel(..., h_per_wg=N)` gives one workgroup N heads x Q_LEN draft
tokens, laid out head-major (`row = hh * Q_LEN + q`), with `grid.y = H / N`.
MLA has a single KV head, so the KV tile, its LDS staging and the whole tile
loop are amortised over all N. Output is **bit-identical** at every N.

| `h_per_wg` | M rows | time | vs shipped | QK work split |
|---:|---:|---:|---:|---|
| 1 (= B3) | 7 | 1,020 µs | 7.0x | D across 8 waves |
| 2 | 14 | 564 µs | 3.9x | D across 8 waves |
| 4 | 28 | 323 µs | 2.2x | D across 8 waves |
| 4 (B5) | 28 | 358 µs | 2.5x | per-wave `(m,n)` tile |
| 4 (B6) | 28 | 337 µs | 2.3x | per-wave `m` tile, all n-tiles |
| 8 (B5) | 56 | 269 µs | 1.85x | per-wave `(m,n)` tile |
| 8 (B6) | 56 | **249 µs** | **1.70x** | per-wave `m` tile, all n-tiles |
| 16 (B7) | 112 | **242 µs** | **1.66x** | per-wave `m` tile, `QK_DSPLIT=1` |

**Why `h_per_wg` stopped at 4, and the QK rewrite that unblocked 8.** The
binding constraint is LDS, and the term that blew up was `s4`, the per-wave
partial score tile the D-split QK needed: `NWAVE x M_ROWS x BLOCK_K x 4`, i.e.
8x the score data itself. At `h_per_wg=8` that put the workgroup at ~197 KB
against a 160 KB limit.

That split only existed because at M=7 there was one MFMA M-tile and nothing
else to give the other waves. `_qk_mfma` now assigns each wave its own
`(m_tile, n_tile)` and splits D only when there are fewer output tiles than
waves (`QK_DSPLIT = max(1, NWAVE // (M_TILES * NT_N))`). When `QK_DSPLIT == 1`
the wave owns the whole contraction for its tile and stores the **already
scaled** score straight into `lds_s`, so `s4` and the entire reduction pass
disappear. `h_per_wg=8` then fits and runs at 269 µs.

Two things to know before building on this:

- **It cost 11 % at `h_per_wg=4`** (323 → 358 µs), where `QK_DSPLIT` is still 2.
  The win at 8 more than pays for it, but if the low arms ever matter again,
  that regression is unexplained.
- **This table's winner has moved twice since it was written.** `q_resident`
  flipped it to hw=8 (the A fragments a wave holds scale with `CH_PER_SLICE`,
  which doubles when `QK_DSPLIT` drops to 1 at hw=16, and hw=16 spilled), and
  `kv_vec=8` flipped it back to hw=16 by removing enough register pressure for
  that arm to stop spilling. **Re-check the h_per_wg sweep after any change to
  the register budget**; it is not a fixed ranking. See "Step 6" and "Step 8".
- **`h_per_wg=16` needed 40 KB it did not have** — Q 112 KB + KV 32 + KVT 32 +
  s 14 + p 7 ≈ 198 KB. `kv_single` and `alias_p` freed exactly that; see
  "Step 5". The failure mode if you get the budget wrong is not a compile
  error: it is `hipErrorIllegalState` at launch.

## Where B5's time goes (`--abl --hw=8`, total 269 µs)

| rung | time | delta | share |
|---|---:|---:|---:|
| launch + epilogue only (zero tile iterations) | **30 µs** | | 11 % |
| + tile loop, no staging | 140 µs | +110 | *invalid, see below* |
| + Q/KV/KVT → LDS | 99 µs | −41 | — |
| + QK MFMA | 209 µs | +110 | **41 %** |
| + fast softmax + PV | 269 µs | +60 | 22 % |

**The `empty` rung is trustworthy and the `gather` rung is not.** Forcing zero
tile iterations gives a clean **30 µs floor** for launch, metadata and the
epilogue at both `h_per_wg` 4 and 8, and that matches the partial-output volume
(`[N, SPLITS, H, D]` fp32 ≈ 88 MB per call, ~25 µs of writes). The `gather`
rung, which keeps the tile loop but skips staging, measures *higher* than the
rung above it — adding work made it faster — so it is over-measuring: its tail
is a loop-carried dependency chain with no memory work to overlap. Read the
ladder as floor 30, loop+staging ~69, QK ~110, softmax+PV ~60.

**QK was the largest item at 41 %, and it was not compute-bound.** ~13 GFLOP of
bf16 in 110 µs is ~119 TFLOP/s, a few percent of peak, so the cost was the LDS
reads feeding the fragments rather than the atoms. The structural reason is
that one `(m_tile, n_tile)` per wave gives **zero operand reuse** — every MFMA
loads both a fresh A and a fresh B fragment.

## Where B6's time goes (`--abl --hw=8 --reuse`, total 249 µs)

**Step 4 is done: `build_kernel(..., qk_reuse=True)`.** Each wave now owns one
`m_tile` and *all* `NT_N` n-tiles, and the A fragment is loaded once per
k-chunk and feeds `NT_N` MFMAs. `QK_ITEMS` drops from `M_TILES * NT_N` to
`M_TILES`, so at `h_per_wg=8` (4 m-tiles, 8 waves) `QK_DSPLIT` goes back to 2
and the 2-way LDS score reduction returns (`s4` ≈ 14 KB, and the workgroup
lands at ~154 KB against the 160 KB limit).

| rung | B5 | B6 |
|---|---:|---:|
| launch + epilogue only | 30 µs | 30 µs |
| + tile loop + staging | ~69 µs | ~69 µs |
| + QK MFMA (+ reduce) | 110 µs | **91 µs** |
| + fast softmax + PV | 60 µs | 59 µs |
| total | 269 µs | **249 µs** |

The QK rung fell 17 % *while also absorbing* a 2-way reduction B5 did not pay
for, so the reuse itself is worth more than the 19 µs the total moved. Output
is unchanged (relL2 1.184e-03 vs torch, same as every arm since B2), and the
same change is worth 6 % at `h_per_wg=4` (358 → 337 µs) where `QK_DSPLIT` goes
2 → 4. `h_per_wg=16` would give 7 m-tiles against 8 waves, i.e. `QK_DSPLIT=1`,
which removes the reduction entirely — that is step 5 and it is now also the
thing that makes this step pay in full.

**Measured null, recorded so it is not retried:** giving each wave 4 independent
QK accumulators instead of 1, to stop back-to-back MFMAs serialising on the
atom's latency, changed nothing at `h_per_wg` 4 (359.8 vs 360.5 µs) or 8 (270.1
vs 270.8). `QK_ACCS` is left at 1. The QK rung is not MFMA-latency bound, which
is consistent with the operand-traffic reading above.

## Step 5: M=112 fits, and why it barely pays

`build_kernel(..., kv_single=True, alias_p=True)` (`B7`) frees the 40 KB
`h_per_wg=16` needed. Both savings are bit-neutral — every B7 arm still scores
1.174e-03 against `ref_torch()`, the same as every arm since B2.

- **`kv_single` (−32 KB).** The second, transposed KV tile is gone. PV's B
  operand is built straight out of the `(kv, d)` tile by reading the atom's
  per-lane `(row, col)` map off `thr_mma.partition_B` applied to two coordinate
  views — the same trick `partition_C` gives for the accumulator — and issuing
  `NREG_B = 8` scalar LDS loads per fragment. **The `LDSReadTrans16_64b` route
  this doc proposed was not needed**; it would have meant guessing the
  transposed copy atom's tiling, and the coordinate-view route has no layout to
  guess. The extra loads are paid once per tile, not per output tile, because
  the same change hoists the wave's `NT_D_PER_WAVE` B fragments out of the
  m-tile loop and loads A once per m-tile (PV's version of `qk_reuse`).
- **`alias_p` (−7 KB).** `p` is a `fx.recast_iter(fx.BFloat16, lds_s)` view of
  the f32 score buffer, which is twice its size. Two conditions, both in the
  code: every thread computes its `p` values into registers and a **barrier**
  separates all the `s` reads from the first `p` store, and the atom pad rows
  (Q_ROWS − M_ROWS, zero at hw=16 but 8 at hw=8) are re-zeroed every tile
  because `s` overwrites them.

LDS at hw=16: Q 112 KB + KV 32 + s 14 + alpha/m/l 1.3 ≈ **159 KB** against the
160 KB limit — it fits with under 1 KB to spare, so **any new LDS at this
geometry has to come out of something else.**

| arm | time | note |
|---|---:|---|
| B6 hw=8 | 248-253 µs | previous best |
| B7 hw=8 | 253-257 µs | the two savings cost ~2 % where LDS was not binding |
| B7 hw=16 | **242-245 µs** | M=112, 1.66x off shipped |

**Where hw=16's time goes (`--abl --hw=16 --reuse --kv1`), against hw=8:**

| rung | hw=8 | hw=16 |
|---|---:|---:|
| launch + epilogue only | 29 µs | 29 µs |
| + tile loop + staging | ~65 µs | **~40 µs** |
| + QK MFMA (+ reduce) | 90 µs | **111 µs** |
| + fast softmax + PV | 73 µs | 64 µs |
| total | 256 µs | 244 µs |

**The design's prediction held and was cancelled out.** Staging *did* halve
(65 → 40 µs) — the KV tile amortised over twice the heads, exactly what folding
heads into M was for. But QK went the other way, +23 % for the same total work,
and the two reasons are geometric, not about registers:

- **7 m-tiles over 8 waves idles one wave in eight.** `QK_DSPLIT` is 1 at
  hw=16, so the 2-way score reduction B6 pays for does disappear as predicted —
  and wave 7 has no m-tile, so 12.5 % of the workgroup runs the MFMAs with
  `mt` forced to 0 and throws the result away.
- **The grid halves: 384 CTAs against 256 CUs.** At hw=8 it is 768, i.e. 3 per
  CU; at hw=16 it is 1.5, so half the CUs run one CTA and idle through a second
  round. 90 µs / 0.875 ≈ 103 µs, and the tail accounts for the rest.

**Register pressure is ruled out as the cause.** The 111 µs QK rung is built
with the ablation tail's fixed `N_ACC = 4`, so it never allocates the 112
fp32-per-lane accumulator that M=112 implies. Whatever is wrong with QK at
hw=16, AGPRs will not fix it — which is why step 6 is re-ordered to put
`KV_SPLITS` and wave utilisation ahead of them.

## Step 6: Q fragments resident (B8), and the q_global null (B9)

**B8 — `q_resident=True`, 254 → 227 µs at hw=8, the new best (1.55x).** With
`qk_reuse` a wave owns one m-tile and a fixed set of k-chunks for the *whole*
tile loop, so its QK A fragments are loop-invariant — and the kernel was
re-reading them out of LDS on every one of the ~37 tiles a production kv_len
(1192 / BLOCK_K 32) produces. They are now loaded once before the loop.
`_load_A`/`_load_B`/`_mma` moved out of the tile-loop body to make that
possible; nothing else changed, and output is unchanged.

**The win landed in the rung it was predicted to** (`--abl --hw=8 --reuse
--kv1 [--qres]`), which is the only reason to believe the mechanism:

| rung | B7 | B8 |
|---|---:|---:|
| launch + epilogue only | 30 µs | 28 µs |
| + tile loop + staging | ~64 µs | ~65 µs |
| + QK MFMA (+ reduce) | 90 µs | **64 µs** |
| + fast softmax + PV | 72 µs | 73 µs |
| total | 256 µs | 229 µs |

**It also moved the best geometry back from hw=16 to hw=8**, and the reason is
the register budget, measured: the A fragments cost `CH_PER_SLICE * NREG_A`
bf16 per lane, which is 8 chunks x 8 = 32 registers at hw=8 (`QK_DSPLIT=2`) but
16 x 8 = 64 at hw=16 (`QK_DSPLIT=1`). On top of the accumulator — 64 fp32/lane
at hw=8, 112 at hw=16 — the first fits and the second spills: **hw=16 regresses
to 330 µs**. That is the AGPR lever's real job now (step 9).

| arm | hw=4 | hw=8 | hw=16 |
|---|---:|---:|---:|
| B7 | 337 µs | 255 µs | 244 µs |
| B8 (`q_resident`) | — | **227 µs** | 330 µs (spill) |
| B9 (`q_global`) | 370 µs | 397 µs | — |

**B9 — `q_global=True` is a measured regression; do not retry it.** The idea
was to delete the Q LDS buffer outright by filling the A fragments straight
from global memory with the `partition_A` coordinate trick (`_load_A_global`,
still in the file), which would have taken the workgroup from 118 KB to ~54 KB
and bought a second resident CTA per CU. It is correct and **397 µs at hw=8**.
Two things to learn from it:

- *The prediction about staging was right and irrelevant.* The staging rung did
  fall as predicted, by 39.7 µs, because Q no longer goes through LDS at all.
- *The cost showed up in a different rung than any of the reasoning covered.*
  The final rung (fast softmax + PV) went from +73 µs to **+232 µs**, while the
  ablation rungs — which carry a 4-register dummy accumulator — barely moved.
  So the damage appears only once the real 64-register accumulator is live,
  which points at the compiler spilling or re-materialising 64 uncoalesced
  global loads inside the tile loop under that pressure. The LDS staging path
  reads Q **fully coalesced** (thread `d` reads column `d`, 512 threads over
  1 KB rows); `_load_A_global` has each lane read a different `(token, head)`
  row, i.e. 16 distinct cache lines per load instruction, 8 dwords per lane per
  fragment.

The occupancy idea itself is still open — it just has to keep the coalesced
staging read and *alias* the dead Q region instead of deleting it. That is
step 7.

## Step 7: the LDS pool is free, the overlap is not (B10)

The plan was to lay the per-tile buffers over the Q tile — dead once
`q_resident` has built its fragments — taking hw=8 from ~118 KB to ~65 KB and
so from one resident workgroup per CU to two. It is implemented
(`lds_pool=True`: one `pool` array, `fx.recast_iter` for the f32 views,
loop-carried `alpha`/`m`/`l` deliberately left *outside* the pool), it is
correct, and it is **a 57 % regression**.

| arm at hw=8 | LDS | time |
|---|---:|---:|
| B8, separate allocations | 118 KB | **227 µs** |
| B10s `pool_split` — one pool, buffers placed *after* Q | 117 KB | **226 µs** |
| B10 `lds_pool` — buffers placed *over* Q | 65 KB | **357 µs** |
| B10p `lds_pool, pool_pad=12288` — over Q, padded back up | 88 KB | **359 µs** |

Read the four rows in order, because each one kills a hypothesis:

- **B10s says pooling is free.** Same single allocation, same computed offsets,
  same `recast_iter` f32 views — 226 vs 227 µs. So neither the pool nor the
  recast is what costs anything, which is what "one pool with dynamic offsets
  defeats LDS alias analysis" would have predicted.
- **B10p says it is not occupancy.** Padding the *overlapped* pool back to
  88 KB puts it in the same one-CTA-per-CU class as the 227 µs arm while
  keeping the overlap, and it stays at 359 µs. So the regression is not the
  register allocator re-targeting a higher occupancy (the obvious suspect on
  AMDGPU, where the LDS footprint sets waves-per-EU and hence the VGPR
  budget), and it is not the smaller footprint in any other guise either.
- **What is left is the overlap itself**: the in-loop KV/s stores provably
  write the bytes the pre-loop Q fragment loads read, and the most likely cost
  is the compiler losing the freedom to schedule LDS accesses across the tile
  loop once that dependence exists. Not proven — an ISA dump would prove it,
  and `FLYDSL_DEBUG_DUMP_ASM=1` did not produce one on this node.

**The strategic conclusion matters more than the mechanism.** Two independent
routes to a smaller LDS footprint (`q_global`, `lds_pool`) both cost far more
than the footprint was worth, and **the occupancy win they were buying has
never actually been measured on this kernel** — the 12-21 % figure in "The
target" is the *shipped* kernel's. Before a third attempt, measure the prize:
find any pair of arms that differ only in CTAs-per-CU and see whether the time
moves at all.

## Step 8: 128-bit KV staging (B11) — and a closed item that had gone stale

**227 → 195 µs, the new best (1.33x), from one change**: `kv_vec=8`. What the
staging loop did at `kv_vec=1`, per tile, per thread:

- **one** bf16 per KV row, fetched by `load_bf16` = dword load, shift, mask,
  bitcast, convert to f32 — and then converted straight back to bf16 to be
  stored. The f32 round trip was pure waste for a copy.
- `BLOCK_K` = 32 of those, plus **32 `kv_indices` loads, with all 512 threads
  loading the same index**.

At `kv_vec=8` a thread owns 8 contiguous columns, `TH_PER_ROW = 64` threads
cover a row, and the tile is staged in `KV_ROUNDS = 4` rounds: one
`dword4` load and one index load per round. Global loads and index loads both
drop 8x, and the conversions disappear. The idiom is copied from
`fused_compress_attn_hca.py::_load_bf16_vec_to_f32`.

**The win landed in the predicted rung**: staging 65 → **28 µs** at hw=8
(`--abl --hw=8 --reuse --kv1 --qres --vec8`), and the total moved by the same
amount. The ladder is now floor 29, staging 28, QK 68, softmax+PV 77.

**It also revived `h_per_wg=16`, which is now the best arm** (195 vs 200 at
hw=8). B8's hw=16 spilled at 330 µs; the scalar staging path's ~200 live VALU
temporaries per tile were evidently part of that pressure, because nothing
else about the register budget changed.

| arm | hw=8 | hw=16 |
|---|---:|---:|
| B8 (`q_resident`) | 227 µs | 330 µs (spill) |
| B11 (`+ kv_vec=8`) | 200 µs | **195 µs** |

**Measured null: `q_vec`** — the same vectorisation on the Q staging loop is
neutral at hw=8 (200.0 vs 200.1) and a **regression at hw=16** (234 vs 195).
Q is staged once per workgroup so the load saving is small, while the store
side gets worse: a thread writing 8 contiguous bf16 puts 64 lanes on a 16-byte
stride, which bank-conflicts, where one-column-per-thread is conflict-free.
Left behind `q_vec=False`; re-try only together with a 128-bit LDS store.

**Why this item was sitting closed, and the rule it gives.** This doc said, in
bold, *"do not spend a day on async 128-bit copies"* — because staging measured
**4 %** of B2's 4,103 µs. That was true and it stopped being true: the same
staging was **28 %** of B8's 227 µs, because everything around it got 18x
faster while it did not. **A closed item re-opens when the total moves.** When
an attribution says "X is only N %", record the total it was N % of, and
re-check X after any change that moves the total by more than ~2x.

## Step 9: the profile, and the bank conflict that was a quarter of the kernel

**195 → 145 µs from `kv_pad=8`**, one integer added to an LDS row stride. This
is the step that took the kernel to parity, and it is also the cheapest change
in the whole project — which is the lesson, because the three steps before it
were structural redesigns aimed at the same unexplained time.

**What the profile said.** `rocprofv3 --pmc OccupancyPercent MfmaUtil
LDSBankConflict MemUnitStalled`, run over a process that builds *both* kernels
so the shipped one is a same-run control:

| counter | B11 hw=16 (before) | B12 hw=16 (after) | shipped |
|---|---:|---:|---:|
| LDSBankConflict | **26.79 %** | 11.28 % | 9.41 % |
| MfmaUtil | 5.85 % | 8.18 % | 11.89 % |
| OccupancyPercent | 15.17 % | 14.78 % | 17.96 % |
| MemUnitStalled | 0.04 % | 0.07 % | 0.19 % |

`MemUnitStalled` ≈ 0 rules out global memory, occupancy barely moved (so the
two failed LDS-footprint experiments were aimed at a non-problem), and the
conflict number is the one that is 3x the control's.

**The mechanism.** The KV tile's LDS row stride was `D` = 512 bf16 = **1024 B**,
a multiple of the 128 B that the 32 LDS banks span, so *every row starts in
bank 0*. A QK B fragment has its 16 lanes reading 16 different rows, so every
`ds_read_b128` was a 16-way conflict — and each of the 8 waves reads the whole
32 KB tile, which is where the 41 % QK rung was going. 8 bf16 of padding puts
consecutive rows 4 banks apart, at a cost of 512 B of LDS.

**The counter moved and the time moved**, which is what makes this an
explanation rather than a coincidence: conflicts 26.8 → 11.3 %, MfmaUtil 5.9 →
8.2 %, time 196 → 145.

**Two limits found while sweeping it:**

- **hw=16 is now hard against the 160 KB LDS wall.** `kv_pad=8` fits;
  `kv_pad=16` is `hipErrorIllegalState` at launch. At hw=8, where Q is 64 KB
  smaller, `kv_pad=16` beats 8 (155.7 vs 160.5 µs) — so there is residual
  conflict worth ~3 % that hw=16 cannot buy with padding. The zero-LDS way to
  get it is an **XOR swizzle** on the KV layout, which `flydsl_mfma_probe.py`
  already showed the atom accepts ("no swizzle is needed for correctness" —
  it was parked as "a bank-conflict optimisation for later"; later is now).
- **Padding the score tile is a measured regression** (`s_pad`, left at 0).
  Its row stride is `BLOCK_K` = 32 fp32 = exactly 128 B, so the fast-softmax
  row scan really does put all 64 lanes of a wave on one bank — but the volume
  is ~3.5k element reads per tile against QK's ~131k, and the flat-pair-index
  decomposition the padding forces costs more than the conflict: hw=8 goes
  155.7 → 164.3 µs, and hw=16 without `kv_pad` is 233 µs. **A real conflict on
  a small volume is not worth fixing.**

## Step 10: the post-parity attribution, and three nulls that map the ceiling

**Re-measured at the winning config** (`--abl --hw=16 --reuse --kv1 --qres
--vec8 --pad8`, total 147 µs) — the `kv_pad` fix moved QK from 80 to 26 µs, so
every earlier ranking in this doc is obsolete:

| rung | µs | share | what it is |
|---|---:|---:|---|
| launch + epilogue floor | 28.3 | 19 % | the 88 MB of fp32 partials; the shipped kernel pays it too |
| tile loop + staging | 19.6 | 13 % | after `kv_vec`; this is the double-buffering prize |
| QK MFMA | 26.5 | 18 % | after `kv_pad`; residual conflict worth ~3 % |
| **fast softmax + PV** | **72.7** | **49 %** | the only large block left |

**Three nulls, run after parity, that say the micro-optimisations are done:**

- **`p_pad` — regression, 144.7 → 175.6 µs.** Padding the P tile the way
  `kv_pad` padded the KV tile. Pad 8 and pad 16 cost the *same*, which is the
  tell: it is not the padding amount, it is that a non-compact P tile drops
  the A-fragment copy off its wide vectorised path.
- **`s_pad` — regression** (see "Step 9"), same family.
- **Hoisting the `alpha` loads out of the `t` loop in `_pv_atoms`** — 145.5 vs
  145.3, i.e. nothing. The row a register holds depends only on `(mt, i)`, so
  this cut a 4x redundancy in the source, and LLVM had already CSE'd it. Kept
  anyway (it is strictly less source-level work); do not expect it to pay.

**The rule the three padding results give.** `kv_pad` bought 50 µs and the
other two cost 30. The difference is not the counter — all three targeted a
real, measurable bank conflict — it is what the tile was before: **padding
pays on a tile that was already read through a strided layout (KV, via
`make_layout((MMA_N, MMA_K), (D, 1))`), and costs on a compact one (P, s),
where the vectorised copy path is what you give up.** Check how a tile is
*read* before padding it.

**Double buffering is now the weakest remaining direction, and at hw=16 it is
blocked.** Its prize is bounded by the staging rung, **19.6 µs**, and a second
KV buffer costs 32 KB of LDS that the winning geometry does not have — hw=16
is at ~159.5 KB of 160 KB, which is why `kv_pad=16` already fails to launch.
It is only buildable at hw=8, which starts 15 µs behind.

**What is left, honestly.** The floor (19 %) is structural and shared with the
kernel we would replace. Staging and QK together are 31 % with maybe 5 µs of
slack between them. **Half the kernel is the softmax+PV block, and the only
structural attack on it is keeping P in registers** — cross-lane (DPP /
permlane) row reductions instead of the LDS round trip — which is blocked on
the QK-C-fragment to PV-A-fragment relayout (they are transposes of each
other). That is the one remaining item with a double-digit prize, and it is
also the hardest thing in the whole plan. Read `fmha_gfx950/op_softmax.py`
before attempting it.

## Step 11: shape verification — where the kernel actually wins

Every number above this section came from one point: bs=12, the production
length distribution, `KV_SPLITS=4`. `/shared_nfs/kk/verify_shapes.py` sweeps
the axes that can falsify it, scoring against `ref_torch()` and timing the
shipped kernel in the same process:

```bash
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/verify_shapes.py     # ~12 s
```

**Correctness holds at every shape** — ours tracks the shipped kernel to
within 0.01e-03 everywhere (1.14e-03 to 1.39e-03 against torch, both kernels
moving together). No shape breaks it.

**Speed, with each kernel given its own best `KV_SPLITS`** (the production
side already picks this per stream, so it is a fair comparison):

| shape | ours | shipped | ratio |
|---|---:|---:|---:|
| bs=1 | 38.4 µs (spl 16) | 46.2 (spl 8) | **0.83x** |
| bs=4 | 61.0 (spl 8) | 62.7 (spl 4) | **0.97x** |
| bs=8 | 88.0 | 104.6 | **0.84x** |
| bs=12 (production p50) | 142.8 | 141.0 | 1.01x |
| bs=24 | 235.1 | 265.0 | **0.89x** |
| bs=32 | 332.6 | 350.9 | **0.95x** |
| SWA-like, kv≈128, spl 2 | 40.2 | 46.7 | **0.86x** |
| CSA-like, kv≈1118, spl 2 | 133.9 | 154.2 | **0.87x** |
| straggler 0.04 / 0.18 / 0.44 | 142.8 / 143.0 / 149.1 | 145.4 / 141.4 / 149.7 | 0.98 / 1.01 / 1.00x |

**Faster at seven of eight shapes, by 3-17 %, and level at the eighth.** Two
findings got it there, both from this sweep rather than from the bs=12 point
everything else was tuned on.

**A dynamic `if` around the kernel body cost 53 µs at short KV.** The kernel
guarded its whole body with `if tile_start < num_tiles:` to skip empty splits.
Removing it (see the epilogue note below) took the SWA-like case from 92.9 to
**40.2 µs**, turning a 1.99x loss into a 0.86x win, while barely moving the
long-KV cases. The AST rewriter runs a dynamic `if` branch as a *separate
function*, so this was a fixed per-CTA cost — invisible when it amortises over
19 tiles, dominant over 2. **Add it to the authoring traps: never wrap the
whole body in a dynamic `if`.**

That change was made for correctness, not speed, and it is required for
integration: the reduce kernel recomputes `act_num_segments` from each
**token's** kv_len, while this kernel plans from the **request's** kv_len_max,
so near a length boundary the reduce reads a split this kernel never wrote —
and the partial buffers are reused across layers, so that read is stale
garbage. Every split now always writes; an empty one writes m=-inf, l=0,
acc=0, which merges to nothing.

**Low batch size was CTA starvation, and the split knob fixes it.** At bs=1,
`KV_SPLITS=4` gives 32 CTAs on 256 CUs and a 1.33x loss; at 16 it is 128 CTAs
and a 0.83x win. Both kernels improve, ours more.

**Measured null: the same knob does not fix bs=12.** The CTA-quantisation
story predicted it would — at hw=16, bs=12, `KV_SPLITS=8` gives exactly 768
CTAs = 3.0 per CU against 384 = 1.5 at `KV_SPLITS=4` — but ours goes 142.8 →
146.5 µs (while the shipped kernel degrades much further, 141.0 → 168.8).
Doubling the splits doubles the 88 MB of fp32 partials, and that costs more
than the half-empty second round it recovers. **bs=12 stays at parity, and the
quantisation hypothesis is only half right.**

## The deployment gates are blocked on plumbing, not on the kernel

`gsm8k_segplan.sh` and `matched_bs.py` both read **server logs**, so both need
the kernel wired into the decode path first. The blocker is the index layout
(see "Shapes and metadata"): production hands down a **per-token** `kv_indptr`
where every draft token owns its own copy of the request's prefix, and this
kernel wants one request-shared slice plus a per-token length.

That is derivable on the backend side without new kernel data: the 7 tokens'
slices are prefixes of the longest, so the request-shared slice *is* the last
token's slice, and `tok_len[i] = kv_indptr[t+1] - kv_indptr[t]`. What has to
be written is the backend glue in `deepseek_v4_backend_hip_radix.py` plus a
per-stream dispatch that keeps SWA on Triton.

## Step 12: the integration works, and it corrected the baseline

**The hook is written and the whole production path runs on it.**
`SGLANG_MLA_FLYDSL=1` swaps stage-1; everything else — the reduce, the
attn_sink fold, the partial buffers — is untouched. Files:
`unified_kv_kernels/flydsl_decode_hook.py` (new, glue only) and a branch plus
a reduce constexpr in `paged_decode.py`. Verify without a server:

```bash
PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/verify_integration.py 12
```

**Three contract mismatches had to be bridged, and all three are now free:**

- **Index layout.** The decode path hands down a per-TOKEN `kv_indptr`; the
  kernel wants a request-shared slice plus per-token lengths. Doing that
  conversion on the host cost **36 µs per call** (7 small tensor ops), so the
  kernel now derives both *in-kernel* from `kv_indptr` (`tok_indptr=True`):
  the request slice is its last token's, and a length is the difference of two
  neighbouring offsets. The hook passes `kv_indptr` for both arguments and
  does no tensor work at all.
- **`BLOCK_K`.** Ours is 32, the shipped bf16 path uses 16, and the reduce
  derives `act_num_segments` from the same constexpr — so the reduce is
  launched with 32 when the hook is on. Free: the shared tail measures 28.3 µs
  at `BLOCK_K=32, FOLD_Q_LEN=7` against 29.8 at `16, 0`.
- **Segmentation.** Ours is per-request (`kv_len_max`), the reduce's is
  per-token, and they disagree near a length boundary — with reused partial
  buffers that is a stale read, not just a lost contribution. Fixed on both
  sides: the kernel always writes every split, and the reduce takes a
  `FOLD_Q_LEN` constexpr and takes the max over the verify window, computed
  from `kv_indptr` in-kernel (fabricating a rounded indptr on the host was the
  other 36 µs).

**The prefix assumption is verified, not assumed.** `SGLANG_MLA_FLYDSL_CHECK=1`
asserts that every draft token's `kv_indices` slice really is a prefix of its
request's last token's slice — the thing that lets one tile load serve the
whole window. It passes on the real layout. It costs ~`R*Q_LEN` device syncs,
so it is debug-only (5.1 ms per call when on).

**A debug cycle worth recording: the masking paths also read `len_buf`.**
Switching the metadata to `tok_indptr` gave relL2 **0.33** — a *masking* error,
not an indexing one, because `_fast_softmax_pv` and the p-value pass re-read
`len_buf[tok0+q]` as a length and it now held a cumulative offset. The tell was
the magnitude: 0.33 is the same order as the `log2e` bug stage A had, i.e. "the
softmax is wrong", not "the data is garbage". Both sites now go through
`dyn_row_len`, a select chain over the constexpr `row_len` list.

**The integration's own worst bug was a host-side one.** Calling the FlyDSL
launch closure directly (`fn(...)`) dispatches through the tracer on every
call; the harness goes through `aiter.ops.flydsl.kernels.tensor_shim.
_run_compiled`, which caches the `CompiledFunction` on the closure. The
difference is a **batch-size-independent ~150 µs floor**, and the tell was
exactly that: the FlyDSL column read 213, 215, 211, 212 µs at bs 1, 4, 8, 12
while Triton scaled 52 -> 141. A flat curve against a scaling one is a fixed
host cost, not a kernel property. With `_run_compiled` the same shapes read
66, 73, 80, 130.

## B0 result: the MFMA atom contract

`/shared_nfs/kk/flydsl_mfma_probe.py` — one wave, one atom, two LDS tiles, one
`fx.gemm`, compared to `torch.matmul`. It passed on the first run and pinned
everything stage B needed:

- Both `MFMA(32, 32, 16, BFloat16)` and `MFMA(16, 16, 32, BFloat16)` work, at
  relL2 **1.95e-08** / 2.79e-08, with full accumulator coverage.
- **Operand convention:** hold *both* operands `(rows, K)` row-major in LDS
  (`fx.make_ordered_layout((rows, K), (1, 0))`, or an explicit
  `fx.make_layout((rows, K), (row_stride, 1))` to slice a k-chunk out of a
  wider tile). The atom then computes `A @ B.T` with A `[M, K]` and B `[N, K]`.
  This is the same convention `gemm_a16w16_gfx950.py` feeds its A fragment and
  its transposed B fragment.
- **No swizzle is needed for correctness**, and a plain unswizzled ordered
  layout works with `UniversalCopy128b`; 64b also works. Swizzles are a
  bank-conflict optimisation for later.
- **Do not hand-derive the accumulator mapping.** `thr_mma.partition_C()`
  applied to two coordinate views hands back the (row, col) of every register:
  ```python
  cRow = thr_mma.partition_C(fx.make_view(0, fx.make_layout((M, N), (1, 0))))
  cCol = thr_mma.partition_C(fx.make_view(0, fx.make_layout((M, N), (0, 1))))
  ```
  For the record, `--dump-map` says 32x32x16 is `col = lane % 32`,
  `row = 4*(lane//32) + i%4 + 8*(i//4)`, 16 regs/lane; 16x16x32 has 4.
- `make_mma_atom` / `make_tiled_mma` / `make_copy_atom` can all be built
  **inside** the kernel body; they do not have to be host-side kernel arguments
  the way `gemm_a16w16_gfx950.py` passes them.
- A single-wave tiled MMA is `fx.make_tiled_mma(atom, fx.make_layout((1,1,1),
  (1,1,0)), fx.make_tile(None, None, fx.make_layout((K//4, 4), (1, K//4))))`,
  and multi-wave kernels slice it per wave with `thr_slice(tid % 64)`.

## The target, in numbers

| | value | source |
|---|---:|---|
| `_paged_decode_split_kernel`, production | 130.7 µs/call, 7.97 ms/step | trace, rank 7 bs=12 |
| B200 `flash_fwd_splitkv_mla_fp8` | 40.2 µs/call | `exchange/FINDINGS.md` |
| gap | **3.25x per call**, `attn` 13.26 vs 8.20 ms/step | |
| occupancy | **12-21 %** | `rocprofv3 OccupancyPercent` |
| MfmaUtil | **3.4 % fp8 path / 11.8 % bf16 path** | same |
| HBM achieved | **~1 % of 8 TB/s** | bytes/time, bf16 576+64 elems/token |
| dispatch floor, in a graph | **1.6 µs/node**, flat to 3,000 nodes | `/tmp/floor2.py` |

**The kernel is latency/occupancy-bound, not bandwidth- and not
balance-bound.** Two things follow, and both are already settled: don't
re-attack the KV traffic, and don't re-attack load balance (see "Closed").

## The design the measurements ask for

1. **Accumulator in AGPRs.** The shipped kernel profiles as `VGPR_Count=128,
   Accum_VGPR_Count=0, LDS_Block_Size=0`. gfx950 splits a 512-register/lane
   file between VGPR and AGPR, and MFMA can accumulate straight into AGPR. We
   use none of it, so `acc[M, 512]` fp32 sits in VGPRs and caps `M`:

   | M | q tile bf16 | acc fp32 | CTA total | vs 256 KB budget |
   |---:|---:|---:|---:|---|
   | 64 | 64 KB | 128 KB | ~196 KB | fits (what ships) |
   | 128 | 128 KB | 256 KB | ~392 KB | spills → measured +28 % |

2. **KV tile staged in LDS**, shared by the whole workgroup, double-buffered.
   Triton never allocated LDS at the shapes that are fast, and at the first
   shape that does (`BLOCK_K=64`) it was **7-11x slower**.

3. **M = 448** — fold the 7 MTP draft tokens (`gamma 6` → verify window 7) into
   the M dimension alongside 64 heads. The prototype proves this is correct and
   cheap to express; it needs (1) and (2) to pay.
   **SUPERSEDED 2026-09-18:** 448 does not fit any register budget on gfx950;
   the mechanism (fold Q tokens into M) survives, the number is now **112**.
   See "Register arithmetic".

4. **QK computed once.** Splitting the output `D` across CTAs to shrink `acc`
   forces every chunk-owner to redo the full-D QK contraction: measured
   **+148 % at best** (M=256, D_CHUNK=128). Inside one workgroup, scores can
   stay live in LDS/registers while PV walks D in chunks — that is the
   structure Triton cannot express and FlyDSL can.

## Register arithmetic — why M=448 is out and 112 is in

Measured/queried on this node: `gfx950`, 256 CUs, wave64, **LDS 160 KB per
workgroup** (in-tree constant `LDS_BYTES_GFX950` in
`aiter/ops/flydsl/kernels/fmha_gfx950/pipeline.py:28`), bf16 MFMA atoms
`mfma_f32_32x32x16_bf16` and `mfma_f32_16x16x32_bf16` both present in
`flydsl.expr.rocdl`.

The accumulator cost per lane is **invariant** to how you split M or D across
the waves of a workgroup — only the workgroup's lane count matters:

```
acc bytes/lane = M * Dv * 4 / (64 * NW)     Dv = 512
NW = 8 (512 threads):  = M * 4 bytes/lane   = M registers/lane
```

gfx950 gives a wave at most **512 registers/lane** (VGPR+AGPR), and that is
1 wave/SIMD. An 8-wave workgroup is 2 waves/SIMD, so its ceiling is
**256 regs/lane**, of which the accumulator should take at most ~3/4 to leave
operand and scratch headroom:

| M | acc regs/lane | verdict |
|---:|---:|---|
| 112 (7 tok x 16 heads) | 112 | fits, with room for Q/K/P — **the target** |
| 224 (7 tok x 32 heads) | 224 | saturates the budget; try only after 112 works |
| 448 | 448 | **1.75x the entire per-lane budget — impossible** |

Going to `NW=16` (1024 threads, the workgroup max) halves regs/lane to `M/2`
but puts 4 waves/SIMD, so the per-SIMD register file binds first — it makes
things worse, not better. The per-CU file is the real wall: 512 regs/lane x 64
lanes x 4 B x 4 SIMDs = 512 KB, i.e. 256 regs/lane for a resident 512-lane CTA.

This also reconciles the earlier measurement: Triton's M=128 attempt "spilled at
~392 KB" even though 392 KB / 512 lanes = 196 regs/lane is under 256 — because
Triton put both acc and the Q tile in **arch VGPRs** (profiled
`VGPR_Count=128, Accum_VGPR_Count=0`), where the cap is 256/lane and it had
allocated 128. **AGPRs are the lever, and Q belongs in LDS, not registers**
(Q[112, 576] bf16 = 126 KB — LDS-sized, not register-sized).

## FlyDSL build plan (staged, per the authoring playbook)

The upstream playbook is explicit that blind iteration on MFMA fragment layout
is the documented dead-end, and that observability + a true reference + a micro
harness must exist first. The harness and reference already exist
(`mla_qfold_proto.py`: `merge()` + relL2 against the shipped kernel), so the
stages are:

- **Stage A — plumbing, no MFMA. DONE 2026-09-18**, in
  `/shared_nfs/kk/flydsl_mla_decode.py`. Paged gather of `kv_indices`, KV tile
  to LDS shared by the workgroup, scores by plain FMA, online softmax with
  per-draft-token masking, PV, `(m, l, acc)` partials in the harness's layout.
  One workgroup owns (request, head, kv-split) and all 7 draft tokens, 512
  threads, thread `t` owning output column `d = t`; the QK contraction is split
  4 ways across threads (112 (token, kv) pairs x 4) and recombined in LDS.
  Repro:
  ```bash
  PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python HIP_VISIBLE_DEVICES=0 \
    python3 /shared_nfs/kk/flydsl_mla_decode.py 4
  ```
- **Stage B — MFMA. DONE 2026-09-18**, same file, behind `mfma_qk` / `mfma_pv`
  build flags so stage A survives as the control. Both inner products use the
  atom form (`fx.make_mma_atom` + `fx.gemm`); no fragments were hand-packed,
  and none needed to be. See "Stage B results".
- **Stage C — the actual wins.** Re-ordered by what stage B learned: attribute
  the 23x first, then fold heads into M (7 → 112), then the AGPR accumulator
  and double-buffered LDS. See "CONTINUE HERE".

Note that stage A has a close in-tree precedent: `fused_compress_attn_hca.py`
is a FlyDSL gfx950 kernel at exactly D=512/RD=64 doing multi-wave K-split
online softmax with **no MFMA at all**, using `ptr_buf_tensor` +
`fx.add_offset(...).load()` for gather and an `fx.struct`/`fx.SharedAllocator`
LDS block. Copy its idioms rather than inventing them.

## FlyDSL toolchain status on this node (verified 2026-09-18)

- `flydsl` imports from `/opt/venv/lib/python3.10/site-packages/flydsl`.
- Working gate command (JIT-compiles and runs real FlyDSL gfx950 kernels,
  ~12 s): `cd /sgl-workspace/aiter && HIP_VISIBLE_DEVICES=0 python3
  op_tests/test_flydsl_compress_attn.py -s hca_main -b 4 -m 3 --modes decode`.
  Its `legacy(NW=1) cache_scale(bit-exact)` failures are **pre-existing** and
  unrelated; the `ksplit` cases pass.
- Dead ends for a gate: `test_flydsl_mla_reduce.py` skips ("unsupported on
  gfx950"), `test_flydsl_fmha.py` skips ("gfx1201/RDNA4 only").
- There is **no** FlyDSL MLA decode stage-1 kernel in aiter today. What exists:
  `mla_reduce.py` (stage-2 combine, gfx950-unsupported),
  `fmha_gfx950/{pipeline,op_gemm,op_lds,op_softmax,op_combine}.py` (fp8
  dual-wave flash attention — the MFMA/LDS structural template), and
  `fused_compress_attn*.py` (the D=512 online-softmax template).

## Where to start in the corpus

**Upstream FlyDSL skills** (17 of them, incl. `flydsl-kernel-authoring`,
`flydsl-tile-programming`, `lds-optimization` with a paged-attention section,
`debug-flydsl-kernel`, `oob-detection`) are richer than the local corpus for
authoring. Checked out sparsely on this node at
`/shared_nfs/kk/flydsl-repo/.claude/skills/` from
<https://github.com/ROCm/FlyDSL>. Index the headings before reading —
`flydsl-kernel-authoring/SKILL.md` alone is 1,008 lines.


`flydsl/SOURCE_MAP.md` first. Then, in order of use here:

- `flydsl/00-foundations/03_mfma_layout.md` — MFMA operand/accumulator layouts.
- `flydsl/00-foundations/02_memory_layout.md` — LDS staging.
- `flydsl/01-playbooks/FLYDSL_KERNEL_AUTHORING.md`, `..._OPT_PLAYBOOK.md`.
- `flydsl/01-playbooks/FLYDSL_KERNEL_DEBUG_TOOLKIT.md` when it misbehaves.
- MegaMoE's shipped FlyDSL kernels are the in-tree example of this style.

Do not read the corpus in bulk — 61 files, ~243k tokens.

## Shapes and metadata (from the model config and the launcher)

`num_hidden_layers` 61, `head_dim` (latent) 512, `qk_rope_head_dim` 64,
`num_attention_heads` 128, `num_key_value_heads` 1. Verify window 7. Per rank
at c128: `running-req` p50 12 → **N ≈ 84-96 tokens**, `BLOCK_H=64` →
2 head tiles, `KV_SPLITS=4` (env-pinned for HCA) → 672 CTAs, which matches the
profiled `Grid_Size/Workgroup_Size`.

Production kv_len, from the `SGLANG_MLA_KVLEN_STATS` probe arm
(`megamoe-eplb-c128-kvlenprobe/server.log`):

```
kvlen mean/p50/p99/max/min: 1192/1148/1492/1901/128   straggler 0.373
```

Three decode streams share this kernel, not just HCA: `_kv_splits_for_stream`
overrides only ratio 128 (→4); SWA (ratio 0, 128-entry window) and CSA (ratio 4,
clamped to 1152) keep `_kv_splits_heuristic`, which at c128 returns 2. So a
rewrite must work for all three.

**Index layout, and this is plumbing the new kernel needs:** `kv_indptr` is per
TOKEN and every token owns its own slice of `kv_indices`, so the 7 draft tokens
of a request cannot share a tile load as the data is handed over today.
`runtime.decode_qo_indptr` is deliberately an `arange` ("one q token per
sequence") and its own comment points at `cu_seqlens_q` as the per-request
grouping that exists upstream but is not passed down. `mla_qfold_proto.py`
builds both layouts so kernels can be compared on equivalent data.

## Closed — do not reopen without new evidence

- **Load balance.** Production straggler is 0.373; at that shape split-K=4 is
  within **5 %** of the perfect-balance floor (149.6 vs 142.1 µs). A full port
  of #39172's length-aware plan is implemented, correct (gsm8k 0.939 vs 0.937)
  and **−0.02 ms on an arm**. The crossover where it starts winning is
  straggler ≈ **0.55-0.6**; #39172's own agentic case maps to ~0.53 after MLA's
  128x compression. Kept env-gated, default off (`SGLANG_MLA_SEG_PLAN`).
- **#39503's exact-CU-share cap.** Does not bind here: the plan's binding term
  is the total budget (13 tiles) not the per-token cap (8).
- **KV traffic / q-folding alone.** Removing 6.98x of duplicate KV reads bought
  **3 %** — 117 MB of working set fits MI355X's 256 MB Infinity Cache.
- **`block_k`.** Production's Triton decode is the **bf16** path
  (`runtime.decode` passes no `kv_scales`), where `block_k` is already 16. The
  32 only ever applied to the quantised path. An arm measured **+0.36 ms**.
- **fp8 two-pool + aiter asm decode** (`SGLANG_DSV4_UNIFIED_KV_FP8`, PRs
  #37413/#38901). #38901 applied to this tree fixes the DSpark dtype assert,
  and the next wall is hard: `NotImplementedError: HiCache offload is not
  supported with SGLANG_DSV4_UNIFIED_KV_FP8=1`. Our recipe runs HiCache
  (`KV_OFFLOADING=dram`), so this path needs HiCache × two-pool integration
  first. #37413's own tables also show fp8 ITL **worse** at every concurrency —
  it is a KV-capacity feature (1.5x), not an MLA speedup.

## The correctness gate is "vs torch", not "vs shipped"

Stage A forced this out. Measured at bs=4 on the merged output:

- stage A vs independent fp32 torch: **4.102e-07**
- stage A vs the shipped Triton kernel: 1.162e-03
- **shipped Triton vs the same torch reference: 1.162e-03**

So the entire discrepancy is the *shipped* kernel's, not ours: Triton's
`tl.dot(p.to(kv.dtype), kv)` rounds P to **bf16** before PV, and stage A keeps
it fp32. Consequences for the rest of this work:

1. Score a new kernel against `ref_torch()` in `flydsl_mla_decode.py`, which is
   built from neither kernel. Scoring only against the shipped kernel is the
   self-referential-oracle trap the FlyDSL authoring playbook warns about.
2. `relL2 ~ 1.2e-3 vs shipped` is the *expected, correct* reading for an
   accurate kernel, not a failure. The original "≤ ~1e-3 vs shipped" pass
   criterion would have rejected a kernel that is 3 orders of magnitude more
   accurate than the thing it replaces.
3. If stage B rounds P to bf16 to feed MFMA (it will), expect it to move toward
   the shipped kernel's 1.2e-3 — that is a dtype choice, so gate it against
   torch and decide deliberately. **Confirmed 2026-09-18:** B2 measures
   1.184e-03 against torch, and the choice is recorded in "Stage B results".

Two other numerics notes found while writing stage A:

- `mla_qfold_proto.py` has a latent masking bug that stage A does not: it sets
  invalid scores to `neg_large` and computes `exp2(score - m_new)` unguarded, so
  a row whose tile is *entirely* past its own `row_len` gets
  `exp2(neg_large - neg_large) = 1` for all 16 lanes and silently accumulates
  garbage into `l` and `acc`. Reachable because the loop bound comes from
  `kv_len_max` while rows are up to 6 shorter. Stage A masks `p` to 0 instead.
- The reference kernels call `exp2` on the **raw scaled scores** (no `log2e`
  factor), so the softmax is base-2 throughout, including in `merge()`. Adding
  the usual `* log2e` gives relL2 0.42 — that was stage A's only real bug.

## FlyDSL authoring traps stage B paid for

- **A host-level `if` inside a kernel is not a host-level `if`.** The AST
  rewriter turns every `if` in a kernel body into an `scf.if` and runs the
  branch as a *separate function*, so names assigned inside it never reach the
  enclosing scope — `NameError: name 'thr_mma' is not defined` at trace time.
  Build-time switches must be `if const_expr(flag):`. The same applies to
  values that have to escape a branch: use the
  `x = a() if const_expr(flag) else b()` ternary.
- **`view[r, c] = v` inside a *dynamic* `if` is rejected**: the rewriter reads
  the subscript assignment as an assignment to `view` and raises
  "Variable(s) ['sScore'] initialized as None before a dynamic if/else". Use
  `fx.ptr_store(v, base + offset)`, which is a call, not an assignment.
- **The bug that cost this session's debug cycle was an LDS *stride*, not
  anything MFMA.** The per-wave score tile's row stride was widened from
  `MMA_N` to `BLOCK_K` while the reduction that reads those tiles back still
  used `MMA_M * MMA_N`. The two agree exactly when `BLOCK_K == MMA_N`, so
  BLOCK_K 16 passed and BLOCK_K 32 returned garbage — a classic
  "works at the tile size you developed at". Derive both the write stride and
  the read stride from the same expression.
- **`fx.union` exists and `SharedAllocator` cannot use it.** A Sum-policy
  composite raises `NotImplementedError: <T> does not support
  __peek_from_ptr__` from `allocate(...).peek()`, so LDS aliasing has to be
  done with `fx.recast_iter(dtype, ptr)` on a normal struct field instead —
  that is how `alias_p` puts the bf16 `p` tile on top of the f32 `s` buffer.
- **You do not need a transposed copy atom to read a transposed operand.**
  `thr_mma.partition_B` on the two coordinate views gives every register's
  `(row, col)`, so a fragment can be filled with plain `ptr_load`s at whatever
  stride the data actually has. Slower per fragment than a 128-bit copy, so
  hoist the load out of the inner loop — but nothing about the atom's tiling
  has to be guessed, which is the part that burns days.
- **Never wrap the kernel body in a dynamic `if`.** The rewriter runs the
  branch as a separate function, which is a fixed per-CTA cost: guarding the
  whole body with `if tile_start < num_tiles:` was invisible at kv≈1200 (19
  tiles) and **53 µs, more than the kernel itself, at kv≈128** (2 tiles).
- **Diff two of your own kernels before reaching for a reference.** Getting a
  torch score reference for a paged, split, masked tile is real work; dumping
  scores from the FMA and MFMA paths at the *same* BLOCK_K and diffing them per
  (draft token, kv) column took one build flag. `build_kernel(dbg_scores=True)`
  overwrites `acc_out` with the last tile's scaled scores for exactly this;
  `flydsl_mla_decode.py --dbg` prints the per-column max difference. It also
  immediately falsified the "only the second n-tile is wrong" hypothesis, which
  is what redirected the search to the stride.
- **A failed bisection can be a broken experiment.** Clamping `tok_len` to make
  a tile fully masked broke the *reference* too, because the harness carries two
  index layouts (`req_indptr`/`tok_len` for this kernel, `tok_indptr`/`tok_idx`
  for the shipped one) and only one was clamped. The tell was the control arms
  failing as well. `--clamp-kv=N` is still in `main()`; it is only valid if both
  layouts are rebuilt.

## Method traps this session paid for

- **FlyDSL's JIT cache key covers the kernel source and its closure scalars, but
  NOT module globals.** `_jit_function_cache_key` hashes `_get_func_source`, the
  dependency sources and `_collect_closure_scalar_vals`, so a build flag passed
  as a `build_kernel` argument is part of the key and an ablation flag read from
  a module-level global is **not**. Session 6 ran four control arms — including
  one that masked *every* KV entry — off a single cached binary and read four
  identical results. The tell was that control arm: masking everything must
  produce zeros, and it produced the correct answer. **Any ablation knob has to
  be an argument that reaches `build_kernel`, and any ablation set needs one arm
  whose expected result is obviously broken.** Without that arm the session
  would have concluded the interval mask was free, and shipped a kernel whose
  mask did nothing. Cache dir: `~/.flydsl/cache`.
- **Timing many kernels in one process inflates the late arms.** B7 hw=16
  measures 308 µs as the eleventh arm of the `--mfma` gate and 242-245 µs when
  two arms are built, reproducibly across runs. Read times from a
  `--only=<tags>` run; the full run is for correctness.
- **`import sglang` does not resolve to the arm's tree.** Bare import gives
  `/sgl-workspace/sglang` (933-line `paged_decode.py`); arms pin
  `/sgl-workspace/sglang-MegaMoE` (1,535 lines). Always
  `PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python`. The tell that you got it
  wrong: `relL2 = 0.000e+00` and total insensitivity to your own knobs.
- **Partials are not comparable between kernels** that segment KV differently.
  Merge with the FlashAttention rule and compare the output; `merge()` in
  `mla_qfold_proto.py` does it. Comparing partials first produced a bogus
  1.2e-01 and cost a debug cycle.
- **Check the harness against a production artefact before pre-registering.**
  Both falsified arms today came from a microbench whose configuration did not
  match production: first fp8-vs-bf16, then the length distribution.
- **Barrier wait is not cashable.** `megamoe_prepare_compact` is a WAIT
  (r = −0.98); a rank that gets faster just waits longer. Only the slowest
  rank's work shortens the step. This is why an MLA regression can also hide:
  the +16 % the plan cost the kernel showed up as −0.02 ms end to end.
- **Init `m_partial` to −inf, not 0**, or an unwritten split wins the max.
- Uncommitted work in this tree, now including the FlyDSL integration:
  `unified_kv_kernels/flydsl_decode_hook.py` (new) and `paged_decode.py`'s
  hook branch plus the reduce's `FOLD_Q_LEN` constexpr, on top of the existing
  env-gated `_FP8_BLOCK_K` and segment-plan kernels; `runtime.py`,
  `deepseek_v4_backend_hip_radix.py`, `metrics_reporter.py`, plus #38901's three
  files. Nothing is committed; do not revert blindly.
