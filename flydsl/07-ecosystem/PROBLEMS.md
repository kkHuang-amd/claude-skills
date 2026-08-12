# Problems encountered (chronological) + fixes

The full journey from "port the concept" to "confirmed +30.6%". Each item is a
real wall hit during this work; record so it is not re-hit.

## 1. First port was only partial (missed the main lever)

- Initial port added only the **SUM fill-target** + **arrive_time/queue guard**
  layered on SGLang's existing PrefillDelayer (slot/queue-ratio + all/none/mixed).
- Result: neutral to slightly negative on uniform workloads (+1.4% best on TBO
  1k/1k c1024), far from ATOM's claimed +16%.
- Root cause (found only after reading ATOM's actual source, not just the PR
  text): the biggest throughput lever is the **alignment gate**
  (`n_prefillable < dp_size → HOLD`, anti-skew) plus the tick-based must-fire
  bounds + stall give-up — none of which were in the partial port.
- Fix: full faithful rewrite of `prefill_delayer.py` as ATOM's tick FIRE/HOLD
  state machine, moved the decision to `get_new_batch_prefill` (once per tick).
- Lesson: read the real implementation, not the PR description; state clearly
  when only doing a partial port.

## 2. Workload dependence (the big time sink)

- Even the faithful port showed ~0% (or −6% from per-tick all_gather cost) on
  `--random-range-ratio 1.0` (equal-length prompts, simultaneous arrival),
  including at the ATOM-headline point TBO 1k/1k c1024.
- Root cause: uniform same-length arrival = no prefill fragmentation and no
  cross-rank token skew → alignment gate always satisfied, fill target trivially
  met → nothing to coalesce; the coalescer only adds a per-tick collective.
- Fix / lesson: **A/B with a fragmented workload** (`--random-range-ratio 0.3`,
  or 0.8 as in ATOM's README). That surfaced +30.6% (SGLang) / +35.1% (ATOM).
- This was confirmed to be workload — not a port bug — by running ATOM's OWN
  coalescer: it too is −0.9% on uniform and +35.1% on fragmented.

## 3. DSV4 MoE cuda-graph capture fails on a plain sglang launch

- Symptom: `Exception: Capture cuda graph failed: Unsupported kernel config for
  moe heuristic dispatch` (from aiter `ck_gemm_moe_2stages_codegen`), at decode
  graph capture, before serving any request. Reproduced on every branch, dp/no-dp,
  triton-moe-runner, small cuda-graph-bs.
- Root cause: **missing DSV4 tuned env vars**, not the model/hardware/change.
- Fix: launch via `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh`,
  which sets `SGLANG_USE_ROCM700A=0`, `SGLANG_USE_AITER=1`,
  `AITER_BF16_FP8_MOE_BOUND=0`, `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`,
  `SGLANG_DP_USE_GATHERV/REDUCE_SCATTER=1`, etc. With these the model boots.
- Note: `--disable-cuda-graph` / `--cuda-graph-backend-decode=disabled` did NOT
  skip decode-graph capture on this tree (init_all_cuda_graphs runs
  unconditionally), so they are not a workaround.

## 4. gsm8k score looked low (0.883)

- Root cause: DSV4 launch env sets `SGLANG_DSV4_REASONING_EFFORT=max`; the long
  reasoning CoT gets truncated at `--max-tokens 2048`, leaving no final answer →
  scored 0.
- Fix: run gsm8k with `--max-tokens 8192` → 0.931 (OFF) / 0.939 (ON).

## 5. cohere2_moe.py `@strict` import crash (known dsv4 gotcha)

- `import sglang` (and pytest collection) crash:
  `StrictDataclassDefinitionError: Class 'Cohere2MoeConfig' must be a dataclass
  before applying @strict` on huggingface_hub>=1.x (here 1.19.0, transformers
  5.2.0).
- Fix: the local no-op `strict` patch in `srt/configs/cohere2_moe.py` (kept on
  the mori-ep working tree; reapply on a clean checkout). This + `utils.cuh`'s
  `getSMVersion` shim are the two "env" files needed to boot sglang here.

## 6. Running ATOM in-place = an aiter/flydsl/triton upgrade cascade

To validate ATOM's own coalescer in the SAME environment, ATOM had to be the
coalescer version (wheel `dev335`, PR #1611). That cascaded:

1. `pip install --no-deps .` of the new ATOM → `ImportError: cannot import name
   'interleave_gate_up_rows' from aiter.ops.shuffle`. Then `moe_shuffle_weight`.
   - Backportable as tiny additive shims (both are thin torch wrappers; current
     `shuffle_weight` already had the needed kwargs) — but next came:
2. `ModuleNotFoundError: aiter.ops.triton.attention.pa_prefill_sparse` — a whole
   new Triton kernel module. Not shim-able → **the shared aiter must be upgraded**
   (it is `d9b3e0d2e` Jul-3; coalescer-era ATOM is ~Jul-16).
3. Upgraded aiter to `origin/main`. After changing the source, `import aiter`
   failed: `module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'`
   (stale prebuilt core .so). **Fix: delete `aiter/jit/module_aiter_core.so`
   (and clear `aiter/jit/`) → next import JIT-rebuilds the core (~9s).**
4. New aiter then required `flydsl >= 0.2.4` (installed 0.2.2). `pip install -e`
   of FlyDSL v0.2.4 needs an **embedded MLIR build** (`scripts/build.sh`) — long
   and failed on the first attempt (needs full LLVM/MLIR toolchain). The user
   ended up building flydsl 0.2.4 out-of-band.
5. With aiter-main + flydsl-0.2.4, ATOM booted far (weights + fused MoE OK) then
   failed compiling aiter's `gemm_a8w8_blockscale_preshuffle` **Triton** GEMM
   with the pinned `triton-custom 3.6.0`: `make_ttgir ... PassManager::run
   failed`. i.e. new-aiter triton GEMM kernel incompatible with this env's triton.
   - **Workaround (used):** env-gate that Triton path to fall through to the
     default CK path. Added to `/sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py`:
     ```python
     if (config is not None and config["libtype"] == "triton"
         and os.environ.get("AITER_DISABLE_BLOCKSCALE_TRITON", "0") != "1"):
     ```
     then launch with `AITER_DISABLE_BLOCKSCALE_TRITON=1`. Env-gated (default
     off) so it does not affect other runs. Fair for A/B (ON and OFF both use CK).
   - This edit is uncommitted in the aiter repo; remove it if triton-custom is
     upgraded to match new aiter.

Lesson: these components (aiter, flydsl, triton-custom) are a tightly version-
pinned web shared with sglang. Upgrading one to match a newer ATOM cascades.
For a clean ATOM measurement prefer the official `rocm/atom-dev:latest` docker
(matching aiter/flydsl/triton) instead of in-place upgrades.

## 7. Lockstep requirement (design constraint, not a bug hit)

`should_allow_prefill` does a cross-DP all_gather and MUST be called every tick
on every DP rank, or ranks deadlock on the collective. It is placed at the top
of `get_new_batch_prefill` (which is called every tick unconditionally) — do not
move it after an early-return. The one wall-clock input (oldest_waiting_age_ms)
is compared locally per rank; only the OR crosses the collective, so ranks never
diverge on a timeout boundary.

## 8. Rollback notes (env safety)

- aiter rollback tag: `rollback-before-atom-test` (@ `d9b3e0d2e`); old core .so
  backed up to `/tmp/module_aiter_core.so.rollback`; flydsl 0.2.2 wheel backed up
  to `/tmp/flydsl-0.2.2-backup.tgz`. (Env was later intentionally left on the
  upgraded aiter/flydsl by the user.)
- The `AITER_DISABLE_BLOCKSCALE_TRITON` gate in `gemm_op_a8w8.py` is the only
  aiter code edit; revert with `git checkout` if unwanted.

## Shell gotcha (meta)

`pkill -9 -f "sglang..."` / `pkill -f atom...` in the SAME chained command as a
follow-up echo/launch can self-terminate the shell wrapper (its command string
matches the pattern), truncating output and skipping the follow-up. Run pkill as
its own command, then the next step separately.
