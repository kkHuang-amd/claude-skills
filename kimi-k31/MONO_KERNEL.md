# Kimi-K3.1 decode / verify mono kernel — design (OPT_TRACKER row 13)

Owner node: mi355-4. Created 2026-10-31. P0 + P1a/P1b step 1 done; history in sections 6, 8, 9.

## CONTINUE HERE (handoff 2026-11-02 14:45, mi355-4 -- NEXT SESSION: wide on / off A/B on the DFLASH config)

**Status:** wide layer launch fault-free and accurate (10.26: gsm8k200 0.985 bundled / 0.98 ROCm, buckets 16..384,
DFLASH, CUDA graphs). Code on the fork kkHuang-amd/sgl-amd-mono `k31/mono-s1` @ 2df98d8 (= the local tree, clean).
SGLang: sglang-k31-amd at origin/dev_amd b588cf7952, unmodified. The open question is speed: with the OLD code
wide was SLOWER than off (10.22, DSpark, SIM 3.84: c4 8.72 vs 7.72, c16 16.61 vs 15.70, c48 31.35 vs 31.28 ms);
the layer launch is now 2-5% faster but that has not been measured end to end. Mode: medium (high if wide is
still slower and the step needs profiling). GPUs free, no server.
**Progress (mi355-4, 2026-11-02): (a) off, (b) S1 done -> section 10.27; (c) S1 + wide running.**
**Next, in order (one server at a time; ~20 min to ready each, kill by the PID in logs/server_<tag>.pid):**
1. Three servers, same DFLASH config, SIM=1 (simulated accept 3.84, so TPOT compares at equal acceptance),
   bundled LLVM (default), each followed by `ab_bench.sh <tag>` (decode 8k/1k c4 / c16 / c48 + prefill 8k):
   (a) off: no `--enable-amd-monokernel`                       -> results/ab/dflash_off_20261102
   (b) S1 only: `--enable-amd-monokernel`                       -> results/ab/dflash_s1_20261102
   (c) S1 + wide: (b) + `SGLANG_AMD_MONO_K31_WIDE=1` (buckets 16..384) -> results/ab/dflash_wide384_20261102
   Check each server log for faults (`rg -c 'Memory access fault|Traceback'`) and that (c) logs `[k31 wide]` steps.
2. Table TPOT / ITL / prefill TTFT for (a) / (b) / (c) into section 10.27 + OPT_TRACKER row 13 (node + date).
   Decision: wide stays opt-in unless (c) <= (b) and (c) <= (a) at c4 / c16 / c48 beyond noise (~1%; rerun a case
   alone if it is an outlier, record both).
3. If (c) is slower: profile one c4 wide step (`capture_trace.sh`, `trace_breakdown.py`, `trace_layer_seq.py`):
   per-layer host work, SGLang routed experts between launches, bank copies; compare to the standard step.
   Optional: (c) again with `SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm` if ROCm's layer time differs.
**Repro (from /tmp):**
```bash
ulimit -c 0; S=/workspace/claude-skills/kimi-k31/scripts; D=/shared_nfs/RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024
DF="--speculative-algorithm DFLASH --speculative-draft-model-path $D"
bash $S/launch_server.sh dflash_off $DF                                                  # (a)
bash $S/launch_server.sh dflash_s1 --enable-amd-monokernel $DF                          # (b)
SGLANG_AMD_MONO_K31_WIDE=1 bash $S/launch_server.sh dflash_wide384 --enable-amd-monokernel $DF   # (c)
rg -m1 'ready to roll|Traceback|Memory access fault' /shared_nfs/kimi-k31/logs/server_<tag>.log   # wait
bash $S/ab_bench.sh <tag>_20261102; rg 'TPOT|TTFT' /shared_nfs/kimi-k31/results/ab/<tag>_20261102/summary.txt
kill $(cat /shared_nfs/kimi-k31/logs/server_<tag>.pid); rm -f /tmp/gpucore*
```

### Previous handoff (2026-11-02 14:35, superseded above): wide on passes server gsm8k200 on both compilers


**Status:** the wide layer launch is fault-free and accurate in the server: gsm8k200 TEMP 0, DFLASH, CUDA graphs,
buckets 16..384: bundled 0.985 / 0.985 (THREADS 16 / 48), ROCm LLVM 0.98 / 0.98; no fault; bucket 384 used
(running-req up to 46-48). Fixes: SGPR pressure cut + per-wave peer puts (10.26), and the `rocm_codegen` scope
now follows the direct caller of FlyDSL (SGLang's GEMM inside wide.backbone kept failing on ROCm clang). All code
committed, pushed to the fork (k31/mono-s1 2df98d8). GPUs free, no server.
**Next, in order:**
1. DONE 2026-11-02 14:50: DEBUG-K31 hooks removed (re-verified: t1-on PASS both compilers, 10.24 / 10.25 repros
   clean), 6 local commits on k31/mono-s1 (ad8388a..2df98d8: wide CUDA graphs, DFLASH, CM_SYS logits read, SGPR
   pressure, ROCm codegen, layer_bench); pre-split state kept as branch k31/mono-s1-backup-20261102.
   Pushed to the fork kkHuang-amd/sgl-amd-mono k31/mono-s1 (2df98d8), 2026-11-02 14:40. SGLang needs no change: the
   server runs sglang-k31-amd at origin/dev_amd b588cf7952 unmodified.
2. A/B wide on vs off on the DFLASH serving config (`ab_bench.sh`: c4 / c16 / c48 TPOT; wide was slower at c4 /
   c16 in 10.22 with the old code). Bundled LLVM unless ROCm is faster; then decide on keeping `rocm_codegen`.
3. Report both compiler bugs (10.24 / 10.25 spill-lane corruption, 10.26 waterfall rsrc clobber) with the evidence.
**Repro (8 GPUs, from /tmp):**
```bash
S=/workspace/claude-skills/kimi-k31/scripts; D=/shared_nfs/RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024
SGLANG_AMD_MONO_K31_WIDE=1 SIM=0 bash $S/launch_server.sh wide384 --enable-amd-monokernel --speculative-algorithm DFLASH \
  --speculative-draft-model-path $D      # + SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm for ROCm codegen; ~20 min cold
NUM=200 THREADS=48 TEMP=0 bash $S/gsm8k_gate.sh wide384_t48 /shared_nfs/kimi-k31/logs/server_wide384.log
```
**Code state (uncommitted, /sgl-workspace/sgl-amd-mono on k31/mono-s1 @ ad8388a):** as in the 13:30 handoff plus
`common/rocm_codegen.py` `_from_repo` (direct caller). Full diff:
`/shared_nfs/kimi-k31/results/m384_fault_20261102/26_sgl_amd_mono_worktree.diff`.

### Previous handoff (2026-11-02 13:30, superseded above): SGPR pressure cut, fault matrix clean on both compilers



**Status:** the wide layer launch no longer faults: SGPR spills cut from ~1100 to ~50 (m128) / ~140 (m384) by
per-stage kernel-arg reloads, region addresses formed at use, a non-unrolled top-16 and per-wave peer puts
(section 10.26). Fault matrix (2 compilers x nvb x t1 x step x 6 buckets, 50 runs): 0 faults; layer 2-5% faster.
All other kimi_k31 kernel benches PASS. Code uncommitted. Server not run yet with these edits. GPUs free.
**Next, in order:**
1. Server gsm8k200 (DFLASH, CUDA graphs, `SGLANG_AMD_MONO_K31_WIDE=1`, default buckets 16..384, THREADS 16 and 48),
   bundled LLVM first (default), then `SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm`. Pass: no fault, gsm8k within the
   run-to-run band of mono-off (0.975-0.98 at 200).
2. Remove `# DEBUG-K31` hooks (layer.py dstop / K31_DBG_LLVM, layer_bench K31_DBG_PAD / K31_DBG_PTRS / K31_DBG_OWN,
   rocm_codegen K31_DBG_ROCM_DUMP), split commits (K1 DFLASH spec, SGPR pressure, bench), ask before pushing.
3. Then: A/B wide vs off (wide was slower at c4 / c16 before, 10.22), report both compiler bugs (spill lane,
   waterfall clobber) with the 10.24-10.26 evidence; decide whether `rocm_codegen` is still needed.
**Repro (8 GPUs, from /tmp):**
```bash
/shared_nfs/kimi-k31/results/m384_fault_20261102/k31_spill.sh <tag> <rocm|bundled> --rows 128,384 --nvb-a 3 --write-a --nvb-m 4
OUT=/tmp/m.txt /shared_nfs/kimi-k31/results/m384_fault_20261102/k31_matrix.sh     # full matrix, ~45 min
```
**Code state (uncommitted, /sgl-workspace/sgl-amd-mono on k31/mono-s1 @ ad8388a):** previous handoff's edits plus
`common/ops.py` (sfresh64, KernArgs, sload64, uniform64), `gfx950/layer.py` (per-stage `ka`), `stages/coll.py`
(lazy `_Context`, `put_each_peer`), `stages/moe_front.py` (top-16 loop, put_each_peer), `chain.py` / `pre_attn.py`
(put_each_peer), `layer_bench.py` (wide.CFGS fallback, near-tie-aware own-logits check). Full diff:
`/shared_nfs/kimi-k31/results/m384_fault_20261102/26_sgl_amd_mono_worktree.diff`.

### Previous handoff (2026-11-02 12:30, superseded above): 10.25 root-caused: ROCm LLVM ALSO miscompiles an SGPR spill lane

**Status:** 10.25 root-caused (section 10.25): with ROCm LLVM the T5 MLP seam's bank descriptor is reloaded from
spill lanes holding (bank.lo, cw_a.hi): ROCm LLVM 23 overwrites an SGPR-spill lane, the same class of bug as the
bundled LLVM in 10.24, on a different variant. IR is correct. Neither compiler is safe for the wide layer launch
at ~1000+ SGPR spills, so the compiler swap is NOT a fix. Keep `SGLANG_AMD_MONO_K31_WIDE` and
`SGLANG_AMD_MONO_ROCM_LLVM` off on the server. Mono S1 (K1 / K2, bundled LLVM) is not affected. GPUs free, no server.
**Next, in order:**
1. Cut SGPR pressure (10.24 plan): (a) lazy region addresses in `coll.region_context` (keep `own` / `peers` / `par`,
   form `<name>_own` / peer addresses at use); (b) kernel args through a device-side pointer table loaded per stage
   (guard against CSE / hoisting back to the entry; this also ends T2 and T5 sharing one launch-long bank rsrc);
   (c) route top-16 not fully unrolled. After each: spill count (`rg -c v_writelane` on the ISA dump),
   `us: layer`, and the matrix in 2 on BOTH compilers.
2. Fault matrix, both compilers: rows 16 / 32 / 64 / 128 / 256 / 384 x (nvb 0, nvb 1/1, a3w m4) x t1 on / off x
   step 1 / 4 / 8 / 68. Pass: no fault anywhere, t1-on variants PASS numerically, `us: layer` within +-2%.
3. Then server gsm8k200 (DFLASH, CUDA graphs, wide on, buckets to 384, THREADS 16 and 48), commit (ask), report to
   FlyDSL and ROCm LLVM (IR + ISA + `rlane.py` evidence; minimal reproducer later).
**Repro of the 10.25 evidence (8 GPUs, ~35 s incl. cold compile, from /tmp):**
```bash
ulimit -c 0; B=/sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/layer_bench.py; export PYTHONPATH=$(dirname $B):$PYTHONPATH
SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm FLYDSL_RUNTIME_ENABLE_CACHE=0 K31_DBG_ROCM_DUMP=/tmp/k31_rocm_dump K31_DBG_PTRS=1 \
  torchrun --nproc-per-node 8 --no-python /shared_nfs/kimi-k31/results/m384_fault_20261102/k31_gdb_ptr.sh $B \
  --rows 128 --nvb-a 1 --nvb-m 1 --no-t1 --launches 8 --step 8      # logs/m384dbg/gdb_m128ptr_r<rank>.log
python3 /shared_nfs/kimi-k31/results/m384_fault_20261102/rlane.py /tmp/k31_rocm_dump/<kernel>.s <line> v205 15,16
rm -f /tmp/gpucore*
```
**Code state (uncommitted, /sgl-workspace/sgl-amd-mono on k31/mono-s1 @ ad8388a):** as in the previous handoff, plus
two more `# DEBUG-K31` hooks: `K31_DBG_ROCM_DUMP=<dir>` in `common/rocm_codegen.py` (rank 0 keeps `.ll` and adds
`.s` of each ROCm compile) and `K31_DBG_PTRS=1` in `layer_bench.py` (each rank prints every buffer's range to
stderr). Remove all `# DEBUG-K31` before any commit.

### Previous handoff (2026-11-02 11:45, superseded: 10.25 root-caused above; ROCm LLVM exposes a 2nd fault)

**Status:** ROCm LLVM codegen (10.24) fixes the bucket-384 fault but is NOT a clean fix: with it the server died
in wide warmup (bucket 128) and a bench variant faults every time. Server path blocked until this is understood.
Do NOT run the server with `SGLANG_AMD_MONO_ROCM_LLVM` until 10.25 is resolved. Mode: high reasoning.
**Repro (8 GPUs, ~25 s warm, from /tmp; deterministic):**
```bash
ulimit -c 0; B=/sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/layer_bench.py
SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm torchrun --nproc-per-node 8 $B --rows 128 --nvb-a 1 --nvb-m 1 --no-t1 --launches 8 --step 8   # faults (3-7 GPUs)
torchrun --nproc-per-node 8 $B --rows 128 --nvb-a 1 --nvb-m 1 --no-t1 --launches 68 --step 68        # bundled LLVM: no fault
rm -f /tmp/gpucore*     # --no-t1 always FAILs numerically (bench does not feed mid): fault check only
```
rocgdb on all ranks: `/shared_nfs/kimi-k31/results/m384_fault_20261102/k31_gdb_wrap.sh` pattern (precise-memory,
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`, one log per rank; see 10.25 for the m128 wrapper).
**Next, in order:**
1. Is the faulting load's descriptor base (s[64:67] = 0x7fbe11dc0000) the right buffer? Print the bench's tensor
   pointers (bank, cw_m, prefix_out ...) on the faulting rank, or chase the descriptor in the ROCm ISA (needs an
   ISA dump of the ROCm build: in rocm_codegen keep the .ll / add `-S` output when a dump dir is set; chase.py and
   rdef.py in the results dir work on ROCm ISA, chase.py follows VGPR-lane spills).
2. If the base is wrong: same method as 10.24 (IR correct? reaching defs of the spill slot) -> compiler bug in
   ROCm LLVM too -> the real fix is SGPR pressure (both compilers fail on different variants).
   If the base is right (in-range load faulting?), re-read: the dmesg says write; the faulting instruction may be
   a store nearby -> kernel race / OOB, possibly the same as 10.22 (2) (bucket 128 cold start, 1 in 3, bundled).
3. Then resume: server gsm8k200 (DFLASH, CUDA graphs, wide on, buckets to 384; THREADS 16 and 48), commit, report
   to FlyDSL.
**Code state (uncommitted, /sgl-workspace/sgl-amd-mono on k31/mono-s1 @ ad8388a):** `common/rocm_codegen.py` (new),
`__init__.py` (install behind SGLANG_AMD_MONO_ROCM_LLVM), `stages/moe_front.py` (rgat read CM_SYS),
`test/kernels/kimi_k31/layer_bench.py` (ROCm option, per-replay barrier, DEBUG-K31 pad), `gfx950/layer.py`
(DEBUG-K31 `K31_DBG_STOP` / `K31_DBG_LLVM` only), plus the older `spec.py` / `wide.py` edits. Remove all
`# DEBUG-K31` before any commit. GPUs free, no server.

### Previous handoff (2026-11-02 12:00 doc time, superseded: "FIXED by ROCm LLVM" was premature)

**Status:** the bucket-384 fault is FlyDSL 0.4.1's bundled AMDGPU backend: the same LLVM IR through ROCm's LLVM
(AMD clang 23.0.0git 8f497e09) is correct. New `sgl_amd_mono/common/rocm_codegen.py` routes FlyDSL device codegen
through ROCm clang (LLVM IR -> hsaco; FlyDSL's own FLYDSL_COMPILE_LLVM_DIR path fails on MLIR syntax). With it the
repro no longer faults and every layer_bench size / nvb / capture variant PASSes at the same speed (+-1%). Bundled
still faults on the repro. Cutting SGPR pressure locally did not work (spills 1078 -> ~1000); not pursued now.
All code uncommitted in /sgl-workspace/sgl-amd-mono (rocm_codegen.py, layer_bench.py option + per-replay barrier,
moe_front.py CM_SYS logits read, DEBUG-K31 hooks in layer.py / layer_bench.py).
**Next:** (1) wire `rocm_codegen.install()` into the server's mono init behind `SGLANG_AMD_MONO_ROCM_LLVM`
(all mono kernels then compile through ROCm LLVM), check cold compile time; (2) gsm8k200 TEMP 0 THREADS 16, DFLASH,
CUDA graphs, `SGLANG_AMD_MONO_K31_WIDE=1` with buckets up to 384, plus mono S1 only (K1/K2 now ROCm-compiled too);
(3) remove DEBUG-K31 hooks, commit (ask the user); (4) report to FlyDSL with the IR + chase.py evidence.
Then the old list below from its item 3 (DFLASH re-baseline, c4 wide profile).
**Repro (8 GPUs, ~25 s warm, from /tmp):**
```bash
ulimit -c 0; B=/sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/layer_bench.py
torchrun --nproc-per-node 8 $B --rows 384 --nvb-a 3 --write-a --nvb-m 4 --no-t1        # bundled: faults
SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm torchrun --nproc-per-node 8 $B --rows 384 --nvb-a 3 --write-a --nvb-m 4 --no-t1  # no fault
rm -f /tmp/gpucore*     # --no-t1 always FAILs numerically (bench does not feed mid): fault check only
```

### Previous handoff (2026-11-02 10:40, root cause; superseded above)
### Previous handoff (2026-11-02 09:45, K1 under DFLASH root-caused + fixed, section 10.23)

**Status:** root cause found and fixed (option B, no K1-decline mitigation). K1's scratch layout grows with the
layer's bank depth (`nblocks`), but `k1.scratch_bytes()` sized the runner's shared scratch for `nblocks=1`. Every
K1 layer with `nblocks >= 2` wrote its `x` rows (bf16) up to ~50 KB past the end of `runner.scratch`. Under DFLASH
the caching allocator placed the KV allocator's `free_pages` right after the scratch, so the next prefill / verify
drew garbage page ids: DSA indexer faults, or two requests sharing pages (cross-talk). Details in 10.23.
**Fix (committed locally, sgl-amd-mono `k31/mono-s1` ad8388a on 581b173, pushed to fork kkHuang-amd 2026-11-02; wide / DFLASH-accept /
layer_bench edits still uncommitted):** `kimi_k3/gfx950/k1.py` `scratch_bytes()` sizes for
`MAX_BLOCKS` (738,560 -> 789,056 B); `warmup()` asserts each K1 layer's `kda.scratch_bytes(key)` fits the runner.
All `# DEBUG-K31` code removed from both trees; sglang-k31-amd clean at `origin/dev_amd` b588cf7952.
**Verified (mi355-4, 2026-11-02):** mono S1 + K1 + DFLASH, wide off, CUDA graphs on: gsm8k200 TEMP 0 THREADS 16
0.975 (accept 5.13 req-weighted; no mono 0.98 / 4.96), 3 x 4 concurrent `wide_req` correct, no fault.
`results/acc/dflash_mono_k1fix`.
DSPARK regression with the fix: mono S1, gsm8k200 TEMP 0 THREADS 1 (K1 every step) 0.98, accept 4.947, 4 concurrent
correct, no fault (`results/acc/dspark_mono_k1fix_t1`).
**NEXT SESSION (planned 2026-11-02 09:55), in order:**
1. Bucket-384 fault (10.22 (1), deterministic bench repro, 8 GPUs ~70 s). First a cheap audit with today's
   lesson: every buffer the wide planner sizes (`pl.scratch_bytes()`, peer offsets, per-stage regions) vs what
   each stage writes at rows 384 / nvb 3-4 (K1 had a layout that grew with nblocks and a size taken at
   nblocks 1). Only if sizes hold, chase the race hypothesis (T2 starting without waiting).
   Also check the bucket-128 one-in-three cold-start warmup fault (10.22 (2)) against the same audit.
2. Commit the remaining uncommitted sgl-amd-mono edits on top of ad8388a (ask the user first): `kimi_k31/wide.py`
   graph-safe KDA tables + widths capped to warmed buckets, `kimi_k31/spec.py` DFLASH accepted + debug logs
   removed, `layer_bench.py --no-t1 --restart`. Before committing: gsm8k200 TEMP 0 THREADS 16 with
   `SGLANG_AMD_MONO_K31_WIDE=1` on DFLASH + CUDA graphs (previous wide-on DFLASH run 0.985 was before ad8388a).
3. Re-baseline on the serving config, which is now DFLASH (draft below): ab_bench c1 (S1) / c4 / c16 / c48 for
   no mono, mono S1, mono S1 + wide. Earlier A/B numbers are DSpark-only (S1 c1 4.96 -> 4.49 ms; wide slower
   c4 +13% / c16 +5.8%).
4. Wide slower than off at c4 / c16: profile one c4 wide step (per-layer host work, SGLang routed experts, bank
   / prefix gathers) and attribute the gap.
5. Lower priority: warmup compile cost (~150 s a bucket cold; nvb as a runtime value, fewer buckets).
Rules for every item: `NUMERICS_RULES.md`; `ulimit -c 0` before a server; after a fault `rm -f /tmp/gpucore*` and
`gpucore*` in the crashed process's cwd; after a scheduler exception kill -9 the whole server tree (a SIGTERM'd
TP0 left TP1-7 spinning at 100%).
**Repro of the old failure (pre-fix k1.py):**
```bash
ulimit -c 0; S=/workspace/claude-skills/kimi-k31/scripts
D=/shared_nfs/RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024
SIM=0 bash $S/launch_server.sh k1dbg --enable-amd-monokernel --speculative-algorithm DFLASH \
  --speculative-draft-model-path $D --cuda-graph-backend-prefill disabled --disable-cuda-graph
python3 $S/wide_req.py 4     # pre-fix: cross-talk or fault; post-fix: 42 / 44 / 46 / 48
rm -f /tmp/gpucore*          # after any fault; after a scheduler exception kill -9 the whole tree
```

### Previous handoff (2026-11-02 09:00)
Task: option B, root-cause mono S1's K1 breaking DFLASH serving (done, see above and 10.23).
### Previous handoff (2026-11-01 22:30)
## CONTINUE HERE (handoff 2026-11-01 22:30, mi355-4 -- gsm8k + CUDA graphs done, A/B half done, m384 fault open)

**Update 2026-11-01 22:30 (section 10.22):** eager gsm8k PASSES (wide 0.99 vs off 0.98), CUDA graphs WORK
(wide widths captured, gsm8k 0.98 vs off 0.99), A/B done (buckets 16..256): wide is SLOWER than off at c4 / c16.
Code changes are UNCOMMITTED in `/sgl-workspace/sgl-amd-mono` (on top of 581b173): `wide.py` graph-safe KDA
tables + widths limited to warmed buckets, `spec.py` debug logs removed, `layer_bench.py --no-t1 --restart`.
**Next (new order):** (a) A/B done: wide SLOWER (TPOT c4 8.72 vs 7.72, c16 16.61 vs 15.70, c48 31.35 vs
31.28) -> profile a c4 wide step (per-layer host work, SGLang routed experts, gathers); (b) bucket 384 fault (10.22, deterministic bench repro); (c) commit code.
After any fault: `rm -f /tmp/gpucore*` AND `gpucore*` in the cwd of the crashed process (torchrun from the repo
root left 133 GB in `/sgl-workspace/sgl-amd-mono`, filled `/`, killed a server with "No space left").

### Previous handoff (2026-11-02 05:00)

**Status:** the wide-row KDA layer launch (`sgl_amd_mono/kimi_k31/gfx950/layer.py`, T1 .. T6 + DSpark capture)
now SERVES verify steps end to end in an eager server: `SGLANG_AMD_MONO_K31_WIDE=1` routes R x 8 steps (R 2 ..
48) to `kimi_k31/wide.py`; 4 concurrent requests answered correctly (rows 24 / 32). Real-model per-hand-off check
passed before integration (10.20). Not yet measured: accuracy (gsm8k), speed (A/B), CUDA graphs. No server
running on mi355-4, GPUs free.
**Code:** `/sgl-workspace/sgl-amd-mono` `k31/mono-s1` @ 581b173 (fork kkHuang-amd/sgl-amd-mono, pushed).
Integration pieces: `kimi_k31/spec.py` (`_K31Built`: rows <= 8 -> Kimi-K3's S1 launches, else `wide.Wide`;
wide widths only with the env), `kimi_k31/wide.py` (step loop, KDA state hand-off, warmup), `wide_check.py`
(eager real-model check, `SGLANG_AMD_MONO_K31_WIDE_CHECK=<layer>`). Design + numbers: sections 10.11 .. 10.21.

**Repro (eager, ~2 min load + warmup):**
```bash
S=/workspace/claude-skills/kimi-k31/scripts
SGLANG_AMD_MONO_K31_WIDE=1 SGLANG_AMD_MONO_K31_WIDE_BUCKETS=16,32 bash $S/launch_server.sh wideN \
  --enable-amd-monokernel --disable-cuda-graph
python3 $S/wide_req.py 4        # 4 concurrent greedy requests; log: rg 'k31 wide\]' <log>
rm -f /tmp/gpucore*             # AFTER ANY GPU FAULT: each crash leaves ~17 GB a rank in /tmp (filled / once)
```
**Next, in order:**
1. Accuracy, eager: gsm8k (`NUM=200 THREADS=1 TEMP=0 bash $S/gsm8k_gate.sh <tag> <log>` with SIM=0, real draft)
   with `SGLANG_AMD_MONO_K31_WIDE_BUCKETS=16,32,64,128` and concurrency that keeps R >= 2; compare to the same
   server without the env. Expect the top-16 flip rate (10.20) to be the main numeric difference.
2. Remove the temporary debug logs in `spec.py` (`routed:` / `wide backbone ...`) once (1) passes.
3. CUDA graphs: the step loop has host work per layer (`kda_table` tensors built per step, `.tolist()`-free but
   `torch.tensor` allocations; `epoch[0] += 1`; bank / prefix gathers) -- make them graph-safe (prebuilt
   tables per bucket and layer, device-side slot rows), then let the framework capture the wide widths.
4. Endpoint A/B c4 / c16 / c48 + gsm8k gate (pass criteria below).
5. Startup cost: warmup compiles 23 variants a bucket (~150 s a bucket cold, ~1 s from the JIT cache); six
   buckets cold ~15 min. Options: nvb as a runtime value (one variant for nvb 0..7), fewer buckets.

**Decisions (user, 2026-10-31):**
1. S1 then S2, but S1 must not lock in PR #13's structure: reuse what is good, redesign what limits S2
   (section 2b). S1 is built on the S2 architecture, with <= 8-row stage variants.
2. New package `sgl_amd_mono/kimi_k31/`.
3. Design docs committed and pushed (skills repo).
4. No K3 c1 reproduction; P0 starts directly on K3.1.
**Code home:** `/sgl-workspace/sgl-amd-mono` (github sgl-project/sgl-amd-mono, detached at 90c2355 = main).
Reference: PR [#13](https://github.com/sgl-project/sgl-amd-mono/pull/13) `billishyahao/kimi_k3`
(Kimi-K3 layer monokernel, open, 7.3k lines, last push 2026-10-29). SGLang side:
`/sgl-workspace/sglang-k31-amd` `origin/dev_amd` 3d2aa95c9d (has `kernel_plugins/amd`,
`--enable-amd-monokernel`, and `forward_pre_o_proj` on both the KDA and the MLA attention).
**Repro of the numbers below:**
```bash
python3 scripts/trace_layer_kinds.py /shared_nfs/kimi-k31/traces/c16_isl8k_osl1k_noradix_20261030/*TP-0.trace.json.gz VERIFY 20
```
**Pass criteria for any phase:** endpoint TPOT moves (ab_bench c4/c16/c48, plus c1 for S1), gsm8k200 greedy
within noise of 0.985 with the real draft, accept length not collapsing, fault injection stops every rank.

## 0. What K3.1 is, and what PR #13 already covers

K3.1 has the same dims as K3 (checked both `config.json`): 93 layers (1-based), hidden 7168, 69 KDA layers
and 24 full-attention layers (4, 8, ..., 92, 93), layer 1 dense FFN, 92 latent-MoE layers
(896 experts, top-16, latent 3584, expert inter 3072, 2 shared experts), attention-residual bank
(`attn_res_block_size` 12). The only architectural difference: K3.1's full-attention layers are DSA
(MLA q_lora 1536 / kv_lora 512 / rope 64, 96 heads, plus an indexer: 32 heads x 128, top-2048 every layer).
`KimiK31ForConditionalGeneration` subclasses the K3 model in sglang-k31-amd.

PR #13 (K3, TP8, DCP1, aiter attention backend, DSpark block 7, 1..8 rows a step):
- K1 per KDA layer: AttnRes seam -> input_layernorm -> in_proj -> f_b -> conv -> recurrence -> gated norm.
- K2 per MoE layer (all 92): o_proj + in-kernel AR -> MLP AttnRes seam -> router top-16 -> latent down +
  shared gate_up -> MXFP4 experts -> shared down + latent up -> in-kernel MoE AR. Router and latent
  down/up are computed 1/8 per rank and all-gathered (the standard path replicates them).
- Full-attention layers keep SGLang's attention up to `forward_pre_o_proj`.
- Measured on K3 c1: 41.63 -> 55.26 decode steps/s (step 24.0 -> 18.1 ms, -5.9 ms); AgentX c1 TPOT -21%.

What PR #13 does NOT give K3.1, and the reason this design exists:
1. Width. All stages are GEMV-style for S <= 8 rows (c1). Our A/B points are c4/c16/c48 = 32/128/384 verify
   rows. Its one-shot AR (every rank pushes its full partial to every peer) costs 7x the data per rank:
   fine at 8 rows (0.8 MB out), ~13 MB out at 128 rows.
2. DSA layers: indexer, KV write, sparse MLA, DCP8 combine stay outside (24 layers, 17.1 ms/step at c16).
3. Config gate: `model_type kimi_k3`, `attention_backends {aiter}`; K3.1 A/B config runs `attention_backend=triton`,
   DCP8, fp32 ReplaySSM ring (`--enable-linear-replayssm-spec`, PR #7). PR #13's best run used bf16 SSM
   state in one arm; K1 reads its own conv buffer (SGLang's spec conv window is one entry short).
4. Tested against SGLang 8e7601ae4a, not our dev_amd.

## 1. Op list per layer from the c16 trace (VERIFY, bs 16 x 8 = 128 rows, TP8 + DCP8)

Trace `c16_isl8k_osl1k_noradix_20261030` TP0, 20 VERIFY steps averaged; kernel-sum 47.93 ms/step
(GPU fully busy, single stream). Segments end at each MoE stage 2, so a segment = the previous layer's
MoE AR tail + this layer's attention + this layer's MoE. GEMM roles come from the M=128 shape log
(`logs/server_i3_gemm_shapes_20261031.log`, N x K per rank) and kernel order; "?" = inferred.

| segment kind | segs/step | kernels/seg | us/seg | ms/step |
|---|---:|---:|---:|---:|
| KDA layer | 67 | 22.2 | 446.0 | 29.88 |
| DSA layer | 24 | 43.5 | 712.2 | 17.09 |
| head (layer 1 dense + layer 2 KDA, embedding) | 1 | 37 | 672.7 | 0.67 |
| tail (final AR/agg/norm, lm_head, logits AG, softcap) | 1 | 10 | 286.3 | 0.29 |

KDA layer, in order (us, avg):

```text
MoE tail of L-1 : AR 2stage [latent|shared] ~20 | add_rmsnorm_quant 4.3 (latent norm) | latent up 7168x3584 20.2
                  | add3 4.4 | _agg (attn-side AttnRes) ~8.5
attention       : in_proj 6288x7168 29.8 | f_b 1536x128 4.3 | conv1d_update 5.9 | KDA recurrence 26.3
                  (Triton; 16.1 with PR #7) | gated norm 4.0 | o_proj 7168x1536 10.3 | AR ~20 | _agg (MLP-side) ~8.5
MoE front       : fused [shared gate|up 1536 | latent down 3584 | router 896] x7168 30.2 | situ_and_mul 4.1
                  | shared down 7168x768 6.6 | topk_reg 5.1 | moe_sorting x2 10.5 | fused_mx_quant_moe_sort 12.5
experts         : stage1 135.8 | stage2 73.9
by class        : experts 209.7 | dense GEMM 101.4 | AR 39.9 | KDA recurrence 26.3 | small ops 67.9
```

DSA layer, in order (MoE tail and MoE front/experts as in the KDA layer):

```text
attention front : q_a|kv_a 2112x7168 13.3 | fused_qk_rmsnorm 4.3 | q_b 18432x1536 19.6 (hipBLASLt; all 96 heads
                  on every rank under DCP8) | index q 4096x1536 8.2 | index k 128x7168 5.6 | layer_norm 4.0
                  | act_quant 5.3 | indexer_k_quant_and_cache 4.0 | index weights 32x7168 5.4 | head_gate_q_scale 4.2
                  | (layer 4 only: 12 x elementwise ~3.8 = per-forward memo, ~0.26 ms/step)
indexer + DCP   : paged_mqa_logits 7.6 | dcp_local_candidates 4.3 | allgather_vec 45.5 | dcp_global_slots 21.6
KV + q path     : W_UK bmm? 14.0 | quantize_k_cache 4.3 | set_mla_kv_buffer 4.2 | localize_compact 4.5 | mxfp8_quant 5.5
attention       : opus_mla_decode_mxfp8 51.1
DCP combine     : dcp_pack_a2a_send 7.9 | NCCL a2a 57.8 | dcp_lse_combine 5.2
attention back  : W_UV bmm? 11.9 | output-gate GEMM 1536x7168? 12.1 | mla_output_gate 3.9 | o_proj 7168x1536 10.5 | AR ~20
by class        : experts 209.4 | dense GEMM 158.6 | comm 155.3 (AR 38.9, DCP 116.4) | sparse MLA 51.1
                  | indexer/topk 33.5 | small ops 104.3
```

Where the 47.9 ms go, and which part a mono kernel can touch at c16:

```text
experts (stage1+2)   19.3 ms  HBM roofline (~8 TB/s; ~807 of 896 experts touched at 2048 pairs) -> not a target
dense GEMM           10.9     ~3-4.3 TB/s, floor 4.8 ms; row 3: in-kernel tiles tie tgemm at best
                              -> gain only by NOT doing replicated work (router, latent down/up)
comm                  6.3     AR 187 x ~20 us (~185 GB/s busbw) + DCP 116 us x 24; already link-bound
                              -> gain by overlap and by sending fewer bytes, not by "faster comm"
small ops + prologue  ~7.2    67.9 x 69 + 104.3 x 24 (+ head/tail); each ~4-13 us, real work ~1 us
                              -> the main mono target
KDA / sparse MLA / indexer ~4.2  compute kernels; keep (PR #7 HIP KDA), fold only their glue
```

Caveat: only c16 is traced. Small-op cost is roughly width-independent, so at c4 (step ~30 ms) the same
absolute saving is a larger fraction; MoE at c4 touches fewer experts. P0 captures c1 and c4.

## 2. Scope options and expected ms/step (c16 VERIFY 47.9 ms unless stated)

Common savings mechanisms (per layer, c16):
- A. Small-op fusion: each fused hand-off still costs a grid-wide data dependency (~2-4 us through the
  mailbox) but not a launch + ramp + tail. KDA 67.9 -> ~20 us, DSA 104.3 -> ~35 us.
  = 3.3 + 1.7 = ~5.0 ms (range 3.5-5.0).
- B. De-replication (PR #13 does it): router + latent down computed 1/8 per rank (6016-row GEMM -> 2096 rows,
  30 -> ~11 us, + 0.9 MB latent / 0.5 MB logits all-gather ~5-7 us), latent up for 1/8 of the output
  columns (20 -> ~4 us) with a column-sharded residual. Net ~20 us x 92 = ~1.8 ms (1.5-2.5).
  Overlaps PR #8's row-sharded tail (standard path, prefill threshold 4096): measure that first (P0.4).
- C. Comm: two-shot AR (RS + AG) pushed tile by tile from the producing GEMM epilogue, seam/norm on the
  owner's 1/8 columns, ~8-12 us of each ~20 us AR hidden -> 1.5-2.2 ms. DCP: apply W_UV (512 -> 128 per head)
  to the partial outputs BEFORE the a2a (linear, so LSE rescale commutes) -> a2a bytes / 4 (11 -> 2.75 MB),
  pack/combine fused: ~45 us x 24 = ~1.1 ms. Candidate all-gather 45.5 us: size unknown, 0-0.6 ms.
  C = 2.6-3.9 ms.
- D. GEMM tiles inside the launch: 0 assumed (row 3), +0..1 ms if the next stage's weights stream
  while a stage waits (PR #13's "issue weights before the first wait").
- E. Experts: 0 at c16.

| option | launches/step | covers | c1 est. (VERIFY) | c16 est. | effort / risk |
|---|---:|---|---|---|---|
| S0 = today | 2581 | - | VERIFY 18.9 | 47.9 (c4: 26.1) | - |
| S1 <= 8 rows on the 2b architecture | c1: ~2150 -> ~900 | KDA layers whole + o_proj .. MoE of the DSA layers; DSA attention + DCP on SGLang | **~-5 ms of 18.9 (-27%)** | 0 (falls back) | medium: planner, gate, DCP8, triton backend, fp32 ring |
| **S2 per-layer, wide rows, experts kept as aiter launches** | ~69 x 3 + 24 x 5 = ~330 | everything except experts stage1/2, sparse MLA | ~-7 ms (-37%) | **-8 to -9 ms (-17..-19%)**; c4 -5 to -8.6 ms (-20..-33%) | high: MFMA-tile stages, waitcnt-flag comm, DCP in kernel |
| S2+ experts inside the layer launch | ~69 + 24 x 3 | S2 + stage1/2 as stages | - | S2 + 0.5-0.9 ms (overlap expert weight issue with prologue, -2 boundaries) | very high: grouped MXFP4 GEMM at roofline inside a multi-role kernel |
| S3 one persistent kernel per step | 1-3 | all 93 layers + tail | - | S2+ + ~0.7-1.0 ms (remaining boundaries x 2-3 us) | highest: one register/LDS budget for every stage, compile time, debuggability |

S2 launch shape (one "layer launch" = from the MoE output of layer L-1 to the expert inputs of layer L):

```text
KDA layer L : [K_L: MoE-tail(L-1) RS/AG + latent up + residual + AttnRes seam + input norm
               + in_proj + f_b + conv + recurrence + gated norm + o_proj + AR + MLP seam
               + router/latent-down/shared gate_up (1/8, AG) + SiTU + shared down + top-16 + sort + mxfp8 quant]
              -> aiter stage1 -> aiter stage2
DSA layer L : [K_L,front: MoE-tail(L-1) ... + q_a/kv_a + norms + indexer q/k/w + logits + local top-k
               + in-kernel candidate AG + global slots + KV write + q_b + W_UK]
              -> opus sparse MLA
              -> [K_L,back: W_UV on partials + in-kernel a2a + LSE combine + gate + o_proj + AR + MLP seam + MoE front]
              -> aiter stage1 -> aiter stage2
```

Recommendation: S1 first (small, gives c1 and validates the PR #13 base on K3.1), then S2 in sub-phases
(section 5). S2+/S3 only if S2 lands and the remaining boundaries measure > 1 ms/step.

## 2b. PR #13: what kimi_k31 reuses and what it redesigns

Rule: S1 and S2 share one architecture. S1 differs from S2 only in which stage variants a width bucket
picks (GEMV <= 8 rows, MFMA tiles >= 16 rows) and in whether the experts run inside the launch.

Reuse (as is, or by import from `common` / `kimi_k3` once #13 merges):
- `common/` runtime: tagged mailbox, guarded waits + fail-stop, uncached peer memory, epoch, stamps.
- Host-bumped step epoch captured in the graph (`tag.py`): no buffer zeroing between steps.
- Role rule: CTAs that poll carry no weight loads in flight; weight CTAs issue the next stage's weights
  before their first wait (MoE weights stream while the AttnRes chain settles).
- De-replication of router / latent down / latent up (1/8 per rank + all-gather).
- In-place weights (bf16 checkpoint layout, MXFP4 experts in `Mxfp4MoEMethod`'s layout): no KV budget cost.
- Correctness lessons: K1 owns its conv window (SGLang's spec window is one entry short); compile by running
  on real weights (FlyDSL compile-only folds zero args into an unselectable pattern); kernel checks against
  SGLang ops, bitwise graph replay, fault injection, `SGLANG_READS` contract.
- The <= 8-row GEMV stages (seam, KDA K1, MoE GEMV) as S1's narrow variants, after a review on K3.1 data.

Redesign (things that would cap S2 or the later path):
| PR #13 | limit | kimi_k31 |
|---|---|---|
| CTA roles as fixed index ranges (0..55 seam, 56..159 o_proj, 160..243 router/latent, ...), `NA = 56`, `ROWS = 16` | one layout for one width; not XCD-aware; every new stage re-cuts the ranges by hand | a build-time planner: stage graph -> tasks -> CTAs per width bucket, grouped by XCD (b % 8), roles derived, not written |
| launch cut at the o_proj input (K1 = KDA front, K2 = o_proj .. MoE output) | the MoE tail and the next layer's front are split by a launch; the full-attention layers can't join | cut at the expert inputs: one launch = MoE tail(L-1) + attention(L) + MoE front(L); experts inside only for the GEMV buckets |
| one-shot AR (full partial to every rank) | 7x bytes per rank; ~13 MB out at 128 rows | width-switched: one-shot <= 8 rows, two-shot RS + AG above, epilogue-pushed tiles |
| residual / AttnRes seam replicated on every rank | 8x the bank reads at wide rows | column-sharded residual and bank (each rank 896 columns), scalars exchanged for the full-row reductions |
| builds per width 1..8 (`MAX_ROWS`) | 48 builds at bs 1..48 x 8 rows | width buckets with masked rows (8, 16, 32, 64, 128, 256, 384) |
| full-attention layers on SGLang up to `forward_pre_o_proj` | 24 DSA layers x 712 us outside | DSA front / back launches (P2b / P2c) on the same planner |
| `model_type kimi_k3`, aiter attention backend, DCP1, bf16 SSM state run | does not start on the K3.1 A/B config | `kimi_k31` spec: triton backend, DCP8, fp32 ReplaySSM ring (PR #7) |
| `MAX_BLOCKS` AttnRes bank bound | fixed at build | derived from config, checked in the spec |

Open on #13: its body says two of the 1..8 widths "abort today" (the section it points to is not in the
body); find which before reusing the GEMV stages.

## 3. Synchronization across 8 XCDs and TP8 comm inside the kernel

Hardware facts (MI355X, from row 3 measurements and the sgl-amd-mono runtime): 8 XCDs x 32 CUs; L2 per XCD,
not coherent across XCDs; workgroup b lands on XCD b % 8 (verified via HW_REG_XCC_ID); a last-CTA fixup
with `__threadfence` cost ~50 us; device atomics are slow (3M fp32 atomics ~8-10 us) and same-XCC scope does
not help; owner stores must be sc0|sc1 (write-through) or another XCD's L2 writeback overwrites them.

3.1 Intra-GPU (reuse `sgl_amd_mono/common`):
- Resident grid, one workgroup per CU (256), role by CTA index (`common/plan.py`), never a grid barrier.
- Dataflow hand-offs through the tagged mailbox (`common/sync.py`): payload words carry the step epoch tag
  and live in uncached scratch (`hipExtMallocWithFlags(..., uncached)`); a consumer polls the data itself,
  so there is no flag + fence pair. Epoch bumped once per forward by the host and captured in the graph
  (PR #13 `tag.py`), so no buffer is zeroed between steps.
- XCD-aware placement (new vs PR #13): split each stage's tiles into 8 XCD groups; a hand-off whose
  producers and consumer share an XCD (e.g. GEMM N-tiles -> the per-column seam of the same columns) uses
  cached L2 stores + L1-bypass loads; only all-to-all hand-offs (full-row RMS / score partials, K-split
  reductions, router logits) go through uncached memory, and those carry scalars or small rows.
- No split-K across CTAs in the dense GEMMs at these widths (row 3: the split-K finish, not the weight
  stream, bounds the design); pick N-tile counts that fill 256 CUs, and reduce K inside a CTA.
- Polling CTAs carry no weight loads in flight (a poll retires behind earlier loads; PR #13 role split);
  weight-streaming CTAs issue the next stage's weights before their first wait.
- Every wait has a deadline (`common/guard.py`, half the watchdog, <= 60 s); a failed wait stops every rank.

3.2 TP8 comm (peer memory, `common/peer.py`, system-scope stores over xGMI, local polling):
- AR replaced by two-shot: the GEMM epilogue writes column slice j of its bf16 partial straight into rank j's
  mailbox (RS); rank j sums the 8 partials in rank order (bit-match the custom AR's order, fp32 accumulate),
  applies residual / AttnRes / norm on its 896 columns, then pushes the result slice to every rank (AG).
  Full-row reductions (RMS, AttnRes score dots) exchange per-row partial scalars only.
  Bytes per rank at 128 rows: RS 1.6 MB out + AG 1.6 MB out vs one-shot 12.6 MB.
- Payload format by size (P0.2, section 6): tagged 64-bit words (bf16 pair + tag) up to ~0.9 MB (<= 32 rows
  x 7168), raw 128-bit stores + `s_waitcnt vmcnt(0)` + one flag per (block, source) above; never
  `__threadfence_system` (~50 us floor). One-shot vs two-shot by width as before.
- MoE tail: AR of the latent (0.9 MB) + RS of the shared partial, latent up on the owner's columns, AG of
  the next layer's input row (needed in full by in_proj / q_a, K = 7168).
- DCP8: (1) indexer candidates: each rank writes its local top candidates to every rank's mailbox, then the
  global-slot merge runs locally; (2) partial attention outputs: W_UV applied locally for all 96 heads
  (+11 MB weight read ~2-3 us), then [rows, head group, 128] + LSE sent to the head owner, combined there.
- Precision floor: bf16 payloads, no quantized collectives (policy 2026-10-30); fp8 AG only as #9/#10
  experiments.
- xGMI traffic competes with HBM-bound stages at the receiving end; overlap gains must be measured (P0.3),
  not assumed.

## 4. Integration

- CUDA graphs: launches are captured inside SGLang's decode/verify graphs. Routing is by graph key (width)
  only, so every rank decides alike and replay repeats the capture decision (sgl-amd-mono "graph-safe
  routing"). Builds per width bucket (8, 16, 32, 64, 128, 256, 384 rows) with masked rows, not per bs
  (bs 1..48 x 8 = 48 widths would be 48 compiles x layer kinds). Widths outside the built set fall back.
- Spec verify (DSPARK): R requests x 8 tokens. KDA: recurrence over the 8 tokens with the fp32 ReplaySSM
  ring layout of PR #7 (`SGLANG_ROCM_KDA_VERIFY_HIP_FP32_RING`), conv window read as PR #13's K1 does;
  post-verify fold (replayssm, 0.33 ms) stays on SGLang. DSA: per-token causal top-k over its own position,
  KV for all 8 tokens written before attention. DSpark draft hidden captures
  (`dspark_layers_to_capture`) written by the layer launch, as PR #13 does. Draft step (1.45 ms) stays on
  the standard path (later: DSV4.1 put its draft on mono).
- DCP8: KV is sequence-sharded across the 8 ranks; the launch reads the same paged fp8_ds_mla cache and
  DCP slot tables SGLang writes (`dcp_local_candidates`, `_localize_compact` semantics), so prefill and
  HiCache stay untouched.
- Plugin / package: new package `sgl_amd_mono/kimi_k31/` (spec `model_type kimi_k31`), importing PR #13's
  `kimi_k3` stages (seam, KDA, MoE GEMV) once #13 merges; K3.1-only stages (wide-row tiles, two-shot AR,
  DSA front/back, DCP) live in `kimi_k31/gfx950/`. Coordinate with the #13 owner (billishyahao) before
  changing shared `kimi_k3` / `common` code. `SGLANG_READS` + `test/unit/test_contract.py` against dev_amd.
- Flags: `--enable-amd-monokernel` (server), plus the existing K3.1 flags. Interplay with PR #7 (KDA ring
  layout must match), #8 (row-sharded tail; prefill only, untouched), #9/#10 (prefill sparse MLA; untouched).
- Check mode: per-layer compare against SGLang ops (`SGLANG_AMD_MONO_CHECK` pattern of glm53/minimax).

## 5. Risks and phased plan

Risks:
1. Wide-row stages are new work: PR #13's stages are GEMVs for <= 8 rows; 32-384 rows need MFMA tiles.
2. Dense GEMM tiles inside a multi-role launch may stream slower than FlyDSL tgemm (row 3: our kernels
   tied at best). Occupancy: HBM floor needs ~8 waves/CU; roles share one VGPR/LDS budget.
3. Comm overlap may not show: AR and DCP a2a are already link-bound (~185-190 GB/s); xGMI vs HBM contention.
4. Numerics: reduction order, bf16 rounding points, fp32 ReplaySSM ring; gate = gsm8k200 + KL vs mono off.
5. Hangs: every wait guarded; fault-injection test on 8 ranks per phase.
6. Compile time / number of builds (width buckets x 3 layer kinds); PR #13 compiles by running on real weights.
7. Upstream churn: PR #13 still open; sgl-amd-mono main moves; our SGLang is a fork (dev_amd).
8. c16 is MoE-bound (19.3 ms of 47.9 untouchable here); the endpoint gain shrinks with concurrency.

Plan (each phase: own branch in sgl-amd-mono off main + PR #13, own A/B tag under
`/shared_nfs/kimi-k31/results/`, default off):

- P0 measure (no mono code; no K3 reproduction, user 2026-10-31): (1) K3.1 c1 and c4 traces (only c16
  exists); (2) P2P microbench in the sgl-amd-mono runtime: mailbox hand-off latency same-XCD vs
  cross-XCD, two-shot AR at 0.9 / 1.8 / 2.75 MB, a2a at 2.75 / 11 MB; (3) PR #8 tail threshold lowered to
  decode widths (bounds mechanism B on the standard path). Output: updated section 2 numbers.
- P1 = S1: `kimi_k31` package on the section 2b architecture (planner, cut at the expert inputs), <= 8-row
  bucket with GEMV stages and experts inside, TP8 + DCP8, triton attention backend, fp32 ring; DSA attention
  still SGLang. Gate: c1 TPOT >= -15%, gsm8k, fault test.
- P2a: KDA layer + MoE front/tail at 16-384 rows (MFMA tiles, two-shot AR, de-replication), DSA layers
  still use SGLang attention up to `forward_pre_o_proj` and the layer launch from o_proj on.
  Expected ~5-6 ms/step at c16 (A for KDA + B + AR part of C).
- P2b: DSA front (norms, indexer, KV write, q path, candidate AG + global slots) in the launch.
- P2c: DSA back (W_UV before a2a, in-kernel a2a + LSE combine, gate). P2b + P2c expected ~3-4 ms/step.
- P3 (optional, re-decide with data): experts inside (S2+), draft on mono, per-step persistent (S3).

## 6. P0 results (mi355-4, 2026-10-31)

6.1 Traces (current tree: sglang-k31-amd k31/sparse-mla-prefill-hip 91ca03666b, flags off;
`traces/p0_c{1,4}_isl8k_20261031/{breakdown,layer_kinds}.txt`). VERIFY span = kernel busy (no idle):

```text
       VERIFY  DRAFT  KDA seg (kernels, us)  DSA seg (kernels, us)  experts/KDA seg  AR/KDA seg  small/KDA seg
c1      18.9   0.99   17.2, 161              40.5, 328              41               20         ~47
c4      26.1   1.06   19.0, 230              41.4, 414              91               22         ~59
c16     47.9   1.45   22.2, 446              43.5, 712              210              40         ~68   (10-30 trace)
```
At c1 / c4 the current tree already runs the HIP KDA verify (7.5 / 8.4 us), `k3_ag_attn_res_kernel` and
`_latent_tail_partial_kernel` (small-width paths), so the per-layer op list differs from the c16 table in
section 1. Small ops + AR are ~42% of a KDA layer at c1 and ~35% at c4: the mono kernel's main target.

6.2 P2P microbench (`scripts/p2p_microbench.py`, torchrun 8 ranks, no server; log /tmp/p2p_mb2.log):

```text
hand-off one-way (1 thread, ping-pong): same XCD 0.39 us, cross XCD 0.42-0.55 us (uncached + system atomics;
                                        hipMalloc + agent atomics 0.47-0.63 us); GPU <-> GPU 0.65-0.69 us
collectives, us per launch incl. back-to-back launch gap, bf16, 256 CTAs x 256 threads, all validated with
per-launch-varying inputs:
                        tagged64   flag+threadfence_system   flag+waitcnt   RCCL
AR    8 x 7168  0.11 MB     8.3          52.1                    14.6        15.8
AR   32 x 7168  0.44 MB    10.4          53.8                    16.0        28.5
AR  128 x 3584  0.88 MB    19.0          54.2                    17.4        30.3
AR  128 x 7168  1.75 MB    35.3          55.5                    23.1        40.9   (custom AR in trace ~15-22)
AR  128 x 10752 2.62 MB    60.1          61.4                    31.5        42.9
AR  384 x 7168  5.25 MB   133.1          90.4                    65.3        59.0
A2A 128 x 12288 3.00 MB    56.6          40.8                    26.4        34.7
A2A 128 x 49152 12.0 MB   227.0         115.8                    98.5        54.0
```
Findings: (1) a hand-off is < 1 us, so a fused stage boundary costs ~1-2 us with skew, cheaper than the
2-4 us section 2 assumed. (2) Tagged words win to ~0.9 MB, then lose half the link to tags. (3) The
system fence is a ~50 us floor; the `s_waitcnt vmcnt(0)` publish (stores to uncached peer memory acked
before the flag) reaches custom-AR speed at 1.75 MB. It passed the varying-data validation, but its
ordering is not a documented guarantee: stress-test (random delays, 10^5 launches) before P2 relies on it.
(4) In-kernel collectives are not faster than custom AR / RCCL at >= 1.75 MB: wide-row comm gains come
from overlap and fewer bytes, not from the transport. (5) The 11 MB DCP a2a must shrink first
(W_UV before the a2a -> 2.75 MB, 26 us in kernel vs NCCL 57.8 today).

6.3 PR #8 row-sharded tail at decode widths (ab_bench, same tree, `results/ab/p0_{control,rowshard8}_20261031`):

```text
                 control   MIN_TOKENS=8
decode c4 TPOT    7.71      8.26  (+7.1%)
decode c16 TPOT  15.80     15.68  (-0.8%, noise)
decode c48 TPOT  31.34     30.35  (-3.2%)
prefill 8k TTFT  528.4     503.5  (-4.7%; prefill takes the same path)
```
Row-sharding the tail with extra standard-path launches loses at 32 rows and pays only at 384: mechanism B
is worth doing only inside the launch (no extra launches), width-switched. Side item for OPT_TRACKER: a
decode threshold of 256-384 rows on the standard path (c48 -3%) is independent of the mono work.

6.4 Revised estimates (section 2 table updated):
- c1, S1: KDA layer small ops ~47 -> ~8 us, AR 20 -> ~8, de-replication ~8 -> ~58 us x 69 = 4.0 ms; DSA layers'
  o_proj .. MoE part ~50 us x 24 = 1.2 ms -> ~-5 ms of 18.9 (-27%); DRAFT unchanged.
- c4, S2: ~72 us per KDA layer (5.0 ms) + DSA ~3.6 ms -> -5 to -8.6 ms of 26.1.
- c16, S2: A ~5.0-5.5 (cheaper hand-offs), B ~1.8, C revised to ~1.5-2.0 (AR 0.5-1.0 + DCP ~1.0), D 0
  -> -8 to -9 ms of 47.9.

## 7. S2 layer launch: dataflow (design, mi355-4 2026-10-31)

One launch per layer L at 16..384 rows (R requests x 8 tokens), between aiter's expert stage 2 of L-1 and
stage 1 of L. The residual stream is column-sharded: rank r owns hidden columns [896 r, 896 (r+1)).
Stages (each a set of tiles; hand-offs through the mailbox; P = partial over ranks):

```text
T1 MoE tail (L-1)  in: expert out latent P [rows,3584], shared down P [rows,7168]
                   latent: two-shot AR (RS + AG)            -> full latent on every rank
                   shared: RS to owners                      -> shared slice [rows,896]
                   latent norm + latent up, owner's 896 output columns only (1/8 of W_up)
                   y slice = up + shared + residual          (column-sharded, no replicated work)
T2 attn seam (L)   AttnRes on the owner's columns; per-(row, source) RMS / score partials and the
                   input-norm RMS exchanged as scalars; bank kept column-sharded (1/8 the reads)
                   x slice -> AG x                           (in_proj needs K = 7168)
T3 attention (L)   KDA: in_proj tiles consume x by K-chunks as the slices land; f_b, conv, recurrence,
                   gated norm per (request, head) task (K1 math, R requests: conv staged per step for
                   R slots, ring or snapshot state); DSA: SGLang's attention between two launches (P2b/c)
T4 o_proj          tiles, epilogue pushes column slices to owners (RS)
T5 MLP seam        owner columns, scalars exchanged, post-attention norm -> x' slice -> AG x'
T6 MoE front       router 112 experts / rank + latent down 448 rows / rank + shared gate_up 1536 rows,
                   one weight pass over x'; AG router logits (0.06 MB) and latent (0.9 MB);
                   top-16, sort, mxfp8 quant -> aiter stage-1 inputs; SiTU + shared down -> shared P
```

Bytes out per rank per layer at 128 rows: latent RS + AG 1.6 MB, shared RS 0.8, x AG 1.6, o_proj RS 1.6,
x' AG 1.6, router + latent AG 0.8: ~8 MB vs ~7.7 MB for today's two all-reduces. Comm volume is unchanged;
what changes is (a) no replicated work (router, latent down / up, seams, norms), (b) every collective is
split into slices that a GEMM produces or consumes tile by tile, so it overlaps, (c) ~22 kernels -> 1 launch
+ 2 expert launches. Formats by size (section 6.2): tagged words <= 0.9 MB, waitcnt-flag above.

Reuse from PR #13: seam / KDA math (stages/seam.py, attention/kda.py), mailbox + guard + peer runtime, the
role rule (pollers carry no weight loads). New: MFMA row-tile GEMM stage (M = 16..384, N-tiles over 256
CTAs grouped by XCD), slice collectives, the planner (stage graph -> tasks -> CTAs per width bucket).
P1b keeps the c1 bucket on PR #13's launches (GEMV, experts inside) and adds the ring; the wide buckets
come from this design (P2a = KDA layers + MoE tail/front, P2b/c = DSA front/back).

Why columns, not rows: DSV4.1's mono runs token parallel (`deepseek_v41/gfx950/token_par.py`: rank r owns
rows r S/TP .., all-reduces become reduce-scatters to the row owner, seams need no exchange). For K3.1 the
latent up (7168 x 3584 bf16, 51 MB) would then be read whole by every rank for its rows -- the replicated
cost we remove (~20 us x 92 at c16; PR #8's row-sharded tail pays it and loses at c4). Columns read 1/8 of
W_up at the price of a scalar exchange per full-row reduction (~1-2 us, section 6.2). Re-decide if the
scalar rounds measure > 4 us a layer.

GEMM tiles at 16..384 rows: x does not fit LDS (128 x 7168 bf16 = 1.8 MB), so the PR #13 GEMV form
(x rows in LDS, weights streamed into MFMA B operands) stops at ~16-48 rows. P2a builds a tile stage from
aiter's FlyDSL a16w16 gfx950 GEMM pieces (`aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`:
`make_gemm_ab_load_context`, `async_load_to_lds`, `write_cshuffle_vec_to_global`), K-chunked x through LDS,
epilogue into the slice mailbox.

## 8. P1a results (mi355-4, 2026-10-31; K3 launches on K3.1, REPLAYSSM=0, SIM acc 3.84)

```text
ab_bench (results/ab/p1a_mono_{on,off}_rs0_20261031)   mono off   mono on
decode c1 TPOT                                          5.06       4.65   (-8.1%)
decode c1 P90 ITL                                       6.36       5.84   (-8.2%)
decode c4 TPOT (not routed: 32 rows)                    7.98       7.92   (noise)
prefill 8k TTFT                                         524.1      522.8
c1 trace mono on (traces/p1a_c1_mono_on_20261031): VERIFY 17.67 ms; K2 ~100 us x ~85 launches, K1 ~37 us x 68
```
Below the -25% estimate. Why: at c1 a layer reads ~300 MB of weights a rank (experts ~250 MB at 128
routed pairs, dense ~70 MB after de-replication), so the step is HBM-bound and launch / small-kernel
overhead is the smaller part. K2 streams at ~3.2 TB/s (100 us for ~320 MB) vs ~120 us for the standard
path's kernels it replaces; K1's in_proj GEMV (90 MB) runs ~2.4 TB/s where the standard hgemm gets 5.1.
Revised c1 levers: K2 weight streaming to ~5 TB/s (~35 us a layer, ~3 ms a step), K1 in_proj at hgemm speed
(~15-20 us x 68 = 1.0-1.4 ms). Next measurement: the launches' stage timeline (`timeline=True` builds,
`gfx950/stamps.py`) to place the K2 / K1 time per stage.

K2 stage timeline, 8 rows, graph-replayed behind plain launches (`time_k2.py`; 103.5 us a launch):
```text
x published (o_proj + AR + seam)  21.5 us | route table (router AG, top-16, queue) 39.9 (one CTA at 38.4,
the rest at 21.7) | expert gate/up done 65 (~25 us, ~6.8 TB/s) | expert down done 86 (~20 us, ~4.4 TB/s) |
latent norm, latent up, MoE AR reduce 102.6
```
The experts stream near the floor; the route phase (18 us) and the tail (16 us) are latency. Halving both
is ~20 us a layer, ~1.8 ms a c1 step: P1b's first items.

gsm8k caveat: the P1a gate (`results/acc/p1a_gsm_{on,off}_rs0_20261031`, 0.99 on / 0.98 off) ran 64 threads,
so most verify steps were wider than 8 rows and took the standard path. It does not gate the mono path.
`gsm8k_gate.sh` now takes THREADS; mono gates run THREADS=1 (every step bs 1, as PR #13's
--max-running-requests 1).

## 9. P1a on the ring, P1b step 1 (mi355-4, 2026-10-31)

```text
ring config (REPLAYSSM=1), SIM acc 3.84      mono off   mono on (K3 launches + K1 ring)
decode c1 TPOT                                 4.96       4.60  (-7.3%)
decode c1 P90 ITL                              6.24       5.81
decode c4 TPOT (not routed)                    7.80       7.79
greedy gsm8k200, THREADS=1, real draft         0.98       0.985   accept len 4.79 / 4.73
(results/ab/p1r_mono_{on,off}_20261031, results/acc/p1r_gsm1_{on,off}_20261031)
```

K2 route chain per rank (8 rows, `time_k2.py`, us after the launch start on that GPU; GPU clocks are not
aligned across ranks, so only per-rank deltas are read): x published +0 | router part pushed +5.1 |
logits in +8.2 | top-16 +16.0 (radix select, one wave: 7.8 us) | route +16.8 | latent part pushed +6.4 |
latent quantized +15.7 (every CTA read ~115 KB of tagged bf16 pairs and quantized all 8 x 3584) | table +18.8.
Step 1 (commit 6e50a20, `stages/moe.py`): top-16 by 16 rounds of a wave max over each lane's 14 sorted
candidates (GLM-5.3-Flash's router form; 53-comparator network) -> 3.9 us, same picks and order; the latent
quantized once by CTAs 160..255 (free first) into a LATQP mailbox that every CTA polls (36 KB each).
Table at +13.3 (was +18.8); K2 103.5 -> 100.6 us a launch.

Tail (step 2 target): experts down done ~81.6 us, but the shared down (448 tasks) mostly runs after it on
every CTA (mark 4 median 89.6), then lnorm / latent up / reduce to ~98. The shared path (SGU -> SiTU ->
shared down) does not depend on the route; it fits the ~13 us where CTAs 56..255 wait for the route table.
Move stage_h (SiTU) to the SGU CTAs and run the shared down there before the experts.

Step 2 measured and rejected (mi355-4, 2026-10-31; not committed): (a) all 448 shared-down tasks on CTAs
56..159 right after their SGU task: 109.5 us a launch (route table last 52.8, those CTAs' experts start
late: each task polls H, pushes FINAL to 8 ranks and publishes, ~4-5 us); (b) only the 56 up-stage tasks
there, the other 392 left after the down: 103.2 us (route table median 39.0 vs 35.4; suspect the early
FINAL pushes competing with the router logits all-gather on xGMI). The tail stays as PR #13 has it.

E2E after step 1 (ring config, SIM acc 3.84; `results/ab/p1b1_mono_on_20261031`): c1 TPOT 4.49 ms,
P90 ITL 5.67 (mono off 4.96 / 6.24, step-0 mono 4.60 / 5.81). gsm8k200 greedy THREADS=1 0.98
(`results/acc/p1b1_gsm1_on_20261031`).

gsm8k per-item (THREADS=1, greedy, real draft; ids matched): mono off vs P1a-ring mono 78/200 identical
generations (correct 196 / 197); off vs step-1 mono 62/200 (196 / 196); P1a-ring mono vs step-1 mono 62/200.
Step 1 is designed to be bit-identical (same picks, order, renorm and quant), so mono-vs-mono 62/200 points
to run-to-run nondeterminism on the mono path (the standard path's greedy band is 188-190/200).
Determinism (mi355-4, 2026-10-31; one server, two greedy THREADS=1 passes each; results/acc/det_mono_{on,off}_20261031_{a,b}):
mono on 66/200 identical (0.97 / 0.98, accept len 4.48 / 4.81); mono off 108/200 identical (0.985 / 0.99,
4.69 / 4.92), and 99-100/200 across two off servers. The standard path is itself nondeterministic at bs 1
(the 188-190/200 band is a 64-thread number; suspect split-K atomics in small-M GEMMs, row 3), so the
mono-vs-mono 62/200 is not evidence that step 1 changed numerics; the K2 kernel replays bit for bit on fixed
inputs. Mono runs a little less reproducible (66 vs 108) and scores 0.97-0.985 over 4 runs vs 0.98-0.99 over 3:
within 200-question noise; a 1319-question gsm8k (or KL vs mono off on fixed prompts) is the next gate
before calling the c1 accuracy equal.

## 10. S1 accuracy gate and P2a step 1: tile GEMM v0 (mi355-4, 2026-11-01)

### 10.1 gsm8k1319 gate (S1 c1)

`launch_server.sh` with SIM=0 (real draft), `NUM=1319 THREADS=1 TEMP=0 gsm8k_gate.sh`; sglang-k31-amd
origin/dev_amd 3d2aa95c9d, sgl-amd-mono k31/mono-s1 @ 6e50a20.

```text
mono on   1287/1319 = 0.9757   accept len 4.709   wall 610 s   results/acc/g1319_mono_on_20261101
mono off  1288/1319 = 0.9765   accept len 4.697   wall 683 s   results/acc/g1319_mono_off_20261101
paired (evaluator's symbolic_correct): on-only 2, off-only 3, McNemar exact p = 1.0;
identical generations 1246/1319 (paired_vs_off.txt; per-item output-rs0.jsonl copied next to metrics.json)
```
-> S1 accuracy equals mono off. gsm8k_gate.sh fixes: `NUM` knob (default 200); the accept-len window now
compares full date-times (it read 0 lines when a run crossed midnight UTC).

### 10.2 Tile GEMM v0 (`kimi_k31/gfx950/stages/tile.py`)

v0 = PR #13 GEMV generalized: task = 16 nb weight rows x all m rows, the 8 waves split K (gemv's K order),
both operands buffer-loaded straight into MFMA registers (weights NT, x from L2), partials summed through
64 KB LDS in rounds; `emit(row, col, v4, live)` hook (bf16 / f32 stores provided), optional K split across
CTAs (`ks`, f32 partials [ks, m, n] for the consumer to sum). `tile_bench.py`: rel L2 1.7e-3 (bf16 output
rounding) at m 16 / 20 / 128 / 384, nb 1 / 2 / 4, ks 1 / 2 / 4, K tails (1536 = 1.5 chunks a wave).

Graph-replayed, MALL-busting (us; tgemm = server's tuned path):

```text
                 m=16 best (nb,ks)   tgemm | m=128 best          tgemm | m=384 best         tgemm
6288x7168        29.5 (2,1)          18.8  | 111.6 (2,1)          29.9  | 789 (2,1)           68.4
7168x3584        17.5 (2,1)          11.2  |  50.7 (2,4)          21.7  | 304 (2,4)           37.3
 896x3584        10.4 (1,1)           4.7  |  11.1 (1,4)           7.0  |  49 (1,4)           13.0
7168x1536         9.7 (2,1)           6.1  |  26.3 (2,2)          10.1  | 150 (2,2)           18.1
```

Why (ISA via `scripts/tile_isa.py`): (1) LLVM's scheduler sank every prefetch load to just before its MFMA
(66 x `s_waitcnt vmcnt(1)`); `rocdl.sched_barrier(0)` around each unit keeps 36-43 loads in flight but
changed nothing (126.9 vs 129.5 us at m=128 nb=1), so latency is not the limit. (2) Bytes per CU are: the
CTAs pull ~25-31 GB/s each (the same per-CU rate as the 6 TB/s HBM read floor), and total traffic is
(N / NT)(m + NT) K 2 -- x is reread by every task, so at m=128, NT=16-32 the x traffic is 4-8x the weights.
`ks` does not change that sum (it only adds CTAs). Cutting it needs NT >= ~m, which with all 8 waves on K
means m/16 x NT/16 x 4 accumulators a lane: 256 VGPRs (the 2-waves-a-SIMD cap) at m=128, NT=32 already
(m=128 nb=2: 256 VGPR + 408 B scratch; m=384: 1576 B scratch). So v0's structure is a dead end past ~16 rows.

aiter's own hgemm at 8 waves (`scripts/hgemm_8wave_sweep.py`, M=128; only t128x128x64 stages 2 w2x4x1
passes the wrapper's config check at 512 threads):

```text
6288x7168  ks4 37.5 / ks2 48.7 / ks1 81.3   tgemm 29.8      7168x3584  ks4 25.7 / ks1 42.9   tgemm 20.6
6016x7168  ks4 34.0 / ks1 80.9              tgemm 27.9      7168x1536  ks2 18.1 / ks1 21.5   tgemm 10.4
```
tgemm's winners are 16-wave (e.g. t128x128x64x4 ks4 w4x4x1). Without split-K the 128x128 tile leaves
~200 CUs idle (49 tasks for in_proj) -> 2.7x; with split-K 4 (atomics) it is still +25%.

### 10.3 Consequence for S2 (superseded by 10.4: the +25% came from the HTI default)

At 128+ rows a dense GEMM inside a 512-thread launch costs ~+25% (best case, aiter structure + split-K) over
tgemm; KDA layer at c16: in_proj + MoE front + latent up + o_proj ~ 90 us of tgemm a layer -> ~+20 us a layer
if moved in naively, against the S2 estimate of -8..-9 ms/step (section 2, ~-90 us a layer). Options:
(a) tile stage = aiter's structure (LDS async DMA, 128x128, w2x4x1) as an in-launch stage, split-K 4 with f32
    partials that the consumer sums (no atomics / finish; fits the column-sharded RS of section 7);
(b) as (a) in a 1024-thread wide-row launch (16 waves, tgemm's shape; 128 VGPRs a wave for every other stage);
(c) keep in_proj / MoE front / o_proj as tgemm launches, the wide-row launch fuses only what lies between them
    (seams, norms, router / top-k / sort prologue, latent up owner slice): ~3-4 launches a layer instead of 1.

### 10.4 8 vs 16 waves, HTI off (mi355-4, 2026-11-01) -> option (a)

`hgemm_8wave_sweep.py` now sweeps use_half_tile_interleaved (the wrapper's default is True; HTI rejects
k_waves > 1 and most 8-wave tiles, which is why 10.2's sweep saw only one config), block_k 64/128, stages 2-4,
split-K 1/2/4; `WAVES=16` sweeps 1024-thread configs. M=128, us (graph-replayed, MALL-busting):

```text
shape          tgemm | 8-wave best                      ks1 best | 16-wave best              ks1 best
6288x7168      30.1  | 27.9 t128x128x64x4 ks4 w2x2x2    40.7     | 28.3 t128x128x64x3 ks4 w2x8x1  51.6
6016x7168      27.6  | 26.6 t128x128x64x4 ks4 w2x2x2    39.9     | 27.1 t128x128x64x4 ks4 w2x8x1  50.9
7168x3584      20.6  | 18.4 t128x128x64x3 ks4 w4x2x1    23.1     | 18.4 t128x128x64x4 ks4 w2x8x1  28.1
 896x3584       7.0  |  8.0 t128x32x128x3 ks4 w4x2x1    19.5     | 10.5 t128x64x128x3 ks4 w4x4x1  26.5
7168x1536      10.1  | 10.9 t128x64x64x4 ks2 w2x4x1     11.4     | 12.0 t128x64x128x3 ks2 w4x4x1  14.7
sum of the 4 big shapes: tgemm 88.4, 8-wave 83.8, 16-wave 85.8 us a KDA layer
```
(all ks > 1 rows use aiter's atomic split-K; ks1 is 1.3-2.4x slower at the K=7168 shapes, so the in-launch
stage needs a K split -- with consumer-summed f32 partials that is extra traffic, to be measured.)

1024 threads for PR #13's stages (`scripts/mono_threads_isa.py`, compile only, THREADS patched in
common.plan): KDA front 205 -> 103 VGPR, no spill (twice the waves, half the chunks a wave); K2 layer_post
219 -> 128 VGPR with 324-400 B scratch at s = 1 / 4 / 8 (a fixed-size stage, not the row count).
-> (b) buys nothing on the GEMMs and costs spills; (a) is the plan.

### 10.5 Own FlyDSL tile GEMM v1 (`kimi_k31/gfx950/stages/tile_mma.py`, mi355-4, 2026-11-01)

No aiter gemm code: a task is a [bm x bn] output tile over one K slice; 8 waves in a wm x wn grid (no
cross-wave reduction); BK = 64 steps, x and W pieces (16 B a thread) loaded `depth` steps ahead into
registers, written into one of two LDS buffers with rows padded to 72 bf16 (16 rows of a ds_read_b128 hit
16 distinct 4-bank groups), MFMA operands read from LDS; ks > 1 leaves f32 partials. Same `emit` hook as v0.
`test/kernels/kimi_k31/mma_bench.py`: rel L2 ~1e-7 (f32 partials) / 1.7e-3 (bf16), m = 20 / 128.

```text
m=128, us        t128x128 w2x4: ks1   ks2   ks4  | best other              | tgemm | aiter 8w best (10.4)
6288x7168                      74.7  41.9  30.1  | t128x64 w4x2 ks4 33.0   | 29.4  | 27.9
6016x7168                      75.4  41.6  29.2  |                         | 27.5  | 26.6
7168x3584                      40.4  24.3  19.0  | t128x64 w4x2 ks4 21.3   | 21.3  | 18.4
 896x3584                      39.0  22.0  13.7  | t128x32 w8x1 ks8  6.3   |  7.1  |  8.0
7168x1536                      20.8  13.9  11.4  |                         | 10.1  | 10.9
```
Prefetch depth 2 / 4 / 6 changes nothing (in_proj 30.1 / 30.2 / 30.3): not load-latency bound; ks8 is worse
for the big shapes (32.8). The stage excludes the partial sum: a separate torch sum adds 3.8 (in_proj) to
10 us (o_proj, mostly its launch); fused into the consumer only the bytes remain (in_proj 4 x 128 x 6288 f32
= 12.9 MB written + read). aiter's 27.9 uses packed bf16 atomics instead -- the next comparison.

### 10.6 Split-K merge and m buckets (mi355-4, 2026-11-01)

Merge modes (`--mode`): part = f32 partials [ks, m, n] for the consumer; atom / atombf = f32 / packed-bf16 adds
straight from the MFMA layout (a lane adds 4 columns of one row: 16 rows a wave instruction); lin / linbf =
the f32 tile shuffled through LDS (aliasing the x / W buffers) and added lane-contiguous (a wave covers
whole 64 B lines, as aiter does). Masked lanes add at the buffer end (0x7FFFFFFF + 4 j wrapped and faulted).
m=128, t128x128 w2x4 ks4, us: in_proj part 28.7 / atom 59.4 / atombf 41.0 / lin 32.5 / linbf 30.2;
o_proj 11.3 / 50.3 / 28.8 / 18.4 / 12.7 -> linbf: the consumer reads plain bf16 (1.6 MB for in_proj vs 12.9 MB
of partials), rel L2 3.2e-3 (bf16 rounding a K slice, as tgemm's own split-K).

Best linbf config a bucket (us, ours / tgemm; tile, ks):

```text
m     in_proj 6288x7168          latent up 7168x3584        owner 896x3584          o_proj 7168x1536
16    16.4 / 18.7 t16x128 ks4    9.8 / 11.1 t16x128 ks4     5.0 / 4.7 t16x128 ks8   5.8 / 6.1 t16x128 ks4
32    17.6 / 20.5 t32x128 ks4    10.7 / 12.9 t32x128 ks4    4.9 / 5.0 t16x128 ks8   6.5 / 6.3 t32x128 ks4
48    21.5 / 22.9 t48x128 ks4    12.0 / 15.2 t48x128 ks4    5.0 / 5.3 t16x128 ks8   7.6 / 7.5 t48x128 ks4
128   30.2 / 30.1 t128x128 ks4   19.5 / 20.3 t128x128 ks4   6.5 / 7.1 t128x32 ks8   12.7 / 10.2 t128x128 ks4
256   46.3 / 51.5 t256x128 ks4   28.8 / 62.2 t128x128 ks2   10.1 / 9.6 t128 ks8     16.9 / 27.7 t128x128 ks2
384   64.7 / 69.1 t192x128 ks2   37.2 / 36.6 t192x128 ks2   11.6 / 12.9 t128 ks8    22.1 / 18.3 t192x128 ks2
```
Waves: bm 16 / 48 -> w1x8, bm 32 / 128 -> w2x4, bm 192 / 256 -> w4x2; bm = m rounded to 16 up to 48. tgemm's
m=256 entries look untuned (7168x3584 62 us). Repro: `mma_bench.py --time --m 16,32,48 --cfg 16:128:1:8:4:2,...
--mode linbf` (cfg = bm:bn:wm:wn:ks:depth; wm x wn must be 8).

### 10.7 Column-slice collectives (`kimi_k31/gfx950/stages/coll.py`, mi355-4, 2026-11-01)

RS / AG / two-shot AR over column slices (rank r owns columns [r sl, (r + 1) sl), sl = H / 8), on PR #13's
`Mailbox`, `PeerBuffer` and guard. "tag": 16 B stores of 2 bf16 pairs + 2 tags; CTA group j (32 of the 256 CTAs)
pushes to rank j, so a wave's destination base is uniform; consumers poll exactly the pairs they read. "flag":
CTA b owns chunk b of every slice, plain 16 B uncached stores, `s_waitcnt vmcnt(0)` + barrier + a tagged flag a
(phase, source, CTA). Both sum in f32 in rank order: bit-exact against the torch reference, deterministic.
Regions are double-buffered by epoch parity. `test/kernels/kimi_k31/coll_bench.py`: every launch of a 20-launch
graph has its own inputs, 3 input sets rotate between replays (stale data cannot pass).

```text
us a launch (graph replay, incl. the launch gap; 256 x 512 threads), all bit-exact
shape       MB    AR tag / flag   AG tag / flag   RS tag / flag | 6.2 AR: HIP tagged64 / waitcnt / RCCL
8 x 7168    0.11   9.0 / 15.7      5.4 / 12.7      6.1 / 11.2   |   8.3 / 14.6 / 15.8
32 x 7168   0.46  10.1 / 18.3      6.9 / 13.5      7.5 / 11.7   |  10.4 / 16.0 / 28.5
128 x 3584  0.92  14.1 / 18.7      9.6 / 15.1      9.9 / 13.0   |  19.0 / 17.4 / 30.3
128 x 7168  1.84  23.3 / 24.5     14.1 / 19.3     14.5 / 17.5   |  35.3 / 23.1 / 40.9
128 x 10752 2.75  32.9 / 33.4     18.7 / 26.1     19.0 / 23.8   |  60.1 / 31.5 / 42.9
384 x 7168  5.51  59.8 / 63.3     32.8 / 52.1     31.7 / 47.9   | 133.1 / 65.3 / 59.0
```
Findings: (1) 16 B tagged stores with a uniform destination take tagged from 35.3 to 23.3 us at 128 x 7168; it
now ties or beats flag at every size, so section 6.2's rule (tagged <= 0.9 MB, flag above) is replaced: tagged by
default, flag kept for overlapped phases where link bytes matter (half the bytes). (2) The standalone AR beats
RCCL to 128 rows and ties it at 384; AG and RS each cost about half an AR, so section 7's split (RS and AG at
different stage boundaries) costs no more than today's AR before any overlap. (3) Stress (`--stress 2000
--launches 50`, random per-CTA start delay 0..200 us and 0..10 us): AR 8 x 7168 and 128 x 7168, both formats,
8 x 100050 launches, 0 mismatches -> the waitcnt publish ordering holds under skew (still not a documented
guarantee; rerun the stress after ROCm / firmware changes). (4) Open: parity. The bench bumps the epoch every
launch; in production a step's launches share an epoch (PR #13 tag scheme, `kimi_k3/gfx950/tag.py`), so two
consecutive launches using one region need alternating parity from (epoch, slot) -- an odd launch count a step
or a per-launch counter. Safe without parity only if every rank's read of launch L's region happens before a
push the producer needs for launch L + 1.

Wiring (next): with linbf the o_proj / shared-down tile is complete only after its last K slice; a per-tile
arrival counter lets that task read the tile, RS-push it and zero it (decision (a)). in_proj takes x by K
chunks from the AG region as the slices land; the AR inside T1 (latent) uses `tag_reduce` with an emit that
feeds the latent norm instead of a store.

Repro: `cd /tmp && torchrun --nproc-per-node 8 /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/coll_bench.py
--time` (add `--shapes 8x7168,128x7168 --kinds ar --launches 50 --stress 2000 --delay 20000` for the stress).
Results: `/shared_nfs/kimi-k31/results/coll_slices_20261101/{summary.txt,sweep.log,coll_stress_*.log}`.

### 10.8 Fused tile GEMM -> RS (`kimi_k31/gfx950/stages/gemm_coll.py`, mi355-4, 2026-11-01)

One launch: GEMM tasks (tiles x K slices, a tile's slices consecutive tasks of one round), each slice merges
into a local scratch tile at device scope (CM_DEV: the reader may sit on another XCD -- plain stores failed
with "part" until then; linbf atomics were already coherent), counts itself in a per-tile counter (monotonic,
+ks a launch, no reset), waits for the tile's other slices, pushes 1 / ks of the rows to the column owner and
zeroes them (decision (a)); after their GEMM tasks the CTAs reduce the owner slice piece by piece in rank order.
`test/kernels/kimi_k31/gemm_coll_bench.py`: rel L2 vs the f32 reference on rotating inputs, "part" replays bit
for bit; times fused | tile GEMM + memset + coll RS (3 launches) | F.linear + RCCL reduce_scatter.

Where the time went (m = 128, t128x128 ks4, linbf; GC_PHASES): GEMM 14.6 -> + arrive 15.8 -> + tile loads 18.2
-> + remote pushes 35.0 us when the last slice pushed the whole 64 KB tile alone: one CTA pushes only ~2-4 GB/s
of remote stores (same per-CTA rate as the standalone RS), so the push must be spread over many CUs. Every
slice pushing its rows: + push 12 us, i.e. ~3.2 MB of tagged bytes a rank at ~270 GB/s, near the links' limit
-> the "flag" push (plain 16 B stores, one flag a (rows, tile) piece, owner CTA waits for the 8 sources' flags of
a piece) halves the bytes: + push 5.8 us. ks = 1 with smaller tiles removes the wait, the atomics and the
zeroing, and wins at 128 / 384 rows.

```text
o_proj 7168 x 1536, flag push, us a launch    fused   (GEMM alone)  3 launches  F.linear + RCCL
m 16  t16x128  w1x8 ks4  linbf / part         13.6 / 14.7   (7.2)     13.7        17.4
m 32  t32x128  w2x4 ks2  linbf / part         16.2 / 16.5   (9.8)     17.4        18.8
m 128 t64x64   w2x4 ks1                       23.1          (13.2)    25.7        32.1
m 384 t192x64  w4x2 ks1                       43.2          (23.1)    52.8        46.6
```
At 16 / 32 rows tag and flag tie (13.3 / 16.3 tagged). The fused GEMM runs ~1-2 us slower than the standalone
tile GEMM (guard, context, persistent loop). Push + reduce still add ~6 us (m 16) to ~20 us (m 384) over the
GEMM: inside the S2 launch this is what overlap with the next stage has to hide. Not yet stressed with random
delays (the coll stress covers the transport; the counter wait is new).
Results: `/shared_nfs/kimi-k31/results/gemm_rs_fused_20261101/{summary.txt,final.log,*sweep.log}`.

### 10.9 Fused AG -> tile GEMM (`gemm_coll.ag_gemm`, mi355-4, 2026-11-01)

One launch: every rank pushes its x slice [m, 896] to all ranks' gather region (plain 16 B stores, CTA group j
feeds rank j); each CTA counts itself in a local counter (+BLOCKS a launch, no reset) and the last one flags
"source r landed" on every rank; a GEMM task waits only for the owners of its K range (2 at ks 4) and reads x
straight from the gather region. `test/kernels/kimi_k31/ag_gemm_bench.py` (linbf; every column includes the
output memset):

```text
in_proj 6288 x 7168, us    fused   (GEMM on the gather region, no push / wait)   coll AG + tile   RCCL AG + F.linear
m 16  t16x128  w1x8 ks4    23.8    (19.9)                                         25.9             35.1
m 32  t32x128  w2x4 ks4    25.8    (20.3)                                         26.1             37.2
m 128 t128x128 w2x4 ks4    39.6    (32.7)                                         46.2             58.5
m 384 t192x128 w4x2 ks2    84.7    (76.1)                                         99.0             98.8
```
rel L2 2.5-3.2e-3. The table-10.6 configs stay best (ks8, t64x128, t128x64 at 128 rows and ks4 / ks1 variants
at 384 rows are 7-24 us worse). Push + wait add 4-9 us over the GEMM: every task waits at its start, so only
skew between owners overlaps. The gather region is uncached peer memory, so x is re-read from memory by each of
the 50 N tiles: ~+10 us at 384 rows vs the standalone tile GEMM (64.7). Options if it matters: K steps of a task
ordered to start on owners already landed; a local cached copy of x by the consumer.
Results: `/shared_nfs/kimi-k31/results/ag_gemm_fused_20261101/`.

### 10.10 Buffer parity across a step's launches (mi355-4, 2026-11-01)

A step bumps the epoch once and runs its layer launches (slots) on one peer buffer. With the half picked by the
epoch (the K3 scheme), a rank that finishes launch n can start n + 1 and write slices into the half a slow peer
still reads for n. Reproduced (`coll_bench.py --kinds ag --step 4 --parity epoch --stress 300 --delay 5000`,
delays also before the final read): flag format 6265 silent mismatches in 14k launches; tagged format hangs
(the next tag never matches) -> guard stop on every rank. Fix (`coll.context`): half = (epoch x launches + slot)
% 2, the global launch index, so consecutive launches alternate across step boundaries too. Two halves suffice
only if every launch has an exchange in which each rank waits for every other (then no rank reaches n + 2 before
all have left n) -- a rule for the planner. Stress with launch parity: ag / ar x tag / flag x 8 / 128 rows, steps
of 4, 8 x 48048 launches, + steps of 3 (odd) 2 x 48048: 0 mismatches. Fused kernels with random CTA delays (before
each arrival in gemm_rs; before / after the push in ag_gemm), steps of 4, 48000 launches each: gemm_rs 8 configs
(ks 4 / 2 / 1, linbf / part, flag) worst rel L2 3.7e-3, ag_gemm 6 configs 3.2e-3 (part 9e-7). Timing unchanged.
PR #13's K1 / K2 keep epoch parity but use tagged pairs only, so the same race there would be a guard stop, not
wrong data; worth an audit of which of its regions are read after the reader's last awaited push.
Results: `/shared_nfs/kimi-k31/results/parity_stress_20261101/`.

### 10.11 Planner v0 and the post-attention chain (mi355-4, 2026-11-01)

`kimi_k31/plan.py`: a launch = stages every CTA walks in order (task t on CTA t % 256), joined by mailbox words /
flags only, no grid barrier; deadlock-free with every CTA resident and waits pointing backwards. `Plan` lays out
named peer regions (two halves) and local scratch, lists stages with their waits, and checks (waits backwards,
one all-ranks stage for the parity rule). `coll.region_context` gives any planned region `<name>_of` / `_own`.

`gfx950/chain.py` (T4 -> T5 -> T6): oproj (gemm_rs tiles, flag push) -> rsred (owner piece sum in rank order +
residual, new residual slice, piece mark in scratch) -> seam (a row a task: 8 waves split the 896 columns, sum of
squares exchanged as tagged f32, RMSNorm, normalized slice pushed to every rank, row flag) -> front (2096 x 7168
tiles waiting for their rows' and owners' flags, x' read from the gather region). v0 ran the seam a band (64
rows) a CTA: 108 us at m = 128, one CTA pushing 0.9 MB; a row a task: 55 -> 48 us with the front at ks 8.
`test/kernels/kimi_k31/chain_bench.py` checks the new residual slice and the front output vs an f32 reference:

```text
us a launch (steps of 4)   o_proj cfg        front cfg          chain linbf / part   standard   rel L2 res / front
m 16                       t16x128 ks4       t16x128 ks8        27.6 / 26.1          43.0       3.7e-3 / 5.7e-3
m 32                       t32x128 ks2       t32x128 ks8        31.4 / 29.7          58.6       3.2e-3 / 5.3e-3
m 128                      t64x64  ks1       t128x128 ks8       48.3 / 45.5          76.1       2.5e-3 / 4.9e-3
m 384                      t192x64 ks1       t128x128 ks4       91.7 / 89.5          110.2      2.5e-3 / 4.2e-3
```
linbf includes the front output memset; "part" leaves f32 partials for the consumer (its sum is not counted).
16k launches a config re-checked clean (no random-delay stress in the chain yet). The standard column is torch
ops, not SGLang's fused add-rmsnorm + custom AR; the endpoint A/B is the real comparison.
Results: `/shared_nfs/kimi-k31/results/chain_post_attn_20261101/`.

### 10.12 Chain timeline (mi355-4, 2026-11-01)

Timeline build (`chain.post_attn(..., timeline=True)`, PR #13's LDS stamps): per CTA the clock at the start, each
stage's first wait done and its end; `chain_bench.py --timeline` shows the last launch of a step behind warm
launches. The GPUs' s_memrealtime clocks are NOT aligned on mi355-4 (hundreds of us apart): each rank is shown
against its own earliest start. Medians, us (m = 128, o t64x64 ks1, front t128x128 ks8):

```text
oproj first tile 9.0 | oproj pushed 14.7 (last 17.5) | RS pieces in 17.2, done 18.8 | seam pieces 20.6,
sum-of-squares exchange 23.0, pushed 27.8 | front x' in 29.3, done 41.1 (launch 43-44 per rank; 48.5 a launch
with the memset and the gaps). m = 16: 3.9 / 8.6 / 11.4 / 16.8 / 17.5 / 23.5. m = 384: 17.5 / 33.2 / 39.3 / 56.9
/ 63.3 / 85.5.
```
Each stage is close to its standalone cost and they do not overlap (the front waits for every row's x'). Tried at
m = 16 / 128 / 384 (us): front tasks on the top CTAs 28.0 / 48.5 / 90.4 vs 28.0 / 48.5 / 92.0; + W prefetch into
L2 during the wait 27.1 / 48.7 / 96.7 (no gain; slower at 384); timing probe with x' from a cached local buffer:
front 11.6 us at 128 rows, unchanged -> neither W nor the uncached x' read bounds the front. The front alone
(mma_bench 2096 x 7168) is 16.5 us at 128 rows (tgemm 18.4), 11.5 in the chain: it is the tile GEMM's
~2.6 TB/s. Defaults: front_top on, prefetch off.
Results: `/shared_nfs/kimi-k31/results/chain_timeline_20261101/`.

### 10.13 AttnRes seam on column-sharded state (`stages/attnres.py`, mi355-4, 2026-11-01)

SGLang (`layers/attn_residual.py`, `aggregate_stream_torch` + out_norm, `AttnResidual.forward`): prefix = previous
prefix + the sublayer's output; sources = bank rows 0 .. nvb - 1 + the prefix; score_r = rsqrt(mean(x_r^2) + eps)
(x_r . cw), cw = score_norm.weight x score_proj.weight; softmax; mixed = sum p_r x_r (bf16); out = RMSNorm(mixed);
at a block boundary the prefix becomes bank row nvb. So the attention-side (T2) and the MLP-side (T5) seams are
the same op; the chain's earlier "residual add + RMSNorm" was its nvb = 0 case.

Column-sharded: a rank holds its 896 columns of every source (bank [m, nbk, 896] per rank). Every full-row sum is
8 partials; all travel in ONE tagged round a row: R dots + R (R + 1) / 2 Gram entries (R = nvb + 1, 54 scalars at
nvb 8), since sum(mixed^2) = sum_rs p_r p_s G_rs (skips the bf16 rounding of mixed before squaring; below output
rounding). A row a task, 8 waves split the columns, wave_sum + LDS per scalar.
`chain_bench.py --nvb N [--write]` vs the eager reference (scores rounded to bf16 as the bf16 Linear does; the
kernel keeps f32 -> front rel L2 grows with nvb: 5.7e-3 at 0, 7.5e-3 at 8, threshold 1e-2):

```text
chain us (linbf)   nvb 0    nvb 1    nvb 4 + write    nvb 8
m 16               28.6     28.8     31.5             36.9
m 128              48.6     49.7     52.3             57.6
m 384              95.0     95.6     99.3             110.2
```
Bank write checked bit-exact against the new prefix. The "standard" column of chain_bench is torch eager
AttnRes (48 -> 106 us at m 16 as nvb grows): not SGLang's fused HIP aggregation (~8 us a call at c16, section 1)
-- use the endpoint A/B for the real comparison. Open: the +9 us at nvb 8 is the per-scalar wave_sum chain (54
butterflies) + 9 source loads a lane; a lane-transposed reduction or 2 rows a task would cut it.
Results: `/shared_nfs/kimi-k31/results/chain_attnres_20261101/`.

### 10.14 Pre-attention chain: MoE tail + AttnRes + in_proj (`gfx950/pre_attn.py`, mi355-4, 2026-11-01)

SGLang's tail (`kimi_k3.KimiK3MoE._column_split_tail`, K3.1: `latent_moe_use_norm` true, eps 1e-5, routed scaling
1.0): latent = RMSNorm_n(AR(latent partials)); y = AR(shared partials + latent @ W_up[cols]^T on each rank's
cols); prefix' = prefix + y. Column-sharded stages: push ((row, owner) tasks: 448 latent + 896 shared columns
to the owner, a flag a (src, row)) -> lred (row: rank-order sums; latent slice + its sum of squares to every
rank, shared sum f32 in local scratch) -> up (896 x 3584 tiles over the gathered latent; norm weight folded into
W_up's columns at load (`fold_up`), the row rsqrt(mean square) applied in the epilogue; K slices merge linbf,
arrive, each finishes its rows: y = bf16(up + shared), prefix' = bf16(y + prefix), zero the tile) -> seam
(`attnres_row` with the attention-side weights, x to every rank) -> in_proj (6288 x 7168, chain's front form).
`test/kernels/kimi_k31/pre_attn_bench.py` (reference in SGLang's op order, f32):

```text
us a launch (steps of 4)   up cfg           in_proj cfg       nvb 0   nvb 4 + write (linbf / part)   torch standard
m 16                       t16x128 ks8      t16x128 ks4       38.2    41.7 / 40.6                    74.8 / 128.0
m 128                      t128x32 ks8      t128x128 ks4      64.8    68.2 / 64.3                    122.4 / 210.6
m 384                      t128x32 ks4      t192x128 ks2      134.5   141.6 / 138.8                  192.1 / 376.7
```
rel L2: prefix' 3.5-3.9e-3, in_proj 4.4-6.4e-3; bank write bit-exact. Alternatives tried (in_proj ks8, up t128x64
/ ks8 at 384) are equal or slower. The standard column is torch ops (eager AttnRes), not SGLang's fused kernels.
Bench note: PeerBuffer close must be fenced by barriers on every rank before the next case allocates
(hipIpcGetMemHandle "invalid argument" when a freed address comes back while still imported by a peer).
Results: `/shared_nfs/kimi-k31/results/pre_attn_20261101/`.

### 10.15 KDA stage in the pre-attention launch (mi355-4, 2026-11-01)

`pre_attn(..., kda=L, state_len, ring=True)`: in_proj (6288 x 7168) tiles merge linbf in a bf16 buffer, arrive on
a per-tile counter, and each K slice converts its rows to the tagged f32 pairs `kimi_k3`'s KDA tasks poll
(scratch "proj", [S][6288]) and zeroes them; then `kimi_k3.gfx950.attention.kda.kda_tasks` unchanged (conv, f_b
gate MFMA, l2norm, recurrence on the ReplaySSM ring, gated RMSNorm -> core_out [S, 1536]); its ~24 arguments
packed in one int64 table (`pre_attn.kda_table`). `pre_attn_bench.py --kda 8` (R = m / 8 requests, num_acc 1..8,
ring rows written, committed state read-only): reference = `test_kimi_k3_k1._kda_reference` vectorized over
requests.

```text
m 16 (R 2):    53.1 us (38.2 without KDA)   core_out rel L2 1.04e-2 end to end, 4.2e-3 KDA alone
m 128 (R 16): 123.5 us (64.8 without KDA)   core_out rel L2 1.00e-2 end to end, 3.9e-3 KDA alone
```
"KDA alone" feeds the reference the rows the stage read (from the scratch pairs): the 4e-3 is K1's own bf16
roundings (conv out, g1, o); end to end the in_proj rounding (5e-3) is amplified by l2norm / gates / recurrence
(threshold 3e-2 end to end, 5e-3 alone). The cost: K1 splits a (request, head) into 8 v-chunk tasks and every
chunk recomputes the head's gate MFMA, q / k l2norm and conv: 1536 tasks at m = 128 (6 rounds) with 8x
redundant front work -> the wide KDA task (Next 1).
Results: `/shared_nfs/kimi-k31/results/pre_attn_kda_20261101/`.

### 10.16 Wide KDA task (`stages/kda_wide.py`, mi355-4, 2026-11-02)

One task a (request, head), the head in one CTA (K1's math and rounding; ReplaySSM ring only, the committed state
read, never written): conv a channel a thread (384 threads: q|k|v x 128), f_a / beta polled once, the gate MFMA
once (wave w rows 16 w ..), q / k l2norm, the 128 x 128 state in registers (KPT K columns a thread, 128 / KPT
lanes a row, 32 / KPT rows a thread), gated RMSNorm a wave a token. 192 tasks at m = 128 (one round) vs K1's 1536.

```text
pre_attn + KDA, us (KDA alone rel L2 3.9-4.3e-3 at every KPT)   m 16    m 128
K1 tasks (8 a head)                                               53.1    123.5
wide, KPT 4 (32 lanes a row, 8 rows a thread)                     61.8     93.8
wide, KPT 8                                                       55.3     86.7
wide, KPT 16 (8 lanes a row, 2 rows a thread; default)            54.4     86.1
```
At m = 16 only 24 tasks run (K1's 192 spread better; equal within 1.3 us, so one path). The KDA tasks wait for
all of in_proj (a head's q / k / v / g / f_a / beta columns span the whole N range), so the stage does not
overlap in_proj; its 21 us at m = 128 is one round of tasks. Results: `/shared_nfs/kimi-k31/results/pre_attn_kda_20261101/`.

### 10.17 One wide-row KDA layer launch (`gfx950/layer.py`, mi355-4, 2026-11-02)

`kda_layer(m, LayerCfg(up, in_proj, o_proj, front), L, nf, nvb_a, nvb_m, write_a, ...)`: `pre_attn`'s stages
(push, lred, up, seam, inproj) + the wide KDA heads + `chain`'s (oproj, rsred, mseam, front) in one launch, one
`Plan` (post regions prefixed `o_`), two `region_context`s on one peer buffer with one tag. In-launch hand-offs at
device scope (reader may be on another XCD): core_out (KDA marks each (request, head) done in scratch; o_proj
tiles wait for their rows' heads, x loads CM_DEV via `mma_tile(..., xcm)`), the prefix after the MoE tail (RS
residual input, CM_DEV), the bank row the attention seam writes (MLP seam reads it CM_DEV).
`test/kernels/kimi_k31/layer_bench.py` (f32 reference in SGLang's op order end to end):

```text
m (R x 8)   layer us   rel L2: mid      h        core_out   front     (configs: layer_bench.CFGS)
16          82.4       3.9e-3   5.5e-3   1.0e-2     7.0e-3
32          89.7       3.9e-3   5.4e-3   1.0e-2     6.9e-3
128         135.8      3.9e-3   5.4e-3   1.0e-2     6.9e-3
384         272.1      3.5e-3   4.9e-3   8.9e-3     6.0e-3
16 / 128, nvb_a 3 + write, nvb_m 4: 87.2 / 140.6, all pass
```
The launch costs the sum of its halves (pre + KDA 54 / 86, chain 28 / 49 at m = 16 / 128): the stages are serial
(10.12), merging saves the gaps only. What it replaces at c16 (section 6.1, per KDA layer): 446 us of kernels of
which ~210 experts -> ~236 us vs 136 here, ~-100 us a layer x 69 = ~-6.9 ms a step, before the T6 tail (router
top-16, latent quant, shared down) is added and before endpoint confirmation.
Results: `/shared_nfs/kimi-k31/results/kda_layer_20261102/`.

### 10.18 MoE front tail in the layer launch (`stages/moe_front.py`, mi355-4, 2026-11-02)

SGLang's boundary kept: the routed experts take (latent bf16 [m, 3584], topk ids / weights [m, 16]); sorting,
mxfp8 quant and the two stages stay aiter's, between layer launches. K3.1 routing (config): sigmoid, `noaux_tc`,
1 group, top 16, renormalize, scaling 1.0, `e_score_correction_bias`. A rank's front rows: 112 router | 448
latent down | 768 shared gate | 768 shared up (SGLang replicates latent down; here 1/8 a rank + AG). The front
merges in f32 (`lin`; router logits are fp32 in SGLang), then per finished (tile, row band): router columns ->
every rank's [m, 896] f32, latent -> every rank's [m, 3584] bf16 (flag (src, band, tile)), shared -> local scratch
(mark). Row tasks: route (copy the gathered latent into lat_out; top-16 in one wave: 14 experts a lane, 16
rounds of a wave argmax, ties to the lower expert; weights renormalized), SiTU (beta 4, linear beta 25, bf16 in /
out), then shared down tiles (7168 x 768, ks 1) -> the shared partial bf16 [m, 7168].

```text
layer us (with the tail / without)   m 16 105.3 / 82.4   m 32 114.1 / 89.7   m 128 167.1 / 135.8   m 384 335.2 / 272.1
rel L2: latent 5.6-6.8e-3, shared partial 8.5-10.2e-3, topk weights 5-7e-4 (rows with the same set)
top-16 sets differing from the f32 reference: 10-12% of rows; on the launch's OWN gathered logits: 0 rows, w 1e-7
```
The end-to-end top-16 flips are the upstream rounding (logits ~N(0, 1), the 16th / 17th keys of 896 ~1e-3 apart,
logits off by ~7e-3): the routing code is exact; whether the flip rate matters is the endpoint gsm8k gate's call.
Results: `/shared_nfs/kimi-k31/results/kda_layer_20261102/tail_*.log`.

### 10.19 DSpark capture from the wide layer launch (mi355-4, 2026-11-02)

SGLang (`kimi_k3.py` bank path, `_dspark_capture_stream`): the capture after layer i is `aggregate_stream(prefix
after layer i, bank, nvb of layer i + 1, layer i + 1's self_attention_res_{proj,norm})` -- the pre-norm mixture
the next attention seam forms, bf16 [m, H] on every rank. DSpark's targets (`dflash_config.target_layer_ids`):
0-based 7, 23, 51, 67, 83; the next layers 8, 24, 52, 68, 84 are all KDA (DSA at 0-based 3, 7, .., 91, 92), so
each capture is the layer launch of layer i + 1 with `capture=True`: `attnres_row(..., emit_mixed=...)` pushes the
mixture slice to every rank's "cap" region (+ a (src, row) flag), and a row stage copies the gathered row into
`cap_out` with row stride `cap_ld` (a column block of the packed [m, 5 x H] aux buffer).
`layer_bench.py --capture` (slot 2 of 5; the other blocks must stay zero):

```text
m 16:  106.1 us (105.3 without)  capture rel L2 3.84e-3   | nvb_a 3 + write: 110.6 us, 5.05e-3
m 128: 172.5 us (167.1 without)  capture rel L2 3.84e-3   | nvb_a 3 + write: 176.4 us, 5.25e-3
```
Results: `/shared_nfs/kimi-k31/results/kda_layer_20261102/capture*.log`.

### 10.20 Real-model check of the wide-row layer launch (`kimi_k31/wide_check.py`, mi355-4, 2026-11-02)

`SGLANG_AMD_MONO_K31_WIDE_CHECK=<layer>` (installed from the kimi_k31 spec's build; eager server): wraps
`AttnResidual.forward` (both seams of the layer: its bank, nvb, write flag, and the (normed, prefix) it returns),
o_proj's input (core_out), the experts' inputs (latent, topk_output incl. router_logits). At the attention seam
it runs `kda_layer(rows, ..., t1=False)` from the seam's prefix with: the bank's column slice, the KDA state as
K1 reads it but for R requests (conv window staged a pool slot, `st_idx` = the request's slot, num_acc 1, ring
rows on copies, the committed state read only), real weights (front = gate rows 112 r .. | latent down rows 448 r
.. | shared gate_up; cw = res_norm x res_proj). SGLang serves every step itself. Layer 8 (KDA, nvb 1), A/B config
(DCP8, DSPARK, ring), 4 / 16 concurrent requests:

```text
rows (R x 8)   x        core_out   x'       h        latent   logits   top-16 rows differing (w on equal sets)
32             2.8e-3   5.8-6.5e-3 3.4-4.1e-3 3.0-3.2e-3 4.8e-3 2.4-2.7e-3  0-12.5% (1.9-2.3e-3)
128            2.8e-3   6.1-6.4e-3 3.7-3.9e-3 3.1e-3   4.7e-3   2.7e-3   7-8% (1.9-2.4e-3)
routing: the noaux_tc rule on SGLang's logits = SGLang's ids (0%); ours = the rule on our logits (0%)
```
The weight / layout conventions (in_proj rows, KDA weights, ring, o_proj, both seams, front rows, bias) hold on
the real model. Not covered yet: T1 (layer 7's MoE tail) and the shared expert output.
Second run (widechk4, T1 on, inputs from layer 7's fused front: `_forward_fused` prefix_sum, `_forward_routed`
latent partial, `_forward_shared` shared partial; the decode path calls neither the shared MLP module nor
`routed_expert_down_proj`): mid (T1) 2.3-2.5e-3, x 3.7e-3, core 6.1-6.6e-3, x' 4.0-4.5e-3, h 3.7-3.8e-3, latent
5.0-5.2e-3, shared 5.3-6.0e-3, logits 2.5-3.1e-3, top-16 sets differing 6-13%, routing rule 0% / 0%. Logs:
`/shared_nfs/kimi-k31/results/wide_check_real_20261102/`.

### 10.21 Serving wide verify steps on the layer launch (eager v0, mi355-4, 2026-11-02)

`kimi_k31/wide.py` (`SGLANG_AMD_MONO_K31_WIDE=1`): the spec adds widths 8 R (R 2 .. 48); `_K31Built._backbone`
sends rows > 8 to `Wide.backbone`. A step: standard layers (layer 0 = dense MLP, the DSA layers) run SGLang's own
forward; each run of covered layers (KDA + latent MoE: 68 layers, runs of <= 3 between DSA layers) goes:
`fold()` the stream -> mid / bank column slices; per layer: `kda_layer` (t1 from the 2nd layer of a run,
`restart` on block-write layers 12, 24, .., 84 = always a run's first layer, capture when i - 1 is a DSpark
target), KDA state as K1 but per request (conv window staged a pool slot, `st_idx` = slot, padded requests -1,
ring rows written into SGLang's ring, `intermediate_conv_window` filled after the launch = K1's finish for R
requests), a written bank row all-gathered into SGLang's bank (`res.write`), SGLang's `_forward_routed` on the
launch's latent + router logits (`logits_out`); leave: all-gather the prefix slice, SGLang's
`_column_split_tail`, `set_written`. Rows pad to buckets 16 / 32 / 64 / 128 / 256 / 384; tags: one epoch bump a
step, slot = the launch's index among the 68, parity = launch index.

Bugs found on the way (all fixed in 581b173): (a) `self.backbone = model.backbone` shadowed `Wide.backbone`, so
the "wide" path silently ran SGLang's model forward -- outputs looked right while nothing of ours ran; check
that the first-step log appears. (b) lazy first-use compile skewed the ranks past the 30 s guard deadline ->
guard stop (fault at 0x0): now every non-capture variant is warmed at build, deadline 900 s
(`SGLANG_AMD_MONO_K31_WIDE_DEADLINE_S`). (c) variants sharing one peer buffer each placed their halves and the
guard's flags differently, so one launch's data landed on another variant's flag words -> peers saw a
"failure" and stopped (fault at 0x0): `kda_layer(..., half_bytes=)` now forces one layout a bucket. (d) GPU
core dumps (8 x 17 GB a crash) filled the root disk; delete `/tmp/gpucore*` after a fault. (e) FlyDSL: a
dynamic `if` body cannot create or use a buffer resource (`as_dsl_value ... llvm.ptr<8>`): do such stores
outside it. Run: warmup 46 launches (buckets 16, 32) 295 s cold; 4 concurrent requests answered 42 / 44 / 46 /
48 with steps of batch 3 and 4 on the wide path. Request script `/tmp/wide_req.py`: N threads posting
`/generate` with `{"text": "Q: What is {12+i} + {30+i}? ...\nA:", "sampling_params": {"max_new_tokens": 40,
"temperature": 0}}`. Logs: `/shared_nfs/kimi-k31/logs/server_wide9.log` (success), `server_wide6/8.log` (bugs
b / c).

### 10.22 Accuracy, CUDA graphs, A/B, the bucket-384 fault (mi355-4, 2026-11-01)

gsm8k200, TEMP 0, THREADS 16 (R up to 16 -> rows <= 128), SIM=0 real draft, `--enable-amd-monokernel`,
buckets 16,32,64,128; off = same server without `SGLANG_AMD_MONO_K31_WIDE`
(`results/acc/wide_{gsm,graph}_{on,off}`):

```text
                 score  accept len (req-weighted)  wall
eager  wide on   0.99   5.005                      170 s (incl. lazy capture-variant compile)
eager  off       0.98   5.022                       63 s
graph  wide on   0.98   4.955                       25 s
graph  off       0.99   5.019                       26 s
```
Identical generations pairwise 91..103 / 200 for ALL pairs, incl. off-eager vs off-graph (91): at 16-way
concurrency batch composition dominates; the wide path adds no visible divergence. Gate passes.

CUDA graphs: SGLang's full-graph backend runs two eager forwards per shape (barrier'd) before capturing, so
lazy work there is fine. Changes: KDA table built once per (bucket, layer) on first (eager) use, pointing at
per-bucket persistent `st_idx` / `num_acc` (step: `fill_(-1)` + slot copy; assert not capturing on build);
`epoch[0].add_(1)`; `wide_widths()` capped at max(`SGLANG_AMD_MONO_K31_WIDE_BUCKETS`) / 8 so no graph meets a
cold bucket (R above it -> standard graphs). Log: "runs target_verify steps of batch size [1..16] (16 of 27
captured decode graphs)"; 4 wide_req answers correct.

A/B (ab_bench, SIM acc 3.84, buckets 16..256 -> c48 steps with R > 32 on standard graphs),
`results/ab/wide_ab_on256_20261101`: c4 TPOT 8.72 / c16 16.61 / c48 31.35 ms, prefill 8k 527.6 ms.
Off control same day (`results/ab/wide_ab_off_20261101`): c4 7.72 / c16 15.70 / c48 31.28, prefill 527.2
-> wide is SLOWER: c4 +13%, c16 +5.8%, c48 +0.2% (accept len equal 3.82-3.85). Next: profile a c4 wide step. Doc baselines (other days):
c4 7.78-7.94, c16 15.58-16.18, c48 31.34-31.94.

Faults:
(1) bucket 384, deterministic: the warmup's first variant faults on every rank (addr ~0x6000_0000_3000).
Bench repro (8 GPUs, ~70 s; run from /tmp so cores land there):
`torchrun --nproc-per-node 8 test/kernels/kimi_k31/layer_bench.py --rows 384 --nvb-a 3 --write-a --nvb-m 4 --no-t1`
Needs t1=False (a run's first layer) + nvb_a 3 write + the 384 CFG; no fault with: t1 on, nvb 0, m 128,
restart alone, up ks 8 instead of 4 (same plan sizes, only up task count 336 -> 672 = CTA task distribution),
in/o bm 128. With t1 off the up GEMM does not run, so ks only moves task scheduling -> suspect a race exposed
by timing (T2 starts without waiting), not a static OOB.
(2) bucket 128 warmup, once in 3 cold starts, one GPU (addr 0x7385f8304000); rerun clean. Not chased.

### 10.23 Rebase onto DFlash2 dev_amd; mono S1 K1 breaks under DFLASH (mi355-4, 2026-11-02, FIXED)

sglang-k31-amd -> rewritten `origin/dev_amd` b588cf7952 (detached; untracked skinny-GEMM files kept). The
`kimi_k3.py` changes (A4W4 MoE env, shared RS overlap on the row-sharded prefill tail) do not touch mono's
contract. Mono fix: kimi_k31 spec accepts `speculative_algorithm` DFLASH (was refused); DFLASH taps
[7, 23, 51, 67, 83, 91] reuse the DSpark bank-path captures (`set_dflash_layers_to_capture` ->
`set_dspark_layers_to_capture`, no +1 shift). Draft
`/shared_nfs/RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024` (DFlash2, 5 SWA-1024 MLA layers, block 8).
Launch: `launch_server.sh <tag> --enable-amd-monokernel --speculative-algorithm DFLASH
--speculative-draft-model-path <draft>` (later args override the script's DSpark ones).

gsm8k200 TEMP 0 THREADS 16 SIM=0 (`results/acc/dflash_*`): wide ON 0.985 (accept 4.67 req-weighted), no mono
0.98 (4.96). Mono ON + wide OFF FAILS every time (5/5): GPU faults in DSA indexer kernels
(`indexer_k_quant_and_cache`, `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`, addr 1-2 pages past a buffer
end) and, without a fault, requests contaminate each other (4 concurrent greedy requests answer each other's
numbers) and gsm8k collapses (acc ~2%, accept len 2.00).
Bisection: no mono OK; DSpark + mono S1 on the new code OK (1 then 2 x 4 requests); THREADS=1 (all steps S1,
R = 1) OK (0.95 / 40, accept 5.19); S1 steps of request A then a prefill of request B -> fault; mono with K1
forced to decline (K2 still on) OK. => K1 (S1's KDA launch) under DFLASH. Checked and ruled out: K1's slot /
SGLang commit slot agree; K1 hand-off row 0 of a 49-row `intermediate_conv_window`; ring shapes rawk / rawv / g
(69, 49, 12, 16, 128), beta (69, 49, 12, 16) = what K1 assumes; conv staging zero-stride; `SGLANG_AMD_MONO_CHECK`
on one request: 1608 checks, 0 violations. Open leads: tap 91 (DSA layer, next layer 92 DSA) is the only
target-side tap DSpark lacks; K1's bank bookkeeping (`attn_res._count_write(None)`) vs what the tap reads.
Mitigation candidate: K1 declines under DFLASH (verified on 2 x 4 requests only).
Temporary debug code (marked `# DEBUG-K31`, remove before any commit): sglang-k31-amd `dsa_indexer.py`,
`hybrid_linear_attn_backend.py`; sgl-amd-mono `kimi_k3/gfx950/k1.py` (env SGLANG_DEBUG_SLOTS / SGLANG_DEBUG_NO_K1
/ SGLANG_DEBUG_INDEXK). Launch with `ulimit -c 0` (core dumps filled `/` twice).

**Root cause (mi355-4, 2026-11-02 09:45): K1 overran the runner's shared scratch.** `kda.scratch_layout` puts the
`part` / `wgt` pair regions first, sized `NA * S * (nblocks + 1) * 2` and `S * (nblocks + 2)`, and the plain bf16
`x` rows (`S * HIDDEN * 2` = 114,688 B) last. `k1.scratch_bytes()` sized it for `nblocks=1` (738,432 B, runner
738,560 after rounding), while K1 layers are built with `nblocks = layer.prev_valid_blocks` up to `MAX_BLOCKS` (8):
need 745,664 at 2 blocks, 789,056 at 8, i.e. 7 KB .. 50 KB of `x` rows written past the buffer end, every K1
layer, every S1 step. How it was found (eager, rank 0, temporary debug code since removed):
```text
fault-step out_cache_loc (DSA indexer)  garbage on all ranks alike (e.g. 6.5e18, -4.3e18 as int64): bf16 bits
req_to_token snapshot per layer          unchanged through S1 steps -> not the source
KV allocator free list per layer         PagedTokenToKVPoolAllocator.free_pages head overwritten after every
                                         K1 layer: 855 entries from layer 13 (nblocks 2), 1759 from layer 25
                                         (nblocks 3) = the 7,104 / 14,336 B overrun minus equal words
address check                            runner.scratch 0x..e1000000 .. 0x..e10b4500, free_pages storage at
                                         0x..e10b4600: same 2 MB caching-allocator segment, right after it
core_out canary / bank rows              clean (K1 writes only bank block write_idx, no core tail)
```
Why only DFLASH: the overrun is config-independent; what sits after the scratch is not. With the DFLASH worker's
allocation order the KV free list lands there, so a later `alloc` hands out garbage page ids (fault) or pages
another request holds (cross-talk; accept 2.00, gsm8k ~2%). Under DSpark something harmless sat there (the earlier
DSpark gates passed); K1 declining or THREADS=1 avoided either the writes or (not investigated) the reuse.
Fix: `scratch_bytes()` sizes for `MAX_BLOCKS` (runner scratch 789,056 B; K2 already sizes for `MAX_BLOCKS`), and
`warmup()` asserts `kda.scratch_bytes(packed.key) <= runner.scratch.numel()` per layer so a future layout change
fails at load instead of corrupting a neighbour. Result: wide_req 4 correct with the debug checks on (no free-list
change, no indexer OOB); gsm8k200 TEMP 0 THREADS 16 with CUDA graphs 0.975, accept 5.126 req-weighted, wall 26.7 s
(`results/acc/dflash_mono_k1fix`; no mono 0.98 / 4.956), 2 more x 4 concurrent correct, no fault.
DSPARK regression (same fix): gsm8k200 TEMP 0 THREADS 1 0.98, accept 4.947, wall 85 s, 4 concurrent correct
(`results/acc/dspark_mono_k1fix_t1`). Commit: sgl-amd-mono `k31/mono-s1` ad8388a (k1.py only, local).

### 10.24 Bucket-384 fault: AMDGPU miscompile under SGPR pressure (mi355-4, 2026-11-02)

Artifacts: `/shared_nfs/kimi-k31/results/m384_fault_20261102/` (`FINDINGS.md`, IR, both ISAs, gdb log, the
CFG / spill-chain scripts `chase.py` / `rdef.py`); logs `logs/m384dbg/`.
Evidence, in order:
- Server and bench faults: "Write access to a read-only page" at 0x6000_0000_0xxx on every rank, TCP client
  (dmesg). The guard's deliberate stop is a store to 0x0 (the m32 faults in `server_wide6/8.log` were that).
- Scratch padded by its own size on both sides (0xA5 canary): still faults -> no scratch overrun.
- Stage bisection (debug `K31_DBG_STOP`): T2 .. route clean; adding `situ_row` faults, yet the faulting store
  is in T5, which runs earlier -> register-allocation (code-shape) effect.
- rocgdb (`set amdgpu precise-memory on`, `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`): `buffer_store_dwordx4 ...
  s[20:23]` at chain.py:114 (`_rs_piece`: store of the new residual to `rsrc(res_out)`, res_out = kernel arg
  `prefix_out`), called from layer.py:292. Descriptor base 0x0000_6000_0000_0000; source pair s[62:63] =
  (lo 0, hi 0x01026000).
- LLVM IR: that store's rsrc is `make.buffer.rsrc(inttoptr %5)`, %5 = prefix_out. IR is correct.
- ISA: s[62:63] is reloaded from SGPR-spill lanes 10/11 (v250 in the default build). Same lane layout with
  `amdgpu-spill-sgpr-to-vgpr=false` (scratch slot 0, lanes 10/11), which still faults with the same value;
  `no-stack-slot-sharing=true` still faults; `verify-machineinstrs` reports nothing.
- CFG reaching definitions on the nospill ISA: the only store reaching T5's reload of that slot is in the kernel
  front (.LBB0_57), lanes 10/11 <- s[54:55] <- `preg` (sync.py:121) = `par*half + offs["agf"]`
  (0x1026000 = the pre-half peer region `agf`), hi / lo swapped -> exactly the observed (0, 0x1026000).
Verdict: the backend feeds T5's prefix_out descriptor from the agf-offset spill slot. Why only bucket 384 /
nvb_a 3 write / t1 off (and why bm 128, ks 8, t1 on hide it): each changes the code shape and so the SGPR
allocation; only some allocations collide. Corrections to 10.22: with t1 off the up loop is empty, so ks changes
the `pdone` size and the code, not "only task scheduling"; the race hypothesis is refuted.

Compiler workaround cost (layer_bench `us: layer`, nvb a3w m4, t1 on), and it does not fix the fault:
```text
                          m128    m384
default                   173.6   343.7
spill-sgpr-to-vgpr=false  341.4   635.5
```

Where the pressure comes from (static spill instrs, default build: writelane 1105 + readlane 1143 of 27,847):
`coll.region_context` builds every region's `_own` address and `peer_bases` (8 peer bases) at kernel entry for
ca / co / cf (~20 x 64-bit live across the launch; sync.py:126 / 121 / 116 / 141 ~520), the 34 64-bit kernel
args stay live, layer.py:215-232 builds 10 scratch region pointers at entry (~180), and the unrolled route
top-16 argmax (moe_front.py:121-130, a 64-bit lane mask per compare, ~290). The wave limit is ~102 SGPRs.

Plan (measure spill count + `us: layer` + the repro after each):
1. Lazy addresses: `region_context` keeps only `own`, `peers` and `par`; `<name>_own` / `<name>_of(p)` form
   the address at use (peer base `s_load` from the `peers` table, scalar-cache hit); scratch regions as
   `scratch + const` at use.
2. Kernel args through a device-side pointer table, loaded per stage (guard against the loads being CSE'd /
   hoisted back to the entry).
3. Route top-16: do not fully unroll (loop over k) so the per-compare masks do not all stay live.
This works around the backend bug rather than fixing it; a minimal LLVM reproducer is a later item.

10.24, continued (mi355-4, 2026-11-02 11:00-12:00):
- ROCm LLVM 23 (`/opt/rocm/llvm/bin/clang -O3`) on the same IR: spills about the same (1097), but the chain.py:114
  descriptor traces back to the entry kernarg load at byte 0x28 = prefix_out: correct. So the bug is the bundled
  backend. `FLYDSL_COMPILE_LLVM_DIR=/opt/rocm/llvm` fails (ROCm mlir-opt rejects the bundled MLIR's rocdl syntax).
- `sgl_amd_mono/common/rocm_codegen.py`: `install(llvm_dir)` sets FLYDSL_COMPILE_LLVM_DIR (FlyDSL then takes its
  external path and keys its cache on it) and replaces that path's codegen: gpu.module -> LLVM IR -> ROCm clang
  (`-O3 -nogpulib -mcode-object-version=6`) -> hsaco -> `gpu.binary`. Results: repro no fault (bundled still
  faults); layer_bench rows 16 / 32 / 128 / 384 x (nvb 0, nvb a3w m4, + capture) all PASS on both compilers,
  `us: layer` within +-1% (m384 a3w m4: ROCm 341.8, bundled 343.4).
- Local SGPR-pressure edits (per-stage contexts, route picks folded per k): spills 1078 -> 987 / 1037, no speed
  change; reverted. Per-stage working sets are near the limit on their own; K31_DBG_STOP leaves loop bodies in
  the code, so spill sweeps with it are meaningless.
- layer_bench "own logits: 1 rows" FAILs (rare, both compilers) were a bench race: a peer's next replay pushed
  logits into this rank's buffer while it read them for the check. Fixed with a sync + barrier per replay
  (6 runs x 2 sizes x 40 replays: 0 rows). route_row now reads the gathered logits with CM_SYS (hardening).

### 10.25 ROCm LLVM exposes a second fault: bucket 128, t1 off, nvb 1/1 (mi355-4, 2026-11-02, ROOT-CAUSED: spill-lane miscompile)

- Wiring: `sgl_amd_mono.register_specs()` calls `rocm_codegen.install()` when `SGLANG_AMD_MONO_ROCM_LLVM` is set;
  scope = kernels whose compile is triggered from this repo's code (call-stack check), aiter's FlyDSL kernels keep
  the bundled codegen (`SGLANG_AMD_MONO_ROCM_LLVM_SCOPE=all` for every kernel). test_contract: 5 failures, the
  same on the unmodified `__init__.py` (pre-existing).
- Server (`launch_server.sh rocm_wide384`, SIM=0, DFLASH draft, `--enable-amd-monokernel`,
  `SGLANG_AMD_MONO_K31_WIDE=1`, default buckets 16..384, CUDA graphs): weights loaded, warmup built buckets
  16 / 32 / 64 / 128, then GPU 7 faulted in `k31_kda_layer_m128_..._a10_m10of8_l68_..._t0_r0` at 0x7e71358b3000
  (heap range), dmesg write + PERMISSION_FAULTS 0x5; rank 7 aborted (`logs/server_rocm_wide384.log`).
- Bench: `--rows 128 --nvb-a 1 --nvb-m 1 --no-t1 --launches S --step S` faults for S = 8 / 16 / 32 / 68 with
  ROCm LLVM (3 / 3 runs at 68), on 3-7 GPUs, heap addresses. Bundled LLVM at S = 68: no fault. ROCm, t1 on,
  S = 68: PASS. The 10.24 sweeps never ran nvb 1/1 + t1 off + step > 4.
- rocgdb on every rank (ROCm build, debug info; wrapper `/tmp/k31_gdb_rocm.sh`: every rank under
  `rocgdb -batch -ex 'set amdgpu precise-memory on' -ex run -ex 'info line *$pc' -ex 'bt 6' -ex 'info registers'`,
  log `logs/m384dbg/gdb_m128rocm_r<rank>.log`): ranks 0 / 4 / 6 SIGSEGV at attnres.py:67 from layer.py:295
  (T5 MLP seam: `post._seam_row` -> `attnres_row`), faulting instr `buffer_load_dwordx4 v[26:29], v2, s[64:67]`,
  s[64:67] = (lo 0x11dc0000, hi 0x7fbe, -1, 0x27000), v2 = 0x42800 = row 19's bank offset ((row*nbk + r)*sl*2
  bytes). Ranks 1 / 2 / 3 / 5 stopped at guard.py:97 (the deliberate stop after a peer failed).
- Open: whether base 0x7fbe11dc0000 is that launch's bank (then the load is in range and something else faults)
  or another buffer; dmesg reports a write. Common factor of 10.24 and 10.25: the SGPR pressure (~1000+ spills);
  each compiler fails on a different variant.
- Root cause (mi355-4, 2026-11-02 11:45-12:30): ROCm LLVM 23 miscompiles an SGPR spill lane, the same class as 10.24.
  Run: the m128 repro with every rank under rocgdb, `FLYDSL_RUNTIME_ENABLE_CACHE=0`, rank 0 keeping the ROCm `.ll` / `.s`
  of the same compile (`K31_DBG_ROCM_DUMP`), and each rank printing every buffer's range (`K31_DBG_PTRS`).
  Rank 2 faults at attnres.py:67 from layer.py:295 with base s[64:65] = 0x7fbe_11a00000; its bank is
  0x7fbc_11a00000 and cw_a 0x7fbe_d4c25000, so the descriptor is (bank.lo, cw_a.hi). No rank's (bank.lo, 0x7fbe)
  falls in any of its tensors: every rank computes a bad address, the first to reach it faults, the guard stops
  the rest (hence "3-7 GPUs").
- IR: T5's rsrc = `make.buffer.rsrc(inttoptr %6)`, %6 = kernel arg 6 = bank (T2's seam has the other one). Correct.
- ISA (matches the gdb disassembly): s[64:65] <- v205 lanes 15 / 16. Flow-sensitive reaching defs of the lanes
  (`rlane.py`, results dir): the only def reaching T5's reload is in the loop header .LBB0_57: it reads lane 16
  (bank.hi) and lane 18 (cw_a.hi), computes `s13 = bank.hi & 0xffff` (bank rsrc word 1, used live by T2's seam
  load, so T2 is right) and `s73 = cw_a.hi & 0xffff`, then writes s73 to lane 16. Entry: `s_load_dwordx16 s[12:27]`
  of kernarg 0 puts bank (arg 6) in lanes 15 / 16 and cw_a (arg 7) in 17 / 18; the back-edge shuffles keep pairs.
- Verdict: both compilers corrupt an SGPR spill slot in this kernel, on different variants; why some variants
  (step >= 8, t1 off, nvb 1/1) and not others is code shape -> allocation, as in 10.24. The server's m128 l68 t0
  fault is likely the same (not traced). Fix: cut SGPR pressure (10.24 plan); the compiler swap only moves the bug.
  Artifacts: results dir `25_*` (IR, ISA, gdb log of rank 2, chase output), `rlane.py`, `k31_gdb_ptr.sh`.

### 10.26 SGPR pressure cut: no fault on either compiler (mi355-4, 2026-11-02)

Evidence and numbers: `/shared_nfs/kimi-k31/results/m384_fault_20261102/FINDINGS.md` (update 12:30-13:30),
`matrix_sgpr_20261102.txt`; diff `26_sgl_amd_mono_worktree.diff`.
- What kept SGPRs live across the launch: all 34 kernel args (loaded at the entry), every region's `_own` and the
  8 peer bases (formed at the entry by `region_context`), scratch region pointers, and GVN merging equal address
  math of different stages (T2 / T5 shared one bank rsrc, the 10.25 victim).
- Edits: (1) `KernArgs`: a stage (or task loop body) re-reads the args it uses from the kernarg segment through
  an opaque `s_mov_b64` (inline asm with side effects) of the segment pointer, so the loads are neither merged
  nor hoisted; the layout comes from the kernel's annotations. (2) `region_context`: `<name>_own` formed at use,
  peer bases read at use with `sload64` (scalar load, address readfirstlane'd). (3) top-16 routing as a
  non-unrolled loop; a taken expert's key ORed to NaN / -inf instead of 14 live "free" masks (bit-identical
  picks and weights). (4) `put_each_peer`: lane 0 of wave p puts to rank p instead of `if tid == p` x 8.
- (4) was forced by a second bundled-LLVM bug: with (2) the 8 `tid == p` branches became a switch on tid with
  the put merged behind a per-lane address -> waterfall loop whose `v_mov v6, -1 / v7, 0x27000` overwrite the
  rsrc address VGPRs it loops on -> fault at 0x7001_0000_0000 on all ranks (ROCm LLVM compiled it correctly).
- Spills (ROCm, 10.25 repro m128 nvb 1/1 t1 off l8): 1048 -> 688 -> 249 -> 59. t1 on a3w m4: m128 ~51, m384
  ~131-150 (was ~1100). Remaining m384 spills are inside stage bodies; moving the GEMM A addresses into the tile
  loops did not help (reverted).
- Speed (`us: layer`, t1 on, a3w m4): bundled m128 173.6 -> 165.8, m384 343.4 -> 335.7; ROCm 166.0 / 336.0.
- Fault matrix, 50 runs x 6 buckets (16..384, bench now uses `wide.CFGS` for 64 / 256): 0 faults on both
  compilers. One "own logits: 1 rows" FAIL (rocm a0 m0 s68 rows 64), not reproduced in 60-replay stress; it is
  a 16th / 17th near-tie under the kernel's v_exp + v_rcp sigmoid (1-3 ulp off torch), seen before these edits
  in 10.24; the bench now ignores swaps of keys closer than 1e-6 (and w_own uses identical-id rows only).
- Other kernels on the edited code: chain / pre_attn / coll / gemm_coll / ag_gemm benches PASS; coll flag stress
  18 x 40020 launches, 0 mismatches.
- Server (mi355-4, 2026-11-02 14:35), gsm8k200 TEMP 0, DFLASH, CUDA graphs, wide on, buckets 16..384, SIM=0
  (`results/acc/wide384_sgpr_*`): bundled THREADS 16 0.985 (accept 5.143) / THREADS 48 0.985 (5.258); ROCm LLVM
  0.98 (5.136) / 0.98 (5.102). No fault; running-req up to 46 / 48. The first ROCm server died in decode graph
  capture: SGLang's FlyDSL bf16 GEMM (llvm option `slp-vectorize-once-used`, bundled-only) went through ROCm clang
  because wide.backbone was on the stack; `rocm_codegen._from_repo` now uses the innermost frame outside FlyDSL.

### 10.27 A/B on the DFLASH config: off / S1 / S1 + wide (mi355-4, 2026-11-02)

Config: `launch_server.sh` (DCP8, SIM=1 accept 3.84, no radix, CUDA graphs) + `--speculative-algorithm DFLASH`
(K31-DFlash2 draft), bundled LLVM; sgl-amd-mono k31/mono-s1 @ 2df98d8, sglang-k31-amd dev_amd b588cf7952.
`ab_bench.sh`: decode 8k/1k c4 / c16 / c48 (n = 4c) + prefill 8k c1. Results `/shared_nfs/kimi-k31/results/ab/<tag>`.

```text
case (tag)                        TPOT c4 / c16 / c48 ms   P90 ITL c4 / c16 / c48 ms   prefill 8k TTFT ms  faults
(a) off   dflash_off_20261102      7.86 / 15.28 / 29.76     9.45 / 15.89 / 26.32        461.4               0
(b) S1    dflash_s1_20261102       7.87 / 15.27 / 29.82     9.47 / 15.96 / 26.31        462.3               0
```
(b) = (a) as expected: S1 builds widths 1..8 only (one request x 8 tokens; log `widths [8]`), so c4 / c16 / c48
never take it; the DFlash2 draft has no mono spec (normal forward). S1's gain is c1 only.
