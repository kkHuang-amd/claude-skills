---
name: mono-kernel
description: >-
  Write, debug and A/B persistent "monokernel" layer launches in sgl-amd-mono (FlyDSL, gfx950) for
  multi-request (c > 1) speculative verify steps, as our Kimi-K3.1 wide-row KDA layer launch does.
  Use when touching sgl_amd_mono/kimi_k31 (wide.py, gfx950/layer.py, pre_attn.py, chain.py, stages/),
  adding a bucket / draft-token count, chasing a wrong latent / GPU fault at address 0x0, reading a
  stage timeline, or deciding whether wide beats SGLang's path. For bs1 latency (K1 / K2, LithosAI
  targets) read the team skill amd-monokernel first (pointer below).
---

# Monokernels for wide verify steps (c > 1)

Owner node: mi355-4 (2026-11-03). Project log: `kimi-k31/MONO_KERNEL.md` (sections 10.x, CONTINUE HERE).
Node-local runs and numbers: `/shared_nfs/kimi-k31/results/<tag>/`.

## 0. Which kernel covers what

| Kernel | Rows a step | Status |
|---|---|---|
| Team K1 / K2 / mla_pre (`kevin-mii/sglang-k31-amd` branch `mono/kimi-k31`) | <= 8 total (`kimi_k3.MAX_ROWS = 8`): bs1 only (6 or 8 draft tokens) | bs1 latency path ("S1") |
| Ours: `kimi_k31/wide.py` + `gfx950/layer.py` (`SGLANG_AMD_MONO_K31_WIDE=1`) | R x L, R 2..48, L = draft tokens (6 or 8) | opt-in; -2.9..-3.6% vs off at c4..c48 (10.37); default-on waits for KVV |

Team skill (bs1, worth reading for mechanisms): `git show ee15183f58:.agents/skills/amd-monokernel/SKILL.md`
in `/sgl-workspace/sglang-k31-amd` (references: handoffs-and-memory.md, flydsl.md, measurement.md).

## 1. The wide layer launch in one paragraph

One persistent launch a KDA layer: T1 MoE tail of layer i-1 (TP push of the latent / shared partials,
latent up, prefix add) -> T2 attention AttnRes seam -> T3 in_proj -> KDA (a task a (request, head), ring)
-> T4 o_proj -> T5 RS + residual, MLP seam -> T6 MoE front (router / latent-down / shared gate-up) ->
route + SiTU + shared down. Stages hand off by flags (device / system scope) across CTAs and ranks. A
"run" is the KDA layers between two DSA layers: entry (first launch takes the DSA layer's MoE tail),
exit (column tail at <= 64 rows, else prefix-in-AR). Rows are padded to a bucket of whole dummy
requests (slot -1); bucket rows = L x requests, configs from the 8-token bucket of the same requests
(`wide.cfg_rows`).

## 2. Rules that cost us a day each

- **Partial tiles (m not a multiple of a GEMM's bm).** Data stores mask `row < m`, but every *flag* a
  split-K piece publishes must too: a piece with `r0 >= m` has band >= nbf, and its flag index lands
  on the next rank's band (consumer reads a latent band before it exists -> latent rel 3-9e-2) or past
  the region (rank 7 -> wait deadline -> "GPU fault at 0x0"). Fixed in `finish_front`, `pre_attn._finish`,
  `gemm_coll._push_band_flag` (guard `r0 < m`, 2026-11-03). Any new finish / flag: same guard.
  Symptom map: one tensor wrong (latent) + others fine = a hand-off, not the math.
- **FlyDSL cache keys miss `module.func` helpers.** kimi_k31 kernels do not use `kernel_sources`
  keying (kimi_k3 does): an edit in `stages/*.py` or `pre_attn.py` can reuse a stale binary. Benches:
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`. Before a server after any kernel edit: move `~/.flydsl/cache` aside
  (never restore an old one). Do not edit the tree a starting server imports from.
- **New L / bucket = new widths end to end.** SGLang's binding keeps device widths with
  `w % tokens == 0`; `wide_widths()` is computed at import (no server args yet), so it lists every
  supported L. The kernel asserts `m % L == 0`. Read L from `runtime_context.get_spec()`, not a default.
- **Prove a change at the width it targets and at every other.** Gate (below) at L=6 rows 12..288 and
  L=8 rows 16..384; partial tiles only appear at L=6 (all L=8 buckets are tile multiples).
- **After a GPU fault:** `rm -f /tmp/gpucore*` and core files in the crashed process's cwd (one torchrun
  from the repo root left 133 GB). `ulimit -c 0` before servers / torchrun.
- **Count tasks against BLOCKS (256).** A stage with 257..512 tasks runs two rounds: the last CTA ends a whole task
  after the median. The front at 192 / 256 rows had 272 split-K tasks -> `wide.FRONT_ROWS` picks one-round configs
  (10.33, -11% a launch at m192); in_proj at m192 / m256 had 400 -> `wide.INP_ROWS` (10.37, -7% / -3%). Check
  `tiles x ks` of every GEMM stage for every new bucket / config, and the row fill: L=6 buckets inherit the 8-token
  bm (192 rows in a 256-row tile pair); `layer_bench --inp <rows>=<cfg>` sweeps in_proj.
- **One 128 B line per counter / mailbox that many CTAs hit at once.** Split-K tile counters packed 32 to a line
  cost ~7.6 us a launch at every size (`layer.CNT_STRIDE = 32`, 10.33). Partial-line XGMI pushes and the `fgf` flag
  stride measured nothing (noise): stride where many writers and spinners share a line, not by default.
- **Reading SGLang tensors in place:** they hold the step's rows, the bucket may hold more -> bound the buffer
  descriptor (`rsrc(ptr, nbytes)`, OOB loads return 0; `pfx_bytes`, 10.34) and test it with NaN rows past the bound
  (`layer_bench --pfx-short`; `rel` maps NaN to inf, plain `max()` would hide it).
- **Bundled LLVM** (default). `SGLANG_AMD_MONO_ROCM_LLVM=/opt/rocm/llvm` faulted at m128 t1-off (10.24-10.26).

## 3. Gate ladder (every kernel edit)

1. Kernel, cache off (`results/pro_20261103/k31_gate_l6.sh <tag> [6|8]`): layer_bench variants a0m0,
   a1m1, a3w/m4, 68 launches, capture, no-t1, restart (a1m2 / a3w4 / a7m8) + pre_attn_bench kda.
   PASS = every check under its threshold, 0 faults. layer_bench prints mid / h / core / windows /
   latent / shared / topk_w rel; a single bad tensor points at its hand-off.
2. Codegen: an edit that should not touch a path -> dump ISA and diff (team flydsl.md:
   `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<d> FLYDSL_RUNTIME_ENABLE_CACHE=0`, `<d>/<kernel>/NN_final_isa.s`).
3. Real model, eager, SIM=0: `SGLANG_AMD_MONO_K31_WIDE_ENTRY_CHECK=40 ... --disable-cuda-graph` +
   `wide_req.py 4` / `16`: entry / exit stream rel ~3-4e-3 median; isolated 1e-2 = top-16 near-tie.
   `SGLANG_AMD_MONO_K31_WIDE_CHECK=<layer>` (wide_check.py) compares one layer launch stage by stage.
4. gsm8k200 TEMP 0, CUDA graphs, SIM=0, THREADS 48 and 16 (`gsm8k_gate.sh`): >= 0.975 and within
   ~1 pt of off on the same weights / draft. Check real accept length too (draft vs target mismatch
   shows there first: K3.1-Pro + step29301 draft 3.7, + step55000 4.9).
4b. KVV (toolcall / OCRBench / MMMU-Pro / BEAM 1M, effort max; `kimi-k31/KVV_ACCURACY.md`): off and the
   changed path, when a round changes many paths, before wide becomes default, before a PR.
5. Serving A/B, same session, off then wide, `ab_bench.sh` with `AB_CONC="4 8 16 32 48"`, SIM=1 at the
   AgentX accept (3.399, 6 draft tokens). Decision rule: wide on only if <= off beyond ~1% at every
   bucket that matters.

## 4. Attribute before optimizing

- Trace: `capture_trace.sh <tag> <log> <c> 8192 1024 <prefill batches> <s>`; `seg_attr.py` splits a
  VERIFY step at each MoE stage 2 (`opus_moe_stage2|moe_stage2|down_ws_kernel|gemm2_a4w4_port` -- the
  m256 stage 2 is `gemm2_a4w4_port`; without it seg_attr sees 1 segment); `cmp_seg.py <wide> <off>`
  gives per-kind diffs: L (mid-run launch), Lfirst (run entry), Dafter_run (exit), D.
- Stage timeline: `layer_bench.py --timeline` (layer.py `timeline=True`: 14 marks into scratch `tl`,
  no new kernel argument, off = unchanged build). Read the LAST CTA per mark; compare with the halves:
  `pre_attn_bench --kda L` and `chain_bench --timeline` at the same configs.
- 10.33-10.36 (2026-11-03): one-round front + counter stride, prefix / bank by stride, sort + pop route, uncached
  scratch -> wide beats off by 1.7-3.2% at c4..c48 (6 draft tokens). Before that:
- 2026-11-03 finding (K3.1-Pro, m256 = c32): wide - off +2.55 ms/step, all in the launch body; pre part
  (T1..KDA) = the pre half; post part +24 us, all in the MoE front stage (finish_front's 8-peer
  gather); T6 tail 20 us (route 11 us at every size).

## 5. Mechanisms from the team skill that apply to wide (amd-monokernel, MI350X / MI355X numbers)

- Hot lines: device-scope stores from many CUs into one 128 B line serialize (~50-100 ns each);
  `FLAG_STRIDE = 16` (sgl-amd-mono#82, open) gave -0.1..-0.2 ms/step at TP8 bs1. Our fgf / pdone /
  band flags are published by many CTAs at once: candidate. Do not stride regions polled in bulk.
- Whole 128 B lines across XGMI: 16 B halves per lane write half lines; a gather round 31.6 -> 13.3 us
  (TP2, 24 rows). finish_front pushes the latent to 8 peers in 16 B stores: first suspect for the
  +24 us front stage at m256.
- `vmcnt` is in order: a poll issued after a prefetch waits for the prefetch. Prefetching front weights
  while waiting on the seam is only a win if measured.
- Uncached launch scratch (one-rank PeerBuffer): -1.35..-2.76 us a launch (x 68 launches a step).
- Fit a stage in one round; place work past busy CTAs (router 19 -> 6 us on GLM-5.3); the team's
  route by per-lane sort + 16 wave-max pops cut 8.4 -> 4.3 us (mono/kimi-k31 05dfa48); ours rescans
  14 candidates a pick.
- One flag hop ~0.5-0.6 us idle, 1.1-1.25 us while CTAs stream weights: removing a hop beats speeding
  a stage.
- Eager `@flyc.jit` launches rebuild their key each call (24-182 us host); graphs replay none of it.

## 6. Repro snippets

```bash
S=/shared_nfs/kimi-k31/results/pro_20261103/scripts   # K3.1-Pro, DFLASH step55000, 6 tokens, SIM 3.399; RADIX=1 for KVV
bash /shared_nfs/kimi-k31/results/pro_20261103/k31_gate_l6.sh mytag 6     # kernel gate, L=6
SGLANG_AMD_MONO_K31_WIDE=1 bash $S/launch_server.sh wide_x --enable-amd-monokernel
AB_CONC="4 8 16 32 48" bash $S/ab_bench.sh wide_x_ab
```
