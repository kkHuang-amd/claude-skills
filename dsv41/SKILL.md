---
name: dsv41
description: >-
  DeepSeek-V4.1-Flash on AMD gfx950 (MI350X/MI355X) with SGLang, tracking PR sgl-project/sglang#39857
  ([AMD] Support DeepSeek-V4.1 on gfx950 with DSpark and fused kernels). Use when working in
  /sgl-workspace/sglang-dsv41 (branch dsv41-amd-main), launching or benchmarking DSV4.1-Flash,
  DSpark speculative decoding, HIP low-ratio attention / KV store / RoPE fusion, BF16 WO-A,
  mHC split-H, TP4 all-reduce+mHC fusion, sharded Engram, or AITER MoE on this branch.
---

# DSV4.1 on gfx950 (PR #39857)

## CONTINUE HERE

**ACTIVE:** ATOM optimization port (details + own CONTINUE HERE: ATOM_PORT.md). /sgl-workspace/sglang-rolao-opt branch
atomport-mxfp8-producers == pushed rolao/dsv41/opt-branch @ 026da361c0 (8c67e0bd51 T2 MXFP8 producer fusions; ba5eb0f439
shared-expert MXFP4 requant ties-to-even fix; 026da361c0 DSpark draft block metadata built in the draft CUDA graph).
**Best config** = that commit + `--enforce-shared-experts-fusion --fp8-gemm-backend aiter` + SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1
+ tuned FMoE CSV /sgl-workspace/aiter-5750/aiter/configs/model_configs/dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv (untracked;
backup patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv), scripts/agentx_colleague_run.sh, TP2 EP1, GPUs 4,5,
aiter-5750 + pydeps-flydsl-0341 via PYTHONPATH. Rebuild in a fresh container: setup_env.sh then
scripts/setup_atomport_env.sh (RUNBOOK.md "ATOM-port worktree").
**AgentX 2026-09-29** (TTT / P90, 0 faults): c1 10,306.1/292.8, c2 10,476.9/279.6, c4 15,680.7/267.0, c8 28,552.6/199.3,
c16 51,219.5/137.8, c32 PDI16 90,876.4/102.0, c32 PDI4 93,855.8/68.7, c64 PDI16 90,671.3/78.8, c64 PDI4 117,189.7/39.7.
vs rolao all-opts +3.8..+7.6% TTT, +3.1..+10.3% P90; vs ATOM behind at c1/c2/c8 (P90 -13/-14/-17%), ahead at c32/c64 PDI4.
Table: results/agentx.md. GSM8K ~0.90 (0.901-0.908, CONC=32 eval server).
**RUNNING:** PDI sweep c8/c2 at PDI 32 and 4 + c1 PDI 4 (ATOM_PORT.md). c8 PDI32: 28,643.3/212.3 (P90 +6.5% vs PDI16).
**Engram host-table GPU fault (2026-09-29, fixed locally, NOT in 026da361c0):** apply
`patches/sglang_local_engram_host_devptr_0001.patch` (`git apply`; setup_atomport_env.sh does it). Needed on any node where
`scripts/hostreg_devptr_check.py` prints `same=False` (crsuse2-m2m-176, -227). Not yet committed/pushed to rolao.
**PR pass criteria:** GSM8K 5-shot 1319 TP4/EP4 ≈ 90.45% (DSpark off) / 90.22% (on).

## History (closed)

Details: /shared_nfs/kk/dsv41/doc_backup_20260929/SKILL.md (CONTINUE HERE history), NOTES.md.
- 09-29 TP2 best config faults deterministically at prefill graph capture (4096) on a 2nd node: engram `_HostTable`
  cudaHostRegister's an mmap but engram_gather got the HOST VA; on ROCm dev VA != host VA on some nodes. Fix: use
  hipHostGetDevicePointer (patches/sglang_local_engram_host_devptr_0001.patch), GSM8K 0.901. The 09-24 TP2
  "MoE" fault was very likely the same bug. Report: /shared_nfs/kk/dsv41/atomport/verify_env/report_crsuse2-m2m-176.md.
- 09-24 fresh-container TP4 repro (opus-prefill 048ffae315): PERF=1 c1/c8/c32 145/927/2344 tok/s, DSpark on (SIM_AL=3.51)
  366/1679/3333. Never bench two servers at once (inflated TTFT, c8 -11% tok/s).
- 09-24 aiter -> v0.1.22.post1 (b4d9154d1) + #5561 + local edits (#5802 dropped); TP4 GSM8K 0.907, 145/924/2350 tok/s.
  Old state: /sgl-workspace/aiter_jit_backup_acf8fdf93, stash@{0}, /shared_nfs/kk/dsv41/aiter_pre_upgrade_full.diff.
- TP2 aiter MoE graph-capture fault (old container) did not reproduce in the fresh one (NOTES "AgentX TP2 bring-up").
- 09-25 c16/c64 "Write access to a read-only page" faults = int32 KV-store `loc` overflow; fixed by upstream #41159
  (patches/sglang_local_kvstore_int64_0001.patch).
- 09-25 B200-port script: RCCL OUT_OF_RESOURCES crash at every conc (not memory/engram/PDI); superseded, never bisected.
- 09-26 colleague recipe reproduced: c16 PDI16 45,804.7/112.5 (colleague 45,518.84/110.02), c64 PDI4 99,666.6/33.6
  (96,837.78/32.15); OPUS +2.0%/+4.6% TTT. Old TP2 gap was the recipe (engram per_rank huge pages etc.).
- 09-26 rolao all-opts (2b875bd95a) sweep, c1 9,818.6/267.5 .. c64 PDI4 111,506.4/38.0, GSM8K 0.893 (results/agentx.md).
- OPUS share patches: patches/sglang_share_opus_prefill_000{1,2}.patch (+ /shared_nfs/kk/dsv41/share/), `git am` clean on e2e824dc58 and pr/41021.

## Folder rules (MUST follow -- keep this dir tidy)

```
dsv41/
  SKILL.md         index: CONTINUE HERE, recipe, gotchas, rules.
  VLLM_COMPARE.md  vLLM-vs-SGLang comparison + reusable-optimization inventory (own CONTINUE HERE).
  OPUS_PORT.md     porting aiter OPUS sparse prefill into the V4.1 HIP path.
  ATOM_PORT.md     porting ATOM V4.1 optimizations (gap list in NOTES.md 2026-09-27).
  RUNBOOK.md       rebuild the validated env from a fresh container (keep in sync with scripts/).
  ENV_1001.md      HaiShaw/sglang dsv41-env + aiter e7d2453f2 + #5967 env rebuild and validation (own CONTINUE HERE).
  MAIN_REGRESS_1002.md  regression check of upstream main + PR #42055 vs env1001 (own CONTINUE HERE).
  NOTES.md         longer findings / investigation log, newest first.
  scripts/         ALL runnable scripts (*.sh, *.py). Nothing executable anywhere else.
  patches/         other repos: <repo>_<upstreamPR>_<what>.patch (`git -C /sgl-workspace/<repo> apply`);
                   our unpushed sglang commits: sglang_local_<topic>_000N.patch (`git am`). Record state in CONTINUE HERE.
  results/         small curated tables (*.md), one per benchmark type; a row per run, never raw logs.
```

- Raw logs, traces, dumps, JSONL -> `/shared_nfs/kk/dsv41/`, never here. Result rows link to them.
- New script: in `scripts/`, header = purpose + env knobs + output location. Prefer a knob over a near-duplicate.
- Scripts resolve paths relative to themselves (`$(dirname "$0")/..`); use `grep -E`, NOT `rg` (agent shell only).
- No other subfolders without updating this section first. Update CONTINUE HERE whenever status changes.

## Scripts (in scripts/)

- `launch_server.sh` TP4 server (PERF=1, DSPARK=1); `setup_env.sh` idempotent env setup (RUNBOOK); `setup_atomport_env.sh`
  idempotent ATOM-port best-config env (sglang rolao worktree, aiter-5750 + patch + tuned CSV, flydsl 0.3.4.1).
- `agentx_colleague_run.sh` **current** AgentX launcher (shim for `agentx_colleague_mi355x_sglang.sh`); knobs `TAG CONC OPUS
  TP EP_SIZE GPUS PREFILL_DECODE_INTERVAL SRC EXTRA_ARGS SERVER_ONLY EVAL_ONLY`; out /shared_nfs/kk/dsv41/agentx/<TAG>/.
- `agentx_series.sh` sequential AgentX points (`RUNS SCRIPT PREFIX`), progress agentx/series.txt.
- Historical: `agentx_dsv41_mi355x_sglang.sh`, `agentx_dsv41_b200port_mi355x.sh`, `agentx_colleague_pipeline.sh`.
- `run_gsm8k.sh` -> results/gsm8k.md; `run_gsm8k_openai.sh` engine-neutral GSM8K (never with simulated acceptance);
  `run_pr_style_c1.sh` PR bs1 method -> results/perf_prstyle.md; `run_throughput.sh` -> results/perf.md;
  `pipeline_eval.sh` wait + gsm8k + throughput; `vllm_launch.sh` vLLM server (inside vLLM container).
- `hostreg_devptr_check.py` node check for the engram host-table fault (host VA vs device VA after cudaHostRegister).
- ATOM-port A/B: `atomport_proxy_bench.py` (decode proxy), `atomport_draft_raw_{ab,ab2,parity}.sh` (in-graph draft
  metadata), `greedy_parity_dump.py`, `shared_expert_requant_check.py`, `test_{ffn_norm,shared_act,wo_a}_mxfp8.py`.
- Profiling: `profile_prefill.sh`, `trace_summary.py`, `trace_comms.py`, `atomport_{trace_breakdown,step_spans,step_sequence}.py`,
  `pyspy_tree.py`. Microbench/repro: `bench_*.py`, `hip_event_sync_latency.py`, `hip_graph_wait_microbench.py`, `repro_moe_tp2.py`.

## Launch recipe (TP4, cookbook deepseek-v4_1.jsx MI350X; image lmsysorg/sglang:dev-dsv41-mi35x)

`launch_server.sh`: TP4+EP4, `SGLANG_USE_AITER=1` (else fp4 experts assert), `SGLANG_MOE_PADDING=1`,
`AITER_FLYDSL_FORCE_REDUCE=1` (determinism), `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`, `AITER_BF16_FP8_MOE_BOUND=0`,
`TRITON_HIP_USE_ASYNC_COPY=0`; `--disable-radix-cache --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 4096`.
`DSPARK=1`: `--mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5
--cuda-graph-max-bs-decode 64`. No hierarchical cache on ROCm; PD + spec incompatible. Branch runs via PYTHONPATH.

## PR status

#39857 superseded by stack #41018 (MXFP8 kernels) -> #41019 (KV layouts + attention) -> #41020 (HIP radix backend) ->
#41021 (model wiring); local refs `pr/41018..41021`. None contain our OPUS prefill. aiter: ROCm/aiter#5561, #5562, #5802.

## Reference numbers (4×MI350X, ISL 4096 / OSL 1024, out tok/s, DSpark off → on, real acceptance)

Random bs1 155.7 → 641.9; real text bs1 155.9 → 335.1, bs8 951.7 → 1446.5, bs32 2182.2 → 2241.9.
Known PR issues: intermittent RCCL graph-capture abort; AITER tolerances unvalidated; no fresh Docker / CP / EAGLE runs.

## Gotchas

- Never pass `cudaHostRegister`'ed host pointers to kernels: use hipHostGetDevicePointer. Check a node with
  `WHICH=devptr|hostptr python3 scripts/hostreg_devptr_check.py` (hostptr faults iff `same=False`).
- Weights 510 GB (experts 296 + ENGRAM 203). TP2 needs `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`; use engram `per_rank`
  (`shared` = 0% huge pages, ~10x slower lookups).
- `AITER_BF16_FP8_MOE_BOUND=0` (missing in cookbook), else `Unsupported kernel config for moe heuristic dispatch`.
- aiter #5561 (LDS-DMA race; GSM8K 0.885 -> 0.905) required. Local aiter edits: /shared_nfs/kk/dsv41/aiter_preexisting_local.diff.
- aiter merges `aiter/configs/model_configs/*tuned_fmoe*.csv` of the IMPORTED aiter; restart the server after changes.
- AgentX lib: `install_agentic_deps` rm -rf's the shared venv (set AIPERF_DEPS_READY=1; never parallel);
  `wait_for_amd_gpu_clean` checks ALL 8 GPUs.
- Never `pgrep -f`/`pkill -f` a pattern in your own command line (kills your shell). Stop by PID:
  `ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'`, AND its renamed children
  (`sglang::scheduler_TP*`, detokenizer; else they keep VRAM): `ps -eo pid,comm | awk '$2 ~ /^sglang::/'`.
- rocm-smi GPU[7] = HIP device 5 here (GPUS=4,5 shows GPU[4]+GPU[7]); rocm-smi shows host PIDs as UNKNOWN.
- Fork `origin/main` is stale: diff against `$(git merge-base HEAD upstream/main)`. `gh` missing; use WebFetch.
- Related skills: `dsv4`, `agentx`, `aiter-custom-allreduce-nan-crash`, `sglang-prefill-coalescer`, `perf-bottleneck-attribution`.
