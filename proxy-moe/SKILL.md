---
name: proxy-moe
description: >-
  Run and optimize Meta's proxy MoE model (hidden 16384, 512 experts top-16 + shared, MXFP4 W4A4
  experts, BF16 GQA attention with 3:1 sliding/full layers, FP8 KV) on SGLang LagunaForCausalLM with
  a locally generated dummy checkpoint on AMD MI35x. Use when setting up, serving, benchmarking,
  profiling, or tuning the proxy MoE / proxy_moe_sglang package.
---

# Proxy MoE (SGLang, dummy weights)

Package (read-only source): `/workspace/models/proxy_moe_sglang/proxy_moe_sglang/`
(`README.md` is the authoritative how-to; `reference/expected_results.md` is the MI350X baseline).

Fixed for comparability: `spec/proxy_spec.json`, 12 layers on 4 GPUs / 24 on 8, MXFP4 experts,
BF16 attention, FP8 KV, dummy ckpt seed 0. Free to tune: SGLang/AITER versions and kernels, server
flags, in-node parallelism.

## Task docs (one per task, owner node on top)

| Doc | Topic |
|-----|-------|
| `ENV_SETUP_mi355-4.md` | Environment bring-up on mi355-4, first baseline run |
| `OPT_mi355-4.md` | Warm profile, roofline estimates, optimization levers and results |
| `OPT_crsuse2-m2m-231.md` | Continued on crsuse2-m2m-231: stage-2 MoE b_nt fix, bf16x3 router, decode MoE analysis, pushed branches, TP4 (T11) |
| `SESSION_PROMPT_1020.md` | Handoff prompt to continue on another node |
| `SESSION_PROMPT_OPT_1021.md` | Handoff prompt: continue optimization on crsuse2-m2m-231 after the sweep |
| `fixed_seq_len/` | InferenceX-style sweep (1k/1k, 8k/1k, 70k/300; TP8 + TP4): scripts, README, session prompt |
| `fixed_seq_len_gb300/` | Same sweep on GB300 (TP4, 12L): Quark MXFP4 on CUDA patch, scripts, RESULTS |
| `MI455_RUN.md` | MI455 (gfx1250) run recipe on h13-16: container, AITER patches, 1K1K / 8K1K / 70K300 commands |
| `MI455_RUN_h13-17.md` | MI455 optimization log on h13-17: P0 attribution, RCCL CUMEM/DDA, FlyDSL fp32 router, aiter custom AR, results (CONTINUE HERE on top) |
| `MI355X_OPT_CATALOG.md` | Catalog of MI355X optimizations with gains and gfx950 assumptions, plus the target numbers |
| `RCCL_DDA_HANG_REPRO.md` | Repro of the RCCL DDA all-reduce hang on gfx1250 TP8 (`NCCL_CUMEM_ENABLE=1`) |
| `SESSION_PROMPT_MI455_1024.md` | Handoff prompt and node-local state for continuing the MI455 work |

Kernels (`kernels/`): `router_gemv_gfx1250.py` (FlyDSL fp32 router GEMV/WMMA), `wmma_f32_layout_probe.py`.
Techniques learned on gfx1250: `../amd-kernel-opt-techniques-gfx1250/SKILL.md`.

Tools (`scripts/`): `trace_kernels.py <trace.json.gz> --steps N` (kernel time by name/category),
`rank_skew.py` / `ar_wait_attrib.py` (cross-rank AR wait attribution), `ep_moe_check.py`
(real `fused_moe` + expert_mask, compare two AITER configs).

Portable state: `patches/aiter_2bb549102_proxymoe_tuning.patch` (local AITER fixes, apply on top
of the package's `aiter_mx_quant_large_hidden.py`), `configs/proxymoe_fp4_tuned_fmoe.csv` (install
into `aiter/configs/model_configs/`), raw tuner outputs `configs/tuned_*_all14.csv`.

Results go under `/shared_nfs/kk/results/proxy_24l/<run-tag>/` (or `proxy_12l`).
