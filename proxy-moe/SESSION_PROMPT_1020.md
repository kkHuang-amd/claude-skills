# Proxy MoE — next-session prompt (handoff from mi355-4, 2026-10-20)

Paste the block below as the first message on the new node (after NEW_WORKSPACE_PROMPT.txt).

```text
Read /workspace/claude-skills/NEW_WORKSPACE_PROMPT.txt and follow it, answer in Chinese.
Then: cd /workspace/claude-skills && git pull --rebase (autostash if dirty), and read ONLY:
  proxy-moe/SKILL.md
  proxy-moe/OPT_mi355-4.md  sections "CONTINUE HERE", "Summary vs baseline", "Task items",
                            "AR wait attribution" (use rg -n '^## ' first, then sed ranges)
Do not reconstruct prior chat history; the docs are the source of truth.

This is a NEW node: `hostname` → create proxy-moe/OPT_<hostname>.md with "Owner node: <hostname>"
and a CONTINUE HERE block; never edit OPT_mi355-4.md (owned by mi355-4). Results go to
/shared_nfs/kk/results/proxy_24l/<run-tag>/ on this node (mi355-4's /shared_nfs is a LOCAL RAID,
its result paths in the docs are not visible here).

Environment bring-up on this node (detect first; package README is authoritative):
  PKG=/workspace/models/proxy_moe_sglang/proxy_moe_sglang   (check it exists)
  1. Check versions: SGLang has models/laguna.py; AITER commit (mi355-4 used 2bb549102).
  2. Free disk for a 1.36 TB 24L checkpoint (NOT /workspace if small); tokenizer + make_hf_config
     (--num-layers 24 --num-gpus 8, config sha256 must be 9b60ba1c…bd7bd) + gen_dummy_ckpt (seed 0).
  3. AITER patches: python3 $PKG/patches/aiter_mx_quant_large_hidden.py ; then
     cd /sgl-workspace/aiter && git apply --check /workspace/claude-skills/proxy-moe/patches/aiter_2bb549102_proxymoe_tuning.patch
     && git apply it (HIDDEN_MAX bucket in mxmoe_dispatcher, tuner ref vectorize, TUNE_MOE_STAGE2_ATOMIC_ONLY).
     cp proxy-moe/configs/proxymoe_fp4_tuned_fmoe.csv /sgl-workspace/aiter/aiter/configs/model_configs/
     If AITER commit differs, check the patch applies; if not, stop and tell me.
  4. Reproduce the two best configs WARM (discard first run per container: cold JIT), run from /tmp:
     EP8: NUM_LAYERS=24 TP=8 EP=8 SKIP_GEN=1 RUN_PROFILE=1 AITER_LARGE_HIDDEN_WORKAROUND=1
          SERVER_EXTRA_ARGS="--enable-aiter-allreduce-fusion" bash $PKG/scripts/run_stage.sh
     EP1: same with EP=1.
     Expected (mi355-4): EP8 813 tok/s / TTFT 308 / TPOT 18.5; EP1 902 / 359 / 16.4.
     Check env ROCM_QUICK_REDUCE_QUANTIZATION=INT8 is set as on mi355-4 (user approved INT8 prefill AR).

Then continue the optimization from the Task items table (TODO: T3 prefill comm overlap / DP-attn,
T4 FP32 router GEMM, T5 rocprof counters, T6 small kernels, T9 decode MoE kernel, T7/T10 upstream).
Ask me which task first if unclear. Analysis tools: proxy-moe/scripts/trace_kernels.py,
rank_skew.py, ar_wait_attrib.py, ep_moe_check.py. Follow perf-bottleneck-attribution: one variable
per experiment, measure before claiming a bound.
```
