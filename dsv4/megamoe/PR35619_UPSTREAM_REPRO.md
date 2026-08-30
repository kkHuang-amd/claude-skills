# sglang#35619 (upstream Aiter MegaMoEv2) — reproduction on stock aiter

Both throughput numbers claimed in
[sgl-project/sglang#35619](https://github.com/sgl-project/sglang/pull/35619)
reproduce on **stock `/sgl-workspace/aiter`** — the `aiter-megamoe-pr4439` /
`FlyDSL-mega_moe_v1` checkouts from the original bring-up are **not** required.

| Arm | PR claims | Measured 2026-08-27 | Delta |
|---|---|---|---|
| MegaMoE, no EPLB | 38,863.10 tok/s | **39,200.88** | +0.87 % |
| MegaMoE + EPLB (interval 200) | 42,382.44 tok/s | **42,505.60** | +0.29 % |
| EPLB gain | +9.05 % | **+8.43 %** | — |

5120/5120 requests successful, 0 failed, on both arms.

## Environment

```
sglang   /sgl-workspace/sglang @ a1f9508dd4 + PR #35619 applied (git apply)
aiter    /sgl-workspace/aiter  @ c16d44b93   (built-in aiter.ops.flydsl.kernels.mega_moe)
mori     /sgl-workspace/mori
model    /shared_nfs/models/DeepSeek-V4-Pro
hw       8x MI355X (gfx950), ROCm
```

`SGLANG_AMD_FLYDSL_KERNELS_PATH` is **not** needed — PR #35619 imports
`aiter.ops.flydsl.kernels.mega_moe`, not an external FlyDSL workspace.

## Full commands — arm 1, no EPLB (39,200.88 tok/s)

Server:

```bash
cd /workspace/useful-scripts/benchmarking/dsv4
MODE=megamoe PORT=30000 \
MODEL=/shared_nfs/models/DeepSeek-V4-Pro \
MEM=0.85 \
MORI_SHMEM_HEAP_SIZE=17179869184 \
bash run_sgl_dsv4_unified.sh
```

Client:

```bash
cd /workspace/useful-scripts/benchmarking
python3 common_oai_benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --model /shared_nfs/models/DeepSeek-V4-Pro --trust-remote-code \
  --input-len 8192 --output-len 1024 --num-prompts 5120 \
  --warmup-requests 1024 --max-concurrency 512 \
  --prompt-manifest /tmp/common_prompt_c512.jsonl.gz \
  --output-dir /tmp/pr35619_noeplb
```

## Full commands — arm 2, EPLB (42,505.60 tok/s)

Server — identical except for the four additions:

```bash
cd /workspace/useful-scripts/benchmarking/dsv4
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/tmp \
MODE=megamoe PORT=30000 \
MODEL=/shared_nfs/models/DeepSeek-V4-Pro \
MEM=0.85 \
MORI_SHMEM_HEAP_SIZE=17179869184 \
SGLANG_AITER_MEGA_RANK_SYNC=1 \
SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY=1 \
SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD=1 \
SGL_EXTRA_ARGS="--enable-eplb --eplb-rebalance-num-iterations 200 --expert-distribution-recorder-mode stat" \
bash run_sgl_dsv4_unified.sh
```

Client: same as arm 1, only `--output-dir` changes. **Reuse the same
`--prompt-manifest`** so both arms replay identical prompts.

Resulting server argv (both arms):
`--tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend megamoe
--moe-dense-tp-size 1 --enable-dp-lm-head --load-balance-method round_robin
--attention-backend dsv4 --kv-cache-dtype fp8_e4m3 --page-size 256
--swa-full-tokens-ratio 0.15 --chunked-prefill-size 65536 --cuda-graph-max-bs 1024
--max-running-requests 1024 --disable-radix-cache --disable-shared-experts-fusion`
— no speculative decoding, no HiCache.

## Five things that silently break it

1. **`MORI_SHMEM_HEAP_SIZE` default (4 GiB) is too small.** MegaMoEV2's
   dispatch/combine buffers need ~3.1 GiB *each*; the default overflows during
   **decode CUDA-graph capture** with `AssertionError: mori_shmem.shmem_malloc
   failed`. 16 GiB is enough — `MEGAMOE_HANDOFF.md`'s 40 G is over-provisioned,
   and the 24 GiB it wastes costs KV pool. Note capture exercises MegaMoE
   unconditionally (`should_use_mega_moe` returns True in capture mode), so this
   fires regardless of load.
2. **`--enforce-shared-experts-fusion` is incompatible with a2a.** Use
   `--disable-shared-experts-fusion` (the unified script already does).
   `forward_mega_moe` sizes MegaMoEV2 with
   `topk = num_experts_per_tok + num_fused_shared_experts`, so fusion changes
   topk from 6 to 7 and the FlyDSL dispatch path faults with
   `HSA_STATUS_ERROR_EXCEPTION 0x1016`.
3. **The four DP-comm envs must be 0**, not 1:
   `SGLANG_SHARED_EXPERT_TP1`, `SGLANG_DP_SHARED_EXPERT_LOCAL`,
   `SGLANG_DP_USE_GATHERV`, `SGLANG_DP_USE_REDUCE_SCATTER`. MoE comm is mori's;
   leaving the DP gatherv path on runs two comm schemes at once. (InferenceX's
   AgentX launcher sets all four to 1 for its non-MegaMoE DP arm — copying that
   block is the trap.)
4. **EPLB without `--expert-distribution-recorder-mode stat` is a regression,
   not a win.** Measured: **36,642.28 tok/s (−6.5 %)** with 1040 rebalance
   events and no load statistics to drive them. Adding the recorder is what
   turns it into +8.4 %.
5. **Do not pass `--ep-num-redundant-experts`.** The validated config leaves it
   at 0. Setting 32 costs ~640 K tokens of KV pool for no gain, and at
   `mem-fraction-static 0.89` it pushes the DSv4 indexer's 4.85 GiB long-context
   scratch allocation into `torch.OutOfMemoryError`.

## Open bug: `SGLANG_AITER_MEGA_RANK_SYNC` + DP attention

`forward_batch_info.py`, in the PR's new idle path:

```python
if mega_moe_idle_materialize:          # RANK_SYNC and is_extend_in_batch and forward_mode.is_idle()
    global_num_tokens = [1]
```

`adjust_num_token_non_padded_for_attn_tp` then indexes that list by
`get_parallel().attn_dp_rank`, which reaches 7 at dp_size 8 →
`IndexError: list index out of range`, killing the scheduler on the first real
forward (the server prints `ready to roll` first, so it looks healthy).

Not hit in the fixed-seq-len runs above, including arm 2 which has
`RANK_SYNC=1`. Reproduced under InferenceX AgentX with
`--enable-prefill-delayer` and the four DP-comm envs at 1 — i.e. it needs an
idle batch to coincide with an extend. A `[1] * len(global_num_tokens)` would
fix the indexing.

## Status on the AgentX (agentic trace replay) scenario

Separate from the numbers above, and **not** yet re-measured with the corrected
EPLB config from this document:

- MegaMoE at conc 32 ran ~27 % below the published non-MegaMoE HiCache arm
  (11,458 vs 15,671 tok/s/GPU). Not a memory artifact — raising the KV pool
  2.42x moved throughput by +0.9 %.
- EPLB runs there failed (OOM at 0.89, RCCL watchdog deadlock at 0.85) but both
  used the *wrong* config: no recorder, 32 redundant experts. Re-test before
  citing.

See `/workspace/claude-skills/agentx/SKILL.md` for the AgentX harness itself.
