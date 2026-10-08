# OPUS sparse prefill for DSV4.1 HIP (gfx950) -- local port (DONE, superseded)

Paths are relative to `/sgl-workspace/sglang-dsv41/python/sglang` unless stated. Full design Q&A (current-path
call chain, index semantics, OPUS contract, vLLM approach, overhead table, test plan):
`/shared_nfs/kk/results/DeepSeek-V4.1-Flash/doc_backup_20260929/OPUS_PORT.md` (sections Q1-Q5, Design, Implementation steps).

## CONTINUE HERE
**Status: DONE; superseded in practice.** The current best config uses the RolaoDenthu opt-branch's own OPUS
prefill (`SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1`, see ATOM_PORT.md / SKILL.md). Our implementation is kept as
`patches/sglang_share_opus_prefill_0001.patch` (OPUS) + `_0002.patch` (= upstream #41159, KV-store int64 loc);
local-branch form: `patches/sglang_local_opus_prefill_*.patch` on branch `opus-prefill` (RUNBOOK.md).
No open work.

## What was built (patch 0001)
- Why: the default V4.1 HIP path (`SGLANG_DSV4_KV_LAYOUT=v4`, 584 B packed fp8 rows) runs the DECODE kernel
  aiter `pa_decode_sparse` (kv_splits=1 for n >= 1024) for prefill too (`srt/layers/attention/hip_flash_mla.py`).
  In a 16k-token prefill profile it was 21% of the chunk (NOTES.md, c32 TTFT). The existing OPUS wrapper in the
  unified-KV path (`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`) is rejected for V4.1 by
  `validate_deepseek_v41_features` (`srt/arg_groups/deepseek_v4_hook.py`) -> a port was needed.
- Approach (vLLM-like): per forward, dequant-gather a request-local bf16 workspace (all visible compressed
  positions `[0, seq_len//R)` + SWA tail), build a CSR from the indexer's request-local `c{R}_sparse_raw_indices`
  plus SWA position arithmetic (no `unique`/sort), call aiter OPUS bf16 once per layer, then `_apply_inverse_rope`.
  fp8 e4m3 x E8M0 fits bf16 exactly, so the KV is bit-exact; only accumulation order differs.
- `kernels/ops/attention/dsv4/opus_prefill_hip.py`: Triton `_dequant_gather_kernel` (584 B row -> bf16 [512]),
  Triton `_csr_fill_kernel`, `OpusPrefillRows` (layer-invariant plan: compressed + SWA source slots, bases;
  rebuilt when layer ids restart, robust to BCG object reuse), `opus_sparse_prefill`.
- `srt/layers/attention/deepseek_v4_backend_hip_radix.py`: `_opus_prefill_eligible` / `_opus_prefill`, called in
  `_forward_attention` before the aiter_sparse length fold.
- Test: `test/registered/kernels/ops/attention/dsv4/test_hip_opus_prefill.py`.

## Gating / knobs (`srt/environ.py`)
| env | default | meaning |
|---|---|---|
| `SGLANG_OPT_DSV41_OPUS_PREFILL` | False | enable |
| `SGLANG_OPT_DSV41_OPUS_PREFILL_MIN_TOKENS` | 2048 | min tokens in batch (1x1024 was 0.5-0.9x = slower) |
| `SGLANG_OPT_DSV41_OPUS_PREFILL_MAX_WS_ROWS` | 524288 | workspace row cap (bf16, 1 KB/row); above -> fallback |
| `SGLANG_DEBUG_DSV41_OPUS_PREFILL_CHECK` | False | bounds-check every CSR index (syncs; debug only) |

Gate: forward mode EXTEND or MIXED (MIXED running rows are length-1 extends; target-verify/draft-extend
excluded), V4.1 low ratios, T >= MIN_TOKENS, T == sum(extend lens) == positions, 584 B layout, no CP, no SWA
replay / request_window, no late-layer tail, not stream-capturing (breakable prefill graphs run attention
eagerly, so it works under `--cuda-graph-backend-prefill breakable`). Not covered: `SGLANG_DSV4_KV_LAYOUT=v41`
(528/288 B rows via `compact_attention_hip`; would need new gather kernels).

## OPUS contract gotchas (aiter `ops/pa_sparse_prefill_opus.py`)
- No -1/padding support and NO bounds check: every CSR entry must be a valid row, or the GPU faults. Count valid
  entries explicitly (clamp-1 `c{R}_sparse_topk_lengths` gives length 1 with index -1 for pos < R-1), and derive
  SWA counts from the same floor as the gather.
- q/kv/out same dtype, D == 512, `attn_sink` fp32 [H], `softmax_scale` passed as-is, prefix and extend regions
  must share `stride(0)` (we pass an empty extend region). Output excludes inverse RoPE.
- First call on a fresh aiter JIT-builds `module_mla_v4_prefill_opus.so` (minutes).
- Likely cause of the vLLM OPUS + prefix-cache crash (unverified hypothesis): -1 / negative CSR entries after a
  prefix hit (backup file, section "vLLM approach").

## Validation (2026-09-24, 4xMI355X TP4, random-ids ISL4096/OSL1024)
- Unit: 3 requests incl. a 5000-token prefix, R=0/1/2 vs `aiter_sparse_decode_fwd`: cos >= 0.9999973,
  max abs <= 0.0039; MIXED-batch case (length-1 rows) also passes.
- Microbench OPUS vs current kernel, per layer (H=16, topk 512 + SWA 128): T=1024 0.176 -> 0.076 ms,
  T=4096 0.673 -> 0.282, T=16384 2.687 -> 1.084 (2.2-2.5x). Full path incl. gather/CSR
  (`scripts/bench_opus_prefill_path.py`): 4x4096 R0 0.77 -> 0.37, R1 2.63 -> 1.00, R2 2.56 -> 0.93 ms; plan build
  0.3-0.7 ms once per forward.
- E2E, DSpark off, QR=NONE: GSM8K 0.902 with CHECK=1 (baseline 0.905-0.908), OPUS on all 4 ranks, 0 faults;
  TTFT c1/c8/c32 164/841/2768 -> 154/827/2484 ms (c32 -10.2%); out tok/s c32 2198 -> 2275 (+3.5%); TPOT unchanged.
- Combined (commit 8de5d6cfe3): OPUS + QR=INT8, DSpark off: c32 TTFT 2273 ms (-17.9%), 2337 tok/s (+6.3%).
  DSpark sim 3.51 + OPUS: c32 TTFT 2736 -> 2485 (-9.2%), 3160 -> 3340 tok/s. With `--enable-mixed-chunk` and
  MIXED via OPUS: c32 TTFT 1520 ms, GSM8K 0.911, 0 faults (details: NOTES.md, VLLM_COMPARE.md).

## Known issues
- AgentX TP2 faults at c16/c64 (2026-09-25, "Write access to a read-only page") were NOT OPUS: OPUS=0 c64
  faulted too. Root cause = int32 `loc` offset overflow in the Triton DSv4 KV-store kernels
  (`kernels/ops/kvcache/triton_store_cache.py`), fixed by upstream #41159 (= patch 0002). With it, TP2 EP1 3600 s
  runs are fault-free (results/agentx.md). Runs before the fix with loc > ~3.67M tokens may have corrupted KV.
- Repro (unit): `HIP_VISIBLE_DEVICES=4 PYTHONPATH=/sgl-workspace/sglang-dsv41/python python3 test/registered/kernels/ops/attention/dsv4/test_hip_opus_prefill.py`
