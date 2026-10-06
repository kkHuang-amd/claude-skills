# B200 vLLM TP4 profile + microbench for DSV4.1-Flash (answer to B200_REQUEST_1006.md rev 2)

Owner node: dgx-025

## CONTINUE HERE

**Status:** P1a done. P1b (D64 c8 clean + profile) running; then c16 (needs a server restart with the c16 recipe entry), P2, P3.
**Next:** P1b c8 -> restart server with `serve.sh 3 c16` -> P1b c16 -> P2 (on c8 server) -> P3.
**Files:** `dsv41/scripts/b200_tp4_1006/` (`serve.sh`, `prefix_sweep.py`, `decode_run.sh`, `extend_prof.py`, recipe copy).
**Repro (P1a):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dsv41/scripts/b200_tp4_1006/serve.sh 2 c8 > server_c8.log 2>&1 &
# wait for "OpenAI server is ready" (Rust frontend; there is no "Application startup complete")
python3 dsv41/scripts/b200_tp4_1006/prefix_sweep.py
```

## Environment

- Container `vllm/vllm-openai:nightly-ac9126e58aa7bbab1856ba6593ba4d5003fea516` (same vLLM commit as the requested
  `nightly-dev-x86_64-cu130-ac9126e58aa7`; image digest is not visible from inside the container).
  `vllm 0.30.1rc1.dev493+gac9126e58`, torch 2.13.0+cu130, flashinfer 0.7.0.post1.
- 4x B200 (GPU 0-3 of 8), driver 580.173.02, power limit 1000 W, max clocks SM 1965 / mem 3996 MHz.
- Weights: `deepseek-ai/DeepSeek-V4.1-Flash` from HF (48 shards).

### Recipe deviations (read before comparing)

- Commit `a8504a430` is not in the public InferenceX repo, so args come from origin/main `5ff11ab20`
  (`inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/b200-fp4-mtp/agentic.yaml`, copied into
  the scripts dir): base args + `zip_override_tp4` entry for the conc. That includes `--tokenizer-mode deepseek_v41`,
  the deepseek_v41 tool/reasoning parsers and `--kernel-config '{"enable_flashinfer_autotune":true}'`.
- Speculative config: `rejection_sample_method: synthetic`, `synthetic_acceptance_length: 3.51` (as requested; the
  InferenceX throughput harness makes the same substitution, see `runners/test_dsv41flash_capture.py`).
- On main the TP4 zip maps conc -> max-num-seqs as c1:8, c8:16, c16:32 (the request text says 8|16 for c8|c16).
  I follow main: the **c8 server** is entry idx 2 (`max-num-seqs 16`, `max-num-batched-tokens 8192`,
  capture size 8190) and also serves P1a and P2; c16 uses entry idx 3 (`max-num-seqs 32`, compilation config adds
  `decoder_replay_cudagraph_capture_sizes`); c1 (P3) uses entry idx 0 (`max-num-seqs 8`, `max-num-batched-tokens 4096`).
- `--profiler-config` (torch, with_stack off, record_shapes on, ignore_frontend, max_iterations 40) is always passed;
  it is inactive outside `/start_profile`..`/stop_profile`.
- Startup: weights 253 s, CUDA graph capture 216 s, then FlashInfer autotune ~20 min; KV cache 78.4 GiB/GPU
  (46.2M tokens).

## P1a -- cached-prefix TTFT sweep (Deliverable 1b), conc 1, c8 server

Method: for each (L, N) a fresh random-token-id prefix P_L (ids in [1000, 120000)), warm with max_tokens 1, then
3x `P_L + N fresh tokens` (max_tokens 1, streamed `/v1/completions`), client-side TTFT, median reported.
Cache hit verified per request from `vllm:prefix_cache_hits_total` / `queries_total` deltas: every measured request hit
exactly L tokens out of L+N.

Median TTFT (ms), 2026-10-06, dgx-025:

| L \ N | 512 | 4096 |
|------:|----:|-----:|
| 32k   | 58.6 | 95.7 |
| 64k   | 63.5 | 113.7 |
| 128k  | 88.6 | 143.4 |
| 256k  | 147.9 | 199.9 |

Linear fit TTFT = a + b x L:
- N=512: **a = 40.3 ms, b = 0.412 ms per 1k prefix**
- N=4096: a = 83.0 ms, b = 0.460 ms per 1k prefix

This matches the InferenceX-trace fit for B200 (58 ms + 0.39 ms/1k). The 3.6k extra new tokens cost ~43 ms at every L.
Raw CSV: `/shared_nfs/kk/dsv41_b200/p1a_sweep.csv` on dgx-025 (first 64k/512 rep was 102.7 ms, an outlier; median
unaffected).
