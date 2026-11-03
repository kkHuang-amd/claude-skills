# Kimi-K3.1-Pro KVV accuracy (Kimi Vendor Verifier)

Owner node: mi355-4 (2026-11-03). Results: `/shared_nfs/kimi-k31/results/kvv/<tag>/` (append-only table below).

## CONTINUE HERE

**Status (2026-11-03):** KVV set up on mi355-4, no score yet. kvv-k3 snapshot (user's zip
`/sgl-workspace/kvv-k3-k3-20261103.zip`) unpacked to `/sgl-workspace/kvv-bench/kvv-k3-20261103`, `uv sync` done
(inspect_ai 0.3.158), OCRBench (1000) / MMMU-Pro standard 10 options (1730) cached in `/shared_nfs/hf_cache`.
**Next:** off server (`SIM=0 RADIX=1`), `preflight`, then `LIMIT=20 ocrbench` as a smoke run, then the four jobs.
**Blocked:** BEAM judge needs `JUDGE_BASE_URL` / `JUDGE_API_KEY` (official judge kimi-k2.6, external API).
DeepSWE needs docker + Pier + kimi-code >= 0.29 (screenshot: 0.40.1); none in this container.

## Targets (official K3.1-Pro, thinking, reasoning_effort max, temperature 1.0, top-p 0.95)

| Job (kvv run_eval.sh) | max tokens | Official |
|---|---|---|
| toolcall_thinking_effort (`toolcall_thinking_samples.stable2.jsonl`) | 32768 | 0.972 |
| ocrbench_thinking_effort | 16384 | 0.872 |
| MMMU_pro_thinking_effort | 98304 | 0.843 |
| BEAM 1M (plain-text, judge kimi-k2.6) | 65536 (default) | 0.3917 |
| DeepSWE, 113 tasks, Pier, kimi-code 0.40.1, effort max, timeout x50, max-step 20000 | - | 73.8-74.6 @4 (avg output 102-103K tokens) |

## Run

```bash
S=/shared_nfs/kimi-k31/results/pro_20261103/scripts
SIM=0 RADIX=1 bash $S/launch_server.sh kvv_off            # + SGLANG_AMD_MONO_K31_WIDE=1 ... --enable-amd-monokernel for wide
bash $S/run_kvv_k31.sh kvv_off_<date> preflight toolcall ocrbench mmmu beam
JUDGE_BASE_URL=... JUDGE_API_KEY=... bash $S/run_kvv_k31.sh kvv_off_<date> beam_judge
```
`run_kvv_k31.sh` passes the official jobs' arguments (`--thinking --thinking-effort max --think-mode opensource
--stream --max-connections 50`), BEAM with `--plain-text` (K3.1 series), the model's own tokenizer, effort max.
`LIMIT=<n>` for smoke runs (not a score).

## When to run (user, 2026-11-03)

Per kernel change the gate stays gsm8k200 (mono-kernel skill section 3). Run the full KVV set, off and the
changed path on the same weights / draft / image, when a round of optimizations changes many paths, before wide
becomes default, and before a PR. Pass: within noise of off and of the official numbers.

## Results (append-only; node + date on every row)

| Date | Node | Tag | Path | toolcall | OCRBench | MMMU-Pro | BEAM 1M | Notes |
|---|---|---|---|---|---|---|---|---|
