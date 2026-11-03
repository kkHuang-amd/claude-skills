# Kimi-K3.1-Pro KVV accuracy (Kimi Vendor Verifier)

Owner node: mi355-4 (2026-11-03). Results: `/shared_nfs/kimi-k31/results/kvv/<tag>/` (append-only table below).

## CONTINUE HERE

**Status (2026-11-03):** KVV set up on mi355-4, no score yet. kvv-k3 snapshot (user's zip
`/sgl-workspace/kvv-k3-k3-20261103.zip`) unpacked to `/sgl-workspace/kvv-bench/kvv-k3-20261103`, `uv sync` done
(inspect_ai 0.3.158), OCRBench (1000) / MMMU-Pro standard 10 options (1730) cached in `/shared_nfs/hf_cache`.
**15:25:** off server `server_kvv_off.log` (SIM=0, RADIX=1, context 1M, mi355-4) -> `run_kvv_k31.sh kvv_off_20261103
ocrbench`: off 0.873, wide 0.869 (+- 0.011; official 0.872): tie. MMMU-Pro postponed (user, 15:34).
**Blocked:** the zip's LFS files are pointers only (130-byte stubs). Toolcall:
`toolcall_benchmark/toolcall_thinking_samples.stable2.jsonl` (30,161,019 B, sha256 7db15a81c84952a0...) exists only
in the kvv-k3 repo's LFS (needs repo access, or a copy; K3's checkout was `/sgl-workspace/kvv-bench/kvv-k3-0727-update`
on the K3 node). eval.py has no `--limit` (toolcall: `SAMPLE=<id>` for one sample).
BEAM 1M data rebuilt (16:30) from the public BEAM repo (`/shared_nfs/kimi-k31/kvv_data/BEAM`, github
mohammadtavakoli78/BEAM @ b2da22e, `chats/1M`) with kvv's `beam/prepare_data.py` -> `kvv_data/beam_prepared`, copied
into `beam/data/`: questions sha256 = the LFS oid (0882ca1e...); chats .gz same size to the byte (35,465,655),
sha differs only by the gzip header. BEAM generate can run; its judge still needs JUDGE_BASE_URL / KEY.
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
| 2026-11-03 | mi355-4 | kvv_off_20261103 | off | - | 0.873 +- 0.011 | - | - | DFLASH step55000 nd6, SIM=0 (accept ~3.35), radix on; 5:29, 1.43 M tokens |
| 2026-11-03 | mi355-4 | kvv_wide_20261103 | wide (MONO_KERNEL 10.34 worktree) | - | 0.869 +- 0.011 | - | - | same server args + WIDE=1; 6:19, 1.47 M tokens; wide steps up to 288 rows; no fault |
| 2026-11-04 | mi355-4 | kvv_wide_inp1_20261104 | wide (MONO_KERNEL 10.37, sgl-amd-mono ce9e2a6) | - | 0.872 +- 0.011 | - | - | same server args + WIDE=1 (`scripts/k31_kvv_inp1_ocr.sh`); 6:30, 1.48 M tokens; first wide step 288 rows, mostly 43-46 requests; no fault |
