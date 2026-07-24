# DSV4 DP+TBO perf-regression log (mainline)

Running record of DP+TBO (`MODE=dp-tbo`) regression checks after the TBO feature was
merged into sglang mainline. Newest entry on top. Compare each row against the
"reference best" (`TBO_RESEARCH.md` §18.3b canonical, 8192/1024) to catch regressions.

Reference best (§18.3b, 8x MI355X, mem0.9, TBO):
- conc256: **32,592 tok/s** | TTFT 14,363 ms | TPOT 56.14 ms | gsm8k 0.948
- conc512: **41,180 tok/s** | TTFT 34,777 ms | TPOT 73.33 ms

---

## 2026-07-07 02:33 UTC — 1k/1k A/B: Non-TBO vs TBO, conc 128/256/512 (mainline)

Same version/config as prior entries (HEAD `24c42c90b`), workload **1024/1024** (light
prefill, decode-heavy). Non-TBO = `MODE=dp` (uncapped); TBO = `MODE=dp-tbo`
(`GPU_MAX_HW_QUEUES=5`). Client: sglang-oai sweep, np=conc×8, warm=conc×2, ratio 1.0.
gsm8k on the TBO server = **0.9507**. Both modes stable, no HSA crash.

| conc | Non-TBO tok/s | TBO tok/s | Δ tok/s | Non-TBO TTFT/TPOT | TBO TTFT/TPOT | Δ TTFT |
|---:|---:|---:|---:|---|---|---:|
| 128 | 7,627 | **7,737** | **+1.4%** | 2,050 / 31.58 | 1,760 / 31.37 | −14.1% |
| 256 | 12,873 | **13,020** | **+1.1%** | 3,038 / 36.81 | 2,844 / 36.53 | −6.4% |
| 512 | 19,673 | **20,104** | **+2.2%** | 5,165 / 46.58 | 4,911 / 46.00 | −4.9% |

**Takeaway:** at 1k/1k the TBO throughput win is small (**+1–2%**, vs +8–12% at 8k/1k) —
expected, because 1k/1k is decode-bound with a tiny (1024-token) prefill, so there's little
gather/combine comm for TBO to overlap. TBO still cuts **TTFT −5% to −14%** (prefill latency)
and TPOT is flat/slightly better. Both stable at mem0.9. Logs: `/workspace/reg_1k1k_dp_0707/`,
`/workspace/reg_1k1k_tbo_0707/`, `/tmp/{dp,tbo}_1k_sweep.log`.

---

## 2026-07-06 07:19 UTC — A/B: `GPU_MAX_HW_QUEUES=5` vs UNCAPPED (dp-tbo, mainline)

Same version/config as the 07:12 entry below, only difference = whether
`GPU_MAX_HW_QUEUES=5` is set. Question: post the §18 `record_stream`→event+ref fix,
is the queue cap still needed, and does dropping it help or hurt?

**Result: uncapped RUNS fine (no HSA crash, 2048/2048 + 4096/4096 all OK) but is SLOWER.**
The record_stream fix removed the real HSA-exhaustion root cause, so the cap is no longer
needed for *stability* — but for sglang DP+TBO the cap is also a small *throughput win*
(opposite of ATOM in §13, where the cap throttled ATOM ~12%).

| conc | capped (=5) tok/s | uncapped tok/s | Δ | capped TTFT/TPOT | uncapped TTFT/TPOT |
|---:|---:|---:|---:|---|---|
| 256 | **33,253** | 30,441 | **−8.5%** | 14,582 / 54.55 | 16,654 / 58.88 |
| 512 | **42,469** | 37,884 | **−10.8%** | 30,124 / 78.51 | 36,331 / 85.57 |

Uncapped: conc256 2048/2048 in 620.0s; conc512 4096/4096 in 996.4s; no HSA/OUT_OF_RESOURCES
in server log. Logs: `/workspace/reg_tbo_nocap_0706/`, `/tmp/nocap_srv.log`.

**Confirmation — capped repeated (run #2, 07:58 UTC, identical config):**

| conc | capped run#1 | capped run#2 | run-to-run var | uncapped | cap effect (avg) |
|---:|---:|---:|---:|---:|---:|
| 256 | 33,253 | 33,119 | −0.4% | 30,441 | **+9.0%** |
| 512 | 42,469 | 42,371 | −0.2% | 37,884 | **+11.9%** |

Cross-run variance for capped is <0.5% (≪ the 9–12% cap effect) → **the cap win is REAL,
not noise.** capped run#2 logs: `/workspace/reg_tbo_cap2_0706/`, `/tmp/cap2_srv.log`.

**Takeaway: keep `GPU_MAX_HW_QUEUES=5` for DP+TBO — it's now a perf knob (+~9–12%), not
just a stability band-aid.**

---

## 2026-07-06 07:12 UTC — mainline post-merge check ✅ NO REGRESSION

**Version**
- Repo: `/sgl-workspace/sglang-upstream`, branch `main`, up to date with `origin/main`.
- HEAD: `24c42c90b` (`24c42c90be82c24433ab10e65db24e52a48d8084`) — "Clean up ServerArgs post-init dispatch (#30186)", committed 2026-07-05 23:05 -0700.
- TBO feature merge commit in history: `81735ecf8` "[AMD] Feat/dsv4 ep tbo prefill (#29362)".
- Working-tree edits (uncommitted):
  - `python/sglang/srt/configs/cohere2_moe.py` — `@strict`→no-op (import fix, required).
  - `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` — decode small-cap mori op
    (`SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS`, default 0 = off). **Inert** for `MODE=dp-tbo`
    (non-EP DP path; mori not used).

**Config** — `run_sgl_dsv4_unified.sh MODE=dp-tbo` defaults:
- `--tp 8 --dp 8 --enable-dp-attention --enable-prefill-delayer --enable-two-batch-overlap`
- `--attention-backend dsv4 --kv-cache-dtype fp8_e4m3 --page-size 256`
- `--mem-fraction-static 0.90 --chunked-prefill-size 65536` (=8192/rank) `--cuda-graph-max-bs 1024 --max-running-requests 1024 --swa-full-tokens-ratio 0.15`
- `--disable-radix-cache --disable-shared-experts-fusion`
- env: `GPU_MAX_HW_QUEUES=5`, `SGLANG_USE_ROCM700A=0`, gatherv+reduce-scatter+SE-local ON, `PYTHONPATH` pinned to upstream tree.
- Model: `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro`. HW: 8x MI355X (gfx950).

**Commands**
```bash
# 1. launch (one server, reused for gsm8k + perf)
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4
PORT=8000 MODE=dp-tbo bash run_sgl_dsv4_unified.sh    # ready in ~150s (weights cached)

# 2. gsm8k 5-shot
bash lm-eval.sh                                        # local-completions, num_concurrent=64

# 3. perf conc256 + conc512 (canonical 8192/1024, sglang-oai client)
WORKLOADS="8192:1024" CONCS="256 512" \
  RESULT_DIR=/workspace/reg_tbo_0706 \
  bash sweep_dsv4_sglang_client.sh                     # np=conc*8, warm=conc*2, ratio 1.0, rate inf
```

**gsm8k:** flexible-extract **0.9507**, strict-match **0.9507** (limit 1319). ✅ (ref ~0.95)

**Perf results (8192/1024, sglang-oai, np=conc×8, warm=conc×2):**

| conc | Total tok/s | vs ref | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | duration (s) | reqs | crash |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 256 | **33,253** | +2.0% | 3,695 | 14,582 | 54.55 | 567.6 | 2048/2048 | none |
| 512 | **42,469** | +3.1% | 4,719 | 30,124 | 78.51 | 888.9 | 4096/4096 | none |

**Verdict:** No regression after mainline integration — both concurrencies meet/slightly
exceed the reference best, gsm8k unchanged (0.9507), no HSA crash at mem0.9. Also consistent
with the 2026-07-02 `perf_reg.sh` baseline (8000/1000, backend=sglang): conc256 32,740 / conc512 42,329.

**Logs:** `/workspace/reg_tbo_0706/`, `/tmp/gsm8k.log`, `/tmp/reg_srv.log`.
