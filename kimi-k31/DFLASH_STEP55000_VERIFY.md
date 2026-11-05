# K3.1-Pro + DFlash2 step55000 draft: accuracy / perf check vs documented DFLASH numbers

Owner node: crsuse2-m2m-v3-131. Started 2026-11-05.

## CONTINUE HERE

**Status (2026-11-05 06:37, crsuse2-m2m-v3-131): DONE.** Cookbook low-latency (TP8) + 3 SkyOpt switches + mono +
step55000, mem 0.90: gsm8k200 0.98 (accept 4.345) PASS; c4 TPOT 7.64 ms = 10.39 (+0.7%); mono vs off TPOT x acc
c1 -29.5% / c4 -4.9%. Open: mono-only first-request TTFT outlier (3.5-5.7 s after flush). Results below.
Scripts: `/tmp/k31v/launch_ll.sh`, `pipeline_ll.sh` (need `export -f rg`). Older DCP8 plan (runs 0-4) not run.
**Next:** runs 1-4 below (AgentX config of MONO_KERNEL.md 10.32-10.39: 6 draft tokens, SIM 3.399, DCP8).
Mono on = `--enable-amd-monokernel` (local sgl-amd-mono e52803f has no SGLANG_AMD_MONO_K31_WIDE knob: wide
covers 2..48-request verify steps whenever mono is on; reference used fork k31/mono-s1 601a423).
```text
run  config                         measure                       reference (mi355-4, 10.32 / 10.38 / 10.39)
0    8 tok, off, SIM=0              gsm8k200 T16                  0.985, accept 5.07 (10.32)
1    6 tok, off, SIM=0              gsm8k200 T48 / T16            0.98 / 0.99 (image aiter, 10.39), accept ~4.35
2    6 tok, mono, SIM=0             gsm8k200 T48 / T16            0.985 / 0.99 (image aiter, 10.39)
3    6 tok, off, SIM=1 3.399        ab c4/c8/c16/c32/c48 + pf 8k  TPOT 8.33/11.14/15.77/23.24/30.71, pf 459-461
4    6 tok, mono, SIM=1 3.399       same                          TPOT 8.07/10.68/15.22/22.27/29.80 (10.38);
                                                                  TPOT x acc vs off -3.3/-3.9/-3.5/-4.4/-3.3% (10.39 img)
```
**Files:** node-local copies in `/tmp/k31v/` (launch_server.sh / ab_bench.sh / gsm8k_gate.sh, sourced
`/tmp/k31v/env.sh` instead of `../env.sh`; only paths changed). sgl-eval 0.1.3 + math_verify unpacked into
`/tmp/sgleval_pkg`, `/tmp/sgleval_deps` (on PYTHONPATH, venv untouched).
**Repro:**
```bash
D=/shared_nfs/RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024-step55000
SIM=0 /tmp/k31v/launch_server.sh <tag> --speculative-algorithm DFLASH --speculative-draft-model-path $D
TEMP=0 THREADS=16 /tmp/k31v/gsm8k_gate.sh <tag> $K31_OUT/logs/server_<tag>.log
```
**Pass criteria:** gsm8k200 >= 0.975 (MONO_KERNEL gate); TPOT within ~2% of the reference rows above;
mono vs off TPOT x acc about -3..-4% at every bucket. (10.27 numbers = Step29301 draft, 8 tokens, SIM 3.84: not comparable.)

## Setup (this node)

- Model `/shared_nfs/models/Kimi-K3.1-Pro` (verified vs HF 6d0a020295, 112 files, sizes match).
- Draft: HF `RadixArk/K31-DFlash2-MLA5-Preview1027-Step29301-SWA1024` revision `step55000`
  (sha 1e8f8811c6, SHA256SUMS OK). NOT the draft used for the reference numbers.
- Stack (installed image, user: up to date, do not update): sglang 0.5.23.dev0+k31pr48490.380ee3f,
  amd-aiter 285579b62, flydsl 0.4.0, torch 2.11.0+rocm10.0.0. Reference used sglang-k31-amd dev_amd
  b588cf7952 + aiter-k31 f0b96f49e + flydsl 0.4.1.
- Results: `/shared_nfs/kk/results/Kimi-K3.1-Pro/dflash-step55000-20261105/` (`logs/`, `results/acc/`, `results/ab/`).

## Cookbook recipe (sglang main docs, fetched 2026-11-05)

Source: `docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx`, cell
`{hw: mi355x, pdMode: unified, strategy: low-latency}` (verified with DFLASH; 1-4 agents) + DFLASH spec flags:
```bash
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 SGLANG_AMD_MONO_K31_WIDE=1 \
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95 SGLANG_ROCM_RUNTIME_RESERVE_GB=6 \
SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 HSA_USERPTR_FOR_PAGED_MEM=0 \
python -m sglang.launch_server --trust-remote-code --model-path <K3.1-Pro> --tp-size 8 \
  --mem-fraction-static 0.85 --chunked-prefill-size 16384 --max-running-requests 4 --cuda-graph-max-bs-decode 4 \
  --schedule-policy lpm --prefill-decode-interval 16 --max-mamba-cache-size 60 \
  --mamba-radix-cache-strategy extra_buffer_lazy --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096 --enable-amd-monokernel \
  --weight-loader-prefetch-checkpoints --model-loader-extra-config '{"enable_multithread_load": true}' \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --speculative-algorithm DFLASH --speculative-draft-model-path RadixArk/Kimi-K3.1-Dflash2-Preview \
  --speculative-dflash-block-size 6 --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --speculative-draft-attention-backend triton --enable-linear-replayssm-spec
```
(K3.1 forces kv fp8_e4m3 / page 64 / dsa backend. No HiCache: mono rejects it. 8-56 agents = DCP8 cell, no mono except at 8.)

SkyOpt switches in the installed sglang (env, default off, prefill batches only; MONO_KERNEL 10.39 ran them at pure TP8):
`SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_FRONT=1 SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_O_PROJ=1 SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1`
(o_proj: eager batches only). Row-sharded MoE tail (old PR #8) is built in at >= 4096 tokens. SkyOpt's DFlash fc shard
= `SGLANG_OPT_DFLASH_TP_AUX_FC` (default follows SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION). 10.39 run config
also: mem 0.90, context 32768, `--min-free-slots-delay 1` at c16 / c48.

## Results

(append rows: node, date)

### Perf: cookbook low-latency + SkyOpt switches + mono, mem 0.90, step55000 (crsuse2-m2m-v3-131, 2026-11-05)

`results/ab/ll_perf_mono` (ab_bench, SIM=1 3.399, AB_CONC "1 4", bench_serving random 8k/1k, n = 4c), 0 faults:
```text
case        TPOT mean / median ms   P90 ITL ms   TTFT mean / median ms   acc    out tok/s
decode_c1   3.84 / 3.83             4.39         1887 / 678              3.41   176.0
decode_c4   7.73 / 7.64             7.81         1112 / 680              3.38   450.6
prefill_8k  -                       -            1182 / 366              -      -
```
vs MONO_KERNEL 10.39 SkyOpt + wide (mi355-4, InferenceX 8k1k client): c4 TPOT p50 7.59 (here 7.64, +0.7%), TTFT p50
360 (prefill 8k median 366). Means are inflated by ONE slow first request per case after `/flush_cache`
(p99 TTFT 3.5-5.7 s); medians are the comparable numbers. No c1 reference for this config.
Server ready 160-320 s with a warm page cache.

### Accuracy: same config, SIM=0 (crsuse2-m2m-v3-131, 2026-11-05)

`results/acc/ll_acc_mono_t16`: gsm8k200 TEMP 0 THREADS 16 (server caps 4 running): **0.98**, truncated 0, error 0,
accept len req-weighted 4.345 (mean 4.354), wall 122 s, 0 faults. Reference (10.32 / 10.38, DCP8, 6 tokens, step55000):
0.985-0.99, accept 4.30-4.36 -> PASS (gate >= 0.975).
Eval setup on this node: sgl-eval 0.1.3 needs editdistance / nltk / wonderwords / pytest-asyncio / math_verify;
installed `--no-deps --target /tmp/sgleval_deps` (venv untouched); without them the registry is silently empty
("Eval `gsm8k` not found. Available: (none registered)").

### Perf A/B: mono on vs off, same config otherwise (crsuse2-m2m-v3-131, 2026-11-05 06:17 / 06:34)

`results/ab/ll_perf_{mono,off}`, SIM=1 3.399, 0 faults both:
```text
case        TPOT median ms (mean)      acc (on/off)   TPOT x acc (means)   TTFT median ms (mean)
            on            off                         on vs off            on            off
decode_c1   3.83 (3.84)   5.44 (5.43)  3.41 / 3.42    -29.5%               678 (1887)    381 (393)
decode_c4   7.64 (7.73)   8.10 (8.08)  3.38 / 3.40    -4.9%                680 (1112)    682 (789)
prefill_8k  -             -            -              -                    366 (1182)    362 (358)
```
vs 10.39 SkyOpt (mi355-4): c4 TPOT p50 off 7.93 -> wide 7.59 (-4.3%); here off 8.10 (+2.1%), on 7.64 (+0.7%).
Mono-only TTFT outlier: with mono on, the first request of each case after `/flush_cache` takes 3.5-5.7 s
(p99); off has none. Medians unaffected. Not investigated (candidate: per-case mono warmup / first-launch cost).
