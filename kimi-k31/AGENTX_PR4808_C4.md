# K3.1-Pro AgentX c4: InferenceX PR 4808 recipe vs + SkyOpt flags (MI355X)

Owner node: crsuse2-m2m-v3-131. Started 2026-11-05.

## CONTINUE HERE

**Status (2026-11-05 17:00, crsuse2-m2m-v3-131): matrix DONE, GPUs free.** Keep mamba150 (c4 +9.8 %), drop
draft8 (c1 -7.2 %, c4 -2.8 %); see "Summary of the gap matrix" at the end. Next candidates: replicate mamba150,
mamba 200+, c1 without `--prefill-decode-interval 16` / lpm (B300 sets neither).
(Earlier, 12:20:) gap-closing matrix ran, `driver_matrix.frozen.sh` pid 130425,
OUT=`/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-gap-20261105` (`status.txt`), points in order:
c4-mamba150 (`--max-mamba-cache-size 150`; KV 1.44 M tokens, ready 12:17) -> c1-base -> c1-draft8 -> c4-draft8
(draft8 = `--speculative-num-draft-tokens 8` + SIM 3.68 as B300). Each one variable vs c4-base (run above) / c1-base.
Done before: c4 base reproduces CI; SkyOpt opt arm invalid (KV 474 k tokens, 400s on >474 k inputs); mono cannot
take HiCache (spec lacks KV_OFFLOAD); PR's c8-c48 HiCache recipe exceeds this node's GTT (1375 GB = RAM/2).
**Next:** read each END line; compare hit / tok/s/GPU / P90 intvty / ITL; one 3600 s arm resolves ~5-6 % only.
**Files (node-local, /shared_nfs/kk/agentx-k31/):** `points/c4-{base,opt}/{server_env,server_args,client_env}`,
`run_point_k31.sh` (copy of `agentx/e2e/run_point.sh`: Kimi model, InferenceX-latest, runtime dir
`/workspace/agentx-runtime-latest`), `driver_c4.sh`; CI log / artifacts under `ci/`.
**Repro:**
```bash
B=/shared_nfs/kk/agentx-k31; cd /workspace
setsid nohup bash $B/driver_c4.frozen.sh < /dev/null > $OUT/driver.log 2>&1 &
# OUT=/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-pr4808-c4-20261105
```
**Pass / read:** single 3600 s arm resolves ~5-10 % only (agentx skill: replicate spread 5.67 %); quote
total tok/s/GPU and P90 interactivity; overall cache hit, not device-tier.

## Setup

- Client: InferenceX `pr-4808` @ 7ad0ba79c (`/workspace/InferenceX-latest`, main was 2ad2c9f97 2026-11-04).
  New layout: client is `python3 -m infx.bench agentic` (`srt_agentic.sh`), aiperf = PyPI `agentx-harness==1.0.13`
  in a uv venv under `AIPERF_RUNTIME_DIR` (no aiperf submodule any more; `AIPERF_WARMUP_REQUESTS_PER_LANE` is not read).
- Point = CI job 127116611069 (run 41977056700, PR 4808 TP8 c4, passed 1h40m) via `agentx/e2e/extract_point.sh`
  (raw extract kept at `agentx/e2e/points/ci41977056700/k31pro-c4`). Local edits only: model / draft local paths,
  `--speculative-draft-model-revision` dropped, split `--model-loader-extra-config` JSON rejoined.
  CI injects `SGLANG_SIMULATE_ACC_LEN=3.43` (`match-expected`, `real-draft-token`) -- not in the recipe yaml.
- Draft: `RadixArk/Kimi-K3.1-Dflash2-Preview` @ d0d8bc2e -> `/shared_nfs/RadixArk/Kimi-K3.1-Dflash2-Preview-d0d8bc2e`
  (sha256 1e07d235..., NOT the same weights as step55000 e883bba6...).
- Server stack: image sglang 0.5.23.dev0+k31pr48490.380ee3f, aiter 285579b62, sgl-amd-mono e52803f (CI image is
  `lmsysorg/sglang:dev-kimi-k31-pro-mi35x@sha256:6af91676...`; not checked to be identical).
- opt arm = base + `SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_{FRONT,O_PROJ,STREAM}=1` + mem 0.90 (0.85 leaves no KV with
  these switches, see `DFLASH_STEP55000_VERIFY.md`). Everything else identical.

## References (CI artifacts)

```text
c4                               tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL p90 ms  ISL     hit    dur
MI355X CI (PR 4808, 41977056700)  3005.9     129.6       0.41 / 2.40      7.72        176637  0.887  3617
B300 SGLang CI (41932307764)      3424.8     189.0       0.40 / 1.19      5.29        174932  0.973  3595
```
B300 c4 runs HiCache (kvdram); not the same config. (User called 41932307764 "vLLM"; its title and artifacts are
SGLang on B300.)

## Results

(append rows: node, date)

### base (PR 4808 c4 as CI ran it), crsuse2-m2m-v3-131, 2026-11-05 08:58-10:05, rc=0, 0 errors

```text
                    tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL p90 ms  ISL     hit    dur
this node, base     2973.7     133.8       0.42 / 2.54      7.47        168937  0.883  3620
MI355X CI c4        3005.9     129.6       0.41 / 2.40      7.72        176637  0.887  3617
delta vs CI         -1.1 %     +3.2 %                                    -4.4 %
```
Reproduces CI within noise (single-arm spread ~5.7 %); ISL 4.4 % lower (trace mix at a fixed window).
Server accept len in decode lines ~3.42 (SIM 3.43).

Trap (new client): `Runtime.active()` is only true when already running the venv python, so EVERY run
`rmtree`s and rebuilds the uv venv under `AIPERF_RUNTIME_DIR`. The opt arm's rebuild on /workspace hit
"Disk quota exceeded (os error 122)" with df still showing 309 G free (quota, not df) -> client exit 2 at
start. Fix: `AIPERF_RUNTIME_DIR=/shared_nfs/kk/agentx-runtime-latest`; opt rerun 10:18.

### Gap to B300 SGLang (CI artifacts, run 41977056700 vs 41932307764) -- analysis, crsuse2-m2m-v3-131, 2026-11-05

```text
        tok/s/GPU  P90 intvty  ITL mean/p90 ms  overall hit (gpu/cpu)    SIM acc  draft tok
MI c1   2191.9     175         4.26 / 5.71      0.971                    3.43     6
B300 c1 2370.0     241         3.71 / 4.15      0.974 (0.974/0.000)      3.68     8
MI c4   3005.9     130         5.66 / 7.72      0.887                    3.43     6
B300 c4 3424.8     189         4.40 / 5.29      0.973 (0.661/0.312)      3.68     8
```
- c1: cache hit already equal. ITL mean gap 14.8 %, of which ~7.3 % is the simulated acceptance alone
  (3.68 / 3.43, B300 runs 8 draft tokens); per-step time gap ~7 %. ITL p90 gap (+38 %) is larger than mean.
- c4: hit 0.887 vs 0.973 is the KDA-state (mamba) pool, not KV. c4 base server metrics (TP0, profiling):
  `mamba_available_tokens` p50 0 / max 4 of `--max-mamba-cache-size 60`, `mamba_evictable` 50-59 (all slots
  hold cached prefix states and get evicted); KV evicted only 2.10 M tokens vs 75.5 M cached (2.8 %), kv
  evictable p50 1.44 M of 1.76 M. A hybrid prefix hit needs the KDA state at that position, so evicted
  states = misses. B300 keeps them in host DRAM (HiCache); mono cannot (spec has no KV_OFFLOAD).
- Mamba slot cost here: ssm 3.08 GB + conv 0.11 GB for 60 slots = ~53 MB / slot / rank.
- B300 c1 flags of note: `--speculative-num-draft-tokens 8`, `--mamba-radix-cache-strategy extra_buffer`,
  `--mamba-track-interval 256`, `--mamba-ssm-dtype float32`, `--mamba-max-states-per-path 1`, no
  `--prefill-decode-interval` / lpm, `--max-running-requests 2`, HiCache 230 GB.

### Gap matrix results (crsuse2-m2m-v3-131, 2026-11-05, OUT=/shared_nfs/kk/results/Kimi-K3.1-Pro/agentx-gap-20261105)

```text
c4                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors
base (08:58)      2973.7     133.8       0.42 / 2.54      5.78 / 7.47      168937  0.883  0/506
mamba150 (12:15)  3265.7     143.1       0.37 / 1.05      5.45 / 6.99      172089  0.951  1/545
delta             +9.8 %     +7.0 %      -12 % / -59 %    -5.7 % / -6.4 %  +1.9 %  +0.068
B300 c4 (CI)      3424.8     189.0       0.40 / 1.19      4.40 / 5.29      174932  0.973
```
mamba150: hit 0.883 -> 0.951 (B300 0.973); gap to B300 tok/s/GPU 13.2 % -> 4.6 %. Mamba slots no longer pinned
(available p50 0 -> 7, p90 43; min still 0); KV evicted 2.10 M -> 4.18 M tokens (KV 1.76 M -> 1.44 M), harmless.
The one error: InvalidInferenceResultError (a response with no content, only usage/[DONE]); no server fault.
+9.8 % is above the 5.67 % replicate spread but below the 10 % bar; every secondary metric moved the same way.

```text
c1                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors
base (13:22)      2186.0     174.7       0.49 / 0.70      4.30 / 5.72      251408  0.971  0/249
MI355X CI c1      2191.9     175         0.49 / 0.71      4.26 / 5.71      251408  0.971
B300 c1 (CI)      2370.0     241         0.56 / 0.77      3.71 / 4.15      260660  0.974
```
c1-base reproduces CI c1 to 0.3 % (same ISL to the token: same traces at c1).

```text
c1                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors  SIM acc
base (13:22)      2186.0     174.7       0.49 / 0.70      4.30 / 5.72      251408  0.971  0/249   3.43
draft8 (14:29)    2028.2     165.7       0.49 / 0.71      4.76 / 6.03      242287  0.969  0/240   3.68
delta             -7.2 %     -5.2 %                       +10.7 % / +5.4 %
```
draft8 LOSES at c1: per decode step (ITL mean x acc) 14.7 -> 17.5 ms (+19 %) for +7.3 % acceptance. 8-token
verify costs more than it returns on this stack at the CI-simulated acceptance. Break-even needs ~+19 % real
acceptance (gsm8k step55000 6 -> 8 tokens measured +17 %, 4.35 -> 5.07: close, not over). Server warmup with
8 tokens took ~20 min (first-time mono / FlyDSL builds for L=8), ~3 min with 6.

```text
c4                tok/s/GPU  P90 intvty  TTFT p50/mean s  ITL mean/p90 ms  ISL     hit    errors  SIM acc
base (08:58)      2973.7     133.8       0.42 / 2.54      5.78 / 7.47      168937  0.883  0/506   3.43
draft8 (15:53)    2889.8     123.0       0.40 / 2.59      6.13 / 8.13      168248  0.894  0/495   3.68
delta             -2.8 %     -8.1 %                       +6.1 % / +8.8 %
```
draft8 loses at c4 too: per step 19.8 -> 22.6 ms (+14 %). Driver done 17:00, GPUs free.

### Summary of the gap matrix (2026-11-05)

- KEEP `--max-mamba-cache-size 150` at c4: +9.8 % tok/s/GPU, +7.0 % P90 intvty, hit 0.883 -> 0.951; gap to
  B300 c4 13.2 % -> 4.6 % (tok/s/GPU). n=1; replicate before landing in the recipe. Untried: 200+ slots.
- DROP 8 draft tokens: c1 -7.2 %, c4 -2.8 % tok/s/GPU at CI-simulated acceptance (3.68 vs 3.43).
- c1: hit already equals B300; remaining gap is decode per-step time (ITL p90 5.72 vs 4.15 ms). Not yet tried:
  B300 runs without `--prefill-decode-interval 16` / `--schedule-policy lpm`.
