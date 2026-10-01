# Porting ATOM V4.1 optimizations into SGLang (rolao opt-branch)

Owner node: mi355-4 (all PIDs / GPU ids / ports / "server up" below refer to mi355-4).

Full pre-condensation history: /shared_nfs/kk/dsv41/doc_backup_20260929/ATOM_PORT.md.

## CONTINUE HERE

**CURRENT STATE (2026-09-30 19:50, read this first):**
- Code: sglang `/sgl-workspace/sglang-rolao-opt` branch atomport-mxfp8-producers = rolao/dsv41/opt-branch **a5e40eca5e**
  (+ uncommitted EVENT_WAIT experiment in managers/overlap_utils.py, do not ship). aiter `/sgl-workspace/aiter-5750`
  branch dsv41-atomport-5750-local 56862fa70 (= PR #5750 head 1053c79bb + #5561 + fp4 prefill cache + torch.Stream +
  tuner datagen + tuned SEF CSV). flydsl /sgl-workspace/pydeps-flydsl-0341. Repro below still valid (SRC=sglang-rolao-opt).
- Best numbers: results/agentx.md "CURRENT BEST" table (5ec406) + router fusion + sort MP rows: c1 11,371.9 / 349.5,
  c2 11,819.8 / 324.6 (both >= ATOM). Per-concurrency chunk: 16384 at c<=32 (c32 mem 0.80), 4096 at c64 (mem 0.85).
- Done today: chunk A/B c32/c64; c8 prefill-overlap attribution; A router fusion (-0.41 ms/step) + D sort multi-phase
  (-0.13 ms/step) pushed as bc8bae5146 / a5e40eca5e; aiter PR ROCm/aiter#5967 (tuned rows + tuner) on
  kkHuang-amd/aiter dsv41-atomport-main; min-aiter-scope study -> AITER_MIN_SCOPE.md.
- Not re-measured on a5e40eca5e: c8..c64 with A+D, and teammate 5562928323 (wo_a M-bucketed tiles) in any AgentX run.
- **C IN PROGRESS (user picked C first, 2026-09-30 20:10):**
  (1) RUNNING c8 baseline on a5e40eca5e, chunk 16384, mem 0.70: TAG c8_a5e_c16k_pdi16 (GPUs 4,5, port 8888, PID 967791;
  first launch lost a venv race, log *.venvrace.nohup) and (2) c8_a5e_c16k_pdi32 (GPUs 6,7, port 8889, PID 964438).
  SRC = new clean worktree /sgl-workspace/sglang-c8-base (detached a5e40eca5e; /sgl-workspace/sglang-router-fuse is gone).
  Do NOT launch two agentx runs at once when /workspace/agentx-runtime/venv is missing (both try to create it).
  (3) CODE, uncommitted, worktree /sgl-workspace/sglang-pdi-time branch pdi-time-interleave: cost-scaled PDI =
  SGLANG_PREFILL_DECODE_STEPS_PER_KTOK (0 = off) + SGLANG_PREFILL_DECODE_PREFIX_KTOK_SCALE (128): defer prefill for
  max(PDI, ceil(k * sum(ext * (1 + pre/(scale*1000))) / 1000)) steps. Token-based, NOT wall-clock: a time-based
  decision differs per TP rank -> collective mismatch. Off under DP attention. UT test_scheduler_prefill_decode_interval.py 7/7.
  RESULTS 21:15: (1) PDI16 TTT 29,378.4 / P90 209.3 (p50 305.9, TTFT p50/p90 0.41/1.16, 1358 ok) = vs ATOM +0.4% /
  -12.8% -> A+D did NOT move c8 P90 (5ec406: 209.7); no-overlap subset P90 271.0 (was 262.4), bottom decile 131/136
  overlapped. (2) PDI32 29,302.1 / 216.2 (TTFT 0.47/1.28, 1351 ok) = +3.3% P90 vs PDI16 (inside noise), -9.9% vs ATOM.
  => fixed PDI is not enough; prefill interruption still the driver.
  CALIBRATION 21:45 (calib_prefill_cost/probe.log, 16384 new tokens, 3 reps, spread <2%): prefix 0/32/128/256/512k
  -> 627/653/719/816/996 ms (ratio 1.00/1.04/1.15/1.30/1.59) => PREFIX_KTOK_SCALE ~870 (default 128 was 6-7x too
  steep); a full chunk is 0.63-1.0 s (the earlier "~3.2 s at long ctx" estimate is wrong). 38.3 ms/ktok at prefix 0,
  decode step ~10-11 ms -> k = 3.5 * d/(1-d) steps/ktok for decode duty d. PDI16 alone gives d ~14-20% on a full chunk.
  **C RESULT 2026-10-01 08:45 -- cost-scaled PDI is a NULL result, do not ship as a c8 P90 fix.**
  Knob verified working (temp PDICOST log, now removed): full 16384 chunk -> 30-32 steps (k1.75) / 59-64 (k3.5) vs 16.
  c8_pdicost_k175_s870: TTT 29,384.8 / P90 213.4 / p50 302.4 / TTFT 0.40/1.17 / 1359 ok / no-overlap P90 268.8.
  c8_pdicost_k350_s870: TTT 29,349.6 / P90 215.4 / p50 301.5 / TTFT 0.42/1.28 / 1354 ok (one 13.6 s intvty outlier).
  Baseline c8 3-run mean 29,385 / 210.9 (runs 209.3/210.0/213.3) -> both k inside run-to-run noise, far from the
  228 pass bar. agentx_prefill_overlap: bottom decile is still ~95% overlapped, and the in-flight 4.5-8.5 bucket
  (~130 req) has intvty median ~208 for all three runs (no-overlap subset in that bucket ~240), so P90 tracks the
  high-concurrency periods, and deferring prefill harder does not move it. Hypothesis, not verified: the tail is
  decode step cost at batch 5-8 (step time / DSpark accept), not the number of prefill interruptions.
  Worktree sglang-pdi-time left uncommitted (code + UT), not committed. GPUs 4-7 free. Next = user decision.
  **C STEP 1 IN PROGRESS (08:55): decode-only bs sweep SGLang vs ATOM** (user: AL is simulated 3.51 on both, so the
  gap must be step cost or scheduling). Client scripts/decode_bs_sweep.py (warm prefix cache, then bs concurrent
  decode; ctx 8192,65536 x bs 1,2,4,6,8, OSL 2048, temp = server default). Servers: SGLang c8-base a5e40eca5e SERVER_ONLY
  GPUs 4,5 port 8888 PID 1054569 (tag bsweep_sgl_c8); ATOM rootfs MODE=server CONC=8 GPUs 6,7 port 8000 PID 1054570
  (log /shared_nfs/kk/atom_run/bsweep_atom_c8.log). Rows -> /shared_nfs/kk/dsv41/atomport/decode_bs_sweep.tsv.
  Decision: step gap at bs 5-8 -> GPU profile both at those bs; no gap -> look at ATOM scheduling.
  RESULT (09:10, ctx 64k and 128k agree, 2 reps each, spread <=0.3 s): per-request decode tok/s ours vs ATOM:
  bs1 +10%, bs2 ~0 (-3/+3), bs4 -10/-12%, bs6 -12%, bs8 +7/+10% (ATOM drops sharply 6->8: 294 -> 218).
  P90-interactivity proxy follows the same pattern (bs4-6 -10..-14%). => REAL step-cost gap at bs 4-6, same size as the
  AgentX c8 P90 gap (-12%), matching the hypothesis that the c8 tail is the high-in-flight periods. ctx 8192 rows of
  tag bsweep1001 are INVALID (first-swept ctx, warmup: reps differ 2-3x on both engines). Rerun 8k + 64k bs 3,5,7 as
  tag bsweep1001b was running at 09:13 (logs /tmp/bsw2_{sglang,atom}.log, aggregator /tmp/agg.py).
  bsweep1001b DONE (warm, reps agree): 8k now matches 64k/128k (bs4 -12.6%, bs6 -12.6%, bs8 +11.7%); 64k bs3 -8.3%,
  bs5 -13.7%, bs7 +13.8%. => gap at bs 3-6 (-8..-14%); ATOM has a cliff between bs6 and bs7 (300 -> 213 tok/s,
  i.e. bs*6 verify tokens 36 -> 42), ours declines smoothly. Context length does not matter (8k = 64k = 128k).
  NEXT: GPU-only profile both engines at ctx 64k bs 4 and 6 (same client, warm), per-kernel/per-step breakdown;
  first suspects are anything whose cost scales with bs*6 tokens (MoE M-buckets / tuned CSV rows at M=24..36,
  attention decode split, sampling). Servers still up: SGLang PID 1054569 (8888), ATOM PID 1054570 (8000).
  **C RUN LOG (launched 2026-10-01 07:25):** both lanes below, lane PIDs 1026775 (k175, 8888) / 1026776 (k350, 8889) (relaunched 07:33; first try crashed on self.tp_rank in temp log, dirs *.tprankcrash);
  kill: `kill -- -<pid>` then free GPUs 4-7. Overnight nohups moved to lane_888{8,9}.night0930.nohup. TEMP log
  `PDICOST ext= pre_max= steps=` (tp_rank 0, marker TMP_PDICOST_LOG) added in scheduler._prefill_cost_decode_steps --
  REMOVE before commit. Verify: `rg -c PDICOST <tag>/server.log`, full 16384 @ prefix 0 -> steps 29 (k1.75) / 58 (k3.5).
  **C NEXT -- START HERE IN THE NEW SESSION (user 2026-10-01 07:15).** Copy-paste launch (both lanes, ~70 min; per-port
  aiperf venvs already fixed in agentx_colleague_run.sh, so parallel launch is safe):
  ```bash
  cd /shared_nfs/kk/dsv41/agentx && export PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori \
    SRC=/sgl-workspace/sglang-pdi-time/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 SGLANG_PREFILL_DECODE_PREFIX_KTOK_SCALE=870 \
    EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" OPUS=0 TP=2 EP_SIZE=1 DURATION=3600
  S=/workspace/claude-skills/dsv41/scripts/agentx_lane.sh
  SGLANG_PREFILL_DECODE_STEPS_PER_KTOK=1.75 GPUS=4,5 PORT=8888 POINTS="c8_pdicost_k175_s870:8:16:16384:0.70" setsid nohup bash $S > lane_8888.nohup 2>&1 < /dev/null &
  SGLANG_PREFILL_DECODE_STEPS_PER_KTOK=3.5  GPUS=6,7 PORT=8889 POINTS="c8_pdicost_k350_s870:8:16:16384:0.70" setsid nohup bash $S > lane_8889.nohup 2>&1 < /dev/null &
  ```
  Verify the knob took effect (outcome, not just the env): add a temporary log or count decode steps between Prefill
  batch lines in server.log for big chunks (expect ~16 at k=0; ~57 / ~115 for a full 16384 chunk at prefix 0).
  Baseline to beat: c8 3-run mean 29,385 / P90 210.9 (night0930_c8, night0930_c8_r2, c8_a5e_c16k_pdi16). Compare with
  scripts/agentx_prefill_overlap.py (no-overlap P90 ~271) and TTFT p50/p90 (0.41/1.16). If a k wins: GSM8K x3, then
  c16/c32 no-regression check, then commit on pdi-time-interleave (no push without asking).
  Details of the (postponed) plan:
  c8_pdicost_k175_s870 (k 1.75, d~1/3; GPUs 4,5, port 8888) and c8_pdicost_k350_s870 (k 3.5, d~1/2; GPUs 6,7, port 8889),
  both PDI16 floor, chunk 16384, SRC /sgl-workspace/sglang-pdi-time/python, env SGLANG_PREFILL_DECODE_STEPS_PER_KTOK=<k>
  SGLANG_PREFILL_DECODE_PREFIX_KTOK_SCALE=870 on top of the repro command (CONC=8 PORT=<p> GPUS=<g>). Delete the stale
  c8_pdicost_* dirs/nohups first. Pass: P90 >= 228 (ATOM 240 -5%) with TTT >= ATOM. Scheduling-only change -> GSM8K
  after k is chosen. Also consider changing the env default scale 128 -> 870 in environ.py.
  (4) (done, see above) calibrate with scripts/prefill_cost_probe.py --port <p> on a finished server (prefix 0..512k x
  16384 new) -> scale = L_k/(ratio-1), k from target decode duty; then GSM8K x3 + c8 AgentX with the knob; ATOM
  prefill comparison skipped (user).
- **OVERNIGHT SWEEP DONE (all 8 points 0 errors, GPUs free; table = results/agentx.md CURRENT BEST 2026-10-01):**
  vs ATOM TTT / P90: c1 +5.1% / +2.8%, c2 +3.8% / -2.6% (r2 -4.2%), c8 +0.5% / -12.5% (3 runs mean -12.1%),
  c16 +0.2% / +5.1%, c32 +6.3% / +5.3%, c64 +15.2% / +73.6%. Pass criteria all met EXCEPT c8 P90 -> C is the only gap.
  c2 vs 09-30 rf_c16k_c2_pdi16 (324.6): 315.7 / 310.3 is tail noise, not a regression (p50 and no-overlap P90 flat,
  see results/agentx.md note 2); c2 3-run mean P90 316.9 = -2.2% vs ATOM.
  Launch details (kept for reference): latest code a5e40eca5e
  (= rolao/dsv41/opt-branch head after fetch) from /sgl-workspace/sglang-c8-base, best config per concurrency, two TP2
  lanes via scripts/agentx_lane.sh. Lane A GPUs 4,5 port 8888 PID 989613: c64 (PDI4, chunk 4096, mem 0.85) -> c16
  (PDI16, 16384, 0.80) -> c2 -> c2_r2 (PDI16, 16384, 0.70). Lane B GPUs 6,7 port 8889 PID 989614: c8 -> c1 -> c8_r2
  (PDI16, 16384, 0.70), then c32 (PDI4, 16384, 0.80) via requeue wrapper PID 996215 (first c32 died in 1 s: venv race).
  Tags night0930_c*; progress lane_{8888,8889}.txt in /shared_nfs/kk/dsv41/agentx. ETA lane A ~02:30, lane B ~03:45 +08.
  VENV TRAP: benchmark_lib.sh resets AIPERF_DEPS_READY=0 and rm -rf's $AIPERF_VENV on EVERY run -> parallel lanes
  delete each other's aiperf. Fixed in agentx_colleague_run.sh: AIPERF_VENV=$AIPERF_RUNTIME_DIR/venv_p$PORT (applies
  from each lane's 2nd point; the running c64/c8 share the old venv, nothing deletes it now).
  Summary per point: TTT = throughput.per_gpu.total_tput_tps, P90 = latency.intvty.p90, TTFT p50/p90, ok count;
  ATOM refs from results/agentx.md. Covers B (c2 x2, c8 x3 incl. c8_a5e_c16k_pdi16) + CURRENT BEST refresh.
- REMAINING WORK (user to prioritize in the new session):
  B. c2 / c8 x2 repeats on the final build (P90 noise ~5%/run).
  C. c8 P90 (-12.6% vs ATOM on 5ec406): prefill interrupts decode (PDI 16 = 16 decode steps per 16384 chunk);
     options PDI 32 at c8 or a time-based prefill/decode interleave. PAUSED by user.
  E. ATOM-style real-sampling verify (greedy draft + sampled target prefix match) for real traffic.
  F. Long-context step-time slope (9.4 ms @64k -> 9.7 ms @>512k).
  G. Move servers to the aiter main-based branch (#5967, without #5561) after GSM8K x3 + one AgentX point.
  H. Housekeeping: claude-skills local commit 33f5621 + later doc edits not pushed; upstream the aiter opus
     moe_sorting auto-policy fix; SGLang compact_attention_hip.py imports aiter _qkpv_fp8 (deleted upstream, #4919),
     only reached for 528/288 KV dims.
  Also: re-run c8/c16/c32/c64 on a5e40eca5e to refresh the CURRENT BEST table with A+D.

**History (condensed log below):**

**c1/c2 GAP ROOT CAUSE FOUND (2026-09-29 11:40):** AgentX (aiperf agentx-mvp) sends no temperature -> SGLang
default 1.0 -> DSpark takes the SAMPLING path even under SGLANG_SIMULATE_ACC_LEN (draft temperature sampling,
SoftmaxTemp over draft rows, AcceptSampling; fold disabled) and then overwrites correct_len with the simulated
value, so the sampling work is thrown away. ATOM with --spec-decode-acceptance-length: draft always greedy,
verify = target argmax + rejection_synthetic_sample_kernel (atom/model_ops/rejection_sampler.py:214), no softmax.
Proof, same c1 server (gap_temp_probe_c1, PID 597393, GPUs 4,5), proxy ctx 2k x3 x2: temp 0 -> 391-409 tok/s,
proxy P90 334-347; temp 1.0 -> 340-347 tok/s, P90 291-296 (-13%; matches AgentX c1 P90 292.8). AgentX c1 server
log: step ~10.45 ms flat over ctx 0-512k (10.42 @<64k, 10.59 @>512k) -> context length is NOT the cause; TTFT =
12.8% of request time. Per-request intvty barely depends on ISL; only OSL<64 requests are slow (n=11).
ATOM's real (non-synthetic) stochastic verify is also cheaper: sample target token per position + exact prefix
match vs a greedy draft (lossless), no draft probs.
**Next:** (a) measurement parity: under SIMULATE_ACC take the greedy draft + greedy/fold accept regardless of
temperature (like ATOM synthetic); expect c1 P90 ~335. (b) real-traffic win: port ATOM-style stochastic verify
(greedy draft + sampled-target prefix match), in-graph. (c) prefill share: ATOM chunk 16384 vs our 4096
(InferenceX PR #3451 applies ATOM's fixed 16384 to the vLLM recipe). ATOM-on-node run (step 2) still to plan;
no docker in this container -> in-place install.
**DONE 2026-09-29 12:50 -- AgentX c1 temp 0 (aiperf --extra-inputs temperature:0, best config, PDI16):**
TTT 11,170.2 / P90 330.7 (p50 371.7), 282 ok -- vs default-temp run 10,306.1 / 292.8: +8.4% / +12.9%; vs ATOM public
10,820.2 / 337: +3.2% / -1.9% (inside the 5% criterion). Server step now 9.15 ms @64-256k -> 9.48 ms @>512k (+0.33 ms
= the remaining context slope); TTFT share 14.4%. Dir gap_temp0_agentx_c1_pdi16. => c1 gap = sampling path.
Next: implement (a) in code (SIMULATE_ACC -> greedy draft + fold accept, no client change), run the workflow, then
c2; then (b) ATOM-style stochastic verify for real traffic; long-context slope and chunk 16384 are secondary.

**(a) IMPLEMENTED, uncommitted (2026-09-29 13:10):** SGLANG_SIMULATE_ACC_GREEDY (EnvBool True, environ.py) ->
dspark_worker_v2 sets sampling_info=None per step when SIMULATE_ACC_LEN>0 and no grammar (greedy draft +
AcceptGreedy; our draft fold is "greedy only", so temp 1.0 used to add an eager Markov sampling pass + SoftmaxTemp
+ AcceptSampling). Inert without SIMULATE (GSM8K/EVAL_ONLY path unchanged). Proxy c1 ctx2k, same server:
temp 1.0 now 392-399 tok/s / P90 335-344 = temp 0 (384-399 / 334-338); before 340-347 / 291-296. Output-id
parity is not testable under SIMULATE (temp0 vs temp0 already diverges at token 3: unverified drafts committed).
**ATOM evidence (source @4685e3cf, not runtime-confirmed):** chat default temperature 1.0 (protocol.py:23
DEFAULT_TEMPERATURE, no generation_config in the V4.1 dir). Draft ALWAYS greedy (deepseek_v4_dspark.py:1398
_DSparkInner.forward_head markov argmax; V4.1 reuses it, deepseek_v41/dspark.py:549). Verify with temp>0 STILL
samples: model_runner.py:3250 sample_verification_tokens (every target row) + bonus sampler with temperature,
then rejection_sampler.py:214 synthetic branch uses target argmax and ignores target_token_ids. So ATOM is not
fully greedy: it skips draft sampling / draft softmax / ratio rejection, but pays target-row sampling. Our
SIMULATE_ACC_GREEDY skips that too -> slightly more favourable than ATOM (bs1: ~6 rows x 129k vocab sampling).
Strict parity option: keep a temperature-sampled bonus/target draw under SIMULATE (measure its cost).
**ATOM PATH CONFIRMED AT RUNTIME (2026-09-29 13:50):** image pulled w/o docker (scripts/atom_image_fetch.py, 35
layers 16.9 GB -> /shared_nfs/kk/atom_image/rootfs 43 GB, 328 s), run by chroot (scripts/atom_chroot_run.sh; ATOM
4685e3cf7, aiter e2d019f15, torch 2.10+rocm7.2.4) -- recipe TP2 server boots in ~4 min on GPUs 4,5, log shows
"Forced speculative acceptance ON: mean acceptance length 3.5100". Probe (scripts/atom_pathprobe/, counters in
/shared_nfs/kk/atom_run/probe_c1.jsonl), c1 ctx2k OSL1024 x3 (scripts/atom_proxy_bench.py): default temperature ->
every verify step = rejection_synthetic + sample_verification_tokens, Sampler.forward all_greedy=False; temp 0 ->
synthetic, no target sampling, all_greedy=True. Draft sample_next only called at graph capture (135), replayed
graph shared by temp 0 and 1.0; no temperature reference in the draft graph/proposer -> draft greedy. ATOM decode
tok/s: default temp 357-368, temp 0 363-370 (~1%: target sampling is cheap). ATOM proxy ~364 vs ours ~395 (proxy).
=> STRICT VERSION implemented: draft + accept greedy, bonus temperature-sampled from ALL verify rows (same work as
ATOM sample_verification_tokens + bonus), fallback to the full sampling path for grammar / penalties / top-p /
top-k / min-p (dspark_verify.sample_simulated_bonus; worker passes simulate_bonus_sampling_info). Unit test
test/registered/spec/dspark/test_dspark_simulated_bonus.py 2/2 on GPU 6. Pre-existing quirk fixed on this path:
simulate bonus used the REAL correct_len row, now the simulated one.
Proxy (strict, same server): temp 0 392-401 tok/s / P90 333-337; temp 1.0 379-390 / 324-333 (-2.6%; ATOM -1%).
**simbonus c1 DONE (15:02):** TTT 10,969.6 / P90 317.5 (p50 361.7), 279 ok, 0 tracebacks; vs BEST 10,306.1 / 292.8
+6.4% / +8.4%; vs ATOM public 10,820.2 / 337 +1.4% / -5.8% (just outside 5%). Server step 9.39-9.76 ms vs 9.15-9.48
in the temp-0 run -> our bonus sampling costs ~0.3 ms/step (~3%) vs ATOM ~1%: sample_simulated_bonus = float
softmax + exponential_ + div + argmax over 6x129k; candidate: fused Gumbel-max on logits (no softmax) in one kernel.
**simbonus c2 DONE (16:07):** TTT 11,253.3 / P90 292.6 (p50 358.7), 434 ok; vs BEST 10,476.9 / 279.6 +7.4% / +4.6%;
vs ATOM 11,266.6 / 324 -0.1% / -9.7%. c2 is mostly bs=1 (server decode lines bs1 3473 vs bs2 437; bs1 median 366.8,
bs2 619.9 tok/s, both up vs BEST 336.0 / 576.0) -> c2 P90 loss vs c1 (317.5) is likely the other lane's new-turn
prefill interrupting decode (chunk 4096 vs ATOM 16384) -- unverified, next to attribute per request.
**(1) DONE 16:40:** fused bonus sampling = kernels/ops/speculative/dspark/simulated_bonus.py (2-stage Gumbel-max:
[rows x 32 vocab splits] partial max, then per-request select at correct_len; seed do_not_specialize -- a
specialized seed recompiled every call, 1.09 ms). UT 3/3 (+bf16 full-vocab case). Microbench vs torch version:
bs1-8 109-144 -> 29-30 us, bs64 515 -> 127 us. Proxy c1 same server: temp 1.0 384-401 vs temp 0 390-396 (~0-1%,
ATOM ~1%).
**(2) DONE:** scripts/agentx_prefill_overlap.py on simbonus_c2: 61/433 requests overlap the other lane's prefill;
overlap>10% (n=43) intvty median 280 vs 362 without overlap; bottom decile 29/44 overlapped; corr -0.65;
no-overlap subset P90 307. ATOM also pauses decode during prefill (scheduler.py:1985 mixed batch TODO; recipe
PDI 0), but uses chunk 16384 vs our 4096 -> next experiment: --chunked-prefill-size 16384 at c2 (and c1).
**(3) DONE:** GSM8K (scripts/run_gsm8k.sh = few_shot_gsm8k, the established metric) 0.899. NOTE: EVAL_ONLY without
SERVER_ONLY runs InferenceX run_eval (lm-eval chat multiturn) -> 0.970, NOT comparable; artifacts moved to agentx/misc/.
**PUSHED 5ec406bb76** on rolao/dsv41/opt-branch, rebased on teammate e5d7c72c33 (1am9trash: HIP+simulate -> greedy
accept only; draft still samples, bonus argmax). Ours adds greedy draft + temperature-sampled bonus; compatible.
SGLANG_SIMULATE_ACC_GREEDY=0 == e5d7c72 behaviour alone (use for A/B of the draft-side cost).
**Step span (c1 ctx2k, TP0, 5ec406):** temp 0 1136 kernels/step, span 8.410 ms, host gap 1.382; temp 1.0 1138, 8.343,
1.396 -> bonus sampling ~free on GPU. Profiling a never-seen sampling path hung once (JIT under profiler?); warm
up with an unprofiled temp-1.0 request first.
**f5ec406 c1 DONE (18:48):** TTT 11,141.9 / P90 321.5 (p50 365.9), 281 ok -> vs ATOM +3.0% / -4.6% (c1 now inside the
5% criterion); vs unfused strict +1.6% / +1.3%. Server step 9.34-9.70 ms (temp-0 client run 9.15-9.48; ~0.15 ms left,
could be noise). TTFT share 12.9%.
**f5ec406 c2 DONE (19:53):** TTT 11,123.3 / P90 288.5 (p50 361.6), 427 ok; vs ATOM -1.3% / -11.0%; vs unfused strict
-1.2% / -1.4% (noise). Overlap attribution unchanged: 60/426 overlap, overlap>10% (n=43) median 269 vs 365 no-overlap,
bottom decile 28/43 overlapped, no-overlap subset P90 310. chunk16k run started 19:53 (server_args chunked_prefill_size=16384 confirmed).
**f5ec406 chunk16k c2 DONE (20:59):** TTT 11,550.4 / P90 309.3 (p50 366.5), 452 ok -> vs chunk 4096 +3.8% / +7.2%;
TTFT p50/p90 0.52/1.19 -> 0.42/0.96 s; vs ATOM 11,266.6 / 324 +2.5% / -4.5% (c2 now inside 5%). Overlap: bottom decile
22/46 overlapped (was 28/43), median overlap 0. KV pool 29.20M -> 28.85M tokens (-1.2%). Queue done, GPUs free.
**Status:** c1 (+3.0% / -4.6%) and c2 chunk16k (+2.5% / -4.5%) both meet the 5% P90 / TTT>=ATOM criteria.
**SWEEP DONE (sw5ec406_c16k, 21:10-02:35 +08; rows "SWEEP" in results/agentx.md), TTT / P90 vs ATOM:**
c1 11,087.2 / 333.4 (+2.5% / -1.1%), c2 11,494.7 / 293.2 (+2.0% / -9.5%), c8 29,157.1 / 209.7 (-0.4% / -12.6%),
c16 53,469.1 / 135.7 (-1.0% / +6.1%). c32 PDI4 mem0.85: HIP OOM after 20 min (FAIL). c64 PDI4 mem0.85: NCCL
watchdog BROADCAST(numel 16) timeout at 18:23:51 UTC under a 42-request / 6M-token prefill queue (FAIL, cause not
yet known; OOM-adjacent?). c2 same config as the 309.3 run gave 293.2 -> single-run P90 spread ~5%: c2/c8 need
repeats before concluding. chunk 16384 is not safe at c>=32 with mem 0.85 (predicted ~10 GB headroom).
**Next (proposal):** (1) c32/c64: chunk 4096 (previous best 93,855.8/68.7, 117,189.7/39.7) or chunk 16384 with mem
0.80; (2) c2/c8 x2 repeats to size the noise; (3) c8 P90 -12.6% = biggest gap -> prefill-overlap attribution.
**c64 NCCL timeout = memory (2026-09-30 07:20, from sw5ec406_c16k_c64_pdi4/server.log):** both ranks' main thread
stuck at the SAME line, fp4_indexer_hip.py:276 prepare_fp4_prefill_workspace `torch.empty(guarded_page_table)`
(called per chunked-prefill step from deepseek_v4_backend_hip_radix._refresh_fp4_prefill_workspace, breakable prefill
graph replay), BROADCAST seq 37727 stuck from 18:23:51 +08. KV pool NOT full (full token usage 0.12-0.14, queue 42,
pending 6M tokens, 16384-token chunks back-to-back). Free device mem after graph capture 10.97 GB; during serving
"free device mem 0.22 GiB" warnings. => allocator at the OOM edge (release-cached-blocks / hipFree syncs while an RCCL
collective is in flight) -- same root as c32 OOM (270.49 GiB allocated + 11.42 GiB reserved-unused,
0 free, 512 MiB alloc in a breakable-graph eager piece). Headroom outside mem-fraction is the constraint, not KV.
Proposed (1): c32/c64 PDI4 chunk 16384 mem 0.80 vs chunk 4096 mem 0.85 re-baselined on 5ec406 (old 4096 numbers are 026da361).
**aiter on main (2026-09-30 08:45):** #5750 is squash-merged upstream (e2d019f15). Our local aiter changes (= patches/
aiter_local_5750_worktree_0001.patch + tuned FMoE CSV) are now ONE commit 7f44ace62 on top of ROCm/aiter main c8325e00c,
pushed to FEATURE BRANCH kkHuang-amd/aiter dsv41-atomport-main (fork main stays = upstream; worktree
/sgl-workspace/aiter-mainport, local branch dsv41-atomport-main). 08:55 user trim -> branch head 7fddd8880 = main +
MoE-tuner GPU datagen + the 23 tuned rows MERGED into dsv41_fp4_tuned_fmoe.csv / shapes into dsv41_fp4_untuned_fmoe.csv
(no separate CSV). DROPPED on the branch (still in the live aiter-5750 env): torch_utils torch.Stream, fp4 prefill
lru_cache->cache, #5561 flydsl stage1 full vmcnt drain (our afp8_wfp4 stage1 path; heterogeneous_b drain in main
only covers FHMoE shared_expert_id). Moving the servers to this branch needs GSM8K x3 + one AgentX point first.
Live servers still use /sgl-workspace/aiter-5750 (PR head 1053c79bb + same diff); switching them to aiter-mainport is
untested (needs JIT rebuild + GSM8K + one AgentX point). Fork branch dsv41-atomport-5750-local is superseded.
**sglang #40204 (small-M MXFP4 MoE, Qwen) review 09:20:** not applicable as-is (guard hidden 4096 / per-rank inter 256 /
10-11 slots / bf16 act; ours hidden 5120, inter 1152/rank, 7 slots, fp8 act on FlyDSL a8w4; not in our tree). Its premise
is weaker for us: its aiter baseline ~0.3 TB/s (1.7 MB experts), our tuned a8w4 at random routing ~3.0 TB/s at M=1,
~5.1-5.7 TB/s at M=4-32 (9.4 MB experts; 37% -> ~70% of 8 TB/s). The PR itself says inter 512 already crosses to FlyDSL at
~20 tok. Transferable idea = the MoE OVERHEAD at small M: c1 step has moe1+moe2 1.42 ms but sorting/gating/router/
append_shared/quant-sort 1.32 ms (0.49+0.27+0.22+0.17+0.17). Candidates: sort-free small-M dispatch (skip moe_sorting +
mx_quant_moe_sort, ~0.5-0.66 ms/step), one-launch router+topk+shared-append (cf. #41133, ~0.4 ms). Est. 5-10% c1 step.
**ROUTER FUSION (2026-09-30 09:50, uncommitted, worktree /sgl-workspace/sglang-router-fuse branch
router-gate-shared-append @5ec406):** with --enforce-shared-experts-fusion each decode layer ran _router_gemv_split_k 4.4 +
_reduce_partials 4.3 + aiter topk_gating_opt 7.2 + _fused_append_shared_experts 3.9 us (prof_5ec406); without SEF the
fused gate was gemv 4.6 + _router_gate 4.8. Cause: select_experts dropped the partials whenever num_fused_shared_experts>0
(non per-rank). Fix: rocm_router_gate(num_shared=S) also writes the shared columns (id 384+s, weight 1.0; the existing
"router emitted shared" branch then applies fused_shared_experts_scaling_factor exactly like the append);
_post_process_topk_ids skips the append when topk_ids already has top_k columns (generalizes the JIT-grouped-topk check),
capture/EPLB recorder get the routed columns only; partials kept for aiter SEF when EPLB remap is off.
Parity: bitwise vs aiter gate + fused_append (M 1/6/64) and select_experts partials vs non-partials (scaling None/0.5),
test_aiter_moe_hip.py 5/5 on GPU 6; new path calls 0 reduce / 0 append. Microbench (hot cache, graph): 11.2 -> 6.35 us/layer
at M=6 (-4.9); in-model expected ~-10 us/layer (trace numbers) -> 0.2-0.4 ms/step. pre-commit fully clean: also added
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd") to test_dspark_simulated_bonus.py (missing since 5ec406;
3/3 on GPU 6) + ruff format of that file, in the same worktree.
**D = MoE sort dispatch policy (13:30, same worktree, uncommitted):** aiter opus moe_sorting auto policy picks ONESHOT
below ~24 tokens: 11.6 / 12.5 / 15.7 us at M 1/6/16 (E385 topk7 BM32) vs MULTI-PHASE (policy 2) 6.7 / 7.0 / 6.9 us; at
M>=64 both equal (7.0 ... 64.2 us at 16384); E129 topk4 (draft) 6.4-7.9 -> 5.8-6.0. Outputs bitwise identical for every
M 1..16384 (sorted ids/weights/expert tiles/num_valid, moe_buf zeroed). Fix: SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY
(EnvInt, default 2) passed as fused_moe(moe_sorting_dispatch_policy=...) in moe_runner/aiter.py. Expect ~-5.5 us/layer
at c1 (~0.22 ms/step). Unit test TestAiterMoeSortingDispatchPolicy (policy 0 vs 2 bitwise), file 6/6 on GPU 6.
Upstream candidate: fix the auto heuristic in aiter moe_sorting_opus.h. A custom single-launch sort+quant kernel was
considered and dropped (bit-exact MX quant re-implementation risk; the policy switch gets most of the sort win).
**RUNNING (13:07 +08, PID 816672, scripts/router_fuse_ab.sh, summary /shared_nfs/kk/dsv41/atomport/router_fuse/summary.txt):**
SIM c1 proxy x3 + GPU-only profile for base (5ec406) / fuse (policy 0) / fuse+sortMP (policy 2), step spans of all
three; then GSM8K 1319 x3 (fuse+sortMP, EVAL_ONLY CONC=32); then AgentX c1, c2 (PREFIX rf_c16k, PDI16, chunk 16384).
ETA ~16:00 +08. **Step-span results (13:30):** kernels/step base 1138 / fuse 1057 / fuse+sortMP 1098 (MP = 2 kernels);
span median 8.502 / 7.865 / 8.020 ms -- the TOTAL span is NOT usable here: MoE GEMM time varies per server
(moe1 955 / 841 / 1002, moe2 526 / 426 / 552 us/step; temp-1.0 proxy tokens -> different routed-expert counts).
Per-kernel deltas (scripts: /tmp/kdiff.py logic): A = -259 topk_gating -168 reduce_partials -166 append +186 _router_gate
= **-0.41 ms/step**; D = opus sort 484 -> 358 us/step = **-0.13 ms/step** (in-model 12.3 -> 2 x ~4.5 us per layer).
GSM8K fuse+sortMP 0.897 / 0.901 / 0.900 (baseline ~0.899). **AgentX c1 A+D: 11,371.9 / P90 349.5 (vs 5ec406 +2.6% / +4.8%; vs ATOM +5.1% / +3.7%).** **AgentX c2 A+D: 11,819.8 / P90 324.6 (vs 5ec406 +2.8% / +10.7%, vs its 309.3 twin +4.9%; vs ATOM +4.9% / +0.2%).** Run DONE 15:36, GPUs free. C (c8 prefill vs decode) PAUSED by user 15:45; nothing running.
**PUSHED 15:50:** rolao/dsv41/opt-branch 5562928323..a5e40eca5e = bc8bae5146 (A router fusion + D sort MP + tests) + a5e40eca5e (register_amd_ci for test_dspark_simulated_bonus.py), rebased onto teammate 5562928323 (wo_a M-bucketed tiles, not in our AgentX runs; 9/9 UT + pre-commit on the rebased tree). Worktree /sgl-workspace/sglang-router-fuse is now at the pushed head.
First attempt aborted 13:05 (the fuse server may have imported the D edit mid-start). DO NOT edit
/sgl-workspace/sglang-router-fuse/python until it finishes.
**(1) c32 A DONE: 97,158.2 / P90 69.5, 0 errors (vs ATOM +6.8% / +4.7%, vs old 4096 +3.5% / +1.2%); c64 A DONE: 121,940.9 / P90 31.5 (vs ATOM +19.1% / +36.4%; vs old 4096 +4.1% / -20.7%: P90 trade-off); free mem touched 0.00 GiB once. B c32 DONE 92,657.2 / 66.0 -> A wins at c32 (+4.9% / +5.3%). B c64 DONE 116,273.6 / 39.3 (A +4.9% TTT but -19.8% P90). DECISION: per-concurrency chunk = 16384 at c<=32 (c32 mem 0.80), 4096 at c64 (mem 0.85). Series done 12:36, GPUs free.**
**(1) RUNNING (started 07:25 +08, wrapper PID 730744, /shared_nfs/kk/dsv41/agentx/run_c32c64_chunk_0930.sh):**
A = sw5ec406_c16k_m080_c{32,64}_pdi4 (chunk 16384, mem 0.80; confirmed in server_args; post-capture free 25.96 GB
vs 10.97 at 0.85), then B = sw5ec406_c4k_c{32,64}_pdi4 (chunk 4096, mem 0.85 re-baseline on 5ec406). VRAM every 30 s ->
vram_c32c64_0930.log. ETA ~11:55 +08. Decision: A both OK and TTT >= B-1% and P90 not worse -> 16384 at c32/c64,
else per-concurrency chunk (16384 at c<=16, 4096 at c>=32). (2) c2/c8 repeats: ON HOLD by user until (1) is read.
**(3) DONE c8 prefill-overlap (sw5ec406_c16k_c8_pdi16; script extended: in-flight load, agent_depth split, wall-time
share by in-flight count; also fixed a bin bug that dropped overlap_frac>1.01):** in-flight during decode p50 2.55,
p90 4.36, max 7.56; wall share by in-flight count 0:0.17 1:0.33 2:0.26 3:0.14 4:0.07 5:0.02, >8: 0.00 -> subagents
did NOT push beyond 8 in this run; c8 is mostly 1-3 in flight. Prefill is the driver: 592/1346 requests have >10% of
their decode window overlapping another prefill (med 255.5 vs 311.3 no-overlap); bottom decile 130/135 overlapped
(median overlap 0.586); corr(intvty, inflight) 0.005 -> load itself is not the cause. No-overlap subset P90 262.4 vs
all 209.7 (ATOM 240). Prefill-busy wall share 16.8% (c2: 7.3%). depth>0 (subagents) p10 199.7 vs depth 0 237.9.
Mechanism: PDI 16 = 16 decode steps (~0.16 s) per prefill chunk; a full 16384 chunk takes ~0.81 s median (p10 tput
5.1k tok/s -> ~3.2 s at long ctx) -> other lanes get ~5-16% decode duty during a big prefill. Only 249 full chunks
(most prefills are radix hits, new-tok p50 1971; 7.9M new tokens total) -> the tail comes from a few big uncached prefills.
Candidates (not run): time-based interleave (guarantee decode N ms per prefill chunk instead of N steps), PDI 32 at c8
(earlier +6.5% P90, TTFT +20%), mixed chunk; ATOM c8 run on chroot to measure ATOM's decode duty during prefill (user: not now).
Note 16:44-16:57: agent shell tool hung (no new process could start); machine fine (252/3023 GB used, no
OOM/hung-task in dmesg); resolved by itself. chroot bind mounts under atom_image/rootfs unmounted 09-30 07:02 (atom_chroot_run.sh remounts on use).

**Status (2026-09-29):** best config pushed = rolao/dsv41/opt-branch **026da361c0** (T2 + requant fix + draft
metadata in graph) + `--enforce-shared-experts-fusion` + local tuned FMoE CSV. Best sweep done (below).
Remaining gap to ATOM = low-concurrency P90 (c1/c2/c8 -13/-14/-17%).
**RUNNING:** PDI sweep PREFIX atomport_best_pdi, reduced by user to c8 PDI32/4 + c2 PDI32 (c2 PDI4, c1 PDI4
cancelled). c2 PDI32 = PID 575737 (orphaned point subshell; series loop killed, so no END line in series.txt),
ETA ~11:10 +08; result in /shared_nfs/kk/dsv41/agentx/atomport_best_pdi_c2_pdi32/.
Done so far: c8 PDI32 28,643.3 / P90 212.3 (vs PDI16 +0.3% / +6.5%; TTFT p50 0.45 -> 0.54 s, p90 1.42 -> 1.78 s;
vs ATOM -2.1% / -11.5%); c8 PDI4 28,586.2 / 200.0 (+0.1% / +0.4%; TTFT p50 0.39 s, p90 1.17 s); c2 PDI32 10,421.0 / 282.0
(-0.5% / +0.9% = noise). PDI sweep DONE: larger PDI helps P90 only at c8; c2 unaffected.
**Env reproducibility (2026-09-29):** scripts/setup_atomport_env.sh rebuilds this env in a fresh container
(RUNBOOK "ATOM-port worktree"); a copy in /sgl-workspace/verify_ap is file-identical. Server from that copy (EVAL_ONLY CONC=32, GPUs 4,5): ready in
380 s incl. first JIT build, aiter imported from the copy, 0 untuned 385/129 warnings, GSM8K 0.895 (1 run; live env
0.901-0.908) -> reproducible (scripts/verify_atomport_env_server.sh, /shared_nfs/kk/dsv41/atomport/verify_env/summary.txt).
**Next (new session, user 2026-09-29):** attribute the c1/c2 gap to ATOM (P90 292.8 vs 337, 279.6 vs 324; TTT
-4.8% / -7.0%). Known: PDI is irrelevant at c1/c2; proxy c1 (ctx 2k) P90 ~336 but AgentX c1 P90 292.8 -> the loss is
AgentX-specific (long/growing contexts, radix hits, prefill of new turns, real traffic mix), not the fixed per-step
cost; proxy host gap only ~0.85 ms/step. Plan: (1) from our AgentX c1 server log/metrics, decode step time vs context
length and TTFT/prefill share; proxy at AgentX-like contexts; (2) run ATOM c1/c2 AgentX on this node (image
rocm/atom-dev nightly, see NOTES 2026-09-27 and the sglang-prefill-coalescer skill for the in-place gotchas) with
the same trace + GPU-only profile, compare per step; (3) port the largest measured difference.
**Files:** tree `/sgl-workspace/sglang-rolao-opt` (branch atomport-mxfp8-producers, tracks rolao/dsv41/opt-branch;
worktree also holds the uncommitted SGLANG_HIP_SPEC_EVENT_WAIT experiment in managers/overlap_utils.py -- do not ship),
aiter `/sgl-workspace/aiter-5750` (+ untracked `aiter/configs/model_configs/dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv`,
+ local tuner patch), flydsl `/sgl-workspace/pydeps-flydsl-0341`, ATOM reference `/workspace/atom-survey` @ 4685e3cf.
**Repro (best config, c1 server; drop SERVER_ONLY to run AgentX; EVAL_ONLY=true for GSM8K):**
```bash
cd /shared_nfs/kk/dsv41/agentx && PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori \
SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 \
EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" \
SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CONC=1 PREFILL_DECODE_INTERVAL=16 TAG=<tag> \
setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > <tag>.nohup 2>&1 < /dev/null &
```
Series: `SCRIPT=agentx_colleague_run.sh RUNS="conc:pdi ..." PREFIX=<p> bash scripts/agentx_series.sh` (same env).
**Pass criteria (project):** AgentX TP2 c1-c8 P90 within 5% of ATOM (337/324/-/240), TTT not below ATOM at c1-c16,
no regression at c32/c64 (89,715 / 111,506), GSM8K >= 0.89.
**Workflow per change:** unit parity -> GSM8K 1319 x3 (EVAL_ONLY, real acceptance) -> step span -> AgentX.
GPU 4,5 = servers, GPU 6 = unit tests. Never force-push; fetch before push (gh account kkHuang-amd has write access).

## Results

Commits on rolao/dsv41/opt-branch (all pushed, fast-forward, author wunhuang):
- **8c67e0bd51** T2: MXFP8 operands emitted by their producers (knobs SGLANG_HIP_SHARED_ACT_MXFP8 / _WO_A_MXFP8 /
  _FFN_NORM_MXFP8, EnvBool(_default_hip)). 1376 -> 1256 kernels/step, GPU span 9.458 -> 8.879 ms (-6.1%), bit-exact
  unit parity, GSM8K 0.904.
- **ba5eb0f439** shared-expert MXFP4 requant: aiter dynamic_mxfp4_quant scale + round-to-nearest-even
  (fp8_utils.quantize_block_fp8_weight_to_mxfp4; test_ties_round_to_even). Makes `--enforce-shared-experts-fusion`
  accurate: GSM8K 0.891 -> 0.906.
- **026da361c0** DSpark draft block metadata built inside the draft CUDA graph (knob
  SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA; DSV4RawDSparkDraftMetadata in deepseek_v4_backend_hip_radix.py, same pattern
  as DSV4RawVerifyMetadata). Host critical path -~0.2 ms/step.

AgentX TP2 c1 PDI16 progression (TTT / P90, 3600 s, single runs): rolao all-opts 9,818.6 / 267.5 -> T2 9,977.9 /
277.5 -> + fusion (fixed requant) 10,036.3 / 282.5 -> + tuned FMoE 10,269.1 / 287.5 -> + draft metadata in graph
10,306.1 / 292.8 (ATOM 10,820.2 / 337).
**Best-config sweep 2026-09-29 (rows "BEST sweep" in results/agentx.md, all 0 faults):** c1 10,306.1/292.8,
c2 10,476.9/279.6, c4 15,680.7/267.0, c8 28,552.6/199.3, c16 51,219.5/137.8, c32 PDI16 90,876.4/102.0,
c32 PDI4 93,855.8/68.7, c64 PDI16 90,671.3/78.8, c64 PDI4 117,189.7/39.7. vs rolao all-opts: TTT +3.8..+7.6%,
P90 +3.1..+10.3% at every point. vs ATOM: c1/c2/c8 TTT -4.8/-7.0/-2.4%, P90 -13.1/-13.7/-17.0%; c16 -5.1%/+7.7%;
c32 PDI4 +3.1%/+3.5%; c64 PDI4 +14.5%/+71.9%.
GSM8K 1319 5-shot with the best config (CONC=32 eval server): 0.901-0.908 over 6 runs.

## Closed work (summaries)

- **Gap attribution (P0, 2026-09-27):** c1 decode is context-insensitive (ctx 2k vs 64k equal) -> fixed per-step
  cost dominates; indexer only 0.09 ms/step. Small kernels cost 4.3-7.2 us each regardless of work -> lever =
  launch count. mHC seam is NOT a gap vs ATOM (ours 2 kernels/seam, ATOM >= 3). Index-build fusion not worth it.
- **T2 (MXFP8 producer folds):** (c) shared SwiGLU -> Triton silu_and_mul_clamp fp8 grid (1376 -> 1335);
  (a) wo_a split-K reduce EMIT_FP8 epilogue (-> 1296, span 9.099); (b) FFN rmsnorm_sinkhorn emit_fp8 for the shared
  gate_up (-> 1256, 8.879). Parity scripts: test_shared_act_mxfp8.py, test_wo_a_mxfp8.py, test_ffn_norm_mxfp8.py.
- **T3 (shared + routed add):** no cheap fold (aiter fused_moe output= overwrites, ar_ll has no input add);
  superseded by shared-expert fusion, which removes the add.
- **Shared-expert fusion:** upstream #32340 path (FP8 shared expert requantized to MXFP4 at load, appended as
  routed slot -> top-7; loader FusedMoE._maybe_load_fp8_shared_expert_as_fp4, layer.py:891). Saves 0.25 ms/step
  (shared gate_up/clamp/down/add gone; moe1/moe2 slightly longer). Accuracy drop 1.3 pt was a REAL BUG: the old
  MXFP4QuantizeUtil.cast_fp4 rounded e2m1 ties toward zero; FP8 sources hit midpoints often -> every matrix gain
  0.949, shared MLP output gain 0.850 (scripts/shared_expert_requant_check.py, all 43 shared experts). Fix -> MLP
  gain 0.937 / rel err 0.190 (residual = inherent MXFP4). Same bug hits DSV4-Pro fusion (ue8m0 [128,128] scales),
  accuracy only: Pro MLP gain 0.85 -> 0.94 (upstream reported no GSM8K drop on Pro). FHMoE (keep shared FP8, #35074)
  no longer needed for accuracy; possible follow-up: per-layer gain fold into the shared topk weight.
- **FMoE tuning of the fused shapes (2026-09-28):** 385/7 tokens 1-4096 and DSpark draft 129/4 tokens 1-512
  (5120/1152, a8w4 per_1x32) had no tuned config. Tuned 23/23 shapes on GPU 6 (all ksplit 0, err 0.0%), CSV
  /shared_nfs/kk/dsv41/moe_tune/ -> deployed into aiter-5750 model_configs/ (backup
  patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv). Tuner needed a local fix: generate_v2_stage1_data
  quantized E x 2I x H weights on CPU per token (42 min without progress) -> `with torch.device(device):`
  (patches/aiter_local_moe_tune_gpu_datagen.patch). Span 1176 -> 1135 kernels, 8.701 -> 8.389 ms; GSM8K unchanged.
  Caveat: unfused 384/6 + 128/3 are still untuned, so fusion-vs-unfused A/Bs slightly favour fusion.
- **Host gap (2026-09-28):** SGLang HIP host-syncs the spec publish event (overlap_utils.resolve_seq_lens_cpu,
  `publish_ready.synchronize()` workaround for #26672). Findings: (1) ATOM also syncs on step N's accept counts before
  preparing N+1; its DSpark draft runs after verify so the sync overlaps the draft. Our dspark_worker_v2 already
  publishes right after accept (dspark_worker_v2.py:912), before the ~0.3 ms commit_hidden tail -> reordering gives
  ~nothing; the DSpark draft here has no MoE kernels and is cheap. (2) Real c1 idle is ~0.85 ms/step (9.2 ms step vs
  8.39 ms GPU); the 1.6-1.9 ms trace gap was profiler-inflated. Event.synchronize wake-up is ~13 us. (3) py-spy per
  step: sync wait 5.51 ms, critical host path after the sync ~0.55 ms (0.43 = draft metadata) -> fixed by
  026da361c0 (proxy c1 384.5 -> 393.6 tok/s, P90 ~328 -> ~336; AgentX c1 +0.4% / +1.8%).
  SGLANG_HIP_SPEC_EVENT_WAIT=1 (device-side wait; host really runs ahead because our backend's
  needs_cpu_seq_lens = SGLANG_OPT_USE_ONLINE_COMPRESS = False) is ~10% SLOWER on the GPU; cause unknown
  (plain graph-after-event-wait microbench is not slower). Hypothesis: TP2 rank skew makes ar_ll spin.
- **PDI survey (2026-09-29):** PDI = scheduler iterations to skip prefill after each prefill batch/chunk
  (scheduler.py _should_defer_prefill; counts iterations even with no decode running -> c1 ~unaffected).
  Smaller PDI: prefill sooner (TTT/TTFT up), decode interrupted more (P90 down). Our c32/c64 16 -> 4: TTT
  +3%/+29%, P90 -33%/-50%. Recipes: 16 (ours c<64, B200/H200/GB300), 4 (ours TP2 c64, B300 c32+), 24/20
  (InferenceX MI355X DSV4 SGLang), ATOM DSV4 script ATOM_PREFILL_DECODE_INTERVAL=10.

## Open candidates

- ATOM trace under AgentX-like load (next, user-approved).
- (B) Explain/fix the EVENT_WAIT=1 GPU slowdown: rocprofv3 --kernel-trace on both ranks, EVENT_WAIT 0 vs 1,
  compare per-kernel durations (esp. ar_ll) -- ceiling ~0.85 ms/step at c1.
- Per-step GPU breakdown (fusion+tuned, c1, ms/step): MoE 2.25 (moe1 0.92, moe2 0.50, moe_sorting 0.49,
  topk_gating 0.27, router gemv 0.22, _fused_append_shared_experts 0.17, fused_mx_quant_moe_sort 0.17), gemm 1.83,
  sparse attn 1.22 (sparse_mla 0.76 + decode_reduce 0.18 + reduce_partials 0.17), mHC 0.55, ar_ll 0.50, wo_a
  split-K 0.35 + 0.17. Cheapest: fold _fused_append_shared_experts into topk (~-0.17 ms); larger: ATOM #2377 gluon
  fused sparse decode (removes the 2 reduces, ~-0.3 ms); wo_a split-K retune.
- Original plan items not done: P1 #2392 row-bounded block maxima (no c1 value), P2 DualRMSNormMXFP8 /
  rope_quant_window, P3 DSpark draft kernelization (ATOM 21.73 -> 6.67 launches/draft step), P4 gluon sparse
  decode, P5 Engram side-stream overlap (our TP2 uses the host table), P6 recipe knobs (mem 0.9, cuda-graph-max-bs
  128, chunk 16384).

## Measurement methods and traps

- **Proxy:** scripts/atomport_proxy_bench.py (random ids, c1, ctx 2k, OSL 1024, 3 reps) for relative A/Bs only;
  proxy P90 is higher than AgentX P90. 1-2% effects need the span metric.
- **Span:** proxy with `--profile-dir`, then scripts/atomport_step_spans.py (kernels/step, GPU span, host gap);
  scripts/atomport_step_sequence.py prints one step's kernel sequence; atomport_trace_breakdown.py groups by module.
  torch profiler: GPU activities only, no stack/shapes, few steps -- CPU+GPU hung both TP ranks in RCCL. Even GPU-only
  profiling occasionally hangs (NCCL watchdog 1 s after /start_profile); retry on a fresh server. The profiled host
  gap is inflated ~2x; use step time minus GPU span without the profiler for real idle.
- **Host:** `py-spy record --nonblocking -r 1000 -d 10 --format raw --pid <sglang::scheduler_TP0>` during a long
  decode, then scripts/pyspy_tree.py (tree under event_loop_overlap).
- **Evals:** GSM8K only on EVAL_ONLY=true servers (SIMULATE_ACC commits unverified drafts -> 0.39). Compare GSM8K on
  the same CONC setting (CONC=32 eval = mem 0.85 / max-running 64; CONC=1 = max-running 2, 5x slower).
  Greedy token parity across servers is unusable: two fresh servers differ at token 0 (target prefill
  nondeterminism); a second run on the same server hits the radix cache. Use GSM8K + real accept length instead.
- **Ops:** kill servers by PID AND their renamed children (`ps -eo pid,comm | awk '$2 ~ /^sglang::/'`), otherwise
  scheduler processes keep ~227 GB of VRAM. rocm-smi GPU[7] == HIP device 5 here and host PIDs show as UNKNOWN.
  Never `pkill -f` a pattern contained in your own command line. Do not write a script and launch it in the same
  parallel tool batch (the launch can run before the file exists).
