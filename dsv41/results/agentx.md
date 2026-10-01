# AgentX (InferenceX inferencex-agentx-mvp trace replay) -- DSV4.1-Flash

## CURRENT BEST per concurrency (2026-10-01, overnight sweep night0930_c*)
rolao/dsv41/opt-branch a5e40eca5e (router fusion + sort multi-phase + teammate wo_a M-bucketed tiles) +
`--enforce-shared-experts-fusion` + tuned FMoE CSV, TP2 EP1 on MI355X, two TP2 lanes in parallel (GPUs 4,5 / 6,7),
max-running-requests = 2 x conc (cap 64). TTT = tok/s/GPU, P90 = P90 interactivity (tok/s/user). 0 errors everywhere.

| 併發 | PDI | chunk | mem-fraction | 我們 TTT / P90 | ATOM TTT / P90 | 我們 vs ATOM |
|---|---|---|---|---|---|---|
| 1 | 16 | 16384 | 0.70 | 11,375.4 / 346.4 | 10,820.2 / 337 | +5.1% / +2.8% |
| 2 ² | 16 | 16384 | 0.70 | 11,692.9 / 315.7 | 11,266.6 / 324 | +3.8% / −2.6% |
| 4 ¹ | 16 | 4096 | 0.70 | 15,680.7 / 267.0 | 沒有這一點 | 沒有這一點 |
| 8 ³ | 16 | 16384 | 0.70 | 29,417.4 / 210.0 | 29,261.3 / 240 | +0.5% / −12.5% |
| 16 | 16 | 16384 | 0.80 | 54,080.7 / 134.5 | 53,993.3 / 128 | +0.2% / +5.1% |
| 32 | 4 | 16384 | 0.80 | 96,768.1 / 69.9 | 91,002.2 / 66.4 | +6.3% / +5.3% |
| 64 | 4 | 4096 | 0.85 | 117,918.7 / 40.1 | 102,389.5 / 23.1 | +15.2% / +73.6% |

(1) c4 is 026da361c0 (chunk 4096), not re-run. (2) c2 runs with the same settings: rf_c16k_c2_pdi16 (09-30, A+D on
5ec406, before the rebase onto 5562928323) 11,819.8 / 324.6, night0930_c2 315.7, night0930_c2_r2 11,639.7 / 310.3;
3-run mean P90 316.9 (-2.2% vs ATOM). Not a regression: p50 384.4 / 383.3 / 384.7 and no-overlap-subset P90 331.9 /
330.7 / 329.1 are flat; the spread is only in the 52-65 requests that overlap another lane's prefill (P90 tail noise
~5%). Same for c1: rf_c16k_c1_pdi16 349.5 vs 346.4, no-overlap P90 370.5 vs 370.4. (3) repeats night0930_c8_r2 29,360.0 / 213.3 and c8_a5e_c16k_pdi16 29,378.4 / 209.3; mean P90
210.9 (-12.1%); c8 PDI 32 on the same build 29,302.1 / 216.2 (c8_a5e_c16k_pdi32). TTFT p50/p90 (s): c1 0.72/1.11,
c2 0.43/0.99, c8 0.41/1.16, c16 0.48/1.73, c32 0.68/2.43, c64 2.25/16.02. Dirs /shared_nfs/kk/dsv41/agentx/night0930_c*.
Previous table (5ec406, 2026-09-30): c1 11,087.2/333.4, c2 11,494.7/293.2, c8 29,157.1/209.7, c16 53,469.1/135.7,
c32 97,158.2/69.5, c64 116,273.6/39.3 (dirs sw5ec406_c16k_c{1,2,8}_pdi16, sw5ec406_c16k_c16_pdi16,
sw5ec406_c16k_m080_c32_pdi4, sw5ec406_c4k_c64_pdi4).

## vLLM MI355X reference (public InferenceX table, pasted by user 2026-09-24)
FP4, TP2, 2 physical GPUs, no DP. Consistency: Total-tokens-per-$1 == TTT x 2400 on every row
(= 3600 s / $1.50 per GPU-hour => TTT is total (input+output) tok/s per GPU). conc 128 TTT halves vs conc 64 --
likely KV / max-num-seqs (=128 in dsv41flash_fp4_mi355x_vllm_mtp.sh) saturation, not a typo.

| conc | Total tokens / $1 | P90 interactivity (tok/s/user) | Throughput/Chip = TTT (tok/s/GPU) |
|---|---|---|---|
| 1 | 23,272,349 | 267 | 9,696.8 |
| 2 | 24,356,411 | 237 | 10,148.5 |
| 4 | 35,590,789 | 206 | 14,829.5 |
| 8 | 62,544,889 | 137 | 26,060.4 |
| 16 | 108,188,820 | 78.7 | 45,078.7 |
| 32 | 178,478,547 | 36.5 | 74,366.1 |
| 64 | 191,554,804 | 15.4 | 79,814.5 |
| 128 | 97,880,813 | 8.7 | 40,783.7 |

Local InferenceX checkout (b5d0e56a2) configs/amd-master.yaml `dsv41flash-fp4-mi355x-vllm-agentic-dspark` says
TP4, conc 1-32, image nightly-eed1f3d... -- the public TP2/conc-to-128 table above comes from a newer config/run.
Official duration: workflow default 3600 s (benchmark-tmpl.yml `duration` default '3600'); 1200 s only with
`agentx-fast`. Compare only 3600 s runs against this table.

## SGLang runs
Config decision (user, 2026-09-24): TP2 **EP1** only (TP2 EP2 hits an aiter MoE memory fault at 192 local experts;
not pursued). 3600 s per point, conc 4 / 16 / 64, run sequentially on GPU 0,1.

| date | commit | tp | conc | TTT (tok/s/GPU) | P90 interactivity | notes | result dir |
|---|---|---|---|---|---|---|---|
| 2026-09-24 | opus-prefill 048ffae315 | 2 (EP1) | 4 | 9,318 | 229.7 | SMOKE 600 s (not comparable), 0 faults, 0/117 errors | /shared_nfs/kk/dsv41/agentx/smoke_tp2ep1_c4_v2 |
| 2026-09-24 | opus-prefill 048ffae315, aiter v0.1.22.post1 | 2 (EP1) | 4 | 9,104 | 216.3 | SMOKE 600 s (not comparable), 0 faults, 0/128 errors | /shared_nfs/kk/dsv41/agentx/smoke_tp2ep1_c4_aiter0122 |
| 2026-09-24 | opus-prefill, aiter v0.1.22.post1, OPUS=1 | 2 (EP1) | 4 | 11,668.7 | 159.3 | 3600 s OK, 0/486 errors | /shared_nfs/kk/dsv41/agentx/tp2ep1_aiter0122_c4 |
| 2026-09-24 | same, OPUS=1 | 2 (EP1) | 16 | CRASH | - | GPU mem fault after ~45 min (OPUS prefill path) | /shared_nfs/kk/dsv41/agentx/tp2ep1_aiter0122_c16 |
| 2026-09-24 | same, OPUS=1 | 2 (EP1) | 64 | CRASH | - | GPU mem fault after ~7 min serving (OPUS prefill path) | /shared_nfs/kk/dsv41/agentx/tp2ep1_aiter0122_c64 |
| 2026-09-25 | opus-prefill c65a3acad7 (+#41159 kvstore int64), OPUS=1, old local script (no PDI) | 2 (EP1) | 64 | 11,985.3 | 1.7 | 3600 s OK, 0 faults, 0/864 errors (vLLM c64: 79,814.5 / 15.4) | /shared_nfs/kk/dsv41/agentx/tp2ep1_kvfix_c64 |
| 2026-09-25 | same + --prefill-decode-interval 4 (PDI_AUTO) | 2 (EP1) | 64 | 10,199.8 | 47.6 | 3600 s OK, 0 faults, 0/676 errors | /shared_nfs/kk/dsv41/agentx/kvfix_pdi_tp2ep1_c64 |
| 2026-09-25 | same + --prefill-decode-interval 16 | 2 (EP1) | 16 | 7,382.0 | 131.3 | 3600 s OK, 0 faults, 0/419 errors | /shared_nfs/kk/dsv41/agentx/kvfix_pdi_tp2ep1_c16 |
| 2026-09-25 | same + --prefill-decode-interval 16 | 2 (EP1) | 4 | 11,044.1 | 143.6 | 3600 s OK, 0 faults, 0/470 errors | /shared_nfs/kk/dsv41/agentx/kvfix_pdi_tp2ep1_c4 |
| 2026-09-26 | opus-prefill c65a3acad7, OPUS OFF, COLLEAGUE recipe (agentx_colleague_mi355x_sglang.sh) | 2 (EP1) | 16 (PDI 16) | 45,804.7 | 112.5 | 3600 s OK, 0 faults; colleague 45,518.84 / 110.02 -> REPRODUCED | /shared_nfs/kk/dsv41/agentx/colleague_noopus_tp2ep1_c16_pdi16 |
| 2026-09-26 | same, OPUS OFF, colleague recipe | 2 (EP1) | 64 (PDI 4) | 99,666.6 | 33.6 | 3600 s OK, 0 faults; colleague 96,837.78 / 32.15 -> REPRODUCED | /shared_nfs/kk/dsv41/agentx/colleague_noopus_tp2ep1_c64_pdi4 |
| 2026-09-26 | same, OPUS ON, colleague recipe | 2 (EP1) | 16 (PDI 16) | 46,717.9 | 117.3 | 3600 s OK, 0 faults; vs OPUS off +2.0% TTT, +4.3% P90, TTFT p50 0.73->0.67 s | /shared_nfs/kk/dsv41/agentx/colleague_opus_tp2ep1_c16_pdi16 |
| 2026-09-26 | same, OPUS ON, colleague recipe | 2 (EP1) | 64 (PDI 4) | 104,232.7 | 35.6 | 3600 s OK, 0 faults; vs OPUS off +4.6% TTT, +6.0% P90, TTFT p50 2.71->2.39 s | /shared_nfs/kk/dsv41/agentx/colleague_opus_tp2ep1_c64_pdi4 |
| 2026-09-26 | RolaoDenthu dsv41/opt-branch 2b875bd95a ALL opts (their OPUS + top-k v2 + aiter group32 GEMM, aiter #5750), colleague recipe | 2 (EP1) | 16 (PDI 16) | 48,522.9 | 126.5 | 3600 s OK, 0 faults; vs our OPUS 46,717.9 / 117.3 (+3.9% / +7.8%) | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c16_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 1 (PDI 16) | 9,818.6 | 267.5 | 3600 s OK, 0 faults; vLLM 9,696.8 / 267 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c1_pdi16 |
| 2026-09-28 | rolao 8c67e0bd51 (T2) + --enforce-shared-experts-fusion + requant RNE fix (uncommitted), colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,036.3 | 282.5 | 3600 s OK; vs rolao all-opts c1 +2.2% / +5.6% | /shared_nfs/kk/dsv41/agentx/atomport_sef_rne_agentx_c1_pdi16 |
| 2026-09-28 | rolao 8c67e0bd51 (T2 only, no fusion), colleague recipe | 2 (EP1) | 1 (PDI 16) | 9,977.9 | 277.5 | 3600 s OK; vs rolao all-opts c1 +1.6% / +3.7%; fusion row above is +0.6% / +1.8% vs this | /shared_nfs/kk/dsv41/agentx/atomport_t2only_agentx_c1_pdi16 |
| 2026-09-28 | rolao ba5eb0f439 + --enforce-shared-experts-fusion + local tuned FMoE CSV (385/7, 129/4), colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,269.1 | 287.5 | 3600 s OK; vs untuned fusion +2.3% / +1.8%, vs T2-only +2.9% / +3.6% | /shared_nfs/kk/dsv41/agentx/atomport_sef_tuned_agentx_c1_pdi16 |
| 2026-09-28 | + SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA (draft metadata in graph, uncommitted) on fusion + tuned FMoE, colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,306.1 | 292.8 | 3600 s OK, 0 faults; vs fusion+tuned +0.4% / +1.8%; vs T2-only +3.3% / +5.5% | /shared_nfs/kk/dsv41/agentx/atomport_draftraw_agentx_c1_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 2 (PDI 16) | 10,476.9 | 279.6 | 3600 s OK, 0 faults; vs rolao all-opts +6.6% / +8.3%; vs ATOM -7.0% / -13.7% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c2_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 4 (PDI 16) | 15,680.7 | 267.0 | 3600 s OK, 0 faults; vs rolao all-opts +3.8% / +10.3%; vs ATOM n/a | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c4_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 8 (PDI 16) | 28,552.6 | 199.3 | 3600 s OK, 0 faults; vs rolao all-opts +5.4% / +10.0%; vs ATOM -2.4% / -17.0% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c8_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 16 (PDI 16) | 51,219.5 | 137.8 | 3600 s OK, 0 faults; vs rolao all-opts +5.6% / +8.9%; vs ATOM -5.1% / +7.7% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c16_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 32 (PDI 16) | 90,876.4 | 102.0 | 3600 s OK, 0 faults; vs rolao all-opts +7.6% / +10.3%; vs ATOM -0.1% / +53.6% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c32_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 32 (PDI 4) | 93,855.8 | 68.7 | 3600 s OK, 0 faults; vs rolao all-opts +4.6% / +8.7%; vs ATOM +3.1% / +3.5% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c32_pdi4 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 64 (PDI 16) | 90,671.3 | 78.8 | 3600 s OK, 0 faults; vs rolao all-opts +7.6% / +3.1%; vs ATOM -11.4% / 3.4x | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c64_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 64 (PDI 4) | 117,189.7 | 39.7 | 3600 s OK, 0 faults; vs rolao all-opts +5.1% / +4.5%; vs ATOM +14.5% / +71.9% | /shared_nfs/kk/dsv41/agentx/atomport_best_sweep_c64_pdi4 |
| 2026-09-29 | BEST config + aiperf --extra-inputs temperature:0 (greedy DSpark path; ATOM synthetic-AL parity) | 2 (EP1) | 1 (PDI 16) | 11,170.2 | 330.7 | 3600 s OK, 282 ok; vs temp-default run +8.4% / +12.9%; vs ATOM +3.2% / -1.9% | /shared_nfs/kk/dsv41/agentx/gap_temp0_agentx_c1_pdi16 |
| 2026-09-29 | BEST config + SIMULATE_ACC_GREEDY strict (greedy draft/accept, temperature-sampled bonus; uncommitted), official client params | 2 (EP1) | 1 (PDI 16) | 10,969.6 | 317.5 | 3600 s OK, 279 ok; vs BEST +6.4% / +8.4%; vs ATOM +1.4% / -5.8% | /shared_nfs/kk/dsv41/agentx/simbonus_c1_pdi16 |
| 2026-09-29 | BEST config + SIMULATE_ACC_GREEDY strict (uncommitted), official client params | 2 (EP1) | 2 (PDI 16) | 11,253.3 | 292.6 | 3600 s OK, 434 ok; vs BEST +7.4% / +4.6%; vs ATOM -0.1% / -9.7% | /shared_nfs/kk/dsv41/agentx/simbonus_c2_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 (greedy draft/accept under SIMULATE + fused Gumbel-max bonus) + fusion + tuned FMoE, official client params | 2 (EP1) | 1 (PDI 16) | 11,141.9 | 321.5 | 3600 s OK, 281 ok; vs BEST 026da361 +8.1% / +9.8%; vs unfused strict +1.6% / +1.3%; vs ATOM +3.0% / -4.6% | /shared_nfs/kk/dsv41/agentx/f5ec406_c1_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 + fusion + tuned FMoE, official client params | 2 (EP1) | 2 (PDI 16) | 11,123.3 | 288.5 | 3600 s OK, 427 ok; vs BEST 026da361 +6.2% / +3.2%; vs unfused strict -1.2% / -1.4% (noise); vs ATOM -1.3% / -11.0% | /shared_nfs/kk/dsv41/agentx/f5ec406_c2_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 + fusion + tuned FMoE + --chunked-prefill-size 16384, official client params | 2 (EP1) | 2 (PDI 16) | 11,550.4 | 309.3 | 3600 s OK, 452 ok; vs chunk 4096 +3.8% / +7.2%, TTFT p50/p90 0.52/1.19 -> 0.42/0.96 s; vs ATOM +2.5% / -4.5% | /shared_nfs/kk/dsv41/agentx/f5ec406_chunk16k_c2_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 1 (PDI 16) | 11,087.2 | 333.4 | 3600 s OK, 281 ok; vs ATOM +2.5% / -1.1%; vs BEST sweep +7.6% / +13.9% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c1_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 2 (PDI 16) | 11,494.7 | 293.2 | 3600 s OK, 451 ok; vs ATOM +2.0% / -9.5%; vs BEST +9.7% / +4.9%; same config as the 309.3 run -> P90 run-to-run spread ~5% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c2_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 8 (PDI 16) | 29,157.1 | 209.7 | 3600 s OK, 1347 ok; vs ATOM -0.4% / -12.6%; vs BEST +2.1% / +5.2% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c8_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 16 (PDI 16, mem 0.80) | 53,469.1 | 135.7 | 3600 s OK, 2520 ok; vs ATOM -1.0% / +6.1%; vs BEST +4.4% / -1.5% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c16_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 32 (PDI 4, mem 0.85) | FAIL | FAIL | HIP OOM after ~20 min (0 bytes free, 512 MiB alloc); partial 710/1064 -- not a result | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c32_pdi4 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 64 (PDI 4, mem 0.85) | FAIL | FAIL | NCCL watchdog: BROADCAST numel 16 timed out 600 s at 18:23:51 (queue 42, pending 6M tokens); partial 489/1196 -- not a result | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_c64_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 32 (PDI 4, mem 0.80) | 97,158.2 | 69.5 | 3610 s OK, 4187 ok, 0 errors; TTFT p50/p90 0.60/2.05 s; vs ATOM +6.8% / +4.7%; vs chunk4096 026da361 +3.5% / +1.2% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_m080_c32_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 64 (PDI 4, mem 0.80) | 121,940.9 | 31.5 | 3627 s OK, 7009 ok; TTFT p50/p90 1.00/4.74 s; free device mem hit 0.00 GiB at 09:22 +08 but survived; vs ATOM +19.1% / +36.4%; vs chunk4096 026da361 +4.1% / -20.7% | /shared_nfs/kk/dsv41/agentx/sw5ec406_c16k_m080_c64_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE, CHUNKED_PREFILL_SIZE=4096 (re-baseline) | 2 (EP1) | 32 (PDI 4, mem 0.85) | 92,657.2 | 66.0 | 3600 s OK, 4087 ok, 0 errors; TTFT p50/p90 0.77/3.43 s; chunk 16384+mem0.80 is +4.9% / +5.3% over this -> c32 uses 16384 | /shared_nfs/kk/dsv41/agentx/sw5ec406_c4k_c32_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE, CHUNKED_PREFILL_SIZE=4096 (re-baseline) | 2 (EP1) | 64 (PDI 4, mem 0.85) | 116,273.6 | 39.3 | 3600 s OK, 6616 ok, 0 errors; TTFT p50/p90 2.33/16.17 s; vs ATOM +13.6% / +70.1%; chunk 16384+mem0.80 is +4.9% / -19.8% vs this -> c64 keeps 4096 | /shared_nfs/kk/dsv41/agentx/sw5ec406_c4k_c64_pdi4 |
| 2026-09-30 | ROUTER FUSION + sort MP (router-gate-shared-append worktree on 5ec406) + fusion + tuned FMoE, chunk 16384 | 2 (EP1) | 1 (PDI 16) | 11,371.9 | 349.5 | 3600 s OK, 285 ok, 0 errors; p50 383.9; TTFT p50/p90 0.67/1.21 s; vs 5ec406 sweep +2.6% / +4.8%; vs ATOM +5.1% / +3.7% | /shared_nfs/kk/dsv41/agentx/rf_c16k_c1_pdi16 |
| 2026-09-30 | ROUTER FUSION + sort MP (router-gate-shared-append worktree on 5ec406) + fusion + tuned FMoE, chunk 16384 | 2 (EP1) | 2 (PDI 16) | 11,819.8 | 324.6 | 3600 s OK, 455 ok, 0 errors; p50 384.4; TTFT p50/p90 0.47/1.12 s; vs 5ec406 sweep +2.8% / +10.7% (vs its 309.3 twin +4.9%); vs ATOM +4.9% / +0.2% | /shared_nfs/kk/dsv41/agentx/rf_c16k_c2_pdi16 |
| 2026-09-29 | BEST config, PDI 32 | 2 (EP1) | 8 (PDI 32) | 28,643.3 | 212.3 | 3600 s OK, 0 faults; vs PDI16 +0.3% / +6.5% (TTFT p50 0.45 -> 0.54 s, p90 1.42 -> 1.78 s); vs ATOM -2.1% / -11.5% | /shared_nfs/kk/dsv41/agentx/atomport_best_pdi_c8_pdi32 |
| 2026-09-29 | BEST config, PDI 4 | 2 (EP1) | 8 (PDI 4) | 28,586.2 | 200.0 | 3600 s OK, 0 faults; vs PDI16 +0.1% / +0.4% (TTFT p50 0.45 -> 0.39 s, p90 1.42 -> 1.17 s); vs ATOM -2.3% / -16.6% | /shared_nfs/kk/dsv41/agentx/atomport_best_pdi_c8_pdi4 |
| 2026-09-29 | BEST config, PDI 32 | 2 (EP1) | 2 (PDI 32) | 10,421.0 | 282.0 | 3600 s OK, 0 faults; vs PDI16 -0.5% / +0.9% (TTFT p50 0.50 -> 0.47 s, p90 1.11 -> 1.07 s); vs ATOM -7.5% / -13.0% | /shared_nfs/kk/dsv41/agentx/atomport_best_pdi_c2_pdi32 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 2 (PDI 16) | 9,832.8 | 258.2 | 3600 s OK, 0 faults; vLLM 10,148.5 / 237 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c2_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 4 (PDI 16) | 15,108.4 | 242.0 | 3600 s OK, 0 faults; vLLM 14,829.5 / 206 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c4_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 8 (PDI 16) | 27,101.0 | 181.1 | 3600 s OK, 0 faults; vLLM 26,060.4 / 137 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c8_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 32 (PDI 16) | 84,485.0 | 92.5 | 3600 s OK, 0 faults; vLLM 74,366.1 / 36.5 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c32_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 32 (PDI 4) | 89,715.2 | 63.2 | 3600 s OK, 0 faults; vLLM 74,366.1 / 36.5 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c32_pdi4 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 64 (PDI 16) | 84,277.3 | 76.4 | 3600 s OK, 0 faults; vLLM 79,814.5 / 15.4 | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c64_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 64 (PDI 4) | 111,506.4 | 38.0 | 3600 s OK, 0 faults; vLLM 79,814.5 / 15.4; our OPUS 104,232.7 / 35.6 (+7.0% / +6.7%) | /shared_nfs/kk/dsv41/agentx/rolao_allopts_tp2ep1_c64_pdi4 |

## ATOM MI355X reference (public InferenceX/SemiAnalysis table, pasted by user 2026-09-27)
FP4, TP2, 2 physical GPUs, no DP. No c4 row in the screenshot.

| conc | P90 interactivity (tok/s/user) | Throughput/Chip = TTT (tok/s/GPU) |
|---|---|---|
| 1 | 337 | 10,820.2 |
| 2 | 324 | 11,266.6 |
| 8 | 240 | 29,261.3 |
| 16 | 128 | 53,993.3 |
| 32 | 66.4 | 91,002.2 |
| 64 | 23.1 | 102,389.5 |

vs rolao all-opts (TTT / P90): c1 -9.3%/-20.6%, c2 -12.7%/-20.3%, c8 -7.4%/-24.5%, c16 -10.1%/-1.2%,
c32 PDI4 -1.4%/-4.8% (PDI16 -7.2%/+39%), c64 PDI4 +8.9%/+64% (PDI16 -17.7%/3.3x).
