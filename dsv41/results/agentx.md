# AgentX (InferenceX inferencex-agentx-mvp trace replay) -- DSV4.1-Flash

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
| 2026-09-29 | BEST config, PDI 32 | 2 (EP1) | 8 (PDI 32) | 28,643.3 | 212.3 | 3600 s OK, 0 faults; vs PDI16 +0.3% / +6.5% (TTFT p50 0.45 -> 0.54 s, p90 1.42 -> 1.78 s); vs ATOM -2.1% / -11.5% | /shared_nfs/kk/dsv41/agentx/atomport_best_pdi_c8_pdi32 |
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
