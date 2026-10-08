# AgentX (InferenceX inferencex-agentx-mvp trace replay) -- DSV4.1-Flash

## SUMMARY vs B200 vLLM NEW (run 37070984585, pulled 2026-10-05 by mi355-4)
Same MI355X numbers as the 2026-10-05 summary below; only the vLLM column changes. vLLM = best TTT of TP / DEP at that
GPU count (TP2 points are DEP2 only; TP4 c64 DEP4 117,168.9 > TP4 116,807.6). Raw table: "Reference: vLLM on B200 NEW".

| 併發 | TP | BCG TTT / P90 | Replay TTT / P90 | B200 vLLM TTT / P90 | BCG vs vLLM | Replay vs vLLM |
|---|---|---|---|---|---|---|
| 1 | 2 | 11,375.4 / 346.4 | 11,697.4 / 357.7 | not in new run | – | – |
| 2 | 2 | 11,692.9 / 315.7 | 12,022.3 / 330.7 | not in new run | – | – |
| 8 | 2 | 29,417.4 / 210.0 | 29,760.8 / 237.2 | 31,045.0 / 287.9 (DEP2) | -5.2% / -27.1% | -4.1% / -17.6% |
| 16 | 2 | 54,080.7 / 134.5 | 54,345.3 / 165.0 | 56,674.2 / 230.1 (DEP2) | -4.6% / -41.5% | -4.1% / -28.3% |
| 32 | 2 | 96,768.1 / 69.9 | 105,800.1 / 93.7 | 117,908.2 / 149.5 (DEP2) | -17.9% / -53.2% | -10.3% / -37.3% |
| 64 | 2 | 117,918.7 / 40.1 | 142,243.2 / 52.0 | not in new run | – | – |
| 1 | 4 | 6,188.1 / 391.8 | 6,271.4 / 406.7 | 7,698.6 / 583.2 (TP4) | -19.6% / -32.8% | -18.5% / -30.3% |
| 2 | 4 | 6,362.9 / 368.0 | 6,322.4 / 373.4 | not in new run | – | – |
| 8 | 4 | 15,290.4 / 278.6 | 15,335.5 / 285.8 | 16,549.2 / 385.0 (TP4) | -7.6% / -27.6% | -7.3% / -25.8% |
| 16 | 4 | 27,946.0 / 203.4 | 28,141.6 / 220.1 | 29,412.5 / 317.6 (TP4) | -5.0% / -36.0% | -4.3% / -30.7% |
| 32 | 4 | 56,039.2 / 120.6 | 57,890.6 / 138.5 | 62,356.4 / 221.4 (TP4) | -10.1% / -45.5% | -7.2% / -37.4% |
| 64 | 4 | 81,735.1 / 60.1 | 88,391.8 / 66.5 | 117,168.9 / 138.8 (DEP4) | -30.2% / -56.7% | -24.6% / -52.1% |

## Reference: vLLM on B200 NEW (InferenceX run 37070984585, 2026-10-02, pulled 2026-10-05 by mi355-4)
[Run Sweep](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37070984585/attempts/1) a8504a430 "Update B200 DSV41flash
vLLM image and retune TP4/DEP2/DEP4", b200-nscale, vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7, spec mtp. No
agg_bmk artifact; per-point bmk_agentic_* JSONs in /shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_b200_vllm/run37070984585/ (same fields as below).
Only these 12 points were swept (no pure TP2, no TP4 c2).
Recipe (a8504a430 srt-slurm-recipes/dsv41flash/vllm/b200-fp4-mtp/agentic.yaml): TP4 = tensor-parallel 4, no EP -> TP MoE +
all-reduce, engram cpu_offload + use_thp. DEP2/DEP4 = TP1 x DP2/DP4 + enable-expert-parallel, kernel-config
moe_backend=deep_gemm_mega_moe, engram embedding_across_dp, vllm-router consistent_hash in front of the DP ranks,
max-num-batched-tokens 4096 (DEP2). EP is A2A, not all-reduce: DeepseekV4MegaMoEExperts (vllm deepseek_v4/nvidia/model.py
@ac9126e58) writes tokens into deep_gemm.get_symm_buffer_for_mega_moe(ep_group) and fp8_fp4_mega_moe fuses dispatch + GEMM
+ combine over NVLink symmetric memory (the "AgRsAll2AllManager" log line is the generic manager, not on this path).
Throughput runs use rejection_sample_method=synthetic, synthetic_acceptance_length 3.51 (server log DEP2 c16).
vLLM engram embedding_across_dp=true = one table sharded over all TP x DP ranks, per-step gather ids / lookup owned rows /
exchange back. SGLang equivalent is built in, no knob: under DP attention tp_size spans all ranks, EngramEmbedding shards
rows over it and _dp_sharded_lookup does dp_gather_replicate ids -> _owned_rows -> dp_reduce_scatter (HIP int32 path),
host table per_rank works with it. SGLang has no counterpart to vLLM's default (one TP-sharded replica per DP rank, no
per-step collective) nor to dp_shared_memory (one /dev/shm copy shared by co-located DP replicas, no collective).

| conc | DEP2 (tp2 ep2 dpa) | TP4 (ep1) | DEP4 (tp4 ep4 dpa) | ok / total requests |
|---|---|---|---|---|
| 1 | – | 7,698.6 / 583.2 | – | TP4 334/345 |
| 4 | – | 10,293.2 / 446.0 | – | TP4 770/814 |
| 8 | 31,045.0 / 287.9 | 16,549.2 / 385.0 | – | 1399/1486, 1437/1524 |
| 16 | 56,674.2 / 230.1 | 29,412.5 / 317.6 | – | 2634/2811, 2722/2900 |
| 32 | 117,908.2 / 149.5 | 62,356.4 / 221.4 | – | 4737/5091, 4927/5281 |
| 64 | – | 116,807.6 / 142.9 | 117,168.9 / 138.8 | 11500/12211, 11543/12250 |
| 128 | – | 166,192.5 / 72.2 | 173,193.9 / 81.2 | 18300/19724, 18804/20226 |

## SUMMARY vs B200 vLLM (2026-10-05): TP2 / TP4, BCG vs Replay
TTT = tok/s/GPU, P90 = P90 interactivity. MI355X, EP1, engram host table, MTP/DSpark. Replay = --enable-decoder-swa-bounded-replay
--cuda-graph-backend-prefill disabled; BCG = breakable prefill graph, no replay. B200 vLLM = InferenceX run 36423355395 (same TP).

| 併發 | TP | PDI | chunk | mem-fraction | BCG TTT / P90 | Replay TTT / P90 | B200 vLLM TTT / P90 | BCG vs vLLM | Replay vs vLLM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 16 | 16384 | 0.70 | 11,375.4 / 346.4 | 11,697.4 / 357.7 | 13,128.0 / 393.1 | -13.4% / -11.9% | -10.9% / -9.0% |
| 2 | 2 | 16 | 16384 | 0.70 | 11,692.9 / 315.7 | 12,022.3 / 330.7 | 13,448.7 / 373.4 | -13.1% / -15.5% | -10.6% / -11.4% |
| 8 | 2 | 16 | 16384 | 0.70 | 29,417.4 / 210.0 | 29,760.8 / 237.2 | 31,009.1 / 252.7 | -5.1% / -16.9% | -4.0% / -6.1% |
| 16 | 2 | 16 | 16384 | 0.80 | 54,080.7 / 134.5 | 54,345.3 / 165.0 | 55,804.0 / 175.0 | -3.1% / -23.1% | -2.6% / -5.7% |
| 32 | 2 | 4 | 16384 | 0.80 | 96,768.1 / 69.9 | 105,800.1 / 93.7 | 93,351.7 / 67.3 | +3.7% / +3.9% | +13.3% / +39.2% |
| 64 | 2 | 4 | 4096 | 0.85 | 117,918.7 / 40.1 | 142,243.2 / 52.0 | 20,502.5 / 28.1 ⁴ | +475.1% / +42.7% | +593.8% / +85.1% |
| 1 | 4 | 16 | 16384 | 0.70 | 6,188.1 / 391.8 | 6,271.4 / 406.7 | 7,282.7 / 512.3 | -15.0% / -23.5% | -13.9% / -20.6% |
| 2 | 4 | 16 | 16384 | 0.70 | 6,362.9 / 368.0 | 6,322.4 / 373.4 | 7,450.1 / 470.2 | -14.6% / -21.7% | -15.1% / -20.6% |
| 8 | 4 | 16 | 16384 | 0.70 | 15,290.4 / 278.6 | 15,335.5 / 285.8 | 16,127.4 / 316.7 | -5.2% / -12.0% | -4.9% / -9.8% |
| 16 | 4 | 16 | 16384 | 0.80 | 27,946.0 / 203.4 | 28,141.6 / 220.1 | 28,773.9 / 226.6 | -2.9% / -10.2% | -2.2% / -2.9% |
| 32 | 4 | 4 | 16384 | 0.80 | 56,039.2 / 120.6 | 57,890.6 / 138.5 | 60,400.2 / 162.1 | -7.2% / -25.6% | -4.2% / -14.6% |
| 64 | 4 | 4 | 4096 | 0.85 | 81,735.1 / 60.1 | 88,391.8 / 66.5 | 107,014.2 / 91.1 | -23.6% / -34.0% | -17.4% / -27.0% |

TP2 BCG = night0930_c* (rolao/dsv41/opt-branch a5e40eca5e, different code from main); TP2 Replay = tp2r_c* (main 58f0d250ec + #42055);
TP4 BCG / Replay = tp4s_c*_bcg / _rep (both main + #42055, same lane). (4) vLLM TP2 c64 collapsed (1,252 profiled requests), not a fair point.

## NEW BEST candidate: + aiter sparse decode #5833 + #6042 (2026-10-01, s6042_c*, 1 run each, 0 errors)
Same as the table below (a5e40eca5e, same PDI / chunk / mem per point) but PYTHONPATH aiter =
/sgl-workspace/aiter-5750-sparse6042 (aiter-5750 + pa_decode_sparse.py and gfx950 sparse_mla.py from #5833+#6042;
not committed anywhere yet). Baseline = night0930 (c2 = mean of 2 runs, c8 = mean of 3 runs).

| conc | s6042 TTT / P90 | baseline TTT / P90 | vs baseline | vs ATOM TTT / P90 |
|---|---|---|---|---|
| 1 | 11,521.6 / 367.5 | 11,375.4 / 346.4 | +1.3% / +6.1% | +6.5% / +9.1% |
| 2 | 11,802.3 / 334.5 | 11,666.3 / 313.0 | +1.2% / +6.9% | +4.8% / +3.2% |
| 8 | 29,529.9 / 225.8 | 29,385.3 / 210.9 | +0.5% / +7.1% | +0.9% / −5.9% |
| 16 | 54,303.0 / 142.8 | 54,080.7 / 134.5 | +0.4% / +6.2% | +0.6% / +11.6% |
| 32 | 98,624.1 / 71.5 | 96,768.1 / 69.9 | +1.9% / +2.3% | +8.4% / +7.7% |
| 64 | 119,969.9 / 40.8 | 117,918.7 / 40.1 | +1.7% / +1.7% | +17.2% / +76.6% |

c64 TTFT p50/p90 2.25/16.02 -> 1.79/14.17 s. Pass criteria: all met except c8 P90 (−5.9% vs ATOM, bar −5%); the c8
gap is attributed to ATOM's synthetic-acceptance routing collapse (ATOM_PORT.md STEP 4).

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
c2 0.43/0.99, c8 0.41/1.16, c16 0.48/1.73, c32 0.68/2.43, c64 2.25/16.02. Dirs /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/night0930_c*.
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
| 2026-09-24 | opus-prefill 048ffae315 | 2 (EP1) | 4 | 9,318 | 229.7 | SMOKE 600 s (not comparable), 0 faults, 0/117 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/smoke_tp2ep1_c4_v2 |
| 2026-09-24 | opus-prefill 048ffae315, aiter v0.1.22.post1 | 2 (EP1) | 4 | 9,104 | 216.3 | SMOKE 600 s (not comparable), 0 faults, 0/128 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/smoke_tp2ep1_c4_aiter0122 |
| 2026-09-24 | opus-prefill, aiter v0.1.22.post1, OPUS=1 | 2 (EP1) | 4 | 11,668.7 | 159.3 | 3600 s OK, 0/486 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/tp2ep1_aiter0122_c4 |
| 2026-09-24 | same, OPUS=1 | 2 (EP1) | 16 | CRASH | - | GPU mem fault after ~45 min (OPUS prefill path) | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/tp2ep1_aiter0122_c16 |
| 2026-09-24 | same, OPUS=1 | 2 (EP1) | 64 | CRASH | - | GPU mem fault after ~7 min serving (OPUS prefill path) | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/tp2ep1_aiter0122_c64 |
| 2026-09-25 | opus-prefill c65a3acad7 (+#41159 kvstore int64), OPUS=1, old local script (no PDI) | 2 (EP1) | 64 | 11,985.3 | 1.7 | 3600 s OK, 0 faults, 0/864 errors (vLLM c64: 79,814.5 / 15.4) | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/tp2ep1_kvfix_c64 |
| 2026-09-25 | same + --prefill-decode-interval 4 (PDI_AUTO) | 2 (EP1) | 64 | 10,199.8 | 47.6 | 3600 s OK, 0 faults, 0/676 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/kvfix_pdi_tp2ep1_c64 |
| 2026-09-25 | same + --prefill-decode-interval 16 | 2 (EP1) | 16 | 7,382.0 | 131.3 | 3600 s OK, 0 faults, 0/419 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/kvfix_pdi_tp2ep1_c16 |
| 2026-09-25 | same + --prefill-decode-interval 16 | 2 (EP1) | 4 | 11,044.1 | 143.6 | 3600 s OK, 0 faults, 0/470 errors | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/kvfix_pdi_tp2ep1_c4 |
| 2026-09-26 | opus-prefill c65a3acad7, OPUS OFF, COLLEAGUE recipe (agentx_colleague_mi355x_sglang.sh) | 2 (EP1) | 16 (PDI 16) | 45,804.7 | 112.5 | 3600 s OK, 0 faults; colleague 45,518.84 / 110.02 -> REPRODUCED | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/colleague_noopus_tp2ep1_c16_pdi16 |
| 2026-09-26 | same, OPUS OFF, colleague recipe | 2 (EP1) | 64 (PDI 4) | 99,666.6 | 33.6 | 3600 s OK, 0 faults; colleague 96,837.78 / 32.15 -> REPRODUCED | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/colleague_noopus_tp2ep1_c64_pdi4 |
| 2026-09-26 | same, OPUS ON, colleague recipe | 2 (EP1) | 16 (PDI 16) | 46,717.9 | 117.3 | 3600 s OK, 0 faults; vs OPUS off +2.0% TTT, +4.3% P90, TTFT p50 0.73->0.67 s | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/colleague_opus_tp2ep1_c16_pdi16 |
| 2026-09-26 | same, OPUS ON, colleague recipe | 2 (EP1) | 64 (PDI 4) | 104,232.7 | 35.6 | 3600 s OK, 0 faults; vs OPUS off +4.6% TTT, +6.0% P90, TTFT p50 2.71->2.39 s | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/colleague_opus_tp2ep1_c64_pdi4 |
| 2026-09-26 | RolaoDenthu dsv41/opt-branch 2b875bd95a ALL opts (their OPUS + top-k v2 + aiter group32 GEMM, aiter #5750), colleague recipe | 2 (EP1) | 16 (PDI 16) | 48,522.9 | 126.5 | 3600 s OK, 0 faults; vs our OPUS 46,717.9 / 117.3 (+3.9% / +7.8%) | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c16_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 1 (PDI 16) | 9,818.6 | 267.5 | 3600 s OK, 0 faults; vLLM 9,696.8 / 267 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c1_pdi16 |
| 2026-09-28 | rolao 8c67e0bd51 (T2) + --enforce-shared-experts-fusion + requant RNE fix (uncommitted), colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,036.3 | 282.5 | 3600 s OK; vs rolao all-opts c1 +2.2% / +5.6% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_sef_rne_agentx_c1_pdi16 |
| 2026-09-28 | rolao 8c67e0bd51 (T2 only, no fusion), colleague recipe | 2 (EP1) | 1 (PDI 16) | 9,977.9 | 277.5 | 3600 s OK; vs rolao all-opts c1 +1.6% / +3.7%; fusion row above is +0.6% / +1.8% vs this | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_t2only_agentx_c1_pdi16 |
| 2026-09-28 | rolao ba5eb0f439 + --enforce-shared-experts-fusion + local tuned FMoE CSV (385/7, 129/4), colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,269.1 | 287.5 | 3600 s OK; vs untuned fusion +2.3% / +1.8%, vs T2-only +2.9% / +3.6% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_sef_tuned_agentx_c1_pdi16 |
| 2026-09-28 | + SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA (draft metadata in graph, uncommitted) on fusion + tuned FMoE, colleague recipe | 2 (EP1) | 1 (PDI 16) | 10,306.1 | 292.8 | 3600 s OK, 0 faults; vs fusion+tuned +0.4% / +1.8%; vs T2-only +3.3% / +5.5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_draftraw_agentx_c1_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 2 (PDI 16) | 10,476.9 | 279.6 | 3600 s OK, 0 faults; vs rolao all-opts +6.6% / +8.3%; vs ATOM -7.0% / -13.7% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c2_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 4 (PDI 16) | 15,680.7 | 267.0 | 3600 s OK, 0 faults; vs rolao all-opts +3.8% / +10.3%; vs ATOM n/a | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c4_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 8 (PDI 16) | 28,552.6 | 199.3 | 3600 s OK, 0 faults; vs rolao all-opts +5.4% / +10.0%; vs ATOM -2.4% / -17.0% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c8_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 16 (PDI 16) | 51,219.5 | 137.8 | 3600 s OK, 0 faults; vs rolao all-opts +5.6% / +8.9%; vs ATOM -5.1% / +7.7% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c16_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 32 (PDI 16) | 90,876.4 | 102.0 | 3600 s OK, 0 faults; vs rolao all-opts +7.6% / +10.3%; vs ATOM -0.1% / +53.6% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c32_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 32 (PDI 4) | 93,855.8 | 68.7 | 3600 s OK, 0 faults; vs rolao all-opts +4.6% / +8.7%; vs ATOM +3.1% / +3.5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c32_pdi4 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 64 (PDI 16) | 90,671.3 | 78.8 | 3600 s OK, 0 faults; vs rolao all-opts +7.6% / +3.1%; vs ATOM -11.4% / 3.4x | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c64_pdi16 |
| 2026-09-29 | BEST sweep: rolao 026da361c0 + fusion + tuned FMoE CSV, colleague recipe | 2 (EP1) | 64 (PDI 4) | 117,189.7 | 39.7 | 3600 s OK, 0 faults; vs rolao all-opts +5.1% / +4.5%; vs ATOM +14.5% / +71.9% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_sweep_c64_pdi4 |
| 2026-09-29 | BEST config + aiperf --extra-inputs temperature:0 (greedy DSpark path; ATOM synthetic-AL parity) | 2 (EP1) | 1 (PDI 16) | 11,170.2 | 330.7 | 3600 s OK, 282 ok; vs temp-default run +8.4% / +12.9%; vs ATOM +3.2% / -1.9% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/gap_temp0_agentx_c1_pdi16 |
| 2026-09-29 | BEST config + SIMULATE_ACC_GREEDY strict (greedy draft/accept, temperature-sampled bonus; uncommitted), official client params | 2 (EP1) | 1 (PDI 16) | 10,969.6 | 317.5 | 3600 s OK, 279 ok; vs BEST +6.4% / +8.4%; vs ATOM +1.4% / -5.8% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/simbonus_c1_pdi16 |
| 2026-09-29 | BEST config + SIMULATE_ACC_GREEDY strict (uncommitted), official client params | 2 (EP1) | 2 (PDI 16) | 11,253.3 | 292.6 | 3600 s OK, 434 ok; vs BEST +7.4% / +4.6%; vs ATOM -0.1% / -9.7% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/simbonus_c2_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 (greedy draft/accept under SIMULATE + fused Gumbel-max bonus) + fusion + tuned FMoE, official client params | 2 (EP1) | 1 (PDI 16) | 11,141.9 | 321.5 | 3600 s OK, 281 ok; vs BEST 026da361 +8.1% / +9.8%; vs unfused strict +1.6% / +1.3%; vs ATOM +3.0% / -4.6% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/f5ec406_c1_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 + fusion + tuned FMoE, official client params | 2 (EP1) | 2 (PDI 16) | 11,123.3 | 288.5 | 3600 s OK, 427 ok; vs BEST 026da361 +6.2% / +3.2%; vs unfused strict -1.2% / -1.4% (noise); vs ATOM -1.3% / -11.0% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/f5ec406_c2_pdi16 |
| 2026-09-29 | rolao 5ec406bb76 + fusion + tuned FMoE + --chunked-prefill-size 16384, official client params | 2 (EP1) | 2 (PDI 16) | 11,550.4 | 309.3 | 3600 s OK, 452 ok; vs chunk 4096 +3.8% / +7.2%, TTFT p50/p90 0.52/1.19 -> 0.42/0.96 s; vs ATOM +2.5% / -4.5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/f5ec406_chunk16k_c2_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 1 (PDI 16) | 11,087.2 | 333.4 | 3600 s OK, 281 ok; vs ATOM +2.5% / -1.1%; vs BEST sweep +7.6% / +13.9% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c1_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 2 (PDI 16) | 11,494.7 | 293.2 | 3600 s OK, 451 ok; vs ATOM +2.0% / -9.5%; vs BEST +9.7% / +4.9%; same config as the 309.3 run -> P90 run-to-run spread ~5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c2_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 8 (PDI 16) | 29,157.1 | 209.7 | 3600 s OK, 1347 ok; vs ATOM -0.4% / -12.6%; vs BEST +2.1% / +5.2% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c8_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 16 (PDI 16, mem 0.80) | 53,469.1 | 135.7 | 3600 s OK, 2520 ok; vs ATOM -1.0% / +6.1%; vs BEST +4.4% / -1.5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c16_pdi16 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 32 (PDI 4, mem 0.85) | FAIL | FAIL | HIP OOM after ~20 min (0 bytes free, 512 MiB alloc); partial 710/1064 -- not a result | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c32_pdi4 |
| 2026-09-29 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 64 (PDI 4, mem 0.85) | FAIL | FAIL | NCCL watchdog: BROADCAST numel 16 timed out 600 s at 18:23:51 (queue 42, pending 6M tokens); partial 489/1196 -- not a result | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_c64_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 32 (PDI 4, mem 0.80) | 97,158.2 | 69.5 | 3610 s OK, 4187 ok, 0 errors; TTFT p50/p90 0.60/2.05 s; vs ATOM +6.8% / +4.7%; vs chunk4096 026da361 +3.5% / +1.2% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_m080_c32_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE + CHUNKED_PREFILL_SIZE=16384 | 2 (EP1) | 64 (PDI 4, mem 0.80) | 121,940.9 | 31.5 | 3627 s OK, 7009 ok; TTFT p50/p90 1.00/4.74 s; free device mem hit 0.00 GiB at 09:22 +08 but survived; vs ATOM +19.1% / +36.4%; vs chunk4096 026da361 +4.1% / -20.7% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c16k_m080_c64_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE, CHUNKED_PREFILL_SIZE=4096 (re-baseline) | 2 (EP1) | 32 (PDI 4, mem 0.85) | 92,657.2 | 66.0 | 3600 s OK, 4087 ok, 0 errors; TTFT p50/p90 0.77/3.43 s; chunk 16384+mem0.80 is +4.9% / +5.3% over this -> c32 uses 16384 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c4k_c32_pdi4 |
| 2026-09-30 | SWEEP rolao 5ec406bb76 + fusion + tuned FMoE, CHUNKED_PREFILL_SIZE=4096 (re-baseline) | 2 (EP1) | 64 (PDI 4, mem 0.85) | 116,273.6 | 39.3 | 3600 s OK, 6616 ok, 0 errors; TTFT p50/p90 2.33/16.17 s; vs ATOM +13.6% / +70.1%; chunk 16384+mem0.80 is +4.9% / -19.8% vs this -> c64 keeps 4096 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/sw5ec406_c4k_c64_pdi4 |
| 2026-09-30 | ROUTER FUSION + sort MP (router-gate-shared-append worktree on 5ec406) + fusion + tuned FMoE, chunk 16384 | 2 (EP1) | 1 (PDI 16) | 11,371.9 | 349.5 | 3600 s OK, 285 ok, 0 errors; p50 383.9; TTFT p50/p90 0.67/1.21 s; vs 5ec406 sweep +2.6% / +4.8%; vs ATOM +5.1% / +3.7% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rf_c16k_c1_pdi16 |
| 2026-09-30 | ROUTER FUSION + sort MP (router-gate-shared-append worktree on 5ec406) + fusion + tuned FMoE, chunk 16384 | 2 (EP1) | 2 (PDI 16) | 11,819.8 | 324.6 | 3600 s OK, 455 ok, 0 errors; p50 384.4; TTFT p50/p90 0.47/1.12 s; vs 5ec406 sweep +2.8% / +10.7% (vs its 309.3 twin +4.9%); vs ATOM +4.9% / +0.2% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rf_c16k_c2_pdi16 |
| 2026-09-29 | BEST config, PDI 32 | 2 (EP1) | 8 (PDI 32) | 28,643.3 | 212.3 | 3600 s OK, 0 faults; vs PDI16 +0.3% / +6.5% (TTFT p50 0.45 -> 0.54 s, p90 1.42 -> 1.78 s); vs ATOM -2.1% / -11.5% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_pdi_c8_pdi32 |
| 2026-09-29 | BEST config, PDI 4 | 2 (EP1) | 8 (PDI 4) | 28,586.2 | 200.0 | 3600 s OK, 0 faults; vs PDI16 +0.1% / +0.4% (TTFT p50 0.45 -> 0.39 s, p90 1.42 -> 1.17 s); vs ATOM -2.3% / -16.6% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_pdi_c8_pdi4 |
| 2026-09-29 | BEST config, PDI 32 | 2 (EP1) | 2 (PDI 32) | 10,421.0 | 282.0 | 3600 s OK, 0 faults; vs PDI16 -0.5% / +0.9% (TTFT p50 0.50 -> 0.47 s, p90 1.11 -> 1.07 s); vs ATOM -7.5% / -13.0% | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/atomport_best_pdi_c2_pdi32 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 2 (PDI 16) | 9,832.8 | 258.2 | 3600 s OK, 0 faults; vLLM 10,148.5 / 237 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c2_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 4 (PDI 16) | 15,108.4 | 242.0 | 3600 s OK, 0 faults; vLLM 14,829.5 / 206 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c4_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 8 (PDI 16) | 27,101.0 | 181.1 | 3600 s OK, 0 faults; vLLM 26,060.4 / 137 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c8_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 32 (PDI 16) | 84,485.0 | 92.5 | 3600 s OK, 0 faults; vLLM 74,366.1 / 36.5 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c32_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 32 (PDI 4) | 89,715.2 | 63.2 | 3600 s OK, 0 faults; vLLM 74,366.1 / 36.5 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c32_pdi4 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 64 (PDI 16) | 84,277.3 | 76.4 | 3600 s OK, 0 faults; vLLM 79,814.5 / 15.4 | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c64_pdi16 |
| 2026-09-26 | same RolaoDenthu ALL opts, colleague recipe | 2 (EP1) | 64 (PDI 4) | 111,506.4 | 38.0 | 3600 s OK, 0 faults; vLLM 79,814.5 / 15.4; our OPUS 104,232.7 / 35.6 (+7.0% / +6.7%) | /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/rolao_allopts_tp2ep1_c64_pdi4 |

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

## Reference: vLLM on B200 (InferenceX run 36423355395, 2026-09-28, pulled 2026-10-02 by mi355-4)
[Run Sweep #14800](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36423355395): dsv41flash fp4, b200-nscale,
vllm/vllm-openai:nightly-ddd6fbca (FlashInfer sparse indexer + fp8 KV), spec mtp (DSpark). From artifact results_bmk/agg_bmk.json
(raw copy /shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_b200_vllm/ix_run36423355395_agg_bmk.json). tok/s/GPU = request_metrics.throughput.per_gpu.total_tput_tps,
P90 = request_metrics.latency.intvty.p90 (same fields as agentx/summary_table.py). Missing requests are warmup drops; errors <= 9 per point.

| conc | TP2 tok/s/GPU / P90 | TP4 tok/s/GPU / P90 |
|---|---|---|
| 1 | 13,128.0 / 393.1 | 7,282.7 / 512.3 |
| 2 | 13,448.7 / 373.4 | 7,450.1 / 470.2 |
| 4 | 18,432.2 / 319.9 | 9,703.1 / 424.5 |
| 8 | 31,009.1 / 252.7 | 16,127.4 / 316.7 |
| 16 | 55,804.0 / 175.0 | 28,773.9 / 226.6 |
| 32 | 93,351.7 / 67.3 | 60,400.2 / 162.1 |
| 64 | 20,502.5 / 28.1 (collapsed, 1252 profiled) | 107,014.2 / 91.1 |
| 128 | 18,694.9 / 25.5 (collapsed, 1491 profiled) | 141,302.6 / 44.9 |

Our best TP2 (MI355X, 2 GPUs) vs vLLM B200 TP2 (same GPU count), tok/s/GPU / P90 (2026-10-02):
c1 s6042 11,521.6 / 367.5 vs 13,128.0 / 393.1 = -12.2% / -6.5%; c2 s6042 11,802.3 / 334.5 vs 13,448.7 / 373.4 = -12.2% / -10.4%;
c4 026da361c0 15,680.7 / 267.0 vs 18,432.2 / 319.9 = -14.9% / -16.5%; c8 s6042 29,529.9 / 225.8 vs 31,009.1 / 252.7 = -4.8% / -10.6%;
c16 s6042 54,303.0 / 142.8 vs 55,804.0 / 175.0 = -2.7% / -18.4%; c32 s6042 98,624.1 / 71.5 vs 93,351.7 / 67.3 = +5.6% / +6.2%;
c64 s6042 119,969.9 / 40.8 vs 20,502.5 / 28.1 (vLLM TP2 collapsed). No SGLang c128. s6042 used aiter-5750-sparse6042 + ATOM-port
worktree (not in the current container); current main + #42055 matches env1001 (MAIN_REGRESS_1002.md).
SGLang TP4 sweep (c1-c64, recipe defaults) running 2026-10-02 on mi355-4, see MAIN_REGRESS_1002.md.
SGLang TP4 c64 (2026-10-02 mi355-4, tp4dbg_hosttbl_c64): main 58f0d250ec + #42055, TP4 EP1, engram host table per_rank,
PDI16 / chunk 4096 / mem 0.70: 65,201.5 / 91.4 (p50 129.6, TTFT p50/p90 12.19/30.19 s, cache hit 0.963, 7029 profiled, 0 err)
vs vLLM B200 TP4 c64 107,014.2 / 91.1 = -39.1% / +0.3%. (Engram GPU-resident tp4m_c64 was invalid: ~0 prefix-cache hits.)
SGLang TP4 c64 + --enable-decoder-swa-bounded-replay --cuda-graph-backend-prefill disabled (tp4_bsr_c64, 2026-10-02 mi355-4):
72,693.2 / 99.4 (p50 143.2, TTFT p50/p90 10.12/25.30 s, cache 0.964, 7798 profiled, 0 err) = +11.5% / +8.8% vs the same
config without it (65,201.5 / 91.4); vs vLLM B200 TP4 c64 107,014.2 / 91.1 = -32.1% / +9.1%. vLLM's --swa-bounded-replay
(default ON, vllm/config/cache.py) also keeps SWA KV out of the prefix cache; SGLang's flag only does the late-layer tail.
Same + PDI 4 (tp4_bsr_pdi4_c64): 87,850.1 / 65.7 (p50 101.5, TTFT p50/p90 1.62/6.91 s, cache 0.964, 9427 profiled, 0 err)
= +20.9% / -33.9% vs PDI16 (72,693.2 / 99.4); vs vLLM B200 TP4 c64 107,014.2 / 91.1 = -17.9% / -27.9%.

## SGLang TP4 sweep tp4s_* (2026-10-02/03 mi355-4, one lane GPUs 0-3; see MAIN_REGRESS_1002.md)
TP4 EP1, engram host table, main 58f0d250ec + #42055, TP2-best per-point PDI/chunk/mem. bcg = breakable prefill graph;
rep = prefill graph disabled + --enable-decoder-swa-bounded-replay. tok/s/GPU / P90; vs vLLM B200 TP4 (run 36423355395).

| conc | bcg | rep | rep vs bcg | bcg vs vLLM | rep vs vLLM |
|---|---|---|---|---|---|
| 1 | 6,188.1 / 391.8 | 6,271.4 / 406.7 | +1.3% / +3.8% | -15.0% / -23.5% | -13.9% / -20.6% |
| 2 | 6,362.9 / 368.0 | 6,322.4 / 373.4 | -0.6% / +1.5% | -14.6% / -21.7% | -15.1% / -20.6% |
| 8 | 15,290.4 / 278.6 | 15,335.5 / 285.8 | +0.3% / +2.6% | -5.2% / -12.0% | -4.9% / -9.8% |
| 16 | 27,946.0 / 203.4 | 28,141.6 / 220.1 | +0.7% / +8.2% | -2.9% / -10.2% | -2.2% / -2.9% |
| 32 | 56,039.2 / 120.6 | 57,890.6 / 138.5 | +3.3% / +14.8% | -7.2% / -25.6% | -4.2% / -14.5% |
| 64 | 81,735.1 / 60.1 | 88,391.8 / 66.5 | +8.1% / +10.6% | -23.6% / -34.1% | -17.4% / -26.9% |

All 12 points 0 errors, 0 faults, cache hit 0.964-0.981. c32/c64 run PDI 4 (TP2-best); TP4 c64 rep at PDI16 earlier gave
72,693.2 / 99.4 (P90 +9% vs vLLM), so the PDI choice trades TTT vs P90 at high conc.

## SGLang TP2 rep sweep tp2r_* (2026-10-03 mi355-4, one lane GPUs 0,1; main 58f0d250ec + #42055, TP2-best per-point settings)
rep = prefill graph disabled + --enable-decoder-swa-bounded-replay. tok/s/GPU / P90.
| conc | tp2r (rep) | baseline (bcg) | rep vs baseline | vs vLLM B200 TP2 |
|---|---|---|---|---|
| 1 | 11,697.4 / 357.7 | night0930 11,375.4 / 346.4 | +2.8% / +3.3% | -10.9% / -9.0% |
| 2 | 12,022.3 / 330.7 | env1001 11,535.8 / 322.2 | +4.2% / +2.6% | -10.6% / -11.4% |
| 8 | 29,760.8 / 237.2 | env1001 29,407.5 / 218.1 | +1.2% / +8.8% | -4.0% / -6.1% |
| 16 | 54,345.3 / 165.0 | night0930 54,080.7 / 134.5 | +0.5% / +22.7% | -2.6% / -5.7% |
| 32 | 105,800.1 / 93.7 | env1001 98,143.5 / 74.8 | +7.8% / +25.3% | +13.3% / +39.3% |
| 64 | 142,243.2 / 52.0 | env1001 119,381.4 / 41.0 | +19.2% / +26.9% | vLLM TP2 c64 collapsed (20,502.5 / 28.1) |

All 6 points 0 errors, 0 faults, cache 0.962-0.980. rep beats every previous TP2 number at c8-c64 (c64 +18.6% / +27.5%
vs s6042, c32 +7.3% / +31.0% vs s6042). NEW TP2 BEST config: main + #42055, TP2 EP1 host table, TP2-best per-point
PDI/chunk/mem, --enable-decoder-swa-bounded-replay --cuda-graph-backend-prefill disabled.
