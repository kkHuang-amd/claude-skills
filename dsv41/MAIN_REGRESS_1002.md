# DSV4.1-Flash regression check: upstream main + PR #42055 (2026-10-02)

Owner node: mi355-4 (PIDs / GPU ids / ports below refer to mi355-4).

## CONTINUE HERE

**TP2 rep SWEEP DONE 2026-10-03 19:56 +08 (6/6 OK; results/agentx.md "SGLang TP2 rep sweep"; new TP2 best). Was: started 2026-10-03 13:01 +08, one lane GPUs 0,1:8888, lane PGID 155005):** tp2r_c{1,2,8,16,32,64},
TP2 EP1 host table, TP2-best per-point PDI/chunk/mem, rep = prefill graph disabled + bounded replay. First server ready
13:05 +08 (flags verified). 6 x ~75 min -> done ~20:30 +08. Baseline for "rep vs bcg" at TP2 = env1001_c{2,8,32,64}
(same aiter, dsv41-env code ~ main) and night0930 for c1/c16 (different code) -- no full-length TP2 bcg run on main+#42055.

**TP4 SWEEP DONE 2026-10-03 12:14 +08 (12/12 OK, table in results/agentx.md "SGLang TP4 sweep tp4s_*"; rep >= bcg everywhere). Was: started 2026-10-02 22:02 +08, ONE lane per user -- two lanes not proven equal to one):** GPUs 0-3:8888,
lane PGID 84451 (`kill -- -84451`, then sglang:: children). TP4 EP1, engram host table, main 58f0d250ec + #42055, TP2-best
per-point PDI/chunk/mem (c1,c2,c8 16/16384/0.70; c16 16/16384/0.80; c32 4/16384/0.80; c64 4/4096/0.85). Pairs per point,
low conc first: tp4s_c{1,2,8,16,32,64}_{bcg,rep}; bcg = breakable prefill graph, no replay; rep = --cuda-graph-backend-prefill
disabled + --enable-decoder-swa-bounded-replay (agentx_lane.sh 6th POINTS field selects EXTRA_ARGS_<var>). 12 runs x ~80 min
-> done ~2026-10-03 14:00 +08. Progress lane_8888.txt. First point verified (flags + host table) 22:08 +08.

**TP4 sweep (started 2026-10-02 12:57 +08, ONE lane per user, GPUs 0,1,2,3 port 8888):** recipe defaults PDI16 / chunk 4096 /
mem 0.70 / engram resident (c<128), same SRC + EXTRA_ARGS + OPUS sparse prefill as TP2. No c128 (user).
Wrapper PID 3355: tp4m_c64 -> tp4m_c16 -> tp4m_c4 -> tp4m_c1; then PID 3650 (waits for 3355): tp4m_c32 -> tp4m_c8 -> tp4m_c2.
~70 min per point, ~8 h total. Progress lane_8888.txt. Compare with vLLM B200 TP4 (results/agentx.md, InferenceX run 36423355395).
Kill: `kill 3650; kill -- -3355`, then sglang:: children.
**TP4 sweep STOPPED 2026-10-02 14:17 +08:** tp4m_c64 (EP1, engram GPU-resident, PDI16) had ~0 prefix-cache hits (24,543 prefill
batches, 56 hits; TP2 c64 10,204 / 2,495), KV usage <= 0.10, queue ~56, still in warmup at 77 min -> invalid.
Recipes (InferenceX origin/main 1f60c15b5, inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/):
B200 SGLang TP4 = EP4, PDI16, engram HOST table per_rank, mem 0.8, chunk 4096; MI355X SGLang TP4 = EP4, engram resident,
mem 0.7, max-total-tokens 3145728, max-running 32, prefill graph disabled, no PDI; B200 vLLM = engram cpu_offload (UVA), no PDI knob.
A/B 1 (single variable): tp4dbg_hosttbl_c64 = tp4m_c64 + ENGRAM_HOST_TABLE=1 (new knob in agentx_colleague_mi355x_sglang.sh),
PID 16569, GPUs 0-3:8888, started 14:20 +08. 14:25 per user: script default is now host table per_rank at every TP
(ENGRAM_HOST_TABLE=0 for resident) -- confirm with this A/B. That edit KILLED PID 16569 (bash reads a running script
lazily: "line 257: y_cmd: command not found"; dir renamed *.broken). NEVER edit a script while a run is executing it.
14:40 relaunch OOMed at model load: the "cancelled" part2 lane was still running (`kill $!` hit the setsid wrapper 3650, the
real lane was PID/PGID 3653 -- always find the PGID with ps after setsid) and held GPUs 0-3 (tp4m_c32 rc=127, tp4m_c8 rc=1,
tp4m_c2 running). Stray lane killed 15:21 +08. Relaunched 15:21 +08, PGID 35387 (kill -- -35387).
RESULT 15:48 +08: host table FIXES the cache: 10-min buckets 2 / 553 / 440 hits (142.6M cached tokens in 07:30 bucket, like
TP2), warmup 662/707 at 12 min (resident: 225/707 at 20 min), running 25 / queue 36. Profiling from ~07:37 UTC; live at 10:59:
92,508 tok/s/GPU, intvty p50 178, TTFT p50 10.1 s, 0 err. Root cause of the resident-engram cache loss not investigated.
FINAL (16:4x +08): 65,201.5 tok/s/GPU / P90 91.4 (p50 129.6, TTFT p50/p90 12.2/30.2 s, cache hit 0.963, 0 err);
vs vLLM B200 TP4 c64 107,014.2 / 91.1 = -39.1% / +0.3%. Live tok/s fell from 92.5k (min 11) -> TTFT-bound; candidate PDI4
(TP2 c64 needed PDI4 for the same reason).
A/B 2 (17:57 +08, PGID 54497, GPUs 0-3:8888): tp4_bsr_c64 = above + EXTRA_ARGS --enable-decoder-swa-bounded-replay
--cuda-graph-backend-prefill disabled (pair required by deepseek_v4_hook.py; HIP radix backend implements enter_late_layer_tail;
B200 high-conc option). Server ready 18:01 +08. RESULT 72,693.2 / 99.4 (+11.5% / +8.8% vs 65,201.5 / 91.4), TTFT p50 10.1 s, 0 err.
A/B 3 (20:17 +08, PGID 71224, GPUs 0-3:8888): tp4_bsr_pdi4_c64 = A/B 2 with PDI 4 (only change). Ready 20:22 +08. RESULT 87,850.1 / 65.7 (TTFT p50 1.62 s; +20.9% / -33.9% vs PDI16). GPUs free. look at cache hits per 10-min bucket after ~20 min, then kill. Next A/B if still 0: EP4.
**Status (TP2):** GSM8K PASS; AgentX c8/c64 no regression on live stats (Results). Both lanes were killed externally at
2026-10-02 03:23:36 UTC (agent shell backend restart; no error in logs), so no final JSON / P90, c2/c32 never ran.
Compare cut runs with `scripts/agentx_live_compare.py MM:SS <a.nohup> <b.nohup>`. Rerun a point only if final P90 is needed.
Was: AgentX started 2026-10-02 10:25 +08, two lanes like env1001:
lane A GPUs 4,5:8888 wrapper PID 23691: main42055_c64 (PDI4, 4096, 0.85) -> main42055_c2 (PDI16, 16384, 0.70);
lane B GPUs 6,7:8889 wrapper PID 23692: main42055_c8 (PDI16, 16384, 0.70) -> main42055_c32 (PDI4, 16384, 0.80).
Progress lane_{8888,8889}.txt in /shared_nfs/kk/dsv41/agentx; kill: `kill -- -<pid>` then the sglang:: children.
**Next:** when a point ends, compare with env1001_c* (ENV_1001.md Results) and fill Results below.
**GSM8K method trap:** EVAL_ONLY=true without SERVER_ONLY runs InferenceX lm_eval (chat template) -> 0.974, NOT comparable with
the 0.89-0.91 history, which is `sglang.test.few_shot_gsm8k` (scripts/run_gsm8k.sh) on a SERVER_ONLY=1 EVAL_ONLY=true server.
**Files:** sglang `/sgl-workspace/sglang` main 58f0d250ec + PR #42055 (3 commits, `cherry-pick -n`, staged, uncommitted;
local ref pr-42055); aiter `/sgl-workspace/aiter` e7d2453f2 + #5967 + uncommitted pa_decode_sparse.py (staged) and
csrc/cpp_itfs/torch_utils.py (torch.Stream) edits; flydsl 0.3.4.1 global. /sgl-workspace/sglang-dsv41-env and the
ATOM-port worktrees do not exist in this container. Node check hostreg_devptr_check.py: same=True (engram devptr patch not needed).
**Repro (GSM8K; drop EVAL_ONLY for AgentX, set CONC/PDI/chunk/mem per point):**
```bash
cd /shared_nfs/kk/dsv41/agentx && PYTHONPATH=/sgl-workspace/mori SRC=/sgl-workspace/sglang/python \
  SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" \
  EVAL_ONLY=true OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 PORT=8888 CONC=32 PREFILL_DECODE_INTERVAL=4 CHUNKED_PREFILL_SIZE=16384 \
  MEM_FRACTION_STATIC=0.80 TAG=main42055_gsm8k setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > main42055_gsm8k.nohup 2>&1 < /dev/null &
```
**Pass:** GSM8K 0.895-0.908 (env1001 0.894 / 0.906); AgentX within noise of env1001 (TTT ~1%, P90 ~5%).

## Results

| date (node) | what | numbers | dir |
|---|---|---|---|
| 2026-10-02 (mi355-4) | GSM8K 1319 5-shot few_shot_gsm8k, TP2 EP1, real acceptance, CONC=32 eval server | 0.901 / 0.897, 40.0 / 38.5 s, accept len 3.39; env1001 0.894 / 0.906, 40.0 / 39.1 s, 3.40 -> PASS | gsm8k_main42055_r{1,2}.log, agentx/main42055_gsm8ksrv |
| 2026-10-02 (mi355-4) | AgentX main42055_c8 (PDI16, 16384, 0.70, GPUs 6,7), cut at 48:12, live stats vs env1001_c8 at 48:00 | done 1116 vs 1116, 0 err; tok/s/GPU 28,506 vs 28,614 (-0.4%); intvty p50 304 vs 301; TTFT p50/p95 360/1414 vs 368/1520 ms -> no regression | agentx/main42055_c8.nohup |
| 2026-10-02 (mi355-4) | AgentX main42055_c64 (PDI4, 4096, 0.85, GPUs 4,5), cut at 33:00, live stats vs env1001_c64 at 32:55 | done 3361 vs 3349, 0 err; tok/s/GPU 108,850 vs 108,584 (+0.2%); intvty p50 53 vs 52; TTFT p50/p95 2378/27106 vs 2481/29673 ms -> no regression | agentx/main42055_c64.nohup |
| 2026-10-02 (mi355-4) | GSM8K InferenceX lm_eval (chat template, EVAL_ONLY run) | 0.9735 flexible / 0.9742 strict (different method, reference only) | agentx/main42055_gsm8k.nohup |
