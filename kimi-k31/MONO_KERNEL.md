# Kimi-K3.1 decode / verify mono kernel — design (OPT_TRACKER row 13)

Owner node: mi355-4. Created 2026-10-31. P0 + P1a/P1b step 1 done; history in sections 6, 8, 9.

## CONTINUE HERE (handoff 2026-11-01 13:00, for the next session)

**Status:** S1 (c1, <= 8 rows) runs on K3.1 in the real A/B config (DCP8 + DSPARK, ReplaySSM ring):
c1 TPOT 4.96 -> 4.49 ms (-9.5%), P90 ITL 6.24 -> 5.67. No server running on mi355-4; GPUs free
(2 external ~1.2 GB processes on one GPU are not ours).
**Code:** `/sgl-workspace/sgl-amd-mono` branch `k31/mono-s1` @ 7a76c4e, pushed to the fork
kkHuang-amd/sgl-amd-mono (remote `fork`; `origin` = sgl-project). Base = PR #13 head 88aee67 (open,
billishyahao). Commits: d7fb03e `kimi_k31` spec (DCP8, dsa backend, drift digests on dev_amd) | 5fb46b4 |
baa73de K1 on the ReplaySSM ring (`KdaPreBuild.ring`) + `test/kernels/test_kimi_k31_k1_ring.py` | 8888d88
`test/kernels/kimi_k3/time_k2.py` (K2 time + stage timeline at TP8) | 6e50a20 K2 route by wave-max rounds +
latent quantized once. The ring and route commits change `kimi_k3` code: coordinate with #13 before upstream.
9cba298 (2026-11-01, pushed): `sgl_amd_mono/kimi_k31/gfx950/stages/tile.py` (tile GEMM v0) +
`test/kernels/kimi_k31/tile_bench.py`; 06263ca: `stages/tile_mma.py` (LDS tile GEMM v1) + `mma_bench.py`; 7a76c4e: split-K merge modes (linbf).
SGLang: `/sgl-workspace/sglang-k31-amd` (editable install), dev_amd-based.

**Done 2026-11-01 (mi355-4):** S1 accuracy gate PASSES -- gsm8k1319 greedy THREADS=1 real draft: mono on
1287/1319 (0.9757) vs off 1288 (0.9765), discordant 2 / 3 (McNemar p = 1.0), 1246/1319 identical
generations, accept len 4.71 / 4.70 (section 10.1). P2a step 1 started: register-operand tile GEMM v0 is
correct but 1.6x (m=16) to 4x (m=128) slower than tgemm; the structure is the cause (section 10.2).
Section 10.3's "+25% at 8 waves" was a sweep artifact (HTI default); corrected in 10.4.

**P2a step 1 done:** tile GEMM stage for m = 16..384, own FlyDSL (`stages/tile_mma.py`; no aiter gemm), split-K
merged by lane-contiguous packed-bf16 atomics after an LDS shuffle ("linbf"); ties or beats tgemm on in_proj /
MoE front / latent up at every bucket, o_proj +20-25% at m=128 / 384 (bucket table: section 10.6).

**P2a step 2 done (2026-11-01 13:00):** column-slice collectives `stages/coll.py` (RS / AG / two-shot AR, tagged
and waitcnt-flag formats), bit-exact at TP8, 0 mismatches in 8 x 10^5 stress launches; tagged ties or beats flag at
every size, so tagged is the default (section 10.7). Commit 6162564 on `k31/mono-s1`, pushed.

**Decisions (user, 2026-11-01):** (a) the consumer zeroes the atomic output it reads (one reader an element;
buffers zeroed once at allocation); (b) linbf by default, an env knob (named when wired) switches the tile GEMMs to
"part" for debugging / accuracy gates; tile-GEMM stages are exempt from "graph replays bit for bit".

**Next, in order:**
1. Wire the collectives to the tile GEMMs (section 10.7 "Wiring"): o_proj / shared-down epilogue -> RS push per
   finished tile (a per-tile K-slice arrival count: with linbf the tile is complete only after its last K slice;
   that task reads, pushes and zeroes it = decision (a)); in_proj consumes AG x by K-chunk polls. Decide the
   parity scheme for consecutive launches of a step (section 10.7 (4)).
2. P2a (section 7): the planner (stage graph -> tasks -> CTAs by XCD) -> the launch. Then P2b/P2c (DSA front / back).
3. c1 leftovers (section 8/9): K1 in_proj ~2.4 TB/s vs hgemm 5.1 (~1-1.4 ms a step); router replicated per
   rank (drop the logits all-gather, ~5 us a layer); top-16 still 3.9 us.
4. Side: the standard path is nondeterministic at bs 1 (108/200 identical greedy generations between two
   passes on one server; mono 66/200); find the source (suspect small-M split-K atomics) before trusting
   per-item comparisons.

**Repro:**
```bash
source /workspace/claude-skills/kimi-k31/env.sh
S=/workspace/claude-skills/kimi-k31/scripts
bash $S/launch_server.sh <tag> --enable-amd-monokernel          # mono on; omit the flag for off
AB_CONC="1" bash $S/ab_bench.sh <tag>                            # c1 TPOT -> results/ab/<tag>/summary.txt
SIM=0 bash $S/launch_server.sh <tag> --enable-amd-monokernel; NUM=1319 THREADS=1 TEMP=0 bash $S/gsm8k_gate.sh <tag> <log>
cd /tmp && python /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/tile_bench.py --time --m 16,128 --nb 1,2 --ks 1,4
cd /tmp && python /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/mma_bench.py --time --m 128 --cfg 128:128:2:4:4:2 --mode part,linbf
cd /tmp && torchrun --nproc-per-node 8 /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/coll_bench.py --time   # collectives
python $S/tile_isa.py 128:6288:7168:1 /tmp/tile_isa      # vgpr / scratch / lds + disassembly, no launch
python $S/hgemm_8wave_sweep.py 128                        # aiter hgemm at 8 waves vs tgemm
cd /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k3 && torchrun --nproc-per-node 8 check_k2.py   # + check_moe.py, time_k2.py
python /sgl-workspace/sgl-amd-mono/test/kernels/test_kimi_k31_k1_ring.py                      # 1 GPU
```
**Pass criteria:** endpoint TPOT moves (c1 for S1; c4/c16/c48 for S2), gsm8k on vs off within noise at
THREADS=1, check_k2 / check_moe within NUMERICS bounds and graph replays bit for bit.

**Decisions (user, 2026-10-31):**
1. S1 then S2, but S1 must not lock in PR #13's structure: reuse what is good, redesign what limits S2
   (section 2b). S1 is built on the S2 architecture, with <= 8-row stage variants.
2. New package `sgl_amd_mono/kimi_k31/`.
3. Design docs committed and pushed (skills repo).
4. No K3 c1 reproduction; P0 starts directly on K3.1.
**Code home:** `/sgl-workspace/sgl-amd-mono` (github sgl-project/sgl-amd-mono, detached at 90c2355 = main).
Reference: PR [#13](https://github.com/sgl-project/sgl-amd-mono/pull/13) `billishyahao/kimi_k3`
(Kimi-K3 layer monokernel, open, 7.3k lines, last push 2026-10-29). SGLang side:
`/sgl-workspace/sglang-k31-amd` `origin/dev_amd` 3d2aa95c9d (has `kernel_plugins/amd`,
`--enable-amd-monokernel`, and `forward_pre_o_proj` on both the KDA and the MLA attention).
**Repro of the numbers below:**
```bash
python3 scripts/trace_layer_kinds.py /shared_nfs/kimi-k31/traces/c16_isl8k_osl1k_noradix_20261030/*TP-0.trace.json.gz VERIFY 20
```
**Pass criteria for any phase:** endpoint TPOT moves (ab_bench c4/c16/c48, plus c1 for S1), gsm8k200 greedy
within noise of 0.985 with the real draft, accept length not collapsing, fault injection stops every rank.

## 0. What K3.1 is, and what PR #13 already covers

K3.1 has the same dims as K3 (checked both `config.json`): 93 layers (1-based), hidden 7168, 69 KDA layers
and 24 full-attention layers (4, 8, ..., 92, 93), layer 1 dense FFN, 92 latent-MoE layers
(896 experts, top-16, latent 3584, expert inter 3072, 2 shared experts), attention-residual bank
(`attn_res_block_size` 12). The only architectural difference: K3.1's full-attention layers are DSA
(MLA q_lora 1536 / kv_lora 512 / rope 64, 96 heads, plus an indexer: 32 heads x 128, top-2048 every layer).
`KimiK31ForConditionalGeneration` subclasses the K3 model in sglang-k31-amd.

PR #13 (K3, TP8, DCP1, aiter attention backend, DSpark block 7, 1..8 rows a step):
- K1 per KDA layer: AttnRes seam -> input_layernorm -> in_proj -> f_b -> conv -> recurrence -> gated norm.
- K2 per MoE layer (all 92): o_proj + in-kernel AR -> MLP AttnRes seam -> router top-16 -> latent down +
  shared gate_up -> MXFP4 experts -> shared down + latent up -> in-kernel MoE AR. Router and latent
  down/up are computed 1/8 per rank and all-gathered (the standard path replicates them).
- Full-attention layers keep SGLang's attention up to `forward_pre_o_proj`.
- Measured on K3 c1: 41.63 -> 55.26 decode steps/s (step 24.0 -> 18.1 ms, -5.9 ms); AgentX c1 TPOT -21%.

What PR #13 does NOT give K3.1, and the reason this design exists:
1. Width. All stages are GEMV-style for S <= 8 rows (c1). Our A/B points are c4/c16/c48 = 32/128/384 verify
   rows. Its one-shot AR (every rank pushes its full partial to every peer) costs 7x the data per rank:
   fine at 8 rows (0.8 MB out), ~13 MB out at 128 rows.
2. DSA layers: indexer, KV write, sparse MLA, DCP8 combine stay outside (24 layers, 17.1 ms/step at c16).
3. Config gate: `model_type kimi_k3`, `attention_backends {aiter}`; K3.1 A/B config runs `attention_backend=triton`,
   DCP8, fp32 ReplaySSM ring (`--enable-linear-replayssm-spec`, PR #7). PR #13's best run used bf16 SSM
   state in one arm; K1 reads its own conv buffer (SGLang's spec conv window is one entry short).
4. Tested against SGLang 8e7601ae4a, not our dev_amd.

## 1. Op list per layer from the c16 trace (VERIFY, bs 16 x 8 = 128 rows, TP8 + DCP8)

Trace `c16_isl8k_osl1k_noradix_20261030` TP0, 20 VERIFY steps averaged; kernel-sum 47.93 ms/step
(GPU fully busy, single stream). Segments end at each MoE stage 2, so a segment = the previous layer's
MoE AR tail + this layer's attention + this layer's MoE. GEMM roles come from the M=128 shape log
(`logs/server_i3_gemm_shapes_20261031.log`, N x K per rank) and kernel order; "?" = inferred.

| segment kind | segs/step | kernels/seg | us/seg | ms/step |
|---|---:|---:|---:|---:|
| KDA layer | 67 | 22.2 | 446.0 | 29.88 |
| DSA layer | 24 | 43.5 | 712.2 | 17.09 |
| head (layer 1 dense + layer 2 KDA, embedding) | 1 | 37 | 672.7 | 0.67 |
| tail (final AR/agg/norm, lm_head, logits AG, softcap) | 1 | 10 | 286.3 | 0.29 |

KDA layer, in order (us, avg):

```text
MoE tail of L-1 : AR 2stage [latent|shared] ~20 | add_rmsnorm_quant 4.3 (latent norm) | latent up 7168x3584 20.2
                  | add3 4.4 | _agg (attn-side AttnRes) ~8.5
attention       : in_proj 6288x7168 29.8 | f_b 1536x128 4.3 | conv1d_update 5.9 | KDA recurrence 26.3
                  (Triton; 16.1 with PR #7) | gated norm 4.0 | o_proj 7168x1536 10.3 | AR ~20 | _agg (MLP-side) ~8.5
MoE front       : fused [shared gate|up 1536 | latent down 3584 | router 896] x7168 30.2 | situ_and_mul 4.1
                  | shared down 7168x768 6.6 | topk_reg 5.1 | moe_sorting x2 10.5 | fused_mx_quant_moe_sort 12.5
experts         : stage1 135.8 | stage2 73.9
by class        : experts 209.7 | dense GEMM 101.4 | AR 39.9 | KDA recurrence 26.3 | small ops 67.9
```

DSA layer, in order (MoE tail and MoE front/experts as in the KDA layer):

```text
attention front : q_a|kv_a 2112x7168 13.3 | fused_qk_rmsnorm 4.3 | q_b 18432x1536 19.6 (hipBLASLt; all 96 heads
                  on every rank under DCP8) | index q 4096x1536 8.2 | index k 128x7168 5.6 | layer_norm 4.0
                  | act_quant 5.3 | indexer_k_quant_and_cache 4.0 | index weights 32x7168 5.4 | head_gate_q_scale 4.2
                  | (layer 4 only: 12 x elementwise ~3.8 = per-forward memo, ~0.26 ms/step)
indexer + DCP   : paged_mqa_logits 7.6 | dcp_local_candidates 4.3 | allgather_vec 45.5 | dcp_global_slots 21.6
KV + q path     : W_UK bmm? 14.0 | quantize_k_cache 4.3 | set_mla_kv_buffer 4.2 | localize_compact 4.5 | mxfp8_quant 5.5
attention       : opus_mla_decode_mxfp8 51.1
DCP combine     : dcp_pack_a2a_send 7.9 | NCCL a2a 57.8 | dcp_lse_combine 5.2
attention back  : W_UV bmm? 11.9 | output-gate GEMM 1536x7168? 12.1 | mla_output_gate 3.9 | o_proj 7168x1536 10.5 | AR ~20
by class        : experts 209.4 | dense GEMM 158.6 | comm 155.3 (AR 38.9, DCP 116.4) | sparse MLA 51.1
                  | indexer/topk 33.5 | small ops 104.3
```

Where the 47.9 ms go, and which part a mono kernel can touch at c16:

```text
experts (stage1+2)   19.3 ms  HBM roofline (~8 TB/s; ~807 of 896 experts touched at 2048 pairs) -> not a target
dense GEMM           10.9     ~3-4.3 TB/s, floor 4.8 ms; row 3: in-kernel tiles tie tgemm at best
                              -> gain only by NOT doing replicated work (router, latent down/up)
comm                  6.3     AR 187 x ~20 us (~185 GB/s busbw) + DCP 116 us x 24; already link-bound
                              -> gain by overlap and by sending fewer bytes, not by "faster comm"
small ops + prologue  ~7.2    67.9 x 69 + 104.3 x 24 (+ head/tail); each ~4-13 us, real work ~1 us
                              -> the main mono target
KDA / sparse MLA / indexer ~4.2  compute kernels; keep (PR #7 HIP KDA), fold only their glue
```

Caveat: only c16 is traced. Small-op cost is roughly width-independent, so at c4 (step ~30 ms) the same
absolute saving is a larger fraction; MoE at c4 touches fewer experts. P0 captures c1 and c4.

## 2. Scope options and expected ms/step (c16 VERIFY 47.9 ms unless stated)

Common savings mechanisms (per layer, c16):
- A. Small-op fusion: each fused hand-off still costs a grid-wide data dependency (~2-4 us through the
  mailbox) but not a launch + ramp + tail. KDA 67.9 -> ~20 us, DSA 104.3 -> ~35 us.
  = 3.3 + 1.7 = ~5.0 ms (range 3.5-5.0).
- B. De-replication (PR #13 does it): router + latent down computed 1/8 per rank (6016-row GEMM -> 2096 rows,
  30 -> ~11 us, + 0.9 MB latent / 0.5 MB logits all-gather ~5-7 us), latent up for 1/8 of the output
  columns (20 -> ~4 us) with a column-sharded residual. Net ~20 us x 92 = ~1.8 ms (1.5-2.5).
  Overlaps PR #8's row-sharded tail (standard path, prefill threshold 4096): measure that first (P0.4).
- C. Comm: two-shot AR (RS + AG) pushed tile by tile from the producing GEMM epilogue, seam/norm on the
  owner's 1/8 columns, ~8-12 us of each ~20 us AR hidden -> 1.5-2.2 ms. DCP: apply W_UV (512 -> 128 per head)
  to the partial outputs BEFORE the a2a (linear, so LSE rescale commutes) -> a2a bytes / 4 (11 -> 2.75 MB),
  pack/combine fused: ~45 us x 24 = ~1.1 ms. Candidate all-gather 45.5 us: size unknown, 0-0.6 ms.
  C = 2.6-3.9 ms.
- D. GEMM tiles inside the launch: 0 assumed (row 3), +0..1 ms if the next stage's weights stream
  while a stage waits (PR #13's "issue weights before the first wait").
- E. Experts: 0 at c16.

| option | launches/step | covers | c1 est. (VERIFY) | c16 est. | effort / risk |
|---|---:|---|---|---|---|
| S0 = today | 2581 | - | VERIFY 18.9 | 47.9 (c4: 26.1) | - |
| S1 <= 8 rows on the 2b architecture | c1: ~2150 -> ~900 | KDA layers whole + o_proj .. MoE of the DSA layers; DSA attention + DCP on SGLang | **~-5 ms of 18.9 (-27%)** | 0 (falls back) | medium: planner, gate, DCP8, triton backend, fp32 ring |
| **S2 per-layer, wide rows, experts kept as aiter launches** | ~69 x 3 + 24 x 5 = ~330 | everything except experts stage1/2, sparse MLA | ~-7 ms (-37%) | **-8 to -9 ms (-17..-19%)**; c4 -5 to -8.6 ms (-20..-33%) | high: MFMA-tile stages, waitcnt-flag comm, DCP in kernel |
| S2+ experts inside the layer launch | ~69 + 24 x 3 | S2 + stage1/2 as stages | - | S2 + 0.5-0.9 ms (overlap expert weight issue with prologue, -2 boundaries) | very high: grouped MXFP4 GEMM at roofline inside a multi-role kernel |
| S3 one persistent kernel per step | 1-3 | all 93 layers + tail | - | S2+ + ~0.7-1.0 ms (remaining boundaries x 2-3 us) | highest: one register/LDS budget for every stage, compile time, debuggability |

S2 launch shape (one "layer launch" = from the MoE output of layer L-1 to the expert inputs of layer L):

```text
KDA layer L : [K_L: MoE-tail(L-1) RS/AG + latent up + residual + AttnRes seam + input norm
               + in_proj + f_b + conv + recurrence + gated norm + o_proj + AR + MLP seam
               + router/latent-down/shared gate_up (1/8, AG) + SiTU + shared down + top-16 + sort + mxfp8 quant]
              -> aiter stage1 -> aiter stage2
DSA layer L : [K_L,front: MoE-tail(L-1) ... + q_a/kv_a + norms + indexer q/k/w + logits + local top-k
               + in-kernel candidate AG + global slots + KV write + q_b + W_UK]
              -> opus sparse MLA
              -> [K_L,back: W_UV on partials + in-kernel a2a + LSE combine + gate + o_proj + AR + MLP seam + MoE front]
              -> aiter stage1 -> aiter stage2
```

Recommendation: S1 first (small, gives c1 and validates the PR #13 base on K3.1), then S2 in sub-phases
(section 5). S2+/S3 only if S2 lands and the remaining boundaries measure > 1 ms/step.

## 2b. PR #13: what kimi_k31 reuses and what it redesigns

Rule: S1 and S2 share one architecture. S1 differs from S2 only in which stage variants a width bucket
picks (GEMV <= 8 rows, MFMA tiles >= 16 rows) and in whether the experts run inside the launch.

Reuse (as is, or by import from `common` / `kimi_k3` once #13 merges):
- `common/` runtime: tagged mailbox, guarded waits + fail-stop, uncached peer memory, epoch, stamps.
- Host-bumped step epoch captured in the graph (`tag.py`): no buffer zeroing between steps.
- Role rule: CTAs that poll carry no weight loads in flight; weight CTAs issue the next stage's weights
  before their first wait (MoE weights stream while the AttnRes chain settles).
- De-replication of router / latent down / latent up (1/8 per rank + all-gather).
- In-place weights (bf16 checkpoint layout, MXFP4 experts in `Mxfp4MoEMethod`'s layout): no KV budget cost.
- Correctness lessons: K1 owns its conv window (SGLang's spec window is one entry short); compile by running
  on real weights (FlyDSL compile-only folds zero args into an unselectable pattern); kernel checks against
  SGLang ops, bitwise graph replay, fault injection, `SGLANG_READS` contract.
- The <= 8-row GEMV stages (seam, KDA K1, MoE GEMV) as S1's narrow variants, after a review on K3.1 data.

Redesign (things that would cap S2 or the later path):
| PR #13 | limit | kimi_k31 |
|---|---|---|
| CTA roles as fixed index ranges (0..55 seam, 56..159 o_proj, 160..243 router/latent, ...), `NA = 56`, `ROWS = 16` | one layout for one width; not XCD-aware; every new stage re-cuts the ranges by hand | a build-time planner: stage graph -> tasks -> CTAs per width bucket, grouped by XCD (b % 8), roles derived, not written |
| launch cut at the o_proj input (K1 = KDA front, K2 = o_proj .. MoE output) | the MoE tail and the next layer's front are split by a launch; the full-attention layers can't join | cut at the expert inputs: one launch = MoE tail(L-1) + attention(L) + MoE front(L); experts inside only for the GEMV buckets |
| one-shot AR (full partial to every rank) | 7x bytes per rank; ~13 MB out at 128 rows | width-switched: one-shot <= 8 rows, two-shot RS + AG above, epilogue-pushed tiles |
| residual / AttnRes seam replicated on every rank | 8x the bank reads at wide rows | column-sharded residual and bank (each rank 896 columns), scalars exchanged for the full-row reductions |
| builds per width 1..8 (`MAX_ROWS`) | 48 builds at bs 1..48 x 8 rows | width buckets with masked rows (8, 16, 32, 64, 128, 256, 384) |
| full-attention layers on SGLang up to `forward_pre_o_proj` | 24 DSA layers x 712 us outside | DSA front / back launches (P2b / P2c) on the same planner |
| `model_type kimi_k3`, aiter attention backend, DCP1, bf16 SSM state run | does not start on the K3.1 A/B config | `kimi_k31` spec: triton backend, DCP8, fp32 ReplaySSM ring (PR #7) |
| `MAX_BLOCKS` AttnRes bank bound | fixed at build | derived from config, checked in the spec |

Open on #13: its body says two of the 1..8 widths "abort today" (the section it points to is not in the
body); find which before reusing the GEMV stages.

## 3. Synchronization across 8 XCDs and TP8 comm inside the kernel

Hardware facts (MI355X, from row 3 measurements and the sgl-amd-mono runtime): 8 XCDs x 32 CUs; L2 per XCD,
not coherent across XCDs; workgroup b lands on XCD b % 8 (verified via HW_REG_XCC_ID); a last-CTA fixup
with `__threadfence` cost ~50 us; device atomics are slow (3M fp32 atomics ~8-10 us) and same-XCC scope does
not help; owner stores must be sc0|sc1 (write-through) or another XCD's L2 writeback overwrites them.

3.1 Intra-GPU (reuse `sgl_amd_mono/common`):
- Resident grid, one workgroup per CU (256), role by CTA index (`common/plan.py`), never a grid barrier.
- Dataflow hand-offs through the tagged mailbox (`common/sync.py`): payload words carry the step epoch tag
  and live in uncached scratch (`hipExtMallocWithFlags(..., uncached)`); a consumer polls the data itself,
  so there is no flag + fence pair. Epoch bumped once per forward by the host and captured in the graph
  (PR #13 `tag.py`), so no buffer is zeroed between steps.
- XCD-aware placement (new vs PR #13): split each stage's tiles into 8 XCD groups; a hand-off whose
  producers and consumer share an XCD (e.g. GEMM N-tiles -> the per-column seam of the same columns) uses
  cached L2 stores + L1-bypass loads; only all-to-all hand-offs (full-row RMS / score partials, K-split
  reductions, router logits) go through uncached memory, and those carry scalars or small rows.
- No split-K across CTAs in the dense GEMMs at these widths (row 3: the split-K finish, not the weight
  stream, bounds the design); pick N-tile counts that fill 256 CUs, and reduce K inside a CTA.
- Polling CTAs carry no weight loads in flight (a poll retires behind earlier loads; PR #13 role split);
  weight-streaming CTAs issue the next stage's weights before their first wait.
- Every wait has a deadline (`common/guard.py`, half the watchdog, <= 60 s); a failed wait stops every rank.

3.2 TP8 comm (peer memory, `common/peer.py`, system-scope stores over xGMI, local polling):
- AR replaced by two-shot: the GEMM epilogue writes column slice j of its bf16 partial straight into rank j's
  mailbox (RS); rank j sums the 8 partials in rank order (bit-match the custom AR's order, fp32 accumulate),
  applies residual / AttnRes / norm on its 896 columns, then pushes the result slice to every rank (AG).
  Full-row reductions (RMS, AttnRes score dots) exchange per-row partial scalars only.
  Bytes per rank at 128 rows: RS 1.6 MB out + AG 1.6 MB out vs one-shot 12.6 MB.
- Payload format by size (P0.2, section 6): tagged 64-bit words (bf16 pair + tag) up to ~0.9 MB (<= 32 rows
  x 7168), raw 128-bit stores + `s_waitcnt vmcnt(0)` + one flag per (block, source) above; never
  `__threadfence_system` (~50 us floor). One-shot vs two-shot by width as before.
- MoE tail: AR of the latent (0.9 MB) + RS of the shared partial, latent up on the owner's columns, AG of
  the next layer's input row (needed in full by in_proj / q_a, K = 7168).
- DCP8: (1) indexer candidates: each rank writes its local top candidates to every rank's mailbox, then the
  global-slot merge runs locally; (2) partial attention outputs: W_UV applied locally for all 96 heads
  (+11 MB weight read ~2-3 us), then [rows, head group, 128] + LSE sent to the head owner, combined there.
- Precision floor: bf16 payloads, no quantized collectives (policy 2026-10-30); fp8 AG only as #9/#10
  experiments.
- xGMI traffic competes with HBM-bound stages at the receiving end; overlap gains must be measured (P0.3),
  not assumed.

## 4. Integration

- CUDA graphs: launches are captured inside SGLang's decode/verify graphs. Routing is by graph key (width)
  only, so every rank decides alike and replay repeats the capture decision (sgl-amd-mono "graph-safe
  routing"). Builds per width bucket (8, 16, 32, 64, 128, 256, 384 rows) with masked rows, not per bs
  (bs 1..48 x 8 = 48 widths would be 48 compiles x layer kinds). Widths outside the built set fall back.
- Spec verify (DSPARK): R requests x 8 tokens. KDA: recurrence over the 8 tokens with the fp32 ReplaySSM
  ring layout of PR #7 (`SGLANG_ROCM_KDA_VERIFY_HIP_FP32_RING`), conv window read as PR #13's K1 does;
  post-verify fold (replayssm, 0.33 ms) stays on SGLang. DSA: per-token causal top-k over its own position,
  KV for all 8 tokens written before attention. DSpark draft hidden captures
  (`dspark_layers_to_capture`) written by the layer launch, as PR #13 does. Draft step (1.45 ms) stays on
  the standard path (later: DSV4.1 put its draft on mono).
- DCP8: KV is sequence-sharded across the 8 ranks; the launch reads the same paged fp8_ds_mla cache and
  DCP slot tables SGLang writes (`dcp_local_candidates`, `_localize_compact` semantics), so prefill and
  HiCache stay untouched.
- Plugin / package: new package `sgl_amd_mono/kimi_k31/` (spec `model_type kimi_k31`), importing PR #13's
  `kimi_k3` stages (seam, KDA, MoE GEMV) once #13 merges; K3.1-only stages (wide-row tiles, two-shot AR,
  DSA front/back, DCP) live in `kimi_k31/gfx950/`. Coordinate with the #13 owner (billishyahao) before
  changing shared `kimi_k3` / `common` code. `SGLANG_READS` + `test/unit/test_contract.py` against dev_amd.
- Flags: `--enable-amd-monokernel` (server), plus the existing K3.1 flags. Interplay with PR #7 (KDA ring
  layout must match), #8 (row-sharded tail; prefill only, untouched), #9/#10 (prefill sparse MLA; untouched).
- Check mode: per-layer compare against SGLang ops (`SGLANG_AMD_MONO_CHECK` pattern of glm53/minimax).

## 5. Risks and phased plan

Risks:
1. Wide-row stages are new work: PR #13's stages are GEMVs for <= 8 rows; 32-384 rows need MFMA tiles.
2. Dense GEMM tiles inside a multi-role launch may stream slower than FlyDSL tgemm (row 3: our kernels
   tied at best). Occupancy: HBM floor needs ~8 waves/CU; roles share one VGPR/LDS budget.
3. Comm overlap may not show: AR and DCP a2a are already link-bound (~185-190 GB/s); xGMI vs HBM contention.
4. Numerics: reduction order, bf16 rounding points, fp32 ReplaySSM ring; gate = gsm8k200 + KL vs mono off.
5. Hangs: every wait guarded; fault-injection test on 8 ranks per phase.
6. Compile time / number of builds (width buckets x 3 layer kinds); PR #13 compiles by running on real weights.
7. Upstream churn: PR #13 still open; sgl-amd-mono main moves; our SGLang is a fork (dev_amd).
8. c16 is MoE-bound (19.3 ms of 47.9 untouchable here); the endpoint gain shrinks with concurrency.

Plan (each phase: own branch in sgl-amd-mono off main + PR #13, own A/B tag under
`/shared_nfs/kimi-k31/results/`, default off):

- P0 measure (no mono code; no K3 reproduction, user 2026-10-31): (1) K3.1 c1 and c4 traces (only c16
  exists); (2) P2P microbench in the sgl-amd-mono runtime: mailbox hand-off latency same-XCD vs
  cross-XCD, two-shot AR at 0.9 / 1.8 / 2.75 MB, a2a at 2.75 / 11 MB; (3) PR #8 tail threshold lowered to
  decode widths (bounds mechanism B on the standard path). Output: updated section 2 numbers.
- P1 = S1: `kimi_k31` package on the section 2b architecture (planner, cut at the expert inputs), <= 8-row
  bucket with GEMV stages and experts inside, TP8 + DCP8, triton attention backend, fp32 ring; DSA attention
  still SGLang. Gate: c1 TPOT >= -15%, gsm8k, fault test.
- P2a: KDA layer + MoE front/tail at 16-384 rows (MFMA tiles, two-shot AR, de-replication), DSA layers
  still use SGLang attention up to `forward_pre_o_proj` and the layer launch from o_proj on.
  Expected ~5-6 ms/step at c16 (A for KDA + B + AR part of C).
- P2b: DSA front (norms, indexer, KV write, q path, candidate AG + global slots) in the launch.
- P2c: DSA back (W_UV before a2a, in-kernel a2a + LSE combine, gate). P2b + P2c expected ~3-4 ms/step.
- P3 (optional, re-decide with data): experts inside (S2+), draft on mono, per-step persistent (S3).

## 6. P0 results (mi355-4, 2026-10-31)

6.1 Traces (current tree: sglang-k31-amd k31/sparse-mla-prefill-hip 91ca03666b, flags off;
`traces/p0_c{1,4}_isl8k_20261031/{breakdown,layer_kinds}.txt`). VERIFY span = kernel busy (no idle):

```text
       VERIFY  DRAFT  KDA seg (kernels, us)  DSA seg (kernels, us)  experts/KDA seg  AR/KDA seg  small/KDA seg
c1      18.9   0.99   17.2, 161              40.5, 328              41               20         ~47
c4      26.1   1.06   19.0, 230              41.4, 414              91               22         ~59
c16     47.9   1.45   22.2, 446              43.5, 712              210              40         ~68   (10-30 trace)
```
At c1 / c4 the current tree already runs the HIP KDA verify (7.5 / 8.4 us), `k3_ag_attn_res_kernel` and
`_latent_tail_partial_kernel` (small-width paths), so the per-layer op list differs from the c16 table in
section 1. Small ops + AR are ~42% of a KDA layer at c1 and ~35% at c4: the mono kernel's main target.

6.2 P2P microbench (`scripts/p2p_microbench.py`, torchrun 8 ranks, no server; log /tmp/p2p_mb2.log):

```text
hand-off one-way (1 thread, ping-pong): same XCD 0.39 us, cross XCD 0.42-0.55 us (uncached + system atomics;
                                        hipMalloc + agent atomics 0.47-0.63 us); GPU <-> GPU 0.65-0.69 us
collectives, us per launch incl. back-to-back launch gap, bf16, 256 CTAs x 256 threads, all validated with
per-launch-varying inputs:
                        tagged64   flag+threadfence_system   flag+waitcnt   RCCL
AR    8 x 7168  0.11 MB     8.3          52.1                    14.6        15.8
AR   32 x 7168  0.44 MB    10.4          53.8                    16.0        28.5
AR  128 x 3584  0.88 MB    19.0          54.2                    17.4        30.3
AR  128 x 7168  1.75 MB    35.3          55.5                    23.1        40.9   (custom AR in trace ~15-22)
AR  128 x 10752 2.62 MB    60.1          61.4                    31.5        42.9
AR  384 x 7168  5.25 MB   133.1          90.4                    65.3        59.0
A2A 128 x 12288 3.00 MB    56.6          40.8                    26.4        34.7
A2A 128 x 49152 12.0 MB   227.0         115.8                    98.5        54.0
```
Findings: (1) a hand-off is < 1 us, so a fused stage boundary costs ~1-2 us with skew, cheaper than the
2-4 us section 2 assumed. (2) Tagged words win to ~0.9 MB, then lose half the link to tags. (3) The
system fence is a ~50 us floor; the `s_waitcnt vmcnt(0)` publish (stores to uncached peer memory acked
before the flag) reaches custom-AR speed at 1.75 MB. It passed the varying-data validation, but its
ordering is not a documented guarantee: stress-test (random delays, 10^5 launches) before P2 relies on it.
(4) In-kernel collectives are not faster than custom AR / RCCL at >= 1.75 MB: wide-row comm gains come
from overlap and fewer bytes, not from the transport. (5) The 11 MB DCP a2a must shrink first
(W_UV before the a2a -> 2.75 MB, 26 us in kernel vs NCCL 57.8 today).

6.3 PR #8 row-sharded tail at decode widths (ab_bench, same tree, `results/ab/p0_{control,rowshard8}_20261031`):

```text
                 control   MIN_TOKENS=8
decode c4 TPOT    7.71      8.26  (+7.1%)
decode c16 TPOT  15.80     15.68  (-0.8%, noise)
decode c48 TPOT  31.34     30.35  (-3.2%)
prefill 8k TTFT  528.4     503.5  (-4.7%; prefill takes the same path)
```
Row-sharding the tail with extra standard-path launches loses at 32 rows and pays only at 384: mechanism B
is worth doing only inside the launch (no extra launches), width-switched. Side item for OPT_TRACKER: a
decode threshold of 256-384 rows on the standard path (c48 -3%) is independent of the mono work.

6.4 Revised estimates (section 2 table updated):
- c1, S1: KDA layer small ops ~47 -> ~8 us, AR 20 -> ~8, de-replication ~8 -> ~58 us x 69 = 4.0 ms; DSA layers'
  o_proj .. MoE part ~50 us x 24 = 1.2 ms -> ~-5 ms of 18.9 (-27%); DRAFT unchanged.
- c4, S2: ~72 us per KDA layer (5.0 ms) + DSA ~3.6 ms -> -5 to -8.6 ms of 26.1.
- c16, S2: A ~5.0-5.5 (cheaper hand-offs), B ~1.8, C revised to ~1.5-2.0 (AR 0.5-1.0 + DCP ~1.0), D 0
  -> -8 to -9 ms of 47.9.

## 7. S2 layer launch: dataflow (design, mi355-4 2026-10-31)

One launch per layer L at 16..384 rows (R requests x 8 tokens), between aiter's expert stage 2 of L-1 and
stage 1 of L. The residual stream is column-sharded: rank r owns hidden columns [896 r, 896 (r+1)).
Stages (each a set of tiles; hand-offs through the mailbox; P = partial over ranks):

```text
T1 MoE tail (L-1)  in: expert out latent P [rows,3584], shared down P [rows,7168]
                   latent: two-shot AR (RS + AG)            -> full latent on every rank
                   shared: RS to owners                      -> shared slice [rows,896]
                   latent norm + latent up, owner's 896 output columns only (1/8 of W_up)
                   y slice = up + shared + residual          (column-sharded, no replicated work)
T2 attn seam (L)   AttnRes on the owner's columns; per-(row, source) RMS / score partials and the
                   input-norm RMS exchanged as scalars; bank kept column-sharded (1/8 the reads)
                   x slice -> AG x                           (in_proj needs K = 7168)
T3 attention (L)   KDA: in_proj tiles consume x by K-chunks as the slices land; f_b, conv, recurrence,
                   gated norm per (request, head) task (K1 math, R requests: conv staged per step for
                   R slots, ring or snapshot state); DSA: SGLang's attention between two launches (P2b/c)
T4 o_proj          tiles, epilogue pushes column slices to owners (RS)
T5 MLP seam        owner columns, scalars exchanged, post-attention norm -> x' slice -> AG x'
T6 MoE front       router 112 experts / rank + latent down 448 rows / rank + shared gate_up 1536 rows,
                   one weight pass over x'; AG router logits (0.06 MB) and latent (0.9 MB);
                   top-16, sort, mxfp8 quant -> aiter stage-1 inputs; SiTU + shared down -> shared P
```

Bytes out per rank per layer at 128 rows: latent RS + AG 1.6 MB, shared RS 0.8, x AG 1.6, o_proj RS 1.6,
x' AG 1.6, router + latent AG 0.8: ~8 MB vs ~7.7 MB for today's two all-reduces. Comm volume is unchanged;
what changes is (a) no replicated work (router, latent down / up, seams, norms), (b) every collective is
split into slices that a GEMM produces or consumes tile by tile, so it overlaps, (c) ~22 kernels -> 1 launch
+ 2 expert launches. Formats by size (section 6.2): tagged words <= 0.9 MB, waitcnt-flag above.

Reuse from PR #13: seam / KDA math (stages/seam.py, attention/kda.py), mailbox + guard + peer runtime, the
role rule (pollers carry no weight loads). New: MFMA row-tile GEMM stage (M = 16..384, N-tiles over 256
CTAs grouped by XCD), slice collectives, the planner (stage graph -> tasks -> CTAs per width bucket).
P1b keeps the c1 bucket on PR #13's launches (GEMV, experts inside) and adds the ring; the wide buckets
come from this design (P2a = KDA layers + MoE tail/front, P2b/c = DSA front/back).

Why columns, not rows: DSV4.1's mono runs token parallel (`deepseek_v41/gfx950/token_par.py`: rank r owns
rows r S/TP .., all-reduces become reduce-scatters to the row owner, seams need no exchange). For K3.1 the
latent up (7168 x 3584 bf16, 51 MB) would then be read whole by every rank for its rows -- the replicated
cost we remove (~20 us x 92 at c16; PR #8's row-sharded tail pays it and loses at c4). Columns read 1/8 of
W_up at the price of a scalar exchange per full-row reduction (~1-2 us, section 6.2). Re-decide if the
scalar rounds measure > 4 us a layer.

GEMM tiles at 16..384 rows: x does not fit LDS (128 x 7168 bf16 = 1.8 MB), so the PR #13 GEMV form
(x rows in LDS, weights streamed into MFMA B operands) stops at ~16-48 rows. P2a builds a tile stage from
aiter's FlyDSL a16w16 gfx950 GEMM pieces (`aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`:
`make_gemm_ab_load_context`, `async_load_to_lds`, `write_cshuffle_vec_to_global`), K-chunked x through LDS,
epilogue into the slice mailbox.

## 8. P1a results (mi355-4, 2026-10-31; K3 launches on K3.1, REPLAYSSM=0, SIM acc 3.84)

```text
ab_bench (results/ab/p1a_mono_{on,off}_rs0_20261031)   mono off   mono on
decode c1 TPOT                                          5.06       4.65   (-8.1%)
decode c1 P90 ITL                                       6.36       5.84   (-8.2%)
decode c4 TPOT (not routed: 32 rows)                    7.98       7.92   (noise)
prefill 8k TTFT                                         524.1      522.8
c1 trace mono on (traces/p1a_c1_mono_on_20261031): VERIFY 17.67 ms; K2 ~100 us x ~85 launches, K1 ~37 us x 68
```
Below the -25% estimate. Why: at c1 a layer reads ~300 MB of weights a rank (experts ~250 MB at 128
routed pairs, dense ~70 MB after de-replication), so the step is HBM-bound and launch / small-kernel
overhead is the smaller part. K2 streams at ~3.2 TB/s (100 us for ~320 MB) vs ~120 us for the standard
path's kernels it replaces; K1's in_proj GEMV (90 MB) runs ~2.4 TB/s where the standard hgemm gets 5.1.
Revised c1 levers: K2 weight streaming to ~5 TB/s (~35 us a layer, ~3 ms a step), K1 in_proj at hgemm speed
(~15-20 us x 68 = 1.0-1.4 ms). Next measurement: the launches' stage timeline (`timeline=True` builds,
`gfx950/stamps.py`) to place the K2 / K1 time per stage.

K2 stage timeline, 8 rows, graph-replayed behind plain launches (`time_k2.py`; 103.5 us a launch):
```text
x published (o_proj + AR + seam)  21.5 us | route table (router AG, top-16, queue) 39.9 (one CTA at 38.4,
the rest at 21.7) | expert gate/up done 65 (~25 us, ~6.8 TB/s) | expert down done 86 (~20 us, ~4.4 TB/s) |
latent norm, latent up, MoE AR reduce 102.6
```
The experts stream near the floor; the route phase (18 us) and the tail (16 us) are latency. Halving both
is ~20 us a layer, ~1.8 ms a c1 step: P1b's first items.

gsm8k caveat: the P1a gate (`results/acc/p1a_gsm_{on,off}_rs0_20261031`, 0.99 on / 0.98 off) ran 64 threads,
so most verify steps were wider than 8 rows and took the standard path. It does not gate the mono path.
`gsm8k_gate.sh` now takes THREADS; mono gates run THREADS=1 (every step bs 1, as PR #13's
--max-running-requests 1).

## 9. P1a on the ring, P1b step 1 (mi355-4, 2026-10-31)

```text
ring config (REPLAYSSM=1), SIM acc 3.84      mono off   mono on (K3 launches + K1 ring)
decode c1 TPOT                                 4.96       4.60  (-7.3%)
decode c1 P90 ITL                              6.24       5.81
decode c4 TPOT (not routed)                    7.80       7.79
greedy gsm8k200, THREADS=1, real draft         0.98       0.985   accept len 4.79 / 4.73
(results/ab/p1r_mono_{on,off}_20261031, results/acc/p1r_gsm1_{on,off}_20261031)
```

K2 route chain per rank (8 rows, `time_k2.py`, us after the launch start on that GPU; GPU clocks are not
aligned across ranks, so only per-rank deltas are read): x published +0 | router part pushed +5.1 |
logits in +8.2 | top-16 +16.0 (radix select, one wave: 7.8 us) | route +16.8 | latent part pushed +6.4 |
latent quantized +15.7 (every CTA read ~115 KB of tagged bf16 pairs and quantized all 8 x 3584) | table +18.8.
Step 1 (commit 6e50a20, `stages/moe.py`): top-16 by 16 rounds of a wave max over each lane's 14 sorted
candidates (GLM-5.3-Flash's router form; 53-comparator network) -> 3.9 us, same picks and order; the latent
quantized once by CTAs 160..255 (free first) into a LATQP mailbox that every CTA polls (36 KB each).
Table at +13.3 (was +18.8); K2 103.5 -> 100.6 us a launch.

Tail (step 2 target): experts down done ~81.6 us, but the shared down (448 tasks) mostly runs after it on
every CTA (mark 4 median 89.6), then lnorm / latent up / reduce to ~98. The shared path (SGU -> SiTU ->
shared down) does not depend on the route; it fits the ~13 us where CTAs 56..255 wait for the route table.
Move stage_h (SiTU) to the SGU CTAs and run the shared down there before the experts.

Step 2 measured and rejected (mi355-4, 2026-10-31; not committed): (a) all 448 shared-down tasks on CTAs
56..159 right after their SGU task: 109.5 us a launch (route table last 52.8, those CTAs' experts start
late: each task polls H, pushes FINAL to 8 ranks and publishes, ~4-5 us); (b) only the 56 up-stage tasks
there, the other 392 left after the down: 103.2 us (route table median 39.0 vs 35.4; suspect the early
FINAL pushes competing with the router logits all-gather on xGMI). The tail stays as PR #13 has it.

E2E after step 1 (ring config, SIM acc 3.84; `results/ab/p1b1_mono_on_20261031`): c1 TPOT 4.49 ms,
P90 ITL 5.67 (mono off 4.96 / 6.24, step-0 mono 4.60 / 5.81). gsm8k200 greedy THREADS=1 0.98
(`results/acc/p1b1_gsm1_on_20261031`).

gsm8k per-item (THREADS=1, greedy, real draft; ids matched): mono off vs P1a-ring mono 78/200 identical
generations (correct 196 / 197); off vs step-1 mono 62/200 (196 / 196); P1a-ring mono vs step-1 mono 62/200.
Step 1 is designed to be bit-identical (same picks, order, renorm and quant), so mono-vs-mono 62/200 points
to run-to-run nondeterminism on the mono path (the standard path's greedy band is 188-190/200).
Determinism (mi355-4, 2026-10-31; one server, two greedy THREADS=1 passes each; results/acc/det_mono_{on,off}_20261031_{a,b}):
mono on 66/200 identical (0.97 / 0.98, accept len 4.48 / 4.81); mono off 108/200 identical (0.985 / 0.99,
4.69 / 4.92), and 99-100/200 across two off servers. The standard path is itself nondeterministic at bs 1
(the 188-190/200 band is a 64-thread number; suspect split-K atomics in small-M GEMMs, row 3), so the
mono-vs-mono 62/200 is not evidence that step 1 changed numerics; the K2 kernel replays bit for bit on fixed
inputs. Mono runs a little less reproducible (66 vs 108) and scores 0.97-0.985 over 4 runs vs 0.98-0.99 over 3:
within 200-question noise; a 1319-question gsm8k (or KL vs mono off on fixed prompts) is the next gate
before calling the c1 accuracy equal.

## 10. S1 accuracy gate and P2a step 1: tile GEMM v0 (mi355-4, 2026-11-01)

### 10.1 gsm8k1319 gate (S1 c1)

`launch_server.sh` with SIM=0 (real draft), `NUM=1319 THREADS=1 TEMP=0 gsm8k_gate.sh`; sglang-k31-amd
origin/dev_amd 3d2aa95c9d, sgl-amd-mono k31/mono-s1 @ 6e50a20.

```text
mono on   1287/1319 = 0.9757   accept len 4.709   wall 610 s   results/acc/g1319_mono_on_20261101
mono off  1288/1319 = 0.9765   accept len 4.697   wall 683 s   results/acc/g1319_mono_off_20261101
paired (evaluator's symbolic_correct): on-only 2, off-only 3, McNemar exact p = 1.0;
identical generations 1246/1319 (paired_vs_off.txt; per-item output-rs0.jsonl copied next to metrics.json)
```
-> S1 accuracy equals mono off. gsm8k_gate.sh fixes: `NUM` knob (default 200); the accept-len window now
compares full date-times (it read 0 lines when a run crossed midnight UTC).

### 10.2 Tile GEMM v0 (`kimi_k31/gfx950/stages/tile.py`)

v0 = PR #13 GEMV generalized: task = 16 nb weight rows x all m rows, the 8 waves split K (gemv's K order),
both operands buffer-loaded straight into MFMA registers (weights NT, x from L2), partials summed through
64 KB LDS in rounds; `emit(row, col, v4, live)` hook (bf16 / f32 stores provided), optional K split across
CTAs (`ks`, f32 partials [ks, m, n] for the consumer to sum). `tile_bench.py`: rel L2 1.7e-3 (bf16 output
rounding) at m 16 / 20 / 128 / 384, nb 1 / 2 / 4, ks 1 / 2 / 4, K tails (1536 = 1.5 chunks a wave).

Graph-replayed, MALL-busting (us; tgemm = server's tuned path):

```text
                 m=16 best (nb,ks)   tgemm | m=128 best          tgemm | m=384 best         tgemm
6288x7168        29.5 (2,1)          18.8  | 111.6 (2,1)          29.9  | 789 (2,1)           68.4
7168x3584        17.5 (2,1)          11.2  |  50.7 (2,4)          21.7  | 304 (2,4)           37.3
 896x3584        10.4 (1,1)           4.7  |  11.1 (1,4)           7.0  |  49 (1,4)           13.0
7168x1536         9.7 (2,1)           6.1  |  26.3 (2,2)          10.1  | 150 (2,2)           18.1
```

Why (ISA via `scripts/tile_isa.py`): (1) LLVM's scheduler sank every prefetch load to just before its MFMA
(66 x `s_waitcnt vmcnt(1)`); `rocdl.sched_barrier(0)` around each unit keeps 36-43 loads in flight but
changed nothing (126.9 vs 129.5 us at m=128 nb=1), so latency is not the limit. (2) Bytes per CU are: the
CTAs pull ~25-31 GB/s each (the same per-CU rate as the 6 TB/s HBM read floor), and total traffic is
(N / NT)(m + NT) K 2 -- x is reread by every task, so at m=128, NT=16-32 the x traffic is 4-8x the weights.
`ks` does not change that sum (it only adds CTAs). Cutting it needs NT >= ~m, which with all 8 waves on K
means m/16 x NT/16 x 4 accumulators a lane: 256 VGPRs (the 2-waves-a-SIMD cap) at m=128, NT=32 already
(m=128 nb=2: 256 VGPR + 408 B scratch; m=384: 1576 B scratch). So v0's structure is a dead end past ~16 rows.

aiter's own hgemm at 8 waves (`scripts/hgemm_8wave_sweep.py`, M=128; only t128x128x64 stages 2 w2x4x1
passes the wrapper's config check at 512 threads):

```text
6288x7168  ks4 37.5 / ks2 48.7 / ks1 81.3   tgemm 29.8      7168x3584  ks4 25.7 / ks1 42.9   tgemm 20.6
6016x7168  ks4 34.0 / ks1 80.9              tgemm 27.9      7168x1536  ks2 18.1 / ks1 21.5   tgemm 10.4
```
tgemm's winners are 16-wave (e.g. t128x128x64x4 ks4 w4x4x1). Without split-K the 128x128 tile leaves
~200 CUs idle (49 tasks for in_proj) -> 2.7x; with split-K 4 (atomics) it is still +25%.

### 10.3 Consequence for S2 (superseded by 10.4: the +25% came from the HTI default)

At 128+ rows a dense GEMM inside a 512-thread launch costs ~+25% (best case, aiter structure + split-K) over
tgemm; KDA layer at c16: in_proj + MoE front + latent up + o_proj ~ 90 us of tgemm a layer -> ~+20 us a layer
if moved in naively, against the S2 estimate of -8..-9 ms/step (section 2, ~-90 us a layer). Options:
(a) tile stage = aiter's structure (LDS async DMA, 128x128, w2x4x1) as an in-launch stage, split-K 4 with f32
    partials that the consumer sums (no atomics / finish; fits the column-sharded RS of section 7);
(b) as (a) in a 1024-thread wide-row launch (16 waves, tgemm's shape; 128 VGPRs a wave for every other stage);
(c) keep in_proj / MoE front / o_proj as tgemm launches, the wide-row launch fuses only what lies between them
    (seams, norms, router / top-k / sort prologue, latent up owner slice): ~3-4 launches a layer instead of 1.

### 10.4 8 vs 16 waves, HTI off (mi355-4, 2026-11-01) -> option (a)

`hgemm_8wave_sweep.py` now sweeps use_half_tile_interleaved (the wrapper's default is True; HTI rejects
k_waves > 1 and most 8-wave tiles, which is why 10.2's sweep saw only one config), block_k 64/128, stages 2-4,
split-K 1/2/4; `WAVES=16` sweeps 1024-thread configs. M=128, us (graph-replayed, MALL-busting):

```text
shape          tgemm | 8-wave best                      ks1 best | 16-wave best              ks1 best
6288x7168      30.1  | 27.9 t128x128x64x4 ks4 w2x2x2    40.7     | 28.3 t128x128x64x3 ks4 w2x8x1  51.6
6016x7168      27.6  | 26.6 t128x128x64x4 ks4 w2x2x2    39.9     | 27.1 t128x128x64x4 ks4 w2x8x1  50.9
7168x3584      20.6  | 18.4 t128x128x64x3 ks4 w4x2x1    23.1     | 18.4 t128x128x64x4 ks4 w2x8x1  28.1
 896x3584       7.0  |  8.0 t128x32x128x3 ks4 w4x2x1    19.5     | 10.5 t128x64x128x3 ks4 w4x4x1  26.5
7168x1536      10.1  | 10.9 t128x64x64x4 ks2 w2x4x1     11.4     | 12.0 t128x64x128x3 ks2 w4x4x1  14.7
sum of the 4 big shapes: tgemm 88.4, 8-wave 83.8, 16-wave 85.8 us a KDA layer
```
(all ks > 1 rows use aiter's atomic split-K; ks1 is 1.3-2.4x slower at the K=7168 shapes, so the in-launch
stage needs a K split -- with consumer-summed f32 partials that is extra traffic, to be measured.)

1024 threads for PR #13's stages (`scripts/mono_threads_isa.py`, compile only, THREADS patched in
common.plan): KDA front 205 -> 103 VGPR, no spill (twice the waves, half the chunks a wave); K2 layer_post
219 -> 128 VGPR with 324-400 B scratch at s = 1 / 4 / 8 (a fixed-size stage, not the row count).
-> (b) buys nothing on the GEMMs and costs spills; (a) is the plan.

### 10.5 Own FlyDSL tile GEMM v1 (`kimi_k31/gfx950/stages/tile_mma.py`, mi355-4, 2026-11-01)

No aiter gemm code: a task is a [bm x bn] output tile over one K slice; 8 waves in a wm x wn grid (no
cross-wave reduction); BK = 64 steps, x and W pieces (16 B a thread) loaded `depth` steps ahead into
registers, written into one of two LDS buffers with rows padded to 72 bf16 (16 rows of a ds_read_b128 hit
16 distinct 4-bank groups), MFMA operands read from LDS; ks > 1 leaves f32 partials. Same `emit` hook as v0.
`test/kernels/kimi_k31/mma_bench.py`: rel L2 ~1e-7 (f32 partials) / 1.7e-3 (bf16), m = 20 / 128.

```text
m=128, us        t128x128 w2x4: ks1   ks2   ks4  | best other              | tgemm | aiter 8w best (10.4)
6288x7168                      74.7  41.9  30.1  | t128x64 w4x2 ks4 33.0   | 29.4  | 27.9
6016x7168                      75.4  41.6  29.2  |                         | 27.5  | 26.6
7168x3584                      40.4  24.3  19.0  | t128x64 w4x2 ks4 21.3   | 21.3  | 18.4
 896x3584                      39.0  22.0  13.7  | t128x32 w8x1 ks8  6.3   |  7.1  |  8.0
7168x1536                      20.8  13.9  11.4  |                         | 10.1  | 10.9
```
Prefetch depth 2 / 4 / 6 changes nothing (in_proj 30.1 / 30.2 / 30.3): not load-latency bound; ks8 is worse
for the big shapes (32.8). The stage excludes the partial sum: a separate torch sum adds 3.8 (in_proj) to
10 us (o_proj, mostly its launch); fused into the consumer only the bytes remain (in_proj 4 x 128 x 6288 f32
= 12.9 MB written + read). aiter's 27.9 uses packed bf16 atomics instead -- the next comparison.

### 10.6 Split-K merge and m buckets (mi355-4, 2026-11-01)

Merge modes (`--mode`): part = f32 partials [ks, m, n] for the consumer; atom / atombf = f32 / packed-bf16 adds
straight from the MFMA layout (a lane adds 4 columns of one row: 16 rows a wave instruction); lin / linbf =
the f32 tile shuffled through LDS (aliasing the x / W buffers) and added lane-contiguous (a wave covers
whole 64 B lines, as aiter does). Masked lanes add at the buffer end (0x7FFFFFFF + 4 j wrapped and faulted).
m=128, t128x128 w2x4 ks4, us: in_proj part 28.7 / atom 59.4 / atombf 41.0 / lin 32.5 / linbf 30.2;
o_proj 11.3 / 50.3 / 28.8 / 18.4 / 12.7 -> linbf: the consumer reads plain bf16 (1.6 MB for in_proj vs 12.9 MB
of partials), rel L2 3.2e-3 (bf16 rounding a K slice, as tgemm's own split-K).

Best linbf config a bucket (us, ours / tgemm; tile, ks):

```text
m     in_proj 6288x7168          latent up 7168x3584        owner 896x3584          o_proj 7168x1536
16    16.4 / 18.7 t16x128 ks4    9.8 / 11.1 t16x128 ks4     5.0 / 4.7 t16x128 ks8   5.8 / 6.1 t16x128 ks4
32    17.6 / 20.5 t32x128 ks4    10.7 / 12.9 t32x128 ks4    4.9 / 5.0 t16x128 ks8   6.5 / 6.3 t32x128 ks4
48    21.5 / 22.9 t48x128 ks4    12.0 / 15.2 t48x128 ks4    5.0 / 5.3 t16x128 ks8   7.6 / 7.5 t48x128 ks4
128   30.2 / 30.1 t128x128 ks4   19.5 / 20.3 t128x128 ks4   6.5 / 7.1 t128x32 ks8   12.7 / 10.2 t128x128 ks4
256   46.3 / 51.5 t256x128 ks4   28.8 / 62.2 t128x128 ks2   10.1 / 9.6 t128 ks8     16.9 / 27.7 t128x128 ks2
384   64.7 / 69.1 t192x128 ks2   37.2 / 36.6 t192x128 ks2   11.6 / 12.9 t128 ks8    22.1 / 18.3 t192x128 ks2
```
Waves: bm 16 / 48 -> w1x8, bm 32 / 128 -> w2x4, bm 192 / 256 -> w4x2; bm = m rounded to 16 up to 48. tgemm's
m=256 entries look untuned (7168x3584 62 us). Repro: `mma_bench.py --time --m 16,32,48 --cfg 16:128:1:8:4:2,...
--mode linbf` (cfg = bm:bn:wm:wn:ks:depth; wm x wn must be 8).

### 10.7 Column-slice collectives (`kimi_k31/gfx950/stages/coll.py`, mi355-4, 2026-11-01)

RS / AG / two-shot AR over column slices (rank r owns columns [r sl, (r + 1) sl), sl = H / 8), on PR #13's
`Mailbox`, `PeerBuffer` and guard. "tag": 16 B stores of 2 bf16 pairs + 2 tags; CTA group j (32 of the 256 CTAs)
pushes to rank j, so a wave's destination base is uniform; consumers poll exactly the pairs they read. "flag":
CTA b owns chunk b of every slice, plain 16 B uncached stores, `s_waitcnt vmcnt(0)` + barrier + a tagged flag a
(phase, source, CTA). Both sum in f32 in rank order: bit-exact against the torch reference, deterministic.
Regions are double-buffered by epoch parity. `test/kernels/kimi_k31/coll_bench.py`: every launch of a 20-launch
graph has its own inputs, 3 input sets rotate between replays (stale data cannot pass).

```text
us a launch (graph replay, incl. the launch gap; 256 x 512 threads), all bit-exact
shape       MB    AR tag / flag   AG tag / flag   RS tag / flag | 6.2 AR: HIP tagged64 / waitcnt / RCCL
8 x 7168    0.11   9.0 / 15.7      5.4 / 12.7      6.1 / 11.2   |   8.3 / 14.6 / 15.8
32 x 7168   0.46  10.1 / 18.3      6.9 / 13.5      7.5 / 11.7   |  10.4 / 16.0 / 28.5
128 x 3584  0.92  14.1 / 18.7      9.6 / 15.1      9.9 / 13.0   |  19.0 / 17.4 / 30.3
128 x 7168  1.84  23.3 / 24.5     14.1 / 19.3     14.5 / 17.5   |  35.3 / 23.1 / 40.9
128 x 10752 2.75  32.9 / 33.4     18.7 / 26.1     19.0 / 23.8   |  60.1 / 31.5 / 42.9
384 x 7168  5.51  59.8 / 63.3     32.8 / 52.1     31.7 / 47.9   | 133.1 / 65.3 / 59.0
```
Findings: (1) 16 B tagged stores with a uniform destination take tagged from 35.3 to 23.3 us at 128 x 7168; it
now ties or beats flag at every size, so section 6.2's rule (tagged <= 0.9 MB, flag above) is replaced: tagged by
default, flag kept for overlapped phases where link bytes matter (half the bytes). (2) The standalone AR beats
RCCL to 128 rows and ties it at 384; AG and RS each cost about half an AR, so section 7's split (RS and AG at
different stage boundaries) costs no more than today's AR before any overlap. (3) Stress (`--stress 2000
--launches 50`, random per-CTA start delay 0..200 us and 0..10 us): AR 8 x 7168 and 128 x 7168, both formats,
8 x 100050 launches, 0 mismatches -> the waitcnt publish ordering holds under skew (still not a documented
guarantee; rerun the stress after ROCm / firmware changes). (4) Open: parity. The bench bumps the epoch every
launch; in production a step's launches share an epoch (PR #13 tag scheme, `kimi_k3/gfx950/tag.py`), so two
consecutive launches using one region need alternating parity from (epoch, slot) -- an odd launch count a step
or a per-launch counter. Safe without parity only if every rank's read of launch L's region happens before a
push the producer needs for launch L + 1.

Wiring (next): with linbf the o_proj / shared-down tile is complete only after its last K slice; a per-tile
arrival counter lets that task read the tile, RS-push it and zero it (decision (a)). in_proj takes x by K
chunks from the AG region as the slices land; the AR inside T1 (latent) uses `tag_reduce` with an emit that
feeds the latent norm instead of a store.

Repro: `cd /tmp && torchrun --nproc-per-node 8 /sgl-workspace/sgl-amd-mono/test/kernels/kimi_k31/coll_bench.py
--time` (add `--shapes 8x7168,128x7168 --kinds ar --launches 50 --stress 2000 --delay 20000` for the stress).
Results: `/shared_nfs/kimi-k31/results/coll_slices_20261101/{summary.txt,sweep.log,coll_stress_*.log}`.
