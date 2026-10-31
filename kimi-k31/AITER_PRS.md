# aiter open PRs relevant to Kimi-K3.1 (gfx950)

Scanned 2026-10-30 on mi355-4: 600 open PRs on ROCm/aiter, filtered on
kimi/k3/kda/situ/dspark/attn_res/f_b_proj, gfx1250/gfx942-only PRs dropped.
All listed PRs are fetched into `/sgl-workspace/aiter-k31` as `refs/remotes/pr/<N>`
(not merged). Base: aiter main `f0b96f49e`.

Note: `gh` cannot read ROCm/aiter (org blocks classic PATs). Use unauthenticated
REST: `curl -s "https://api.github.com/repos/ROCm/aiter/pulls?state=open&per_page=100&page=N"`.
Re-triage: `bash /tmp/k31_pr_triage.sh` (recreate from this doc if /tmp is gone:
merge-base, `git rev-list --count`, `git merge-tree --write-tree origin/main pr/N`).

`behind` = commits on main since the PR's merge-base. `conflict` = `git merge-tree` against main.

## Priority A — directly on the K3.1 hot path, clean on main

| PR | What | Size | Notes |
|----|------|------|-------|
| 7169 | Kimi-K3 gfx950 BF16 GEMM rows (>=3% over default) | 1 file +37 | overlaps 6959/6089 on `kimik3_bf16_tuned_gemm.csv`; sglang also ships its own table (`kimi_k3_amd/bf16_gemm_table.py`) — diff against it |
| 6959 | Kimi-K3 MI355X GEMM tuning tables | 4 files +151 | same csv as 7169 |
| 6026 | Fused KDA decode kernels (FlyDSL, gfx950) | +2428 | backs `SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter`; sglang may vendor its own (`SGLANG_K3_FLYDSL_SOURCE`) |
| 7167 | Fused SiTU-and-mul (Triton), draft | +784 | K3.1 `hidden_act=situ` |
| 7028 | causal_conv1d_split_qkv update (KDA prefill) | +1038 | |
| 7034 | KDA cleanup (bf16 state) | +73/-106 | |
| 6912 | KDA: cu_seqlens_cpu, async varlen metadata, draft | +322 | overlaps 6498 |
| 6484 | MoE: caller-provided precomputed BM=16 sort | +1149 | overlaps 6485 (`fused_moe.py`, tests) |
| 6485 | MoE a4w4 adaptive aux sort, in-place output | +417 | stack with 6484 |
| 6270 | Custom allreduce + fused residual add + row-selective RMSNorm | +2311 | overlaps 6960 (custom_all_reduce.*) |
| 6960 | `all_gather_lastdim_add` in custom all-reduce | +403 | dev_amd commit 14603f485 (LL all-gather MoE tail) may need it — check |
| 6961 | mla_gluon on-device kernel selection for CUDA-graph MLA | +123 | |
| 7088 | unified_attention length-aware KV split, d64 bf16-Q/fp8-KV | +601 | |
| 7069 | GDR decode optional deferred spec-decode state | +263 | relevant to DSPARK verify |

## Priority B — fixes / infra likely needed, clean

| PR | What |
|----|------|
| 6965 | fused_moe: stage-2 reduce target device-memory exhaustion crash (+3/-2) |
| 7104 | FlyDSL moe_reduce: guard -1 topk_ids padding (+4/-1) |
| 7172 | MoE: skip unused intermediate alloc for fused stage-1 quant |
| 7161 | MoE sorting: include HIP bf16 header (+1) |
| 7134 | gfx950 E=128..255 decode M>=17 multi-phase sorting (K3.1 E=896 -> probably no effect, verify) |
| 7101 | torch_to_c_types accepts generic torch.Stream |
| 6260 | JIT build fixes (metadata-only setup.py, empty rocminfo) |
| 6712 | FlyDSL A16W16 AOT semaphore dtype fix |
| 7063 | Precompile MoE deployments (EP/TP) |
| 7022 | fp8_mqa_logits: no recompile per batch size (gfx950), draft — DSA indexer |
| 7128 / 6952 | OPUS FP4 MQA writer / MXFP4 MQA logits unified API (DSv4-oriented; K3.1 indexer_use_fp8=false -> likely N/A) |
| 7185 | OPUS fallback when A0 strict unsupported (mostly gfx1250 dirs; check gfx950 effect) |

## Conflicting with main (stale) — only rebase if needed

6089 (K3 bf16 decode GEMM configs, behind 481), 6543 (FlyDSL Kimi K2 grouped MoE, 344),
6597 (A16W16 param cache, 260), 6498 (KDA backward — training, skip), 6764 (non-causal PA
decode), 6550 (fused K3 KDA whole-layer ops, clean but behind 337), 5756 / 5487 / 5368
(old K3 A4W4/A8W4 configs and latent FHMoE prototype).

## SGLang K3-on-gfx950 PR queues (checked 2026-10-30; none merged upstream)

kevin-mii queue (owner of sglang-k31-amd; "land one at a time"; tested with EAGLE3 4 draft tokens,
not DSPARK): #44978 verify argmax split + aiter tuned bf16 rows (open draft) · #44979 MLA FP8 prefix
K/V prefill (closed) · #44980 streaming AttnRes bank read for prefill (closed) · #44981 prefill MXFP4
MoE act / SiTU epilogue / row-sharded tails (open draft) · #44982 KDA verify tuning + four-step
ReplaySSM commit + chunk-major prefill scan (open draft) · #44983 MLA one-launch q/k prep, split
combine, fp8 verify query (closed) · #44984 MLA split combine + W_vc + output gate fused (closed) ·
#44985 TP8 LL collectives, column-parallel latent tail (open draft; cf. dev_amd 14603f485) · #44986
attention-output TP8 AR fused with the FFN-side read (closed) · #44987 fused router top-k + route
sort (<=16 tokens, quant <=8) + zero-copy MoE output + shared-down FlyDSL out= (open draft) · #44988
column-parallel MoE front + route-sort bank partials (+8317, open draft) · #44989 MLA q_a/kv_a norms
+ q_b one launch (closed).

yuychang stack (closed/unmerged): sglang#43126 radix-4 route + sort + MXFP8 quant (closed: 1-2.6%
slower TPOT) needing aiter#6208 (closed); sglang#43130 ROCm fused kernels integration (closed;
`hip_fused_ar_rmsnorm` uses aiter#6270); aiter#6484 `register_presorted_moe` BM=16 aux-sort (open).

DSPARK caveat for all decode-size gates: verify runs 8 tokens/request, so a "<=16 tokens" fast path
covers c1-c2 only.

## Decisions

(none yet — record accept/reject with A/B evidence here, one row per PR)
