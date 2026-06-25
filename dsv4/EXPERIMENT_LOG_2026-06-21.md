# DeepSeek-V4-Pro serving perf — experiment log (2026-06-21)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp 55 — Port shared-expert-local to sglang-upstream + A/B; debunk the "upstream is slower" / BLOCK_M scare (2026-06-21)

### What was done
1. Ported the **shared-expert-local (SE-local)** PoC from the dev clone (`/sgl-workspace/sglang`)
   onto **`/sgl-workspace/sglang-upstream`** (which already has CK-GEMM + batched/contig
   rope DEFAULT-ON, Exp 54). Edits: `models/deepseek_v2.py` (`skip_shared_experts` param
   in forward/forward_normal), `models/deepseek_v4.py` (`_SHARED_EXPERT_LOCAL` flag +
   compute-local / skip / add-after-reduce-scatterv), `configs/cohere2_moe.py` (the
   `@strict` no-op patch, SKILL §2a, needed to import on upstream too).
2. Also applied the **rope contig-kernel `BLOCK_M=32 → 8` + `num_warps=4`** tweak
   (today's microbench: 1.6–1.8× faster kernel) to both clones.
3. A/B on upstream (user's choice = "plan B"): baseline = **gatherv ONLY**;
   SE-local = **gatherv + TP1 + se-local**. ROCM700A=0, chunk 65536 (=8192/rank,
   server resolves to chunked_prefill_size=8192), ratio 1.0, np=conc*8, ATOM-aligned
   launcher (`run_sgl_dsv4_aligned.sh`).

### Correctness
- gsm8k 5-shot (SE-local ON, upstream): **flex 0.9500 / strict 0.9507** — correct.

### Performance (upstream, plan B baseline = gatherv only)
| workload | baseline (gatherv) | + TP1 + se-local | delta |
|---|---:|---:|---:|
| pure-prefill 1k (OSL=1, c256) | 48,941 | **52,235** | **+6.7%** |
| pure-prefill 8k (OSL=1, c256) | 49,639 | **52,674** | **+6.1%** |
| c512 1k/1k (total tok/s) | 17,738 | 17,136 | **−3.4%** |
| c512 8k/1k (total tok/s) | 33,828 | 34,064 | +0.7% |
- pure-prefill +6–7% (consistent with Exp 50). c512 diluted/negative because **plan B
  bundles TP1**: TP1 replicates the shared expert → in DECODE each rank does the full
  shared-expert GEMM (~8× FLOPs, only saves the all-reduce); at c512 1k/1k (decode-bound)
  that cost outweighs se-local's tiny prefill benefit → −3.4%. 8k/1k (prefill-heavier)
  roughly breaks even (+0.7%, TTFT 48.0s→44.9s).

### The "upstream 34k vs this-morning 36k" investigation — it was an editable-finder BUG
User flagged that c512 8k/1k SE-local hit ~36k earlier (Exp 52 logged 36,031 on the dev
clone, 6/18) but only 34k now on upstream. Root-caused as follows:
- **BUG**: `pip install -e` to "restore" the dev clone did NOT switch the import — TWO
  `__editable__` finders coexisted in site-packages: `…dev380…` → `/sgl-workspace/sglang`
  (dev clone) and `…dev14280…` → `/sgl-workspace/sglang-upstream` (upstream). The
  upstream finder won, so **every run labelled "dev clone" was actually upstream**.
  Fix: `rm` the upstream `.pth` + `_finder.py` + `dist-info`; import then resolves to
  the dev clone. (Lesson: after `pip install -e`, ALWAYS verify `python -c "import
  sglang.…; print(.__file__)"` AND the server log's `Editable project location`.)
- After the fix, re-ran the **TRUE dev clone** 8k/1k c512 SE-local (full ALL3 env:
  FORCE_CK + ROPE_BATCHED + SE-local + TP1 + gatherv):

| config | c512 8k/1k total tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| dev clone **BLOCK_M=32** (= Exp 52 cfg) | 35,101 | 38,987 | 88.31 |
| dev clone **BLOCK_M=8** (current default) | 35,086 | 39,062 | 88.34 |
| upstream BLOCK_M=8 | 34,064 / 35,084 | — | — |
| Exp 52 logged (dev clone BM32, 6/18) | 36,031 | — | — |

### Conclusions
1. **rope BLOCK_M=8 vs 32 has ZERO c512 end-to-end effect** (35,086 vs 35,101, −0.04%,
   pure noise). As predicted — rope is a sub-ms prefill op, c512 is decode-bound. The
   BLOCK_M=8 default stays (it's a real prefill kernel win, Exp 53/microbench, and costs
   nothing at c512). User's BLOCK_M hypothesis = ruled out.
2. **upstream ≈ dev clone** today (both 34–35k). The earlier "dev clone +3%" claim was
   the finder bug (both were upstream); there is NO clone/version regression.
3. **36,031 (6/18) vs ~35,100 (today) = ~2.5% cross-day run-to-run variance** (c512
   high-conc single-run variance is 1–3%), not any code change.

### Artifacts
- upstream: `/sgl-workspace/{base_prefill,base_c512,se_prefill,se_c512,se_c512_8k,gsm8k_se_local}.log`
- dev clone: `/sgl-workspace/{D32_c512_8k,D8_c512_8k}.log`, servers `/sgl-workspace/dsv4_{D32,D8}.log`
- Final state: both clones rope = BLOCK_M=8 + num_warps=4; editable → dev clone
  (`/sgl-workspace/sglang`); servers stopped, VRAM ~0.3 GB/GPU.

---

## Exp 56 — C5: extend shared-expert-local to DECODE → fixes the c512 1k/1k regression (2026-06-21)

### Motivation
Exp 55 (plan B) showed SE-local (TP1 + se-local) REGRESSED c512 1k/1k by −3.4% while
helping prefill +6–7%. Root cause analysis: the −3.4% is NOT prefill FLOPs. Per-rank
shared-expert FLOPs are identical between the two schemes:
- normal (TP-sharded, global buffer): `M_global * dim/tp`
- SE-local (TP1, local tokens):        `M_local * dim = M_global/tp * dim`
The penalty came from SE-local being PREFILL-ONLY (`is_extend()` gate): in DECODE the
normal path ran with the **replicated (TP1) weights on the gathered global batch at full
dim = ~dp_size x** the sharded cost. c512 1k/1k is decode-bound, so that decode penalty
dominated.

### Change (one-liner: broaden the gate)
`deepseek_v4.py` `_do_shared_local`: drop `is_extend()`, switch `_use_gatherv_pair` →
`_use_tp_moe_gather`, so SE-local applies to BOTH prefill (gatherv/reduce_scatterv) and
decode (dp_scatter). The shared expert is a per-token MLP → computing on this rank's
local tokens ≡ computing on the global buffer then taking the local slice (gsm8k-verified).
The existing add-back (after the if/else) already covers both reduce_scatterv and
dp_scatter. Stable for CUDA graph (gate no longer depends on padding mode). Amended into
the SE-local commit `30fa179536`.

### Correctness
gsm8k 5-shot (C5, gatherv+TP1+se-local): **flex 0.9507 / strict 0.9515** — correct.

### c512 results (ratio 1.0, conc 512, same client, dev clone, ROCM700A=0, chunk 8192/rank)
| config | 1k/1k | vs base | 8k/1k | vs base |
|---|---:|---:|---:|---:|
| baseline (gatherv only) | 17,738 | — | 33,828 | — |
| SE-local prefill-only (Exp 55) | 17,136 | **−3.4%** | 34,064 | +0.7% |
| **SE-local C5 (prefill+decode)** | **17,954** | **+1.2%** | **35,403** | **+4.7%** |

Decode TPOT confirms the ~dp_size x removal:
- 1k/1k TPOT 48.08 → **46.54** ms; 8k/1k TPOT 79.10 → **73.69** ms.
vs ATOM (same-day): 1k/1k 91.2%, 8k/1k 90.7% of ATOM (was ~87%).

### Net
SE-local is now a positive lever at c512 for BOTH workloads (no 1k/1k regression). Still
env-gated (`SGLANG_DP_SHARED_EXPERT_LOCAL` + `SGLANG_SHARED_EXPERT_TP1`); TP1 weight-memory
cost (~+0.5 GB/rank) unchanged — C5 only removes the decode COMPUTE penalty, not the
replication memory.

### Artifacts
- `/sgl-workspace/{gsm8k_c5,c5_c512}.log`, server `/sgl-workspace/dsv4_c5.log`.
- Code: dev clone `deepseek_v4.py` (amended into commit `30fa179536`); server stopped,
  VRAM ~0.3 GB/GPU.

---

