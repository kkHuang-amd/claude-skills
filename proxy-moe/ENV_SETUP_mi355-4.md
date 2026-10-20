Owner node: mi355-4

# Proxy MoE — environment setup on mi355-4

## CONTINUE HERE

**Status:** env DONE — baseline TP8/EP8 24L passes (smoke + bench), ~4% above MI350X reference.
**Next:** none here — long-smoke slowness confirmed as cold JIT (warm 0.48 s). Optimization
continues in `OPT_mi355-4.md`.
**Files:** package `/workspace/models/proxy_moe_sglang/proxy_moe_sglang/`; model
`/shared_nfs/kk/proxy_moe/proxy_24l`; AITER patch target `/sgl-workspace/aiter/aiter/ops/quant.py`.
**Repro (inside this container, NOT from /sgl-workspace):**
```bash
PKG=/workspace/models/proxy_moe_sglang/proxy_moe_sglang; W=/shared_nfs/kk/proxy_moe
TAG=baseline_$(date -u +%Y%m%d_%H%M); ART=/shared_nfs/kk/results/proxy_24l/$TAG
cd /tmp && NUM_LAYERS=24 TP=8 EP=8 SKIP_GEN=1 AITER_LARGE_HIDDEN_WORKAROUND=1 \
  TOKENIZER_DIR=$W/tokenizer MODEL_DIR=$W/proxy_24l ART=$ART \
  bash $PKG/scripts/run_stage.sh > $ART.stage.log 2>&1
python3 $PKG/tools/summarize_run.py $ART --out $ART/summary.json
```
**Pass criteria:** `smoke.json` `"passed": true`; bench 64/64. MI350X reference (8 GPU, 24L):
output 753 tok/s, median TTFT 461 ms, median TPOT 19.3 ms, weights 160.7 GiB/GPU.

## Environment (mi355-4, 2026-10-20)

- Already inside a container (no docker); 8x MI355X (gfx950), all idle at start.
- SGLang `ed0b8f433f` (0.5.22.dev20261019, `/sgl-workspace/sglang`, has `models/laguna.py`);
  AITER `0.1.25.dev178+g2bb549102` (`/sgl-workspace/aiter`, source install); torch
  `2.11.0+rocm10.0.0`. Newer than the package's pinned image (SGLang `e00930c548`, AITER 0.1.22,
  ROCm 7.2) — expect differences vs reference.
- Disk: `/workspace` (= `/`) has only ~25 GB free → work dir is `/shared_nfs/kk/proxy_moe`
  (`/dev/md0`, local xfs RAID, 45 TB free). 24L ckpt = 1.36 TB.
- HF reachable; tokenizer hashes match reference.

## Steps done

1. Tokenizer → `/shared_nfs/kk/proxy_moe/tokenizer` (3 files, sha256 match).
2. `make_hf_config.py --num-layers 24 --num-gpus 8` → `proxy_24l/config.json`, sha256
   `9b60ba1c…bd7bd` = reference.
3. `gen_dummy_ckpt.py --workers 32` (seed 0) → log `/shared_nfs/kk/proxy_moe/gen_24l.log`.
4. AITER known issue 1 still present in AITER 2bb549102; applied
   `patches/aiter_mx_quant_large_hidden.py` → `/sgl-workspace/aiter/aiter/ops/quant.py`
   (uncommitted edit in the AITER tree; marker `# proxy_moe_sglang: fused kernel supports cols <= 8192`).

## Results

Append-only; every row carries node + date.

| Date | Node | Run tag | Config | Out tok/s | TTFT med ms | TPOT med ms | Notes |
|------|------|---------|--------|-----------|-------------|-------------|-------|
| (ref) | MI350X | expected_results.md | TP8/EP8 24L | 753 | 461 | 19.3 | pinned image |
| 2026-10-20 | mi355-4 | `/shared_nfs/kk/results/proxy_24l/baseline_20261020_0146` | TP8/EP8 24L, defaults | 781 | 467 | 19.0 | smoke pass; ready 101 s, load 11.3 s, weights 160.8 GiB/GPU, peak HBM 225.7 GiB; long smoke 28.8 s (ref 11.5) |

## Notes / gotchas

- Run from a directory other than `/sgl-workspace` (it shadows `sglang.benchmark`).
- `--mem-fraction-static 0.88` is lowered by SGLang to 0.748 (aiter backend, ctx > 8192); expected.
