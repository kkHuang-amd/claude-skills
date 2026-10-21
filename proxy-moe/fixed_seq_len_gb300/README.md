Owner node: pod4-gb300-2-tray01-f3

# Proxy MoE fixed_seq_len sweep on GB300

GB300 counterpart of `../fixed_seq_len/` (MI355X). Same shapes, CONC lists, client settings,
one server per point, DONE markers, same `report.py`. Platform differences are listed below.

## CONTINUE HERE

**Status (2026-10-21 05:40 UTC):** FULL SWEEP DONE, 25/25, no warnings → `RESULTS.md`.
**Next:** investigate CONC=1024 ITL doubling (see RESULTS.md notes); compare with MI355X
`../fixed_seq_len` once its RESULTS.md lands. Regenerate the report with
`python3 ../fixed_seq_len/report.py /shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300 --md` → `RESULTS.md`.
**Repro (full sweep):**
```bash
setsid nohup bash /workspace/claude-skills/proxy-moe/fixed_seq_len_gb300/sweep_proxy.sh \
  > /shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300/sweep.log 2>&1 < /dev/null &
echo $! > /shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300/sweep.pid
```
Use `setsid`: on this node, background jobs started from an agent shell call die when the call ends.
**Progress:** `find /shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300 -name DONE | wc -l` (of 25).
**Pass criteria:** every point DONE; report.py prints no WARNINGS.

## Environment (pod4-gb300-2-tray01-f3)

4x NVIDIA GB300 (SM 10.3, 284 GB each), driver 595.58.03, aarch64. SGLang 0.5.22 (`baa9bef`,
`/sgl-workspace/sglang`) + `../patches/sglang_quark_mxfp4_cuda.patch`, FlashInfer 0.7.1,
torch 2.14.1+cu130.

Node-local (container disk, not shared): package `/sgl-workspace/proxy_moe_sglang/proxy_moe_sglang`,
work dir `/sgl-workspace/proxy_moe_work` (`proxy_12l/` checkpoint, `tokenizer/`, gen log).
Rebuild if lost (~11 min with 13 workers, ~1.1 GB/s):
```bash
PKG=/sgl-workspace/proxy_moe_sglang/proxy_moe_sglang; W=/sgl-workspace/proxy_moe_work
# tokenizer: openai/gpt-oss-120b @ b5c939de8f754692c1647ca79fbf85e8c1e70f8a (3 files) into $W/tokenizer
python3 $PKG/tools/make_hf_config.py --spec $PKG/spec/proxy_spec.json --num-layers 12 \
  --out-dir $W/proxy_12l --tokenizer-dir $W/tokenizer --num-gpus 4
cd $PKG/tools && setsid nohup python3 gen_dummy_ckpt.py --model-dir $W/proxy_12l --workers 13 \
  > $W/gen_12l.log 2>&1 < /dev/null &
```
Generated config.json is identical to `configs/4gpu_12l/config.json`; manifest: seed 0, 13 shards,
685,748,582,400 bytes.

## Platform mapping vs MI355X

| Item | MI355X (`../fixed_seq_len`) | GB300 (this dir) |
|---|---|---|
| Recipes | tp8ep1 (24L, 8 GPU), tp4ep1 (12L, 4 GPU) | tp4ep1 only: one tray = 4 GPUs; 24L (1.36 TB) does not fit in 4x284 GB |
| Checkpoint / config | Quark MXFP4, seed 0 | same files, unchanged |
| Routed experts | AITER fused MoE, MXFP4 W4A4 | FlashInfer trtllm-gen MoE (`--moe-runner-backend flashinfer_mxfp4`), MXFP4 weights x MXFP8 activations (W4A8; no W4A4 MoE kernel on this stack) |
| Shared expert | AITER `gemm_afp4wfp4`, MXFP4 W4A4 | FlashInfer `mm_fp4` (cuDNN), MXFP4 W4A4 |
| AR fusion | `--enable-aiter-allreduce-fusion` | `--enable-flashinfer-allreduce-fusion` |
| Other server flags | ctx 73728, chunk 8192, mem 0.88, fp8 KV, max-running = cuda-graph-max-bs-decode = CONC | same |

User decisions 2026-10-21: W4A8 for routed experts; shared expert handled as on MI355X (MXFP4);
tp4ep1 only.

### SGLang patch (`../patches/sglang_quark_mxfp4_cuda.patch`)

Upstream Quark MXFP4 MoE raises `NotImplementedError` without the AITER runner. On non-HIP:
- `QuarkW4A4MXFp4MoE` delegates to `Mxfp4MoEMethod` (same `w13/w2_weight[_scale]` params),
  sets `gate_up_interleaved=False` (Quark stores gate and up as row blocks; the mxfp4 path
  defaults to the GPT-OSS interleaved layout) and clears `gemm1_alpha/beta/clamp_limit`
  (plain SiLU-gated, not the GPT-OSS swiglu 1.702 / clamp 7 default).
- `QuarkW4A4MXFP4` linear: weight scale → 128x4 swizzle (`block_scale_interleave`, 2D view),
  activations `mxfp4_quantize`, GEMM `mm_fp4(block_size=32, use_nvfp4=False, backend="cudnn")`
  (cutlass / trtllm backends reject MXFP4). Unit test with exactly representable inputs:
  relL2 0.0017 (= bf16 output rounding) for M in {1, 37, 300, 8192}.

## Smoke (2026-10-21, pod4-gb300-2-tray01-f3)

Server tp4ep1: weights 163.3 GB/GPU, load 62 s, ready ~240 s, max_total_num_tokens 16.08M at
CONC=2. Package `smoke_test.py`: passed. Output `/shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300_smoke/`.
Mini sweep (NP_MULT=2, so "completed" warnings are expected); launch time 240-330 s per point:

| Shape | CONC | TTT/GPU | Median TTFT (ms) | Median ITL (ms) |
|---|---|---|---|---|
| 1024/1024 | 2 | 252 | 81 | 3.85 |
| 1024/1024 | 64 | 1,301 | 786 | 23.30 |
| 70000/300 | 2 | 10,913 | 1,380 | 3.89 |

MI355X tp4ep1 1024/1024 c2 smoke for reference: 217.9 TTT/GPU, TTFT 86.6 ms, ITL 4.49 ms.
