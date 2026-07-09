# Per-node results — gfx950 reference node (8x MI355X, TP2)

Raw run log for the gfx950 cross-validation box only. Write new runs here first, then promote
a short summary into the shared `EXPERIMENT_LOG.md` / `STATUS.md`.

Tree: `/sgl-workspace/sglang_gfx-1250`. This node's shared-log entries so far: **E26** (absorb
ablation = neutral), **E27/E28** (cross-node dump + partial diff), **E30** (bf16 MoE emul: a8w4
scheme neutral), **E33** (chunked-emul control = 0.950). Native a4w4 GSM8K reference ~0.93-0.945.
See `EXPERIMENT_LOG.md`.

---

## (append new gfx950 runs below)

## E42 (2026-07-09) bf16 matmul microbench (matmul_prec.py, HIP_VISIBLE_DEVICES=1)
gfx950 bf16 vs fp64: K256=2.872e-3, K512=2.828e-3, K2048=2.892e-3 (fp32 ~1-4e-7).
same-vals bf16: ~1.65e-3 == gfx1250 (E41) 1.65e-3. => gfx950 bf16 matmul NOT more accurate;
identical to gfx1250. E41 hypothesis REFUTED (bf16 output-rounding dominated, arch-independent).
