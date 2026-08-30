# Per-node results — gfx1250 @ ctheliosr-rck-g02-j19-10 (4x gfx1250, primary)

Raw run log for THIS node only. Write new runs here first (freely, no merge risk), then
promote a 1-3 line summary into the shared `EXPERIMENT_LOG.md` / `STATUS.md`. This keeps
concurrent multi-node pushes from clobbering the shared files.

Env: sglang `000a61a2`, aiter `8815f4b5` (bisect fix REVERTED, E38), model
`/shared_nfs/huggingface_models/amd/DeepSeek-R1-0528-MXFP4`. Launch scripts: `run_ds-r1.sh` (serve),
`run_ds-r1_dump.sh` / `run_ds-r1_emuldump.sh` (per-layer dump), `run_ds-r1_nonmoeprobe.sh`
(non-MoE probe).

This node's shared-log entries so far: **E31** (token-sweep a8w4 bisect), **E32** (bf16 MoE
emul summary), **E34** (non-MoE per-op probe), **E36** (DSv4 0.925), **E37/E38** (DSv4 crash
= our bisect fix, reverted). See `EXPERIMENT_LOG.md` for the write-ups.

---

## (append new j19-10 runs below)
