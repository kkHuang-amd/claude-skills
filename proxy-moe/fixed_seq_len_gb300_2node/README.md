Owner node: pod4-gb300-2-tray05-f3

# Proxy MoE fixed_seq_len sweep, TP8 on 2x GB300 (MNNVL)

2-node counterpart of `../fixed_seq_len_gb300/` (tp4ep1, 1 tray). Recipe **tp8ep1, 24 layers**,
same shapes / CONC / client / server flags; only the topology differs (8 GPUs across 2 trays,
cross-tray NCCL over MNNVL).

## CONTINUE HERE

**Status (2026-10-21 11:15 UTC):** FULL SWEEP DONE, 25/25 points, report.py no warnings.
Tables (all 25 points + per-point MI355X comparison): `RESULTS.md` here (raw: `/shared_nfs/kk/results/proxy_24l/fixed_seq_len_gb300_2node/`).
Sweep 06:58-11:15 UTC (4h17m). Servers torn down; `_2node` containers left running (idle).
**Next:** none queued (tp8ep8 deliberately not run; ask user).
**Repro (on tray05 HOST, not in a container):**
```bash
G=/SFS-aGqda6ct/amd/kk/claude-skills/proxy-moe/fixed_seq_len_gb300_2node
R=/SFS-aGqda6ct/models/kk/results/proxy_24l
setsid nohup bash $G/smoke_2node.sh > $R/fixed_seq_len_gb300_2node_smoke/smoke.log 2>&1 < /dev/null &
setsid nohup bash $G/sweep_proxy_2node.sh > $R/fixed_seq_len_gb300_2node/sweep.log 2>&1 < /dev/null &
echo $! > $R/fixed_seq_len_gb300_2node/sweep.pid
```
Stop: `kill $(cat .../sweep.pid)`, then `source $G/lib_2node.sh; teardown_2node`.
**Progress:** `find $R/fixed_seq_len_gb300_2node -name DONE | wc -l` (of 25).
**Report:** `python3 ../fixed_seq_len/report.py $R/fixed_seq_len_gb300_2node --md`.
**Pass criteria:** smoke.json passed, NCCL uses MNNVL (not NET/Socket), every point DONE.

## Topology

| | node0 (rank 0) | node1 (rank 1) |
|---|---|---|
| host | pod4-gb300-2-tray05-f3 | pod4-gb300-2-tray01-f3 |
| IP (bond0.113) | 10.13.113.15 | 10.13.113.11 |
| container | kk_proxy-moe_gb300-2-tray05-f3_1021 | kk_proxy-moe_gb300-2-tray01-f3_1021 |
| role | HTTP :30000 (127.0.0.1), dist-init 10.13.113.15:5000, client, sweep driver (host) | worker |

Slurm reservations (user wunhuang): tray01 `gh-wunhuang-29a09c79` until 12:30 UTC 10-21,
tray05 `gh-wunhuang-c4d3b825` until 16:41 UTC 10-21. Both trays are in the same IMEX domain
(`/etc/nvidia-imex/nodes_config.cfg` lists 10.13.113.11-28).

The sweep driver runs on the tray05 host and reaches tray01 with `ssh` (key
`~/.ssh/id_ed25519` on tray05, added to tray01's authorized_keys) + `docker exec`.

## Environment

Image `lmsysorg/sglang:v0.5.22` (id fd8439732b16, same on both) = SGLang `baa9bef`, torch
2.14.1+cu130, FlashInfer 0.7.1; + `../patches/sglang_quark_mxfp4_cuda.patch` (identical to tray01's
working-tree diff). Containers: `--privileged --gpus all --network=host --ipc=host --shm-size 64G`,
`/SFS-aGqda6ct/amd/kk:/workspace`, `/SFS-aGqda6ct/models:/shared_nfs` (both Ceph, shared).
`--privileged` exposes `/dev/nvidia-caps-imex-channels` without `--device`.

Shared (Ceph): package `/workspace/models/proxy_moe_sglang/proxy_moe_sglang` (copied from tray01),
tokenizer `/shared_nfs/kk/proxy_moe/tokenizer`, ckpt `/shared_nfs/kk/proxy_moe/proxy_24l`
(`make_hf_config --num-layers 24 --num-gpus 8`, config sha256 9b60ba1c...bd7bd = reference,
`gen_dummy_ckpt --workers 32` seed 0, log `/shared_nfs/kk/proxy_moe/gen_24l.log`).
`/SFS-aGqda6ct/models/kk` is root-owned: create dirs from a container; results dirs are chowned
to uid 10003 so the host driver can write.

### 2-node gotchas

- Both trays must be in the same IMEX domain (`/etc/nvidia-imex/nodes_config.cfg`; host
  `systemctl is-active nvidia-imex` = active, `nvidia-imex-ctl -q` = READY).
- Containers need `--network=host` (bridge 172.17.x cannot reach the other tray) and the IMEX
  channel (`--privileged`, or `--device /dev/nvidia-caps-imex-channels`); without it NCCL fails with
  `Cuda failure 800 'operation not permitted'` / `MNNVL ... not working`.
- `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` = interface name `bond0.113` on both nodes;
  `--dist-init-addr` = node0 IP on both, only `--node-rank` differs.
- Do not work around MNNVL errors with `NCCL_MNNVL_ENABLE=0`: it falls back to TCP sockets and the
  numbers are meaningless. Check with `NCCL_DEBUG=INFO` that cross-tray channels say `via P2P/MNNVL`.
- `EADDRINUSE` on :5000 = leftover server: `ss -ltnp | grep :5000`, then `teardown_2node`.

### Node-local checkpoint and containers

Containers `kk_proxy-moe_gb300-2-tray0{5,1}-f3_1021_2node` (same flags as above, plus
`/mnt/local/home/wunhuang/proxy_moe:/local`), created by `/tmp/kk/mk2node.sh`-style commands: run the
image, `git apply` the patch, copy the non-weight files of the Ceph `proxy_24l`, then
`gen_dummy_ckpt.py --model-dir /local/proxy_24l --workers 32` (seed 0, byte-identical to Ceph copy).
tray01's original `kk_proxy-moe_gb300-2-tray01-f3_1021` (12L ckpt for tp4ep1) is left untouched.

## Deviations from the 1-tray sweep

- `--mem-fraction-static 0.85` (1-tray: 0.88). At 0.88, TP8 2-node OOMs in prefill CUDA graph capture:
  after the KV pool 32.9 GB is left (same as 1 tray), but PyTorch then holds 15 GB reserved-unallocated
  (fragmentation after FlashInfer autotune) and the 2-node MNNVL setup needs more. KV capacity is not
  binding (16.7M full-layer tokens at 0.88 vs. <= 9.4M needed at 8k/1k c1024).

## Smoke

Smoke #1 (2026-10-21 06:05, tray05+tray01, Ceph ckpt, mem 0.88): rendezvous OK, all 8 ranks up.
NCCL: `MNNVL 1`, cross-tray channels `3[3] -> 4[0] via P2P/MNNVL`, comm `nNodes 1 localRanks 8`
(the NVL domain is one NCCL node), no NET/Socket; FlashInfer AR fusion `backend=mnnvl, world_size=8`;
CustomAllreduce and multimem all-gather auto-disabled (TP group spans nodes). Weights 160.72 GB/GPU.
Then OOM as above. Logs: `.../fixed_seq_len_gb300_2node_smoke/smoke1_ceph_mem088/`.

Smoke #2 (2026-10-21 06:38, local ckpt, mem 0.85): load 103 s, first launch ready 374 s (incl. ~3 min
FlashInfer autotune, cached afterwards), later launches 167-302 s. KV pool 15.0M tokens, 22.6 GB free
after graph capture (1 tray: 21.7 GB). Package `smoke_test.py`: passed. Mini sweep (NP_MULT=2):

| Shape | CONC | TTT (8 GPU) | TTT/GPU | Median TTFT (ms) | Median ITL (ms) |
|---|---|---|---|---|---|
| 1024/1024 | 2 | 743 | 93 | 96 | 5.25 |
| 1024/1024 | 64 | 4,360 | 545 | 1,233 | 27.37 |
| 70000/300 | 2 | 29,837 | 3,730 | 2,088 | 5.32 |

## Results

Append-only; every row carries node + date. Full tables (E2EL, interactivity, P90) in `RESULTS.md` (this directory).

| Date | Nodes | Run | Shape | Peak TTT/GPU (CONC) | c2 TTFT / ITL med (ms) | Notes |
|---|---|---|---|---|---|---|
| 2026-10-21 | tray05+tray01 | `.../proxy_24l/fixed_seq_len_gb300_2node` | 1024/1024 | 2,123 (c512) | 97 / 5.24 | c1024 below c512 (ITL 103 ms) |
| 2026-10-21 | tray05+tray01 | same | 8192/1024 | 3,525 (c512) | 255 / 5.27 | c1024 3,423 |
| 2026-10-21 | tray05+tray01 | same | 70000/300 | 3,958 (c32) | 2,089 / 5.33 | |

vs 1-tray tp4ep1 12L (`../fixed_seq_len_gb300`): per-GPU work per step is the same, so with equal
step time TP8 TTT/GPU would be 0.5x TP4 (same tokens, 2x GPUs). Measured TP8/TP4 TTT/GPU geomean over
25 points: GB300 0.380, MI355X 0.403 (`../fixed_seq_len`), i.e. both platforms lose about the same
from TP4 12L to TP8 24L; GB300 2-tray only ~6% more. c2 ITL 5.24 vs 3.86 ms.

### Geomean MI355X / GB300, tp8ep1 (24 layers, 8 GPUs) — 2026-10-21, tray05

Same method as `../fixed_seq_len/RESULTS.md` (MI355X value / GB300 value per point, geomean;
TTT/GPU and P90 interactivity < 1 = MI355X behind; ITL / TTFT > 1 = MI355X slower). 25 matched
points, chunk 8192 both. Script: `cmp_tp8.py` here. Same precision / prefill-AR caveats as tp4ep1
(GB300 W4A8 routed experts, bf16 prefill AR; MI355X W4A4, INT8 quick-reduce prefill AR); GB300 here
additionally runs mem 0.85 and spans 2 trays over MNNVL.

| Shape / subset | points | TTT/GPU | P90 interactivity | Median ITL | Median TTFT |
|---|---|---|---|---|---|
| 1024/1024 | 10 | 0.947 | 0.941 | 1.040 | 1.032 |
| 8192/1024 | 10 | 0.940 | 0.960 | 1.051 | 1.084 |
| 70000/300 | 5 | 0.946 | 0.887 | 1.128 | 1.071 |
| all 3 shapes | 25 | 0.944 | 0.938 | 1.061 | 1.060 |
| 1024/1024, CONC <= 512 | 9 | 0.930 | 0.923 | 1.083 | 1.033 |
| 8192/1024, CONC <= 512 | 9 | 0.928 | 0.932 | 1.096 | 1.094 |
| all, CONC <= 512 | 23 | 0.933 | 0.919 | 1.097 | 1.065 |

- GB300 lead at TP8 (0.933 at CONC <= 512) is smaller than at tp4ep1 (0.874).
- Gap is largest at low CONC (c2 0.83-0.92) and closes with CONC; ~parity at c512 (0.99-1.00).
- CONC=1024 MI355X ahead (1.117 / 1.059): GB300 ITL 103 ms vs 74 ms, same open item as tp4ep1.
