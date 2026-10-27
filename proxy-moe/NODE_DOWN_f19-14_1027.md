Owner node: ctheliosp-zuse-f19-14

# f19-14 node down at first TP8 server start after the RCCL DDA fix swap (2026-10-27)

## CONTINUE HERE

**Status (2026-10-27 03:05, f19-14):** NOT confirmed either way. Timing coincides with distributed init (where the DDA fabric init runs), but the crash signature (hard reset, IO bus MCEs) points at the platform, and the code delta of the installed lib over the version validated on h13-17 is host-side only. GPUs are not usable on this node right now: `amdgpu` is blacklisted on the kernel cmdline and has not been loaded since the reboot, so the container has no `/dev/kfd` / `/dev/dri`.
**Next:** (1) get the host / BMC side: SEL + serial console (ttyS1) around 2026-10-27 01:52 UTC, kdump under the host's `/var/crash` (crashkernel=128M is set); (2) once amdgpu is loaded, check `amd-smi ras` / bad-page / ECC counters; (3) only with user OK (each try may take the node down again), A/B in this order: stock 2.30.7 server start -> fix lib with `RCCL_DDA_ENABLE=0` -> fix lib with DDA on.
**Files:** `/workspace/proxy_work/rccl_swap.sh` (`restore` = stock 2.30.7), fix lib `/workspace/proxy_work/rccl_16ded0c/librccl.so.1.0`, crashed session transcript `/root/.claude/projects/-sgl-workspace/2623c8a9-*.jsonl`.
**Repro (the command that preceded the crash):**
```bash
export NCCL_CUMEM_ENABLE=1 SGLANG_OPT_LAGUNA_ROUTER_FLYDSL=1 AITER_CUSTOM_AR_MAX_SIZE=1048576 SMOKE=1 EP=8 PKG=/workspace/proxy_moe_sglang/proxy_moe_sglang
bash /workspace/claude-skills/proxy-moe/scripts/mi455_point.sh 1k1k 2 <run-tag>
```
**Pass criteria:** to blame RCCL: node survives N starts on stock / DDA-off and dies on the fix lib with DDA on. To clear it: node survives N starts on the fix lib with DDA on.

## Timeline (UTC, f19-14)

| Time | Event |
|---|---|
| 01:48:57 | shallow clone of `users/wenkaidu/dda-fabric-init-memset-race`; tip is now **`16ded0c`**, not `9c52e34` (the one A/B-tested on h13-17) |
| 01:51:18 | `rccl_swap.sh install` of 16ded0c into the venv SDK (both `_rocm_sdk_libraries` and `_rocm_sdk_devel`) |
| 01:51:40 | TP8 server start, `NCCL_CUMEM_ENABLE=1`, **DDA on** (no `RCCL_DDA_ENABLE=0`), EP=8 |
| 01:52:06.98 | last write that survived: `custom_all_reduce*.pyc` + comgr cache, i.e. workers in distributed init |
| ~01:52 | node down. Transcript ends in NUL bytes, server log 0 bytes (NFS) -> hard reset, dirty pages lost, no clean shutdown |
| 02:48:41 | host boot (about 57 min later) |
| +333 s | first MCE poll: 7 corrected errors, banks MC26/MC27/MC60, 4 of them `Over` + `UECC`, all `mem/io: IO`, bus-error class (`0x080b`), same syndrome; possibly latched before the reset |

## Why RCCL is not the likely cause

- 16ded0c vs 9c52e34 (`fabric_init.cu` only, +37/-23): replaces the `bootstrapBarrier` after the zero fills with a `bootstrapAllGather` of a per-rank "ready" flag so a local failure falls back instead of blocking peers. Same memsets and same sync; host-side control flow only.
- 9c52e34 ran 750 comm-init iterations at 4 ranks on h13-17 without incident; the unpatched base hangs GPU waves (rocgdb: spinning in `ddaAllReduceFlatLL`) but never took the node down.
- A userspace bug in a collective normally gives a GPU hang or page fault, not a host hard reset. IO bus-error MCEs with overflow are a platform / PCIe / fabric signature.

## Why it is not excluded

- The crash falls in the window where `ncclCommInit` and `ncclDdaFabricCommInit` run, and DDA was on. This was also the first GPU workload on this node after the rebuild, so any GPU load would have hit the same window.
- The exact crash second is unknown (writes after 01:52:07 may have been lost), so this cannot pin it to RCCL init vs model load.
- TP8 (not TP4) DDA init with the fix was never run before, on any node.
