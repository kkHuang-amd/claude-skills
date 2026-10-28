Owner node: ctheliosp-zuse-f19-14

# Prefill MoE optimization, gfx1250, TP2 4L shapes (2026-10-28)

Continues the last sections of `TP2_4L_f19-14_1027.md` (WMMA peak, prefill MoE process, tile sweep status).
No other node is available, so f19-14 keeps the work; the handoff block in that doc is superseded by this doc.

## CONTINUE HERE

### HANDOFF (2026-10-28 ~00:50 UTC; f19-14 stops, work moves to another node)

- **Closed for f19-14.** The next node opens its own doc (e.g. `proxy-moe/MOE_PREFILL_<host>_<date>.md`, `Owner node:` on top) and cites
  this one. Do not act on f19-14 GPU / PID / incident state from here.
- **Result to carry over:** grouped a8w4 MoE at T=8192 (one EP2 rank: E=256, topk 8, H 16384, I 4096, no bias), single-GPU interleaved:
  stage1 `AITER_TDM_TILE_M/N/K/NUM_BUFFERS=128/256/128/4` + stage2 `..._M2/N2/K2/NUM_BUFFERS2=128/256/256/2` = 14.40 ms vs kernel
  defaults (64/256/128/3 both stages) 16.46 ms, -12.6%, noise < 0.3%. Stage2 128/256/128/3 is within 0.6%. Measured on f19-14 only, idle clocks.
- **Not done yet:** (1) reproduce the A/B on the new node; (2) ISA resource check of the two winning kernels (VGPR / scratch / LDS from
  the `counter_collection.csv` columns, or `FLYDSL_DUMP_IR`); (3) make it the default without a TP2-specific CSV (the generic
  `no CSV matched` fallback in `grouped_moe_gfx1250.py`, or a shape rule), then server 8k1k TTFT A/B at serving clocks + GSM8K/smoke;
  (4) HBM bytes / L2 hit per kernel (FETCH_SIZE and GL2C in separate passes).
- **Tool facts:** rocprofv3 on torch scripts needs `PYTHONPATH=proxy-moe/scripts/rocprof_env_fix` (fixed sitecustomize); on gfx1250 the
  instruction-level SQ counters read 0, so WMMA / wait attribution is not available from counters; ATT hung once here (unconfirmed cause).
- **Files:** scripts copied to `/shared_nfs/kk/proxy_work_f19-14_1027/` (`moe_ab_1028.sh`, `moe_s2_sweep_1028.sh`, `moe_counters_1028.sh`,
  `bench/moe_prefill_bench.py`); they hard-code `/workspace/proxy_work` paths and GPU defaults -> adjust. Sweep script:
  `proxy-moe/scripts/gfx1250_moe_tile_sweep.sh`. Results: `/shared_nfs/kk/results/proxy_4l/moe_prefill_{ab,s2}_f19-14_1028/`.

**Status (2026-10-28 00:45 UTC, f19-14): STOPPED -- GPU 2/3 physical device (0002:04:00.0) SMU not responding** (see "Node incident" at the end).
Best tiles so far: stage1 128/256/128/4 + stage2 128/256/256/2 = 14.38-14.43 ms vs default 16.44-16.49 (-12.6%), confirmed by the stage2 sweep.
**Next:** user decides on node recovery (GPU reset / reboot by admins). After recovery: verify GPU 2/3 with a short known-good run before any timing;
then counter pass (FETCH_SIZE / GL2C only, see below) on default vs best, ISA check from the kernel trace (VGPR / scratch), server 8k TTFT.
Do NOT run rocprofv3 --att on this node until the incident is understood.
**GPUs:** use GPU 2 and 3 only on f19-14 (other GPUs may be used by others). Never run timing on several GPUs at once,
and not even profiling on the sibling partition: GPU 2 and 3 share one physical device (0002:04:00.0).
**Files:** `proxy-moe/scripts/gfx1250_moe_tile_sweep.sh`, `/sgl-workspace/aiter/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py`,
standalone bench `/workspace/proxy_work/bench/moe_prefill_bench.py`. aiter `kk/feat/proxy-model-gfx1250-1023` @ `7ea6f2f46`.
**Repro (one point, server-path shape of one EP2 rank):**
```bash
GPU=4 STAGE=0 E=256 TOPK=8 INTER=4096 bash /workspace/claude-skills/proxy-moe/scripts/gfx1250_moe_tile_sweep.sh \
  /shared_nfs/kk/results/proxy_4l/<run-tag> 8192 a8w4
```
**Pass criteria:** rc=0, check `pass=True`; a tile is a winner only if it beats default by more than the run-to-run noise
(measured from 3 default repeats), passes the ISA resource check (no scratch), and holds at serving clocks.

## Shape

One EP2-rank equivalent: E=256 local experts, 8 local routes/token, H 16384, I 4096, a8w4 (fp8 act x fp4 weight), T=8192, no bias
(server path: fused stage1 act + MX quant epilogue, needs `(tile_n / n_warp) % 64 == 0`). Server default for this shape:
no CSV row matches, so both stages run the kernel defaults 64/256/128, num_buffers 3 (the log line says num_buffers=2, but that value is not passed).

## Results

### Tile sweep, remaining cases (2026-10-28, f19-14; `moe_prefill_tiles_f19-14_1028/`, GPUs 4/5/6 concurrently)

end-to-end us, `test_flydsl_grouped_gemm.py`. default (no env) x3: 23455 / 21439 / 21763. stage1 `m128_n256_k256` b2/b3/b4: 21537 / 23887 /
23875. stage2 `m128_n128_k256` b3/b4: 18002 / 21518; `m128_n256_k128` b2/b3/b4: 17066 / 16507 / 16551; `m128_n256_k256` b2/b3/b4: 16366 / 18122 / 18055.
From the 1027 logs: stage2 `m128_n128_k256_b2` 17631 (missing from summary.txt).

**The sweep cannot rank tiles.** With no CSV and no env, the kernel defaults are 64/256/128, num_buffers 3 for BOTH stages
(`grouped_moe_gfx1250.py` signature; the logged "using defaults ... num_buffers=2" is never passed to the kernel when no CSV row matches).
That is exactly the config the 1027 sweeps pinned the other stage to, and the 1027 stage2 "winner" `m64_n256_k128_b3` (16149 us) IS the
default, yet the default ran 21.4-23.5 ms today. Same-config spread is ~9% within one day and ~35% across days (3 GPUs at once,
PPT-capped clocks). Next: interleaved single-GPU A/B (`/workspace/proxy_work/moe_ab_1028.sh`, GPU 2, 3 rounds) ->
`moe_prefill_ab_f19-14_1028/`.

### rocprofv3 aborts on any torch process (2026-10-28, f19-14) -- FIXED (see the fix note below)

`rocprofv3 ... -- python3 -c 'import torch'` aborts: `CommandLine Error: Option 'spirv-expand-step' registered more than once!`
(`print(1)` alone works). Two byte-identical `libamd_comgr.so.3` (md5 e640743a...) are loaded: `_rocm_sdk_core/lib` (torch) and
`_rocm_sdk_devel/lib` (the rocprofv3 tool, via RPATH `$ORIGIN/..`, so `LD_LIBRARY_PATH` cannot override it). `LD_PRELOAD` of the devel copy:
same error; `LD_PRELOAD` of the core copy: `rocprofiler_at_intercept_table_registration ... error code 16` (configuration outside valid period).
`/opt/rocm` is a symlink to `_rocm_sdk_devel`, so there is no other rocprofv3. Proposed fix (venv change, reversible): replace
`_rocm_sdk_devel/lib/libamd_comgr.so.3` with a symlink to the core copy (keep a backup).

Fix (works, venv unchanged): `LD_DEBUG=files` showed torch's `torch/_rocm_init.py` -> `rocm_sdk.initialize_process(preload_shortnames=[
'amd_comgr', ..., 'rocprofiler-sdk', 'roctx64', ...])` ctypes-loads `_rocm_sdk_core` copies by full path while the tool already mapped the
`_rocm_sdk_devel` copies, so comgr AND librocprofiler-sdk load twice (the symlink of comgr alone only moved the abort to "error code 16").
`proxy-moe/scripts/rocprof_env_fix/sitecustomize.py` now skips preloading any shortname already mapped when running under rocprofv3.
Use: `PYTHONPATH=/workspace/claude-skills/proxy-moe/scripts/rocprof_env_fix rocprofv3 ... -- python3 <script>`. The trial comgr symlink was
reverted (original file back, md5 e640743a...; backup `/workspace/proxy_work/comgr_backup_1028/`). Not an image change: this preload code dates
from the image build (10-20), so profiling torch scripts with rocprofv3 never worked on this image without the fix.

### Interleaved single-GPU tile A/B (2026-10-28, f19-14, GPU 2; `moe_prefill_ab_f19-14_1028/`, `/workspace/proxy_work/moe_ab_1028.sh`)

end-to-end us, 3 interleaved rounds, all `pass=True`:

| Config (stage1 M/N/K/NB ; stage2) | r1 | r2 | r3 | vs def |
|---|---|---|---|---|
| def (no env; 64/256/128/3 both) | 16442 | 16469 | 16489 | - |
| env_def (same values via env) | 16451 | 16459 | 16450 | 0.0% |
| s1best: 128/256/128/4 ; 64/256/128/3 | 15721 | 15747 | 15746 | -4.4% |
| both: 128/256/128/4 ; 128/256/256/2 | 14397 | 14407 | 14382 | -12.6% |

Run-to-run spread < 0.3% on one GPU running alone, vs 9-35% for the 3-GPU concurrent sweeps (those numbers are void for ranking).
The env path matches the no-env path, so env tiles are a valid way to test. Most of the gain is in stage2 (down, K=4096): -8.5% on top of s1best.

### Stage2 sweep with stage1 fixed at 128/256/128/4 (2026-10-28 00:1x-00:29 UTC, f19-14, GPU 2; `moe_prefill_s2_f19-14_1028/`)

Controls before: def 16451, env_def 16452, s1best 15727, both 14383 (matches the A/B within 0.1%). stage2 (M2/K2/NB2, N2=256), us:
m64 k128 b2/b3/b4 15789 / 15744 / 15740; m64 k256 16426 / 16355 / 18218; m128 k128 15089 / 14521 / 14622; **m128 k256 b2 14430** / 16165 / 16093.
Winner unchanged: stage2 128/256/256/2 (= "both"); 128/256/128/3 is within 0.6%. Controls AFTER the sweep are void (see incident):
def 16482 normal, env_def 18933 (+15%, GPU 3 counter tests were running on the sibling partition), s1best hung (rc=124).

### rocprofv3 counters on gfx1250: instruction-level SQ counters read 0 (2026-10-28, f19-14, GPU 3, bf16 4096^3 hipBLASLt GEMM)

Non-zero: SQ_WAVES, SQ_BUSY_CYCLES, GRBM_COUNT, GRBM_GUI_ACTIVE, FETCH_SIZE, GL2C_HIT / GL2C_MISS / GL2C_EA_RDREQ.
Always 0: SQ_WAVE_CYCLES, SQ_INSTS_VALU, SQ_INSTS_LDS, SQ_WAIT_ANY, SQ_WAIT_INST_ANY, SQ_INST_CYCLES_{VALU_WMMA,VMEM_LOAD,LDS},
SQ_INSTS_VEC32_VALU_WMMA. FETCH_SIZE + GL2C in one pass -> "error code 38: Request exceeds the capabilities of the hardware".
So WMMA busy / wait attribution is NOT available from counters on this stack; only HBM bytes, L2 hit rate and resources
(VGPR / SGPR / scratch / LDS columns of counter_collection.csv). Summarizer: `proxy-moe/scripts/rocprof_counter_summary.py <pass dirs>`.
ATT (`rocprofv3 --att`, GPU 3, 00:30:15) produced no output at all and was killed by `timeout` at 300 s.

### Node incident: SMU of 0002:04:00.0 (GPU 2 + GPU 3) stopped responding (2026-10-28, f19-14)

- 00:28:37-00:29:43 rocprofv3 counter tests on GPU 3; GPU 2 ran the sweep controls. 00:29:47 env_def ran 15% slow.
- ~00:30:00 the GPU 2 test (s1best) stopped writing and hung until its 600 s timeout. 00:30:15 ATT test started on GPU 3 (after the hang began).
- By 00:31 `amdgpu_dpm_get_gpu_metrics` -> `smu_v15_0_8_get_metrics_table_internal` blocked; 00:33:04 hung-task warnings, including
  other users' processes (podman, grpcpp_sync_ser, sysstat sa1). 00:40:32 `amdgpu 0002:04:00.0: SMU: No response msg_reg: 7 resp_reg: 0`,
  `Failed to export SMU metrics table!`. `amd-smi metric` calls hang in D state (cannot be killed).
- Context: ALL four devices log correctable `pcie_pl` hardware errors every ~5.5 min (0002 at 112 -> 133 -> 156 by 00:22, before any profiling;
  0001 139, 0003 88, 0004 107), consistent with the IO-bus MCEs in `NODE_DOWN_f19-14_1027.md`. Cause of the SMU hang NOT established:
  candidates are the platform PCIe issue, rocprofv3 counter collection on the sibling partition, or the workload; ATT started after the hang.
- Actions: sweep processes killed (00:43). No further GPU commands, no amd-smi. Mapping: KFD domain 2 = 0002:04:00.0 = GPU 2 (1024) + GPU 3 (1025).
