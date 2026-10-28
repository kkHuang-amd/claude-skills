Owner node: ctheliosp-zuse-f19-14

# Prefill MoE optimization, gfx1250, TP2 4L shapes (2026-10-28)

Continues the last sections of `TP2_4L_f19-14_1027.md` (WMMA peak, prefill MoE process, tile sweep status).
No other node is available, so f19-14 keeps the work; the handoff block in that doc is superseded by this doc.

## CONTINUE HERE

**Status (2026-10-28, f19-14):** tile A/B on one GPU: stage1 128/256/128/4 + stage2 128/256/256/2 = 14.40 ms vs default 16.47 ms (-12.6%), noise < 0.3%. rocprofv3 works again (sitecustomize fix, no venv change).
**Next:** (1) stage2 sweep around 128/256/256 with stage1 fixed at 128/256/128/4, single GPU (2 or 3) only; (2) rocprofv3 counters on the winner vs default; (3) ISA resource check; (4) server validation (8k prefill TTFT) at serving clocks.
**GPUs:** use GPU 2 and 3 only on f19-14 (other GPUs may be used by others). Never run timing on several GPUs at once.
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
