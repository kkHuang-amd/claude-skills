---
name: aiter-custom-allreduce-nan-crash
description: Debug intermittent GPU ASSERT_TRAP / NaN crashes in SGLang P/D-disaggregated decode on AMD ROCm caused by aiter's custom all-reduce. Use when decode dies with HSA_STATUS_ERROR_EXCEPTION 0x1016 from at::native::_assert_async_cuda_kernel during sampling, or when an all-reduce is suspected of producing NaN under load. Captures the full bisection methodology, instrumentation pitfalls, aiter AR kernel anatomy, JIT-rebuild + bind-mount workflow, and the root cause (missing end_sync in cross_device_reduce_1stage, fixed by aiter PR #3514).
---

# Debugging aiter custom all-reduce NaN / GPU ASSERT_TRAP in SGLang disagg decode

A reusable playbook from a real investigation: DeepSeek-R1-0528-MXFP4, TP=8,
1P1D disaggregated, MI355X (gfx950), SGLang + aiter. Decode intermittently
crashed with a GPU hardware exception. Root cause turned out to be a missing
exit barrier in aiter's `cross_device_reduce_1stage` all-reduce kernel
(upstream **ROCm/aiter PR #3514**, issue #3515).

## 1. Recognize the signature

- Decode scheduler dies with: `HSA_STATUS_ERROR_EXCEPTION` code `0x1016`,
  `Fatal Python error: Aborted`, subprocess `crashed with exit code -6` (SIGABRT).
- rocm-debug-agent shows faulting kernel
  `at::native::_assert_async_cuda_kernel<bool>(bool const*, at::native::Msg)`,
  reason `ASSERT_TRAP`, `trapsts=0x80000000`, wavefront registers full of
  `0xffc00000` (= bf16/fp32 NaN bit pattern).
- Python traceback surfaces (async) at a sync point, e.g.
  `event_loop_overlap_disagg_decode -> process_batch_result_decode ->
  result.copy_done.synchronize()` — that is where it is *reported*, not where the
  bad kernel launched.
- **Sampling-dependent:** `temperature>0` (multinomial) crashes; greedy
  (`argmax`) does NOT crash but silently emits garbage tokens. This is a strong
  tell that a NaN reached the sampler.
- Intermittent: reproduces after a handful of benchmark runs, not deterministically.

## 2. Reproduce reliably (repeat loop)

Run the real agentic workload in a loop and auto-detect the crash by polling each
worker's `/health`; stop the moment a worker stops answering. Key knobs:
`CONCURRENCY`, `DURATION`, skip re-installing pip deps each iteration (the
agentic-benchmark install can git-clone transformers and dominate runtime), and
pin node IPs. Crash typically lands within ~3-6 short runs.

## 3. Bisect with a variant matrix (most important step)

Run the repeat loop under each config; this localizes the layer fast:

| Variant | How | Meaning if it STOPS crashing |
|---|---|---|
| Disable custom AR entirely | `--disable-custom-all-reduce` (uses RCCL) | bug is in custom AR (any impl) |
| Use sglang-native custom AR | `SGLANG_USE_AITER_AR=0` | bug is specifically in **aiter** AR |
| Toggle fused AR+RMSNorm | `--enable-aiter-allreduce-fusion` on/off | rules fusion in/out |
| Toggle CUDA graph | `--disable-cuda-graph` | tells you if it's graph-triggered |

In the case study: both `SGLANG_USE_AITER_AR=0` and `--disable-custom-all-reduce`
were 100% stable (12/12 runs); fusion on/off both crashed identically. => the bug
is aiter's custom all-reduce, independent of fusion.

## 4. Capture GPU faults (core dumps + rocm-debug-agent)

- Launch the decode container with `--ulimit core=-1` and `HSA_TOOLS_LIB=
  /opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1`. The agent prints
  the faulting wavefront/trap to the log on a GPU exception — cheap and far more
  useful than a full coredump.
- **GPU coredumps are huge** (~mem_fraction × VRAM per rank; e.g. 8×~150GB ≈ 1TB)
  and can fill the disk. Cap with a small `ulimit -c` or disable, and rely on the
  debug agent text. `ROCM_DEBUG_AGENT_OPTIONS="-s <dir>"` saves loaded GPU code
  objects (small, MB-scale) for later `llvm-objdump`.
- **NFS root_squash gotcha:** the container runs as root → mapped to `nobody` on
  NFS → kernel/coredump writes become 0-byte. Put `core_pattern` and the dump dir
  on **node-local disk** (`/var/tmp/cores`), set it via
  `/proc/sys/kernel/core_pattern` + `fs.suid_dumpable=2`, and bind-mount that path
  into the container so the in-container path resolves. Back up & restore the
  original `core_pattern` (default is the apport pipe on Ubuntu).

## 5. Pin the NaN source — and the instrumentation pitfalls

A `sitecustomize.py` monkeypatch (auto-imported via `PYTHONPATH`) can wrap
`aiter ...CustomAllreduce.all_reduce` to flag, after each call, output NaN/Inf and
whether the **input was already bad**. Pitfalls that made this misleading:

- **CUDA graph blinds Python instrumentation.** During capture a `.item()` sync
  is illegal (`operation not permitted when stream is capturing`); during replay
  the Python wrapper does not run at all (only captured kernels replay). So in the
  production (graph-on) runs the check never fires. You must run with
  `--disable-cuda-graph` for a Python wrapper to observe AR outputs — but that
  changes the trigger.
- **A per-rank wrapper only sees the LOCAL input.** A multi-rank reduce
  (`cross_device_reduce_2stage`) sums all ranks; it can output NaN with a clean
  *local* input when a *peer's* input is already NaN. `INPUT_ALREADY_BAD=False`
  means "my local input is clean", NOT "this AR created the NaN". Do not conclude
  "kernel X is the source" from this alone — the NaN may have propagated from an
  upstream call on another rank.
- **Don't `import torch` at sitecustomize top level**, and **don't eagerly import
  aiter there** — both perturb ROCm init ordering and hang startup. Patch lazily
  (only after the target module is already in `sys.modules`).

## 6. Identify the trapping assert (it's torch-internal)

The `_assert_async_cuda_kernel` is NOT sglang's `maybe_detect_nan`
(`SGLANG_SPEC_NAN_DETECTION`) nor the sampler's `enable_nan_detection` — both
default OFF and were off. aiter has no device asserts. With
`sampling_backend="pytorch"` (the ROCm default) the sampler uses
`torch.multinomial` and a `@torch.compile` path; **torch.compile/inductor inserts
`_assert_async` runtime guards**. So the assert fires from torch internals
whenever a NaN reaches sampling, regardless of any sglang flag. Confirm via the
`aten::_assert_async.msg` symbols in `libtorch_hip.so`.

## 7. aiter AR kernel anatomy (`csrc/include/custom_all_reduce.cuh`)

- Dispatch (`allreduce(...)`, `use_new=true`): `bytes = numel*sizeof(T)` (whole
  tensor). For full xGMI: `world_size<=8 && bytes < 80KB` -> **1stage**, else
  **2stage**. `write_mode` only on `gfx942` (not gfx950). So decode batch size
  decides the kernel: e.g. hidden=7168 bf16 -> <=5 tokens (<80KB) = 1stage, >=6 =
  2stage. Compute which kernel YOUR shapes hit before blaming one.
- All AR kernels bracket the reduction with `start_sync` (entry) and `end_sync`
  (exit) cross-rank barriers. `end_sync` publishes a flag (SYSTEM-scope release)
  and spins on peers' flags; it makes peer data writes visible across GPUs.

## 8. Root cause (PR #3514) and why scope/fence "fixes" don't help

`cross_device_reduce_1stage` was the **only** AR kernel missing the `end_sync`
call at kernel exit. Without an exit barrier: a fast rank A finishes its peer-IPC
reads and exits; A's next graph kernel (PyTorch `graph_pool` aggressively reuses
the AR input slot under a captured CUDA graph) overwrites A's input slot while a
slow rank B is still reading it via IPC -> B reads garbage -> B's AR output is
NaN/Inf. Fix (PR #3514):

```cpp
        buf = next_buf;
    }
+   end_sync<ngpus, true>(sg, self_sg, rank);   // match every other AR kernel
}
```

Pitfall that cost time here: it is tempting to "strengthen" `end_sync` (change the
acquire from `__MEMORY_SCOPE_DEVICE` to `__MEMORY_SCOPE_SYSTEM`, add
`__threadfence_system`). Those were built, disassembly-verified
(`buffer_inv sc0 sc1`, `buffer_wbl2`) and loaded — and still crashed — **because
the buggy kernel never calls `end_sync` at all**, so making `end_sync` stronger is
a no-op for it. Lesson: confirm the suspect kernel is actually on your path
(`__builtin_trap()` at its entry, or check the dispatch) before fixing its
internals.

## 9. Rebuild aiter + test a patched kernel without a new image

aiter modules are JIT-built but the runtime image ships prebuilt `.so` and the
`aiter_meta` build alias is removed. To rebuild one module:

1. Edit `csrc/include/custom_all_reduce.cuh` (the header is inlined into the kernel).
2. Recreate the build alias: `ln -sfn /sgl-workspace/aiter /sgl-workspace/aiter/aiter_meta` and `mkdir -p .../build/module_custom_all_reduce/blob`.
3. `cd .../jit/build/module_custom_all_reduce/build && rm -f *.o *.so && ninja`
   (hipcc compile is CPU-only — does not need exclusive GPU).
4. `cp` the rebuilt `module_custom_all_reduce.so` over `.../jit/module_custom_all_reduce.so`, extract it to a host path.
5. Launch decode with the patched `.so` bind-mounted read-only over the image's:
   `-v <patched.so>:/sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so:ro`.
6. **Verify it's really applied**: `docker inspect` the mount; and disassemble
   both `.so` (roc-obj-ls to get the gfx950 code-object offset, `dd` it out,
   `llvm-objdump -d --mcpu=gfx950`, diff) so you don't chase a false negative.

## 10. Fix & workarounds

- **Fix:** upgrade aiter to include PR #3514 (commit `3895df5`).
- **Workarounds (validated stable):** `SGLANG_USE_AITER_AR=0` (sglang-native
  custom AR) or `--disable-custom-all-reduce` (RCCL). Until the image carries the
  PR, bind-mount a rebuilt patched `module_custom_all_reduce.so`.

## Key takeaways (transferable)

1. Bisect with a flag matrix before reading kernels — it localizes the bug in hours.
2. A NaN that trips `_assert_async` in sampling almost always came from upstream
   (often an all-reduce); the crash site (`copy_done.synchronize`) is async noise.
3. Multi-rank reduce: "local input clean + output NaN" ≠ "this kernel is the
   source"; a peer's input may already be NaN.
4. CUDA-graph-triggered bugs are invisible to Python-level instrumentation —
   instrument device-side or use a deterministic input probe (each rank writes
   `1.0`; correct AR sum = world_size).
5. Verify any binary patch reached the loaded `.so` (mount + disasm) before
   trusting a negative result.
