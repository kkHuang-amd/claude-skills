# Same logic as sglang PR #44292, for any Python process (e.g. aiter op tests that spawn
# workers): drop an inherited ROCPROFILER_REGISTER_LIBRARY so torch.profiler (kineto on
# rocprofiler-sdk, torch >= 2.12) can register in this process. Use via PYTHONPATH.
import os
if os.environ.get("ROCPROFILER_REGISTER_LIBRARY") is not None:
    try:
        loaded = "librocprofiler-sdk" in open("/proc/self/maps").read()
    except OSError:
        loaded = True
    if not loaded:
        os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)

# Under rocprofv3 the tool LD_PRELOADs librocprofiler-sdk (and, through it, libamd_comgr) from
# _rocm_sdk_devel/lib. torch's rocm_sdk.initialize_process then dlopens byte-identical copies from
# _rocm_sdk_core/lib by full path, so they load twice and the process aborts ("Option
# 'spirv-expand-step' registered more than once", then rocprofiler "error code 16"). Skip preloading
# any library whose soname is already mapped.
if "librocprofiler-sdk-tool" in os.environ.get("LD_PRELOAD", ""):
    try:
        import rocm_sdk

        _orig_preload = rocm_sdk.preload_libraries

        def _preload_unloaded(*shortnames, **kw):
            try:
                maps = open("/proc/self/maps").read()
            except OSError:
                maps = ""
            return _orig_preload(*[n for n in shortnames if f"/lib{n}.so" not in maps], **kw)

        rocm_sdk.preload_libraries = _preload_unloaded
    except Exception:
        pass
