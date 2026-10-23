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
