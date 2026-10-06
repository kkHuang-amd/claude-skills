"""Run DSV4.1 (vLLM, NVIDIA path) forward kernels on one CUDA stream, without editing the installed vLLM.

Active only when DSV41_SINGLE_STREAM=1 and this directory is on PYTHONPATH (spawned workers inherit both).
Also set VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 and VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0.
Same kernels as the multi-stream run; only the stream they are issued on changes:
  - attention: compressor / insert_cache / aux input GEMMs run inline instead of on aux streams
  - mHC: DecoderLayer.mhc_stream resolves to the current stream, so the overlap path's ops and joins stay on it
  - engram: cpu-offload lookups always take the inline path (no prefetch stream)
Left alone: model-runner D2H copy streams (output_copy_stream, spec-decode copy_stream).
"""
import importlib.abc
import importlib.machinery
import os
import sys

_orig = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_orig):
    exec(compile(open(_orig).read(), _orig, "exec"))


def _patch_attention(m):
    f_maybe, f_exec = m.maybe_execute_in_parallel, m.execute_in_parallel

    def maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream=None):
        return f_maybe(fn0, fn1, event0, event1, None)

    def execute_in_parallel(default_fn, aux_fns, start_event, done_events, aux_streams=None, enable=False):
        return f_exec(default_fn, aux_fns, start_event, done_events, None, enable=False)

    m.maybe_execute_in_parallel, m.execute_in_parallel = maybe_execute_in_parallel, execute_in_parallel


def _patch_model(m):
    import torch

    # forward reads self.mhc_stream once and passes it to every mHC op and wait_stream; returning the
    # stream current at that moment keeps the overlap code path (same kernels) but on the capture stream.
    def get(self):
        return torch.cuda.current_stream() if self.__dict__.get("_mhc_stream_orig") is not None else None

    def set_(self, v):
        self.__dict__["_mhc_stream_orig"] = v

    m.DeepseekV4DecoderLayer.mhc_stream = property(get, set_)


def _patch_engram(m):
    f = m._engram_lookup_thresholds
    m._engram_lookup_thresholds = lambda device: (f(device)[0], 0)


_PATCHES = {
    "vllm.models.deepseek_v41.attention": _patch_attention,
    "vllm.models.deepseek_v41.nvidia.model": _patch_model,
    "vllm.models.deepseek_v41.common.engram": _patch_engram,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner, self.fn = inner, fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)
        print(f"[single_stream] patched {module.__name__} pid={os.getpid()}", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = _PATCHES.get(name)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, fn)
        return spec


if os.environ.get("DSV41_SINGLE_STREAM") == "1":
    sys.meta_path.insert(0, _Finder())
