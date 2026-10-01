# MoE routing probe for SGLang and ATOM (loaded via PYTHONPATH; EAGER mode only -- Python hooks do not run inside
# CUDA graphs / torch.compile). Wraps aiter.fused_moe.fused_moe at import time and records, per call keyed by
# (E = w1.shape[0], num_tokens): calls, sum/min/max of DISTINCT routed experts (ids < E rounded down to even, so a
# fused shared-expert slot 384/128 is excluded). Cumulative stats are rewritten every 2 s to $MOE_PROBE_OUT.<pid>.
import collections, importlib.abc, importlib.machinery, json, os, sys, time

_OUT = os.environ.get("MOE_PROBE_OUT", "/tmp/moe_probe")
_S = collections.defaultdict(lambda: [0, 0, 1 << 30, 0])  # calls, sum_distinct, min, max
_last = [0.0]
_dumped = [0]
hidden_states_ref = [None]


def _record(w1, topk_ids):
    import torch
    E = int(w1.shape[0])
    routed = E - (E % 2)
    ids = topk_ids.reshape(-1)
    d = int(torch.unique(ids[(ids >= 0) & (ids < routed)]).numel())
    T = int(topk_ids.shape[0])
    dump_n = int(os.environ.get("MOE_PROBE_DUMP", "0"))
    if dump_n and 1 < T <= 64 and E >= 384 and _dumped[0] < dump_n and os.path.exists(_OUT + ".dump_on"):
        _dumped[0] += 1
        h = hidden_states_ref[0].float()
        rec = {"E": E, "T": T, "distinct": d, "topk_ids": topk_ids.tolist(),
               "h_row_norm": h.norm(dim=-1).tolist(), "h_nan": int(torch.isnan(h).sum()),
               "h_rows_unique": int(torch.unique(h.round(decimals=3), dim=0).shape[0]),
               "h_dtype": str(hidden_states_ref[0].dtype), "h_shape": list(hidden_states_ref[0].shape)}
        with open(f"{_OUT}.dump.{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    s = _S[f"{E}:{T}"]
    s[0] += 1; s[1] += d; s[2] = min(s[2], d); s[3] = max(s[3], d)
    now = time.time()
    if now - _last[0] > 2.0:
        _last[0] = now
        tmp = f"{_OUT}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"t": now, "stats": _S}, f)
        os.replace(tmp, f"{_OUT}.{os.getpid()}")


_SIG_DONE = set()


def _desc(v):
    import torch
    if isinstance(v, torch.Tensor):
        return {"tensor": True, "shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
    if hasattr(v, "value") and hasattr(v, "name"):  # enum
        return {"enum": type(v).__name__, "value": v.value}
    return {"repr": repr(v)}


def _signature(hidden_states, w1, w2, topk_weight, topk_ids, a, k):
    # One call signature per (E, num_tokens) for decode-sized calls -> microbench rebuilds same shapes/kwargs.
    key = (int(w1.shape[0]), int(topk_ids.shape[0]))
    if key in _SIG_DONE or key[1] > 64:
        return
    _SIG_DONE.add(key)
    sig = {"E": key[0], "T": key[1], "pos": [_desc(x) for x in (hidden_states, w1, w2, topk_weight, topk_ids) + a],
           "kw": {n: _desc(v) for n, v in k.items()}}
    with open(f"{_OUT}.sig.{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(sig) + "\n")


def _wrap(mod):
    orig = mod.fused_moe

    def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, *a, **k):
        try:
            hidden_states_ref[0] = hidden_states
            _record(w1, topk_ids)
            _signature(hidden_states, w1, w2, topk_weight, topk_ids, a, k)
        except Exception:
            pass
        return orig(hidden_states, w1, w2, topk_weight, topk_ids, *a, **k)

    mod.fused_moe = fused_moe


def _wrap_atom_api(mod):
    # ATOM synthetic-acceptance runs replace the API text with "synthetic "; log the real completion ids instead.
    orig = mod.delivered_text

    def delivered_text(token_ids):
        try:
            with open(f"{_OUT}.tokens.jsonl", "a") as f:
                f.write(json.dumps(list(map(int, token_ids))) + "\n")
        except Exception:
            pass
        return orig(token_ids)

    mod.delivered_text = delivered_text


_PDS_DONE = set()


def _wrap_pa_decode_sparse(mod):
    # Records one call signature per decode size (q rows <= 64); graph capture runs this once per batch size.
    orig = mod.pa_decode_sparse

    def pa_decode_sparse(*a, **k):
        try:
            n = int(a[0].shape[0])
            key = (n, k.get("extra_cache") is not None, k.get("kv_splits"))
            if n <= 64 and key not in _PDS_DONE:
                _PDS_DONE.add(key)
                sig = {"n": n, "pos": [_desc(x) for x in a], "kw": {kk: _desc(v) for kk, v in k.items()}}
                with open(f"{_OUT}.pds.{os.getpid()}.jsonl", "a") as f:
                    f.write(json.dumps(sig) + "\n")
        except Exception:
            pass
        return orig(*a, **k)

    mod.pa_decode_sparse = pa_decode_sparse


_WRAPPERS = {"aiter.fused_moe": _wrap, "atom.entrypoints.openai.api_server": _wrap_atom_api,
             "aiter.ops.triton.attention.pa_decode_sparse": _wrap_pa_decode_sparse}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner, self.fn = inner, fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _WRAPPERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, _WRAPPERS[name])
        return spec


sys.meta_path.insert(0, _Finder())
