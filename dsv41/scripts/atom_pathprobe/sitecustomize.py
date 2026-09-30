# Runtime path probe for ATOM DSpark decode (loaded via PYTHONPATH, no image edits).
# Counts, per process: Sampler.sample_verification_tokens calls, rejection_sample calls split by
# branch (synthetic / target_token_ids / other), Sampler.forward calls by all_greedy, and draft
# DSparkMarkovHead.sample_next calls (graph capture only if the draft is graph-replayed).
# Dumps a JSON line to $ATOM_PATHPROBE_OUT every 200 rejection_sample calls and at exit.
import atexit, collections, importlib.abc, importlib.machinery, json, os, sys, time

_C = collections.Counter()
_OUT = os.environ.get("ATOM_PATHPROBE_OUT", "/tmp/atom_pathprobe.jsonl")


def _dump(tag="tick"):
    try:
        with open(_OUT, "a") as f:
            f.write(json.dumps({"t": time.time(), "pid": os.getpid(), "tag": tag, **_C}) + "\n")
    except Exception:
        pass


atexit.register(_dump, "exit")


def _patch_sampler(m):
    S = m.Sampler
    svt, fwd = S.sample_verification_tokens, S.forward

    def sample_verification_tokens(self, *a, **k):
        _C["sample_verification_tokens"] += 1
        return svt(self, *a, **k)

    def forward(self, *a, **k):
        ag = k.get("all_greedy", a[4] if len(a) > 4 else None)
        _C[f"sampler_forward_all_greedy={ag}"] += 1
        return fwd(self, *a, **k)

    S.sample_verification_tokens, S.forward = sample_verification_tokens, forward


def _patch_rejection(m):
    rs = m.rejection_sample

    def rejection_sample(*a, **k):
        if k.get("synthetic_acceptance_rates") is not None:
            _C["rejection_synthetic"] += 1
        elif k.get("target_token_ids") is not None:
            _C["rejection_target_ids"] += 1
        else:
            _C["rejection_other"] += 1
        _C["rejection_with_target_ids_given"] += int(k.get("target_token_ids") is not None)
        n = sum(_C[x] for x in ("rejection_synthetic", "rejection_target_ids", "rejection_other"))
        if n % 200 == 0:
            _dump()
        return rs(*a, **k)

    m.rejection_sample = rejection_sample


def _patch_draft(m):
    H = m.DSparkMarkovHead
    sn = H.sample_next

    def sample_next(self, *a, **k):
        _C["draft_markov_sample_next"] += 1
        return sn(self, *a, **k)

    H.sample_next = sample_next


_HOOKS = {"atom.model_ops.sampler": _patch_sampler, "atom.model_ops.rejection_sampler": _patch_rejection,
          "atom.models.deepseek_v4_dspark": _patch_draft}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _HOOKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _name=name):
            _orig(module)
            try:
                _HOOKS[_name](module)
                _C[f"patched:{_name}"] = 1
            except Exception as e:
                _C[f"patch_failed:{_name}:{type(e).__name__}"] = 1

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
