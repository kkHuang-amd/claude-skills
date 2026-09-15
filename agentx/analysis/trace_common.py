"""Shared trace parsing for AgentX analysis. Platform-neutral by design.

The whole point of this module is that a CUDA trace and a ROCm trace cannot be
compared by kernel name — `deep_gemm::smN_fp8_fp4_mega_moe_impl` and aiter's
`asm_moe`/`ck_moe` are the same *role* under different names. Everything here
classifies kernels by ROLE, and always reports what it failed to classify so a
missing pattern is visible instead of silently landing in `other`.
"""
import collections
import gzip
import json
import re

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# Order matters: the first match wins. `mega_moe` must be tested before `gemm`,
# communication before everything (a wait shows up as comm, not as compute).
ROLES = [
    ("comm", r"all_gather|allgather|all_reduce|allreduce|nccl|rccl|gloo|"
             r"a2a|dispatch|combine|barrier|broadcast"),
    ("moe", r"mega_moe|fused_moe|asm_moe|ck_moe|moe_|_moe|expert|grouped_gemm|group_gemm"),
    ("attn", r"attn|attention|fmha|mha|flash|paged|mla|mqa_logits|indexer|"
             r"mhc_"),                     # mhc_* is DSv4's MLA pre/post fusion
    ("gemm", r"gemm|matmul|hipblas|cublas|_mm_|linear"),
    ("quant", r"quant|dequant|scale|fp8|fp4|ue8m0|cast"),
    ("norm_rope", r"norm|rope|rms|embed"),
    ("sample", r"sample|topk|argmax|logit"),
    ("copy", r"memcpy|memset|copy|cat_|index_|gather|scatter"),
]
ROLES = [(name, re.compile(pat, re.I)) for name, pat in ROLES]

STEP_RE = re.compile(r"step\[(?P<type>[A-Z_]+)(?:\s+bs=(?P<bs>\d+))?(?:\s+toks=(?P<toks>\d+))?")

_NORM = [(re.compile(r"<.*?>"), ""), (re.compile(r"\d{2,}"), "N"), (re.compile(r"\s+"), " ")]


def classify(name):
    for role, pat in ROLES:
        if pat.search(name):
            return role
    return "other"


def norm_name(name):
    for pat, rep in _NORM:
        name = pat.sub(rep, name)
    return name.strip()[:66]


def load(path):
    """-> (steps, kernels) where steps is a list of dicts and kernels are
    (ts, dur, name) tuples. Kernel->step attribution is by timestamp."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)["traceEvents"]

    # `gpu_user_annotation` step[...] events NEST AND OVERLAP heavily: one
    # TARGET_VERIFY step of ~30 ms also emits hundreds of sub-slices of ~0.4 ms.
    # Taking them all as steps gave 2077 "steps" of mean 2.89 ms for what were
    # really 62 steps of 16-30 ms, and made kernel->step attribution arbitrary
    # (a kernel lands in whichever overlapping slice is checked first). Keep
    # only the OUTERMOST annotations. The CPU-side `user_annotation` count is a
    # free cross-check: it agreed at 62.
    raw = []
    for e in ev:
        if (e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"
                and e.get("dur") and str(e.get("name", "")).startswith("step[")):
            raw.append((e["ts"], e["ts"] + e["dur"], e["name"]))
    raw.sort(key=lambda a: (a[0], -(a[1] - a[0])))

    steps = []
    for lo, hi, name in raw:
        if steps and lo >= steps[-1]["lo"] and hi <= steps[-1]["hi"]:
            continue                       # strictly inside the previous outer step
        m = STEP_RE.match(name)
        steps.append({
            "type": m.group("type") if m else "UNKNOWN",
            "bs": int(m.group("bs")) if m and m.group("bs") else None,
            "toks": int(m.group("toks")) if m and m.group("toks") else None,
            "lo": lo, "hi": hi, "ms": (hi - lo) / 1000.0,
        })

    kernels = [(e["ts"], e["dur"], e["name"]) for e in ev
               if e.get("ph") == "X" and e.get("cat") in GPU_CATS and e.get("dur")]
    kernels.sort()
    return steps, kernels


def attribute(steps, kernels):
    """-> {step_index or None: [(dur, name), ...]}. None = outside any step."""
    out = collections.defaultdict(list)
    for ts, dur, name in kernels:
        idx = None
        for i, s in enumerate(steps):
            if s["lo"] <= ts <= s["hi"]:
                idx = i
                break
        out[idx].append((dur, name))
    return out


def rank_of(path):
    m = re.search(r"-TP-(\d+)-", str(path))
    return int(m.group(1)) if m else -1
