#!/usr/bin/env python3
"""Compile the DeepSeek-V4.1 mono kernels without launching them and write each kernel's disassembly (task 20 F6:
the shared-common convergence must not change the generated code).

  PYTHONPATH=<tree>/python HIP_VISIBLE_DEVICES=4 python3 mono_isa_dump.py OUT_DIR [--quick] [--tok-par | --tok-par-only]
  diff -r OUT_A OUT_B      # identical ISA <=> no output

One file per build: <kernel symbol>.s (llvm-objdump -d of the code object, addresses stripped) and a summary line in
OUT_DIR/index.txt (symbol, sha256 of the disassembly, vgpr / sgpr / lds / scratch).
"""

import hashlib
import inspect
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["COMPILE_ONLY"] = "1"
os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", tempfile.mkdtemp(prefix="mono_isa_cache_"))

try:  # RFC #43949 layout (perf/v41-mono-rfc step 1 on)
    from sglang.kernels.ops.mono.deepseek_v41_gfx950 import layer as L  # noqa: E402
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer import k2i  # noqa: E402
except ImportError:
    from sglang.kernels.ops.mono.gfx950.deepseek_v41 import layer as L  # noqa: E402
    from sglang.kernels.ops.mono.gfx950.deepseek_v41.indexer import k2i  # noqa: E402

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
_BIN = re.compile(r'bin\s*=\s*"')


def code_object(module_text: str) -> bytes:
    m = _BIN.search(module_text)
    if m is None:
        raise RuntimeError("no gpu.binary in the compiled module")
    out, i = bytearray(), m.end()
    while module_text[i] != '"':
        ch = module_text[i]
        if ch != "\\":
            out.append(ord(ch))
            i += 1
        elif module_text[i + 1] in '\\"':
            out.append(ord(module_text[i + 1]))
            i += 2
        else:
            out.append(int(module_text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def note(elf: bytes, key: bytes):
    i = elf.find(key)
    if i < 0:
        return None
    j = i + len(key)
    tag = elf[j]
    if tag < 0x80:
        return tag
    if tag == 0xCC:
        return elf[j + 1]
    if tag == 0xCD:
        return int.from_bytes(elf[j + 1 : j + 3], "big")
    if tag == 0xCE:
        return int.from_bytes(elf[j + 1 : j + 5], "big")
    return None


def disasm(elf: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(elf)
        f.flush()
        text = subprocess.run(
            [OBJDUMP, "-d", "--no-show-raw-insn", "--no-leading-addr", f.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    lines = [ln for ln in text.splitlines() if ln.strip() and "file format" not in ln]
    return "\n".join(re.sub(r"\s*//.*$", "", ln).rstrip() for ln in lines) + "\n"


def jit_of(launcher):
    if hasattr(launcher, "_mem_cache"):
        return launcher
    for obj in [c.cell_contents for c in (getattr(launcher, "__closure__", None) or ())]:
        if hasattr(obj, "_mem_cache"):
            return obj
    raise RuntimeError(f"{launcher}: no compiled jit object")


def compile_one(launcher):
    sig = inspect.signature(getattr(launcher, "__wrapped__", None) or getattr(launcher, "func", launcher))
    args = [0 for p in sig.parameters.values() if p.name != "stream" and p.kind is p.POSITIONAL_OR_KEYWORD]
    kw = {"stream": 0} if "stream" in sig.parameters else {}
    launcher(*args, **kw)
    return [code_object(a._ir_text) for a in jit_of(launcher)._mem_cache.values()]


def builds(quick: bool):
    tokens = (1, 6) if quick else (1, 2, 6, 12, 24, 48)
    target = dict(experts=384, topk=6, ug_groups=2)
    draft = dict(experts=128, topk=3, ug_groups=1, swa_width=192)
    for t in tokens:
        for r in (0, 1, 2):
            for ex in (False, True) if r else (False,):
                b = L.MonoBuild(t, 4, r, kv41=True, export_idx=ex, cfp4=r > 0, **target)
                yield f"k1_t{t}_r{r}_ex{int(ex)}", L.build_mono_k1(b)
                yield f"k2_t{t}_r{r}_ex{int(ex)}", L.build_mono_k2(b)
                if ex and "index_q" in L.MonoBuild.__dataclass_fields__:  # task 17 S3
                    bq = L.MonoBuild(t, 4, r, kv41=True, export_idx=True, index_q=True, cfp4=True, **target)
                    yield f"k1iq_t{t}_r{r}", L.build_mono_k1(bq)
        if hasattr(L, "build_mono_ffn"):  # removed after 134919476c (task 26b)
            yield f"ffn_t{t}", L.build_mono_ffn(L.MonoBuild(t, 4, 0, **target))
        bd = L.MonoBuild(t, 4, 0, kv41=True, **draft)
        yield f"draft_k1_t{t}", L.build_mono_k1(bd)
        yield f"draft_k2_t{t}", L.build_mono_k2(bd)
        yield f"k2i_t{t}", k2i.build_k2i(k2i.K2iBuild(t, 8192, tree=False))


def tok_par_builds():
    """Task 27b: token-parallel K2 builds (not in the default set, which must stay comparable to older dumps)."""
    target = dict(experts=384, topk=6, ug_groups=2)
    for tp in (2, 4):
        for t in (12, 24, 48):
            for r in (0, 1, 2):
                b = L.MonoBuild(t, tp, r, kv41=True, export_idx=r > 0, cfp4=r > 0, tok_par=True, **target)
                yield f"k2tok_tp{tp}_t{t}_r{r}", L.build_mono_k2(b)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sets = tok_par_builds() if "--tok-par-only" in sys.argv else builds("--quick" in sys.argv)
    if "--tok-par" in sys.argv:
        import itertools

        sets = itertools.chain(sets, tok_par_builds())
    for name, launcher in sets:
        try:
            elfs = compile_one(launcher)
        except Exception as e:  # report and go on: one broken build must not hide the others
            rows.append(f"{name} ERROR {type(e).__name__}: {str(e)[:160]}")
            print(rows[-1], flush=True)
            continue
        for k, elf in enumerate(elfs):
            text = disasm(elf)
            tag = name if len(elfs) == 1 else f"{name}_{k}"
            (out / f"{tag}.s").write_text(text)
            res = {
                key: note(elf, b"." + key.encode())
                for key in ("vgpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size")
            }
            rows.append(f"{tag} {hashlib.sha256(text.encode()).hexdigest()[:16]} {res}")
            print(rows[-1], flush=True)
    (out / "index.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
