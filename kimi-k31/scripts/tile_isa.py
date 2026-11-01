#!/usr/bin/env python3
"""Compile (no launch) K3.1 row-tile GEMM builds; print vgpr / agpr / scratch / lds a build and
optionally write the disassembly. Code-object helpers from dsv41/scripts/mono_isa_dump.py.
Usage: tile_isa.py M:N:K:NB[,...] [OUT_DIR]"""
import os
import re
import subprocess
import sys
import tempfile

os.environ["COMPILE_ONLY"] = "1"
os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", tempfile.mkdtemp(prefix="tile_isa_"))
OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"


def code_object(text):
    i = re.search(r'bin\s*=\s*"', text).end()
    out = bytearray()
    while text[i] != '"':
        if text[i] != "\\":
            out.append(ord(text[i]))
            i += 1
        elif text[i + 1] in '\\"':
            out.append(ord(text[i + 1]))
            i += 2
        else:
            out.append(int(text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def note(elf, key):
    i = elf.find(key)
    if i < 0:
        return None
    j = i + len(key)
    tag = elf[j]
    if tag < 0x80:
        return tag
    return {0xCC: lambda: elf[j + 1], 0xCD: lambda: int.from_bytes(elf[j + 1 : j + 3], "big"),
            0xCE: lambda: int.from_bytes(elf[j + 1 : j + 5], "big")}.get(tag, lambda: None)()


def jit_of(launcher):
    if hasattr(launcher, "_mem_cache"):
        return launcher
    for c in getattr(launcher, "__closure__", None) or ():
        if hasattr(c.cell_contents, "_mem_cache"):
            return c.cell_contents
    raise RuntimeError("no compiled jit object")




def regs(launch):
    """Compile ``launch`` (zero args) -> [(elf, {vgpr, agpr, scratch, lds})] a variant."""
    import inspect

    sig = inspect.signature(getattr(launch, "__wrapped__", None) or getattr(launch, "func", launch))
    launch(*[0 for p in sig.parameters.values() if p.name != "stream" and p.kind is p.POSITIONAL_OR_KEYWORD],
           stream=0)
    keys = ("vgpr_count", "agpr_count", "private_segment_fixed_size", "group_segment_fixed_size")
    return [(e, {k: note(e, b"." + k.encode()) for k in keys})
            for e in (code_object(a._ir_text) for a in jit_of(launch)._mem_cache.values())]


if __name__ == "__main__":
    from sgl_amd_mono.kimi_k31.gfx950.stages import tile  # noqa: E402

    out = sys.argv[2] if len(sys.argv) > 2 else None
    for spec in sys.argv[1].split(","):
        m, n, k, nb = map(int, spec.split(":"))
        launch = tile.standalone(m, n, k, nb)
        launch(0, 0, 0, stream=0)
        for a in jit_of(launch)._mem_cache.values():
            elf = code_object(a._ir_text)
            keys = ("vgpr_count", "agpr_count", "private_segment_fixed_size", "group_segment_fixed_size")
            print(spec, {kk: note(elf, b"." + kk.encode()) for kk in keys}, flush=True)
            if out:
                with tempfile.NamedTemporaryFile(suffix=".co") as f:
                    f.write(elf)
                    f.flush()
                    text = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", f.name], capture_output=True,
                                          text=True, check=True).stdout
                os.makedirs(out, exist_ok=True)
                open(f"{out}/tile_{spec.replace(':', '_')}.s", "w").write(re.sub(r"\s*//.*", "", text))
