#!/usr/bin/env python3
"""Vendor the mono kernel sources into sglang (MONO_KERNEL_mi355-4.md, packaging = in-tree). Re-runnable: overwrites the
vendored trees from the pinned upstream checkouts, rewrites their package imports to sglang paths and adds an origin
header (repo, commit, path) after each file's SPDX lines.

    python3 mono_vendor.py [--sglang /sgl-workspace/sglang] [--atom /tmp/mono_src/ATOM] [--vllm /tmp/mono_src/vllm]

L0  ATOM atom/mono/{device,plan,runtime} (MIT)        -> python/sglang/srt/layers/mono/common/   (runtime/deployment.py
    is ATOM-config specific and replaced by sglang's guards: not vendored)
L3  vLLM vllm/models/deepseek_v41/amd/mono (Apache-2.0) -> python/sglang/srt/models/deepseek_common/amd/mono/kernels/
Prints one line per tree (files copied, imports rewritten); exits non-zero if a foreign import is left.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path


def head_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def vendor(src: Path, dst: Path, repo_name: str, commit: str, src_rel: str, old_pkg: str, new_pkg: str,
           skip: set[str], foreign: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    n_files = n_imports = 0
    left = []
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_dir() or "__pycache__" in rel.parts or str(rel) in skip:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix != ".py":
            shutil.copy2(f, out)
            continue
        text = f.read_text()
        text, k = re.subn(rf"\b{re.escape(old_pkg)}\b", new_pkg, text)
        n_imports += k
        lines = text.splitlines(keepends=True)
        i = 0
        while i < len(lines) and lines[i].startswith("# SPDX-"):
            i += 1
        origin = f"# Vendored from {repo_name} @ {commit[:10]}: {src_rel}/{rel.as_posix()}\n"
        text = "".join(lines[:i]) + origin + "".join(lines[i:])
        out.write_text(text)
        n_files += 1
        if re.search(foreign, text):
            left.append(str(rel))
    print(f"{dst}: {n_files} py files, {n_imports} import rewrites, foreign imports left: {left or 'none'}")
    if left:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang", default="/sgl-workspace/sglang")
    ap.add_argument("--atom", default="/tmp/mono_src/ATOM")
    ap.add_argument("--vllm", default="/tmp/mono_src/vllm")
    a = ap.parse_args()
    srt = Path(a.sglang) / "python/sglang/srt"
    atom, vllm = Path(a.atom), Path(a.vllm)
    ac, vc = head_commit(atom), head_commit(vllm)

    lib = srt / "layers/mono/common"
    lib.mkdir(parents=True, exist_ok=True)
    for sub in ("device", "plan", "runtime"):
        vendor(atom / "atom/mono" / sub, lib / sub, "ROCm/ATOM", ac, f"atom/mono/{sub}",
               "atom.mono", "sglang.srt.layers.mono.common",
               skip={"deployment.py"}, foreign=r"^\s*(from|import) (atom|vllm)\b")
    init = (atom / "atom/mono/__init__.py").read_text().splitlines(keepends=True)
    (lib / "__init__.py").write_text("".join(init[:2]) + f"# Vendored from ROCm/ATOM @ {ac[:10]}: atom/mono/__init__.py\n"
                                     + "".join(init[2:]))

    k = srt / "models/deepseek_common/amd/mono/kernels"
    vendor(vllm / "vllm/models/deepseek_v41/amd/mono", k, "vllm-project/vllm", vc,
           "vllm/models/deepseek_v41/amd/mono", "vllm.models.deepseek_v41.amd.mono",
           "sglang.srt.models.deepseek_common.amd.mono.kernels",
           skip=set(), foreign=r"(?m)^\s*(from|import) (atom|vllm)\b")
    pkg = k.parent / "__init__.py"
    if not pkg.exists():
        pkg.write_text("")


if __name__ == "__main__":
    main()
