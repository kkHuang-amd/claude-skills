#!/usr/bin/env python3
"""RFC #43949 step 1 (move only): mono framework + kernels under kernels/ops/mono/.

  python3 rfc_step1_move.py /sgl-workspace/sglang_v41mono     # run once, on a clean tree
"""

import re
import subprocess
import sys
from pathlib import Path

tree = Path(sys.argv[1])
P = "python/sglang"
K = f"{P}/kernels/ops/mono"


def git(*a):
    subprocess.run(["git", "-C", str(tree), *a], check=True)


# framework: srt/layers/mono/* -> kernels/ops/mono/ (its __init__ re-export is dropped: the package stays import-light)
fw = tree / P / "srt/layers/mono"
for f in sorted(fw.iterdir()):
    if f.name in ("__pycache__", "__init__.py"):
        continue
    git("mv", str(f.relative_to(tree)), f"{K}/{f.name}")
git("rm", "-q", f"{P}/srt/layers/mono/__init__.py")
git("mv", f"{K}/gfx950/common", f"{K}/common_hip")
git("mv", f"{K}/gfx950/deepseek_v41", f"{K}/deepseek_v41_gfx950")
git("rm", "-q", f"{K}/gfx950/__init__.py")
amd = f"{P}/srt/models/deepseek_common/amd/mono"
for n in ("spec.py", "indexer_hook.py"):
    git("mv", f"{amd}/{n}", f"{K}/deepseek_v41_gfx950/{n}")
git("rm", "-q", f"{amd}/__init__.py")
(tree / K / "minimax_m3_gfx950").mkdir()
git("mv", f"{P}/srt/models/minimax_m3_mono/spec.py", f"{K}/minimax_m3_gfx950/spec.py")
git("mv", f"{P}/srt/models/minimax_m3_mono/__init__.py", f"{K}/minimax_m3_gfx950/__init__.py")

SUBS = [
    (r"from sglang\.srt\.layers import mono\b", "from sglang.kernels.ops.mono import layer as mono"),
    (r"from sglang\.srt\.layers\.mono import (?=MonoRuntime|attach)", "from sglang.kernels.ops.mono.layer import "),
    (r"sglang\.kernels\.ops\.mono\.gfx950\.common\b", "sglang.kernels.ops.mono.common_hip"),
    (r"sglang\.kernels\.ops\.mono\.gfx950\.deepseek_v41\b", "sglang.kernels.ops.mono.deepseek_v41_gfx950"),
    (r"sglang\.srt\.models\.deepseek_common\.amd\.mono\b", "sglang.kernels.ops.mono.deepseek_v41_gfx950"),
    (r"sglang\.srt\.models\.minimax_m3_mono\b", "sglang.kernels.ops.mono.minimax_m3_gfx950"),
    (r"sglang\.srt\.layers\.mono\b", "sglang.kernels.ops.mono"),
    (r"\bgfx950/common\b", "common_hip"),
    (r"\bgfx950/deepseek_v41\b", "deepseek_v41_gfx950"),
    (r"srt/layers/mono/", "kernels/ops/mono/"),
]
changed = []
files = subprocess.run(
    ["git", "-C", str(tree), "ls-files", "python", "test", "docs"], check=True, capture_output=True, text=True
).stdout.split()
for rel in files:
    if not rel.endswith((".py", ".md")):
        continue
    f = tree / rel
    if not f.is_file():
        continue
    s = f.read_text()
    t = s
    for a, b in SUBS:
        t = re.sub(a, b, t)
    if t != s:
        f.write_text(t)
        changed.append(rel)
print(f"rewrote {len(changed)} files")
for rel in changed:
    print(" ", rel)
