#!/usr/bin/env python3
"""Diff two arms' `sglang_command.txt`. Run this ~60 s after launching an A/B
arm and abort the run if anything other than the variable under test differs.

    python3 cmd_diff.py <ref result dir> <new result dir> [--expect FLAG ...]

Earned the hard way: a `total_tokens` arm was launched against a reference that
had `mem-fraction-static 0.85`, but the launcher's MegaMoE+DP branch defaults it
to 0.65 (`MEM_FRACTION_STATIC_DP_MEGAMOE`). That changes the KV pool, so it
changes cache hit, batch composition and every number the arm exists to produce.
Caught 90 s in; it would otherwise have cost an hour and produced a confounded
result that looked fine.

The rule this enforces is the standing one: verify flags from the produced
`sglang_command.txt`, never from what the script appears to intend.

Exit status is 1 when an unexpected difference is present, so it can gate a run.
"""
import sys
from pathlib import Path


def flags(path):
    t = Path(path).read_text().split()
    out, i = {}, 0
    while i < len(t):
        if t[i].startswith("--"):
            v = t[i + 1] if i + 1 < len(t) and not t[i + 1].startswith("--") else ""
            out[t[i]] = v
            i += 2 if v else 1
        else:
            i += 1
    return out


def main(argv):
    expect = set()
    if "--expect" in argv:
        k = argv.index("--expect")
        expect = {f if f.startswith("--") else "--" + f for f in argv[k + 1:]}
        argv = argv[:k]
    ref, new = (Path(a) / "sglang_command.txt" if Path(a).is_dir() else Path(a)
                for a in argv[:2])
    A, B = flags(ref), flags(new)

    bad = []
    for f in sorted(set(A) | set(B)):
        if A.get(f) != B.get(f):
            tag = "expected" if f in expect else "UNEXPECTED"
            print(f"{tag:>10s}  {f:<38s} ref={A.get(f, '<absent>'):<20s} "
                  f"new={B.get(f, '<absent>')}")
            if f not in expect:
                bad.append(f)
    if not bad:
        print(f"OK - {len(A)} flags, only the expected differences" if expect
              else f"OK - identical, {len(A)} flags")
        return 0
    print(f"\n{len(bad)} unexpected difference(s). Kill the arm and fix them "
          f"before spending the hour.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
