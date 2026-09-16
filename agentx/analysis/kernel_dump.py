#!/usr/bin/env python3
"""MI355X-side data for the three things B200 asked for in FINDINGS.md:
   (a) per-role kernel listing at bs=10, full-model verify
   (b) compute by (rank, bs) -- does any bs repeat across ranks?
   (c) compute-vs-bs within ONE rank, which deconfounds bs from rank
Also prints summed-kernel/wall overlap per rank.
"""
import collections
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import attribute, classify, load, rank_of, verify_classes


def prettify(names):
    """Demangle in one batch and strip signatures/template args.

    Do NOT use trace_common.norm_name here: it collapses runs of 2+ digits to
    `N`, which destroys Itanium mangling (`_ZN5aiter37foo` -> `_ZNNaiterNfoo`)
    and hides real kernel identity -- `flash_cN_prefill` is actually
    `flash_c128_prefill`, and the megamoe tile shapes carry the config.
    """
    mangled = [n for n in names if n.startswith("_Z")]
    demap = {}
    if mangled:
        try:
            out = subprocess.run(["c++filt"], input="\n".join(mangled),
                                 capture_output=True, text=True, check=True).stdout
            demap = dict(zip(mangled, out.splitlines()))
        except (OSError, subprocess.CalledProcessError):
            pass                      # no demangler: fall back to raw names
    out = {}
    for n in names:
        s = demap.get(n, n)
        s = re.sub(r"^void\s+", "", s)
        s = re.sub(r"\(.*", "", s)                          # drop the signature
        s = re.sub(r"<[^<>]*(<[^<>]*>)?[^<>]*>", "<..>", s)  # collapse templates
        if s.startswith("_Z"):
            # c++filt cannot demangle some aiter kernels (bf16 appears as the
            # non-standard `DF16b`), so read the Itanium length prefixes
            # directly: each `<len><identifier>` gives one name component.
            parts = []
            for m in re.finditer(r"\d+", s):
                ln = int(m.group())
                seg = s[m.end():m.end() + ln]
                # exact length, or a `Li128E`-style template arg truncated by
                # the end of the string wins `max(len)` with pure junk
                if len(seg) == ln and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", seg):
                    parts.append(seg)
            if parts:
                # prefer a component that looks like a kernel name: a stray
                # template arg can produce a longer exact-length alnum run
                # Itanium puts the function name before its template args, so
                # take the FIRST kernel-looking component, not the longest: a
                # stray template arg can yield a longer exact-length alnum run.
                role_ish = [p for p in parts if re.search(
                    r"kernel|gemm|moe|attn|quant|norm|rope|combine|dispatch", p)]
                name = role_ish[0] if role_ish else max(parts, key=len)
                ns = [p for p in parts[:parts.index(name)]
                      if p in ("aiter", "ck", "sglang", "opus", "at", "native")]
                s = ("::".join(ns + [name])) + " [mangled]"
        out[n] = s.strip()
    return out

COMPUTE = ("attn", "gemm", "quant", "norm_rope", "sample")
BARRIER = ("moe", "comm")


def full_steps(path):
    """-> list of (bs, wall_ms, [(dur_us, name), ...]) for full-model verify."""
    steps, kernels = load(path)
    owned = attribute(steps, kernels)
    vc = verify_classes(steps, owned)
    return [(s["bs"], s["ms"], owned.get(i, []))
            for i, s in enumerate(steps)
            if s["type"] == "TARGET_VERIFY" and vc.get(i) == " full"]


def main(d):
    files = sorted(Path(d).glob("*.trace.json.gz"), key=rank_of)
    print(f"=== {d}")

    # (b) + (c): per (rank, bs)
    cells = {}
    for f in files:
        r = rank_of(f)
        by_bs = collections.defaultdict(list)
        for bs, wall, ks in full_steps(f):
            by_bs[bs].append((wall, ks))
        for bs, entries in by_bs.items():
            walls = [w for w, _ in entries]
            n = len(entries)
            roles_ps, ksum_ps = [], []
            for _w, ks in entries:
                acc = collections.defaultdict(float)
                for dur, name in ks:
                    acc[classify(name)] += dur / 1000.0
                roles_ps.append(acc)
                ksum_ps.append(sum(acc.values()))
            def p50(fn):
                return st.median([fn(a) for a in roles_ps])
            cells[(r, bs)] = {
                "n": n,
                "wall": st.median(walls),
                "ksum": st.median(ksum_ps),
                "compute": p50(lambda a: sum(a[x] for x in COMPUTE)),
                "barrier": p50(lambda a: sum(a[x] for x in BARRIER)),
                "attn": p50(lambda a: a["attn"]),
                "gemm": p50(lambda a: a["gemm"]),
                "moe": p50(lambda a: a["moe"]),
                "comm": p50(lambda a: a["comm"]),
                "copy": p50(lambda a: a["copy"]),
                "quant": p50(lambda a: a["quant"]),
            }

    print("\n-- (b) full-verify p50 by (rank, bs)")
    print(f"{'rank':>4s} {'bs':>3s} {'n':>3s} {'wall':>7s} {'ksum':>7s} {'ovl':>5s}"
          f" {'compute':>8s} {'barrier':>8s} {'attn':>6s} {'gemm':>6s} {'moe':>6s}"
          f" {'comm':>6s} {'copy':>6s} {'quant':>6s}")
    for (r, bs) in sorted(cells):
        c = cells[(r, bs)]
        print(f"{r:4d} {bs:3d} {c['n']:3d} {c['wall']:7.2f} {c['ksum']:7.2f}"
              f" {c['ksum'] / max(c['wall'], 1e-9):5.2f} {c['compute']:8.2f}"
              f" {c['barrier']:8.2f} {c['attn']:6.2f} {c['gemm']:6.2f} {c['moe']:6.2f}"
              f" {c['comm']:6.2f} {c['copy']:6.2f} {c['quant']:6.2f}")

    bs_ranks = collections.defaultdict(list)
    for (r, bs) in cells:
        bs_ranks[bs].append(r)
    rep = {bs: rs for bs, rs in bs_ranks.items() if len(rs) > 1}
    print(f"\n   bs repeated across ranks: {rep if rep else 'NONE'}")
    per_rank_bs = collections.defaultdict(list)
    for (r, bs) in cells:
        per_rank_bs[r].append(bs)
    multi = {r: sorted(v) for r, v in per_rank_bs.items() if len(v) > 1}
    print(f"   ranks with >1 bs (deconfounds bs at fixed rank): {multi}")
    return cells


def kernel_dump(path, want_bs, top_thresh=0.02, width=100):
    ents = [(bs, w, ks) for bs, w, ks in full_steps(path) if bs == want_bs]
    n = len(ents)
    if not n:
        print(f"\n-- (a) no full TARGET_VERIFY at bs={want_bs} on rank {rank_of(path)}")
        return
    print(f"\n-- (a) per-kernel, TARGET_VERIFY full bs={want_bs}, "
          f"rank {rank_of(path)}, n={n} steps")
    by_role = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    tot = 0.0
    for _bs, _w, ks in ents:
        for dur, name in ks:
            slot = by_role[classify(name)][name]
            slot[0] += dur / 1000.0
            slot[1] += 1
            tot += dur / 1000.0
    nice = prettify([k for r in by_role.values() for k in r])
    print(f"   summed kernel {tot / n:.2f} ms/step")
    for role in sorted(by_role, key=lambda r: -sum(v[0] for v in by_role[r].values())):
        rtot = sum(v[0] for v in by_role[role].values()) / n
        ks = sorted(by_role[role].items(), key=lambda kv: -kv[1][0])
        print(f"   * {role} {rtot:.2f} ms/step  {100 * rtot / (tot / n):.1f}%"
              f"  ({len(ks)} distinct kernels)")
        small = 0
        for name, (dur, c) in ks:
            if dur / n < top_thresh:
                small += 1
                continue
            print(f"       {dur / n:7.3f} ms {c / n:5.0f} calls"
                  f" {1000 * dur / max(c, 1):8.1f} us/call  {nice[name][:width]}")
        if small:
            rest = sum(v[0] for k, v in ks if v[0] / n < top_thresh) / n
            print(f"       {rest:7.3f} ms  (+{small} kernels below {top_thresh} ms)")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/shared_nfs/kk/pr35619/trace_c128_pdi24_steady"
    main(d)
    if len(sys.argv) > 2 and sys.argv[2] == "nodump":
        sys.exit(0)
    f7 = next(Path(d).glob("*TP-7-*.gz"))
    kernel_dump(f7, 10)
