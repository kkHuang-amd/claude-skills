#!/usr/bin/env python3
"""Per-DP-rank KV / request skew from a server.log. RUN THIS ON BOTH NODES.

    python3 kv_skew.py <server.log> [--tail-frac 0.5]

Answers, with no trace and no GPU: **is the DP load actually balanced, and
balanced on what?** Reads the scheduler's own `Decode batch` lines, which carry
per-rank `#running-req` and `#full token`.

Why it matters: `--load-balance-method total_requests` equalises *request
count*, but decode attention cost tracks *KV tokens*. If `#running-req` is
level across ranks while `#full token` is skewed, the balancer is levelling the
wrong quantity and the straggler -- which sets the group step -- is whoever
holds the most KV.

Three numbers per rank, over the steady-state tail of the run:

  running-req    what `total_requests` balances
  full token     what attention actually costs (`total_tokens` balances this)
  tok/req        full token / running-req -- how long this rank's requests are

The verdict line compares the two spreads. A large `full token` spread with a
small `running-req` spread is the signature that switching to
`--load-balance-method total_tokens` has something to fix.
"""
import collections
import re
import statistics as st
import sys

LINE = re.compile(
    r"DP(?P<dp>\d+) TP\d+.*?Decode batch.*?"
    r"#running-req:\s*(?P<req>\d+).*?"
    r"#full token:\s*(?P<tok>\d+)")


def main(path, tail_frac=0.5):
    per_rank = collections.defaultdict(list)
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                rows.append((int(m.group("dp")), int(m.group("req")),
                             int(m.group("tok"))))
    if not rows:
        print("no `Decode batch` lines matched -- check the log format")
        return
    cut = int(len(rows) * (1.0 - tail_frac))
    for dp, req, tok in rows[cut:]:
        per_rank[dp].append((req, tok))

    print(f"=== {path}")
    print(f"{len(rows)} decode lines; using the last {tail_frac:.0%} "
          f"({len(rows) - cut} lines) as steady state\n")
    print(f"{'rank':>4s} {'lines':>6s} {'running-req':>12s} {'full token':>13s}"
          f" {'tok/req':>10s}")
    reqs, toks, tprs = {}, {}, {}
    for dp in sorted(per_rank):
        v = per_rank[dp]
        r = st.median([a for a, _ in v])
        t = st.median([b for _, b in v])
        # per-request KV, computed per line then medianed -- not median/median
        tpr = st.median([b / a for a, b in v if a > 0]) if any(a > 0 for a, _ in v) else 0
        reqs[dp], toks[dp], tprs[dp] = r, t, tpr
        print(f"{dp:4d} {len(v):6d} {r:12.1f} {t:13.0f} {tpr:10.0f}")

    def spread(d):
        lo, hi = min(d.values()), max(d.values())
        return lo, hi, (hi / lo if lo > 0 else float("inf"))

    print()
    for label, d in (("running-req", reqs), ("full token", toks),
                     ("tok/req", tprs)):
        lo, hi, ratio = spread(d)
        print(f"  {label:<12s} {lo:12.0f} - {hi:12.0f}   max/min {ratio:6.2f}x")

    rq = spread(reqs)[2]
    tk = spread(toks)[2]
    print()
    if tk > 1.3 and tk > 1.5 * rq:
        print(f"  VERDICT: `full token` is {tk:.2f}x skewed while `running-req` is only "
              f"{rq:.2f}x\n           -> the balancer is levelling request COUNT while the "
              f"cost follows KV.\n           -> `--load-balance-method total_tokens` has "
              f"something to fix here.")
    elif tk > 1.3:
        print(f"  VERDICT: both are skewed ({tk:.2f}x KV, {rq:.2f}x requests) -- the "
              f"balancer is\n           not keeping up at all; look at admission, not just "
              f"the metric.")
    else:
        print(f"  VERDICT: KV is level within {tk:.2f}x -- no KV-skew problem on this run; "
              f"a\n           straggler here would have to come from something other than "
              f"KV volume.")
    print("\n  Caveat: these lines are sampled independently per rank, not "
          "synchronised\n  per step, so this measures the sustained distribution, "
          "not one step's skew.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    frac = 0.5
    if "--tail-frac" in sys.argv:
        frac = float(sys.argv[sys.argv.index("--tail-frac") + 1])
    main(sys.argv[1], frac)
