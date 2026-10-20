"""Per-rank kernel-time split (comm vs moe vs rest) across all ranks of one phase.

Usage: python3 rank_skew.py <profile_dir> [DECODE|EXTEND]
If comm time differs a lot across ranks while total is equal, comm kernels are absorbing
wait-for-slowest-rank time; the lowest-comm rank approximates the real collective cost.
"""

import glob
import gzip
import json
import re
import sys

COMM = re.compile(r"all_reduce|allreduce|cross_device|reduce_scatter|quickreduce|allgather", re.I)
MOE = re.compile(r"moe", re.I)


def main() -> None:
    d = sys.argv[1]
    ph = sys.argv[2] if len(sys.argv) > 2 else "DECODE"
    files = sorted(glob.glob(f"{d}/**/*-TP-*-{ph}.trace.json.gz", recursive=True))
    print(f"{'rank':>4} {'total_ms':>9} {'comm_ms':>8} {'moe_ms':>8} {'rest_ms':>8}")
    for f in files:
        r = re.search(r"-TP-(\d+)-", f).group(1)
        ev = json.load(gzip.open(f, "rt"))["traceEvents"]
        tot = comm = moe = 0.0
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") == "kernel":
                t = e.get("dur", 0) / 1000
                tot += t
                if COMM.search(e["name"]):
                    comm += t
                elif MOE.search(e["name"]):
                    moe += t
        print(f"{r:>4} {tot:9.2f} {comm:8.2f} {moe:8.2f} {tot - comm - moe:8.2f}")


if __name__ == "__main__":
    main()
