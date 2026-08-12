#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


CANONICAL = (
    "kimi-k3-optimization-scan.canvas.tsx",
    "kimi-k3-hardware-comparison.canvas.tsx",
    "kimi-k3-experiment-history.canvas.tsx",
)

LEGACY = (
    "aiter-kimi-k3-optimization-scan.canvas.tsx",
    "b300-vs-mi355X-kimi-k3.canvas.tsx",
    "kimi-k3-batch1-fusions.canvas.tsx",
    "kimi-k3-optimization-execution.canvas.tsx",
    "kimi-k3-optimization-priorities.canvas.tsx",
    "kimi-k3-pr-stack-validation.canvas.tsx",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prune-legacy", action="store_true")
    args = parser.parse_args()

    source = Path(__file__).resolve().parent
    args.target.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, args.target / name)
        print(f"synced {name}")

    if args.prune_legacy:
        for name in LEGACY:
            path = args.target / name
            if path.exists():
                path.unlink()
                print(f"removed legacy {name}")


if __name__ == "__main__":
    main()
