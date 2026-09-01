#!/usr/bin/env bash
# /workspace/claude-skills/dsv4 is read-only to the container (uid 100454,
# NFS root-squash). Run this AS THE OWNING USER to install. Idempotent.
set -euo pipefail
SRC=/workspace/results/dsv4-fixed_seq_len
DST=/workspace/claude-skills/dsv4/fixed_seq_len
mkdir -p "$DST"
for f in serve_b200.sh serve_mi355x.sh sweep_mi355x.sh report.py report_70k.py README.md RESULTS.md NOTES.md; do
    cp -v "$SRC/$f" "$DST/$f"
done
chmod +x "$DST"/serve_*.sh
echo "installed to $DST"
