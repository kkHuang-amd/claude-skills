#!/usr/bin/env bash
# Mean device time per kernel name for a torch-free binary (rocprofv3 kernel trace).
# Usage: rocprof_ktime.sh <binary> [args...]
set -uo pipefail
OUT=/tmp/rocprof_kt/$(basename "$1")
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"
rocprofv3 --kernel-trace --output-format csv -d "$OUT" -o kt -- "$@" > "$OUT.log" 2>&1 || { tail -3 "$OUT.log"; exit 1; }
F=$(find "$OUT" -name '*kernel_trace.csv' | head -1)
python3 - "$F" <<'EOF'
import csv, sys, collections
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[r["Kernel_Name"]].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000)
for k, v in sorted(d.items(), key=lambda x: -sum(x[1])):
    v = v[3:] or v  # drop warmup
    print(f"{sum(v) / len(v):8.2f} us  x{len(v):3d}  {k[:90]}")
EOF
