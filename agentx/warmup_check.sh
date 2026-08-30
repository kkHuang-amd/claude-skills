#!/usr/bin/env bash
# Health of a RUNNING arm: warmup progress vs the reference arm at matched elapsed.
# Reference is tbo-tp8-c64 (52/94/262/701 at 300/600/900/1200 s) -- the REFERENCE
# arm, never the fastest arm (trap 11). Never judge health from /metrics (trap 1)
# or decode-batch counts (trap 2).
D=${1:?usage: warmup_check.sh <arm-dir-or-name>}
[ -d "$D" ] || D=/workspace/results/overnight/$D
REF=${REF:-/workspace/results/tbo-tp8-c64/benchmark.log}
LOG=$D/benchmark.log
[ -f "$LOG" ] || { echo "no benchmark.log yet in $D (launcher still starting?)"; exit 0; }
# NOTE: aiperf prints thousands separators once the warmup set exceeds 999
# ("returned=223/2,845" at c256). A [0-9]+ regex matches NOTHING there and the
# check reports silence, which looks like health. Always allow [0-9,] and strip
# the commas before arithmetic.
series(){ grep -oE "Phase [a-z]+ progress \| returned=[0-9,]+/[0-9,]+ \| sent=[0-9,]+ \| in_flight=[0-9,]+ \| errors=[0-9]+ \| elapsed=[0-9.]+s" "$1"; }
echo "--- $(basename $D): last 3 progress lines"
series "$LOG" | tail -3
echo "--- vs reference tbo-tp8-c64 at matched elapsed"
for t in 300 600 900 1200; do
  a=$(series "$LOG" | grep -oE "returned=[0-9,]+/[0-9,]+ .*elapsed=$t\.[0-9]s" | head -1 | grep -oE "^returned=[0-9,]+" | cut -d= -f2 | tr -d ,)
  b=$(series "$REF" | grep -oE "returned=[0-9,]+/[0-9,]+ .*elapsed=$t\.[0-9]s" | head -1 | grep -oE "^returned=[0-9,]+" | cut -d= -f2 | tr -d ,)
  [ -n "$a" ] && printf "  %5ss  arm %-5s ref %-5s  %s\n" "$t" "$a" "${b:--}" \
     "$(python3 -c "print('%+.0f %%'%(($a-$b)/$b*100))" 2>/dev/null)"
done
echo "--- errors / phase"
grep -oE "Phase [a-z]+ (started|progress)" "$LOG" | sort | uniq -c | tail -4
series "$LOG" | tail -1 | grep -oE "errors=[0-9]+"
echo "--- stall check (arm 2 froze at returned=223 for 98 min, errors=0)"
u=$(series "$LOG" | tail -20 | grep -oE "returned=[0-9,]+" | sort -u | wc -l)
if [ "$u" -le 1 ]; then
  echo "  **STALLED** -- last 20 progress lines all identical. Check the server:"
  echo "  grep -E '#running-req|#queue-req' \$D/server.log | tail -3"
  echo "  0 running + non-zero queued + last log line long ago = DP collective hang."
else
  echo "  progressing ($u distinct returned= values in last 20 lines)"
fi
