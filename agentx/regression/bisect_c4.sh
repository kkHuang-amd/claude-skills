#!/usr/bin/env bash
# Autonomous c4 bisect of the DSV4-Pro AgentX regression (crsuse2-m2m-049).
#
#   0. references: GOOD = mean of two c73f7077eb c4 runs (049 Phase A + rerun),
#      BAD = B0 c4 (HEAD 41cbe65de0). Two metrics: tok/s/GPU and aiperf P90
#      interactivity (slow tail). A metric is used only if its gap >= MIN_GAP and
#      the same-code spread of the two GOOD runs is < half that gap. None usable -> stop.
#   1. c4 at 425a1f8f24 (parent of #41019). bad -> bisect c73f7077eb..425a1f8f24.
#      good -> test effb752188 (#41019, the next commit): bad -> it is the first bad;
#      good -> bisect effb752188..41cbe65de0 (first-parent, history is linear).
#   2. per usable metric: good >= mid+band, bad <= mid-band, else ambiguous. Any metric
#      bad -> bad; all good -> good; otherwise one more run, and the means are
#      classified against mid (any bad -> bad). A run that fails twice is skipped.
#   3. confirm: rerun the first bad commit and its good neighbour once more.
#
# Each measurement is one matrix.sh c4 run (~72 min) in its own detached worktree
# /sgl-workspace/sglang-bis-<sha>. Results are cached by TAG, so re-launching
# resumes: finished points are skipped and their JSON re-read.
# Decisions: BISECT_C4.log (this dir). Launch:
#   setsid nohup bash bisect_c4.sh < /dev/null > /workspace/results/bisect_c4.driver.log 2>&1 &
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E=/workspace/claude-skills/agentx/e2e
REPO=/sgl-workspace/sglang
RES=/workspace/results
LOG=$R/BISECT_C4.log
GOOD_C=c73f7077eb GOOD_TAG=sglc73f-049      # 049 Phase A
HEAD_C=41cbe65de0 HEAD_TAG=head41cbe-049    # B0
PRE_C=425a1f8f24                            # parent of #41019
SUS_C=effb752188                            # #41019
MIN_GAP=${MIN_GAP:-0.04}

exec 9> /tmp/bisect_c4.lock
flock -n 9 || { echo "another bisect_c4.sh is running"; exit 1; }
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }  # stderr: callers capture stdout

metrics() {  # <result dir> -> "tok intvty_p90 succ", empty unless the point finished OK
  local j; j=$(ls "$1"/dsv4_fp4_sglang_*_agentic.json 2>/dev/null | head -1)
  [ -f "$1/.ok" ] && [ -n "$j" ] || return 0
  python3 -c "import json;d=json.load(open('$j'));r=d['request_metrics']
print('%.1f %.1f %d' % (r['throughput']['per_gpu']['total_tput_tps'], r['latency']['intvty']['p90'], d['num_requests_successful']))"
}

worktree() {  # <sha> -> worktree path
  local c; c=$(git -C "$REPO" rev-parse --short=10 "$1")
  case $c in
    41cbe65de0) echo /sgl-workspace/sglang-head; return ;;
    c73f7077eb) echo /sgl-workspace/sglang-ci; return ;;
  esac
  local wt=/sgl-workspace/sglang-bis-$c
  if [ ! -d "$wt" ]; then
    git -C "$REPO" worktree add --detach "$wt" "$c" > /dev/null 2>&1 || { log "ERROR worktree $c"; return 1; }
    cp "$REPO"/python/sglang/srt/rust_extensions/_multimodal*.so "$wt/python/sglang/srt/rust_extensions/"
  fi
  echo "$wt"
}

run_c4() {  # <sha> <tag> -> metrics or empty; reruns a failed run once
  local wt m try; wt=$(worktree "$1") || return 0
  for try in 1 2; do
    m=$(metrics "$RES/e2e-c4-$2")
    [ -n "$m" ] && { echo "$m"; return 0; }
    log "RUN c4 $1 tag=$2 (attempt $try)"
    (cd /workspace && env TAG="$2" POINTS=c4 SKIP_SMOKE=1 PYTHONPATH="$wt/python" SGL_DIR="$wt" \
      bash "$E/matrix.sh" < /dev/null > "$RES/e2e-matrix-$2.driver.log" 2>&1)
  done
  metrics "$RES/e2e-c4-$2"
}

# verdict <tok> <intvty> [strict] -> good | bad | ambig over the usable metrics.
# Non-strict uses mid +- band; strict (means of two runs) compares to mid only.
verdict() {
  python3 - "$1" "$2" "${3:-}" <<PY
import sys
tok, itv, strict = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "strict"
refs = {"tok": (tok, $TOK_MID, $TOK_BAND, $TOK_ON), "intvty": (itv, $ITV_MID, $ITV_BAND, $ITV_ON)}
vs = []
for v, mid, band, on in refs.values():
    if not on: continue
    if strict: vs.append("good" if v >= mid else "bad")
    else: vs.append("good" if v >= mid + band else "bad" if v <= mid - band else "ambig")
print("bad" if "bad" in vs else "good" if all(x == "good" for x in vs) else "ambig")
PY
}

# classify <sha> -> good | bad | skip, logging every measurement
classify() {
  local c tag m m2 v t i
  c=$(git -C "$REPO" rev-parse --short=10 "$1"); tag=bis-$c-049
  m=$(run_c4 "$c" "$tag")
  if [ -z "$m" ]; then log "SKIP $c (two failed runs)"; echo skip; return; fi
  log "MEASURE $c tok/s/GPU,intvty_p90,succ = $m  ($(git -C "$REPO" log -1 --format=%s "$c" | cut -c1-90))"
  read -r t i _ <<< "$m"
  v=$(verdict "$t" "$i")
  if [ "$v" = ambig ]; then
    m2=$(run_c4 "$c" "$tag-r2")
    if [ -n "$m2" ]; then
      log "MEASURE $c rerun = $m2"
      t=$(python3 -c "print('%.1f' % (($t+${m2%% *})/2))")
      i=$(python3 -c "print('%.1f' % (($i+$(echo "$m2" | cut -d' ' -f2))/2))")
      log "  means tok $t (mid $TOK_MID) intvty $i (mid $ITV_MID)"
    fi
    v=$(verdict "$t" "$i" strict)
  fi
  log "VERDICT $c $v"
  echo "$v"
}

log "==== bisect_c4 start (pid $$)"
while pgrep -f run_c4_fix_then_head.sh > /dev/null; do sleep 60; done   # B0 c4 is in this chain
G1=$(metrics "$RES/e2e-c4-$GOOD_TAG")
B0=$(run_c4 "$HEAD_C" "$HEAD_TAG")
G2=$(run_c4 "$GOOD_C" "$GOOD_TAG-r2")
[ -n "$G1" ] && [ -n "$G2" ] && [ -n "$B0" ] || { log "ABORT: missing reference (g1='$G1' g2='$G2' b0='$B0')"; exit 1; }
log "REF good $GOOD_C = $G1 | rerun = $G2 | B0 $HEAD_C = $B0"
eval "$(python3 - "$G1" "$G2" "$B0" "$MIN_GAP" <<'PY'
import sys
g1, g2, b = (list(map(float, a.split()[:2])) for a in sys.argv[1:4]); min_gap = float(sys.argv[4])
for k, name in ((0, "TOK"), (1, "ITV")):
    good = (g1[k] + g2[k]) / 2; bad = b[k]
    gap = (good - bad) / good; spread = abs(g1[k] - g2[k]) / good
    on = int(gap >= min_gap and spread < gap / 2)
    print(f"{name}_GOOD={good:.1f} {name}_BAD={bad:.1f} {name}_GAP={gap:.4f} {name}_SPREAD={spread:.4f} "
          f"{name}_MID={(good + bad) / 2:.1f} {name}_BAND={(good - bad) / 4:.2f} {name}_ON={on}")
PY
)"
log "tok/s/GPU: good $TOK_GOOD bad $TOK_BAD gap $TOK_GAP spread $TOK_SPREAD -> used=$TOK_ON (mid $TOK_MID band $TOK_BAND)"
log "intvty:    good $ITV_GOOD bad $ITV_BAD gap $ITV_GAP spread $ITV_SPREAD -> used=$ITV_ON (mid $ITV_MID band $ITV_BAND)"
if [ "$TOK_ON" = 0 ] && [ "$ITV_ON" = 0 ]; then
  log "DONE: no c4 gap that the same-code spread can resolve (need gap >= $MIN_GAP and spread < gap/2) -- nothing to bisect"
  exit 0
fi

v=$(classify "$PRE_C")
case $v in
  bad)  LO=$GOOD_C HI=$PRE_C ;;
  good)
    v2=$(classify "$SUS_C")
    if [ "$v2" = bad ]; then
      log "FIRST BAD $SUS_C (direct child of good $PRE_C): $(git -C "$REPO" log -1 --format=%s "$SUS_C")"
      m=$(run_c4 "$SUS_C" "bis-$SUS_C-049-confirm"); log "CONFIRM $SUS_C = $m"
      m=$(run_c4 "$PRE_C" "bis-$PRE_C-049-confirm"); log "CONFIRM $PRE_C = $m"
      log "==== bisect_c4 finished: first bad $SUS_C, last good $PRE_C"; exit 0
    fi
    LO=$SUS_C HI=$HEAD_C ;;
  *)    log "ABORT: $PRE_C could not be measured"; exit 1 ;;
esac
log "range: good $LO .. bad $HI"

mapfile -t L < <(git -C "$REPO" rev-list --first-parent --reverse "$LO..$HI" | cut -c1-10)
lo=-1 hi=$((${#L[@]} - 1))          # L[hi] is bad; index -1 stands for LO (good)
while [ $((hi - lo)) -gt 1 ]; do
  m=$(((lo + hi) / 2))
  log "step: ${#L[@]} commits, window $((hi - lo - 1)) untested, testing ${L[$m]}"
  v=$(classify "${L[$m]}")
  case $v in
    good) lo=$m ;;
    bad)  hi=$m ;;
    skip) L=("${L[@]:0:$m}" "${L[@]:$((m + 1))}"); hi=$((hi - 1)) ;;
  esac
done

FB=${L[$hi]}; [ $lo -ge 0 ] && PG=${L[$lo]} || PG=$LO
log "FIRST BAD (candidate) $FB: $(git -C "$REPO" log -1 --format=%s "$FB")"
log "confirming: rerun $FB and $PG"
for c in "$FB" "$PG"; do
  c=$(git -C "$REPO" rev-parse --short=10 "$c")
  if [ "$c" = "$GOOD_C" ]; then log "CONFIRM $c = Phase A (not rerun)"; continue; fi
  m=$(run_c4 "$c" "bis-$c-049-confirm"); log "CONFIRM $c = $m"
done
log "==== bisect_c4 finished: first bad $FB, last good $PG"
