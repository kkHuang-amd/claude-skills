#!/usr/bin/env bash
# Per-socket clock lock for the AR-skew test (h13-17 1029). Runs on the HOST (needs sudo + PM interfaces).
# Mechanism: AGT SMC messages, per /workspace/skills/mi450-clock-control/SKILL.md. Never uses AGT -eng= / -fclk=.
#
#   agt_clock.sh probe                   read-only: -clkstatus on $CAND (default "0"; e.g. CAND="0 5 10 15") -> $OUT, progress on screen
#   agt_clock.sh cleanup                 kill a stuck agt_internal (only if it is yours) and clear the SMU mutexes
#   agt_clock.sh status                  read-only: -clkstatus on $IDX + sysfs GFXCLK
#   IDX="0 5 10 15" agt_clock.sh lock <gfx> <fclk> <gl2>   force (MHz, 500-2700; "-" skips a domain); auto-unlocks on any failure
#   IDX="0 5 10 15" agt_clock.sh unlock  unforce gfx/fclk/gl2 on $IDX and read back
#
# AGT is found in order: $AGT_BIN, agt_internal on PATH, /tmp/agt159, $AGT_TGZ, the vendored tarball under /workspace/skills.
set -u
CMD=${1:-}; shift || true
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
CAND=${CAND:-"0"}
TMO=${TMO:-20}
OUT=${OUT:-/home/wunhuang/tmp/agt_probe.txt}

find_agt() {
  [ -n "${AGT_BIN:-}" ] && [ -x "$AGT_BIN" ] && { echo "$AGT_BIN"; return; }
  command -v agt_internal 2>/dev/null && return
  local p; p=$(find /tmp/agt159 -maxdepth 3 -name agt_internal -type f 2>/dev/null | head -1)
  [ -n "$p" ] && { chmod +x "$p"; echo "$p"; return; }
  local tgz=${AGT_TGZ:-}
  [ -z "$tgz" ] && tgz=$(ls /workspace/skills/mi450-pm-capture/scripts/vendor/AGT_Internal_*.tar.gz 2>/dev/null | sort -V | tail -1)
  [ -f "$tgz" ] || { echo "no AGT found: set AGT_BIN=/path/agt_internal or AGT_TGZ=/path/AGT_Internal_*.tar.gz" >&2; exit 1; }
  mkdir -p /tmp/agt159 && tar -xzf "$tgz" -C /tmp/agt159 || exit 1
  p=$(find /tmp/agt159 -maxdepth 3 -name agt_internal -type f | head -1); chmod +x "$p"; echo "$p"
}
[ -z "$SUDO" ] || sudo -n true 2>/dev/null || { echo "need passwordless sudo: run 'sudo -v' first" >&2; exit 1; }
echo "locating AGT (first run extracts ~138 MB to /tmp/agt159)..."
AGT=$(find_agt) || exit 1
AD=$(dirname "$AGT")
echo "agt=$AGT"

clear_mutex() { $SUDO rm -f /dev/shm/TOOLS_SMU_*MUTEX* /dev/shm/SMN_0_5; }
agt() {  # -k: SIGKILL if AGT ignores SIGTERM; stdin closed so it can never wait for input
  $SUDO timeout -k 5 $TMO env LD_LIBRARY_PATH="$AD" "$AGT" "$@" < /dev/null; local rc=$?
  [ $rc = 124 ] || [ $rc = 137 ] && { echo "agt TIMEOUT after ${TMO}s ($*)" >&2; $SUDO pkill -9 -x agt_internal; clear_mutex; }
  return $rc
}
sysfs_clk() { for f in /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input; do [ -r "$f" ] && echo "$f $(( $(cat "$f") / 1000000 )) MHz"; done; }
need_idx() { [ -n "${IDX:-}" ] || { echo "set IDX (socket device indices from 'probe'), e.g. IDX=\"0 5 10 15\"" >&2; exit 1; }; }

if [ "$CMD" = cleanup ]; then
  pgrep -a -x agt_internal || echo "no agt_internal running"
  $SUDO pkill -9 -x agt_internal; sleep 1; clear_mutex; echo "cleanup done"; exit 0
fi
# Refuse to run next to another AGT instance (its mutex would be clobbered).
if pgrep -x agt_internal >/dev/null; then
  echo "agt_internal already running (pid $(pgrep -x agt_internal | tr '\n' ' ')) -- if it is a stuck one of yours: $0 cleanup; otherwise someone else is using AGT." >&2; exit 1
fi
clear_mutex

send() {  # $1=idx $2=msg $3=arg -> 0 only if FW acknowledged
  local o; o=$(agt -i=$1 -smcmsgsend=$2,$3,0x0 2>&1)
  echo "$o" | grep -q 'successfully sent' && { echo "  i=$1 $2 $3 ok: $(echo "$o" | grep -o 'FW response.*' | head -1)"; return 0; }
  echo "  i=$1 $2 $3 FAILED: $(echo "$o" | tail -2 | tr '\n' ' ' | cut -c1-200)"; return 1
}

unlock() {
  local i rc=0
  for i in $IDX; do
    send $i 0x1E 0x0 || rc=1    # GFXCLK
    send $i 0x22 0x0 || rc=1    # FCLK
    send $i 0x20 0x0 || rc=1    # GL2CLK
  done
  sleep 2; sysfs_clk
  [ $rc = 0 ] && echo "UNLOCK OK" || echo "UNLOCK HAD FAILURES -- rerun unlock; if it keeps failing, the GPU needs a reset"
  return $rc
}

case "$CMD" in
  probe)
    : > "$OUT"
    for i in $CAND; do
      echo "[$(date +%T)] -i=$i -clkstatus (timeout ${TMO}s) ..."
      { echo "== -i=$i"; agt -i=$i -clkstatus 2>&1; echo "rc=$?"; } >> "$OUT"
      echo "[$(date +%T)]   $(tail -1 "$OUT"), $(wc -l < "$OUT") lines so far"
    done
    sysfs_clk >> "$OUT"
    echo "wrote $OUT ($(wc -l < "$OUT") lines)"; sysfs_clk ;;
  status)
    need_idx
    for i in $IDX; do
      echo "== -i=$i [$(date +%T)]"
      agt -i=$i -clkstatus 2>&1 | grep -E 'VendorID|Engine Clock|GL2CLK|FCLK|MCLK_a'
      agt -i=$i -fclk 2>&1 | grep -v -E '^\s*$|AGT version|stty'
    done
    sysfs_clk ;;
  lock)
    need_idx
    [ $# = 3 ] || { echo "usage: IDX=\"...\" $0 lock <gfx> <fclk> <gl2>   (MHz, '-' = leave on DPM)" >&2; exit 1; }
    for v in "$@"; do [ "$v" = - ] || { [ "$v" -ge 500 ] 2>/dev/null && [ "$v" -le 2700 ]; } || { echo "bad MHz '$v' (500-2700 or -)" >&2; exit 1; }; done
    GFX=$1 FCLK=$2 GL2=$3; rc=0
    echo "locking IDX=[$IDX] gfx=$GFX fclk=$FCLK gl2=$GL2 (order gfx, fclk, gl2)"
    for i in $IDX; do
      [ "$GFX"  = - ] || send $i 0x1D $(printf 0x%x $GFX)  || rc=1
      [ "$FCLK" = - ] || send $i 0x21 $(printf 0x%x $FCLK) || rc=1
      [ "$GL2"  = - ] || send $i 0x1F $(printf 0x%x $GL2)  || rc=1
    done
    if [ $rc != 0 ]; then echo "LOCK FAILED -> unlocking everything"; unlock; exit 1; fi
    sleep 2; sysfs_clk
    echo "LOCKED. When done: IDX=\"$IDX\" $0 unlock" ;;
  unlock)
    need_idx; unlock ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
