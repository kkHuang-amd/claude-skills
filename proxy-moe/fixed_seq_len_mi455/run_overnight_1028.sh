#!/usr/bin/env bash
# Overnight 2026-10-28 (h13-17): new best config sweep, then the 10-23 config rerun, then both reports.
# Resumable: rerun this script; finished points (DONE) are skipped.
HERE="$(cd "$(dirname "$0")" && pwd)"
NEW=/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_allon_h13-17_1028
OLD=/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_oldcfg_h13-17_1028
mkdir -p "$NEW" "$OLD"
echo "=== overnight start $(date '+%F %T') pid $$"
MODE=new OUT=$NEW bash "$HERE/sweep_mi455_1028.sh" >> "$NEW/sweep.log" 2>&1
echo "=== new done $(date '+%F %T'): $(find $NEW -name DONE | wc -l) DONE, failed $(cat $NEW/FAILED.txt 2> /dev/null | wc -l)"
MODE=old OUT=$OLD bash "$HERE/sweep_mi455_1028.sh" >> "$OLD/sweep.log" 2>&1
echo "=== old done $(date '+%F %T'): $(find $OLD -name DONE | wc -l) DONE, failed $(cat $OLD/FAILED.txt 2> /dev/null | wc -l)"
for D in $NEW $OLD; do python3 "$HERE/../fixed_seq_len/report.py" "$D" --md > "$D/REPORT.md" 2> "$D/report.err"; done
echo "=== overnight end $(date '+%F %T')"
