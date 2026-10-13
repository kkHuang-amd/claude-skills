#!/usr/bin/env bash
# Download MiniMax-M3 MXFP4 target + EAGLE3 draft (ATOM recipes/MiniMax-M3-Agentic-InferenceX.md paths); second model
# for the SGLang mono framework (MONO_KERNEL_mi355-4.md). ~249 GB total, resumable (re-run to resume).
# Knobs: DEST (default /shared_nfs/huggingface_models), LOG.
# Output: $DEST/amd/MiniMax-M3-MXFP4, $DEST/Inferact/MiniMax-M3-EAGLE3-GQA; prints only exit codes.
set -u
DEST=${DEST:-/shared_nfs/huggingface_models}
LOG=${LOG:-/shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/download_minimax_m3.log}
mkdir -p "$(dirname "$LOG")"
rc_all=0
for r in Inferact/MiniMax-M3-EAGLE3-GQA amd/MiniMax-M3-MXFP4; do
  echo "[$(date +%F' '%T)] start $r" >> "$LOG"
  hf download "$r" --local-dir "$DEST/$r" >> "$LOG" 2>&1
  rc=$?; echo "[$(date +%F' '%T)] done $r rc=$rc" >> "$LOG"
  echo "$r rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
