#!/bin/bash
# For each fetched upstream PR head (refs/remotes/upstream-pr/<N>), list its new files and new env
# vars relative to the queue's base, and whether dev_amd already has them.
# Usage: audit_pr_queue.sh <base-commit> <N>...
set -u
cd /sgl-workspace/sglang-k31-amd
BASE=$1; shift
for n in "$@"; do
  H=refs/remotes/upstream-pr/$n
  mb=$(git merge-base "$BASE" "$H")
  title=$(git log -1 --format=%s "$H" | cut -c1-90)
  nfiles=$(git diff --name-only "$mb" "$H" | wc -l)
  added=$(git diff --name-only --diff-filter=A "$mb" "$H" -- python | rg -v '__init__.py$')
  envs=$(git diff "$mb" "$H" -- python/sglang/srt/environ.py | rg -o '^\+\s+(SGLANG_[A-Z0-9_]+|AITER_[A-Z0-9_]+)\s*=' | rg -o '(SGLANG|AITER)_[A-Z0-9_]+' | sort -u)
  na=0; ha=0; ne=0; he=0; miss_f=""; miss_e=""
  for f in $added; do na=$((na+1)); if git cat-file -e "origin/dev_amd:$f" 2>/dev/null; then ha=$((ha+1)); else miss_f="$miss_f $(basename $f)"; fi; done
  for e in $envs; do ne=$((ne+1)); if git grep -q "$e" origin/dev_amd -- python 2>/dev/null; then he=$((he+1)); else miss_e="$miss_e $e"; fi; done
  echo "#$n  files=$nfiles  new_files_in_dev_amd=$ha/$na  new_envs_in_dev_amd=$he/$ne  | $title"
  [ -n "$miss_f" ] && echo "     missing files:$(echo $miss_f | cut -c1-300)"
  [ -n "$miss_e" ] && echo "     missing envs: $(echo $miss_e | cut -c1-300)"
done
