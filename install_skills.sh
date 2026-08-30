#!/usr/bin/env bash
# Install this repo's skills into a Claude Code skills directory as symlinks.
#
# Re-run after any image/container swap: the repo lives on shared storage and
# survives, but ~/.claude/skills does not.
#
#   bash install_skills.sh                 # link into ~/.claude/skills
#   bash install_skills.sh --dry-run       # show what would happen
#   bash install_skills.sh --target DIR    # link somewhere else
#   bash install_skills.sh --force         # replace real dirs, not just symlinks
#   bash install_skills.sh --cursor-rules  # also run kimi-k3/install_cursor_rules.sh
#
# Idempotent: safe to run repeatedly. Existing symlinks are refreshed; real
# directories are left alone unless --force is given.

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET="${HOME}/.claude/skills"
DRY=0; FORCE=0; CURSOR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)      DRY=1 ;;
    --force)        FORCE=1 ;;
    --cursor-rules) CURSOR=1 ;;
    --target)       shift; TARGET="${1:?--target needs a directory}" ;;
    -h|--help)      sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '  %-34s %s\n' "$1" "$2"; }

echo "repo   : $REPO"
echo "target : $TARGET"
[ "$DRY" = 1 ] && echo "mode   : dry-run (nothing will be written)"
echo

[ "$DRY" = 1 ] || mkdir -p "$TARGET"

linked=0; refreshed=0; skipped=0; blocked=0

# Every directory holding a SKILL.md, up to two levels deep (catches nested
# skills such as gpt-oss/aiter-pa-decode-gluon-design). Each is linked under
# its own basename so it is discoverable on its own.
while IFS= read -r skill; do
  src="$(dirname -- "$skill")"
  name="$(basename -- "$src")"
  dest="$TARGET/$name"

  if [ -L "$dest" ]; then
    cur="$(readlink -- "$dest")"
    if [ "$cur" = "$src" ]; then say "$name" "already linked"; linked=$((linked+1)); continue; fi
    [ "$DRY" = 1 ] || { rm -f -- "$dest"; ln -s -- "$src" "$dest"; }
    say "$name" "relinked (was $cur)"; refreshed=$((refreshed+1)); continue
  fi

  if [ -e "$dest" ]; then
    if [ "$FORCE" = 1 ]; then
      [ "$DRY" = 1 ] || { rm -rf -- "$dest"; ln -s -- "$src" "$dest"; }
      say "$name" "replaced real directory (--force)"; refreshed=$((refreshed+1))
    else
      say "$name" "SKIPPED: real directory exists (use --force)"; blocked=$((blocked+1))
    fi
    continue
  fi

  [ "$DRY" = 1 ] || ln -s -- "$src" "$dest"
  say "$name" "linked"; linked=$((linked+1))
done < <(find "$REPO" -maxdepth 3 -name SKILL.md -not -path '*/.git/*' | sort)

# Directories with no SKILL.md are document corpora, not skills. Reaching them
# by path is intentional -- see NEW_WORKSPACE_PROMPT.txt for the size warnings.
echo
while IFS= read -r d; do
  [ -f "$d/SKILL.md" ] && continue
  say "$(basename -- "$d")" "not a skill (no SKILL.md) - reach by path"
  skipped=$((skipped+1))
done < <(find "$REPO" -maxdepth 1 -mindepth 1 -type d -not -name '.git' | sort)

if [ "$CURSOR" = 1 ]; then
  echo
  if [ -x "$REPO/kimi-k3/install_cursor_rules.sh" ] || [ -f "$REPO/kimi-k3/install_cursor_rules.sh" ]; then
    echo "running kimi-k3/install_cursor_rules.sh /sgl-workspace"
    [ "$DRY" = 1 ] || bash "$REPO/kimi-k3/install_cursor_rules.sh" /sgl-workspace
  else
    echo "kimi-k3/install_cursor_rules.sh not found - skipped"
  fi
fi

echo
echo "linked=$linked refreshed=$refreshed blocked=$blocked non-skill-dirs=$skipped"
[ "$blocked" -gt 0 ] && echo "re-run with --force to replace the blocked entries" || true
echo
echo "If skills still are not picked up, the prompt path always works:"
echo "  讀 $REPO/NEW_WORKSPACE_PROMPT.txt 並遵守裡面的規則"
