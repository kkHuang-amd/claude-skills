#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/cursor-rules" && pwd)"
WORKSPACE="${1:-/sgl-workspace}"
TARGET_DIR="$WORKSPACE/.cursor/rules"

test -d "$WORKSPACE" || {
  echo "error: workspace does not exist: $WORKSPACE" >&2
  exit 1
}

mkdir -p "$TARGET_DIR"
for source in "$SOURCE_DIR"/*.mdc; do
  install -m 0644 "$source" "$TARGET_DIR/$(basename "$source")"
done

echo "Installed Kimi-K3 Cursor rules:"
for source in "$SOURCE_DIR"/*.mdc; do
  target="$TARGET_DIR/$(basename "$source")"
  cmp --silent "$source" "$target"
  echo "  $target"
done
