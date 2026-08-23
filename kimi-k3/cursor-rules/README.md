# Durable Kimi-K3 Cursor rules

These files are the persistent source of truth. Cursor loads project rules only
from `<workspace>/.cursor/rules/`, so install them after creating or replacing
an image/workspace:

```bash
bash /workspace/claude-skills/kimi-k3/install_cursor_rules.sh /sgl-workspace
```

The installer is idempotent and verifies each copied rule. Edit the canonical
files in this directory, then rerun the installer; do not make durable edits
only in the generated workspace copies.
