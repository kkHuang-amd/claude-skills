"""Print the latest probe counters per pid (cumulative): python3 snap.py <probe.jsonl> [label]"""
import json, sys

last = {}
for line in open(sys.argv[1]):
    d = json.loads(line); last[d["pid"]] = d
label = sys.argv[2] if len(sys.argv) > 2 else ""
for pid, d in sorted(last.items()):
    c = {k: v for k, v in d.items() if k not in ("t", "pid", "tag") and not k.startswith("patched:")}
    print(label, pid, json.dumps(c, sort_keys=True))
