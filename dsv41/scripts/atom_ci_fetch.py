#!/usr/bin/env python3
"""Download an InferenceX ATOM AgentX run's per-point artifacts (agentic / bmk_agentic / server_logs) into
ref_atom_<run>/tp<TP>_c<C>/<kind>/, the newest attempt's artifact when a name repeats.
    atom_ci_fetch.py <run_id>"""
import io, json, os, re, subprocess, sys, zipfile

run = sys.argv[1]
out = f"/shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_atom_{run}"
os.makedirs(out, exist_ok=True)
arts = json.loads(subprocess.check_output(
    ["gh", "api", f"repos/SemiAnalysisAI/InferenceX/actions/runs/{run}/artifacts?per_page=100"]))["artifacts"]
latest = {}
for a in arts:
    if a["name"] not in latest or a["created_at"] > latest[a["name"]]["created_at"]:
        latest[a["name"]] = a
pat = re.compile(r"^(agentic|bmk_agentic|server_logs)_dsv41flash_tp(\d+)_conc(\d+)_")
with open(f"{out}/names.txt", "w") as f:
    f.write("\n".join(sorted(latest)) + "\n")
for name, a in sorted(latest.items()):
    m = pat.match(name)
    if not m or a.get("expired"):
        continue
    kind, tp, c = m.groups()
    dst = f"{out}/tp{tp}_c{c}/{kind}"
    if os.path.isdir(dst) and os.listdir(dst):
        print("skip", dst, flush=True)
        continue
    data = subprocess.check_output(
        ["gh", "api", f"repos/SemiAnalysisAI/InferenceX/actions/artifacts/{a['id']}/zip"])
    zipfile.ZipFile(io.BytesIO(data)).extractall(dst)
    print(f"ok {dst} {len(data) >> 20} MB", flush=True)
print("DONE", flush=True)
