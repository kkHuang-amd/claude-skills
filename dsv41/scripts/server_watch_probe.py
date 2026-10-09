#!/usr/bin/env python3
"""Log which pid fails InferenceX server_watch's health check (same rule: /proc/<pid>/stat state not Z/X and
start time unchanged), using the newest /tmp/inferencex-server-state.* snapshot. Polls every 2 s; logs snapshot switches
and each pid's first healthy -> failing transition (pids already dead when a snapshot is picked up are ignored). Usage: server_watch_probe.py <logfile>   (run under setsid nohup)"""
import glob, json, os, sys, time
log = open(sys.argv[1], "a", buffering=1)
cur = None
ok = {}
while True:
    files = glob.glob("/tmp/inferencex-server-state.*")
    if files:
        f = max(files, key=os.path.getmtime)
        if f != cur:
            cur, req, ok = f, json.load(open(f)), {}
            log.write(f"{time.strftime('%F %T')} snapshot {f} pids={list(req)}\n")
        for pid, start in req.items():
            try:
                st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                bad = st[0][0] in "ZX" or st[19] != start
                why = f"state={st[0]} start={st[19]} want={start}"
            except Exception as e:
                bad, why = True, f"read failed: {e!r}"
            if not bad:
                ok[pid] = True
            elif ok.get(pid):
                ok[pid] = False
                try: comm = open(f"/proc/{pid}/comm").read().strip()
                except Exception: comm = "?"
                log.write(f"{time.strftime('%F %T')} FAIL pid={pid} comm={comm} {why}\n")
    time.sleep(2)
