#!/usr/bin/env python3
"""Task 8: compile the DSpark draft builds of the mono kernels (no launch, no TP group): K1 + K2 at ratio 0 with the
draft's routing (128 experts / top-3, ug_groups 1) and block-window keys (swa_width 192), plus the target default build
of the same width as a control.  PYTHONPATH=<sglang>/python HIP_VISIBLE_DEVICES=4 python3 mono_draft_build_check.py [S..]"""
import sys
import time

from sglang.srt.models.deepseek_common.amd.mono.kernels.layer import MonoBuild, build_mono_k1, build_mono_k2

for s in [int(x) for x in sys.argv[1:]] or [5]:
    for tag, kw in (("target", {}), ("draft", dict(experts=128, topk=3, ug_groups=1, swa_width=192))):
        b = MonoBuild(s, 4, 0, kv41=True, **kw)
        t0 = time.time()
        try:
            build_mono_k1(b), build_mono_k2(b)
            print(f"S={s} {tag}: K1+K2 built in {time.time() - t0:.1f}s")
        except Exception as e:  # report and keep going
            print(f"S={s} {tag}: FAILED {type(e).__name__}: {str(e)[:300]}")
