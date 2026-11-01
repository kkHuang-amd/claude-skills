#!/usr/bin/env python3
"""Would PR #13's stages fit a 1024-thread (16-wave, 128 VGPRs a wave) launch? Compiles (no launch) the
K3 mono KDA front (attention/kda.py) and layer_post (layer.py: o_proj, seam, MoE prologue) with
common.plan THREADS patched before any stage imports it; prints vgpr / agpr / scratch / lds.
Usage: THREADS=512|1024 mono_threads_isa.py [tokens=8]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tile_isa import regs  # noqa: E402  (sets COMPILE_ONLY)

import sgl_amd_mono.common.plan as plan  # noqa: E402

plan.THREADS = int(os.environ.get("THREADS", "512"))
plan.WAVES = plan.THREADS // 64

from sgl_amd_mono.kimi_k3.gfx950 import layer  # noqa: E402
from sgl_amd_mono.kimi_k3.gfx950.attention import kda  # noqa: E402

s = int(sys.argv[1]) if len(sys.argv) > 1 else 8
builds = {
    f"kda_pre_s{s}_ring": lambda: kda.build(kda.KdaPreBuild(tokens=s, qlen=s, nblocks=8, delta=True,
                                                            state_len=10, ring=True)),
    f"k2_s{s}": lambda: layer.build(layer.K2Build(tokens=s, nblocks=8)),
}
for name, mk in builds.items():
    try:
        for _, r in regs(mk()):
            print(f"THREADS={plan.THREADS} {name} {r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"THREADS={plan.THREADS} {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
