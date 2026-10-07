#!/usr/bin/env python3
"""P3 sizing (FP4_INDEX_PLANE_PORT.md): SGLang decode scorer (A) vs row-group (B) at DSpark target-verify shapes.
Verify = speculative_num_draft_tokens=6 rows per request (server.log), ctx = ISL of the decode profiles (64k / 32k).
Reuses fp4_scorer_vs_sglang_bench.run (GPU time via CUDA-graph replay; COLD=1 for the cold-cache method).
  PYTHONPATH=/sgl-workspace/aiter-6145 HIP_VISIBLE_DEVICES=7 python3 fp4_verify_shapes_bench.py
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "fp4_scorer_vs_sglang_bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

Q = 6
for ratio in (2, 1):
    for bs, ctx in ((1, 65536), (4, 65536), (8, 65536), (16, 32768), (32, 32768), (64, 32768)):
        bench.run(f"verify bs{bs} ctx{ctx // 1024}k q{Q}", [(ctx - Q, Q)] * bs, ratio, is_decode=True)
