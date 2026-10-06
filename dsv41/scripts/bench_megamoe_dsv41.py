#!/usr/bin/env python3
"""MegaMoEV2 config microbench on DSv4.1-Flash shapes (any EP size), vs a TP-sharded fused_moe baseline.

Why: dep2_c32 decode profile (DEP_1005.md) put the whole DEP2-vs-TP2 decode gap in MegaMoE, whose config rules
(aiter mega_moe_config.py) were derived on DSv4-Pro EP8 (d7168 i3072, 48 experts/rank). This sweeps per-rank token
counts at a fixed MTPR and compares config variants built from those rules:
  default        what MegaMoEV2 picks (MTPR 16384 -> large-MTPR class, fp8 p2p quant)
  tn256          default with stage1 tile_n 256 (DSv4.1 inter 2304 = 9 x 256, but 4.5 x 512)
  bounded        the bounded-MTPR-class rule set for the same bucket (p2p quant none)
  bounded_tn256  bounded with stage1 tile_n 256
Each variant is checked against default (rel L2 of the output). TP baseline: all world*tokens tokens through
aiter fused_moe with inter_dim/world sharded experts (the non-DP TP<world> layout), with and without an RCCL
all-reduce (SGLang uses aiter custom AR, ~14 us at decode sizes, so tp_moe+ar is pessimistic).

  HIP_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 bench_megamoe_dsv41.py --tokens 8,16,24,32,48,64,128
  knobs: --mtpr 16384 --variants default,tn256,bounded,bounded_tn256 --model-dim 5120 --inter-dim 2304 --iters 50
Output: one [ROW] line per (tokens, variant) on rank 0; times are rank-mean / rank-max ms per MoE layer call.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "17179869184")
os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
sys.path.insert(0, "/sgl-workspace/aiter/op_tests/multigpu_tests")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import aiter  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as mcfg  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
import mori.shmem as ms  # noqa: E402
from bench_mega_moe_v2 import (  # noqa: E402
    SWIGLU_LIMIT,
    barrier,
    capture,
    make_inputs,
    make_weights,
    setup_dist,
    time_graph,
)


def variant_config(name, base, tokens, mtpr, model_dim, inter_dim):
    if name == "default":
        return base
    cfg = base
    if name.startswith("bounded"):
        bucket = mcfg.nearest_token_bucket(tokens)
        cfg = mcfg._select_bucket_config(bucket, min(mtpr, mcfg.P2P_FP8_MIN_MTPR), model_dim, inter_dim, False)
    if name.endswith("tn256"):
        cfg = replace(cfg, stage1=replace(cfg.stage1, tile_n=256))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,16,24,32,48,64,128", help="per-rank tokens (decode: reqs x 6)")
    ap.add_argument("--mtpr", type=int, default=16384)
    ap.add_argument("--variants", default="default,tn256,bounded,bounded_tn256")
    ap.add_argument("--model-dim", type=int, default=5120)
    ap.add_argument("--inter-dim", type=int, default=2304)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-tp", action="store_true")
    args = ap.parse_args()

    rank, world, device = setup_dist()
    epr = args.experts // world
    token_list = [int(t) for t in args.tokens.split(",") if t]
    variants = [v for v in args.variants.split(",") if v]

    w1, w1_s, w2, w2_s = make_weights(epr, args.model_dim, args.inter_dim, rank, device)
    mega = MegaMoEV2(rank=rank, world_size=world, model_dim=args.model_dim, inter_dim=args.inter_dim,
                     experts=args.experts, topk=args.topk, quant="a8w4", w1=w1, w1_scale=w1_s, w2=w2,
                     w2_scale=w2_s, max_tok_per_rank=args.mtpr, swiglu_limit=SWIGLU_LIMIT)
    default_select = mega._select_config
    tp_w = None
    if not args.no_tp:
        tp_w = make_weights(args.experts, args.model_dim, args.inter_dim // world, rank, device)

    for tokens in token_list:
        x, wts, ids = make_inputs(tokens, rank, world, args.model_dim, args.experts, args.topk,
                                  "uniform", 0.0, device)
        holder = {}
        ref = None
        for name in variants:
            def select(t, _name=name):
                base = default_select(t)
                cfg = variant_config(_name, base, t, args.mtpr, args.model_dim, args.inter_dim)
                mega._active_config = cfg
                return cfg

            mega._select_config = select
            err = ""
            try:
                out = mega(x, wts, ids).clone()
                barrier()
                if ref is None:
                    ref = out
                rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).reshape(1)
                dist.all_reduce(rel, op=dist.ReduceOp.MAX)

                def body():
                    holder["o"] = mega(x, wts, ids)

                g = capture(body)
                e2e = time_graph(g, args.iters, device)
                cfg = mega._active_config
                if rank == 0:
                    s1 = cfg.stage1
                    print(f"[ROW] world={world} tokens={tokens} variant={name} mega={e2e[0]:.4f}/{e2e[1]:.4f}ms "
                          f"rel_l2={rel.item():.2e} p2p={cfg.p2p_quant} sbm={s1.sort_block_m} tile_n={s1.tile_n} "
                          f"waves={s1.num_waves} s2_bm={cfg.stage2.block_m} s2_bn={cfg.stage2.block_n}", flush=True)
                del g
            except Exception as exc:  # report and keep sweeping
                err = f"{type(exc).__name__}: {str(exc)[:160]}"
                if rank == 0:
                    print(f"[ROW] world={world} tokens={tokens} variant={name} ERROR {err}", flush=True)
                barrier()
        mega._select_config = default_select

        if tp_w is not None:
            gt = tokens * world
            gx, gw, gids = make_inputs(gt, 0, world, args.model_dim, args.experts, args.topk, "uniform", 0.0,
                                       device)

            def tp_body(with_ar):
                def run():
                    out = fused_moe(gx, tp_w[0], tp_w[2], gw, gids, None, quant_type=aiter.QuantType.per_1x32,
                                    w1_scale=tp_w[1], w2_scale=tp_w[3], a1_scale=None, dtype=torch.bfloat16,
                                    swiglu_limit=SWIGLU_LIMIT, gate_mode=GateMode.INTERLEAVE.value)
                    if with_ar:
                        dist.all_reduce(out)
                    holder["t"] = out
                return run

            tp = time_graph(capture(tp_body(False)), args.iters, device)
            tpa = time_graph(capture(tp_body(True)), args.iters, device)
            if rank == 0:
                print(f"[ROW] world={world} tokens={tokens} variant=tp_fused_moe global_tokens={gt} "
                      f"tp_moe={tp[0]:.4f}/{tp[1]:.4f}ms tp_moe+rccl_ar={tpa[0]:.4f}/{tpa[1]:.4f}ms", flush=True)
        torch.cuda.empty_cache()

    ms.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
