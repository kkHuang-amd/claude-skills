#!/usr/bin/env python3
"""FFN-only mono launch microbench (vLLM #60397 DSV4.1 kernels) at SGLang TP shapes, outside any serving framework.

Per rank: L layers with their own random weights (cold: L x ~1.9 GB at TP4 >> MALL), the FFN launch of each layer
captured back to back in one HIP graph, replayed N times -> us per layer for each width M.
Kernel source is loaded through a namespace shim (no vllm install needed).
Knobs (env): TP (4), GPUS ("4,5,6,7"), LAYERS (8), ITERS (50), WIDTHS ("1,2,4,6,8,12,16,24,32,48"), UNION (0), MONO_IMPORT (shim|sglang),
  MONO_VLLM_SRC (/tmp/mono_src/vllm/vllm/models/deepseek_v41/amd/mono), OUT (csv path, optional).
Output: one line per width "M=<m> us_per_layer=<x> finite=<bool>" from rank 0 (+ csv if OUT is set).
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MONO_IMPORT = os.environ.get("MONO_IMPORT", "shim")  # shim: pinned vLLM source via namespace map; sglang: vendored copy
PKG = "sglang.srt.models.deepseek_common.amd.mono.kernels" if MONO_IMPORT == "sglang" else "vllm.models.deepseek_v41.amd.mono"


def map_namespace(pkg: str, src_dir: str) -> None:
    parts = pkg.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(
        pkg, f"{src_dir}/__init__.py", submodule_search_locations=[src_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)


def rand_e4m3(shape, dev):
    return (torch.randn(shape, device=dev) * 0.5).to(torch.float8_e4m3fn)


def rand_e8m0(shape, dev):
    return torch.randint(118, 124, shape, dtype=torch.uint8, device=dev)


def rand_fp4(shape, dev):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)


def union_bias(E, dev):
    """UNION=k > 0: a large routing bias on k random experts, so every row's top-6 falls in those k (the step's expert
    union <= k, like real decode where rows share experts); 0: unbiased random routing (union ~ min(E, 6M))."""
    k = int(os.environ.get("UNION", "0"))
    b = torch.zeros(E, dtype=torch.float32, device=dev)
    if k > 0:
        g = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 1 << 30, (1,)).item()))
        b[torch.randperm(E, generator=g)[:k].to(dev)] = 100.0
    return b


def make_weights(tp, dev):
    import importlib

    runner = importlib.import_module(f"{PKG}.runner")
    MoeDims = importlib.import_module(f"{PKG}.stages.dims").Dims
    E = importlib.import_module(f"{PKG}.stages.moe_shape").EXPERTS
    H = 5120
    d = MoeDims(tp)
    f32 = dict(dtype=torch.float32, device=dev)
    w = runner.MonoLayerWeights(
        attn=None,
        hc_attn_fn=torch.randn(24, 4 * H, **f32) * 0.01,
        hc_attn_scale=torch.full((3,), 0.1, **f32),
        hc_attn_base=torch.zeros(24, **f32),
        attn_norm=torch.ones(H, dtype=torch.bfloat16, device=dev),
        hc_ffn_fn=torch.randn(24, 4 * H, **f32) * 0.01,
        hc_ffn_scale=torch.full((3,), 0.1, **f32),
        hc_ffn_base=torch.zeros(24, **f32),
        ffn_norm=torch.ones(H, dtype=torch.bfloat16, device=dev),
        gate_w=(torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16),
        bias=union_bias(E, dev),
        w13=rand_fp4((E, 2 * d.inter, H // 2), dev),
        w13_s=rand_e8m0((E * 2 * d.inter, H // 32), dev),
        w2=rand_fp4((E, H, d.inter // 2), dev),
        w2_s=rand_e8m0((E * H, d.down_scale_cols), dev),
        sgu=rand_e4m3((2 * d.sh_inter, H), dev),
        sgu_s=rand_e8m0((2 * d.sh_inter // 32, H // 32), dev),
        sw2=rand_e4m3((H, d.sh_inter), dev),
        sw2_s=rand_e8m0((H // 32, d.sh_inter // 32), dev),
    )
    w.check(tp)
    return w


def worker(rank, tp, gpus, layers, iters, widths, src, out):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.cuda.set_device(gpus[rank])
    dev = torch.device("cuda", gpus[rank])
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    if MONO_IMPORT != "sglang":
        map_namespace(PKG, src)
    import importlib

    runner = importlib.import_module(f"{PKG}.runner")
    torch.manual_seed(1234)  # replicated tensors (gate, bias, mHC params) must agree across ranks, as in the model
    mono = runner.DSV41MonoLayer(tp, rank, dist.group.WORLD, dev)
    ws = [make_weights(tp, dev) for _ in range(layers)]
    H, HC = 5120, 4
    rows = []
    for M in widths:
        part = torch.randn(M, H, device=dev).to(torch.bfloat16)
        res = torch.randn(M, HC, H, device=dev).to(torch.bfloat16)
        post = torch.full((M, HC, 1), 0.5, dtype=torch.float32, device=dev)
        comb = (torch.eye(HC, device=dev) / HC).expand(M, HC, HC).contiguous()
        pre = torch.full((M, HC), 0.25, dtype=torch.float32, device=dev)
        outs = [mono._outs(M, res) for _ in range(layers)]
        dist.barrier()
        for i in range(layers):  # eager first step: compile + scratch allocation, outside capture
            mono.ffn(ws[i], part, res, post, comb, pre, outs[i])
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                for i in range(layers):
                    mono.ffn(ws[i], part, res, post, comb, pre, outs[i])
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        us = t0.elapsed_time(t1) * 1000.0 / (iters * layers)
        finite = all(bool(torch.isfinite(o[0]).all()) for o in outs)
        t = torch.tensor([us], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"M={M} us_per_layer={t.item():.2f} finite={finite}", flush=True)
            rows.append((M, t.item(), finite))
        del g
        dist.barrier()
    if rank == 0 and out:
        with open(out, "w") as f:
            f.write("M,us_per_layer,finite\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.3f},{r[2]}\n")
    dist.barrier()
    dist.destroy_process_group()


def main():
    tp = int(os.environ.get("TP", "4"))
    gpus = [int(x) for x in os.environ.get("GPUS", "4,5,6,7").split(",")][:tp]
    layers = int(os.environ.get("LAYERS", "8"))
    iters = int(os.environ.get("ITERS", "50"))
    widths = [int(x) for x in os.environ.get("WIDTHS", "1,2,4,6,8,12,16,24,32,48").split(",")]
    src = os.environ.get("MONO_VLLM_SRC", "/tmp/mono_src/vllm/vllm/models/deepseek_v41/amd/mono")
    mp.spawn(worker, args=(tp, gpus, layers, iters, widths, src, os.environ.get("OUT", "")), nprocs=tp)


if __name__ == "__main__":
    main()
