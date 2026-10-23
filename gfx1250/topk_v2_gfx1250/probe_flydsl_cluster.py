#!/usr/bin/env python3
"""Probe gfx1250 FlyDSL cluster identity and barriers without remote LDS."""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T


def pointer(tensor: torch.Tensor):
    return flyc.from_c_void_p(fx.Int32, tensor.data_ptr())


def build_probe(use_barrier: bool):
    kernel_name = "probe_flydsl_cluster_barrier" if use_barrier else "probe_flydsl_cluster_identity"

    @flyc.kernel(name=kernel_name, known_block_size=[32, 1, 1])
    def kernel(output: fx.Pointer):
        tid = gpu.thread_idx.x
        _, rank_y = cluster.compute_cluster_position()

        if tid == fx.Int32(0):
            slot = fx.Int64(fx.index_cast(T.i32, rank_y))
            fx.ptr_store(fx.Int32(10) + fx.Int32(slot), output + slot)

        if use_barrier:
            cluster.cluster_barrier()
            if tid == fx.Int32(0):
                slot = fx.Int64(2) + fx.Int64(fx.index_cast(T.i32, rank_y))
                fx.ptr_store(fx.Int32(20) + fx.Int32(slot), output + slot)

    @flyc.jit
    def launch(output: fx.Pointer, stream: fx.Stream):
        kernel(output, value_attrs={"rocdl.cluster_dims": "1,2,1"}).launch(
            grid=(1, 2, 1),
            block=(32, 1, 1),
            cluster=(1, 2, 1),
            stream=stream,
        )

    return launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--barrier", action="store_true")
    args = parser.parse_args()

    output = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    launch = build_probe(args.barrier)
    flyc.compile(launch, pointer(output), stream)
    torch.cuda.synchronize()

    result = output.cpu().tolist()
    expected = [10, 11, 22, 23] if args.barrier else [10, 11, -1, -1]
    print(f"barrier={args.barrier} result={result}")
    if result != expected:
        raise SystemExit(f"expected {expected}")


if __name__ == "__main__":
    main()
