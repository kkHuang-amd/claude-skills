#!/usr/bin/env python3
"""Compare gfx1250 register-multicast TopK v2 against the default ROCm path."""

import os
import sys

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)

SHAPES = (
    (1, 131072),
    (4, 131072),
    (1, 262144),
    (31, 131072),
    (64, 131072),
    (64, 262144),
)
TOPK = 512
PAGE_SIZE = 64


def set_cluster(enabled: bool) -> None:
    if enabled:
        os.environ["SGLANG_GFX1250_TOPK_CLUSTER"] = "1"
    else:
        os.environ.pop("SGLANG_GFX1250_TOPK_CLUSTER", None)


def bench_shape(batch: int, seq: int) -> tuple[float, float, int]:
    torch.manual_seed(batch * 100003 + seq)
    scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    out_off = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
    out_on = torch.full_like(out_off, -1)

    set_cluster(False)
    metadata_off = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    set_cluster(True)
    metadata_on = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()

    def run_off():
        set_cluster(False)
        topk_transform_paged_v2(
            scores, seq_lens, None, out_off, PAGE_SIZE, metadata_off
        )

    def run_on():
        set_cluster(True)
        topk_transform_paged_v2(
            scores, seq_lens, None, out_on, PAGE_SIZE, metadata_on
        )

    run_off()
    run_on()
    torch.cuda.synchronize()
    ref = torch.topk(scores, TOPK, dim=1, sorted=False).indices.cpu().tolist()
    off = out_off.cpu().tolist()
    on = out_on.cpu().tolist()
    for row, (expected, got_off, got_on) in enumerate(zip(ref, off, on)):
        expected_set = set(expected)
        if set(got_off) != expected_set:
            raise AssertionError(f"default mismatch at row {row}")
        if set(got_on) != expected_set:
            raise AssertionError(f"cluster mismatch at row {row}")

    kwargs = {
        "use_cuda_graph": False,
        "warmup_iters": 20,
        "replay_iters": 100,
        "metrics": (0.5, "avg"),
        "memory_args": (scores, seq_lens),
    }
    off_time = marker.do_bench(run_off, **kwargs).times[0] * 1e6
    on_time = marker.do_bench(run_on, **kwargs).times[0] * 1e6
    cluster_items = int(metadata_on[0, 1].item()) if batch > 30 else batch
    return off_time, on_time, cluster_items


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name} arch={getattr(props, 'gcnArchName', None)}")
    print("| batch | seq | cluster items | default us | cluster us | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    shapes = (
        ((int(sys.argv[1]), int(sys.argv[2])),)
        if len(sys.argv) == 3
        else SHAPES
    )
    for batch, seq in shapes:
        off, on, items = bench_shape(batch, seq)
        print(
            f"| {batch} | {seq} | {items} | {off:.2f} | {on:.2f} | {off / on:.3f}x |"
        )


if __name__ == "__main__":
    main()
