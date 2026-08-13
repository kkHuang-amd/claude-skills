# Triton 3.7 gfx950 extend-attention N32 results — 2026-08-13

## Root cause

The Kimi-K3 late-prefill `extend_attention.py::_fwd_kernel` specialization is:

```text
Lq=576 / Lv=512
BLOCK_M=64 / BLOCK_N=64 / num_warps=4
```

Triton 3.7 raises this kernel from 483 to 512 VGPRs and introduces a
472-byte private segment with 186 scratch load/store instructions.

Rank0 trace:

```text
p50:  5.99 -> 13.78 ms (+130.03%)
total: 143.73 -> 330.90 ms
explained prefill span increase: 84.34%
```

## Isolated sweep

Matched shape:

```text
B=2
extend lengths=[8192, 7661]
prefix lengths=[0, 0]
Hq/Hkv=12/1
Lq/Lv=576/512
BF16 / gfx950 / Triton 3.7
```

Results:

```text
config       p50 ms   private/scratch          correctness
M64 N32 W4     5.24   0 bytes / 0 instructions pass, max abs 0.00390625
M64 N32 W8     6.45   12-32 bytes / 4-10        pass
M64 N64 W8     7.07   104-128 bytes / 44-56     pass
M32 N64 W4     8.27   0 / 0                     pass
M32 N32 W4     9.42   0 / 0                     pass
M32 N64 W8     9.49   0 / 0                     pass
M64 N64 W4    12.57   472-484 / 186-214         pass
M16 N64 W4    13.96   0 / 0                     pass
M16 N64 W8    14.27   0 / 0                     pass
M32 N128 W4   33.53   1712 / 618                pass
M32 N128 W8   84.66   2412 / 602                pass
```

Selected:

```text
BLOCK_M=64
BLOCK_N=32
num_warps=4
```

This keeps the query tile and launch count unchanged while reducing KV tile
register pressure.

## Triton 3.6 compatibility

The N32 tile is not universally better:

```text
Triton 3.6 N64: 5.34 ms
Triton 3.6 N32: 6.26 ms (+17.2%)
```

The production change is therefore version-gated to Triton >=3.7 and
independently controlled by:

```text
SGLANG_TRITON_37_EXTEND_LQ576_N32=1
```

Default remains off pending accuracy validation and policy review.

## Focused validation

```text
block-selection test: passed
Lq576/Lv512 N64-vs-N32 correctness test: passed
production-path isolated correctness: passed
max abs difference: 0.00390625
production-path p50: 5.24 ms
candidate VGPR count: 392-433
candidate private segment: 0
candidate scratch instructions: 0
```

## Endpoint validation

TP8, 8192/1024, 64 warmups, no radix cache:

```text
C    Triton 3.7 base   N32 candidate   handover target
2          970.53          973.94          968.57
4         1718.97         1748.71         1741.98
8         2839.40         2901.37         2881.25
16        4309.25         4460.33         4432.25
32        5881.93         6198.56         6191.41
```

Candidate improvement over Triton 3.7 baseline:

```text
C2  +0.35%
C4  +1.73%
C8  +2.18%
C16 +3.51%
C32 +5.38%
```

All requests succeeded. Capacity remained:

```text
max_total_num_tokens=933883
```

## Decision

Retain the version-gated N32 profile as the Triton 3.7 endpoint fix. It fully
recovers the C2-C32 performance matrix and is independently revertible.

Do not enable it by default until GSM8K accuracy validation is recorded.

## Artifacts

```text
/dockerx/var/amdsgl/kk/workspace/kimi-k3-runs/stage2-runs/
  2026-08-13-extend-attention-triton37-tuning/
```

Includes:

```text
bench_candidate.py
verify_production_path.py
sweep-summary.json
candidate JSON/log files
focused test evidence
C2-C32 endpoint JSONL/log files
server log
```
