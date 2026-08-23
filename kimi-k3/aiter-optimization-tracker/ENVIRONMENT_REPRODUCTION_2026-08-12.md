# Kimi-K3 environment reproduction — 2026-08-12

## Result

The fresh environment reproduced the selected vendored SGLang/core-only AITER
stack on 8x MI355X. Torch, Triton and `sglang-kernel` were not changed.

```text
Machine:        crsuse2-m2m-002.crusoe.amd.com
Torch:          2.9.1+rocm7.2.0.lw.git7e1940d4
Triton:         3.6.0+git42270451
HIP:            7.2.26015-fc0010cf6a
FlyDSL:         0.3.0
sglang-kernel:  0.4.6.post1

SGLang: f9dd3a0661b472d5fba1632adebcffc5c7c4021e
AITER:  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
```

Both repositories were clean after installation and validation.

## Installation

The repositories were cloned as:

```text
/sgl-workspace/sglang
/sgl-workspace/aiter
```

SGLang and AITER were installed editable with dependency resolution disabled.
AITER was installed with `AITER_USE_SYSTEM_TRITON=1`. The only approved
non-repository package change was:

```text
flydsl 0.2.4 -> 0.3.0
```

Pre/post package snapshots and installation logs are retained under:

```text
/workspace/kimi-k3-runs/
```

## Focused validation

```text
SGLang-vendored Kimi FlyDSL tests: 46 passed, 3 warnings
Elapsed:                            26.85 s
```

## Aligned production launch

The launch matched the recorded production command, with only current path
mapping:

```text
PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/aiter
model=/shared_nfs/models/Kimi-K3
AITER_JIT_DIR=/tmp/aiter-jit-vendor-core-0812
TP8, BF16, mem fraction 0.85
max running requests / graph BS = 256 / 256
radix cache disabled
Triton attention and prefill, AITER decode
PyTorch sampling
```

Capacity:

```text
recorded:     933883
reproduced:   934463
delta:           580 (+0.06%)
```

The first model read from NFS took 4147 seconds. A second launch with warm host
cache loaded weights in 132 seconds.

## Accuracy

```text
GSM8K 50:          1.000, invalid 0.000
GSM8K 200 run 1:   0.985, invalid 0.005
GSM8K 200 rerun:   0.985, invalid 0.005
recorded GSM8K 200 0.990
```

The 200-question result is one question below the recorded score and includes
one deterministic invalid response. This is retained as a minor unresolved
reproduction difference rather than silently replacing the recorded `0.990`.

## Fixed 8192/1024 endpoint

All measured requests succeeded. The anomalous first C4 pass (`790.16 tok/s`)
was retained but excluded: a warmed rerun produced `1764.84 tok/s`.

```text
C2:    980.24 tok/s  (+1.20% vs recorded  968.57), median TPOT 17.47 ms
C4:   1764.84 tok/s  (+1.31% vs recorded 1741.98), median TPOT 18.92 ms
C8:   2921.82 tok/s  (+1.41% vs recorded 2881.25), median TPOT 22.15 ms
C16:  4492.89 tok/s  (+1.37% vs recorded 4432.25), median TPOT 27.71 ms
C32:  6295.63 tok/s  (+1.68% vs recorded 6191.41), median TPOT 37.59 ms
```

## B2 profile

The pasted B2 command omitted `--random-range-ratio 1.0`; that run used
variable lengths and is retained only as diagnostic evidence. The comparable
fixed-length rerun added the same `1.0` setting as production.

```text
production C2: 980.24 tok/s, median TPOT 17.47 ms
B2 C2:        1073.02 tok/s, median TPOT 15.89 ms
delta:          +9.47% throughput

production C4: 1764.84 tok/s
B2 C4:         1770.31 tok/s
delta:           +0.31%
```

This reproduces the prior policy result: B2 materially improves C2 and is flat
at C4.

## Artifacts

```text
/workspace/kimi-k3-runs/focused-vendored-tests-2026-08-12.log
/workspace/kimi-k3-runs/aligned-production-server.log
/workspace/kimi-k3-runs/gsm8k-50.log
/workspace/kimi-k3-runs/gsm8k-200.log
/workspace/kimi-k3-runs/gsm8k-200-rerun.log
/workspace/kimi-k3-runs/production-endpoint/
/workspace/kimi-k3-runs/aligned-b2-server.log
/workspace/kimi-k3-runs/b2-endpoint/
```

No server or benchmark process was left running.
