# Kimi-K3 ATOM C2-C64 stream A/B — 2026-08-14

## Workload

```text
input/output:       8192/1024 fixed
concurrency:        2, 4, 8, 16, 32, 64
num prompts:        concurrency * 8
warmup requests:    concurrency * 2
random range ratio: 1.0
ignore EOS:         enabled
CUDA graph mode:    FULL
tensor parallel:    8
```

Multi-stream used ATOM's default dual-stream MoE policy. Single-stream set:

```bash
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
```

## Results

Each entry is `total token throughput; median TPOT`.

```text
C2:  multi  756.23 tok/s, 22.87 ms | single  671.05 tok/s, 25.91 ms
C4:  multi 1391.64 tok/s, 24.40 ms | single 1240.28 tok/s, 27.47 ms
C8:  multi 2470.31 tok/s, 26.61 ms | single 2243.62 tok/s, 29.51 ms
C16: multi 4107.02 tok/s, 30.72 ms | single 3797.69 tok/s, 33.57 ms
C32: multi 6377.19 tok/s, 37.07 ms | single 5958.70 tok/s, 40.36 ms
C64: multi 8787.26 tok/s, 50.36 ms | single 8411.14 tok/s, 53.12 ms
```

Multi-stream throughput advantage versus single-stream:

```text
C2 +12.69% | C4 +12.20% | C8 +10.10%
C16 +8.15% | C32 +7.02% | C64 +4.47%
```

The dual-stream advantage is positive at every tested concurrency and
decreases as concurrency rises. Multi-stream also has lower median TPOT at
every point.

## Artifacts

```text
/workspace/kimi-k3-runs/c2-c64-8k1k-stream-ab-2026-08-14/
  summary.csv
  summary.json
  multi/c{2,4,8,16,32,64}-8k1k.{json,log}
  single/c{2,4,8,16,32,64}-8k1k.{json,log}
  multi/server.log
  single/server.log
```

All 12 measured runs completed successfully. No benchmark or server process
was left running.
