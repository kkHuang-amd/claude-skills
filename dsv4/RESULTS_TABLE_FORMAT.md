# DSV4 sweep results — preferred presentation format

When presenting a DSV4 serving sweep, render **one markdown table per parallelism
config** (`tp8`, then `tp8 + dp8`), each grouped by workload (1k/1k block, then
8k/1k block), rows sorted by concurrency.

## Columns (exact order)

| column | meaning / formula |
|--------|-------------------|
| `workload` | `{isl//1024}k/{osl//1024}k` (e.g. `8k/1k`) |
| `conc` | max concurrency |
| `total tok/s` | `total_throughput` (input+output), thousands comma-separated |
| `tok/s/gpu` | `total tok/s / 8` (TP=8) |
| `out tok/s` | `output_throughput` (decode only) |
| `TTFT ms` | median TTFT, 1 decimal |
| `TPOT ms` | median TPOT, 2 decimals |
| `ITL ms` | median ITL, 2 decimals |
| `interact (tok/s/u)` | `1000 / median_TPOT_ms`, 1 decimal — **goes right before E2E** |
| `E2E ms` | median end-to-end latency, comma-separated |

Notes:
- Put the dp8 launch knobs in the table header line, e.g.
  `tp8 + dp8 (--dp 8 --enable-dp-attention, --enable-prefill-delayer
  --prefill-delayer-max-delay-ms 5000, SGLANG_USE_ROCM700A=0)`.
- Insert a blank row between the 1k/1k and 8k/1k blocks.
- Always state the bench methodology once (ratio, num-prompts, warmups, ignore-eos)
  and whether the `intermediate_pad` fix is in.

## How to generate

```bash
python3 /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/summarize_sgl_dsv4.py \
  tp8=<result_dir_tp8> tp8dp8=<result_dir_tp8dp8>
```
The summarizer already emits all columns above (including `interact`); just
reformat its output into the markdown layout below.

---

## Worked example — SGLang ratio 0.8, np=conc*8, warm=conc*2, with intermediate_pad fix

Methodology: `bench_dsv4.py` (sglang-oai, ignore-eos on), random ratio 0.8,
num-prompts = conc*8, warmup = conc*2. Results dir `/workspace/bench_r08/`.

### tp8

| workload | conc | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | interact (tok/s/u) | E2E ms |
|---|---|---|---|---|---|---|---|---|---|
| 1k/1k | 2 | 274 | 34 | 136 | 163.8 | 14.52 | 14.33 | 68.9 | 13,478 |
| 1k/1k | 4 | 518 | 65 | 256 | 166.1 | 15.07 | 14.66 | 66.4 | 13,570 |
| 1k/1k | 8 | 944 | 118 | 470 | 165.6 | 16.46 | 15.44 | 60.8 | 14,895 |
| 1k/1k | 16 | 1,593 | 199 | 801 | 169.9 | 19.29 | 17.10 | 51.8 | 17,789 |
| 1k/1k | 32 | 2,494 | 312 | 1,254 | 177.1 | 24.58 | 20.32 | 40.7 | 23,000 |
| 1k/1k | 64 | 3,740 | 467 | 1,867 | 188.1 | 33.90 | 25.57 | 29.5 | 31,087 |
| 8k/1k | 4 | 2,227 | 278 | 242 | 337.2 | 15.89 | 14.92 | 63.0 | 14,322 |
| 8k/1k | 8 | 3,884 | 485 | 424 | 332.3 | 17.81 | 15.71 | 56.2 | 16,508 |
| 8k/1k | 16 | 6,276 | 785 | 692 | 336.1 | 22.14 | 17.35 | 45.2 | 20,801 |
| 8k/1k | 32 | 9,214 | 1,152 | 1,025 | 347.2 | 30.19 | 20.71 | 33.1 | 28,665 |
| 8k/1k | 64 | 12,289 | 1,536 | 1,360 | 361.2 | 45.31 | 25.92 | 22.1 | 42,145 |

### tp8 + dp8 (`--dp 8 --enable-dp-attention`, `--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000`, `SGLANG_USE_ROCM700A=0`)

| workload | conc | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | interact (tok/s/u) | E2E ms |
|---|---|---|---|---|---|---|---|---|---|
| 1k/1k | 64 | 3,894 | 487 | 1,944 | 782.4 | 31.20 | 28.81 | 32.1 | 29,378 |
| 1k/1k | 128 | 6,531 | 816 | 3,261 | 686.1 | 36.99 | 32.88 | 27.0 | 34,404 |
| 1k/1k | 256 | 10,602 | 1,325 | 5,306 | 542.4 | 45.99 | 38.20 | 21.7 | 42,755 |
| 1k/1k | 512 | 16,147 | 2,018 | 8,073 | 3,017.0 | 57.04 | 44.57 | 17.5 | 55,236 |
| 1k/1k | 1024 | 16,523 | 2,065 | 8,264 | 58,643.4 | 57.29 | 44.58 | 17.5 | 111,344 |
| 8k/1k | 64 | 11,798 | 1,475 | 1,306 | 1,932.9 | 45.97 | 28.87 | 21.8 | 43,774 |
| 8k/1k | 128 | 17,750 | 2,219 | 1,966 | 1,957.9 | 61.40 | 32.90 | 16.3 | 58,119 |
| 8k/1k | 256 | 23,821 | 2,978 | 2,646 | 1,945.3 | 92.74 | 38.09 | 10.8 | 88,008 |
| 8k/1k | 512 | 28,165 | 3,521 | 3,129 | 16,302.5 | 141.90 | 45.02 | 7.0 | 147,924 |
