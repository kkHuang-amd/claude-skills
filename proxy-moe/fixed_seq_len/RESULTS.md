Owner node: crsuse2-m2m-231

# Proxy MoE fixed_seq_len results — 2026-10-21, 8x MI355X (gfx950)

## Config

| | |
|---|---|
| model | proxy MoE (hidden 16384, 512 experts top-16 + shared, MXFP4 experts, BF16 attention, FP8 KV), dummy weights seed 0 |
| recipes | `tp8ep1` = TP8 EP1, **24 layers**, 8 GPUs; `tp4ep1` = TP4 EP1, **12 layers** (proxy spec for 4 GPUs), GPUs 0-3 |
| code | AITER `kkHuang-amd/aiter@feat/proxy-model-1020` (fd537aaa4), SGLang `HaiShaw/sglang@feat/proxy-model-1020` (effe332427), FlyDSL 0.4.0 |
| server | `--enable-aiter-allreduce-fusion`, `SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1`, tuned MoE rows (TP8 + TP4), `--max-running-requests CONC --cuda-graph-max-bs-decode CONC`, `--context-length 73728`, `--chunked-prefill-size 8192`, `--mem-fraction-static 0.88`, fp8 KV, `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`; one server per point |
| client | `sglang.bench_serving --backend sglang`, `random-ids --tokenize-prompt` (exact lengths), range-ratio 1.0, np = CONC*8 (CONC*4 for 70000/300), warmup = CONC*2, request-rate inf, ignore_eos |
| driver | `sweep_proxy.sh` (this dir), results `/shared_nfs/kk/results/proxy_24l/fixed_seq_len/` |
| run | 2026-10-21 01:18 → 07:43 UTC (49 points) + retry of 1 point 07:43 → 07:50 |

Columns: TTT = total_throughput (input + output tok/s); TTT/GPU = TTT / GPUs; Interactivity =
1000 / median ITL; P90 interactivity = 1000 / p90 ITL (tok/s/user). **TP8 and TP4 run different
model depths (24 vs 12 layers), so their TTT/GPU are not directly comparable.**

## Integrity

All 50 points DONE; `report.py` prints no warnings (completed = np, exact input / output lengths,
no client errors). One incident: the first run of `tp8ep1 isl8192-osl1024-c64` (02:47) failed when
the server died mid-decode (64 running, KV 6%) with no traceback, GPU fault or cgroup OOM; the
retry at 07:43 with identical flags completed normally (kept as `...c64.failed_0247`). Not reproduced.

## Tables

# 1024/1024  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1024 | 4,1,1 | 2 | 879.175642 | 219.793910 | 4661.169196 | 83.380500 | 4.439901 | 225.230247 | 221.460595 |
| 1024 | 1024 | 4,1,1 | 4 | 1320.391913 | 330.097978 | 6206.042853 | 143.952091 | 5.925494 | 168.762300 | 166.539181 |
| 1024 | 1024 | 4,1,1 | 8 | 1813.159183 | 453.289796 | 9035.182065 | 216.888538 | 8.620976 | 115.996147 | 114.819211 |
| 1024 | 1024 | 4,1,1 | 16 | 2360.508381 | 590.127095 | 13881.038508 | 254.301326 | 13.186035 | 75.837809 | 75.205878 |
| 1024 | 1024 | 4,1,1 | 32 | 3173.495999 | 793.374000 | 20647.859275 | 425.634583 | 19.458823 | 51.390570 | 50.945819 |
| 1024 | 1024 | 4,1,1 | 64 | 4640.904956 | 1160.226239 | 28234.954567 | 774.846288 | 26.186274 | 38.187945 | 37.902736 |
| 1024 | 1024 | 4,1,1 | 128 | 7744.785069 | 1936.196267 | 33834.446076 | 1452.589019 | 30.294310 | 33.009499 | 32.842878 |
| 1024 | 1024 | 4,1,1 | 256 | 13027.846255 | 3256.961564 | 40031.372421 | 2898.769091 | 33.611961 | 29.751314 | 29.390972 |
| 1024 | 1024 | 4,1,1 | 512 | 19877.636352 | 4969.409088 | 52603.612997 | 5610.919214 | 40.187364 | 24.883443 | 23.019184 |
| 1024 | 1024 | 4,1,1 | 1024 | 21490.448573 | 5372.612143 | 97405.123204 | 10806.658749 | 66.830240 | 14.963286 | 10.885433 |

# 8192/1024  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1024 | 4,1,1 | 2 | 3692.913650 | 923.228413 | 5008.251285 | 206.542756 | 4.542019 | 220.166407 | 218.836351 |
| 8192 | 1024 | 4,1,1 | 4 | 5365.950503 | 1341.487626 | 6885.943095 | 394.702989 | 6.028503 | 165.878660 | 164.740113 |
| 8192 | 1024 | 4,1,1 | 8 | 7117.661151 | 1779.415288 | 10370.968958 | 758.700262 | 8.740574 | 114.408962 | 113.668488 |
| 8192 | 1024 | 4,1,1 | 16 | 8964.010992 | 2241.002748 | 16457.489497 | 1464.932102 | 13.301343 | 75.180378 | 74.702064 |
| 8192 | 1024 | 4,1,1 | 32 | 11462.963820 | 2865.740955 | 25728.011236 | 2981.013568 | 19.579431 | 51.074007 | 50.651946 |
| 8192 | 1024 | 4,1,1 | 64 | 15364.862940 | 3841.215735 | 38387.796382 | 5744.551176 | 26.356151 | 37.941807 | 37.675643 |
| 8192 | 1024 | 4,1,1 | 128 | 21760.350204 | 5440.087551 | 54193.345642 | 11520.831881 | 30.709143 | 32.563592 | 32.400911 |
| 8192 | 1024 | 4,1,1 | 256 | 29245.126488 | 7311.281622 | 80646.883345 | 22930.916984 | 34.238950 | 29.206503 | 28.890133 |
| 8192 | 1024 | 4,1,1 | 512 | 34991.341001 | 8747.835250 | 134424.677606 | 45550.329181 | 41.700858 | 23.980322 | 21.447183 |
| 8192 | 1024 | 4,1,1 | 1024 | 37227.097835 | 9306.774459 | 241866.106579 | 76114.013597 | 68.526390 | 14.592918 | 10.323571 |

# 70000/300  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 70000 | 300 | 4,1,1 | 2 | 33382.085539 | 8345.521385 | 4602.224526 | 1713.269466 | 4.564363 | 219.088598 | 217.864592 |
| 70000 | 300 | 4,1,1 | 4 | 35634.843081 | 8908.710770 | 8258.293371 | 3416.318694 | 6.096292 | 164.034151 | 162.804025 |
| 70000 | 300 | 4,1,1 | 8 | 37300.246715 | 9325.061679 | 15435.616601 | 6590.214685 | 8.858138 | 112.890548 | 112.252847 |
| 70000 | 300 | 4,1,1 | 16 | 38446.677698 | 9611.669424 | 29576.326262 | 12933.083849 | 13.552588 | 73.786642 | 73.264088 |
| 70000 | 300 | 4,1,1 | 32 | 39676.819882 | 9919.204971 | 57020.169283 | 25617.499066 | 20.089482 | 49.777291 | 49.383060 |

# 1024/1024  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1024 | 8,1,1 | 2 | 627.413780 | 78.426722 | 6529.705277 | 115.413310 | 6.245469 | 160.116078 | 156.532027 |
| 1024 | 1024 | 8,1,1 | 4 | 1002.437391 | 125.304674 | 8162.098067 | 185.141310 | 7.796191 | 128.267766 | 125.877080 |
| 1024 | 1024 | 8,1,1 | 8 | 1491.843618 | 186.480452 | 10978.327160 | 281.928558 | 10.457814 | 95.622278 | 94.253097 |
| 1024 | 1024 | 8,1,1 | 16 | 2028.327134 | 253.540892 | 16152.246424 | 347.648090 | 15.284577 | 65.425428 | 64.748618 |
| 1024 | 1024 | 8,1,1 | 32 | 2779.754525 | 347.469316 | 23575.649762 | 577.762421 | 22.072156 | 45.305950 | 44.854906 |
| 1024 | 1024 | 8,1,1 | 64 | 4138.399666 | 517.299958 | 31660.668892 | 1046.505975 | 29.062291 | 34.408850 | 34.113886 |
| 1024 | 1024 | 8,1,1 | 128 | 6892.558259 | 861.569782 | 38018.510690 | 1973.142992 | 33.424033 | 29.918592 | 29.697006 |
| 1024 | 1024 | 8,1,1 | 256 | 11359.472457 | 1419.934057 | 46053.392616 | 3832.453540 | 37.603277 | 26.593427 | 26.183969 |
| 1024 | 1024 | 8,1,1 | 512 | 17045.911210 | 2130.738901 | 61608.412902 | 7432.667259 | 43.898464 | 22.779840 | 19.275153 |
| 1024 | 1024 | 8,1,1 | 1024 | 17544.006653 | 2193.000832 | 122220.858375 | 14555.714455 | 74.410313 | 13.438997 | 8.808421 |

# 8192/1024  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1024 | 8,1,1 | 2 | 2626.689404 | 328.336176 | 7039.085239 | 289.366385 | 6.405885 | 156.106455 | 155.404447 |
| 8192 | 1024 | 8,1,1 | 4 | 4057.203288 | 507.150411 | 9107.261417 | 664.519724 | 7.958503 | 125.651771 | 125.041060 |
| 8192 | 1024 | 8,1,1 | 8 | 5775.876705 | 721.984588 | 12780.862330 | 1028.350202 | 10.618508 | 94.175189 | 93.672542 |
| 8192 | 1024 | 8,1,1 | 16 | 7512.010712 | 939.001339 | 19640.685630 | 2101.265278 | 15.458754 | 64.688267 | 64.406206 |
| 8192 | 1024 | 8,1,1 | 32 | 9708.883424 | 1213.610428 | 30360.980756 | 4009.259123 | 22.217519 | 45.009525 | 44.660159 |
| 8192 | 1024 | 8,1,1 | 64 | 12990.313861 | 1623.789233 | 45407.401759 | 7828.479356 | 29.420927 | 33.989412 | 33.784294 |
| 8192 | 1024 | 8,1,1 | 128 | 18062.549124 | 2257.818641 | 65297.859335 | 15512.080007 | 33.953648 | 29.451916 | 29.332322 |
| 8192 | 1024 | 8,1,1 | 256 | 23496.140561 | 2937.017570 | 100449.947967 | 30725.570536 | 38.496546 | 25.976356 | 25.704962 |
| 8192 | 1024 | 8,1,1 | 512 | 27986.669022 | 3498.333628 | 168520.212595 | 61201.930394 | 45.474651 | 21.990273 | 21.294685 |
| 8192 | 1024 | 8,1,1 | 1024 | 28992.604147 | 3624.075518 | 321217.918289 | 118183.371402 | 73.956583 | 13.521447 | 10.131650 |

# 70000/300  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 70000 | 300 | 8,1,1 | 2 | 24730.204448 | 3091.275556 | 6198.321063 | 2276.279050 | 6.437271 | 155.345344 | 154.309727 |
| 70000 | 300 | 8,1,1 | 4 | 27053.446530 | 3381.680816 | 10878.705271 | 4513.406607 | 8.055601 | 124.137231 | 123.586586 |
| 70000 | 300 | 8,1,1 | 8 | 28590.778063 | 3573.847258 | 20150.365166 | 8744.860599 | 10.749589 | 93.026817 | 92.545143 |
| 70000 | 300 | 8,1,1 | 16 | 29534.569051 | 3691.821131 | 38526.767616 | 18141.894524 | 15.713770 | 63.638452 | 63.374663 |
| 70000 | 300 | 8,1,1 | 32 | 30449.518899 | 3806.189862 | 74188.268690 | 35004.057222 | 22.779255 | 43.899592 | 43.596004 |

## Observations

- Decode per user at low concurrency: ~155 tok/s (TP8, 24 layers) and ~220 tok/s (TP4, 12 layers).
  At CONC 1024 the median ITL reaches 67-74 ms and P90 interactivity drops to ~9-11 tok/s/user.
- 8192/1024: throughput saturates by CONC 512 (TP8 3,498 → 3,624 tok/s/GPU at 1024) while median
  TTFT doubles (61 s → 118 s): high-concurrency points are prefill-queue bound.
- 70000/300: TTT/GPU grows only ~23% from CONC 2 to 32 (TP8 3,091 → 3,806) and median TTFT scales
  ~linearly with CONC → prefill bound; this is where prefill work (T3/F) would pay off.
- 1024/1024 at CONC 1024 adds little over 512 (TP8 2,131 → 2,193 tok/s/GPU) and costs P90
  interactivity (19.3 → 8.8).

## 70000/300: `--chunked-prefill-size` 8192 vs 32768 (one variable, 2026-10-21 08:12-08:47 UTC)

All other launch flags identical to the main sweep (the main sweep uses one launch command for all
ISLs; only CONC changes). Chunk verified in every server log. Results:
`/shared_nfs/kk/results/proxy_24l/fixed_seq_len_chunk32768/` (`CHUNK=32768` env to `sweep_proxy.sh`).
One run per point; CONC 2 medians come from 8 requests.

| Recipe | CONC | TTT/GPU 8192 → 32768 | Median TTFT (ms) | Median ITL (ms) | P90 interactivity |
|---|---|---|---|---|---|
| tp8ep1 | 2 | 3,091.3 → 3,129.9 (+1.3%) | 2,276 → 2,924 (+28.5%) | 6.44 → 6.44 (+0.0%) | 154.3 → 154.4 (+0.1%) |
| tp8ep1 | 4 | 3,381.7 → 3,457.5 (+2.2%) | 4,513 → 4,896 (+8.5%) | 8.06 → 8.02 (-0.4%) | 123.6 → 124.0 (+0.3%) |
| tp8ep1 | 8 | 3,573.8 → 3,656.6 (+2.3%) | 8,745 → 8,330 (-4.7%) | 10.75 → 10.75 (+0.0%) | 92.5 → 92.5 (-0.0%) |
| tp8ep1 | 16 | 3,691.8 → 3,780.6 (+2.4%) | 18,142 → 16,555 (-8.7%) | 15.71 → 15.73 (+0.1%) | 63.4 → 63.3 (-0.1%) |
| tp8ep1 | 32 | 3,806.2 → 3,884.5 (+2.1%) | 35,004 → 33,220 (-5.1%) | 22.78 → 22.87 (+0.4%) | 43.6 → 43.4 (-0.4%) |
| tp4ep1 | 2 | 8,345.5 → 8,841.2 (+5.9%) | 1,713 → 2,051 (+19.7%) | 4.56 → 4.57 (+0.2%) | 217.9 → 216.7 (-0.5%) |
| tp4ep1 | 4 | 8,908.7 → 9,640.8 (+8.2%) | 3,416 → 3,422 (+0.2%) | 6.10 → 6.10 (+0.1%) | 162.8 → 162.8 (-0.0%) |
| tp4ep1 | 8 | 9,325.1 → 10,095.6 (+8.3%) | 6,590 → 5,863 (-11.0%) | 8.86 → 8.87 (+0.1%) | 112.3 → 112.0 (-0.2%) |
| tp4ep1 | 16 | 9,611.7 → 10,426.5 (+8.5%) | 12,933 → 11,680 (-9.7%) | 13.55 → 13.57 (+0.1%) | 73.3 → 73.2 (-0.1%) |
| tp4ep1 | 32 | 9,919.2 → 10,786.4 (+8.7%) | 25,617 → 23,278 (-9.1%) | 20.09 → 20.16 (+0.3%) | 49.4 → 49.2 (-0.3%) |

- Throughput up at every point: TP8 +1.3..2.4%, TP4 +5.9..8.7% (TP4 carries more prefill per GPU).
- Median TTFT -5..-11% at CONC >= 8, but WORSE at CONC 2 (TP8 +28.5%, TP4 +19.7%) and TP8 CONC 4
  (+8.5%); cause not investigated (small sample at CONC 2; repeat before concluding).
- Decode unchanged (ITL / P90 interactivity within ±0.5%).
- The main tables above keep chunk 8192 for every ISL; InferenceX derives chunk from ISL, so align
  the chunk setting before comparing with other platforms.

## Comparing with other platforms (e.g. GB300, `../fixed_seq_len_gb300/`)

Settings that differ between this sweep and `fixed_seq_len_gb300` (pod4-gb300-2-tray01-f3), to read
the GB300 vs MI355X tp4ep1 comparison correctly:

| Item | MI355X (this dir) | GB300 (`../fixed_seq_len_gb300/`) | Effect on the comparison |
|---|---|---|---|
| Routed-expert precision | MXFP4 W4A4 (MXFP4 activations) | MXFP4 weights x MXFP8 activations (W4A8; no W4A4 MoE kernel on that stack) | not the same precision or activation bytes; not an iso-precision comparison |
| Prefill all-reduce | `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (container env): lossy INT8 quick-reduce for prefill AR | no equivalent (bf16 AR) | MI355X prefill AR moves less data at some accuracy cost |
| AR fusion | `--enable-aiter-allreduce-fusion` | `--enable-flashinfer-allreduce-fusion` | platform equivalents |
| Router GEMM | `SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1` (prefill, fp32-accurate) | fp32 | numerically equivalent |
| Chunk / context / mem / KV / CONC binding / client | chunk 8192, ctx 73728, mem 0.88, fp8 KV, max-running = cuda-graph-max-bs-decode = CONC, same client | same | aligned |

- For an iso-precision number, either run MI355X with W4A8 routed experts or GB300 with W4A4 when a
  kernel exists; and run MI355X with `ROCM_QUICK_REDUCE_QUANTIZATION` unset (bf16 prefill AR).
- Measured: chunk 32768 raises MI355X tp4ep1 70000/300 TTT/GPU by 5.9-8.7% (section above). GB300
  has not run chunk 32768, so iso comparisons at 70k use chunk 8192 on both sides.
- Both sweeps use 12 layers for tp4ep1 (proxy spec); TP8 / 24-layer numbers here have no GB300
  single-tray counterpart (`../fixed_seq_len_gb300_2node/` tracks the two-tray run).

### Geomean MI355X / GB300, tp4ep1 (12 layers, 4 GPUs)

Every ratio is MI355X value / GB300 value, per point, then geometric mean over the points. TTT/GPU
and P90 interactivity: higher is better, so < 1 = MI355X behind. ITL and TTFT: lower is better, so
> 1 = MI355X slower. Full-precision values from `RESULTS.md` here and in `../fixed_seq_len_gb300/`
(25 matched points, both chunk 8192 unless noted).

| Shape / subset | points | TTT/GPU | P90 interactivity | Median ITL | Median TTFT |
|---|---|---|---|---|---|
| 1024/1024 | 10 | 0.923 | 0.946 | 1.067 | 1.091 |
| 8192/1024 | 10 | 0.885 | 0.928 | 1.084 | 1.172 |
| 70000/300 | 5 | 0.843 | 0.874 | 1.145 | 1.209 |
| all 3 shapes | 25 | 0.891 | 0.924 | 1.089 | 1.146 |

| Shape / subset | points | TTT/GPU | P90 interactivity | Median ITL | Median TTFT |
|---|---|---|---|---|---|
| 1024/1024, CONC <= 512 | 9 | 0.896 | 0.911 | 1.118 | 1.081 |
| 8192/1024, CONC <= 512 | 9 | 0.871 | 0.899 | 1.137 | 1.194 |
| all, CONC <= 512 | 23 | 0.874 | 0.898 | 1.131 | 1.151 |
| 70000/300, MI355X chunk 8192 (iso) | 5 | 0.843 | 0.874 | 1.145 | 1.209 |
| 70000/300, MI355X chunk 32768 vs GB300 chunk 8192 (NOT iso) | 5 | 0.909 | 0.872 | 1.147 | 1.178 |

- CONC=1024 is GB300's open item (GB300 behind there), so it lifts the MI355X ratio; CONC <= 512 is
  the cleaner view (TTT/GPU 0.874 over 23 points).
- The 70k chunk-32768 row mixes two variables (platform and chunk); GB300 has no chunk-32768 run.
  Quote the iso row (0.843) unless GB300 runs 70000/300 with chunk 32768 too.
- Precision / prefill-AR differences in the table above apply to every ratio here.

## tp4ep1 with `ROCM_QUICK_REDUCE_QUANTIZATION` unset (bf16 prefill AR, GB300-like), 2026-10-21/22

One variable vs the main tp4ep1 sweep (crsuse2-m2m-231): the container's
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8` unset for the server (verified absent in the server process
environ), everything else identical (`sweep_proxy.sh`, same flags/client). Results
`/shared_nfs/kk/results/proxy_24l/fixed_seq_len_qr_unset/tp4ep1/`; comparison script
`/data/proxy_moe/f2/iso_compare.py` (GB300 values parsed from `../fixed_seq_len_gb300/RESULTS.md`; the
INT8 column reproduces the geomeans above exactly). One run per point.

| ISL/OSL | CONC | TTT/GPU INT8 → bf16 AR | TTFT med | ITL med | P90 int. |
|---|---|---|---|---|---|
| 1024/1024 | 2 | 219.8 → 219.3 (-0.2%) | +1.5% | +0.2% | -0.2% |
| 1024/1024 | 4 | 330.1 → 324.9 (-1.6%) | +5.6% | +1.5% | -1.5% |
| 1024/1024 | 8 | 453.3 → 448.6 (-1.0%) | +9.1% | +0.8% | -0.8% |
| 1024/1024 | 16 | 590.1 → 589.5 (-0.1%) | +10.1% | -0.2% | +0.2% |
| 1024/1024 | 32 | 793.4 → 794.5 (+0.1%) | +11.6% | -0.6% | +0.6% |
| 1024/1024 | 64 | 1,160.2 → 1,156.7 (-0.3%) | +11.5% | -0.3% | +0.3% |
| 1024/1024 | 128 | 1,936.2 → 1,919.5 (-0.9%) | +12.6% | -0.2% | +0.1% |
| 1024/1024 | 256 | 3,257.0 → 3,216.9 (-1.2%) | +10.2% | +0.0% | +0.2% |
| 1024/1024 | 512 | 4,969.4 → 4,669.1 (-6.0%) | +13.7% | +0.7% | -15.5% |
| 1024/1024 | 1024 | 5,372.6 → 5,207.3 (-3.1%) | +14.8% | +2.7% | +0.4% |
| 8192/1024 | 2 | 923.2 → 899.2 (-2.6%) | +12.7% | +1.9% | -2.0% |
| 8192/1024 | 4 | 1,341.5 → 1,304.3 (-2.8%) | +13.5% | +1.7% | -1.7% |
| 8192/1024 | 8 | 1,779.4 → 1,738.5 (-2.3%) | +24.1% | +0.7% | -0.7% |
| 8192/1024 | 16 | 2,241.0 → 2,193.7 (-2.1%) | +12.7% | -0.0% | +0.1% |
| 8192/1024 | 32 | 2,865.7 → 2,800.9 (-2.3%) | +12.3% | -0.6% | +0.6% |
| 8192/1024 | 64 | 3,841.2 → 3,715.8 (-3.3%) | +12.4% | -0.5% | +0.5% |
| 8192/1024 | 128 | 5,440.1 → 5,189.1 (-4.6%) | +13.2% | -1.0% | +1.1% |
| 8192/1024 | 256 | 7,311.3 → 6,869.6 (-6.0%) | +12.6% | -1.1% | +1.4% |
| 8192/1024 | 512 | 8,747.8 → 8,067.5 (-7.8%) | +12.8% | -0.5% | -1.2% |
| 8192/1024 | 1024 | 9,306.8 → 8,450.0 (-9.2%) | +33.7% | -2.0% | +11.4% |
| 70000/300 | 2 | 8,345.5 → 7,685.0 (-7.9%) | +12.0% | +1.8% | -1.8% |
| 70000/300 | 4 | 8,908.7 → 8,129.2 (-8.7%) | +12.1% | +1.6% | -1.5% |
| 70000/300 | 8 | 9,325.1 → 8,468.9 (-9.2%) | +12.1% | +0.1% | -0.6% |
| 70000/300 | 16 | 9,611.7 → 8,697.1 (-9.5%) | +12.2% | +0.0% | +0.1% |
| 70000/300 | 32 | 9,919.2 → 8,938.9 (-9.9%) | +12.3% | -0.2% | +0.2% |

| Shape / subset | n | geomean bf16/INT8 TTT | GB300 ratio TTT (INT8) | GB300 ratio TTT (bf16 AR, iso) | TTFT ratio vs GB300 INT8 → bf16 |
|---|---|---|---|---|---|
| 1024/1024 | 10 | 0.986 | 0.923 | 0.910 | 1.091 → 1.200 |
| 8192/1024 | 10 | 0.957 | 0.885 | 0.846 | 1.172 → 1.357 |
| 70000/300 | 5 | 0.909 | 0.843 | 0.766 | 1.209 → 1.356 |
| all | 25 | 0.958 | 0.891 | 0.854 | 1.146 → 1.292 |
| all, CONC <= 512 | 23 | 0.960 | 0.874 | 0.840 | 1.151 → 1.288 |

- bf16 prefill AR costs TTFT +12% almost everywhere and TTT/GPU -1.4% (1k/1k) / -4.3% (8k/1k) / -9.1%
  (70k) geomean; decode (ITL) unchanged except noisy CONC-512/1024 points.
- Iso-AR MI355X / GB300 TTT/GPU geomean: 0.891 → 0.854 (all 25 points), 0.874 → 0.840 (CONC <= 512),
  70k 0.843 → 0.766. Routed-expert precision still differs (MI355X W4A4 vs GB300 W4A8), so this is
  iso-AR, not fully iso-precision.

## Regenerating

`python3 report.py /shared_nfs/kk/results/proxy_24l/fixed_seq_len [--md]`
