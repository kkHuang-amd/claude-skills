Owner node: pod4-gb300-2-tray01-f3

# Proxy MoE fixed_seq_len results on GB300 (tp4ep1, 12 layers, 4 GPUs)

Run 2026-10-21 02:16-05:40 UTC on pod4-gb300-2-tray01-f3; 25/25 points DONE, `report.py` printed
no warnings (completed = CONC*8 or CONC*4 for 70k, exact input/output lengths, no client errors).
Raw results: `/shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300/tp4ep1/isl*-osl*-c*/`.

## Configuration

| Item | Value |
|---|---|
| Hardware | 1 tray, 4x NVIDIA GB300 (284 GB), driver 595.58.03 |
| Software | SGLang 0.5.22 `baa9bef` + `../patches/sglang_quark_mxfp4_cuda.patch`, FlashInfer 0.7.1, torch 2.14.1+cu130 |
| Model | proxy MoE 12 layers, package dummy checkpoint (seed 0, Quark MXFP4 config unchanged) |
| Routed experts | MXFP4 weights x MXFP8 activations, FlashInfer trtllm-gen (`--moe-runner-backend flashinfer_mxfp4`) |
| Shared expert | MXFP4 W4A4, FlashInfer `mm_fp4` (cuDNN) |
| Attention / KV | BF16 / FP8 E4M3 |
| Parallelism | TP4, EP1, DP1 |
| Server | `--context-length 73728 --chunked-prefill-size 8192 --mem-fraction-static 0.88 --enable-flashinfer-allreduce-fusion`, `--max-running-requests` = `--cuda-graph-max-bs-decode` = CONC |
| Client | `bench_serving --backend sglang --dataset-name random-ids --tokenize-prompt --random-range-ratio 1.0 --request-rate inf`, np = CONC*8 (CONC*4 for 70k), warmup = CONC*2 |

TTT = total (input + output) tok/s; Interactivity = 1000 / median ITL; P90 interactivity = 1000 / p90 ITL.

## Tables

# 1024/1024  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1024 | 4,1,1 | 2 | 1002.067511 | 250.516878 | 4088.618068 | 80.514380 | 3.860865 | 259.009314 | 257.587365 |
| 1024 | 1024 | 4,1,1 | 4 | 1503.613596 | 375.903399 | 5451.536571 | 149.999082 | 5.179334 | 193.075017 | 191.408697 |
| 1024 | 1024 | 4,1,1 | 8 | 2040.420891 | 510.105223 | 8032.279424 | 206.873806 | 7.646290 | 130.782379 | 129.734498 |
| 1024 | 1024 | 4,1,1 | 16 | 2643.860951 | 660.965238 | 12393.199617 | 298.813056 | 11.727397 | 85.270414 | 84.534136 |
| 1024 | 1024 | 4,1,1 | 32 | 3507.277956 | 876.819489 | 18678.943279 | 368.008455 | 17.611707 | 56.780413 | 56.363593 |
| 1024 | 1024 | 4,1,1 | 64 | 5216.164439 | 1304.041110 | 25119.130285 | 658.069163 | 23.307442 | 42.904752 | 42.661563 |
| 1024 | 1024 | 4,1,1 | 128 | 8748.136992 | 2187.034248 | 29947.205899 | 1238.571259 | 26.835644 | 37.263872 | 37.070697 |
| 1024 | 1024 | 4,1,1 | 256 | 15064.295952 | 3766.073988 | 34753.542522 | 2422.026676 | 29.141553 | 34.315261 | 33.864109 |
| 1024 | 1024 | 4,1,1 | 512 | 20218.931171 | 5054.732793 | 51904.456446 | 4725.780750 | 39.702212 | 25.187513 | 19.454621 |
| 1024 | 1024 | 4,1,1 | 1024 | 17706.127454 | 4426.531863 | 118364.962588 | 9101.678493 | 95.317455 | 10.491258 | 8.171949 |

# 8192/1024  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1024 | 4,1,1 | 2 | 4344.256422 | 1086.064106 | 4256.819616 | 169.790444 | 3.856752 | 259.285567 | 257.839347 |
| 8192 | 1024 | 4,1,1 | 4 | 6243.986936 | 1560.996734 | 5916.078612 | 323.009301 | 5.182745 | 192.947927 | 191.628718 |
| 8192 | 1024 | 4,1,1 | 8 | 8167.373036 | 2041.843259 | 9034.849435 | 625.776306 | 7.642315 | 130.850412 | 129.771759 |
| 8192 | 1024 | 4,1,1 | 16 | 10202.923036 | 2550.730759 | 14452.797647 | 1232.595722 | 11.750024 | 85.106209 | 84.382982 |
| 8192 | 1024 | 4,1,1 | 32 | 12873.742259 | 3218.435565 | 22902.874406 | 2513.053716 | 17.641902 | 56.683231 | 56.261652 |
| 8192 | 1024 | 4,1,1 | 64 | 17519.066771 | 4379.766693 | 33660.765879 | 4862.231258 | 23.392758 | 42.748273 | 42.508607 |
| 8192 | 1024 | 4,1,1 | 128 | 25016.261335 | 6254.065334 | 47130.227399 | 9784.310839 | 27.068602 | 36.943171 | 36.768085 |
| 8192 | 1024 | 4,1,1 | 256 | 34068.102748 | 8517.025687 | 69202.741170 | 19433.268332 | 29.628208 | 33.751619 | 33.330989 |
| 8192 | 1024 | 4,1,1 | 512 | 39395.861378 | 9848.965344 | 119534.550329 | 38577.271114 | 38.328175 | 26.090467 | 19.296599 |
| 8192 | 1024 | 4,1,1 | 1024 | 36716.152968 | 9179.038242 | 256157.474444 | 76971.246808 | 97.158593 | 10.292450 | 8.379443 |

# 70000/300  tp4ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 70000 | 300 | 4,1,1 | 2 | 39675.758625 | 9918.939656 | 3866.033852 | 1377.659224 | 3.890141 | 257.060090 | 255.708003 |
| 70000 | 300 | 4,1,1 | 4 | 42382.097459 | 10595.524365 | 6935.084560 | 2821.569490 | 5.238536 | 190.893028 | 189.639954 |
| 70000 | 300 | 4,1,1 | 8 | 44230.736897 | 11057.684224 | 13018.918129 | 5493.958835 | 7.749852 | 129.034722 | 127.875717 |
| 70000 | 300 | 4,1,1 | 16 | 45563.160284 | 11390.790071 | 24981.488678 | 10799.805378 | 11.954301 | 83.651901 | 82.998256 |
| 70000 | 300 | 4,1,1 | 32 | 46998.089294 | 11749.522324 | 48119.124746 | 21412.145537 | 18.053742 | 55.390179 | 54.982516 |

## Notes

- Peak TTT/GPU: 1024/1024 5,055 at CONC=512; 8192/1024 9,849 at CONC=512; 70000/300 11,750 at CONC=32.
- CONC=1024 is slower than CONC=512 on both 1k and 8k shapes (median ITL 95-97 ms vs 38-40 ms).
  Not KV pressure (no retraction, full-token usage 13-55%) and decode stays on CUDA graph; server-side
  decode step at 1001 running requests is ~45 ms (gen 22.8k tok/s). The client ITL doubling is most
  likely prefill chunks (8192 tokens) interleaving with decode under continuous arrivals. Open item.
- Low-CONC ITL (3.9 ms at CONC=2) vs MI355X tp4ep1 smoke 4.49 ms (crsuse2-m2m-231, 2026-10-21).
- Server launch is 2-6 min per point (JIT + CUDA graph capture up to CONC); not in the measurement.
