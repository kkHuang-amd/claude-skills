Owner node: pod4-gb300-2-tray05-f3

# Proxy MoE fixed_seq_len results on GB300 2-tray (tp8ep1, 24 layers, 8 GPUs, MNNVL)

Run 2026-10-21 06:58-11:15 UTC, driver on pod4-gb300-2-tray05-f3 (rank 0) + pod4-gb300-2-tray01-f3
(rank 1); 25/25 points DONE, `report.py` printed no warnings (completed = CONC*8 or CONC*4 for 70k,
exact input/output lengths, no client errors).
Raw results: `/shared_nfs/kk/results/proxy_24l/fixed_seq_len_gb300_2node/tp8ep1/isl*-osl*-c*/`.

## Configuration

| Item | Value |
|---|---|
| Hardware | 2 trays (same IMEX/NVL domain), 2x4 NVIDIA GB300 (284 GB), cross-tray NCCL P2P/MNNVL (no NET/Socket) |
| Software | SGLang 0.5.22 `baa9bef` + `../patches/sglang_quark_mxfp4_cuda.patch`, FlashInfer 0.7.1, torch 2.14.1+cu130 |
| Model | proxy MoE 24 layers, package dummy checkpoint (seed 0, Quark MXFP4 config unchanged), node-local copy per tray |
| Routed experts | MXFP4 weights x MXFP8 activations, FlashInfer trtllm-gen (`--moe-runner-backend flashinfer_mxfp4`) |
| Shared expert | MXFP4 W4A4, FlashInfer `mm_fp4` (cuDNN) |
| Attention / KV | BF16 / FP8 E4M3 |
| Parallelism | TP8, EP1, DP1, `--nnodes 2` (FlashInfer AR fusion `backend=mnnvl`; CustomAllreduce / multimem auto-disabled) |
| Server | `--context-length 73728 --chunked-prefill-size 8192 --mem-fraction-static 0.85 --enable-flashinfer-allreduce-fusion`, `--max-running-requests` = `--cuda-graph-max-bs-decode` = CONC |
| Client | `bench_serving --backend sglang --dataset-name random-ids --tokenize-prompt --random-range-ratio 1.0 --request-rate inf`, np = CONC*8 (CONC*4 for 70k), warmup = CONC*2 |

TTT = total (input + output) tok/s; Interactivity = 1000 / median ITL; P90 interactivity = 1000 / p90 ITL.
Only deviation from the 1-tray sweep besides topology: mem 0.85 instead of 0.88 (0.88 OOMs in prefill
CUDA graph capture on 2 trays, see README); KV pool 15.0M tokens, not binding.

## Tables

# 1024/1024  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1024 | 8,1,1 | 2 | 740.243134 | 92.530392 | 5535.267363 | 96.618971 | 5.243500 | 190.712310 | 189.721173 |
| 1024 | 1024 | 8,1,1 | 4 | 1127.578483 | 140.947310 | 7259.385230 | 201.131817 | 6.898320 | 144.962832 | 144.341448 |
| 1024 | 1024 | 8,1,1 | 8 | 1629.109328 | 203.638666 | 10057.262592 | 293.969675 | 9.540791 | 104.813112 | 104.119564 |
| 1024 | 1024 | 8,1,1 | 16 | 2191.670968 | 273.958871 | 14954.299720 | 320.295291 | 14.096418 | 70.940008 | 70.543343 |
| 1024 | 1024 | 8,1,1 | 32 | 2933.399243 | 366.674905 | 22332.772744 | 548.688119 | 20.851187 | 47.958900 | 47.724088 |
| 1024 | 1024 | 8,1,1 | 64 | 4378.240274 | 547.280034 | 29929.155996 | 1007.235621 | 27.359850 | 36.549908 | 36.405870 |
| 1024 | 1024 | 8,1,1 | 128 | 7232.029950 | 904.003744 | 36229.477959 | 1900.485510 | 31.700952 | 31.544794 | 31.421462 |
| 1024 | 1024 | 8,1,1 | 256 | 11955.660529 | 1494.457566 | 43804.367909 | 3761.113445 | 35.422193 | 28.230889 | 27.894645 |
| 1024 | 1024 | 8,1,1 | 512 | 16982.983209 | 2122.872901 | 61295.356139 | 7355.710290 | 43.276938 | 23.106995 | 18.445482 |
| 1024 | 1024 | 8,1,1 | 1024 | 15708.304244 | 1963.538030 | 133227.253021 | 14258.933944 | 103.013946 | 9.707423 | 7.873534 |

# 8192/1024  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1024 | 8,1,1 | 2 | 3153.390394 | 394.173799 | 5868.718322 | 255.175422 | 5.270099 | 189.749756 | 188.886164 |
| 8192 | 1024 | 8,1,1 | 4 | 4599.168639 | 574.896080 | 8028.099681 | 489.829571 | 6.924929 | 144.405823 | 143.799130 |
| 8192 | 1024 | 8,1,1 | 8 | 6334.485709 | 791.810714 | 11653.498072 | 957.223739 | 9.554096 | 104.667155 | 103.934708 |
| 8192 | 1024 | 8,1,1 | 16 | 8112.628921 | 1014.078615 | 18184.304462 | 1891.550935 | 14.115145 | 70.845889 | 70.431063 |
| 8192 | 1024 | 8,1,1 | 32 | 10230.516111 | 1278.814514 | 28825.813522 | 3761.211304 | 20.864167 | 47.929064 | 47.692353 |
| 8192 | 1024 | 8,1,1 | 64 | 13706.827960 | 1713.353495 | 43033.068248 | 7499.883525 | 27.417800 | 36.472656 | 36.327651 |
| 8192 | 1024 | 8,1,1 | 128 | 18832.764778 | 2354.095597 | 62618.742974 | 14858.208267 | 31.924595 | 31.323812 | 31.205037 |
| 8192 | 1024 | 8,1,1 | 256 | 24419.510245 | 3052.438781 | 96525.476220 | 29798.089619 | 35.961735 | 27.807334 | 27.532352 |
| 8192 | 1024 | 8,1,1 | 512 | 28200.227636 | 3525.028455 | 167284.486745 | 59734.190875 | 44.238926 | 22.604527 | 18.153370 |
| 8192 | 1024 | 8,1,1 | 1024 | 27387.665303 | 3423.458163 | 344460.967763 | 119012.756873 | 102.705649 | 9.736563 | 8.034244 |

# 70000/300  tp8ep1
| Input_len | output_len | TP,DP,EP | Concurrency | TTT (tok/s) | TTT/GPU (tok/s) | Median E2EL (ms) | Median TTFT (ms) | Median ITL (ms) | Interactivity (tok/s/user) | P90 interactivity (tok/s/user) |
|---|---|---|---|---|---|---|---|---|---|---|
| 70000 | 300 | 8,1,1 | 2 | 26964.279798 | 3370.534975 | 5713.129283 | 2089.032040 | 5.326210 | 187.750747 | 186.851555 |
| 70000 | 300 | 8,1,1 | 4 | 28703.229396 | 3587.903674 | 10273.784817 | 4298.626683 | 6.985883 | 143.145837 | 142.554560 |
| 70000 | 300 | 8,1,1 | 8 | 30040.213415 | 3755.026677 | 19190.352276 | 8367.141601 | 9.656537 | 103.556793 | 102.627223 |
| 70000 | 300 | 8,1,1 | 16 | 30876.896338 | 3859.612042 | 36890.406477 | 16468.897766 | 14.312627 | 69.868378 | 69.462958 |
| 70000 | 300 | 8,1,1 | 32 | 31665.087382 | 3958.135923 | 71453.625798 | 32650.243121 | 21.253591 | 47.050872 | 46.804156 |

## Notes

- Peak TTT/GPU: 1024/1024 2,123 at CONC=512; 8192/1024 3,525 at CONC=512; 70000/300 3,958 at CONC=32.
- CONC=1024 is slower than CONC=512 on both 1k and 8k shapes (median ITL ~103 ms vs 43-44 ms), same
  pattern as the 1-tray tp4ep1 sweep (open item there).
- Low-CONC ITL 5.24 ms at CONC=2 (1-tray tp4ep1 12L: 3.86 ms).
- vs 1-tray tp4ep1 12L: per-GPU work per step is the same, so with equal step time TP8 TTT/GPU would be
  0.5x TP4. Measured TP8/TP4 TTT/GPU geomean over 25 points: GB300 0.380, MI355X 0.403.

## GB300 vs MI355X, tp8ep1 (24 layers, 8 GPUs)

MI355X numbers from `../fixed_seq_len/RESULTS.md` (crsuse2-m2m-231, 2026-10-21, tp8ep1 tables). GB300
routed experts run W4A8 (MXFP8 activations), bf16 prefill AR, mem 0.85, 2 trays over MNNVL; MI355X W4A4,
INT8 quick-reduce prefill AR, single node. Same shapes, client and server flags otherwise.
Generated with `cmp_tp8.py` (geomean summary in README).

| ISL/OSL | CONC | TTT/GPU MI355X | TTT/GPU GB300 | GB300/MI355X | TTFT MI355X (ms) | TTFT GB300 (ms) | ITL MI355X (ms) | ITL GB300 (ms) | ITL ratio (MI/GB) |
|---|---|---|---|---|---|---|---|---|---|
| 1024/1024 | 2 | 78 | 93 | 1.18x | 115 | 97 | 6.25 | 5.24 | 1.19x |
| 1024/1024 | 4 | 125 | 141 | 1.12x | 185 | 201 | 7.80 | 6.90 | 1.13x |
| 1024/1024 | 8 | 186 | 204 | 1.09x | 282 | 294 | 10.46 | 9.54 | 1.10x |
| 1024/1024 | 16 | 254 | 274 | 1.08x | 348 | 320 | 15.28 | 14.10 | 1.08x |
| 1024/1024 | 32 | 347 | 367 | 1.06x | 578 | 549 | 22.07 | 20.85 | 1.06x |
| 1024/1024 | 64 | 517 | 547 | 1.06x | 1,047 | 1,007 | 29.06 | 27.36 | 1.06x |
| 1024/1024 | 128 | 862 | 904 | 1.05x | 1,973 | 1,900 | 33.42 | 31.70 | 1.05x |
| 1024/1024 | 256 | 1,420 | 1,494 | 1.05x | 3,832 | 3,761 | 37.60 | 35.42 | 1.06x |
| 1024/1024 | 512 | 2,131 | 2,123 | 1.00x | 7,433 | 7,356 | 43.90 | 43.28 | 1.01x |
| 1024/1024 | 1024 | 2,193 | 1,964 | 0.90x | 14,556 | 14,259 | 74.41 | 103.01 | 0.72x |
| 8192/1024 | 2 | 328 | 394 | 1.20x | 289 | 255 | 6.41 | 5.27 | 1.22x |
| 8192/1024 | 4 | 507 | 575 | 1.13x | 665 | 490 | 7.96 | 6.92 | 1.15x |
| 8192/1024 | 8 | 722 | 792 | 1.10x | 1,028 | 957 | 10.62 | 9.55 | 1.11x |
| 8192/1024 | 16 | 939 | 1,014 | 1.08x | 2,101 | 1,892 | 15.46 | 14.12 | 1.10x |
| 8192/1024 | 32 | 1,214 | 1,279 | 1.05x | 4,009 | 3,761 | 22.22 | 20.86 | 1.06x |
| 8192/1024 | 64 | 1,624 | 1,713 | 1.06x | 7,828 | 7,500 | 29.42 | 27.42 | 1.07x |
| 8192/1024 | 128 | 2,258 | 2,354 | 1.04x | 15,512 | 14,858 | 33.95 | 31.92 | 1.06x |
| 8192/1024 | 256 | 2,937 | 3,052 | 1.04x | 30,726 | 29,798 | 38.50 | 35.96 | 1.07x |
| 8192/1024 | 512 | 3,498 | 3,525 | 1.01x | 61,202 | 59,734 | 45.47 | 44.24 | 1.03x |
| 8192/1024 | 1024 | 3,624 | 3,423 | 0.94x | 118,183 | 119,013 | 73.96 | 102.71 | 0.72x |
| 70000/300 | 2 | 3,091 | 3,371 | 1.09x | 2,276 | 2,089 | 6.44 | 5.33 | 1.21x |
| 70000/300 | 4 | 3,382 | 3,588 | 1.06x | 4,513 | 4,299 | 8.06 | 6.99 | 1.15x |
| 70000/300 | 8 | 3,574 | 3,755 | 1.05x | 8,745 | 8,367 | 10.75 | 9.66 | 1.11x |
| 70000/300 | 16 | 3,692 | 3,860 | 1.05x | 18,142 | 16,469 | 15.71 | 14.31 | 1.10x |
| 70000/300 | 32 | 3,806 | 3,958 | 1.04x | 35,004 | 32,650 | 22.78 | 21.25 | 1.07x |

- CONC 2-256: GB300 is 1.04-1.20x on TTT/GPU and 1.05-1.22x on median ITL. The lead is largest at
  CONC=2 (1.18-1.20x for 1k/8k) and shrinks with CONC, unlike tp4ep1 where it stayed flat at 1.11-1.19x.
- 70000/300: TTT/GPU lead only 1.04-1.09x while ITL lead is 1.07-1.21x, i.e. prefill (TTFT) gains less
  at TP8 2-tray than decode.
- CONC 512: about parity (1.00x / 1.01x).
- CONC 1024: GB300 loses (1k/1k 0.90x, 8k/1k 0.94x); median ITL ~103 ms vs MI355X ~74 ms (0.72x),
  same open item as tp4ep1.
