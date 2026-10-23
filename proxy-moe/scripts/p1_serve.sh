#!/usr/bin/env bash
# Fused FFN launch in serving, TP8, CONC 16 graphs (widths 1..16):
#   check : eager, SGLANG_AMD_MONO_CHECK=1, a few requests at CONC 1/4/16
#   base / mono : graphs on, ISL 1024 OSL 256 at CONC 1, 4, 16 (one variable: --monokernel-backend amd)
set -uo pipefail
B=/shared_nfs/kk/results/proxy_24l/c231_mono_p1_1022; mkdir -p $B
FS="$(cd "$(dirname "$0")/.." && pwd)/fixed_seq_len"
bench() { # $1 dir $2 conc $3 prompts $4 osl [$5 isl, default 1024]
  ( source /data/proxy_moe/mono_env.sh; cd /tmp && python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
     --tokenizer /data/proxy_moe/proxy_24l --dataset-name random-ids --tokenize-prompt --random-range-ratio 1.0 \
     --request-rate inf --warmup-requests 0 --random-input-len ${5:-1024} --random-output-len $4 --num-prompts $3 \
     --max-concurrency $2 --seed $((100 + $2)) --output-file $1/result.jsonl ) > $1/client_c$2${5:+_i$5}.log 2>&1; echo "$(basename $1) c$2 ${5:+isl$5 }client exit=$?"
}
run() { # $1 tag, $2 extra args, rest: "conc:prompts:osl[:isl]" ...
  local tag=$1 extra=$2; shift 2
  local RD=$B/$tag; mkdir -p $RD
  ENV_FILE=/data/proxy_moe/mono_env.sh EXTRA_ARGS="$extra" CONC=16 GPUS=${SGPUS:-0,1,2,3,4,5,6,7} \
    SERVER_LOG=$RD/server.log SERVER_PID_FILE=$RD/server.pid bash $FS/serve_proxy.sh > $RD/launch.log 2>&1 || { echo "$tag server failed"; tail -n 5 $RD/launch.log; }
  if [ -s $RD/server.pid ] && kill -0 $(cat $RD/server.pid) 2>/dev/null; then
    for c in "$@"; do IFS=: read conc np osl isl <<< "$c"; bench $RD $conc $np $osl $isl; done
  fi
  SPID=$(cat $RD/server.pid 2>/dev/null); [ -n "$SPID" ] && { kill $SPID; for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null; }
  left=$(ps -eo pid,args | grep -E '[s]glang::' | awk '{print $1}'); [ -n "$left" ] && kill -9 $left
  for i in $(seq 60); do b=$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): [0-9]+' | awk '$NF>20{c++}END{print c+0}'); [ "$b" -eq 0 ] && break; sleep 5; done
}
STEPS=${STEPS:-check base mono}
for s in $STEPS; do case $s in
  check) SGLANG_AMD_MONO_CHECK=${CHECK_MODE:-1} run check${CHECK_MODE:-} "--monokernel-backend amd --disable-cuda-graph" ${CHECK_LOADS:-1:2:16 4:8:16 16:16:16} ;;
  base)  run base "" 1:8:256 4:16:256 16:64:256 ;;
  mono)  run mono "--monokernel-backend amd" 1:8:256 4:16:256 16:64:256 ;;
  base_sweep) run base_sweep "" 2:8:256 3:12:256 5:20:256 6:24:256 8:32:256 ;;
  mono_sweep) run mono_sweep "--monokernel-backend amd" 2:8:256 3:12:256 5:20:256 6:24:256 8:32:256 ;;
  mono_nt)    SGLANG_AMD_MONO_LAGUNA_MAX_ROWS=16 run mono_nt16 "--monokernel-backend amd" 6:24:256 8:32:256 16:64:256 ;;
  mono_v2)    SGLANG_AMD_MONO_LAGUNA_MAX_ROWS=16 run mono_v2_16 "--monokernel-backend amd" 1:8:256 4:16:256 6:24:256 8:32:256 16:64:256 ;;
  mono_cap)   SGLANG_AMD_MONO_LAGUNA_MAX_ROWS=$CAP run mono_cap$CAP "--monokernel-backend amd" 1:8:256 4:16:256 8:32:256 16:64:256 ;;
  base_long)  run base_long "" 1:4:1024:4096 4:8:1024:4096 8:16:1024:4096 16:32:1024:4096 1:4:1024:8192 8:16:1024:8192 16:32:1024:8192 ;;
  mono_long)  run mono_long "--monokernel-backend amd" 1:4:1024:4096 4:8:1024:4096 8:16:1024:4096 16:32:1024:4096 1:4:1024:8192 8:16:1024:8192 16:32:1024:8192 ;;
  # TP4: the 12-layer checkpoint on GPUs 0-3 (T11), the launch built for TP4
  check_tp4)  SGLANG_AMD_MONO_LAGUNA_TP=4 SGLANG_AMD_MONO_CHECK=${CHECK_MODE:-1} MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run check_tp4 "--monokernel-backend amd --disable-cuda-graph" 1:2:16 4:8:16 16:16:16 ;;
  base_tp4)   MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 run base_tp4 "" 1:8:256 4:16:256 8:32:256 16:64:256 ;;
  mono_tp4)   SGLANG_AMD_MONO_LAGUNA_TP=4 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run mono_tp4 "--monokernel-backend amd" 1:8:256 4:16:256 8:32:256 16:64:256 ;;
  check_tp4p) SGLANG_AMD_MONO_LAGUNA_TP=4 SGLANG_AMD_MONO_CHECK=1 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run check_tp4_mspack "--monokernel-backend amd --disable-cuda-graph" 1:2:16 4:8:16 16:16:16 ;;
  mono_tp4p)  SGLANG_AMD_MONO_LAGUNA_TP=4 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run mono_tp4_mspack "--monokernel-backend amd" 1:8:256 4:16:256 8:32:256 16:64:256 ;;
  check_p)    SGLANG_AMD_MONO_CHECK=1 run check_tp8_mspack "--monokernel-backend amd --disable-cuda-graph" 1:2:16 4:8:16 16:16:16 ;;
  mono_p)     run mono_tp8_mspack "--monokernel-backend amd" 1:8:256 4:16:256 6:24:256 8:32:256 16:64:256 ;;
  # same-session A/B of a build: base and mono back to back, tags suffixed with $TAG
  ab_tp4)     MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 run base_tp4_$TAG "" 1:8:256 4:16:256 8:32:256 16:64:256
              SGLANG_AMD_MONO_LAGUNA_TP=4 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run mono_tp4_$TAG "--monokernel-backend amd" 1:8:256 4:16:256 8:32:256 16:64:256
              SGLANG_AMD_MONO_LAGUNA_TP=4 SGLANG_AMD_MONO_CHECK=1 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run check_tp4_$TAG "--monokernel-backend amd --disable-cuda-graph" 1:2:16 4:8:16 16:16:16 ;;
  ab_tp8)     run base_tp8_$TAG "" 1:8:256 4:16:256 6:24:256 8:32:256 16:64:256
              run mono_tp8_$TAG "--monokernel-backend amd" 1:8:256 4:16:256 6:24:256 8:32:256 16:64:256
              SGLANG_AMD_MONO_CHECK=1 run check_tp8_$TAG "--monokernel-backend amd --disable-cuda-graph" 1:2:16 4:8:16 16:16:16 ;;
  mono_tp8_only) run mono_tp8_$TAG "--monokernel-backend amd" 1:8:256 4:16:256 6:24:256 8:32:256 16:64:256 ;;
  chk_tp4)    SGLANG_AMD_MONO_LAGUNA_TP=4 SGLANG_AMD_MONO_CHECK=1 MODEL=/data/proxy_moe/proxy_12l TP=4 SGPUS=0,1,2,3 \
                run check_tp4_$TAG "--monokernel-backend amd --disable-cuda-graph" ${CHECK_LOADS:-1:2:16 4:8:16 16:16:16} ;;
  chk_tp8)    SGLANG_AMD_MONO_CHECK=1 run check_tp8_$TAG "--monokernel-backend amd --disable-cuda-graph" ${CHECK_LOADS:-1:2:16 4:8:16 16:16:16} ;;
  mono_tp8_pts) run mono_tp8_$TAG "--monokernel-backend amd" $LOADS ;;
esac; done
echo DONE
