# Variants: 8 ranks (2 partitions/socket) vs 4 ranks one partition per socket (siblings idle) vs 2 ranks same socket / cross socket; gemm ~1 ms and light.
cd "$(dirname "$0")"; source <(grep -E '^(NCCL_|RCCL_)' /workspace/proxy_work/allon_tp8.env | sed 's/^/export /')
run() { echo "### $1"; CUDA_VISIBLE_DEVICES=$2 torchrun --nproc-per-node $3 --master-port 29611 mb_skew.py --gpus $2 ${@:4} 2>&1 | grep -E 'load=|compute|corr|Error|Traceback' | cut -c1-320; }
run "8r gemm n=2048"        0,1,2,3,4,5,6,7 8 --load gemm --size 2048 --iters 1500
run "4r 1-per-socket gemm"  0,2,4,6         4 --load gemm --size 2048 --iters 1500
run "2r same socket gemm"   0,1             2 --load gemm --size 2048 --iters 1500
run "2r cross socket gemm"  0,2             2 --load gemm --size 2048 --iters 1500


echo DONE
