#!/bin/bash
# e2e_queue.sh: sequential GSM8K e2e for each model-touching PR branch (see PLAN.md).
E=/workspace/claude-skills/dsv4-upstream-prs/scripts/e2e_one.sh
W=/tmp/prwt; PIN=/sgl-workspace/aiter_pin; OLD=/sgl-workspace/aiter
$E main_old  $W/main/python $OLD
$E P1_pin    $W/dsv4-fp8-unified-kv-decode/python $PIN
$E P1_old    $W/dsv4-fp8-unified-kv-decode/python $OLD
$E P2_old    $W/dsv4-fp8-verify-swa-ring/python $OLD
$E P4_old    $W/dsv4-c4-indexer-qlora-quant-reuse/python $OLD
$E P5_pin_rg $W/dsv4-fp4-indexer-rowgroup/python $PIN SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP=1
$E P6_pin_mx $W/dsv4-dense-fp8-mxscale/python $PIN SGLANG_DSV4_DENSE_MXSCALE=1
$E P7_old    $W/dsv4-intree-gemm-tune/python $OLD
echo QUEUE_DONE
