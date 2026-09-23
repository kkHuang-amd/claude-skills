#!/usr/bin/env bash
# c256, FP4 indexer ON, DP attention + TBO, HiCache CPU tier at ratio 3.0.
# Identical to hicache_fp4_c192.sh in EVERY resolved flag except CONC=256, so
# the pair hicache-fp4-c256 <-> hicache-fp4-c192 has concurrency as its single
# variable.
#
# WHY THIS ARM. `hicache-fp4-c192` produced 36,414 tok/s/chip with TTFT 9.11 s
# and ITL p90 flat (sec 13), beating the old c160 "knee" of 31,549 -- which means
# that knee was an artefact of running with no CPU cache tier, and the curve is
# still climbing. Two independent signals say c256 has room: `#queue-req` p90 is
# back to 7 (c160's value, versus 78 on the broken c192 baseline) and KV
# occupancy median is only 0.53. Both axes had headroom at c192.
#
# THE TARGET IS ATOM's c256: 44,722 tok/s/chip, ITL p90 97.6 ms, TTFT 13.4 s
# (ROCm/ATOM PR #2068). We are at 36,414 / 99.3 ms / 9.11 s at c192, i.e. already
# matched on ITL and 4.3 s BETTER on TTFT, needing +22.8 % throughput. This is
# the first arm on this node that can be compared to ATOM's best published point.
#
# RATIO STAYS AT 3.0 deliberately, even though c256 will push the tier harder.
# At c192 the host tier ran 52.8 % full on average (peak 69 %) with 0 dropped
# tokens, so 3.0 was NOT the binding constraint there and holding it fixed keeps
# concurrency as the only variable. If this arm reports the tier near full AND
# the miss rate has climbed, that is the signal to raise it -- the script prints
# both.
#
# WHAT TO WATCH, in priority order:
#   1. tok/s/chip vs 36,414 (c192) and 44,722 (ATOM c256).
#   2. TTFT avg. The c192 *baseline* collapse was queue wait (59.71 s); if c256
#      re-enters that regime, `#queue-req` p90 will leave ~7 and TTFT will skew
#      mean-over-median. That is the capacity ceiling reasserting itself at a
#      higher concurrency, not a hicache failure.
#   3. Measurement-window miss rate. c192+hicache = 5.01 %. Use the WINDOWED
#      number only; the whole-log figure is warmup-contaminated (sec 12).
#   4. Free VRAM. c192+hicache had median 6.77 GB / min 2.71 GB and 0 late
#      Triton loads. c256 raises KV pressure, and the fp8-path OOR cliff (sec 3)
#      is still unfixed, so a min approaching 0 with a late device load is the
#      known abort mode.
#
# CONTAINS the fix for the trap that made c192 look like a 95-minute loss:
# KV_OFFLOAD_BACKEND_METADATA. Without it the launcher exits 1 in the aggregation
# step AFTER a fully successful benchmark and writes no result JSON.
set -u

HERE=/workspace/claude-skills/agentx
RESULT_DIR=/workspace/results/hicache-fp4-c256
mkdir -p "$RESULT_DIR"

# Shared node. REFUSES rather than kills: a blind kill preamble destroyed
# another session's c96 arm at 05:12:25 on 2026-09-02. Bracketed patterns only,
# so this script's own command line is not matched.
busy_count() {
    ps -eo args | grep -Ec "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf (system_controller|profile)"
}
idle=0
for _ in $(seq 1 60); do        # up to 30 min at 30 s
    n=$(busy_count)
    if [ "$n" -eq 0 ]; then idle=$((idle + 1)); else
        [ "$idle" -ne 0 ] && echo "$(date '+%F %T') busy again ($n), idle streak reset"
        idle=0
    fi
    [ "$idle" -ge 3 ] && break
    sleep 30
done
if [ "$idle" -lt 3 ] ; then
    echo "FATAL: node still busy after 30 min ($(busy_count) procs) -- refusing to start"
    exit 3
fi
echo "$(date '+%F %T') node idle x3 -- launching"

cd /workspace/InferenceX
source "$HERE/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=256 DURATION=3600 PORT=8888
export IS_AGENTIC=1

# Held FIXED at the c192 arm's values; CONC is the variable.
# `KV_OFFLOADING=hicache` is INVALID and exits 1 before the model loads --
# benchmark_lib.sh:44-67 accepts only none|dram and requires the backend and a
# positive DRAM figure separately.
export KV_OFFLOADING="dram"
export KV_OFFLOAD_BACKEND="hicache"
# Undocumented THIRD requirement: process_agentic_result.py:89 needs metadata
# whose .name equals KV_OFFLOAD_BACKEND, or the arm exits 1 AFTER a fully
# successful benchmark and writes no result JSON.
export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
export TOTAL_CPU_DRAM_GB=2048      # declared budget only; this launcher merely
                                   # echoes it (:118) and it does not size the
                                   # pool. Actual host bytes are ratio x device
                                   # KV pool x TP. Must exceed the projection or
                                   # validation fails, hence 2048 for ratio 3.
export HICACHE_RATIO=3.0
export HICACHE_WRITE_POLICY=write_through
export HICACHE_IO_BACKEND=direct
export HICACHE_MEM_LAYOUT=page_first_direct

export ENABLE_TBO=1          # settled: TBO off is worse on ITL, TTFT and tok/s
export CHUNK_PER_RANK=16384
export HSA_NO_SCRATCH_RECLAIM=0
export MEM_FRACTION_STATIC=0.90   # do NOT lower: ATOM runs 0.9 and every arm on
                                  # the board is 0.90

# FP4 indexer ON, matching hicache-fp4-c192. No interval override:
# the launcher's own --prefill-decode-interval 10 stands, as in the baseline.
# The interval sweep is a separate track and must not be entangled with this one.
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer"

export RESULT_DIR
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c256"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }

(cd /sgl-workspace/sglang && git log -2 --format='%h %s' && git status --porcelain | wc -l) \
    >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
md5sum \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
    >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" 2>&1

free -g >"$RESULT_DIR/host_dram_before.txt"

bash "$HERE/vram_sampler.sh" "$RESULT_DIR/vram.csv" 15 &
SAMPLER_PID=$!
echo "vram sampler PID: $SAMPLER_PID"
trap 'kill -9 "$SAMPLER_PID" 2>/dev/null' EXIT

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"

free -g >"$RESULT_DIR/host_dram_after.txt"
kill -9 "$SAMPLER_PID" 2>/dev/null
trap - EXIT
sleep 90        # let aiperf finish its export / certification

# The launcher never kills its own server; it holds ~275 GB/GPU and the next arm
# would then sit in the idle gate and fail. Skip our own PID.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" | awk '{print $1}'); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
done
sleep 10

echo "=== single variable vs hicache-fp4-c192 (CONC only)? FP4 ON, hicache ON ==="
echo "fp4=$(grep -c -- '--enable-deepseek-v4-fp4-indexer' "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0) tbo=$(grep -c -- '--enable-two-batch-overlap' "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0) hicache=$(grep -c -- '--enable-hierarchical-cache' "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0)"
rg -o "'hicache_ratio': [0-9.]+|'enable_hierarchical_cache': [A-Za-z]+|'prefill_decode_interval': [0-9]+" \
    "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -5

echo "=== did the FP4 scale pool get a host mirror? (the old rust worry) ==="
rg -o "Allocating [0-9.]+ GB host memory for V4 paged pool '[a-z0-9_]+'" "$RESULT_DIR/server.log" 2>/dev/null \
    | sort | uniq -c | head -12
echo -n "host pool total: "
rg -o 'Allocating ([0-9.]+) GB host memory' -r '$1' "$RESULT_DIR/server.log" 2>/dev/null \
    | awk '{s += $1} END {printf "%.1f GB\n", s}'

echo "=== did it die? ==="
rg -c 'Traceback|HSA_STATUS_ERROR|Aborting with error|OutOfMemory' "$RESULT_DIR/server.log" 2>/dev/null || echo "no fatal patterns: 0"

echo "=== measurement-window miss rate (c192+hicache = 5.01 %; windowed only) ==="
python3 - "$RESULT_DIR/server.log" <<'PY'
import re, sys
pat = re.compile(r'(\d{2}):(\d{2}):(\d{2}).*#new-token: (\d+), #cached-token: (\d+)')
rows = []
for line in open(sys.argv[1], errors='ignore'):
    m = pat.search(line)
    if m:
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        rows.append((t, int(m.group(4)), int(m.group(5))))
if not rows:
    print("no Prefill batch lines"); raise SystemExit
# Measurement window = the last 3600 s of log activity. Anything earlier is the
# ~37 min cold-cache aiperf warmup, which contributes about half of all prefill
# batches at a far higher miss rate and invalidated the original criterion.
end = rows[-1][0]
for lo, label in ((end - 3600, "measurement window (last 3600 s)"), (0, "whole log (WARMUP-CONTAMINATED, do not quote)")):
    n = sum(r[1] for r in rows if r[0] >= lo)
    c = sum(r[2] for r in rows if r[0] >= lo)
    k = sum(1 for r in rows if r[0] >= lo)
    if n + c:
        print(f"  {label:52s} miss = {100 * n / (n + c):5.2f} %  over {k} batches")
PY
echo "  reference: hicache-fp4-c192 = 5.01 % | c160 = 5.96 % | broken c192 baseline = 8.15 %"

echo "=== headline vs c192+hicache: 36,414 tok/s, TTFT 9.11 s, ITL p90 99.3 ms ==="
python3 "$HERE/arm_report.py" hicache-fp4-c256 hicache-fp4-c192
echo "--- and against ATOM c256: 44,722 tok/s/chip, ITL p90 97.6 ms, TTFT 13.4 s ---"

echo "=== tier actually used? (smoke: 74.8 % full at ratio 1.5) ==="
python3 - "$RESULT_DIR/aiperf_artifacts/server_metrics_export.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))['metrics']
except Exception as e:
    print("  no server metrics:", e); raise SystemExit
for name in ('sglang:hicache_host_total_tokens', 'sglang:hicache_host_used_tokens',
             'sglang:hicache_dropped_tokens'):
    m = d.get(name)
    if not m:
        print(f"  {name}: ABSENT"); continue
    a = sum((s.get('stats') or {}).get('avg') or 0 for s in m.get('series', []))
    x = sum((s.get('stats') or {}).get('max') or 0 for s in m.get('series', []))
    print(f"  {name:36s} sum_avg={a:,.0f} sum_max={x:,.0f}")
tot = d.get('sglang:hicache_host_total_tokens'); use = d.get('sglang:hicache_host_used_tokens')
if tot and use:
    T = sum((s.get('stats') or {}).get('max') or 0 for s in tot['series'])
    U = sum((s.get('stats') or {}).get('avg') or 0 for s in use['series'])
    if T: print(f"  -> tier {100 * U / T:.1f} % full on average (raise HICACHE_RATIO again if ~full and the miss rate still has not landed)")
PY

echo "=== free VRAM: c192+hicache had med 6.77 / min 2.71 GB, 0 late Triton loads."
echo "    c256 raises KV pressure and the fp8 OOR cliff (sec 3) is still unfixed ==="
awk -F, 'NR>1 {print $5}' "$RESULT_DIR/vram.csv" | sort -n \
    | awk '{a[NR]=$1} END {printf "  free_gb  n=%d min=%.2f p10=%.2f med=%.2f max=%.2f\n", NR, a[1], a[int(NR*0.1)+1], a[int(NR/2)], a[NR]}'
echo "  late Triton device loads: $(rg -c 'device-loaded after serving started' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)"
echo "=== host DRAM used, before -> after (GB); pool is demand-paged, expect << request ==="
paste <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_after.txt")
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
