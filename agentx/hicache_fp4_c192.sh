#!/usr/bin/env bash
# FULL ARM: c192, FP4 indexer ON, HiCache CPU tier ON at ratio 3.0.
#
# WHY FP4 ON. This makes `fp4-dptbo-c192` (FP4 on, hicache off) a genuine
# single-variable partner: hicache is the only difference. The `hicache-smoke-c192`
# run had FP4 off, which was defensible for miss rate but not for tok/s (§11).
# It also means the FP4 indexer's *scale* pool is now in play, i.e. the exact
# condition the old "rust DeepseekV4C4IndexerScale" worry was about. Pre-checked
# and expected to build: `hybrid_pool_assembler.py:530` creates an
# `_IndexerRegion` for DEEPSEEK_V4_C4_INDEXER_SCALE alongside the payload region,
# and `:1343` registers its hit-policy pair. Zero `.rs` references to the name
# exist in this tree and no --hicache-storage-backend is set, so no rust tier is
# involved at all.
#
# WHY RATIO 3.0, not the launcher's 1.5. The smoke measured the host tier at
# **74.8 % full** (hicache_host_used_tokens 62.35 M of 83.38 M) after a single
# warmup pass, so at 1.5 the tier is plausibly the binding constraint rather than
# the mechanism being wrong. 1.5 cost 743.6 GB of a node with 2,955 GB available,
# so 3.0 (~1.5 TB, a bit more with FP4's extra scale pool) is comfortable.
# NOTE this keeps hicache-vs-no-hicache as the single variable against the
# baseline; only the smoke-to-arm comparison changes ratio.
#
# PASS CRITERION -- USE THE CORRECTED ONE (§12). The measurement-window
# token-weighted miss rate Sum(new)/Sum(new+cached) must fall from **8.15 %**
# toward **5.96 %** (c160's level), and TTFT must follow (baseline avg 59.71 s).
# Do NOT use the old "11.70 % -> below 8.6 %": both of those numbers were
# computed over the whole server.log, which includes a ~37 min cold-cache warmup
# that contributes about half of all `Prefill batch` lines. The baseline already
# beats 8.6 % on the honest window. This script computes the windowed number
# itself, at the end.
# If the miss rate falls but TTFT does not, prefill demand is not the driver and
# the mechanism in §9 is wrong.
#
# FREE SECOND RESULT (§4b, never yet verified): this is the first FP4 arm since
# 33979a814b's bounded prefill logits buffer. Driver-visible free VRAM should
# rise from the old FP4 arms' ~0.09 GB toward ~12 GB. The smoke (FP4 off) already
# showed median 8.36 GB.
set -u

HERE=/workspace/claude-skills/agentx
RESULT_DIR=/workspace/results/hicache-fp4-c192
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
export CONC=192 DURATION=3600 PORT=8888
export IS_AGENTIC=1

# THE VARIABLE UNDER TEST vs fp4-dptbo-c192.
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

# FP4 indexer ON, matching the fp4-dptbo-c192 baseline. No interval override:
# the launcher's own --prefill-decode-interval 10 stands, as in the baseline.
# The interval sweep is a separate track and must not be entangled with this one.
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer"

export RESULT_DIR
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c192"
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

echo "=== single variable vs fp4-dptbo-c192? FP4 must be ON, hicache ON ==="
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

echo "=== PASS CRITERION: measurement-window miss rate, 8.15 % -> target 5.96 % ==="
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
echo "  baseline fp4-dptbo-c192 = 8.15 % (windowed) | c160 target = 5.96 %"

echo "=== TTFT must follow (baseline avg 59.71 s, p50 23.02 s) ==="
python3 "$HERE/arm_report.py" hicache-fp4-c192 fp4-dptbo-c192

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

echo "=== §4b verification: first FP4 arm since 33979a814b. Free VRAM should be"
echo "    ~12 GB, not the old FP4 arms' 0.09 GB. Smoke (FP4 off) had median 8.36 GB ==="
awk -F, 'NR>1 {print $5}' "$RESULT_DIR/vram.csv" | sort -n \
    | awk '{a[NR]=$1} END {printf "  free_gb  n=%d min=%.2f p10=%.2f med=%.2f max=%.2f\n", NR, a[1], a[int(NR*0.1)+1], a[int(NR/2)], a[NR]}'
echo "  late Triton device loads: $(rg -c 'device-loaded after serving started' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)"
echo "=== host DRAM used, before -> after (GB); pool is demand-paged, expect << request ==="
paste <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_after.txt")
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
