#!/usr/bin/env bash
# POST-SWAP REPLICATE of hicache-fp4-int20-c128. Same six settings: c128, DP
# attention, TBO, FP4 indexer, HiCache CPU tier, `--prefill-decode-interval 20`.
#
# IT HAS TWO JOBS.
#
# 1. BRIDGE ARM. The 2026-09-03 image swap moved sglang, aiter, ROCm and torch
#    all at once, so every number in DATA_AND_ANALYSIS sec 1 is a record of the
#    old image and no cross-swap delta may be quoted until one arm has a
#    post-swap twin. Its pre-swap partner is `hicache-fp4-int20-c128`, which is
#    why this writes to a NEW result dir and does not overwrite it.
#
# 2. IT VALIDATES THE P0 OOR FIX (findings sec 17). This exact arm is the one
#    that ran with free VRAM p10 0.08 GB, min 0.01 GB and 10 late Triton device
#    loads -- the signature that killed fp4-dptbo-c64-reclaim0. Two things
#    landed since: the prefill index kernels are now preloaded at engine init,
#    and the fp8 logits rectangle is bounded. FP4 is ON here, so only the
#    preload half is exercised; the bounded fp8 buffer needs a non-FP4 arm.
#    The gate at the bottom is: late device loads == 0.
#
# Tree: sglang e485dc2436 (P0 fix) on 1e41776161 (PR #37660, resolved for the
# new base 2641e427be). The md5s below will differ from every pre-swap arm --
# that is the image swap, not tampering.
#
# --- original rationale, still the scientific point of the settings ---
#
# c128 + `--prefill-decode-interval 20` + FP4 indexer + HiCache CPU tier.
# The two levers we have are being combined for the first time.
#
# WHY THIS ARM EXISTS. Task 1's pass criterion was never met by interval tuning
# alone. At c128, interval 10 -> 20 moved ITL p90 78.39 -> 58.78 ms, PAST ATOM's
# 61.5 ms, but paid for it in TTFT: 8.50 -> 13.22 s, OVERSHOOTING ATOM's 10.9 s.
# The criterion is scored as a pair, so that is a fail, and the conclusion was
# "the optimum interval is between 10 and 20".
#
# HiCache attacks the other side of that trade directly. At c192 it cut the
# measurement-window prefill miss rate 8.15 -> 5.01 % and TTFT 59.71 -> 9.11 s
# (sec 13), i.e. it reduces exactly the prefill demand whose deferral is what
# interval 20 charges to TTFT. So the bet is that the interval's ITL win survives
# while hicache pays off its TTFT bill -- which would satisfy BOTH axes at once
# instead of needing an interval between 10 and 20.
#
# PASS CRITERION (unchanged from task 1, still scored as a PAIR):
#   ITL p90 <= 61.5 ms  AND  TTFT avg <= 10.9 s  SIMULTANEOUSLY.
# That is the reportable result: "SGLang matches ATOM on both axes at c128".
#   - If ITL holds near 58.8 ms and TTFT lands under 10.9 -> done, no interval
#     sweep needed at all.
#   - If TTFT lands between 10.9 and 13.2, hicache helped but not enough; THEN
#     sweep the interval down (15, then 12) with hicache left on.
#   - If ITL has drifted back above 61.5, the interval's effect does not survive
#     hicache and the two levers interact rather than add.
#
# NOT A SINGLE-VARIABLE ARM, state this when reporting. Against
# `interval20-c128` it adds TWO things, hicache and the FP4 indexer. That is
# deliberate -- the goal here is the ATOM pass criterion, not a clean delta. The
# FP4 half is the minor one: the only matched FP4 pair on the board is +2.57 % at
# c128, inside the 5.67 % replicate spread, i.e. null. If the arm passes and a
# clean attribution is then wanted, the follow-up is this arm with FP4 off.
#
# REFERENCE POINTS
#   interval20-c128 (FP4 off, hicache off): 29,130 tok/s, ITL 58.8 ms, TTFT 13.22 s
#   dptbo-c128      (interval 10, both off): 27,895 tok/s, ITL 78.4 ms, TTFT  8.50 s
#   ATOM c128:                               30,709 tok/s, ITL 61.5 ms, TTFT 10.9  s
#   hicache-fp4-c192:                        36,414 tok/s, ITL 99.3 ms, TTFT  9.11 s
#
# HICACHE_RATIO 3.0, same as both hicache arms. At c192 the tier ran 52.8 % full
# and was not the constraint; at c256 it hit 99.98 % and WAS (sec 14). c128 is a
# smaller working set than c192, so 3.0 has ample margin here.
set -u

HERE=/workspace/claude-skills/agentx
RESULT_DIR=/workspace/results/hicache-fp4-int20-c128-postswap
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

# SECOND GATE: VRAM, not just processes. After hicache-fp4-c256 finished the node
# had 0 matching processes yet ~14.5 GB held on EVERY one of the 8 GPUs, 116 GB
# total, and it did not drain. Even distribution over exactly 8 devices is a TP8
# job's signature, and this node is shared -- a process in another container's
# PID namespace is invisible to busy_count() but its allocation is not. (After
# hicache-fp4-c192 the node read 0.28 GB/GPU, so 14.5 GB is not normal residue.)
#
# This matters for validity, not just for OOM: MEM_FRACTION_STATIC=0.90 sizes the
# KV pool, so starting with 14.5 GB/GPU already gone yields a SMALLER pool than
# every arm on the board used, and the memory matching that the whole ATOM
# comparison rests on is broken. A quietly shrunk pool would look like a real
# result. So REFUSE rather than produce an uncomparable arm -- and, as with the
# process gate, never kill: we cannot tell whose allocation it is.
held_gb() {
    rocm-smi --showmeminfo vram 2>/dev/null \
        | grep -oE 'Total Used Memory \(B\): [0-9]+' \
        | awk '{s += $NF} END {printf "%.0f", s / 1e9}'
}
vram_ok=0
for _ in $(seq 1 60); do        # up to 30 min at 30 s
    h=$(held_gb)
    if [ "${h:-999}" -lt 10 ]; then vram_ok=$((vram_ok + 1)); else
        [ "$vram_ok" -ne 0 ] && echo "$(date '+%F %T') VRAM back up (${h} GB), streak reset"
        vram_ok=0
    fi
    [ "$vram_ok" -ge 3 ] && break
    sleep 30
done
if [ "$vram_ok" -lt 3 ]; then
    echo "FATAL: $(held_gb) GB still held across the 8 GPUs after 30 min with no"
    echo "visible process. Refusing: the KV pool would be smaller than every other"
    echo "arm's and the result would not be comparable. Do NOT kill blindly --"
    echo "check 'ps -eo pid,lstart,args' and ask before touching another session's job."
    exit 4
fi
echo "$(date '+%F %T') VRAM clear ($(held_gb) GB) -- launching"

cd /workspace/InferenceX
source "$HERE/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=128 DURATION=3600 PORT=8888
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
# The launcher hardcodes `--prefill-decode-interval 10` inside PARALLEL_ARGS
# (b200align_mtp.sh:235, expanded at :353) and expands EXTRA_ARGS later at :372,
# so this later occurrence is the one argparse keeps. BOTH 10 and 20 will appear
# in the resolved command; the last must be 20. Verified below and in server.log.
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer --prefill-decode-interval 20"

export RESULT_DIR
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }

(cd /sgl-workspace/sglang && git log -2 --format='%h %s' && git status --porcelain | wc -l) \
    >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
md5sum \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
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

echo "=== all four landed? FP4 ON, TBO ON, hicache ON, interval 20 ==="
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

echo "=== interval really 20? (both 10 and 20 appear; LAST one wins) ==="
tr ' ' '\n' <"$RESULT_DIR/sglang_command.txt" | grep -A1 -x -- '--prefill-decode-interval' | grep -v '^--' | tr '\n' ' '; echo
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
echo "  reference: interval20-c128 (no hicache) and dptbo-c128 for the c128 level"

echo "=== JOB 1, THE BRIDGE: how much did the image alone move? ==="
echo "    Same six settings, different image. Any delta here is image + our two"
echo "    patches, nothing else. Pre-swap partner: 7,187,200-token KV pool,"
echo "    ITL p90 57.98 ms, TTFT 13.77 s. A pool >2 % below that means VRAM was"
echo "    held at startup and this arm is NOT comparable -- check the gate log."
echo "    Replicate spread on this node is 5.67 %: a smaller tok/s delta is NULL."
python3 "$HERE/arm_report.py" hicache-fp4-int20-c128-postswap hicache-fp4-int20-c128
echo "=== the pair criterion, for reference: ITL p90 <= 61.5 ms AND TTFT <= 10.9 s ==="
echo "    (sec 15 already failed this pair on TTFT; re-anchor it post-swap)"
python3 "$HERE/arm_report.py" hicache-fp4-int20-c128-postswap interval20-c128 2>&1 | tail -12
echo "--- ATOM c128: 30,709 tok/s/chip, ITL p90 61.5 ms, TTFT 10.9 s ---"

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

echo "=== free VRAM. c192+hicache med 6.77 / min 2.71 GB; c256 med 2.97 / min 1.20."
echo "    c128 is the smallest working set of the three, so expect the most room ==="
awk -F, 'NR>1 {print $5}' "$RESULT_DIR/vram.csv" | sort -n \
    | awk '{a[NR]=$1} END {printf "  free_gb  n=%d min=%.2f p10=%.2f med=%.2f max=%.2f\n", NR, a[1], a[int(NR*0.1)+1], a[int(NR/2)], a[NR]}'

echo "=== JOB 2, THE P0 GATE (findings sec 17) ==="
echo -n "  preload ran?              "
rg -o 'Preloaded unified_kv prefill index kernels for compress ratios [^ ]*' \
    "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -3 || echo "ABSENT <!> preload never ran"
echo -n "  preload failed anywhere?  "
rg -c 'prefill index kernel preload failed' "$RESULT_DIR/server.log" 2>/dev/null || echo 0
LATE=$(rg -c 'device-loaded after serving started' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)
echo "  late Triton device loads: $LATE   (pre-swap partner: 10; PASS is 0)"
[ "$LATE" -eq 0 ] && echo "  -> P0 GATE PASS" || {
    echo "  -> P0 GATE FAIL. Which kernels, and with how much room:"
    rg -o "Triton kernel '[^']+' device-loaded after serving started \(free device mem: [^)]*\)" \
        "$RESULT_DIR/server.log" 2>/dev/null | sort | uniq -c | head -8
}
echo "  free VRAM above should no longer sit near zero: partner was p10 0.08 / min 0.01 GB"
echo "=== host DRAM used, before -> after (GB); pool is demand-paged, expect << request ==="
paste <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_after.txt")
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
