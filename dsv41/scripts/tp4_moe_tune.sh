#!/usr/bin/env bash
# Tune the DSV4.1 TP4 EP1 fused MoE shapes that #5967 (TP2 only) does not cover -- see TP4_GAP_1006.md.
# Shapes = the 23 TP2 rows of dsv41_fp4_untuned_fmoe.csv with inter_dim 1152 -> 640 (TP4: 576 padded to 640):
#   640/385/7 tokens 1..4096 (target, shared experts fused) + 640/129/4 tokens 1..512 (DSpark draft); a8w4 per_1x32.
#   CHECK_ONLY=1 bash tp4_moe_tune.sh   # build the untuned CSV, check the tuner patch, print the command; no GPU use
#   bash tp4_moe_tune.sh                # tune on GPUS (default 0-7); refuses while an AgentX lane / sglang server runs
#   DEPLOY=1 bash tp4_moe_tune.sh       # back up and merge the tuned rows into aiter model_configs (restart servers)
set -euo pipefail
AITER=/sgl-workspace/aiter
CFG=$AITER/aiter/configs/model_configs
PATCH=/workspace/claude-skills/dsv41/patches/aiter_local_moe_tune_gpu_datagen.patch
TUNER=csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
OUT=${OUT:-/shared_nfs/kk/results/DeepSeek-V4.1-Flash/moe_tune_tp4}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(tr ',' '\n' <<< "$GPUS" | wc -l)
mkdir -p "$OUT"

if [ "${DEPLOY:-0}" = 1 ]; then
    [ -s "$OUT/tuned.csv" ] || { echo "no $OUT/tuned.csv"; exit 1; }
    ts=$(date +%m%d_%H%M); mkdir -p "$OUT/backup_$ts"
    cp "$CFG/dsv41_fp4_tuned_fmoe.csv" "$CFG/dsv41_fp4_untuned_fmoe.csv" "$OUT/backup_$ts/"
    for kind in tuned untuned; do
        src=$OUT/$kind.csv; dst=$CFG/dsv41_fp4_${kind}_fmoe.csv; n=0
        # the tuner may emit extra trailing columns (e.g. nt); keep the destination's column count
        ncol=$(head -1 "$dst" | awk -F, '{print NF}')
        while IFS= read -r row; do row=$(cut -d, -f1-"$ncol" <<< "$row"); grep -qxF "$row" "$dst" || { echo "$row" >> "$dst"; n=$((n+1)); }; done < <(tail -n +2 "$src")
        echo "$kind: +$n rows -> $dst"
    done
    echo "backup: $OUT/backup_$ts (restore by copying back). Restart sglang servers to pick up the rows."
    exit 0
fi

# 0. never tune next to a running benchmark (shared host memory / PCIe with the engram host table)
if pgrep -f agentx_lane.sh >/dev/null || pgrep -f '^sglang::' >/dev/null; then
    if [ "${CHECK_ONLY:-0}" = 1 ]; then echo "note: lane/server running (fine for CHECK_ONLY)"
    else echo "REFUSING: an AgentX lane or sglang server is running"; exit 1; fi
fi

# 1. untuned CSV for TP4 (header + TP2 rows with inter_dim 1152 -> 640)
U=$OUT/untuned.csv
head -1 "$CFG/dsv41_fp4_untuned_fmoe.csv" > "$U"
grep -E '^[0-9]+,5120,1152,(385,7|129,4),' "$CFG/dsv41_fp4_untuned_fmoe.csv" | sed 's/,5120,1152,/,5120,640,/' >> "$U"
echo "untuned shapes: $(($(wc -l < "$U") - 1)) (expect 23) -> $U"

# 2. tuner patch (GPU-side datagen; without it stage1 data is built on CPU and stalls ~40 min per shape)
if git -C "$AITER" apply -R --check "$PATCH" 2>/dev/null; then echo "tuner patch: already applied"
elif git -C "$AITER" apply --check "$PATCH" 2>/dev/null; then
    if [ "${CHECK_ONLY:-0}" = 1 ]; then echo "tuner patch: applies cleanly (not applied in CHECK_ONLY)"
    else git -C "$AITER" apply "$PATCH" && echo "tuner patch: applied"; fi
else echo "tuner patch: DOES NOT APPLY"; exit 1; fi

CMD=(python3 "$TUNER" -i "$U" -o "$OUT/tuned.csv" --mp "$NGPU")
echo "cmd: cd $AITER && HIP_VISIBLE_DEVICES=$GPUS ${CMD[*]} > $OUT/tune.log 2>&1"
[ "${CHECK_ONLY:-0}" = 1 ] && exit 0


cd "$AITER"
start=$(date +%s)
rc=0
HIP_VISIBLE_DEVICES=$GPUS "${CMD[@]}" > "$OUT/tune.log" 2>&1 || rc=$?
echo "tuner exit=$rc after $(( ($(date +%s) - start) / 60 )) min; log $OUT/tune.log"
grep -E 'Traceback|Error' "$OUT/tune.log" | tail -5 | cut -c1-200 || true

# 4. summary: every shape tuned, errors, kernels
[ -s "$OUT/tuned.csv" ] || { echo "no tuned.csv produced"; exit 1; }
python3 -I - "$U" "$OUT/tuned.csv" <<'EOF'
import csv, sys
key = lambda r: (r["token"], r["inter_dim"], r["expert"], r["topk"])
want = {key(r) for r in csv.DictReader(open(sys.argv[1]))}
got = {key(r): r for r in csv.DictReader(open(sys.argv[2]))}
print(f"tuned {len(want & got.keys())}/{len(want)}; missing {sorted(want - got.keys())}")
for k in sorted(want & got.keys(), key=lambda k: (k[2], int(k[0]))):
    r = got[k]
    print(f"  {k[2]}/{k[3]} tok {k[0]:>5}: {float(r['us']):8.1f} us ksplit {r['ksplit']} err {r['err1']}/{r['err2']} "
          f"{r['kernelName1'][:40]} | {r['kernelName2'][:40]}")
EOF
