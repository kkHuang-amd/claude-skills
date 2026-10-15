#!/usr/bin/env bash
# c1 TPOT +2% after C: torch-profiler GPU trace of 40 steady-state decode steps, 8k c1, s5 (srt skips aiter's decode
# schedule for K2i-scored ratios) vs c (2d998c2496, srt back to the PR base). fw_ab.sh server config, GPUs 4-7, TP4.
# Output: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_rfc_c1/<variant>/ (torch traces), load out under agentx/.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_rfc_c1
for v in s5:/sgl-workspace/snap_rfc_s5_1014 c:/sgl-workspace/snap_rfc_c_1014; do
  name=${v%%:*}; mkdir -p "$P/$name"
  SGLANG_TORCH_PROFILER_DIR=$P/$name VARIANTS="$name:${v#*:}:MONO_ARG=--enable-monokernel" PHASES=ab ISLS=8192 REPS=1 \
    AB_CONCS=1 LOAD_EXTRA="--profile-dir $P/$name --profile-steps 40 --activities GPU" TAGP=rfcprof bash "$D/fw_ab.sh"
  echo "[$name] traces: $(find "$P/$name" -name '*.json*' | wc -l)"
done
echo "RFC C1 PROFILE DONE"
