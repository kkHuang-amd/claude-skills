#!/usr/bin/env bash
# Print the exact software stack a run used. Two runs are comparable only if
# everything here matches except the one thing under test.
SGL=${SGL_DIR:-/sgl-workspace/sglang}
AIT=${AITER_DIR:-/sgl-workspace/aiter}
INFX=${INFX_REPO:-/workspace/InferenceX-agentx}
echo "host        $(hostname)  $(date -u +%FT%TZ)"
echo "gpu         $(rocm-smi --showproductname 2>/dev/null | grep -m1 -oE 'Card Series:\s+\S.*' | sed 's/Card Series:\s*//') x$(rocm-smi --showid 2>/dev/null | grep -c 'Device ID')"
echo "torch       $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
v=$(pip show sglang 2>/dev/null | grep '^Version' | cut -d' ' -f2)
echo "sglang pkg  $v   (build commit = the g<hash> suffix)"
echo "sglang HEAD $(git -C $SGL rev-parse --short=10 HEAD)  dirty_tracked=$(git -C $SGL status --short --untracked-files=no | wc -l)"
v=$(pip show amd-aiter 2>/dev/null | grep '^Version' | cut -d' ' -f2)
echo "aiter pkg   $v"
echo "aiter HEAD  $(git -C $AIT rev-parse --short=10 HEAD)  dirty_tracked=$(git -C $AIT status --short --untracked-files=no | wc -l)"
git -C $AIT status --short --untracked-files=no | sed 's/^/  aiter M /'
echo "infx HEAD   $(git -C $INFX rev-parse --short=10 HEAD)"
echo "aiperf      $(git -C $INFX submodule status inferencex-e2e/utils/aiperf | cut -c2-11)"
