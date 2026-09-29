#!/usr/bin/env bash
# Idempotent setup of the ATOM-port best-config environment (DSV4.1-Flash TP2, see RUNBOOK.md "ATOM-port worktree").
# Run AFTER scripts/setup_env.sh (which provides /sgl-workspace/sglang-dsv41, the aiter pin and the sgl-kernel rebuild).
# Creates / checks:
#   AP_SGL_DIR     worktree of AP_SGL_BASE at AP_SGL_COMMIT (RolaoDenthu/sglang dsv41/opt-branch), branch AP_BRANCH ('' = detached)
#   AP_AITER_DIR   worktree of AP_AITER_BASE at AP_AITER_COMMIT (ROCm/aiter PR #5750 head) + CK submodule
#               + patches/aiter_local_5750_worktree_0001.patch (#5561 LDS-DMA drain, MoE-tuner GPU data gen, 2 local fixes)
#               + tuned FMoE CSV patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv -> aiter/configs/model_configs/
#   AP_FLYDSL_DIR  flydsl 0.3.4.1 from PyPI (pip --no-deps --target; the image's global flydsl 0.3.2 stays untouched)
# Env knobs: AP_SGL_BASE AP_SGL_DIR AP_SGL_COMMIT AP_BRANCH AP_AITER_BASE AP_AITER_DIR AP_AITER_COMMIT AP_FLYDSL_DIR VERIFY_ONLY=1 (check only)
# Output: one line per step to stdout; exit 1 on the first failed step. First server start JIT-builds aiter-5750 modules.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
AP_SGL_BASE=${AP_SGL_BASE:-/sgl-workspace/sglang-dsv41}
AP_SGL_DIR=${AP_SGL_DIR:-/sgl-workspace/sglang-rolao-opt}
AP_SGL_COMMIT=${AP_SGL_COMMIT:-026da361c0e7baa9ff689d10a4c2c87aafc44d9d}
AP_BRANCH=${AP_BRANCH-atomport-mxfp8-producers}
AP_AITER_BASE=${AP_AITER_BASE:-/sgl-workspace/aiter}
AP_AITER_DIR=${AP_AITER_DIR:-/sgl-workspace/aiter-5750}
AP_AITER_COMMIT=${AP_AITER_COMMIT:-1053c79bb0bac2b7aecaf2da35c10c69a928bcc5}
AP_FLYDSL_DIR=${AP_FLYDSL_DIR:-/sgl-workspace/pydeps-flydsl-0341}
AP_FLYDSL_VER=0.3.4.1
AP_AITER_PATCH=$D/patches/aiter_local_5750_worktree_0001.patch
AP_FMOE_CSV=$D/patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv
AP_MODEL=/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash
[ -d /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash ] && AP_MODEL=/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash
V=${VERIFY_ONLY:-0}
ok(){ echo "OK   $*"; }
die(){ echo "FAIL $*"; exit 1; }
act(){ if [ "$V" = 1 ]; then echo "TODO $*"; return 1; fi; echo "DO   $*"; }

# 0. prerequisites
[ -d "$AP_SGL_BASE/.git" ] || die "missing $AP_SGL_BASE (run scripts/setup_env.sh first)"
[ -d "$AP_AITER_BASE/.git" ] || die "missing $AP_AITER_BASE (image aiter checkout)"
[ -d /sgl-workspace/mori ] || die "missing /sgl-workspace/mori (image component, used via PYTHONPATH)"
[ -f "$AP_MODEL/config.json" ] && ok "model $AP_MODEL" || die "model not found under /shared_nfs/{models/,}deepseek-ai/DeepSeek-V4.1-Flash"
[ -f "$AP_AITER_PATCH" ] && [ -f "$AP_FMOE_CSV" ] || die "missing $AP_AITER_PATCH or $AP_FMOE_CSV"

# 1. sglang worktree (rolao opt-branch)
if ! git -C "$AP_SGL_BASE" remote | grep -qx rolao; then
  act "git remote add rolao" && git -C "$AP_SGL_BASE" remote add rolao https://github.com/RolaoDenthu/sglang.git
fi
if ! git -C "$AP_SGL_BASE" cat-file -e "$AP_SGL_COMMIT^{commit}" 2>/dev/null; then
  act "fetch rolao dsv41/opt-branch" && git -C "$AP_SGL_BASE" fetch -q rolao dsv41/opt-branch
  git -C "$AP_SGL_BASE" cat-file -e "$AP_SGL_COMMIT^{commit}" 2>/dev/null || die "commit $AP_SGL_COMMIT not in rolao dsv41/opt-branch"
fi
if [ -e "$AP_SGL_DIR/.git" ]; then
  [ "$(git -C "$AP_SGL_DIR" rev-parse HEAD)" = "$AP_SGL_COMMIT" ] && ok "sglang $AP_SGL_DIR @ ${AP_SGL_COMMIT:0:10}" \
    || echo "WARN sglang $AP_SGL_DIR HEAD $(git -C "$AP_SGL_DIR" rev-parse --short HEAD) != ${AP_SGL_COMMIT:0:10} (left as is)"
elif act "git worktree add $AP_SGL_DIR ${AP_SGL_COMMIT:0:10} ${AP_BRANCH:-(detached)}"; then
  if [ -z "$AP_BRANCH" ]; then git -C "$AP_SGL_BASE" worktree add -q --detach "$AP_SGL_DIR" "$AP_SGL_COMMIT"
  elif git -C "$AP_SGL_BASE" rev-parse -q --verify "refs/heads/$AP_BRANCH" >/dev/null; then
    git -C "$AP_SGL_BASE" worktree add -q "$AP_SGL_DIR" "$AP_BRANCH" && git -C "$AP_SGL_DIR" reset -q --hard "$AP_SGL_COMMIT"
  else git -C "$AP_SGL_BASE" worktree add -q -b "$AP_BRANCH" "$AP_SGL_DIR" "$AP_SGL_COMMIT"; fi
  [ "$(git -C "$AP_SGL_DIR" rev-parse HEAD)" = "$AP_SGL_COMMIT" ] && ok "sglang $AP_SGL_DIR @ ${AP_SGL_COMMIT:0:10}" || die "sglang worktree"
fi

# 2. aiter worktree (PR #5750 head + local patch + tuned CSV)
if ! git -C "$AP_AITER_BASE" cat-file -e "$AP_AITER_COMMIT^{commit}" 2>/dev/null; then
  act "fetch aiter pull/5750" && git -C "$AP_AITER_BASE" fetch -q origin "refs/pull/5750/head:refs/remotes/pr/5750"
  git -C "$AP_AITER_BASE" cat-file -e "$AP_AITER_COMMIT^{commit}" 2>/dev/null || \
    { git -C "$AP_AITER_BASE" fetch -q origin "$AP_AITER_COMMIT" || die "aiter commit $AP_AITER_COMMIT not fetchable"; }
fi
if [ ! -e "$AP_AITER_DIR/.git" ]; then
  act "git worktree add $AP_AITER_DIR ${AP_AITER_COMMIT:0:9}" && git -C "$AP_AITER_BASE" worktree add -q --detach "$AP_AITER_DIR" "$AP_AITER_COMMIT" \
    || die "aiter worktree"
fi
if [ -e "$AP_AITER_DIR/.git" ]; then
  [ "$(git -C "$AP_AITER_DIR" rev-parse HEAD)" = "$AP_AITER_COMMIT" ] || die "aiter $AP_AITER_DIR HEAD != ${AP_AITER_COMMIT:0:9}"
  if git -C "$AP_AITER_DIR" submodule status 3rdparty/composable_kernel | grep -q '^-'; then
    act "CK submodule init" && git -C "$AP_AITER_DIR" submodule update -q --init 3rdparty/composable_kernel
  fi
  git -C "$AP_AITER_DIR" submodule status 3rdparty/composable_kernel | grep -q '^ ' && ok "aiter CK submodule" \
    || echo "TODO aiter CK submodule not initialised"
  if git -C "$AP_AITER_DIR" apply --reverse --check "$AP_AITER_PATCH" 2>/dev/null; then ok "aiter local patch applied"
  elif git -C "$AP_AITER_DIR" apply --check "$AP_AITER_PATCH" 2>/dev/null; then
    act "apply $(basename "$AP_AITER_PATCH")" && git -C "$AP_AITER_DIR" apply "$AP_AITER_PATCH" && ok "aiter local patch applied"
  else die "aiter local patch neither applied nor applicable (tree has other edits?)"; fi
  DST=$AP_AITER_DIR/aiter/configs/model_configs/dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv
  if cmp -s "$AP_FMOE_CSV" "$DST"; then ok "tuned FMoE CSV"
  else act "install tuned FMoE CSV" && cp "$AP_FMOE_CSV" "$DST" && ok "tuned FMoE CSV"; fi
fi

# 3. flydsl 0.3.4.1 side install
if [ -f "$AP_FLYDSL_DIR/flydsl-$AP_FLYDSL_VER.dist-info/METADATA" ]; then ok "flydsl $AP_FLYDSL_VER in $AP_FLYDSL_DIR"
elif act "pip install flydsl==$AP_FLYDSL_VER --target $AP_FLYDSL_DIR"; then
  python3 -m pip install -q --no-deps --target "$AP_FLYDSL_DIR" "flydsl==$AP_FLYDSL_VER" || die "flydsl install"
  ok "flydsl $AP_FLYDSL_VER in $AP_FLYDSL_DIR"
fi

# 4. import resolution (no aiter import: that would JIT-build)
PP=$AP_FLYDSL_DIR:$AP_AITER_DIR:/sgl-workspace/mori
got=$(PYTHONPATH=$PP:$AP_SGL_DIR/python python3 -c "import importlib.util as u
print(u.find_spec('aiter').origin, u.find_spec('sglang').origin)
import flydsl, importlib.metadata as m; print(m.version('flydsl'))" 2>&1 | tr '\n' ' ')
case "$got" in *"$AP_AITER_DIR/aiter/__init__.py $AP_SGL_DIR/python/sglang/__init__.py $AP_FLYDSL_VER"*) ok "imports resolve: $got";;
  *) echo "WARN import resolution: $got";; esac

# 5. AgentX benchmark deps (built by the agentx skill: /workspace/claude-skills/agentx/SKILL.md)
. /workspace/claude-skills/agentx/agentx_env.sh 2>/dev/null
[ -d "${INFMAX_CONTAINER_WORKSPACE:-/nonexistent}/benchmarks" ] && [ -x "${AIPERF_VENV:-/nonexistent}/bin/aiperf" ] \
  && ok "AgentX deps ($INFMAX_CONTAINER_WORKSPACE, $AIPERF_VENV)" \
  || echo "TODO AgentX deps missing -> follow /workspace/claude-skills/agentx/SKILL.md (GSM8K/proxy work without them)"

echo "LAUNCH: cd /shared_nfs/kk/dsv41/agentx && PYTHONPATH=$PP SRC=$AP_SGL_DIR/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1" \
  "EXTRA_ARGS=\"--fp8-gemm-backend aiter --enforce-shared-experts-fusion\" SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5" \
  "CONC=1 PREFILL_DECODE_INTERVAL=16 TAG=<tag> setsid nohup bash $D/scripts/agentx_colleague_run.sh > <tag>.nohup 2>&1 < /dev/null &"
