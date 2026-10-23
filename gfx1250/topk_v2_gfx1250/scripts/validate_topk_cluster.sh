#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
PYTHON="${PYTHON:-python3}"
TEST_FILE="test/registered/kernels/ops/attention/test_topk_v2.py"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

if [[ ! -d "${SGLANG_ROOT}" ]]; then
  echo "error: SGLANG_ROOT does not exist: ${SGLANG_ROOT}" >&2
  exit 1
fi

cd "${SGLANG_ROOT}"

echo "Duplicate-heavy correctness"
SGLANG_GFX1250_TOPK_CLUSTER=1 timeout --signal=KILL "${TIMEOUT_SECONDS}s" \
  "${PYTHON}" -m pytest -q \
  "${TEST_FILE}::test_topk_v2_split_duplicate_scores" -s

echo "Full TopK v2 suite with cooperative route enabled"
SGLANG_GFX1250_TOPK_CLUSTER=1 timeout --signal=KILL "${TIMEOUT_SECONDS}s" \
  "${PYTHON}" -m pytest -q "${TEST_FILE}" -s

echo "Full TopK v2 suite on the default route"
env -u SGLANG_GFX1250_TOPK_CLUSTER \
  timeout --signal=KILL "${TIMEOUT_SECONDS}s" \
  "${PYTHON}" -m pytest -q "${TEST_FILE}" -s

echo "Fresh-process performance and output-set checks"
shapes=(
  "1 131072"
  "4 131072"
  "1 262144"
  "31 131072"
  "64 131072"
  "64 262144"
)
for shape in "${shapes[@]}"; do
  # Intentional word splitting passes batch and sequence as separate arguments.
  # shellcheck disable=SC2086
  "${PYTHON}" "${ARTIFACT_ROOT}/benchmark_cluster_load_topk.py" ${shape}
done
