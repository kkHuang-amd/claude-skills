#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPCC="${HIPCC:-$(command -v hipcc || true)}"
ARCH="${GPU_ARCH:-gfx1250}"
ARCH="${ARCH%%:*}"
ARCH="${ARCH%%-*}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"

if [[ -z "${HIPCC}" ]]; then
  echo "error: hipcc not found; set HIPCC=/path/to/hipcc" >&2
  exit 1
fi

build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT

run_probe() {
  local name="$1"
  local source="${ROOT}/${name}.hip"
  local binary="${build_dir}/${name}"

  echo "Compiling ${name} for ${ARCH}"
  "${HIPCC}" -O3 -std=c++20 --offload-arch="${ARCH}" "${source}" -o "${binary}"
  echo "Running ${name}"
  timeout --signal=KILL "${TIMEOUT_SECONDS}s" "${binary}"
}

run_probe probe_hip_split_cluster_barrier
run_probe probe_hip_cluster_load

echo "All gfx1250 cluster primitive probes passed."
