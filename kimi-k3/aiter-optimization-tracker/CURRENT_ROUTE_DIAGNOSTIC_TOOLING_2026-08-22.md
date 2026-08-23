# Current-route A8W4/A16W4 diagnostic tooling — 2026-08-22

## Status

Reusable tooling is implemented under
`/workspace/useful-scripts/benchmarking/kimi-k3/contracts/`. No server was
started and no route evidence has been collected yet.

Files:

- `sitecustomize.py`: opt-in, idempotent early wrapper around
  `aiter.fused_moe.fused_moe`; atomic bounded route-count JSON only. Dumping
  and call-index advancement require a valid `K3_ROUTE_DUMP_ARM_FILE`.
- `run_route_dump.sh`: exact common-manifest 8192/128 C64 eager diagnostic for
  SGLang A8W4 or ATOM A16W4, TP8, zero warmups, single-stream ATOM.
- `analyze_route_dumps.py`: exact rank/call alignment plus per-layer and
  cross-engine JSON/CSV/Markdown analysis.
- `README.md`, `test_sitecustomize.py`, and
  `test_analyze_route_dumps.py`.

The model config at `/shared_nfs/models/Kimi-K3/config.json` reports 93 hidden
layers. Layer 0 is dense, so the runner requires exactly 92 qualifying
`topk_ids=[64,16]` calls (`0..91`) on each of eight ranks.

Validation completed without launching servers:

```text
python compilation: pass
unit tests:        6 passed
bash -n:           pass
CLI help smoke:    pass
IDE lints:         no errors
git diff --check:  pass
legacy dump gate:  rejected with status 2
```

## Next

The first route run used the pre-arm wrapper and is invalid. Run the two
engines sequentially with the hardened common C64 contract, then analyze:

```bash
MANIFEST=/workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/prompt-manifest-c64-8192.jsonl.gz
ROOT=/workspace/kimi-k3-runs/current-route-contract-2026-08-22

bash /workspace/useful-scripts/benchmarking/kimi-k3/contracts/run_route_dump.sh \
  sglang "$ROOT/sglang" "$MANIFEST"
bash /workspace/useful-scripts/benchmarking/kimi-k3/contracts/run_route_dump.sh \
  atom "$ROOT/atom" "$MANIFEST"
python /workspace/useful-scripts/benchmarking/kimi-k3/contracts/analyze_route_dumps.py \
  "$ROOT/sglang/route-dumps" "$ROOT/atom/route-dumps" \
  --left-name sglang-a8w4 --right-name atom-a16w4 \
  --output-dir "$ROOT/analysis"
```

These are eager route-contract runs. Do not use or report their endpoint
timings as engine performance.
