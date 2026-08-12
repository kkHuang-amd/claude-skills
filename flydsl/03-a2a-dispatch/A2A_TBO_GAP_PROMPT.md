# Prompt for a new agent

Read this handover first:

`/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/A2A_TBO_GAP_HANDOVER.md`

Then investigate the remaining DeepSeek-V4-Pro performance gap between DP-TBO and
FlyDSL-EP-TBO on 8x MI355X.

Current validated numbers, same 8k/1k c256 harness with guarded delayer:

- DP-TBO: 33,076.53 tok/s, TPOT 52.93ms, TTFT 16.91s
- FlyDSL-EP-TBO: 32,368.95 tok/s, TPOT 54.36ms, TTFT 17.30s

Use:

- SGLang repo: `/sgl-workspace/sglang-flydsl-a2a`
- branch: `feat/flydsl-a2a`
- pinned aiter: `/sgl-workspace/aiter` at `9127c94a1`
- full trace and benchmark paths listed in the handover

Do not re-debug correctness, decode deadlock, dynamic recv, PrefillDelayer, or the
initial missing comm stream. Those are fixed and data-backed.

Primary tasks:

1. Build an overlap-aware critical-path parser for the existing DP/FlyDSL traces.
   Report wall-time union per stream/layer, dispatch vs combine separately, event
   waits, and exact hidden communication. Do not rely only on summed kernel time.
2. Determine which component explains the remaining ~2.1% gap:
   - A2A kernel inflation under overlap
   - split expert-GEMM weight reload / small-M efficiency
   - DSV4 cross-layer fused-mHC loss
   - prefill recv-buffer padding
   - event/stream idle gaps
3. Run controlled experiments in this order:
   - separate dispatch/combine block-count sweep
   - verify actual prefill aiter M for each TBO child
   - TBO split-ratio sweep
   - comm-stream priority probe
4. For every claimed mechanism, state the observable signature and show the
   measurement. Do not infer a bottleneck from a profiler stall site.
5. Use fresh servers, one changed variable, 2048 prompts / 512 warmups for final
   serving decisions, focused 8-GPU correctness, and full GSM8K for any code change.
6. Keep MegaMoE separate. This task concerns the vendored FlyDSL-EP backend and DP.

Write durable results and exact repro commands back into
`A2A_TBO_GAP_HANDOVER.md`. Do not commit or push unless explicitly requested.
