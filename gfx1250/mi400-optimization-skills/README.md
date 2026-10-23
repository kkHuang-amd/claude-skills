# MI400 Optimization Skills

Topic-oriented skills distilled from the MI400/MI450 training series for performance agents.

## Routing
1. Start with `mi400-profiling` when the bottleneck is not established.
2. Route to GEMM, cache, TDM, synchronization, HIP runtime, networking, or XNACK based on evidence.
3. Always use `shared/measurement-contract.md` and `shared/correctness-checklist.md`.
4. Finish with `shared/result-report-template.md`.

## Skills
- mi400-profiling
- mi400-gemm-optimization
- mi400-cache-optimization
- mi400-tdm-data-movement
- mi400-sync-memory-ordering
- mi400-hip-runtime
- mi400-network-collectives
- mi400-xnack-replay

## Deep references
The Profiling, GEMM, Cache and TDM skills include:
- `decision-rules.md`
- `anti-patterns.md`
- `optimization-checklist.md`

Agents should read the relevant `SKILL.md` first, then load only the references needed for the active task. All runs must also follow `shared/skill-quality-gates.md`.
