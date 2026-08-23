# ATOM prefill trace campaign — 2026-08-22

Status: complete. Existing SGLang prefill cases and ATOM C2 were not rerun. No source, package, commit, or raw trace was changed.

## ATOM C2 recovery

- Recovered the existing case after the wave-active gate was relaxed. `stop_while_wave_active=false` is recorded and accepted.
- 8 gzip-valid traces, exact ranks 0-7, 16017523 bytes total; per trace 2000647-2003817 bytes, all below 500 MiB.
- HTTP start/stop 200/200; profile elapsed 2.000898s.
- FULL graph and `enforce_eager=False` evidence passed. Raw checksums were unchanged across analysis.
- Per rank: kernel 10043, CPU op 47347, runtime 25006-25008, user/GPU annotations 196/196, ac2g correlation 35436-35438.
- Prefill shape: `prefill[bs=1 tok=8192 ctx=8192 sqsq=67108864 sqsk=67108864 sk=8192]`, count 4/rank, aggregate annotation duration 2.453-2.493s.
- Analysis: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c2/analysis`; inventory: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c2/trace-inventory.json`; marker: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c2/RECOVERED_CASE_COMPLETE`.

## Fresh ATOM C64

- Updated canonical runner and the C64 exact manifest completed all normal gates; `stop_while_wave_active=true`.
- 8 gzip-valid traces, exact ranks 0-7, 22847176 bytes total; per trace 2852334-2862701 bytes, all below 500 MiB.
- HTTP start/stop 200/200; profile elapsed 2.000277s; FULL/non-eager mode passed.
- Per rank: kernel 10839, CPU op 70500, runtime 37026-37028, user/GPU annotations 288/288, ac2g correlation 48125-48127.
- Prefill shape: `prefill[bs=2 tok=16384 ctx=[8192, 8192] sqsq=134217728 sqsk=134217728 sk=16384]`, count 6/rank, aggregate annotation duration 3.843-3.966s.
- Analysis: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c64/analysis`; inventory: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c64/trace-inventory.json`; wrapper: `/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/prefill-steps/atom-c64/wrapper.log`.

## Cleanup

The runner released VRAM to baseline. Final process and VRAM checks found no benchmark/server processes and only the normal ~284 MiB per-GPU baseline allocation. No rebuildable K3 experiment caches remained.
