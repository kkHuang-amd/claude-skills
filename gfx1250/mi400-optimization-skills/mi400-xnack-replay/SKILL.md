---
name: mi400-xnack-replay
description: Verify MI400/MI450 XNACK replay-mode correctness, hazards, compatibility and measured performance.
---
# MI400 XNACK Replay
## Trigger
XNACK/replay/page-fault behavior, replay-mode hazards, or binary compatibility is relevant.
## Workflow
1. Record XNACK enablement and replay mode.
2. Prefer conservative single-group behavior unless multi-group scheduling freedom is justified by evidence.
3. For multi-group mode audit memory-operation hazards and explicit waits/order.
4. Validate required binary/kernel paths under supported XNACK configurations relevant to release.
5. Use replay/fault-relevant tests for XNACK performance claims.
