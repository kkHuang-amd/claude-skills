---
name: mi400-network-collectives
description: Optimize MI450 collective and point-to-point communication by topology, message size, protocol and movement engine.
---
# MI400 Network and Collectives
## Trigger
All-reduce/all-gather/reduce-scatter, scale-up/out traffic, P/D communication, or vector/TDM/SDMA selection.
## Inputs
topology, message-size distribution, collective, GPU count, concurrency, overlap requirement, available WGPs, latency/bandwidth target.
## Workflow
1. Establish actual topology/domain.
2. Measure target message sizes.
3. Compare vector, TDM and DMA/copy-engine paths instead of hard-coding one.
4. Include WGP consumption when TDM competes with compute.
5. Compare protocol/algorithm candidates against latency, bandwidth and synchronization cost.
6. Benchmark representative model traffic as well as microbenchmarks.
