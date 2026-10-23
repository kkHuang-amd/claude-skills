# GEMM Anti-Patterns

- Replacing MFMA with WMMA and keeping old tile/schedule unchanged.
- Maximizing tile size without tracking VGPR, LDS, occupancy and wave tails.
- Adding TDM when movement is not on the critical path.
- Treating multicast as guaranteed; drift/timeouts can fall back to separate requests.
- Adding barriers until behavior stabilizes without measuring synchronization overhead.
- Prefetching a memory-bound GEMM merely because prefetch exists.
- Reporting one M/N/K point as the optimization result.
- Ignoring output stores because GEMM is assumed to be input/compute dominated.
- Comparing kernels with different accumulation precision, epilogues or correctness tolerance.
