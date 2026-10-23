# GEMM Benchmarking
Do ablation: baseline; coalescing; tiling; staging; pipelining; multicast; output store; cache/prefetch. Keep dtype and M/N/K with every result. Never assign a fixed gain to TDM, WMMA or multicast independent of workload.
