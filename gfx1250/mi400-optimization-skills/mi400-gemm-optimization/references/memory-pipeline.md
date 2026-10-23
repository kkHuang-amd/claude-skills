# GEMM Memory Pipeline
Candidate stages: coalesced global access -> LDS staging -> WMMA -> output store. Compare direct/global-to-LDS paths, asynchronous movement, selective waits and double buffering. Multicast is only a candidate when clustered workgroups share input data.
