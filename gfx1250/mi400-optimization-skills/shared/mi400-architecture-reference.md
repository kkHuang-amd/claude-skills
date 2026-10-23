# MI400 Architecture Reference

Use this as shared context, not as a substitute for measurement.

- MI400/GFX12 kernel optimization should account for wave32 execution and WMMA-oriented matrix execution.
- MI450 adds direct tensor data movement between global memory and LDS through TDM.
- Clustered workgroups enable cluster synchronization and TDM multicast use cases.
- Cache optimization involves scope/coherency plus temporal hints and prefetch behavior.
- MI450 communication optimization may choose among vector load/store, TDM, and DMA/copy engines based on workload characteristics.

Always record exact target and silicon revision because behavior and enablement may differ.
