# TDM Anti-Patterns

- Introducing TDM without proving data movement/address generation is critical.
- Divergent lane control or descriptor values for a wave-wide operation.
- Treating dimensions and strides as interchangeable.
- Assuming normal-mode bound clamping proves iterative-mode safety.
- Issuing too many operations from low-level code without explicit waits.
- Having two TDM engines target the same LDS partition and expecting doubled throughput.
- Assuming multicast always occurs because a multicast mask was set.
- Waiting only in the issuing wave while other waves consume the LDS tile.
- Reading a TDM-updated global location through a stale normal cached path.
- Comparing TDM with an untuned/vector baseline only.
