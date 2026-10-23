# TDM Optimization Checklist

## Descriptor and bounds
- [ ] element size, dimensions, strides, tile and offsets explicit
- [ ] base-pointer adjustments preserve strides
- [ ] normal and iterative bounds tested separately
- [ ] edge tiles/padding behavior tested
- [ ] wave-uniform issue verified

## Completion and ordering
- [ ] consumer waves identified
- [ ] correct wave-local or LDS-barrier completion selected
- [ ] workgroup/cluster barrier added where required
- [ ] in-flight operation count bounded
- [ ] subsequent normal loads protected from stale cache

## Performance
- [ ] alignment/granularity logged
- [ ] one-port/two-port LDS placement compared
- [ ] address-generation contention checked
- [ ] multicast effective behavior measured
- [ ] vector/async-copy/TDM baselines compared
- [ ] full kernel and end-to-end workload measured
