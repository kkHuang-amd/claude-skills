# GEMM Optimization Checklist

## Shape and baseline
- [ ] M/N/K and distribution recorded
- [ ] dtype/accumulation/output precision recorded
- [ ] layouts, strides, transpose and epilogue recorded
- [ ] baseline numerical tolerance and performance repeated

## Access and compute
- [ ] lane accesses checked for coalescing
- [ ] tile sweep includes VGPR/LDS/occupancy
- [ ] wave32/WMMA mapping reviewed
- [ ] wave-tail boundaries included

## Pipeline
- [ ] staging alternatives ablated
- [ ] async/TDM completion precedes consumption
- [ ] double-buffer regions do not overlap incorrectly
- [ ] split-barrier independent window is truly independent
- [ ] multicast measured against non-multicast baseline

## Output and validation
- [ ] output stores profiled for small-K cases
- [ ] representative shape matrix tested
- [ ] regression cases reported
- [ ] end-to-end model effect checked, not kernel-only result
