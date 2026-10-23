# Cache Optimization Checklist

- [ ] Data ranges classified by reuse/lifetime
- [ ] Readers/writers and coherency scope documented
- [ ] Baseline cache/traffic/stall evidence collected
- [ ] RT baseline retained
- [ ] NT/HT/LU hypothesis tied to a specific range
- [ ] Residency-control assumption verified for HT
- [ ] Partial-line and 256-byte effects considered
- [ ] Prefetch region is valid and aligned
- [ ] Prefetch distance/size sweep is bounded
- [ ] Traffic/pollution and end-to-end time both checked
- [ ] Silicon revision recorded
- [ ] Correctness and regression matrix passes
