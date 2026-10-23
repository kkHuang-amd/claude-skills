# Profiling Checklist

## Reproduction
- [ ] Exact source/container/image recorded
- [ ] Exact command and environment recorded
- [ ] Warmup and measurement windows separated
- [ ] Correctness baseline passes
- [ ] Representative workload matrix defined

## Collection
- [ ] Systems trace used if host/communication overlap matters
- [ ] Kernel-level metrics collected for top critical kernels
- [ ] Kernel instances matched by name and launch parameters where relevant
- [ ] Sampling/thread trace used only after coarse classification
- [ ] Capture overhead/perturbation checked

## Diagnosis
- [ ] Primary bottleneck and confidence stated
- [ ] Competing explanation listed
- [ ] Evidence points to a concrete region/kernel
- [ ] Expected metric movement for the proposed optimization stated
- [ ] Correct specialized skill selected
