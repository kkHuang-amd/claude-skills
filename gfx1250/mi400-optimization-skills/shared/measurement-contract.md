# Measurement Contract

Before optimization record:
- GPU target and silicon revision
- ROCm/compiler/tool versions
- source revision
- build and launch commands
- model/kernel, shapes, dtypes and layouts
- concurrency/batch configuration
- correctness command and baseline result
- repeated performance baseline

Rules:
1. Profile before choosing a feature.
2. Change one material factor at a time where practical.
3. Use a representative workload matrix.
4. Repeat measurements and retain raw results.
5. Never claim improvement from a single favorable point.
6. Report regressions as well as wins.
