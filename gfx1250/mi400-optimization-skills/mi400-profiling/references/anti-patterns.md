# Profiling Anti-Patterns

- Optimizing whichever kernel has the most launches rather than the most elapsed critical-path impact.
- Treating one counter as a diagnosis without checking workload phase and kernel identity.
- Comparing captures made with different shapes, warmup, concurrency, clocks, software revision, or profiling overhead.
- Enabling every trace/counter at once and perturbing the workload beyond usefulness.
- Calling a kernel memory-bound solely because it loads memory; prove memory limits progress.
- Calling a workload compute-bound solely from theoretical FLOPs.
- Sending a specialized optimization skill a vague statement such as "GPU utilization is low" without trace evidence.
- Claiming an agent ran a profiler when no output artifact or command result exists.
