# Correctness Checklist

- Baseline is reproducible and correct.
- Numerical tolerance is defined.
- Boundary and tail shapes are tested.
- Synchronization participant counts and scopes are valid.
- Memory ordering matches producer/consumer communication.
- Async copies/TDM are complete before consumption.
- Pointer bounds and alignment are validated.
- Cache-management requirements after special store paths are respected.
- Optimized and fallback paths are both tested when supported.
- No performance result is accepted before correctness passes.
