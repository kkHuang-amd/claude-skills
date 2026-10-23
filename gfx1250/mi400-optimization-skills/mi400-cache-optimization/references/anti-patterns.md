# Cache Anti-Patterns

- Equating NT with complete cache bypass.
- Applying HT globally without identifying valuable reuse or enabling relevant residency behavior.
- Using a wider coherency scope than communication requires.
- Ignoring cross-AID ownership ping-pong on shared writes.
- Prefetching outside the allocation or across uncertain pointer bounds.
- Increasing prefetch distance/region until all cache lines are requested.
- Evaluating prefetch on one silicon revision and generalizing without recording the revision.
- Inferring improved locality from kernel time alone without checking whether another stage regressed.
