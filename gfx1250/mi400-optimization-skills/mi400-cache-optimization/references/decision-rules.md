# Cache Decision Rules

## Start from reuse and scope
For each range record writer, readers, reuse distance, reuse domain, lifetime, size and whether sharing crosses AIDs/workgroups/kernels.

## Temporal-hint selection
- RT: baseline/default caching.
- NT read: candidate for data consumed once or with very low reuse; remember this is not universally equivalent to bypass.
- NT store: candidate for streaming output with no expected reuse; assess interaction with dirty/owned lines and later consumers.
- HT: candidate for high-value reused data only when residency control/configuration can make it meaningful; otherwise it may behave like ordinary temporal traffic.
- LU: candidate when a read represents last use; treat behavior as aging/lifetime guidance and validate dirty-data semantics.

## Scope/coherency
Choose the smallest scope that still covers all communicating agents. Avoid patterns that repeatedly move ownership of the same line across AIDs. Account for 256-byte line/transaction effects where relevant, especially partial coherent accesses.

## Prefetch selection
- Short kernel with cold input and predictable valid region: consider launch/runtime L2 prefetch.
- In-kernel future tile and compute slack: consider shader/vector/L2 prefetch.
- Invalid address possible: do not use a path that can fault; speculative requests can be dropped when translation is unavailable, so measure usefulness.
- Prefetch bandwidth competes with useful traffic: disable when memory-bound or cache pollution rises.

## Experiment
A/B test one hint/prefetch choice, retain identical workload, and measure hit/traffic/stall and end-to-end time. Record silicon revision for unexplained behavior.
