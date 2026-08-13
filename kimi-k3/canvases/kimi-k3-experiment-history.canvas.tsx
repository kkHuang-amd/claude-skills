import {
  Callout,
  Grid,
  H1,
  H2,
  Link,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

export default function KimiK3ExperimentHistory() {
  return (
    <Stack gap={18} style={{ padding: 20, maxWidth: 1450, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Kimi-K3 optimization history</H1>
        <Text tone="secondary">
          Stable experiment ledger · MI355X/gfx950 · updated 2026-08-13
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="0.990" label="Current GSM8K 200" tone="success" />
        <Stat value="933,883" label="Max token capacity" tone="success" />
        <Stat value="46" label="Vendored FlyDSL tests" tone="success" />
        <Stat value="+8.8%" label="Optional B2 C2 gain" tone="info" />
      </Grid>

      <Callout tone="success" title="Current validated state">
        Kimi-specific FlyDSL kernels are maintained in SGLang. The core-only
        AITER branch retains only caller-owned fused_moe output and stage1
        scratch reuse. C2-C32 endpoint throughput is within 0.25% of the golden
        stack.
      </Callout>

      <H2>Milestones</H2>
      <Table
        striped
        headers={["Date", "Milestone", "Result", "Decision"]}
        rows={[
          ["2026-08-07", "Production baseline and runtime attribution", "C2 921.88 · C32 6087.04 tok/s", "Established matched baseline"],
          ["2026-08-10", "KDA, MoE zero-copy, M16384 and scratch reuse", "C2-C32 +1.92–5.05%; graph 9.16→3.47 GB/GPU", "Selected foundation"],
          ["2026-08-10", "Batch-1 fusion family", "#4497/#4499 selected; #4503/#4504 opt-in", "Keep independent flags"],
          ["2026-08-11", "Production trace refresh", "Route prep, fixed ops and attention residual ranked", "Reject forced B32 all-reduce"],
          ["2026-08-11", "Route V3/V3-R/V4 prototypes", "Full-chain gates failed; B16/B32 regressions", "Remove prototypes"],
          ["2026-08-11", "B2 fusion solidification", "C2 +8.70%; C4 flat after M>=4 fallback", "Retain optional B2"],
          ["2026-08-12", "Fresh-clone migration", "GSM8K 0.990; C2-C32 within 0.2%; capacity matched", "Migration accepted"],
          ["2026-08-12", "SGLang-owned FlyDSL kernels", "46 tests; endpoints within 0.25%; B2 reproduced", "AITER reduced to core-only"],
          ["2026-08-12", "SGLang #34490 Radix-4 router", "45 tests; C2 +2.34%; C4/C8/C16/C32 +2.05/+1.71/+1.34/+0.68%", "Retain default-off with local tie/NaN fixes"],
          ["2026-08-13", "Reclone Triton attribution", "Triton 3.6 restored C8/C16/C32 by +1.69/+2.98/+5.01% vs 3.7; C4 repeat passed", "Retain 3.6; full matrix reproduced"],
          ["2026-08-13", "Triton 3.6/3.7 compact trace", "extend_attention _fwd_kernel +130%; 512 VGPR and 472-byte scratch under 3.7", "Prefill root cause localized; retain Triton 3.6"],
          ["2026-08-13", "Triton 3.7 extend-attention N32", "12.57→5.24 ms; C32 5881.93→6198.56 tok/s", "Retain opt-in pending GSM8K"],
        ]}
        rowTone={[
          "neutral",
          "success",
          "success",
          "info",
          "warning",
          "success",
          "success",
          "success",
          "info",
          "warning",
          "warning",
          "success",
        ]}
      />

      <H2>Current production endpoint</H2>
      <Table
        striped
        headers={["Concurrency", "Golden tok/s", "Vendored tok/s", "Delta"]}
        rows={[
          ["C2", "969.04", "968.57", "-0.05%"],
          ["C4", "1,746.24", "1,741.98", "-0.24%"],
          ["C8", "2,885.97", "2,881.25", "-0.16%"],
          ["C16", "4,437.72", "4,432.25", "-0.12%"],
          ["C32", "6,202.46", "6,191.41", "-0.18%"],
        ]}
        rowTone={["success", "success", "success", "success", "success"]}
      />

      <H2>2026-08-13 Triton 3.6 A/B</H2>
      <Table
        striped
        headers={["Concurrency", "Handover tok/s", "Triton 3.7 tok/s", "Triton 3.6 tok/s", "3.6 delta", "0.5% gate"]}
        rows={[
          ["C2", "968.57", "970.53", "966.08", "−0.26%", "Pass"],
          ["C4", "1,741.98", "1,718.97", "1,743.73", "+0.10%", "Pass on repeat"],
          ["C8", "2,881.25", "2,839.40", "2,887.31", "+0.21%", "Pass"],
          ["C16", "4,432.25", "4,309.25", "4,437.74", "+0.12%", "Pass"],
          ["C32", "6,191.41", "5,881.93", "6,176.87", "−0.23%", "Pass"],
        ]}
        rowTone={["success", "success", "success", "success", "success"]}
      />
      <Callout tone="warning" title="Triton 3.7 caused the high-concurrency loss">
        Changing only Triton from 3.7 to the handover commit 3.6.0+git42270451
        restored C2-C32 to the 0.5% gate after a focused C4 repeat. All requests
        succeeded and capacity stayed 933,883. Paired traces could not localize
        the gap to one kernel because ROCTracer did not expand HIP CUDA Graph
        replays.{" "}
        <Link href="file:///dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_AB_2026-08-13.md">
          Detailed report
        </Link>
        {" "}·{" "}
        <Link href="file:///dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_37_C32_TRACE_2026-08-13.md">
          Trace report
        </Link>
      </Callout>

      <H2>Triton 3.6 → 3.7 stage attribution</H2>
      <Table
        striped
        headers={["Scope", "Metric", "3.6", "3.7", "Interpretation"]}
        rows={[
          ["CPU", "aiter::fused_moe_ calls", "1,564", "1,564", "Identical model work executed"],
          ["GPU", "MoE stage1 events", "11,868", "92", "3.7 visibility −99.22%"],
          ["GPU", "_agg_kernel events", "27,156", "186", "3.7 visibility −99.32%"],
          ["GPU", "Fused KDA events", "8,901", "0", "Hidden under 3.7"],
          ["GPU", "MLA merge events", "3,096", "0", "Hidden under 3.7"],
          ["Workload", "Profiled median TPOT", "108.98 ms", "118.29 ms", "3.7 is +8.54% slower"],
        ]}
        rowTone={["success", "warning", "warning", "warning", "warning", "warning"]}
      />
      <H2>Compact Rank0 prefill root cause</H2>
      <Table
        striped
        headers={["Metric", "Triton 3.6", "Triton 3.7", "Delta"]}
        rows={[
          ["Prefill span", "4,153.94 ms", "4,375.87 ms", "+221.93 ms"],
          ["extend_attention _fwd_kernel p50", "5,990.80 µs", "13,780.42 µs", "+130.03%"],
          ["extend_attention _fwd_kernel total", "143.73 ms", "330.90 ms", "+187.17 ms"],
          ["VGPR count", "483", "512", "+29"],
          ["Private segment", "0 bytes", "472 bytes", "Scratch spill"],
          ["Scratch load/store instructions", "0", "186", "New in 3.7"],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning", "warning"]}
      />
      <Callout tone="warning" title="Triton 3.7 spills the extend-attention kernel">
        The compact trace restores comparable graph visibility and identifies
        python/sglang/kernels/ops/attention/extend_attention.py::_fwd_kernel as
        the dominant prefill regression. Its +187.17 ms explains 84.34% of the
        +221.93 ms Rank0 prefill span increase. Decode retains several smaller
        2.7–4.5% regressions but no comparable single-kernel culprit.{" "}
        <Link href="file:///dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_37_STAGE_ANALYSIS_2026-08-13.md">
          Root-cause analysis
        </Link>
      </Callout>
      <Callout tone="info" title="Full-workload trace coverage caveat">
        CPU API counts prove MoE still executes, but ROCTracer loses nearly all
        CUDA Graph replay kernels in the oversized Triton 3.7 full trace.
        Therefore full-trace total GPU time remains invalid; only the matched
        compact windows support the kernel-level comparison above.
      </Callout>

      <H2>Triton 3.7 N32 endpoint recovery</H2>
      <Table
        striped
        headers={["Concurrency", "3.7 baseline tok/s", "N32 candidate tok/s", "Handover target tok/s"]}
        rows={[
          ["C2", "970.53", "973.94", "968.57"],
          ["C4", "1,718.97", "1,748.71", "1,741.98"],
          ["C8", "2,839.40", "2,901.37", "2,881.25"],
          ["C16", "4,309.25", "4,460.33", "4,432.25"],
          ["C32", "5,881.93", "6,198.56", "6,191.41"],
        ]}
        rowTone={["success", "success", "success", "success", "success"]}
      />
      <Callout tone="success" title="Version-gated spill fix passes the endpoint matrix">
        BLOCK_N 64→32 removes scratch while preserving BLOCK_M=64 and four
        warps. C32 improves 5.38% over Triton 3.7 and is 0.12% above handover.
        Capacity remains 933,883. Keep default-off until GSM8K validation.{" "}
        <Link href="file:///dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON37_EXTEND_N32_RESULTS_2026-08-13.md">
          N32 results
        </Link>
      </Callout>

      <H2>MI355X · Kimi-K3 Radix-4 endpoint matrix</H2>
      <Table
        stickyHeader
        striped
        headers={[
          "Concurrency",
          "TP",
          "TTT (tok/s)",
          "TTT per GPU (tok/s/GPU)",
          "Output throughput (tok/s)",
          "Median E2EL (ms)",
          "Median TTFT (ms)",
          "Median TPOT (ms)",
          "Median ITL (ms)",
        ]}
        rows={[
          ["2", "8", "993.11", "124.14", "110.35", "18,547.02", "943.71", "17.21", "17.21"],
          ["4", "8", "1,777.66", "222.21", "197.52", "20,752.40", "1,594.63", "18.71", "18.32"],
          ["8", "8", "2,930.39", "366.30", "325.60", "25,159.37", "2,634.72", "22.01", "20.62"],
          ["16", "8", "4,491.62", "561.45", "499.07", "32,856.92", "4,720.36", "27.65", "24.11"],
          ["32", "8", "6,233.41", "779.18", "692.60", "47,270.03", "8,762.63", "37.90", "30.09"],
        ]}
        rowTone={["success", "success", "success", "success", "success"]}
      />
      <Callout tone="info" title="Measurement definition">
        TTT is total token throughput; per-GPU TTT is TTT ÷ 8. TP8,
        8192-input/1024-output random workload, 64 warmups, no radix cache and
        DCP off. C2 values are five-round medians; C4-C32 are the accepted
        candidate runs. Source: PR #34490 artifacts · 2026-08-12.
      </Callout>

      <H2>Radix-4 + B2 composition · exploratory</H2>
      <Table
        striped
        headers={[
          "Concurrency",
          "TP",
          "TTT (tok/s)",
          "TTT per GPU (tok/s/GPU)",
          "Output throughput (tok/s)",
          "Median E2EL (ms)",
          "Median TTFT (ms)",
          "Median TPOT (ms)",
          "Median ITL (ms)",
        ]}
        rows={[
          ["2", "8", "1,082.78", "135.35", "120.31", "17,010.74", "939.70", "15.71", "15.71"],
          ["4", "8", "1,781.59", "222.70", "197.95", "20,704.56", "1,588.34", "18.68", "18.29"],
        ]}
        rowTone={["success", "success"]}
      />
      <Callout tone="warning" title="Single-run result">
        C2 is +9.03% over Radix-4 alone, +2.71% over the prior B2-only result
        and +11.58% over the paired baseline median. C4 is +0.22% over
        Radix-4 alone, consistent with B2 failing closed for M≥4. Repeat paired
        validation before changing the profile policy.
      </Callout>

      <H2>Feature policy</H2>
      <Table
        striped
        headers={["Feature", "Status", "Evidence"]}
        rows={[
          ["Fused KDA + f_b", "Production", "Kernel boundary active; correctness retained"],
          ["MoE caller-owned output", "Production", "Routed-output copy removed"],
          ["Stage1 scratch reuse", "Production manifest", "Capacity 933,883; graph memory reduced"],
          ["MLA gate + KDA group64", "Production manifest", "Focused tests and endpoint gates passed"],
          ["B2 preroute/shared-down", "Optional", "C2 968.57→1,054.19 tok/s; C4 flat"],
          ["FP8 latent tail", "Optional off", "C1 +2.39%; capacity −9.55%"],
          ["A4W4 profile", "C16 only", "C16 +1.50%; other points regress"],
          ["Radix-4 K3 TopK", "Validated optional", "C2 paired +2.34%; GSM8K 0.985; capacity 933,883"],
          ["Radix-4 + B2", "Exploratory optional", "Single-run C2 1,082.78 tok/s; repeat paired validation"],
          ["Triton 3.7 extend N32", "Validated performance-only", "C32 +5.38%; full endpoint matrix recovered; GSM8K pending"],
        ]}
        rowTone={["success", "success", "success", "success", "info", "warning", "warning", "success", "info", "info"]}
      />

      <H2>Rejected architecture experiments</H2>
      <Table
        headers={["Experiment", "Observed result", "Do not retry unless"]}
        rows={[
          ["Route sort+quant only", "At most 1.04 µs/layer", "A larger boundary is removed"],
          ["One-wave TopK+quant", "10.8–12.4 µs slower", "Quant is parallel rather than serialized"],
          ["V3 256-thread TopK", "Slower and tie-order incompatible", "Wave64 order is preserved"],
          ["V3-R role grid + P23", "B16/B32 regressions", "Sorted ABI is eliminated"],
          ["V4 multi-CU persistent prep", "226–339 µs vs Opus 14.6 µs", "Cross-CU barriers are removed"],
          ["Generic TILE_M", "M4 only +2.77 µs/layer; M8/M16 regress", "MFMA small-M architecture replaces GEMV"],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning", "warning"]}
      />

      <H2>Next work</H2>
      <Stack gap={7}>
        <Text>1. Reconcile the local #34490 exact-tie, NaN, flag and gfx guard fixes with upstream before default enablement.</Text>
        <Text>2. Analyze B300 normal versus single-stream/no-PDL summaries.</Text>
        <Text>3. Track AITER #4617/#4647 merges and remove the core-only patch stack.</Text>
        <Text>4. Run five-round paired C2 before enabling the B2 profile by default.</Text>
        <Text>5. Target fixed tiny-kernel chains, copies, and cross-boundary fusions.</Text>
      </Stack>

      <Text size="small" tone="tertiary">
        Canonical source:{" "}
        <Link href="file:///dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/SUMMARY.md">
          SUMMARY.md
        </Link>
        {" "}· Experiment evidence is retained under stage2-runs without raw profiler traces.
      </Text>
    </Stack>
  );
}
