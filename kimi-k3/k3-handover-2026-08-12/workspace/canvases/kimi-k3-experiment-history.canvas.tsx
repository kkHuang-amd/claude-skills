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
          Stable experiment ledger · MI355X/gfx950 · updated 2026-08-12
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
        ]}
        rowTone={["success", "success", "success", "success", "info", "warning", "warning"]}
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
        <Text>1. Track AITER #4617/#4647 merges and remove the core-only patch stack.</Text>
        <Text>2. Run five-round paired C2 before enabling the B2 profile by default.</Text>
        <Text>3. Analyze B300 normal versus single-stream/no-PDL summaries.</Text>
        <Text>4. Target fixed tiny-kernel chains, copies, and cross-boundary fusions.</Text>
        <Text>5. For route work, prefer a one-CTA E896 sorter or direct stage1 metadata ABI.</Text>
      </Stack>

      <Text size="small" tone="tertiary">
        Canonical source:{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/SUMMARY.md">
          SUMMARY.md
        </Link>
        {" "}· Experiment evidence is retained under stage2-runs without raw profiler traces.
      </Text>
    </Stack>
  );
}
