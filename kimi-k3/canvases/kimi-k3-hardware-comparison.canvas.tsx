import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

export default function B300VsMI355XKimiK3() {
  return (
    <Stack gap={18} style={{ padding: 20, maxWidth: 1500, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Kimi-K3 · B300 versus MI355X</H1>
        <Text tone="secondary">
          TP8 · no DCP · random 8192/1024 · no radix cache · single-stream
          endpoint target with supporting multi-stream trace evidence
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="60.6%" label="MI355X / B300 single-stream C2 throughput" tone="warning" />
        <Stat value="1.67×" label="MI355X C2 TPOT multiplier" tone="warning" />
        <Stat value="1.26 ms" label="B300 endpoint benefit from overlap / PDL" tone="info" />
        <Stat value="1 vs 1" label="Primary comparison GPU streams" tone="success" />
      </Grid>

      <Callout tone="warning" title="Primary finding">
        The single-stream control shows that concurrency is not the majority of
        the remaining endpoint gap. Disabling B300 streams/PDL changes C2 TPOT
        from 9.33 to 10.59 ms, while MI355X remains at 17.66 ms. The dominant
        residual is serial software structure: fixed launch chains,
        route/sort/quant, copies, attention residual, and KDA boundaries.
      </Callout>
      <Callout tone="success" title="Matched no-cache control">
        The primary endpoint comparison now disables radix cache on both
        systems and uses B300 single-stream/no-PDL. Earlier cached,
        multi-stream B300 traces remain useful only for diagnosing what overlap
        hides, not for the primary throughput ratio.
      </Callout>

      <H2>Endpoint comparison</H2>
      <Table
        striped
        headers={["Concurrency", "B300 TTT", "MI355X TTT", "MI / B300", "B300 output", "MI output"]}
        rows={[
          ["C2", "1598.21", "969.04", "60.6%", "177.58", "107.67"],
          ["C4", "2747.60", "1746.24", "63.6%", "305.29", "194.03"],
          ["C8", "4282.09", "2885.97", "67.4%", "475.79", "320.66"],
          ["C16", "6062.39", "4437.72", "73.2%", "673.60", "493.08"],
          ["C32", "8136.58", "6202.46", "76.2%", "904.06", "689.16"],
        ]}
        rowTone={["warning", "warning", "neutral", "neutral", "neutral"]}
      />

      <Table
        striped
        headers={["Concurrency", "B300 TPOT ms", "MI355X TPOT ms", "TPOT ratio", "B300 ITL ms", "MI355X ITL ms"]}
        rows={[
          ["C2", "10.59", "17.66", "1.67×", "10.59", "17.67"],
          ["C4", "11.85", "18.71", "1.58×", "11.75", "18.71"],
          ["C8", "14.61", "21.00", "1.44×", "14.10", "21.00"],
          ["C16", "20.15", "24.62", "1.22×", "18.33", "24.62"],
          ["C32", "29.21", "30.46", "1.04×", "24.52", "30.46"],
        ]}
        rowTone={["warning", "warning", "neutral", "neutral", "neutral"]}
      />

      <H2>Original multi-stream trace diagnostic</H2>
      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader trailing="C2">B300</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>Step span: 9.66 ms</Text>
              <Text>Summed kernels: 15.09 ms</Text>
              <Text>Busy union: 9.07 ms</Text>
              <Text>Overlap saved: 6.02 ms</Text>
              <Text>GPU streams: 5</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing="C2">MI355X</CardHeader>
          <CardBody>
            <Stack gap={7}>
              <Text>Step span: 18.21 ms</Text>
              <Text>Summed kernels: 18.85 ms</Text>
              <Text>Busy union: 18.85 ms</Text>
              <Text>Overlap saved: approximately 0 ms</Text>
              <Text>Effective overlapping streams: 1</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Table
        striped
        headers={["C2 component", "B300 ms/step", "MI355X ms/step", "Interpretation"]}
        rows={[
          ["Other GEMMs", "6.618", "6.874", "Similar serial cost; not the 2× gap"],
          ["Collectives", "2.573", "1.835", "MI355X is faster in summed kernel time"],
          ["MoE compute", "2.248", "1.115", "MI355X is faster at C2"],
          ["Route / sort / quant", "0.901", "2.615", "MI355X is 2.9× slower"],
          ["Attention residual", "1.180", "1.580", "MI355X is 34% slower"],
          ["KDA decode", "0.464", "0.745", "MI355X is 61% slower"],
          ["Copies", "0.024", "0.525", "MI355X has about 22× more copy time"],
          ["Unclassified fixed ops", "0.548", "3.035", "Additional launch-fragmentation gap"],
        ]}
        rowTone={["neutral", "success", "success", "warning", "warning", "warning", "warning", "warning"]}
      />
      <Callout tone="info" title="What B300 overlaps at C2">
        Visible cross-stream overlap includes collective + MoE compute
        (0.84 ms/step), GEMM + GEMM (1.39 ms), GEMM + route preparation
        (0.49 ms), and collective + routing (0.36 ms). The remainder includes
        multiway overlap and same-timeline PDL between CUTLASS kernels. MI355X
        shows approximately zero overlap for these pairs.
      </Callout>

      <Callout tone="info" title="C32 cross-check">
        B300 still overlaps 6.71 ms per decode step, but the total step gap grows
        to 15.94 ms, so overlap explains about 42% at C32. MI355X MoE stage
        compute is close to B300 at C32. This C32 component comparison remains
        directional because the current MI355X C32 diagnostic trace used a
        shorter context; a matched C32 stage trace has not yet been captured.
      </Callout>

      <H2>Software-path differences</H2>
      <Table
        striped
        headers={["Area", "B300", "MI355X", "Likely impact"]}
        rows={[
          ["Concurrency", "CUDA alt streams + PDL", "HIP alt streams explicitly disabled; no measured PDL overlap", "Largest C2 opportunity"],
          ["Radix cache", "Enabled; profiled request reuses 8128/8192 tokens", "Explicitly disabled", "Biases B300 TTFT/E2E/TTT favorably"],
          ["MoE", "flashinfer_mxfp4 / TRT-LLM kernels", "AITER Opus A8W4", "Comparable compute at C32; B300 route prep is better"],
          ["Route prep", "route_quant_fused", "grouped top-k + sort/quant kernels", "About 1.73 ms/step C2 gap"],
          ["Attention", "trtllm_mla", "Triton prefill + AITER MLA decode", "B300 has TMA/persistent kernels"],
          ["Attention residual", "TMA fused kernel", "ROCm register-tile Triton kernel", "About 0.43 ms/step C2 gap"],
          ["KDA", "CUDA fused many-heads kernel", "ATT-tuned AITER fused KDA + f_b", "69-layer graph improved 9.20→8.38 µs/layer; still opt-in"],
        ]}
        rowTone={["warning", "warning", "neutral", "warning", "info", "info", "info"]}
      />
      <Callout tone="info" title="2026-08-18 serial KDA and preroute update">
        SGLang K3 sets <Text weight="semibold">alt_streams=None</Text> on HIP,
        so the retained MI355X execution is single-stream; do not attribute M4
        shared-down behavior to stream overlap. ATT-backed KDA changes reduce a
        production-like 69-layer graph by 8.9%, while combined B2 projections
        raise C2 throughput 9.36%. A new single-launch M4 mixed preroute front
        raises C4 1.80%; M8/M16 and M4 shared-down fail their endpoint gates.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/KDA_B2_M4_OPTIMIZATION_2026-08-18.md">
          Detailed handoff
        </Link>
      </Callout>

      <H2>Recommended order</H2>
      <Stack gap={8}>
        <Text>
          1. Reduce serial fixed-op chains, copies, route preparation, attention
          residual, and KDA launch boundaries. The single-stream control shows
          these dominate the remaining C2 gap.
        </Text>
        <Text>
          2. Keep the validated B2 fusion profile available while measuring
          graph and weight-capacity policy over repeated C2 runs.
        </Text>
        <Text>
          3. Do not pursue sort+quant alone: the gfx950 prototype saved at most
          1.04 µs/layer. A one-wave TopK+quant producer was also 10.8–12.4 µs
          slower because quant serialized behind routing. A 256-thread
          multi-block TopK was then 0.4–1.15 µs slower and changed tie IDs.
          Future work must preserve the existing wave64 TopK.
        </Text>
        <Text>
          4. Treat HIP multi-stream as a secondary experiment after serial
          cleanup; B300 normal versus single-stream changes C2 TPOT by 1.26 ms.
        </Text>
      </Stack>

      <Text tone="tertiary" size="small">
        Sources: uploaded B300 C2/C32 stage-separated TP0 traces and client logs;
        matched MI355X C2 8192/1024 stage-separated TP0 trace and selected-final
        endpoint logs.
        Kernel groups are name-based classifications; overlap is duration sum
        minus merged busy-union. ROCm profiler exported child kernels for one
        graph replay inside the five-step decode window; its component totals
        are normalized to that captured replay.
      </Text>
    </Stack>
  );
}
