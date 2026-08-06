import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const buckets = [
  ["Dense GEMMs", 8.48, 21.1, 698],
  ["MoE stage 1", 6.95, 17.3, 92],
  ["Full attention", 6.35, 15.8, 24],
  ["MoE stage 2", 4.11, 10.2, 92],
  ["MoE route/quant", 2.97, 7.4, 368],
  ["PyTorch native", 2.86, 7.1, 622],
  ["Collective", 2.33, 5.8, 187],
  ["KDA decode", 1.84, 4.6, 69],
  ["Attn residual", 1.66, 4.1, 186],
  ["Activation/add", 0.8, 2.0, 185],
  ["Other + sampling", 1.75, 4.4, 303],
] as const;

export default function KimiK3Tp8RocmTrace() {
  const theme = useHostTheme();

  return (
    <Stack
      gap={20}
      style={{
        padding: 24,
        minHeight: "100%",
        background: theme.bg.editor,
        color: theme.text.primary,
      }}
    >
      <Stack gap={6}>
        <Row justify="space-between" align="center">
          <H1>Kimi-K3 TP8 ROCm decode trace</H1>
          <Pill tone="info">No DCP · No DSPARK</Pill>
        </Row>
        <Text tone="secondary">
          8× MI355X · fixed 8192/1024 · concurrency 32 · Triton attention
        </Text>
      </Stack>

      <Grid columns={4} gap={18}>
        <Stat value="842 tok/s" label="Production graph decode throughput" tone="success" />
        <Stat value="38.0 ms" label="Derived production decode step" />
        <Stat value="−92" label="Native launches after retained fix" tone="success" />
        <Stat value="−0.40 ms" label="Native GPU time per step" tone="success" />
      </Grid>

      <Card>
        <CardHeader trailing="Representative production step">
          GPU time by kernel role
        </CardHeader>
        <CardBody>
          <BarChart
            categories={buckets.map((row) => row[0])}
            series={[{ name: "GPU time", data: buckets.map((row) => row[1]), tone: "info" }]}
            horizontal
            height={390}
            valueSuffix=" ms"
            showValues
          />
          <Text size="small" tone="tertiary">
            X-axis: GPU time per representative decode step (milliseconds).
            Y-axis: kernel role. Source: rank-median CUDA Graph trace.
          </Text>
        </CardBody>
      </Card>

      <Stack gap={10}>
        <H2>Kernel composition</H2>
        <Table
          headers={["Role", "Time/step", "Share", "Launches"]}
          rows={buckets.map((row) => [
            row[0],
            `${row[1].toFixed(2)} ms`,
            `${row[2].toFixed(1)}%`,
            row[3].toString(),
          ])}
          columnAlign={["left", "right", "right", "right"]}
          rowTone={buckets.map((row) =>
            row[0] === "PyTorch native"
              ? "warning"
              : row[2] >= 15
                ? "info"
                : undefined,
          )}
          striped
        />
      </Stack>

      <Callout tone="info" title="The main gap is not naive PyTorch">
        MoE route/GEMMs consume about 35% of the step. Dense GEMMs and full
        attention consume another 37%. These paths already use AITER, FlyDSL,
        Triton, or HIP kernels. Replacing every PyTorch-native operation has a
        hard ceiling near 7%; a realistic first pass is 3–5%.
      </Callout>

      <Stack gap={10}>
        <H2>PyTorch-native materializations</H2>
        <Table
          headers={["Callsite", "Operation", "Time/step", "Calls/step", "Candidate"]}
          rows={[
            ["kimi_k3.py::_forward_routed", "latent.copy_", "0.34 ms", "92", "Honor zero-copy output"],
            ["topk.py::biased_grouped_topk_gpu", "bias dtype copy", "0.32 ms", "92", "Cache converted bias"],
            ["fused_norm_gate.py", "output/residual copy", "0.27 ms", "69", "Return Triton output directly"],
            ["forward_mla.py::forward_absorb_core", "torch.cat", "0.28 ms", "48", "Fuse concat + cache write"],
            ["memory_pool.py::set_kv_buffer", "indexed KV write", "0.13 ms", "24", "Triton scatter"],
            ["forward_mla.py", "weight scale/copy", "0.30 ms", "24 layers", "Cache scaled weight"],
          ]}
          columnAlign={["left", "left", "right", "right", "left"]}
          rowTone={["warning", "warning", "warning", "warning", "warning", "warning"]}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Measured optimization decisions</H2>
          <Pill tone="success">GSM8K 0.980–0.990</Pill>
        </Row>
        <Table
          headers={["Experiment", "Output throughput", "Median TPOT", "Median ITL", "Decision"]}
          rows={[
            ["MoE copy cleanup", "+0.64%", "−0.54%", "−0.47%", "Retained"],
            ["ROCm KDA fusion (short screen)", "+0.82%", "−0.78%", "−0.71%", "Provisional"],
            ["MLA preparation fusion", "+0.14%", "−0.19%", "−0.24%", "Reverted: noise-level"],
            ["MoE + KDA + MLA (matched full run)", "+0.23%", "−0.06%", "−0.13%", "KDA/MLA reverted"],
          ]}
          columnAlign={["left", "right", "right", "right", "left"]}
          rowTone={["success", "warning", "warning", "warning"]}
          striped
        />
        <Text size="small" tone="tertiary">
          Source: TP8 serving A/B · DCP/DSPARK off · Triton attention · fixed
          8192 input / 1024 output · concurrency 32. MLA and combined rows use
          64 warmups + 256 measured requests; the KDA row was an earlier
          32-request screen.
        </Text>
      </Stack>

      <Grid columns={2} gap={20}>
        <Card>
          <CardHeader>Completed first pass</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Retained: cache AITER correction-bias cast.</Text>
              <Text>Retained: remove routed-MoE latent fallback copy.</Text>
              <Text>Reverted: MLA concat/cache fusion was noise-level.</Text>
              <Text>Reverted: KDA fusion did not cross the 1% gate.</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Higher-effort kernel work</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Tune Triton grouped MLA decode (15.8%).</Text>
              <Text>Improve FlyDSL MoE stage 1/2 and routing (34.9%).</Text>
              <Text>Port K3 AR/norm fusion beyond SM100/SM103.</Text>
              <Text>Reduce or overlap 187 custom AR launches (5.8%).</Text>
              <Text>Improve dense GEMM selection/fusion (21.1%).</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="Trace interpretation">
        CUDA Graph traces expose one representative captured kernel set. Eager
        traces provide reliable kernel-to-code mapping but run around 3.3×
        slower because they execute thousands of host launches. Collective
        durations include rank-dependent spin, so raw slow-rank sums are not
        transfer cost.
      </Callout>
    </Stack>
  );
}
