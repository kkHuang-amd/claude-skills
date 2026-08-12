import {
  BarChart,
  Callout,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const concurrency = ["C2", "C4", "C8", "C16", "C32"];

export default function KimiK3PerformanceOptimizationReport() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Stack gap={8}>
        <Row align="center" justify="space-between" wrap>
          <H1>Kimi-K3 performance optimization ledger</H1>
          <Pill active>Best C32: 6087 tok/s</Pill>
        </Row>
        <Text tone="secondary">
          Consolidated effective optimizations · MI355X ×8 · TP8 · DCP1 ·
          fixed 8192 input / 1024 output · BF16 KV
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="+13.59%" label="C32 cumulative throughput" tone="success" />
        <Stat value="-14.12%" label="C32 cumulative TPOT" tone="success" />
        <Stat value="-16.96%" label="C32 cumulative ITL" tone="success" />
        <Stat value="3" label="Retained optimizations" />
      </Grid>

      <Stack gap={8}>
        <H2>Throughput progression across retained optimizations</H2>
        <BarChart
          categories={concurrency}
          series={[
            {
              name: "Starting point",
              data: [874.94, 1569.72, 2504.34, 3868.73, 5358.96],
              tone: "neutral",
            },
            {
              name: "Remove MoE copies",
              data: [882.44, 1594.32, 2527.54, 3893.59, 5381.41],
              tone: "info",
            },
            {
              name: "Fused AITER KDA",
              data: [911.52, 1640.52, 2668.38, 4068.83, 5544.38],
              tone: "warning",
            },
            {
              name: "AITER MLA decode",
              data: [921.88, 1676.44, 2781.64, 4343.66, 6087.04],
              tone: "success",
            },
          ]}
          valueSuffix=" tok/s"
          showValues
          height={340}
        />
        <Text size="small" tone="tertiary">
          X-axis: serving concurrency · Y-axis: total token throughput
          (tok/s). Sources: retained 2026-08-06 and 2026-08-07 matched sweeps.
          The final stage keeps Triton prefill and changes only decode to AITER.
        </Text>
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>Incremental throughput gain</H2>
          <LineChart
            categories={concurrency}
            series={[
              {
                name: "Remove MoE copies",
                data: [0.86, 1.57, 0.93, 0.64, 0.42],
                tone: "info",
              },
              {
                name: "Fused AITER KDA",
                data: [3.3, 2.9, 5.57, 4.5, 3.03],
                tone: "warning",
              },
              {
                name: "AITER MLA decode",
                data: [1.14, 2.07, 4.08, 6.58, 9.62],
                tone: "success",
              },
            ]}
            valueSuffix="%"
            showValues
            height={300}
          />
          <Text size="small" tone="tertiary">
            X-axis: serving concurrency · Y-axis: throughput gain versus each
            optimization&apos;s matched immediate baseline (%).
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>Cumulative throughput gain</H2>
          <BarChart
            categories={concurrency}
            series={[
              {
                name: "All retained optimizations",
                data: [5.36, 6.8, 11.07, 12.28, 13.59],
                tone: "success",
              },
            ]}
            valueSuffix="%"
            showValues
            height={300}
          />
          <Text size="small" tone="tertiary">
            X-axis: serving concurrency · Y-axis: observed gain from the
            starting point to the final configuration (%). Cross-day baseline
            reproduction differed by at most 0.17%.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Optimization ledger</H2>
        <Table
          headers={[
            "Optimization",
            "C2",
            "C4",
            "C8",
            "C16",
            "C32",
            "Primary effect",
          ]}
          rows={[
            [
              "Remove repeated MoE copies",
              "+0.86%",
              "+1.57%",
              "+0.93%",
              "+0.64%",
              "+0.42%",
              "Less per-layer tensor movement",
            ],
            [
              "AITER FlyDSL fused KDA boundary",
              "+3.30%",
              "+2.90%",
              "+5.57%",
              "+4.50%",
              "+3.03%",
              "KDA bucket −32.8%",
            ],
            [
              "Triton prefill + AITER MLA decode",
              "+1.14%",
              "+2.07%",
              "+4.08%",
              "+6.58%",
              "+9.62%",
              "Decode ITL falls with concurrency",
            ],
          ]}
          columnAlign={["left", "right", "right", "right", "right", "right", "left"]}
          rowTone={["success", "success", "success"]}
          striped
        />
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>Final configuration metrics</H2>
          <Table
            headers={["Concurrency", "Throughput", "Median TTFT", "Median TPOT", "Median ITL"]}
            rows={[
              ["2", "921.88 tok/s", "855.29 ms", "18.70 ms", "18.52 ms"],
              ["4", "1676.44 tok/s", "1617.03 ms", "19.91 ms", "19.52 ms"],
              ["8", "2781.64 tok/s", "2569.28 ms", "23.35 ms", "21.91 ms"],
              ["16", "4343.66 tok/s", "4783.78 ms", "28.57 ms", "25.06 ms"],
              ["32", "6087.04 tok/s", "9024.01 ms", "38.61 ms", "31.05 ms"],
            ]}
            columnAlign={["center", "right", "right", "right", "right"]}
            rowTone={["success", "success", "success", "success", "success"]}
            striped
          />
        </Stack>

        <Stack gap={8}>
          <H2>Fused KDA kernel evidence</H2>
          <LineChart
            categories={["B1", "B2", "B4", "B8", "B12", "B16", "B24", "B32"]}
            series={[
              {
                name: "AITER FlyDSL",
                data: [11.69, 12.8, 12.96, 13.34, 13.74, 14.52, 18.5, 19.78],
                tone: "success",
              },
              {
                name: "SGLang Triton experiment",
                data: [29.5, 30.05, 30.05, 30.06, 31.19, 31.54, 37.43, 38.55],
                tone: "warning",
              },
            ]}
            valueSuffix=" µs"
            showValues
            height={280}
          />
          <Text size="small" tone="tertiary">
            X-axis: batch size · Y-axis: graph replay latency (µs). Source: 21
            paired gfx950 trials. AITER is 1.95–2.52× faster.
          </Text>
        </Stack>
      </Grid>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>AITER decode: BF16 versus FP8 KV throughput</H2>
          <LineChart
            categories={concurrency}
            series={[
              {
                name: "BF16 KV",
                data: [921.88, 1676.44, 2781.64, 4343.66, 6087.04],
                tone: "success",
              },
              {
                name: "FP8 E4M3 KV",
                data: [915.42, 1652.9, 2736.77, 4254.73, 5925.5],
                tone: "warning",
              },
            ]}
            valueSuffix=" tok/s"
            showValues
            height={290}
          />
          <Text size="small" tone="tertiary">
            X-axis: serving concurrency · Y-axis: total token throughput
            (tok/s). Both runs use Triton prefill and AITER decode; only KV
            dtype changes.
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>FP8 KV delta with AITER decode</H2>
          <Table
            headers={["Concurrency", "Throughput", "TTFT", "TPOT", "ITL"]}
            rows={[
              ["2", "-0.70%", "-0.11%", "+0.75%", "+0.76%"],
              ["4", "-1.40%", "+0.00%", "+0.95%", "+0.92%"],
              ["8", "-1.61%", "-0.01%", "+1.37%", "+1.05%"],
              ["16", "-2.05%", "+4.05%", "+1.93%", "+1.44%"],
              ["32", "-2.65%", "+3.27%", "+2.69%", "+1.87%"],
            ]}
            columnAlign={["center", "right", "right", "right", "right"]}
            rowTone={["warning", "warning", "warning", "warning", "warning"]}
            striped
          />
          <Text size="small" tone="tertiary">
            Delta is FP8 relative to BF16. AITER decode cuts the prior all-Triton
            FP8 throughput penalty from 1.44–6.30% to 0.70–2.65%, but does not
            make FP8 faster at 8k/1k.
          </Text>
        </Stack>
      </Grid>

      <Callout tone="success" title="Retained configuration">
        Keep the MoE-copy removal, AITER FlyDSL fused KDA boundary, and Triton
        prefill + AITER MLA decode. Together they raise observed C32 throughput
        from 5358.96 to 6087.04 tok/s while lowering median TPOT from 44.96 to
        38.61 ms and median ITL from 37.39 to 31.05 ms.
      </Callout>

      <Callout tone="warning" title="Measured but not retained as an 8k optimization">
        FP8 E4M3 KV still regresses the non-DCP 8k workload: 1.44–6.30% with
        all-Triton attention and 0.70–2.65% with AITER decode. It is excluded
        from the current production optimization total. The final experimental
        PR #4450/#4480 hybrid becomes up to 3.79% faster than matched BF16 after
        fixing FP8 CUDA-Graph runtime split selection, but its newer AITER/Triton
        base still trails the production BF16 C32 result by 2.57%.
      </Callout>

      <Text size="small" tone="tertiary">
        Artifacts: stage2-runs/2026-08-06-concurrency-sweep-current,
        2026-08-06-concurrency-sweep-moe-copy,
        2026-08-06-kda-flydsl-vs-triton, and
        2026-08-07-8k1k-attn-backend-ab (including aiter-decode-fp8kv).
      </Text>
    </Stack>
  );
}
