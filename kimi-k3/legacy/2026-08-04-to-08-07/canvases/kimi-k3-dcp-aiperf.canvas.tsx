import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
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

const concurrency = ["6", "12", "24", "32"];
const historicalThroughput = [64.06, 83.76, 101.59, 109.63];
const baselineThroughput = [69.21, 82.27, 113.51, 101.72];
const dcpThroughput = [43.12, 50.66, 67.08, 72.46];
const tunedThroughput = [75.47, 97.38, 112.31, 114.23];
const customThroughput = [75.33, 94.22, 120.02, 121.46];
const prThroughput = [79.43, 100.69, 122.04, 142.41];
const fp8Throughput = [91.08, 120.7, 150.83, 172.68];
const baselineTtft = [10.09, 24.32, 27.04, 44.73];
const dcpTtft = [18.96, 39.84, 61.5, 71.69];
const tunedTtft = [7.03, 15.95, 26.18, 34.35];
const customTtft = [7.72, 14.73, 22.51, 32.95];
const prTtft = [7.52, 14.0, 21.78, 31.14];
const fp8Ttft = [5.47, 10.53, 15.82, 24.53];
const baselineItl = [38.74, 68.14, 96.17, 138.96];
const dcpItl = [29.72, 37.46, 59.62, 73.02];
const tunedItl = [26.5, 43.38, 70.09, 117.31];
const customItl = [33.13, 45.49, 80.41, 126.4];
const prItl = [31.66, 43.04, 79.67, 89.39];
const fp8Itl = [30.32, 39.64, 64.67, 80.42];
const baselineHit = [73.4, 75.0, 80.9, 80.0];
const dcpHit = [77.9, 79.3, 83.3, 85.2];
const tunedHit = [77.8, 80.8, 83.0, 85.3];
const customHit = [78.8, 81.1, 84.1, 86.4];
const prHit = [78.9, 81.2, 84.0, 86.3];
const fp8Hit = [78.8, 81.2, 84.0, 86.0];

const rows = concurrency.map((value, index) => [
  value,
  baselineThroughput[index].toFixed(2),
  (baselineThroughput[index] / 8).toFixed(2),
  dcpThroughput[index].toFixed(2),
  (dcpThroughput[index] / 8).toFixed(2),
  `${(((dcpThroughput[index] / baselineThroughput[index]) - 1) * 100).toFixed(1)}%`,
  `${baselineTtft[index].toFixed(2)} / ${dcpTtft[index].toFixed(2)}`,
  `${baselineItl[index].toFixed(2)} / ${dcpItl[index].toFixed(2)}`,
  `${baselineHit[index].toFixed(1)} / ${dcpHit[index].toFixed(1)}`,
]);

export default function KimiK3DcpAiperf() {
  const theme = useHostTheme();

  return (
    <Stack
      gap={20}
      style={{
        padding: 24,
        background: theme.bg.editor,
        color: theme.text.primary,
        minHeight: "100%",
      }}
    >
      <Stack gap={6}>
        <Row justify="space-between" align="center">
          <H1>Kimi-K3 long-context: cfg7 vs DCP8</H1>
          <Pill tone="success">4/4 runs, zero errors</Pill>
        </Row>
        <Text tone="secondary">
          8× MI355X · ISL ≈68,086 · OSL 350 · 8 shared prefixes · aiperf 0.11.0
        </Text>
      </Stack>

      <Grid columns={4} gap={18}>
        <Stat value="113.51 tok/s" label="Baseline peak (concurrency 24)" />
        <Stat value="172.68 tok/s" label="FP8 KV peak (concurrency 32)" tone="success" />
        <Stat value="+52.1%" label="FP8 KV peak vs baseline" tone="success" />
        <Stat value="0" label="Request errors across both sweeps" tone="success" />
      </Grid>

      <Callout tone="warning" title="Comparison boundary">
        This reproduces the cfg7 workload on the current e6311f7-based working tree,
        not the historical clean 0e756912 commit. The baseline matches the published
        range, but this is the same-code A/B comparison needed to isolate DCP.
      </Callout>

      <Grid columns="1.35fr 1fr" gap={20} align="stretch">
        <Card>
          <CardHeader trailing="Higher is better">
            Output token throughput
          </CardHeader>
          <CardBody>
            <BarChart
              categories={concurrency}
              series={[
                { name: "Published cfg7", data: historicalThroughput, tone: "neutral" },
                { name: "Current cfg7 baseline", data: baselineThroughput, tone: "success" },
                { name: "Current DCP8", data: dcpThroughput, tone: "warning" },
                { name: "Tuned DCP8", data: tunedThroughput, tone: "info" },
                { name: "Custom 132k DCP8", data: customThroughput, tone: "success" },
                { name: "Custom + PR 33599", data: prThroughput },
                { name: "Custom + PR + FP8 KV", data: fp8Throughput },
              ]}
              height={300}
              valueSuffix=" tok/s"
              showValues
            />
            <Text size="small" tone="tertiary">
              X-axis: client concurrency · Y-axis: aggregate output tokens/second.
              Source: aiperf sweep artifacts, 2026-08-05.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing="Lower is better">Median TTFT</CardHeader>
          <CardBody>
            <LineChart
              categories={concurrency}
              series={[
                { name: "Current cfg7 baseline", data: baselineTtft, tone: "success" },
                { name: "Current DCP8", data: dcpTtft, tone: "warning" },
                { name: "Tuned DCP8", data: tunedTtft, tone: "info" },
                { name: "Custom 132k DCP8", data: customTtft, tone: "success" },
                { name: "Custom + PR 33599", data: prTtft },
                { name: "Custom + PR + FP8 KV", data: fp8Ttft },
              ]}
              height={300}
              valueSuffix=" s"
              showValues
            />
            <Text size="small" tone="tertiary">
              X-axis: client concurrency · Y-axis: p50 time to first token (seconds).
              Source: aiperf sweep artifacts, 2026-08-05.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={2} gap={20}>
        <Card>
          <CardHeader trailing="Lower is better">Median ITL</CardHeader>
          <CardBody>
            <LineChart
              categories={concurrency}
              series={[
                { name: "Current cfg7 baseline", data: baselineItl, tone: "success" },
                { name: "Current DCP8", data: dcpItl, tone: "warning" },
                { name: "Tuned DCP8", data: tunedItl, tone: "info" },
                { name: "Custom 132k DCP8", data: customItl, tone: "success" },
                { name: "Custom + PR 33599", data: prItl },
                { name: "Custom + PR + FP8 KV", data: fp8Itl },
              ]}
              height={250}
              valueSuffix=" ms"
              showValues
            />
            <Text size="small" tone="tertiary">
              X-axis: client concurrency · Y-axis: p50 inter-token latency (milliseconds).
              Source: aiperf sweep artifacts, 2026-08-05.
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing="Higher is better">Average prefix-cache hit</CardHeader>
          <CardBody>
            <BarChart
              categories={concurrency}
              series={[
                { name: "Current cfg7 baseline", data: baselineHit, tone: "success" },
                { name: "Current DCP8", data: dcpHit, tone: "warning" },
                { name: "Tuned DCP8", data: tunedHit, tone: "info" },
                { name: "Custom 132k DCP8", data: customHit, tone: "success" },
                { name: "Custom + PR 33599", data: prHit },
                { name: "Custom + PR + FP8 KV", data: fp8Hit },
              ]}
              height={250}
              valueSuffix="%"
              yMin={65}
              yMax={90}
              showValues
            />
            <Text size="small" tone="tertiary">
              X-axis: client concurrency · Y-axis: average prompt-cache read percentage.
              Source: aiperf usage metrics, 2026-08-05.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>Measured results</H2>
        <Table
          headers={[
            "Concurrency",
            "Baseline out tok/s",
            "Baseline tok/s/GPU",
            "DCP8 out tok/s",
            "DCP8 tok/s/GPU",
            "DCP delta",
            "TTFT p50 base/DCP (s)",
            "ITL p50 base/DCP (ms)",
            "Hit base/DCP (%)",
          ]}
          rows={rows}
          columnAlign={[
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          rowTone={["warning", "warning", "danger", "warning"]}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Matched page-size A/B at concurrency 24</H2>
          <Pill tone="warning">Keep page size 32</Pill>
        </Row>
        <Table
          headers={[
            "Page size",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["32", "63.88", "7.99", "61.48", "227.23", "59.43", "82.7%"],
            ["64", "59.86", "7.48", "64.57", "235.67", "86.42", "82.3%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "danger"]}
        />
        <Text size="small" tone="secondary">
          Same seed, 32 warmups, 96 measured requests. Page size 64 reduced
          throughput by 6.3%, increased TTFT p50 by 5.0%, and increased ITL p50
          by 45.4%.
        </Text>
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Tuned DCP8 at concurrency 24</H2>
          <Pill tone="success">TTFT p50 −57.4%</Pill>
        </Row>
        <Table
          headers={[
            "Configuration",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["Non-DCP cfg7", "113.51", "14.19", "27.04", "90.23", "96.17", "80.9%"],
            ["Current DCP8", "67.08", "8.39", "61.50", "220.69", "59.62", "83.3%"],
            ["Tuned DCP8", "112.31", "14.04", "26.18", "131.47", "70.09", "83.0%"],
          ]}
          columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
          rowTone={["neutral", "danger", "success"]}
        />
        <Callout tone="success" title="Mixed prefill recovers most of the gap">
          Triton prefill + AITER DCP decode, a concurrency-sized Mamba pool, no
          INT8 Mamba checkpoint, and matched graph limits increased concurrency-24
          throughput by 67.4% versus current DCP. Median TTFT is now slightly
          better than the non-DCP cfg7 result while median ITL remains lower.
        </Callout>
      </Stack>

      <Stack gap={10}>
        <H2>Tuned DCP8 full sweep</H2>
        <Table
          headers={[
            "Concurrency",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["6", "75.47", "9.43", "7.03", "28.93", "26.50", "77.8%"],
            ["12", "97.38", "12.17", "15.95", "51.25", "43.38", "80.8%"],
            ["24", "112.31", "14.04", "26.18", "131.47", "70.09", "83.0%"],
            ["32", "114.23", "14.28", "34.35", "114.38", "117.31", "85.3%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success"]}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Custom 132k / Mamba 320 full sweep</H2>
          <Pill tone="success">Peak 121.46 tok/s</Pill>
        </Row>
        <Table
          headers={[
            "Concurrency",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["6", "75.33", "9.42", "7.72", "26.28", "33.13", "78.8%"],
            ["12", "94.22", "11.78", "14.73", "62.71", "45.49", "81.1%"],
            ["24", "120.02", "15.00", "22.51", "112.68", "80.41", "84.1%"],
            ["32", "121.46", "15.18", "32.95", "115.69", "126.40", "86.4%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success"]}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Custom 132k + PR 33599 full sweep</H2>
          <Pill tone="success">Peak 142.41 tok/s</Pill>
        </Row>
        <Table
          headers={[
            "Concurrency",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["6", "79.43", "9.93", "7.52", "27.59", "31.66", "78.9%"],
            ["12", "100.69", "12.59", "14.00", "52.83", "43.04", "81.2%"],
            ["24", "122.04", "15.25", "21.78", "110.75", "79.67", "84.0%"],
            ["32", "142.41", "17.80", "31.14", "111.70", "89.39", "86.3%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success"]}
          striped
        />
        <Text size="small" tone="secondary">
          Relative to the same Custom 132k launch before the PR, throughput
          changed by +5.4%, +6.9%, +1.7%, and +17.2% at concurrency
          6/12/24/32. Concurrency-32 median ITL improved by 29.3%.
        </Text>
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>Custom 132k + PR 33599 + FP8 KV</H2>
          <Pill tone="success">Peak 172.68 tok/s</Pill>
        </Row>
        <Table
          headers={[
            "Concurrency",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Cache hit",
          ]}
          rows={[
            ["6", "91.08", "11.39", "5.47", "21.07", "30.32", "78.8%"],
            ["12", "120.70", "15.09", "10.53", "35.41", "39.64", "81.2%"],
            ["24", "150.83", "18.85", "15.82", "96.76", "64.67", "84.0%"],
            ["32", "172.68", "21.58", "24.53", "90.90", "80.42", "86.0%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success"]}
          striped
        />
        <Callout tone="warning" title="Experimental path">
          AITER DCP with FP8 KV is blocked by default because it was previously
          unvalidated. This run used an explicit experimental opt-in. GSM8K
          scored 0.990 and the full sweep had zero request errors. Synthetic
          retrieval passed through 114.6k tokens, but broader reasoning and
          multimodal validation are still required before production use.
        </Callout>
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>FP8 KV + DSPARK ReplaySSM</H2>
          <Pill tone="warning">Avg accept 2.84</Pill>
        </Row>
        <Table
          headers={[
            "Concurrency",
            "Output tok/s",
            "Tok/s/GPU",
            "TTFT p50 (s)",
            "TTFT p90 (s)",
            "ITL p50 (ms)",
            "Vs no-spec throughput",
          ]}
          rows={[
            ["6", "94.65", "11.83", "1.04", "9.47", "37.78", "+3.9%"],
            ["12", "115.36", "14.42", "1.34", "35.59", "60.61", "−4.4%"],
            ["24", "139.01", "17.38", "2.10", "94.79", "102.26", "−7.8%"],
            ["32", "160.92", "20.12", "2.94", "84.96", "128.11", "−6.8%"],
          ]}
          columnAlign={["right", "right", "right", "right", "right", "right", "right"]}
          rowTone={["success", "warning", "danger", "danger"]}
          striped
        />
        <Callout tone="warning" title="Speculation only wins at low concurrency">
          DSPARK sharply lowers median TTFT, but its average accept length of
          2.84 does not cover verify overhead at concurrency 12 and above.
          The current CLI flag is --enable-gdn-replayssm-spec and DCP requires
          static ragged verification.
        </Callout>
        <Callout tone="info" title="8k/1k workload recovers expected acceptance">
          A separate non-DCP test with 7,998 prompt tokens, 1,000 forced output
          tokens, concurrency 32, and 128 requests measured mean accept length
          5.678 and median 7.634. The 2.84 average is therefore specific to the
          68k long-context workload.
        </Callout>
        <Callout tone="success" title="Standard SGLang 8192/1024 result">
          With fixed 8,192/1,024 lengths, concurrency 32, 64 warmups, and 256
          measured requests, sglang.benchmark.serving reached 4,344.35 total
          tok/s and 482.71 output tok/s. Scheduler average accept length was
          7.008, with median TTFT 1.63s and median ITL 42.97ms.
        </Callout>
      </Stack>

      <Stack gap={10}>
        <Row justify="space-between" align="center">
          <H2>FP8 KV long-context accuracy</H2>
          <Pill tone="success">12/12 identical</Pill>
        </Row>
        <Grid columns={4} gap={18}>
          <Stat value="12/12" label="BF16 exact retrieval" tone="success" />
          <Stat value="12/12" label="FP8 exact retrieval" tone="success" />
          <Stat value="0.002064" label="Maximum token logprob delta" />
          <Stat value="-16.5%" label="FP8 total latency vs BF16" tone="success" />
        </Grid>
        <Table
          headers={[
            "Actual prompt tokens",
            "Needle positions",
            "BF16 exact",
            "FP8 exact",
            "BF16 mean latency",
            "FP8 mean latency",
          ]}
          rows={[
            ["~7,660", "10/50/90%", "3/3", "3/3", "3.60s", "3.99s"],
            ["~31,168", "10/50/90%", "3/3", "3/3", "4.16s", "3.90s"],
            ["~64,870", "10/50/90%", "3/3", "3/3", "9.13s", "7.70s"],
            ["~114,609", "10/50/90%", "3/3", "3/3", "20.15s", "15.34s"],
          ]}
          columnAlign={["right", "center", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success"]}
          striped
        />
        <Text size="small" tone="secondary">
          BF16 and FP8 returned identical response text for every case. Mean
          absolute aligned output-token logprob difference was 0.0000434.
        </Text>
      </Stack>

      <Callout tone="info" title="Why the original DCP8 launch loses throughput">
        The original DCP8 launch lowers decode ITL substantially, but AITER prefill
        and queueing dominate this closed-loop workload. Its effective static memory fraction is 0.7905, token
        capacity is 511,680 per rank (about 4,093,440 virtual tokens across eight
        DCP shards), and effective running-request capacity is 29. DCP therefore
        has more aggregate KV capacity than the 1,225,230-token baseline, but its
        slower prefill and Mamba request cap increase TTFT enough to reduce
        aggregate output throughput.
      </Callout>
    </Stack>
  );
}
