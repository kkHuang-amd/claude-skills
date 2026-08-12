import {
  BarChart,
  Callout,
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

export default function KimiK3FP8KVCrossoverAnalysis() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Stack gap={8}>
        <Row align="center" justify="space-between" wrap>
          <H1>Why FP8 KV loses at 8k but wins at 68k</H1>
          <Pill active>Kernel-regime crossover</Pill>
        </Row>
        <Text tone="secondary">
          Kimi-K3 · MI355X ×8 · Triton prefill · AITER decode · BF16 versus
          FP8 E4M3 KV
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="-0.70–2.65%" label="8k AITER-decode FP8 throughput" tone="warning" />
        <Stat value="+14.7–23.6%" label="68k DCP8 FP8 throughput" tone="success" />
        <Stat value="8.2k vs 8.5k" label="KV tokens read per rank" />
        <Stat value="1 vs 32" label="KV page size: 8k vs 68k" />
      </Grid>

      <Callout tone="warning" title="The context-length explanation alone is insufficient">
        Non-DCP 8k reads about 8192 KV tokens per rank. DCP8 shards the 68k
        sequence to roughly 8500 KV tokens per rank. The local working-set
        lengths are therefore similar. The sign change comes from different
        kernels, page layouts, query-head shapes, and capacity behavior—not
        merely from 68k being a larger number.
      </Callout>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>Observed FP8 throughput delta</H2>
          <BarChart
            categories={["Low", "Mid-low", "Mid-high", "High"]}
            series={[
              {
                name: "8k non-DCP, AITER decode",
                data: [-0.7, -1.4, -2.05, -2.65],
                tone: "warning",
              },
              {
                name: "68k DCP8, AITER decode",
                data: [14.7, 19.9, 23.6, 21.3],
                tone: "success",
              },
            ]}
            valueSuffix="%"
            showValues
            height={300}
          />
          <Text size="small" tone="tertiary">
            X-axis: increasing serving concurrency (8k C2/4/16/32; 68k
            C6/12/24/32) · Y-axis: FP8 throughput delta versus matched BF16
            (%). Sources: retained 2026-08-07 and historical matched sweeps.
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>Measured latency signature</H2>
          <Table
            headers={["Workload", "FP8 TTFT", "FP8 TPOT", "FP8 ITL", "Interpretation"]}
            rows={[
              [
                "8k, AITER decode",
                "−0.1% to +4.1%",
                "+0.8% to +2.7%",
                "+0.8% to +1.9%",
                "Write/dequant + kernel overhead dominates",
              ],
              [
                "68k, DCP8 AITER",
                "−21% to −27%",
                "not retained",
                "−4% to −19%",
                "Both admission/prefill and decode improve",
              ],
            ]}
            columnAlign={["left", "right", "right", "right", "left"]}
            rowTone={["warning", "success"]}
            striped
          />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Confirmed execution-path differences</H2>
        <Table
          headers={["Dimension", "8k non-DCP", "68k DCP8", "Why it matters"]}
          rows={[
            [
              "Decode implementation",
              "aiter.mla ASM wrapper",
              "Gluon MLA skip-reduce + SGLang base-2 reduce",
              "Different kernels; dtype A/B is not one identical kernel",
            ],
            [
              "BF16 / FP8 mode",
              "BF16 forced non-persistent; FP8 enables persistent metadata",
              "Same DCP Gluon dispatch for both dtypes",
              "8k changes dtype and scheduling regime together",
            ],
            [
              "KV page",
              "page_size=1",
              "page_size=32",
              "DCP path reads contiguous page-sized Gluon tiles",
            ],
            [
              "Query heads per rank",
              "12 padded to 16",
              "96 gathered DCP heads",
              "Different reuse, occupancy, and reduction geometry",
            ],
            [
              "FP8 handling",
              "FP8 write + persistent metadata + in-kernel descale",
              "Fused paged write + native FP8 Gluon read/descale",
              "DCP stack better amortizes conversion and metadata",
            ],
            [
              "KV capacity",
              "838k BF16 tokens already sufficient; FP8 gives 1.676M",
              "454,528 BF16 to 909,088 FP8 in historical run",
              "8k gets no admission benefit; long run gains headroom",
            ],
          ]}
          columnAlign={["left", "left", "left", "left"]}
          striped
        />
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>8k cost model</H2>
          <Table
            headers={["Effect", "Direction", "Evidence"]}
            rows={[
              ["Halved KV read bytes", "helps", "FP8 stores twice as many tokens in equal 21.58 GB"],
              ["FP8 quantized cache write", "hurts", "TTFT rises 3–4% at C16/C32"],
              ["FP8 descale in decode", "hurts", "TPOT and ITL regress at every concurrency"],
              ["Persistent-path switch", "likely hurts", "BF16 explicitly disables it for padded 16-head gfx950"],
              ["Extra capacity", "neutral", "BF16 already holds far more than 32 × 9k live tokens"],
            ]}
            columnAlign={["left", "center", "left"]}
            rowTone={["success", "warning", "warning", "warning", "neutral"]}
            striped
          />
        </Stack>

        <Stack gap={8}>
          <H2>68k DCP8 benefit model</H2>
          <Table
            headers={["Effect", "Direction", "Evidence"]}
            rows={[
              ["Halved KV bytes", "helps", "Native FP8 read/descale inside DCP Gluon"],
              ["Page-32 contiguous tiles", "helps", "Historical tuning: page 32 fastest"],
              ["Same kernel family across dtype", "helps", "Avoids 8k persistent/non-persistent mismatch"],
              ["Larger effective capacity", "helps", "Historical token capacity doubles"],
              ["Cross-rank merge", "cost", "Base-2 LSE reduce remains required for both dtypes"],
            ]}
            columnAlign={["left", "center", "left"]}
            rowTone={["success", "success", "success", "success", "neutral"]}
            striped
          />
        </Stack>
      </Grid>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>PR #4450/#4480 isolated kernel</H2>
          <Table
            headers={["Batch", "BF16", "FP8", "FP8 speedup"]}
            rows={[
              ["1", "70.85 µs", "85.21 µs", "0.83×"],
              ["2", "52.03 µs", "48.16 µs", "1.08×"],
              ["8", "34.19 µs", "27.73 µs", "1.23×"],
              ["32", "65.51 µs", "43.90 µs", "1.49×"],
              ["64", "118.00 µs", "69.11 µs", "1.71×"],
            ]}
            columnAlign={["center", "right", "right", "right"]}
            rowTone={["warning", "success", "success", "success", "success"]}
            striped
          />
          <Text size="small" tone="tertiary">
            Context 8192, 12 heads, page 1. Output and LSE checks pass for both
            dtypes.
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>PR #4450/#4480 matched serving</H2>
          <Table
            headers={["Concurrency", "BF16 tok/s", "FP8 tok/s", "FP8 delta"]}
            rows={[
              ["2", "906.28", "783.65", "-13.53%"],
              ["4", "1665.03", "1433.30", "-13.92%"],
              ["8", "2741.79", "2439.18", "-11.04%"],
              ["16", "4169.76", "3874.29", "-7.09%"],
              ["32", "5702.74", "5564.52", "-2.42%"],
            ]}
            columnAlign={["center", "right", "right", "right"]}
            rowTone={["danger", "danger", "danger", "warning", "warning"]}
            striped
          />
          <Text size="small" tone="tertiary">
            Triton prefill, unified Gluon decode, 8192/1024. The kernel gain
            does not survive the SGLang serving integration.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Old padded ASM versus new native 12-head Gluon BF16</H2>
        <Table
          headers={[
            "Batch",
            "Old ASM16",
            "12→16 pad",
            "Old combined",
            "New Gluon12",
            "Gluon delta",
          ]}
          rows={[
            ["1", "30.93 µs", "8.79 µs", "39.72 µs", "70.85 µs", "+78.37%"],
            ["2", "33.17 µs", "10.17 µs", "43.34 µs", "52.03 µs", "+20.05%"],
            ["8", "35.90 µs", "10.34 µs", "46.24 µs", "34.19 µs", "-26.06%"],
            ["32", "68.97 µs", "10.53 µs", "79.50 µs", "65.51 µs", "-17.60%"],
            ["64", "120.07 µs", "10.44 µs", "130.51 µs", "118.00 µs", "-9.59%"],
          ]}
          columnAlign={["center", "right", "right", "right", "right", "right"]}
          rowTone={["danger", "warning", "success", "success", "success"]}
          striped
        />
        <Text size="small" tone="tertiary">
          Context 8192, BF16 KV. Positive delta means Gluon is slower. The old
          total conservatively adds independently measured F.pad latency to the
          16-head ASM kernel.
        </Text>
      </Stack>

      <Callout tone="warning" title="BF16 dispatch must be batch-gated">
        Native 12-head Gluon is 20–78% slower at B1/B2 but 10–26% faster from
        B8 upward after accounting for old padding. An unconditional switch to
        PR #4450 explains the low-concurrency regression; a hybrid ASM/Gluon
        dispatch is required.
      </Callout>

      <Stack gap={8}>
        <H2>Controlled BF16 serving: same AITER base and Triton runtime</H2>
        <Table
          headers={["Concurrency", "Old padded ASM", "New Gluon", "Gluon delta"]}
          rows={[
            ["2", "922.54 tok/s", "906.28 tok/s", "-1.76%"],
            ["4", "1672.78 tok/s", "1665.03 tok/s", "-0.46%"],
            ["8", "2737.91 tok/s", "2741.79 tok/s", "+0.14%"],
            ["16", "4146.75 tok/s", "4169.76 tok/s", "+0.55%"],
            ["32", "5684.14 tok/s", "5702.74 tok/s", "+0.33%"],
          ]}
          columnAlign={["center", "right", "right", "right"]}
          rowTone={["warning", "warning", "success", "success", "success"]}
          striped
        />
        <Text size="small" tone="tertiary">
          This removes the newer-AITER/Triton confounder. The kernel crossover
          survives end to end, but B8+ gains shrink to 0.1–0.6% because MLA is
          only part of the full network.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Old padded ASM: BF16 versus FP8 KV</H2>
        <Table
          headers={[
            "Concurrency",
            "BF16 tok/s",
            "FP8 tok/s",
            "FP8 throughput",
            "FP8 TTFT",
            "FP8 TPOT",
            "FP8 ITL",
          ]}
          rows={[
            ["2", "922.54", "916.55", "-0.65%", "+0.08%", "+0.68%", "+0.68%"],
            ["4", "1672.78", "1659.26", "-0.81%", "+0.14%", "+1.08%", "+0.82%"],
            ["8", "2737.91", "2740.87", "+0.11%", "+0.10%", "+0.04%", "+0.80%"],
            ["16", "4146.75", "4183.65", "+0.89%", "-5.89%", "-0.21%", "+1.25%"],
            ["32", "5684.14", "5771.46", "+1.54%", "-6.48%", "-0.40%", "+2.01%"],
          ]}
          columnAlign={["center", "right", "right", "right", "right", "right", "right"]}
          rowTone={["warning", "warning", "neutral", "success", "success"]}
          striped
        />
        <Text size="small" tone="tertiary">
          Same newer AITER base and Triton 3.7 runtime, with the unified Gluon
          path disabled. FP8 is slightly negative at C2/C4, neutral at C8, and
          improves total throughput at C16/C32 primarily through lower TTFT.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Fixed FP8 CUDA-Graph runtime split results</H2>
        <Table
          headers={["Concurrency", "Gluon BF16", "Fixed Gluon FP8", "FP8 gain", "TPOT delta", "ITL delta"]}
          rows={[
            ["2", "906.28 tok/s", "914.19 tok/s", "+0.87%", "-0.92%", "-0.91%"],
            ["4", "1665.03 tok/s", "1674.83 tok/s", "+0.59%", "+0.05%", "+0.20%"],
            ["8", "2741.79 tok/s", "2781.77 tok/s", "+1.46%", "-0.74%", "-0.05%"],
            ["16", "4169.76 tok/s", "4271.59 tok/s", "+2.44%", "-1.98%", "-0.74%"],
            ["32", "5702.74 tok/s", "5936.61 tok/s", "+4.10%", "-3.25%", "-1.59%"],
          ]}
          columnAlign={["center", "right", "right", "right", "right", "right"]}
          rowTone={["success", "success", "success", "success", "success"]}
          striped
        />
      </Stack>

      <Callout tone="success" title="Defect found and fixed">
        SGLang CUDA Graph capture supplies a placeholder sequence length of 1.
        FP8 bh16bn128 used that host value to compile one KV split, so every
        replay scanned the full 8k sequence with one workgroup. Extending
        PR #4450&apos;s device-side runtime split policy from BF16 to FP8 turns
        the regression into a 0.59–4.10% throughput gain.
      </Callout>

      <Grid columns={2} gap={20}>
        <Stack gap={8}>
          <H2>Runtime 2×2 attribution at C32</H2>
          <Table
            headers={["AITER base", "Triton 3.6", "Triton 3.7"]}
            rows={[
              ["Production AITER", "6087.04 tok/s", "5817.34 tok/s"],
              ["New PR AITER", "5945.07 tok/s", "5684.14 tok/s"],
            ]}
            columnAlign={["left", "right", "right"]}
            rowTone={["success", "warning"]}
            striped
          />
          <Text size="small" tone="tertiary">
            Triton 3.7 contributes about −4.43%; the newer AITER base adds about
            −2.33%. The effects are approximately additive.
          </Text>
        </Stack>

        <Stack gap={8}>
          <H2>Profile attribution</H2>
          <Table
            headers={["Bucket", "Best runtime", "Worst runtime", "Finding"]}
            rows={[
              ["BF16 prefill attention", "2090 ms", "4775 ms", "Triton 3.7: 2.28× slower"],
              ["FP8 prefill attention", "2949 ms", "3295 ms", "Triton 3.7: 1.12× slower"],
              ["Decode MoE stage2", "3.86 ms", "4.22 ms", "New AITER kernel selection"],
              ["MLA ASM microbench", "baseline", "within 1%", "Not the runtime regression"],
              ["KDA fused decode", "1.25 ms", "1.25 ms", "Stable across AITER bases"],
            ]}
            columnAlign={["left", "right", "right", "left"]}
            striped
          />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Final batch-gated hybrid</H2>
        <Table
          headers={["Concurrency", "Hybrid BF16", "Hybrid FP8", "FP8 delta"]}
          rows={[
            ["2", "923.63 tok/s", "919.29 tok/s", "-0.47%"],
            ["4", "1672.91 tok/s", "1676.42 tok/s", "+0.21%"],
            ["8", "2769.79 tok/s", "2784.47 tok/s", "+0.53%"],
            ["16", "4172.76 tok/s", "4278.00 tok/s", "+2.52%"],
            ["32", "5714.25 tok/s", "5930.87 tok/s", "+3.79%"],
          ]}
          columnAlign={["center", "right", "right", "right"]}
          rowTone={["neutral", "success", "success", "success", "success"]}
          striped
        />
        <Text size="small" tone="tertiary">
          BF16 uses padded ASM below B8; FP8 uses padded ASM below B4. Higher
          batches use native Gluon with device-side runtime split selection.
        </Text>
      </Stack>

      <Callout tone="warning" title="Final decision">
        The FP8 defect is fixed and the hybrid passes retrieval/logprob parity,
        but the new runtime&apos;s C32 result (5930.87 tok/s) remains 2.57%
        below production AITER/Triton 3.6 BF16 (6087.04 tok/s). Retain the
        production runtime until the Triton 3.7 prefill regression and the new
        AITER MoE stage2 regression are fixed or selectively backported.
      </Callout>
    </Stack>
  );
}
