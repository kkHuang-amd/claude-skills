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
          Stable experiment ledger · MI355X/gfx950 · updated 2026-08-21
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="0.990" label="Current GSM8K 200" tone="success" />
        <Stat value="933,883" label="Max token capacity" tone="success" />
        <Stat value="46" label="Vendored FlyDSL tests" tone="success" />
        <Stat value="+9.36%" label="KDA + optional B2 C2 gain" tone="info" />
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
          ["2026-08-12", "Fresh environment reproduction · crsuse2-m2m-002", "C2-C32 +1.20–1.68%; B2 C2 +9.47%", "Environment validated; GSM8K 200 caveat retained"],
          ["2026-08-12", "ATOM Kimi-K3 recipe · crsuse2-m2m-002", "C2 799.99 · C64 8380.39 tok/s", "ATOM trails through C32, then leads C64 by 4.57%"],
          ["2026-08-13", "ATOM C64 stream A/B · current stack", "Multi 8742.34 · single 8399.50 tok/s", "Dual-stream MoE retained; single-stream is −3.92%"],
          ["2026-08-21", "ATOM current-main C2 stream A/B", "Five-run median 780.43 vs 767.18 tok/s; TPOT 22.17 vs 22.64 ms", "Multi-stream positive in 5/5 runs; use +1.73% conservative gain for SGLang direction"],
          ["2026-08-21", "ATOM current-main C64 stream A/B", "Multi 8680.09 vs single 9878.91 tok/s; TPOT 51.22 vs 44.65 ms", "Reject multi-stream at C64: throughput −12.14%, TTFT +9.76%, TPOT +14.70%"],
          ["2026-08-21", "Same-AITER common-client SGLang/ATOM C64", "SGLang 9800.69 vs ATOM 9846.50 tok/s; median E2E near 60 s", "Throughput parity within 0.47%; native-client 10% gap invalid"],
          ["2026-08-21", "Same-AITER common-client SGLang/ATOM C2", "Five-run median 1138.02 vs 768.04 tok/s; TPOT 14.90 vs 22.62 ms", "Real low-concurrency gap: SGLang +48.17% throughput and −34.1% TPOT"],
          ["2026-08-21", "SGLang old/current AITER C2 A/B", "Five-run median 1146.00 vs 1136.17 tok/s; TTFT 844.71 vs 871.71 ms", "Current main validated but remains experimental: throughput −0.86%, outside 0.5% gate"],
          ["2026-08-13", "ATOM/SGLang C64 trace attribution", "Single-stream ATOM leads SGLang by 4.81%", "Isolate precision/runtime/kernel gap before dual-stream work"],
          ["2026-08-13", "SGLang Kimi MLA Q/cache fusion", "BF16 C64 +0.53% · FP8 C64 +3.57%", "Retain default-off; accepted for FP8 KV only"],
          ["2026-08-13", "SGLang direct A16W4 switch", "Server ready; GSM8K 50 = 0.040", "Reject flag-only switch; caller/layout contract differs"],
          ["2026-08-13", "SGLang A16W4 caller-contract fix", "Correctness restored; matched sweep −3.13% to −9.86% throughput vs A8W4", "Keep layout fix; reject A16W4 production mode"],
          ["2026-08-14", "A8W4 + fused Q/KV prep + FP8 KV sweep", "C2-C64 throughput +0.73% to +3.38%; TPOT improved at every point", "Retain FP8-KV fusion profile"],
          ["2026-08-17", "AITER prefill + tuned Triton decode", "C64 +6.51% throughput · TTFT −16.05% · eight-rank compact trace", "Retain optional high-concurrency profile; C2 is −3.07%"],
          ["2026-08-17", "Triton MLA Q/cache fusion", "GSM8K 0.951 · Q CatArray removed · C64 +0.68% · TPOT −1.02%", "Retain BF16 Q + FP8 KV; reject FP8 Q/Q-PE"],
          ["2026-08-18", "ATT-driven KDA decode", "69-layer graph 9.20→8.38 µs/layer; C2 endpoint +0.19%", "Retain opt-in; endpoint below 0.5% gate"],
          ["2026-08-18", "KDA winner + B2 projections", "C2 1002.26→1096.04 tok/s; TPOT −8.99%", "Retain optional B2 composition"],
          ["2026-08-18", "M4 single fused mixed preroute", "C4 1834.22→1867.15 tok/s; TPOT −1.90%", "Superseded by cooperative producer; raw dispatch removed"],
          ["2026-08-18", "M4 cooperative preactivated producer", "Five-round C4 1862.80→1920.22 tok/s; TPOT −3.33%; GSM8K 0.955", "Retain exact-M4 SGLang-only opt-in"],
          ["2026-08-18", "M2/M4 MoE design unification", "M2 five-round +1.08%; unified M2+KDA 1111.97 tok/s; GSM8K 0.951", "Remove old M2 tri/shared-down; retain one cooperative design"],
          ["2026-08-18", "Current all-winner C2-C64 sweep", "C2/C4/C8/C16/C32/C64 = 1142/1968/3092/4798/6708/8711 tok/s", "All 1,008 requests succeeded; retain winner composition"],
          ["2026-08-18", "KDA and shared-down redesigns", "split-V/async/MFMA hybrid/shared-down all miss E2E gates", "No-go decisions documented; rejected source removed"],
        ]}
        rowTone={[
          "neutral",
          "success",
          "success",
          "info",
          "success",
          "success",
          "success",
          "success",
          "warning",
          "success",
          "info",
          "success",
          "success",
          "info",
          "warning",
          "success",
          "success",
          "warning",
          "success",
          "warning",
          "warning",
          "success",
          "info",
          "info",
          "success",
          "info",
          "warning",
          "warning",
          "warning",
          "success",
          "info",
          "success",
          "success",
          "warning",
        ]}
      />
      <H2>Current ATOM C2 multi-stream reference</H2>
      <Grid columns={3} gap={14}>
        <Stat value="+1.73%" label="Median throughput gain" tone="success" />
        <Stat value="−2.08%" label="Median TPOT change" tone="success" />
        <Stat value="160/160" label="Measured requests succeeded" tone="success" />
      </Grid>
      <Table
        striped
        headers={["Round", "Multi-stream tok/s", "Single-stream tok/s", "Multi TPOT", "Single TPOT", "Throughput delta"]}
        rows={[
          ["1", "775.32", "762.95", "22.20 ms", "22.80 ms", "+1.62%"],
          ["2", "780.43", "758.09", "22.17 ms", "22.66 ms", "+2.95%"],
          ["3", "779.71", "767.18", "22.18 ms", "22.64 ms", "+1.63%"],
          ["4", "800.92", "767.26", "22.15 ms", "22.64 ms", "+4.39%"],
          ["5", "839.38", "767.57", "20.56 ms", "22.64 ms", "+9.36%"],
        ]}
        rowTone={["success", "success", "success", "success", "success"]}
      />
      <Callout tone="warning" title="Use the median as the conservative SGLang input">
        Fixed 8,192/1,024 C2, 16 measured requests and four warmups per round,
        TP8, FULL graph mode, current ATOM 27f8639b and AITER dc4bdf1c. The
        multi-stream side warmed upward in rounds four and five, while
        single-stream stayed stable; the mean gain is +3.99% but is not the
        production claim. Multi-stream still won every round, supporting a
        default-off SGLang prototype with explicit event and TP collective
        ordering.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/ATOM_C2_STREAM_AB_2026-08-21.md">
          Full current-main A/B report
        </Link>
      </Callout>
      <Grid columns={3} gap={14}>
        <Stat value="−12.14%" label="C64 multi-stream throughput" tone="warning" />
        <Stat value="+9.76%" label="C64 multi-stream TTFT" tone="warning" />
        <Stat value="+14.70%" label="C64 multi-stream TPOT" tone="warning" />
      </Grid>
      <Callout tone="warning" title="Current-main multi-stream crosses over before C64">
        Fixed 8,192/1,024 C64 with 128 warmups and 512 measured requests:
        multi-stream reached 8,680.09 tok/s versus 9,878.91 single-stream.
        Generated graphs confirm the switch was active. The current single path
        is 17.45% faster than the historical single baseline, invalidating the
        old +4.47% C64 multi-stream claim. Gate any SGLang prototype to exact C2
        initially and preserve its existing fused _add3 tail.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/ATOM_C64_STREAM_AB_2026-08-21.md">
          Full C64 stream A/B report
        </Link>
      </Callout>
      <H2>Common-client SGLang versus ATOM C64</H2>
      <Grid columns={4} gap={14}>
        <Stat value="+0.47%" label="ATOM throughput delta" tone="success" />
        <Stat value="−46.74%" label="ATOM median TTFT delta" tone="info" />
        <Stat value="+40.82%" label="ATOM median TPOT delta" tone="warning" />
        <Stat value="−0.64%" label="ATOM median E2E delta" tone="success" />
      </Grid>
      <Table
        striped
        headers={["Engine", "Total tok/s", "Output tok/s", "Median TTFT", "Median TPOT", "Median E2E"]}
        rows={[
          ["SGLang", "9,800.69", "1,088.97", "27,579.66 ms", "31.74 ms", "60,029.54 ms"],
          ["ATOM", "9,846.50", "1,094.06", "14,689.19 ms", "44.70 ms", "59,643.74 ms"],
        ]}
        rowTone={["success", "success"]}
      />
      <Callout tone="success" title="The native-client throughput gap disappears under one client">
        Both frameworks used current AITER dc4bdf1c, FlyDSL 0.3.1, the same
        640-prompt manifest and one exact streaming client. SGLang delays most
        first tokens to about 27.6 seconds and then decodes faster; ATOM
        staggers first tokens across the C64 wave and decodes more slowly.
        Makespan and median E2E are nearly identical. Use this common client for
        every future cross-engine claim.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C64_2026-08-21.md">
          Full common-client report
        </Link>
      </Callout>
      <H2>Common-client SGLang versus ATOM C2</H2>
      <Grid columns={4} gap={14}>
        <Stat value="−32.51%" label="ATOM throughput delta" tone="warning" />
        <Stat value="−9.37%" label="ATOM median TTFT delta" tone="info" />
        <Stat value="+51.77%" label="ATOM median TPOT delta" tone="warning" />
        <Stat value="+48.20%" label="ATOM median E2E delta" tone="warning" />
      </Grid>
      <Table
        striped
        headers={["Engine", "Median total tok/s", "Median TTFT", "Median TPOT", "Median E2E", "Measured requests"]}
        rows={[
          ["SGLang", "1,138.02", "960.14 ms", "14.90 ms", "16,192.43 ms", "80/80"],
          ["ATOM", "768.04", "870.18 ms", "22.62 ms", "23,997.45 ms", "80/80"],
        ]}
        rowTone={["success", "warning"]}
      />
      <Callout tone="info" title="Low-concurrency decode remains SGLang's clear advantage">
        Same current AITER, FlyDSL 0.3.1, exact 8,192/1,024 prompts and five
        common-client rounds. ATOM returns the first token about 90 ms earlier,
        but SGLang's 34.1% lower TPOT dominates the long completion. Unlike C64,
        this 32.5% throughput gap survives client control and is stable across
        rounds. Attribute KDA, MLA, MoE and TP collective decode boundaries
        with matched compact traces before porting code.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C2_2026-08-21.md">
          Full common-client C2 report
        </Link>
      </Callout>
      <H2>Single-step GPU trace attribution</H2>
      <Grid columns={4} gap={14}>
        <Stat value="+7.99 ms" label="ATOM C2 graph gap" tone="warning" />
        <Stat value="−2.38 ms" label="ATOM C64 graph gap" tone="success" />
        <Stat value="133.9 / 18.0" label="SGLang / ATOM active experts" tone="info" />
        <Stat value="3.4–7.4%" label="A8W4 full-chain advantage" tone="success" />
      </Grid>
      <Callout tone="info" title="Routed-MoE inversion is workload, not A16W4 superiority">
        Armed route dumps align all 736 TP-rank/layer pairs. SGLang spreads each
        1,024-route C64 call over 133.87 experts on average; ATOM uses only
        18.03. Replaying both exact route sets through current kernels shows
        A8W4 beats A16W4 by 3.43% and 7.44% for the complete chain. The
        production-trace A16W4 advantage comes from much less BM32-padded work.
      </Callout>
      <Grid columns={3} gap={14}>
        <Stat value="M32+" label="KDA in-proj MXFP4 crossover" tone="info" />
        <Stat value="M64" label="MLA qkv-A PTPC crossover" tone="info" />
        <Stat value="+3.05 GiB" label="Combined prepared storage / GPU" tone="warning" />
      </Grid>
      <Callout tone="warning" title="Dense candidates must remain separate and default-off">
        Complete-chain graph micros pass numerical and input-change gates, but
        the combined hybrid policy costs 3.048 GiB per GPU. Evaluate one
        candidate at a time with the same common-client workload and recapture
        the same prefill/decode GPU step. Production remains unchanged.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md">
          Full GPU-first attribution report
        </Link>
      </Callout>
      <H2>SGLang current-AITER integration gate</H2>
      <Grid columns={3} gap={14}>
        <Stat value="−0.86%" label="Current-main C2 throughput" tone="warning" />
        <Stat value="+3.20%" label="Current-main median TTFT" tone="warning" />
        <Stat value="50/50" label="Vendored FlyDSL tests passed" tone="success" />
      </Grid>
      <Table
        striped
        headers={["Round", "Old AITER tok/s", "Current AITER tok/s", "Old TTFT", "Current TTFT", "Current TPOT"]}
        rows={[
          ["1", "1,141.78", "1,135.36", "845.02 ms", "871.82 ms", "14.99 ms"],
          ["2", "1,147.53", "1,134.44", "844.71 ms", "872.49 ms", "14.99 ms"],
          ["3", "1,145.32", "1,136.31", "845.16 ms", "871.71 ms", "14.99 ms"],
          ["4", "1,146.00", "1,136.17", "842.83 ms", "871.19 ms", "14.99 ms"],
          ["5", "1,147.04", "1,136.50", "842.74 ms", "870.06 ms", "14.99 ms"],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning"]}
      />
      <Callout tone="warning" title="Current AITER is compatible but not promotion-ready">
        Current main dc4bdf1c was adapted with caller-owned MoE output,
        default-off stage1 scratch reuse and the missing N4480/N6016 BF16 rows.
        All 160 measured requests and focused correctness tests passed. The
        corrected run still misses the 0.5% throughput gate and adds 27 ms
        median TTFT, so production stays on b56d27be pending compact kernel
        attribution.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/CURRENT_AITER_SGLANG_C2_AB_2026-08-21.md">
          Full SGLang AITER A/B report
        </Link>
      </Callout>
      <Callout tone="warning" title="A16W4 caller fixed, but endpoint performance rejected">
        SGLang now uses the separated A16W4 weight/scale contract and fresh
        traces show gemm1/gemm2 A16W4 kernels plus the fused FP8 MLA cache
        write. The full GSM8K set scored 0.948 with one invalid response, which
        was accepted. Capacity remains 1,868,927 tokens, but the matched
        C2-C64 sweep regresses throughput by 3.13–9.86% and TPOT by
        2.61–10.54%. Retain A8W4 for production.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/A16W4_FP8_MLA_Q_CACHE_SMOKE_2026-08-13.md">
          Validation report
        </Link>
      </Callout>

      <H2>A8W4 + fused Q/KV prep + FP8 KV endpoint</H2>
      <Table
        striped
        headers={[
          "Concurrency",
          "TP",
          "TTT (tok/s)",
          "TTT per GPU",
          "Output throughput",
          "Median E2EL",
          "Median TTFT",
          "Median TPOT",
          "Median ITL",
        ]}
        rows={[
          ["2", "8", "1,013.37", "126.67", "112.60", "18,176.31 ms", "908.73 ms", "16.88 ms", "16.87 ms"],
          ["4", "8", "1,811.92", "226.49", "201.32", "20,279.11 ms", "1,749.26 ms", "18.30 ms", "17.89 ms"],
          ["8", "8", "2,980.01", "372.50", "331.11", "24,755.61 ms", "2,642.38 ms", "21.58 ms", "20.16 ms"],
          ["16", "8", "4,563.04", "570.38", "507.00", "32,365.70 ms", "4,655.95 ms", "27.09 ms", "23.54 ms"],
          ["32", "8", "6,356.34", "794.54", "706.26", "46,476.69 ms", "8,705.64 ms", "36.90 ms", "29.06 ms"],
          ["64", "8", "8,072.59", "1,009.07", "896.95", "73,067.48 ms", "17,417.53 ms", "54.58 ms", "38.55 ms"],
        ]}
        rowTone={["success", "success", "success", "success", "success", "success"]}
      />
      <Callout tone="success" title="Fusion profile improves the full endpoint sweep">
        All 1,008 measured requests succeeded. Fixed 8,192/1,024 workload,
        64 warmups per concurrency, eight measured requests per concurrency
        unit, seed 42 and disabled radix cache. Gains taper with concurrency but
        remain positive through C64; median TPOT also improves at every point.
        The comparison uses the prior same-machine production reproduction,
        not an interleaved paired A/B.
      </Callout>
      <Callout tone="success" title="Optional B2 profile raises fused C2 to 1,109.82 tok/s">
        Enabling MOE preroute FP8 and B2 fusions improves the same C2 profile
        from 1,013.37 to 1,109.82 tok/s (+9.52%). Median TPOT falls from 16.88
        to 15.34 ms (−9.12%). All 16 measured requests succeeded; B2 remains an
        optional batch-two specialization.
      </Callout>
      <Callout tone="info" title="AITER prefill + tuned Triton decode favors C16-C64">
        The mixed backend keeps 1,861,342 FP8 KV tokens and raises C64 to
        8,597.81 tok/s (+6.51%) while reducing median TTFT by 16.05%. C2
        throughput regresses 3.07%; optional B2 raises C2 from 982.24 to
        1,071.03 tok/s (+9.04%). This remains a high-concurrency profile. The
        compact trace confirms Opus gqa_d192_v128 prefill and two-stage tuned
        Triton decode.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/K3_AITER_PREFILL_TRITON_DECODE_2026-08-17.md">
          Full result and trace report
        </Link>
      </Callout>
      <Callout tone="success" title="BF16 Q fusion restores the optimized Q/cache boundary for Triton decode">
        Triton MLA decode now uses one AITER fused launch for BF16 Q
        materialization plus FP8 KV cache write. GSM8K 1319 is 0.951, the target
        Q CatArray is absent, and C64 improves to 8,656.21 tok/s (+0.68%). FP8
        Q/Q-PE was rejected at 0.929 accuracy and −1.19% throughput.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON_MLA_Q_CACHE_FUSION_2026-08-17.md">
          Fusion A/B report
        </Link>
      </Callout>
      <Callout tone="success" title="KDA/B2 and M4 policies are independently gated">
        The ATT-driven KDA kernel lowers a production-like 69-layer graph from
        9.20 to 8.38 µs/layer but remains opt-in because its isolated endpoint
        gain is only 0.19%. M2 and M4 now share the cooperative preactivated
        MoE producer: M2 improves 1.08% over its old tri/shared-down design,
        while M4 improves C4 throughput 3.08% and TPOT 3.33%. Composed with KDA,
        M2 reaches 1,111.97 tok/s and GSM8K 1319 is 0.951. The older separate
        M2 and raw M4 MoE paths were removed.{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/KDA_B2_M4_OPTIMIZATION_2026-08-18.md">
          Full KDA/B2/M4 handoff
        </Link>
      </Callout>

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
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_AB_2026-08-13.md">
          Detailed report
        </Link>
        {" "}·{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_37_C32_TRACE_2026-08-13.md">
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
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON36_37_STAGE_ANALYSIS_2026-08-13.md">
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
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/TRITON37_EXTEND_N32_RESULTS_2026-08-13.md">
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

      <H2>Fresh environment reproduction</H2>
      <Table
        striped
        headers={["Metric", "Recorded", "Reproduced", "Delta / note"]}
        rows={[
          ["Vendored FlyDSL tests", "46 passed", "46 passed", "Matched"],
          ["GSM8K 50", "1.000", "1.000", "Matched"],
          ["GSM8K 200", "0.990", "0.985", "One deterministic invalid response"],
          ["Max token capacity", "933,883", "934,463", "+0.06%"],
          ["Production C2", "968.57 tok/s", "980.24 tok/s", "+1.20% · TPOT 17.47 ms"],
          ["Production C4", "1,741.98 tok/s", "1,764.84 tok/s", "+1.31% · TPOT 18.92 ms"],
          ["Production C8", "2,881.25 tok/s", "2,921.82 tok/s", "+1.41% · TPOT 22.15 ms"],
          ["Production C16", "4,432.25 tok/s", "4,492.89 tok/s", "+1.37% · TPOT 27.71 ms"],
          ["Production C32", "6,191.41 tok/s", "6,295.63 tok/s", "+1.68% · TPOT 37.59 ms"],
          ["Production C64", "7,906.61 tok/s", "8,014.22 tok/s", "+1.36% · TPOT 55.58 ms"],
          ["B2 C2", "1,054.19 tok/s", "1,073.02 tok/s", "+1.79% vs record · +9.47% vs reproduced production"],
          ["B2 C4", "1,743.45 tok/s", "1,770.31 tok/s", "+1.54% vs record · +0.31% vs reproduced production"],
        ]}
        rowTone={[
          "success",
          "success",
          "warning",
          "success",
          "success",
          "success",
          "success",
          "success",
          "success",
          "success",
          "info",
          "info",
        ]}
      />
      <Text size="small" tone="tertiary">
        Source: crsuse2-m2m-002.crusoe.amd.com · aligned 8x MI355X rerun on
        2026-08-12 · fixed 8192/1024 endpoint, 64 warmups ·{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/ENVIRONMENT_REPRODUCTION_2026-08-12.md">
          full reproduction report
        </Link>
      </Text>

      <H2>ATOM recipe versus reproduced SGLang production</H2>
      <Table
        striped
        headers={["Concurrency", "ATOM tok/s", "SGLang tok/s", "ATOM delta", "ATOM median TPOT"]}
        rows={[
          ["C2", "799.99", "980.24", "-18.39%", "21.56 ms"],
          ["C4", "1,483.11", "1,764.84", "-15.96%", "22.42 ms"],
          ["C8", "2,521.14", "2,921.82", "-13.71%", "25.59 ms"],
          ["C16", "4,290.71", "4,492.89", "-4.50%", "28.54 ms"],
          ["C32", "6,117.82", "6,295.63", "-2.82%", "38.27 ms"],
          ["C64", "8,380.39", "8,014.22", "+4.57%", "52.66 ms"],
        ]}
        rowTone={["warning", "warning", "warning", "info", "info", "success"]}
      />
      <Text size="small" tone="tertiary">
        Source: crsuse2-m2m-002.crusoe.amd.com · ATOM f782218a5 · 8x MI355X
        TP8 · fixed 8192/1024 · 64 warmups · C2-C32 496/496 and C64 512/512
        requests passed · ATOM recipe uses FP8 KV and PTPC-FP8 online
        quantization ·{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/ATOM_KIMI_K3_PERFORMANCE_2026-08-12.md">
          full ATOM report
        </Link>
      </Text>

      <H2>Current ATOM C64 stream policy</H2>
      <Table
        striped
        headers={["Mode", "Total tok/s", "Median TPOT", "Delta versus multi-stream"]}
        rows={[
          ["Multi-stream MoE", "8,742.34", "49.84 ms", "Baseline"],
          ["Single-stream MoE", "8,399.50", "53.33 ms", "−3.92% throughput · +7.01% TPOT"],
        ]}
        rowTone={["success", "warning"]}
      />
      <Text size="small" tone="tertiary">
        Source: crsuse2-m2m-002.crusoe.amd.com · ATOM 5479c5af3 · AITER
        e8b4507e5 · 8x MI355X TP8 · fixed 8192/1024 · 64 warmups · 512/512
        requests passed in both modes · GPU kernel times retained as two
        eight-rank RTL full-mode trace sets · updated 2026-08-13 ·{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/C64_ATOM_STREAM_AB_2026-08-13.md">
          full stream A/B report
        </Link>
      </Text>

      <H2>C64 single-stream ATOM versus single-stream SGLang</H2>
      <Table
        striped
        headers={["Component", "Evidence", "Contribution / implication"]}
        rows={[
          ["Primary: ATOM single-stream", "8,399.50 tok/s · 53.33 ms TPOT", "+4.81% throughput · −4.04% TPOT vs SGLang"],
          ["SGLang", "8,014.22 tok/s · 55.58 ms TPOT", "Matched current single-stream baseline"],
          ["SGLang decode topology", "15.446 s main stream · 0.011 s secondary stream", "Effectively single-stream on MI355"],
          ["MLA Q materialization", "Fused op replaced 24 CatArray launches/step", "BF16 below gate; FP8 C32 +3.26% · C64 +3.57%"],
          ["Unisolated single-stream gap", "+385.28 tok/s", "FP8/BF16, MoE kernels, collectives and runtime remain mixed"],
          ["Separate: ATOM dual-stream", "8,742.34 tok/s · +342.84 over ATOM single", "Future optimization; excluded from primary comparison"],
        ]}
        rowTone={["success", "info", "warning", "success", "warning", "info"]}
      />
      <Text size="small" tone="tertiary">
        Trace structure is diagnostic only: ATOM RTL used Torch 2.13/ROCm 7.14,
        while SGLang PyTorch profiling used Torch 2.9/ROCm 7.2. Absolute kernel
        durations are not compared across profilers. Updated 2026-08-13 ·{" "}
        <Link href="file:///workspace/claude-skills/kimi-k3/aiter-optimization-tracker/C64_ATOM_SGLANG_TRACE_COMPARISON_2026-08-13.md">
          full trace and code comparison
        </Link>
      </Text>

      <H2>Feature policy</H2>
      <Table
        striped
        headers={["Feature", "Status", "Evidence"]}
        rows={[
          ["Fused KDA + f_b", "Production", "Kernel boundary active; correctness retained"],
          ["MoE caller-owned output", "Production", "Routed-output copy removed"],
          ["Stage1 scratch reuse", "Production manifest", "Capacity 933,883; graph memory reduced"],
          ["MLA gate + KDA group64", "Production manifest", "Focused tests and endpoint gates passed"],
          ["KDA ATT winner", "Optional off", "69-layer graph −8.9%; isolated C2 endpoint +0.19%"],
          ["KDA B2 group64", "Optional", "With unified M2 MoE + KDA winner: C2 1,111.97 tok/s; TPOT 15.33 ms"],
          ["M2/M4 cooperative preactivation", "Validated optional", "M2 +1.08%; M4 +3.08%; unified GSM8K 0.951"],
          ["FP8 latent tail", "Optional off", "C1 +2.39%; capacity −9.55%"],
          ["A4W4 profile", "C16 only", "C16 +1.50%; other points regress"],
          ["Radix-4 K3 TopK", "Validated optional", "C2 paired +2.34%; GSM8K 0.985; capacity 933,883"],
          ["Radix-4 + B2", "Exploratory optional", "Single-run C2 1,082.78 tok/s; repeat paired validation"],
          ["Triton 3.7 extend N32", "Validated performance-only", "C32 +5.38%; full endpoint matrix recovered; GSM8K pending"],
          ["A16W4 caller contract", "Rejected for performance", "Correctness restored; C2-C64 throughput −3.13% to −9.86% versus A8W4"],
          ["AITER prefill + tuned Triton decode", "Validated optional", "C64 +6.51%; TTFT −16.05%; C2 throughput −3.07%; capacity 1,861,342"],
          ["Triton MLA Q/cache fusion", "Validated optional", "BF16 Q + FP8 KV: GSM8K 0.951; C64 +0.68%; FP8 Q/Q-PE rejected"],
        ]}
        rowTone={["success", "success", "success", "success", "info", "info", "success", "warning", "warning", "success", "info", "info", "warning", "info", "success"]}
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
          ["KDA three-stage split-V", "15.93–16.45 µs vs 12.79 µs baseline; source removed", "Prepare/workspace/finalize launches are removed"],
          ["KDA async state-to-LDS", "12.99 µs; direct register path is faster", "gfx950 gains a lower-cost async state path"],
          ["A8W8 preroute hybrid", "32.10–36.20 µs vs 22–23 µs BF16", "Quantization and router are fused without extra launches"],
          ["M4 fused shared-down", "Micro win; C4 endpoint regressed; source removed", "Production-equivalent fused SiTU + BF16 baseline is beaten"],
          ["M8 cooperative preactivation", "Best 36.97 µs vs 32.91 µs production; allowance removed", "A non-tiled producer avoids both weight rereads and 114 KiB LDS pressure"],
        ]}
        rowTone={["warning", "warning", "warning", "warning", "warning", "warning", "warning", "warning", "warning", "warning", "warning"]}
      />

      <H2>Next work</H2>
      <Stack gap={7}>
        <Text>1. Reconcile the local #34490 exact-tie, NaN, flag and gfx guard fixes with upstream before default enablement.</Text>
        <Text>2. Complete GSM8K validation for the Triton 3.7 extend-attention N32 recovery.</Text>
        <Text>3. Decide whether the correctness-approved KDA/B2 and cooperative M4 profiles remain opt-in or become exact-bucket defaults.</Text>
        <Text>4. Continue route/sort/quant handoff work; it remains the largest serial module gap.</Text>
        <Text>5. Keep MLA Q/cache, KDA winner, B2 and the cooperative M4 policy independently gated and revertible.</Text>
        <Text>6. Track AITER #4617/#4647 merges and run five-round paired C2 before enabling B2 by default.</Text>
        <Text>7. Analyze B300 normal versus single-stream/no-PDL summaries.</Text>
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
