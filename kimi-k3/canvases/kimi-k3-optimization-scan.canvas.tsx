import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Link,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const pr = (number: number) => (
  <Link href={`https://github.com/ROCm/aiter/pull/${number}`}>#{number}</Link>
);

const sglPr = (number: number) => (
  <Link href={`https://github.com/sgl-project/sglang/pull/${number}`}>#{number}</Link>
);

export default function KimiK3OptimizationScan() {
  return (
    <Stack gap={18} style={{ padding: 20, maxWidth: 1500, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>Kimi-K3 optimization scan</H1>
        <Text tone="secondary">
          46 tracked PRs: 43 direct AITER matches, 2 KDA dependencies, and 1
          SGLang integration candidate · canonical PR inventory, integration
          state, and measured local effects.
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="46" label="Tracked PRs (43 direct + 3 related)" />
        <Stat value="32" label="Open PRs" tone="info" />
        <Stat value="9" label="Merged PRs in search set" tone="success" />
        <Stat value="3" label="Closed without merge" tone="warning" />
      </Grid>

      <Grid columns={4} gap={14}>
        <Stat value="6" label="Kimi kernel families vendored in SGLang" tone="success" />
        <Stat value="2" label="Hard AITER core dependencies" tone="info" />
        <Stat value="5" label="Optional profiles / flags" tone="warning" />
        <Stat value="23" label="Open PRs still unintegrated" tone="info" />
      </Grid>

      <Callout tone="info" title="Current execution state">
        Kimi-specific gfx950 FlyDSL kernels are now maintained in SGLang
        commit 13e6937. The AITER core-only branch retains #4617 and #4647.
        Vendored validation passed GSM8K 200 at 0.990, matched C2-C32 within
        0.25%, and preserved capacity at 933,883 tokens. #4503/#4504 and the
        B2 path remain independent optional flags; #4603 remains C16-only.
        SGLang #34490 is validated default-off with local tie/NaN fixes.
      </Callout>

      <H2>Local integration ledger</H2>
      <Table
        stickyHeader
        striped
        headers={["PR", "Feature", "Local commits", "Measured result", "Production status"]}
        rows={[
          [sglPr(34490), "Radix-4 E896 top-16 router", "experiment/pr34490-radix4", "45 tests; paired C2 +2.34%; C4-C32 +0.68–2.05%", "Validated optional"],
          [pr(4495), "Fused KDA decode + f_b", "SGLang 13e6937", "Vendored kernel tests and endpoint validation passed", "Production manifest"],
          [pr(4617), "Caller-owned fused_moe output", "AITER integration/k3-core-only", "Removed routed-output copy; strict alias contract passed", "Core dependency"],
          [pr(4647), "Reusable MoE stage1 scratch", "AITER integration/k3-core-only", "Capacity 933,883; graph memory reuse retained", "Core dependency"],
          [pr(4497), "Fused MLA output gate", "SGLang 13e6937", "Vendored focused tests passed", "Production manifest"],
          [pr(4499), "KDA group64 input projection", "SGLang 13e6937", "B1/B2 focused tests passed", "Production manifest"],
          [pr(4504), "FP8 MoE pre-route/shared-down", "SGLang 13e6937", "B2 C2 +8.8%; C4 flat; C1 capacity cost remains", "Optional B2/C1"],
          [pr(4503), "FP8 latent-MoE tail", "SGLang 13e6937", "C1 +2.39%, token capacity −9.55%", "Optional off"],
        ]}
        rowTone={["success", "success", "success", "success", "success", "success", "warning", "warning"]}
      />

      <Callout tone="success" title="Final selected-stack validation">
        The SGLang-vendored/core-only-AITER stack scored GSM8K 200 at 0.990.
        C2/4/8/16/32 measured 968.57 / 1741.98 / 2881.25 / 4432.25 /
        6191.41 tok/s, all within 0.25% of golden. The optional B2 profile
        reached 1054.19 tok/s at C2 and stayed flat at C4.
      </Callout>
      <Callout tone="success" title="Radix-4 validation">
        The default-off #34490 profile passed 45 focused tests and exact AITER
        tie/NaN contracts. Five paired C2 rounds improved throughput
        970.38→993.11 tok/s (+2.34%); C4/C8/C16/C32 improved
        +2.05/+1.71/+1.34/+0.68%. GSM8K 200 was 0.985 and capacity remained
        933,883.
      </Callout>

      <H2>Production trace refresh · 2026-08-11</H2>
      <Table
        striped
        headers={["Rank", "Bottleneck", "Trace evidence", "Decision / next action"]}
        rows={[
          ["1", "TP8 communication", "8–9% of decode kernel time; 22% of prefill", "Dispatch-verified B32: 1-stage 21.47 µs vs 2-stage 12.44 µs; exact C32 −1.66%. B6/B16 are separate shape-specific candidates."],
          ["2", "MoE route preparation and stages", "Route/top-k/quant-sort 10–14% at C2; stage1+stage2 about 30% at C32", "B2-only profile remains +8.70%. Generic TILE_M: M4 shared+tri saved only 2.77 µs/layer vs 10 µs gate; M8/M16 regressed, so experiment was removed."],
          ["3", "Attention-residual aggregate", "6–8% of decode and about 8% of prefill", "#4572 fresh-cache C1 endpoint was flat; standalone replacement rejected."],
        ]}
        rowTone={["warning", "info", "neutral"]}
      />
      <Callout tone="info" title="Trace coverage">
        TP8 trace summaries cover prefill C8, decode C2, and decode C32. Raw
        trace files were removed after summary JSON/CSV and decision documents
        were retained. The selected KDA, MLA, MoE, and zero-copy paths remained
        active with no routed [M,3584] copy regression.
      </Callout>
      <Callout tone="warning" title="All-reduce microbench reconciliation">
        The preliminary B32 gain reused a stale JIT module and did not switch
        stages. Fresh-JIT dispatch logs, five paired TP8 rounds, and three
        production traces agree: one-stage is about 72% slower at B32. HIP
        model alt streams are disabled and measured same-rank kernel overlap
        is below 0.1%; neither overlap nor rank skew caused the endpoint loss.
      </Callout>

      <H2>Highest-impact open optimization PRs</H2>
      <Table
        stickyHeader
        striped
        headers={["PR", "Area", "Target", "Reported impact", "Local disposition", "Branch"]}
        rows={[
          [sglPr(34490), "Radix-4 K3 TopK router", "MI355X · E896 top-16 decode", "Measured 4.2-5.2 µs/layer saved at M1-M64", "Validated default-off; local exact-tie, NaN and gfx guard fixes required", "experiment/pr34490-radix4"],
          [pr(4507), "MLA split sizing from page table", "gfx950 · long context", "Up to 4.80× E2E at 327K context", "Deferred: Triton 3.7 isolated env", "fix/mla-gluon-splitkv-sizing-from-page-table"],
          [pr(4450), "12-head MLA split scheduling", "gfx950 · TP8", "TPOT up to 3.78× at c1 / 100K", "Deferred: Triton 3.7 isolated env", "perf/mla-gluon-h12-split-tuning"],
          [pr(4509), "Split-major MLA grid + blocked reduce", "gfx950 · small nhead", "+10.86% median throughput on top of #4507", "Deferred: depends on #4507/Triton 3.7", "perf/mla-gluon-split-major-grid-blocked-reduce"],
          [pr(4487), "SiTUv2 MoE block_m=64", "gfx950 · DSpark verify", "+18% output tok/s", "Pending: only when DSpark is enabled", "tune/kimik3-moe-verify-block-m"],
          [pr(4603), "A4W4 MoE v1/v2 retune", "gfx950", "c16 203→237 tok/s; graph memory 36.6→0.59 GiB", "Integrated as 7f7693c1f; opt-in C16 profile, A8W4 remains default", "tune/kimi-k3-a4w4-v1-v2"],
          [pr(4480), "FP8 KV small-head MLA", "gfx950", "Capacity +45%; kernel up to 1.87×", "Deferred: Triton 3.7 isolated env", "feat/mla-gluon-fp8-kv-serving"],
          [pr(4504), "Fused FP8 pre-route projections", "gfx950 · batch 1", "Dual 1.28× / tri 1.90× kernel", "Fresh-JIT C1 +9.59%; optional profile, −13.44% token capacity", "perf/kimi-k3-preroute-fp8-tri-upstream-20260801"],
          [pr(4497), "Fused MLA output gate", "gfx950 · batch 1", "12.76→5.42 µs (2.35×)", "Integrated; enabled (C4 +0.37%)", "perf/kimi-k3-mla-gate-clean"],
          [pr(4503), "FP8 latent-MoE tail", "gfx950 · batch 1", "13.79→7.14 µs (1.93×)", "Fresh-JIT C1 +2.39%, but −9.55% token capacity; opt-in off", "perf/kimi-k3-latent-tail-fp8-upstream-20260801"],
          [pr(4499), "KDA group64 projection", "gfx950 · FP8", "15.10→9.24 µs (1.63×)", "Integrated; enabled (incremental C4 +0.20%)", "perf/kimi-k3-kda-group64-dco-clean"],
          [pr(4645), "FP8 D192/V128 prefill MHA", "gfx942", "1.11–1.18× vs BF16 ASM", "Deferred: gfx942 hardware", "maeehart/kimi-k3-gfx942-fp8-prefill"],
          [pr(4607), "Fused A4W4 stage1 quantization", "gfx1250", "3.2–35.1% MoE speedup by M", "Deferred: gfx1250 hardware", "perf/gfx1250-fuse-a4w4-quant"],
          [pr(4647), "Reusable MoE stage1 scratch", "graph capture", "Saves about 6.7 GiB/GPU", "Integrated; enabled (−5.69 GB/GPU)", "xiaohuguo/pr-f-moe-stage1-workspace"],
        ]}
        rowTone={["info", "info", "info", "info", "neutral", "neutral", "info", "warning", "success", "warning", "success", "info", "info", "success"]}
      />

      <H2>Other open K3 work</H2>
      <Table
        striped
        headers={["PR", "Type", "What it changes", "Upstream note", "Local disposition"]}
        rows={[
          [pr(4479), "Perf", "gfx950 prefill BF16 GEMM tuning", "Up to 9.46% kernel gain", "Reviewed; adapted to 9 observed M=8192 shapes — six kernel wins, but endpoint C8/C16/C32 was flat; rejected"],
          [pr(4495), "Perf", "Fuse KDA decode + f_b projection", "1.23× kernel", "Integrated and enabled"],
          [pr(4496), "Perf", "Fuse BF16 latent-MoE tail", "1.37× kernel", "Reviewed; skip — benchmark lacks the production all-reduce + RMSNorm fusion, and FP8 #4503 had no E2E gain"],
          [pr(4498), "Perf", "Fuse BF16 pre-route projections", "Supersedes #4442", "Reviewed; skip — production already uses a three-way fused front; FP8 tri-projection #4504 had no E2E gain"],
          [pr(4510), "Perf/fix", "Mixed-MoE stage2 b_nt + retune", "Mean +1.8% over 588 rows", "Reviewed; skip — current K3 A8W4 has no common FlyDSL _bnt2 rows; newer layout/Opus tuning supersedes its configs"],
          [pr(4572), "Perf", "Triton attn_res_gate fusion", "WIP; no benchmark posted", "Fresh-cache C1 endpoint flat (60.63 tok/s, TPOT 15.99 ms); rejected"],
          [pr(4582), "Perf", "gfx942 CDNA3 12-head MLA decode", "1.58× kernel", "Deferred: gfx942 hardware"],
          [pr(4617), "Perf/API", "Caller-provided fused_moe output buffer", "Removes per-layer D2D copy", "Integrated and enabled"],
          [pr(4471), "Enablement", "gfx942 packed-int4 SiTUv2 epilogue", "Depends on #4463 plumbing", "Deferred: gfx942 hardware"],
          [pr(4494), "Correctness/perf", "Graph-safe ASM split-K semaphore", "Unblocks DSpark graph capture", "Pending if split-K graph issue reproduces"],
          [pr(4537), "Correctness/tune", "gfx1250 GEMM fixes + K3 a16w16 configs", "Pairs with merged #4552", "Deferred: gfx1250 hardware"],
          [pr(4577), "Enablement", "KDA per-channel decay gate in GDR decode", "gfx942/gfx950", "Reviewed; 1.25–1.45× over standalone Triton kernel, but C2/C4 regressed 3.49%/2.54% versus fully-fused production AITER; rejected"],
          [pr(4622), "Correctness/perf", "FlyDSL split-K workspace + reduce", "Graph-safe replacement for atomics", "Pending if split-K is selected"],
          [pr(4625), "Enablement", "96-head × 128-dim MLA reduction", "Needed for PP8 K3", "Deferred: PP8-specific"],
        ]}
        rowTone={["neutral", "success", "neutral", "neutral", "neutral", "neutral", "info", "success", "info", "neutral", "info", "neutral", "neutral", "info"]}
      />

      <H2>What is already on origin/main</H2>
      <Table
        striped
        headers={["Merged PR / commit", "Landing", "K3 contribution"]}
        rows={[
          [pr(4397), "313502261 · Jul 28", "Foundational SiTUv2 2-stage MoE, strided grouped-topk, K3 configs"],
          [pr(4435), "b67ea2db8 · Aug 1", "A8W8 bpreshuffle PTPC GEMM tuned configs"],
          [pr(4463), "4045b2de7 · Aug 3", "Opt-in A4W4 SiTUv2 path + tuner correctness fixes"],
          [pr(4482), "e99784470 · Aug 4", "gfx1250 >512-expert scan + real SiTUv2 epilogue"],
          [pr(4534), "6dc26b7a8 · Aug 5", "Opus A8W4 SiTUv2 dispatch and tuning"],
          [pr(4552), "0874216b4 · Aug 5", "gfx1250 missing K3 fused BF16 GEMM routed to Triton"],
          [pr(4586), "7c5e20170 · Aug 10", "Opus/FlyDSL sorted intermediate layout; +3.0–7.7% tuned rows"],
          [pr(4474), "110055925 · Aug 4", "Fix >2 GiB MLA KV offset overflow"],
          [pr(4502), "868ac1f7a · Aug 8", "Faster A16W4 SiTUv2 kernel + K3 FP4 retuning"],
        ]}
        rowTone={["success", "success", "success", "success", "success", "success", "success", "neutral", "success"]}
      />

      <H2>Pending queue · 23 open PRs not integrated locally</H2>
      <Table
        striped
        headers={["Bucket", "Count", "PRs", "Next-action rule"]}
        rows={[
          [
            "Current-runtime backlog",
            "7",
            <Text>{pr(4405)}, {pr(4487)}, {pr(4488)}, {pr(4494)}, {pr(4526)}, {pr(4566)}, {pr(4622)}</Text>,
            "Prioritize only with a reproduced bottleneck or active workload",
          ],
          [
            "Reviewed against current production stack",
            "6",
            <Text>{pr(4479)}, {pr(4496)}, {pr(4498)}, {pr(4510)}, {pr(4572)}, {pr(4577)}</Text>,
            "Skip unless the runtime architecture or selected kernels change",
          ],
          [
            "Triton 3.7 isolated experiment",
            "4",
            <Text>{pr(4450)}, {pr(4480)}, {pr(4507)}, {pr(4509)}</Text>,
            "Do not change the validated Triton 3.6 container",
          ],
          [
            "Other hardware / PP-specific",
            "6",
            <Text>{pr(4471)}, {pr(4537)}, {pr(4582)}, {pr(4607)}, {pr(4625)}, {pr(4645)}</Text>,
            "Validate on gfx942/gfx1250 or PP8 hardware configuration",
          ],
        ]}
        rowTone={["neutral", "warning", "info", "info"]}
      />

      <H2>Related KDA dependencies · outside the original 43 matches</H2>
      <Table
        striped
        headers={["PR", "Role for Kimi-K3", "Upstream result", "Local disposition"]}
        rows={[
          [
            pr(4568),
            "Introduces chunk_delta_attn as a first-class Triton operator for KDA prefill",
            "Correctness passed on gfx950/gfx1250; KDA E2E used Triton 3.7",
            "Deferred: requires isolated Triton 3.7 environment",
          ],
          [
            pr(4602),
            "Optimizes #4568 with MI350X Gluon stages and cache-tuned Triton dispatch",
            "41 tests passed; DA_USE_GLUON / DA_TRITON_OPT graceful fallback",
            "Deferred: depends on #4568 and Triton 3.7 validation",
          ],
        ]}
        rowTone={["info", "info"]}
      />
      <Text tone="secondary">
        These PRs were missed by the original keyword query because their
        titles describe generic chunk delta attention rather than Kimi-K3.
        They are tracked here but do not alter the direct-match counts above.
      </Text>

      <Grid columns="1fr 1fr" gap={16}>
        <Card>
          <CardHeader trailing="not on main">Branch-only optimization tracks</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Text weight="semibold">tune/kimi-k3-shared-gemm-nosplitk</Text> — non-split-K shared BF16 GEMM tuning.</Text>
              <Text><Text weight="semibold">tune/kimi-k3-a4w4-v1-v2</Text> — A4W4 MoE and projection retunes; maps to #4603.</Text>
              <Text><Text weight="semibold">zejun/customer_k3</Text> — A4W4 projection/MoE tuning plus MLA msk0 blobs.</Text>
              <Text><Text weight="semibold">zejun/k3-mmd-mla-bf16-non-mask-20260806</Text> — non-causal MLA ASM kernels.</Text>
              <Text><Text weight="semibold">waqahmed/KIMI-K3_KDA-Decode</Text> — per-channel KDA decode tuning; maps to #4577.</Text>
              <Text><Text weight="semibold">k3-for-amd</Text> — original integration branch; core work landed through #4397 and follow-ons.</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing="triage">Closed or superseded</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>{pr(4442)} — early BF16 pre-route fusion; superseded by #4498.</Text>
              <Text>{pr(4533)} — A8W4 route reduction, closed unmerged despite reported 1.10× full-MoE gain.</Text>
              <Text>{pr(4574)} — no-split-K shared GEMM retune; closed after 2.9–4.1% graph replay regression.</Text>
              <Text>{pr(4405)} — still open, but its body says #4450 may subsume it.</Text>
              <Text>{pr(4526)} — multi-model MXFP4 infrastructure; K3 is incidental rather than the measured target.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="Tracking caveats">
        The 43 direct AITER PRs come from the 2026-08-10 Kimi-K3 text-search
        snapshot at origin/main 7c5e20170; AITER #4568/#4602 and SGLang #34490
        were added manually as functional dependencies or candidates. Local
        integration status is newer and comes from the integration branches
        plus measured MI355X results. Upstream PR states can change, and
        PR-author speedups are not directly interchangeable with local endpoint
        measurements.
      </Callout>

      <Text tone="tertiary" size="small">
        Sources: public GitHub PR API, git history/diffs, local integration
        commits, and MI355X benchmark artifacts · canonical status updated
        2026-08-12.
      </Text>
    </Stack>
  );
}
