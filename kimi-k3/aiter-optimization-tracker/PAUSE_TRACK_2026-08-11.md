# Kimi-K3 pause / continuation track — 2026-08-11

## Status

Optimization work is intentionally paused after the V4 persistent route-prep
gate failed. No V3-R/V4 experimental source remains and no new commit was
created.

The subsequent B300-80% campaign stopped at its Phase-A gate, but the useful
subset was restored as an opt-in B2-only profile. Paired C2 improved 8.70% and
C4 stayed flat after B4 dispatch was removed. See
[`B2_FUSION_SOLIDIFICATION_2026-08-11.md`](B2_FUSION_SOLIDIFICATION_2026-08-11.md).

Resume from:

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/
aiter-optimization-tracker/HANDOFF_2026-08-11.md
```

## New B300 artifacts

Two matched no-radix-cache cases were added:

```text
/sgl-workspace/b300-kimi-k3-traces-c2-c32-two-cases/
  no-radix-cache/
  no-radix-cache-single-stream-no-pdl/
```

Each case retains:

```text
server.log
c2/result.jsonl
c2/client.log
c32/result.jsonl
c32/client.log
```

The server logs record TP8 EXTEND and DECODE profile captures. They reference
the original profile output directories, but the visible `/sgl-workspace`
snapshot currently contains result/client/server logs rather than the profile
subdirectories themselves.

Both result files explicitly report:

```text
disable_radix_cache = true
8192 input / 1024 output
TP8, no DCP
moe_runner_backend = flashinfer_mxfp4
attention_backend = trtllm_mla
```

Therefore this two-case comparison isolates normal overlap/PDL versus
single-stream/no-PDL. It does not isolate radix-cache on versus off.

## Endpoint evidence

The supplied B300 performance sheet compares:

```text
A = non-Dspark, no radix cache
B = non-Dspark, no radix cache, single stream, no PDL
```

```text
conc  output tok/s A  output tok/s B  B delta   TPOT A  TPOT B  B delta
2          200.61          177.58     -11.5%      9.32   10.59   +13.6%
4          339.25          305.29     -10.0%     10.46   11.85   +13.3%
8          516.32          475.79      -7.9%     13.40   14.61    +9.0%
16         714.11          673.60      -5.7%     18.96   20.15    +6.3%
32         958.35          904.06      -5.7%     27.47   29.21    +6.3%
64        1204.66         1146.72      -4.8%     40.06   42.10    +5.1%
```

The trace-run JSON endpoints show the same direction:

```text
C2:
  output throughput 186.46 -> 167.24 tok/s
  median TPOT          9.33 ->  10.59 ms

C32:
  output throughput 913.20 -> 867.88 tok/s
  median TPOT         27.47 ->  29.21 ms
```

Conclusion: B300 multi-stream scheduling and PDL are material. Their advantage
is largest at low concurrency but remains about 5-6% at C32-C64. This supports
the prior trace attribution that overlap explains a substantial part of the
B300 versus MI355X decode gap.

The separate observation that radix-cache on/off has little effect should be
retained as a working conclusion, but it is not proven by these two cases
because both disable radix cache. A future cache-specific claim needs a matched
run where stream/PDL settings are held constant.

## Resume options

Do not restart V3-R or multi-CU persistent V4.

If route work resumes, choose one of:

1. one-CTA LDS E896 sorter for M<=32; or
2. stage1 ABI change consuming route metadata/token-major scale directly.

Before kernel work, analyze the retained B300 single-stream/no-PDL traces
against the normal no-cache trace to quantify:

```text
busy union
summed kernel duration
lost overlap by category
PDL versus cross-stream contribution
```

Performance sheet image retained at:

```text
/root/.cursor/projects/sgl-workspace/assets/c__Users_wunhuang_AppData_Roaming_Cursor_User_workspaceStorage_72c0c5a0c1eb00313cfa49faf81a98bd_images_image-addc5063-15e7-4838-9163-14ec435980b6.png
```
