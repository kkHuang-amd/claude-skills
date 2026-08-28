# B200 recipe flags we do not set, incl. the missing chat template

_Split out of `SKILL.md` on 2026-08-28. Section numbers are
preserved so every `§N` cross-reference in the skill still resolves._

## 24. B200 recipe flags we do not set (2026-08-28)

Audit of `dsv4_fp4_b200_sglang_mtp.sh` against
`dsv4_fp4_mi355x_sglang_{,tbo_}mtp.sh`. **Every flag below exists in our
SGLang** — none is blocked by the platform.

| flag | B200 | ours | touches |
|---|---|---|---|
| `--chat-template` | `deepseek_v4_thinking.jinja` | **not passed** | prompt formatting — see §24.1 |
| `--tokenizer-worker-num` | `$TP` = 8 | 1 | front-end tokenization parallelism |
| `--stream-interval` | 20 | 1 | streamed-token flush frequency |
| `--incremental-streaming-output` | on | off | per-chunk streaming payload |
| `--prefill-decode-interval` | 10 | 0 | scheduler prefill/decode interleave |
| `--enable-dp-attention-local-control-broadcast` | on | off | DP control plane (DP arms only) |
| `--enable-deepseek-v4-fp4-indexer` | on | off | indexer precision — **not NV-only**, read by `deepseek_v4_backend_hip_radix.py:459`, which is what `--attention-backend dsv4` resolves to on ROCm |
| `--weight-loader-prefetch-checkpoints` | on | off | startup only, not perf |

**Correctly different, do not align:** `--moe-runner-backend flashinfer_mxfp4`
and `--disable-flashinfer-autotune` (NVIDIA-only); `--moe-a2a-backend megamoe`
(set by `agentx_megamoe.sh`, deliberately absent from the ep1 TP8 arm);
`--dist-init-addr` / `--endpoint` / `--log` / `--pid` (harness plumbing).
Ours that B200 lacks are equally legitimate: `--attention-backend dsv4`,
`--kv-cache-dtype fp8_e4m3`, `--page-size 256`, `--cuda-graph-max-bs`,
`--enable-metrics`.

### 24.1 We have been running AgentX with NO chat template

**DeepSeek-V4-Pro ships no chat template**: `tokenizer_config.json` has no
`chat_template` key and the model dir has no `.jinja`. Every arm on file logged

```
No HuggingFace chat template found
No chat template found, defaulting to 'string' content format
```

while aiperf posts to `/v1/chat/completions` with `--endpoint-type chat`. So
our runs flattened the message list to a string with **no role markers at
all**. B200 renders:

```
<｜begin▁of▁sentence｜>{sys}<｜User｜>{q}<｜Assistant｜>{a}<｜end▁of▁sentence｜><｜User｜>{q}<｜Assistant｜><think>
```

On a 4-message sample that is +7 tokens, so ISL barely moves — **the length is
not the issue, the missing structure is.** In particular the trailing `<think>`
is absent from our prompts, so `SGLANG_DEFAULT_THINKING=1` and
`--reasoning-parser deepseek-v4` may never have been exercised as intended.

This is a **correctness/comparability defect, not a tuning knob**, and it
changes the workload. Land `--chat-template` *before* any further A/B, or every
baseline has to be re-measured afterwards. It also means our published-arm
reproductions (§12, §19) matched the leaderboard *without* the template the
B200 recipe uses — which is worth understanding before claiming parity.

**Objection raised and closed (2026-08-28).** The b200align launcher argues in
a comment that the template must stay off because `deepseek_v4_thinking.jinja`
handles only system/user/assistant and would silently drop tool definitions and
`role: tool` messages, truncating prompts and distorting ISL. Checked against
the loader: **the AgentX payload contains neither.** Segment roles are
`Literal["system", "user", "assistant"]`
(`utils/aiperf/src/aiperf/dataset/loader/weka_synth_buf.py:62`); `raw_tools` is
never set by the weka path (0 occurrences in `weka_trace.py` /
`weka_trace_models.py`); and `endpoints/openai_chat.py:56` adds a `tools` key
only `if raw_tools is not None`. Tool definitions are pre-folded into the
**system** segment as synthetic tokens (`weka_synth_buf.py:156`), so they reach
the model as ordinary system text that the template renders correctly. The
three-role template is exactly complete for this workload. If a future arm
switches to a loader that does populate `raw_tools` (e.g. `exgentic.py`,
`mooncake_trace.py`, `dag_jsonl.py`, `sagemaker_data_capture.py` — all four do),
this objection becomes live again and the template needs a tools/tool-role
branch first.

