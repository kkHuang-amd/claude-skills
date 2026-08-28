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

### 24.1 The chat-template gap is NOT a defect — RETRACTED 2026-08-28

**This section previously called for landing `--chat-template` as a correctness
fix. That was wrong, it was implemented, and it broke the arm. Do not redo it.**

The reasoning was: DeepSeek-V4-Pro ships no chat template (`tokenizer_config.json`
has no `chat_template`, no `.jinja` in the model dir), every arm logs

```
No chat template found, defaulting to 'string' content format
```

while aiperf posts to `/v1/chat/completions`, therefore prompts must be
role-marker-less strings with no trailing `<think>` and
`SGLANG_DEFAULT_THINKING=1` / `--reasoning-parser deepseek-v4` must never have
been exercised. **Every step of that is wrong except the log line itself.**

SGLang does not use a jinja template for this model at all. It uses a **native
DSv4 encoder**: `entrypoints/openai/chat_encoding.py:112`
`resolve_chat_encoding_spec()` returns `"dsv4"` when
`tool_call_parser == "deepseekv4"` — which every AgentX launcher passes — and
`entrypoints/openai/encoding_dsv4.py` then owns thinking (`<think>`/`</think>`),
tool calls, tool results, EOS and reasoning history. The function's own
docstring: *"None means the default path (HF chat template); any non-None spec
also owns reasoning-history rendering."* The "No chat template found" line is
**misleading log noise**, not evidence of a defect.

Passing `--chat-template` therefore does not add structure — it **overrides the
native encoder** with `chat_templates/deepseek_v4_thinking.jinja`, nine lines
that render system/user/assistant text and nothing else.

**Measured consequence (2026-08-28, b200align c64, killed after ~65 min):**

| arm | output tokens / request |
|---|---|
| `tbo-tp8-c64` (native encoder) | **956** |
| `armB-tp8-c64` (native encoder) | **919** |
| b200align **with** `--chat-template` | **~1** |

Live server `/metrics` during that run: `prompt_tokens_total` **62,704,990**
against `generation_tokens_total` **331**, over 329 returned requests, with
`num_running_reqs` 0.0 on all 8 DP ranks. Generation had collapsed. The
secondary symptoms were a 3-4x slower warmup (324/707 at 2430 s where tbo hit
701/707 at 1200 s) and a prefill-only profile (2147 prefill batches, 2 decode) —
both effects, not causes: with ~1 token generated per turn the replay still
feeds the whole history back, so prompts snowball to ~163k tokens/request.

**Standing rule: never pass `--chat-template` to a DSv4 AgentX arm.** The flag
has been removed from all four MI355X launchers. The B200 recipe passes it
(`dsv4_fp4_b200_sglang_mtp.sh:198`) and also passes `--tool-call-parser
deepseekv4`; whether B200 is silently degraded the same way, or its SGLang
resolves the spec differently, is **open and worth asking the B200 owners** —
but it is not a reason to copy the flag here.

**Method note.** Two claims in the retracted argument were checked and are still
true — the AgentX weka loader emits only `system`/`user`/`assistant` roles
(`weka_synth_buf.py:62`) and never sets `raw_tools`, so the template drops no
tool content. They were just the wrong question. The question that mattered was
*what renders the prompt when no template is passed*, and it went unasked. When
removing or adding a "missing" knob, identify the component that currently owns
the behaviour before concluding nothing owns it.
