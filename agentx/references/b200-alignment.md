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

### 24.1 chat-template: unnecessary, NOT shown to be harmful — twice-corrected

**Read the correction at the end of this section before using anything in it.**
This section first argued for landing `--chat-template`, then blamed it for
collapsing generation. Both claims are wrong. Current standing rule: **do not
pass it** (it is unnecessary and every reference arm runs without it), but do
**not** cite it as a cause of any failure.

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

**What was measured (2026-08-28) — and why it does not mean what it says:**

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

**CORRECTION (2026-08-28, later the same day).** The "~1 output token per
request" figures above came from `sglang:generation_tokens_total` on a server
running `--tokenizer-worker-num 8`. That metric **under-reports across tokenizer
worker processes**: a later run reading "~1 tok/req" by the same method finished
with `request_metrics.tokens.output_actual.mean` = **965.3**. The next run after
the template was removed showed the identical "~1 tok/req" reading. **There was
probably never a generation collapse, and the template was never shown to break
anything.**

What survives: `--tool-call-parser deepseekv4` makes
`resolve_chat_encoding_spec()` return `"dsv4"` (`chat_encoding.py:112`), so
`encoding_dsv4.py` renders DSv4 prompts natively and `--chat-template` overrides
that. The startup line `No chat template found, defaulting to 'string' content
format` is misleading log noise, not evidence of a defect. So the flag is
**unnecessary** — that is why it stays off, matching every reference arm — but
"harmful" is unproven.

Two real failures in the 04:23 and 04:56 runs remain **unexplained**: warmup
324/707 at 2430 s (tbo: 701/707 at 1200 s) and
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, `Available Free mem : 318 MB`. Between
those and the first healthy run, the template was removed **and**
`--enable-prefill-delayer` + `--enable-two-batch-overlap` were added back. Two
variables moved together; neither is established. See SKILL.md CONTINUE HERE,
"RETRACTION OF A RETRACTION", and next-action 4 for the experiment that would
settle it.

**Standing rule: do not pass `--chat-template` to a DSv4 AgentX arm** (it is
redundant with the native encoder),  The flag
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
