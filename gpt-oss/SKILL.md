---
name: gpt-oss-sglang-rocm
description: >-
  Serve gpt-oss-120b with SGLang on AMD ROCm (MI355X / gfx950) using the AITER
  attention backend. Use when launching gpt-oss, when enabling the SHUFFLE 5D KV
  layout (SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d) or fp8 KV cache, when
  decode accuracy or stability regresses on the Plan A path, or when comparing
  AITER pa_decode_gluon against SGLang's legacy unified_attention.
---

# gpt-oss on SGLang + AITER (ROCm)

Entry point for this directory. Two documents, read the one you need — do not
read both up front.

| Your task | Start at | Size |
|---|---|---|
| Hitting a correctness / stability / perf problem on the Plan A SHUFFLE 5D path | `KNOWN_ISSUES.md` | 21 KB |
| Understanding why `pa_decode_gluon` is faster, or porting/tuning it | `aiter-pa-decode-gluon-design/SKILL.md` | 11 KB |

`KNOWN_ISSUES.md` sections: `Correctness`, `Stability`, `Performance (TP=1)`,
`Sweep data`. Jump to one with:

```bash
rg -n '^## ' KNOWN_ISSUES.md
sed -n 'A,Bp' KNOWN_ISSUES.md
```

## Typical launch shape

```bash
SGLANG_AITER_KV_CACHE_LAYOUT="vectorized_5d" python3 -m sglang.launch_server \
  --model-path <gpt-oss-120b> --tp 1 --trust-remote-code \
  --prefill-attention-backend aiter --decode-attention-backend aiter \
  --page-size 64 --disable-radix-cache --port 8000
```

Wait for `The server is fired up and ready to roll!` before sending requests —
`Application startup complete` is too early.

## Rules

Apply `../DEV_RULES.md`-style discipline here too: keep the full server log in a
file and grep it, never paste startup output into the conversation. One
`server_args=ServerArgs(...)` line alone is ~17,000 chars (~4k tokens).

```bash
python3 -m sglang.launch_server ... > /tmp/server.log 2>&1 &
rg 'ready to roll|Traceback|Error|max_total_num_tokens' /tmp/server.log | cut -c1-200
```
