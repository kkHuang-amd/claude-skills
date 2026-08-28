# MegaMoE on AgentX — MTPR vs chunked-prefill, and why it loses on this corpus

_Split out of `SKILL.md` on 2026-08-28. Section numbers are
preserved so every `§N` cross-reference in the skill still resolves._

## 13. MegaMoE: `SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR` vs `--chunked-prefill-size`

Only relevant if you enable the Aiter MegaMoEv2 path on this recipe
(`--moe-a2a-backend megamoe` + `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1`). That path
arrives with **sgl-project/sglang#35619, which is still open** — the env var
does not exist on stock sglang.

MTPR is the MegaMoEV2 `max_tok_per_rank` buffer capacity, and it is **validated
at startup against the chunked prefill budget**, not merely advisory.
`python/sglang/srt/arg_groups/deepseek_v4_hook.py`:

```
required_tokens_per_rank = ceil(local_chunked_prefill_size / token_alignment) * token_alignment
MTPR < required_tokens_per_rank  ->  ValueError, the server refuses to start
```

`local_chunked_prefill_size` depends on how tokens are partitioned:

| Mode | token_partition_size | local_chunked_prefill_size | token_alignment |
|---|---|---|---|
| pure TP / PP static chunking | 1 | `chunked_prefill_size` | `max(tp_size / attn_cp_size, 1)` |
| DP attention | `dp_size` | `chunked_prefill_size // dp_size` | as above |
| prefill CP (`--enable-prefill-cp`) | `attn_cp_size` | `ceil(chunked_prefill_size / attn_cp_size)` | 1 |

`--chunked-prefill-size=-1` is rejected outright: MegaMoE's per-rank token
requirement would have no strict prefill-forward bound.

### How this lines up with the AgentX dsv4 recipe

MTPR's default of 8192 matching the launcher's `CHUNKED_PREFILL_SIZE=8192` is
not a coincidence — it sits exactly on the boundary:

- `DP_ATTENTION=false` (the published arms): pure TP, so
  `local = chunked_prefill_size = 8192` = MTPR. Passes with nothing to spare.
- `DP_ATTENTION=true` (dormant branch): the launcher widens chunked prefill to
  `8192 * TP`, and `token_partition_size = dp_size = TP`, so
  `local = 8192 * TP / TP = 8192` again.

**Raising `--chunked-prefill-size` without raising MTPR is what breaks.** Under
pure TP, 16384 needs `required = 16384` and the default 8192 aborts startup.

### Two enforcement layers, different failure modes

Startup validation covers the **prefill** path only. At runtime:

- `should_use_mega_moe()` returns `max_tokens <= _mtpr()` — over the cap it
  **silently falls back** to the fused MoE path. No error, just no MegaMoE.
- `forward_mega_moe` asserts `x_in.shape[0] <= selected_mtpr`.
- With `SGLANG_AITER_MEGA_RANK_SYNC=1`, any batch with an extend sets
  `sync_tokens = mtpr` outright (`forward_batch.is_extend_in_batch`), so every
  prefill pads its all-to-all to the **full** MTPR. Oversizing MTPR is a real
  bandwidth cost, not a free safety margin.

Decode is not bounded by chunked prefill — it is governed by
`cuda_graph_max_bs` times the MTP draft-token count. The upstream assert in
`layers/moe/mega_moe.py` names both: *"raise the env var or shrink
cuda_graph_max_bs / chunked_prefill_size accordingly"*. A disaggregated decode
node skips the startup hook entirely.

**MTPR must be a power of two** (`_mtpr()` rejects it with
`mtpr & (mtpr - 1)`), so you cannot dial it to exactly `required` — round up to
the next power of two.

