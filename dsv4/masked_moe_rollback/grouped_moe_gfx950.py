# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 masked (deep_gemm-style) grouped MoE stage1 entry.

Milestone-1 of the mori-EP decode padded-M fix (see
``MORI_EP_DECODE_ROOTCAUSE.md`` §5z-b/§5z-c/§5z-d): drive the
``grouped_masked_m`` path of ``compile_mixed_moe_gemm1`` from host side. The
input/output are expert-major ``[E, max_m, *]`` and the per-expert live token
count ``masked_m[E]`` selects how many rows each expert actually computes, so
decode compute scales with the *real* per-expert tokens instead of the padded
recv buffer.

This is deliberately a thin, standalone launcher (not wired into
``fused_moe``) so the new kernel can be validated in isolation against
``torch_moe_stage1``.
"""

from typing import Optional

import torch

from .moe_kernels import (
    _run_compiled,
    _s1_args_fp4,
    _s2_args_fp4,
    _get_dtypes,
    runtime_swiglu_limit,
)


_MXFP4_LUT: Optional[torch.Tensor] = None


def _mxfp4_dequant_bf16_graphsafe(
    x_fp4x2: torch.Tensor, scale_e8m0: torch.Tensor, M: int, model_dim: int, blk: int = 32
) -> torch.Tensor:
    """Graph-safe mxfp4 (fp4x2 + per-1x32 e8m0) -> bf16 dequant.

    Avoids the host syncs in ``fp4_utils.mxfp4_to_f32`` (per-call
    ``torch.tensor(list, device=)``) and ``e8m0_to_f32`` (boolean-mask assign,
    which triggers ``nonzero()``). The 16-entry mxfp4 LUT is built ONCE and
    cached (during warmup, not capture); everything else is static-shape.
    """
    global _MXFP4_LUT
    dev = x_fp4x2.device
    if _MXFP4_LUT is None or _MXFP4_LUT.device != dev:
        _MXFP4_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=dev,
        )
    u8 = x_fp4x2.view(torch.uint8)
    lo = (u8 & 0xF).long()
    hi = (u8 >> 4).long()
    vals = torch.stack([_MXFP4_LUT[lo], _MXFP4_LUT[hi]], dim=-1).view(M, model_dim)
    # e8m0 -> f32 via bit ops + torch.where (no boolean-mask assignment).
    s_u8 = scale_e8m0.view(torch.uint8).to(torch.int32)
    s_bits = s_u8 << 23
    s_bits = torch.where(s_u8 == 0, torch.full_like(s_bits, 0x00400000), s_bits)
    s_bits = torch.where(s_u8 == 0xFF, torch.full_like(s_bits, 0x7F800001), s_bits)
    sf = s_bits.view(torch.float32).view(M, model_dim // blk, 1)
    return (vals.view(M, model_dim // blk, blk) * sf).view(M, model_dim).to(torch.bfloat16)


def _quant_per1x32_fp8(x: torch.Tensor):
    """(M, K) bf16 -> (fp8 M,K), (e8m0 M, K//32). Per-1x32 mxfp8 quant.

    Matches ``torch_moe_stage1``'s per_1x32 fp8 activation quant so the kernel
    (which reads the fp8 value and multiplies by the e8m0 block scale) and a
    dequant reference reconstruct the same value.
    """
    dtypes = _get_dtypes()
    block = 32
    dtype_max = 448.0
    M, K = x.shape
    flat = x.contiguous().float().view(M, K // block, block)
    max_abs = flat.abs().amax(dim=-1)  # [M, K/32]

    # per-1x32 mxfp8 e8m0 block scale, RoundUp == ceil_pow2(max_abs / 448).
    # Graph-safe reimplementation of fp4_utils.f32_to_mx_e8m0_scale +
    # e8m0_to_f32: both use boolean-mask assignment (nonzero) -> NOT
    # cuda-graph-capturable; here we use bit ops + torch.where only.
    xr = (max_abs / dtype_max).contiguous()
    u32 = xr.view(torch.int32)
    exp = (u32 >> 23) & 0xFF
    mant_nz = (u32 & 0x7FFFFF) != 0
    exp = torch.where(mant_nz & (exp < 0xFF), exp + 1, exp)  # ceil bump
    scale_e8m0 = exp.to(torch.uint8).view(dtypes.fp8_e8m0)

    # e8m0 -> f32 (graph-safe): value = 2^(exp-127) via exponent bits.
    sbits = exp << 23
    sbits = torch.where(exp == 0, torch.full_like(sbits, 0x00400000), sbits)
    sbits = torch.where(exp == 0xFF, torch.full_like(sbits, 0x7F800001), sbits)
    scale_f32 = sbits.view(torch.float32)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32 = torch.where(scale_f32 == 0, torch.ones_like(scale_f32), scale_f32)

    q = (flat / scale_f32.unsqueeze(-1)).clamp(min=-dtype_max, max=dtype_max)
    q = q.to(dtypes.fp8).view(M, K)
    return q, scale_e8m0.view(M, K // block)


def flydsl_masked_moe_gfx950(
    hidden_states: torch.Tensor,  # [T, K] bf16
    w1_shuf: torch.Tensor,  # [E, 2*inter, K/2] preshuffled fp4
    w2_shuf: torch.Tensor,  # [E, model_dim, inter/2] preshuffled fp4
    w1_scale_shuf: torch.Tensor,  # preshuffled e8m0 (E*2*inter, K//32)
    w2_scale_shuf: torch.Tensor,  # preshuffled e8m0 (E*model_dim, inter//32)
    topk_weight: torch.Tensor,  # [T, topk]
    topk_ids: torch.Tensor,  # [T, topk] int
    *,
    E: int,
    model_dim: int,
    inter_dim: int,
    max_m: int,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    gate_mode: str = "separated",
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    out: Optional[torch.Tensor] = None,
):
    """Full masked (deep_gemm-style) grouped MoE on gfx950, end-to-end.

    route -> scatter+per-1x32-fp8-quant into expert-major ``[E, max_m, K]`` ->
    masked stage1 -> per-1x32-fp8-quant -> masked stage2 (grouped
    ``[E, max_m, model_dim]``) -> weighted gather-reduce combine -> ``[T, model_dim]``.

    Weights/scales are expected pre-shuffled (``shuffle_weight(16,16)`` +
    ``e8m0_shuffle``), mirroring the standalone stage entries. This is the
    aiter-side entry the sglang mori-EP bridge calls with mori's grouped tokens
    and ``num_recv_tokens_per_expert`` as ``masked_m``.
    """
    from aiter.utility.fp4_utils import e8m0_shuffle
    from .grouped_moe_gfx1250 import _build_route_maps_naive, flydsl_moe_gather_reduce

    dtypes = _get_dtypes()
    dev = hidden_states.device
    T, K = hidden_states.shape
    topk = topk_ids.shape[1]
    scale_blk = 32

    # 1. route maps (arch-neutral torch): topids_to_rows[t,k] = expert*max_m + slot.
    topids_to_rows, rows_to_tokens, masked_m = _build_route_maps_naive(
        topk_ids.to(torch.int32), E, max_m
    )

    # 2. scatter hidden -> grouped [E, max_m, K] (gather by rows_to_tokens), then
    #    per-1x32 fp8 quant + e8m0 preshuffle for the A-scale.
    grouped_x = torch.zeros((E * max_m, K), dtype=hidden_states.dtype, device=dev)
    valid_rows = rows_to_tokens >= 0
    grouped_x[valid_rows] = hidden_states[rows_to_tokens[valid_rows].long()]
    a1_fp8, a1_scale = _quant_per1x32_fp8(grouped_x)
    a1_fp8 = a1_fp8.view(E, max_m, K)
    a1_scale_shuf = e8m0_shuffle(a1_scale)

    # 3. masked stage1 -> grouped [E, max_m, inter] bf16
    s1 = flydsl_masked_moe_stage1(
        a_grouped=a1_fp8,
        w1=w1_shuf,
        masked_m=masked_m,
        max_m=max_m,
        w1_scale=w1_scale_shuf,
        a1_scale=a1_scale_shuf,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        act=activation,
        gate_mode=gate_mode,
        swiglu_limit=swiglu_limit,
    )

    # 4. per-1x32 fp8 quant of the stage1 output
    a2_fp8, a2_scale = _quant_per1x32_fp8(s1.view(E * max_m, inter_dim))
    a2_fp8 = a2_fp8.view(E, max_m, inter_dim)
    a2_scale_shuf = e8m0_shuffle(a2_scale)

    # 5. masked stage2 -> grouped [E, max_m, model_dim] bf16
    s2 = flydsl_masked_moe_stage2(
        a_grouped=a2_fp8,
        w2=w2_shuf,
        masked_m=masked_m,
        max_m=max_m,
        model_dim=model_dim,
        w2_scale=w2_scale_shuf,
        a2_scale=a2_scale_shuf,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    # 6. weighted combine (un-permute): out[t] = sum_k w[t,k] * s2[topids_to_rows[t,k]]
    gather_w = topk_weight.to(s2.dtype).contiguous()
    out = flydsl_moe_gather_reduce(s2, topids_to_rows, gather_w, out=out)
    return out


def flydsl_masked_moe_gfx950_recv(
    recv_hidden: torch.Tensor,  # [M, K] bf16 (dispatch=bf16) OR fp8
    recv_scale: Optional[torch.Tensor],  # [M, K//32] e8m0 for fp8; None for bf16
    recv_topk_ids: torch.Tensor,  # [M, topk] GLOBAL expert ids (int32)
    recv_topk_weights: torch.Tensor,  # [M, topk] float32 topk weights
    w1_shuf: torch.Tensor,
    w2_shuf: torch.Tensor,
    w1_scale_shuf: torch.Tensor,
    w2_scale_shuf: torch.Tensor,
    *,
    E: int,  # num LOCAL experts
    expert_base: int,  # global id of local expert 0 (= ep_rank * E)
    total_recv: int,  # #valid recv rows: rows [0, total_recv) are real, rest padding
    model_dim: int,
    inter_dim: int,
    max_m: int,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    gate_mode: str = "separated",
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    out: Optional[torch.Tensor] = None,
):
    """Masked MoE on the mori/deepep **flat recv buffer** (the sglang bridge
    primitive).

    The mori NORMAL recv buffer is **token-major**: row m is a received token and
    ``recv_topk_ids[m]`` are its ``topk`` **GLOBAL** expert ids (see §6c). Validity
    is per-(row,slot) = "is this global expert local" (== ``expert_mask[topk_ids]``,
    the shipping ``get_topk_valid_mask``); padding rows are zero-inited (weight 0).
    NOT a ``[0,total_recv)`` prefix. This:
    1. maps global->local (``local = global - expert_base``; keeps only local slots),
    2. routes each (recv-row, local-slot) into expert-major ``[E, max_m, K]``,
    3. masked stage1 -> per-1x32 fp8 quant -> masked stage2 (grouped),
    4. **weighted** gather-reduce back to ``[M, model_dim]``:
       ``out[m] = sum_k w[m,k] * s2[row(m,k)]`` over the row's LOCAL experts.
    This matches the sorted ``fused_moe(mori)`` path (topk weight applied in the
    MoE, stage2 ``mulWeightStage2``; mori ``combine`` then just SUMS the per-rank
    partials -- ``_combine_kwargs`` passes no weights). Non-local / padding / cap-
    overflow slots contribute 0 (weight 0).

    Two activation modes: **bf16 recv** (``recv_scale=None``) does a single per-1x32
    fp8 quant (like DP); **fp8 recv** reuses the dispatched fp8 + e8m0 scale.

    ``max_m`` is the static per-expert row cap (bridge sets a decode per-expert bound).
    """
    import torch.nn.functional as F

    from aiter.utility.fp4_utils import e8m0_shuffle
    from .grouped_moe_gfx1250 import flydsl_moe_gather_reduce

    dev = recv_hidden.device
    M, K = recv_hidden.shape
    topk = recv_topk_ids.shape[1]
    scale_blk = 32

    # env-gated per-section timing (EAGER only: uses cuda sync). Prints avg ms
    # over the last _prof.N calls every N calls. SGLANG_MORI_MASKED_PROF=1.
    import os as _osp, time as _timep
    _prof_on = _osp.environ.get("SGLANG_MORI_MASKED_PROF", "0") in ("1", "true", "True")

    def _ck():
        if _prof_on:
            torch.cuda.synchronize()
            return _timep.perf_counter()
        return 0.0

    def _rec(acc, key, t0):
        if _prof_on:
            torch.cuda.synchronize()
            acc[key] = acc.get(key, 0.0) + (_timep.perf_counter() - t0)

    _pacc = flydsl_masked_moe_gfx950_recv.__dict__.setdefault("_pacc", {})
    _pn = flydsl_masked_moe_gfx950_recv.__dict__.get("_pn", 0) + 1
    flydsl_masked_moe_gfx950_recv._pn = _pn
    _t = _ck()
    is_fp8_recv = recv_scale is not None

    # --- global->local + per-(row,slot) routing (torch; eager) ---
    # A slot is valid iff (a) its GLOBAL expert is local (local in [0, E)) AND
    # (b) its recv row is REAL: row < total_recv. mori's valid recv tokens occupy
    # the PREFIX [0, total_recv) (= sum num_recv_tokens_per_expert); rows beyond
    # are PADDING that the default path zeros via num_valid_ids. NOTE: weight>0 is
    # NOT a valid-row signal — padding rows carry STALE nonzero weights, so
    # filtering on weight processes garbage rows the default excludes (this was the
    # e2e-mismatch root cause; §6c-viii). Real rows: default==0 iff row>=total_recv.
    # graph-safe routing: NO host syncs (no .item()) and NO boolean-index (which
    # produces data-dependent shapes). expert_base / total_recv may be 0-d DEVICE
    # tensors; all ops below are static-shape so the path is cuda-graph-capturable.
    local = recv_topk_ids.to(torch.long) - expert_base  # [M, topk]
    row_idx = torch.arange(M, device=dev).unsqueeze(1)
    valid = (local >= 0) & (local < E) & (row_idx < total_recv)
    flat_valid = valid.reshape(-1)

    import os as _osr
    _route_mode = _osr.environ.get("SGLANG_MORI_MASKED_ROUTE", "ondevice").lower()
    if _route_mode == "ondevice":
        # On-device atomic route (~us) replacing the one_hot+cumsum (57% of masked
        # MoE time; see profile). Invalid slots -> a TRASH expert E: the atomic on
        # atomic_buffer[E] is in-bounds (size E+1) and the rows_to_tokens store for
        # slot>=max_m is a buffer-resource OOB drop (safe). Yields grouped rows in
        # atomic (not cumsum) order -- fine, GEMM/combine only care per-expert set.
        from .grouped_moe_gfx1250 import _get_compiled_route_maps_guarded

        # invalid slots -> id E (>= experts) so the GUARDED kernel skips them (no
        # OOB store; no trash region needed). topids pre-zeroed so skipped routes
        # read 0 (weight 0). rows_to_tokens pre-filled -1 (empty rows).
        e_in = torch.where(valid, local, torch.full_like(local, E)).to(torch.int32).reshape(-1).contiguous()
        numel = M * topk
        atomic_buffer = torch.zeros(E, dtype=torch.int32, device=dev)
        topids = torch.zeros(numel, dtype=torch.int32, device=dev)
        rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device=dev)
        _get_compiled_route_maps_guarded()(
            e_in, atomic_buffer, topids, rows_to_tokens, numel, topk, max_m, E,
            (numel + 255) // 256, stream=torch.cuda.current_stream(),
        )
        # cap at max_m: atomic counts ALL valid slots but only slot<max_m are
        # stored; the GEMM must not read rows >= max_m (overflow tokens dropped).
        masked_m = atomic_buffer.clamp(max=max_m)
        # combine index: invalid slots -> row 0 (weight 0 -> harmless read).
        grouped_row = torch.where(flat_valid, topids, torch.zeros_like(topids)).view(M, topk)
        keep = flat_valid
        # gather source per grouped row (empty rows -> tok 0, skipped by masked_m).
        src_rows = rows_to_tokens[: E * max_m].clamp(min=0).long()
        _use_gather = True
    else:
        flat_local = local.reshape(-1)
        lc = flat_local.clamp(min=0, max=E - 1)
        oh_valid = F.one_hot(lc, E) * flat_valid.unsqueeze(1)
        slot = (oh_valid.cumsum(0).gather(1, lc[:, None]).squeeze(1) - 1).clamp(min=0)
        keep = flat_valid & (slot < max_m)
        masked_m = (F.one_hot(lc, E) * keep.unsqueeze(1)).sum(0).to(torch.int32)
        grouped_row = torch.where(keep, lc * max_m + slot, torch.zeros_like(lc)).view(M, topk)
        trash = E * max_m
        scatter_row = torch.where(keep, lc * max_m + slot, torch.full_like(lc, trash))
        src_tok = row_idx.expand(M, topk).reshape(-1)
        _use_gather = False

    import os as _os
    if _os.environ.get("SGLANG_MORI_MASKED_DUMP", "0") in ("1", "true", "True"):
        _lc = local.reshape(-1).clamp(min=0, max=E - 1)
        _want = torch.bincount(_lc[flat_valid], minlength=E)[:E]  # counts before cap
        _ovf = int((_want > max_m).sum().item())
        if _ovf and not getattr(flydsl_masked_moe_gfx950_recv, "_ovf_logged", False):
            print(
                f"[masked-recv OVERFLOW] {_ovf} local experts exceed max_m={max_m}; "
                f"max_count={int(_want.max().item())} (dropped tokens -> WRONG). "
                f"valid_slots={int(flat_valid.sum().item())}",
                flush=True,
            )
            flydsl_masked_moe_gfx950_recv._ovf_logged = True

    _rec(_pacc, "route", _t); _t = _ck()
    # --- scatter token hidden into expert-major grouped [E, max_m, Kw] ---
    # a1 activation dtype for stage1 (SGLANG_MORI_MASKED_A1_DTYPE):
    #  - "fp8" (default): DSV4 default is a8w4 (FP8 act) + gate_mode INTERLEAVE
    #    (fused_moe key q_dtype_a=fp8, kernel flydsl_moe1_afp8_wfp4_..._gui). mori
    #    dispatches mxfp4 -> DEQUANT fp4->bf16 then re-quant per-1x32 FP8.
    #  - "fp4": a4w4 -- scatter the mxfp4 recv bytes DIRECTLY (NO dequant/requant
    #    round-trip; big perf win). fp4 activation is coarser than the default's
    #    fp8 -> validate accuracy before trusting.
    import os as _os1
    _a1dt = _os1.environ.get("SGLANG_MORI_MASKED_A1_DTYPE", "fp8").lower()
    fp4x2 = _get_dtypes().fp4x2
    s_cols = model_dim // scale_blk  # per-1x32 e8m0 scale columns
    _direct_fp4 = recv_hidden.dtype == fp4x2 and _a1dt == "fp4"
    if recv_hidden.dtype == fp4x2 and not _direct_fp4:
        recv_hidden = _mxfp4_dequant_bf16_graphsafe(
            recv_hidden, recv_scale, M, model_dim, scale_blk
        )
        recv_scale = None
        is_fp8_recv = False
    # Materialise expert-major grouped [E*max_m, Kw]. ondevice route -> GATHER
    # (grouped[r] = recv[src_rows[r]]); torch route -> trash-row index_put scatter.
    if _direct_fp4:
        u8 = recv_hidden.view(torch.uint8)
        Kw = u8.shape[1]  # K/2 packed
        if _use_gather:
            grouped_a = u8[src_rows].view(fp4x2)
            grouped_scale = recv_scale[src_rows]
        else:
            grouped_u8 = torch.zeros((E * max_m + 1, Kw), dtype=torch.uint8, device=dev)
            grouped_u8[scatter_row] = u8[src_tok]
            grouped_scale = torch.zeros((E * max_m + 1, s_cols), dtype=recv_scale.dtype, device=dev)
            grouped_scale[scatter_row] = recv_scale[src_tok]
            grouped_a = grouped_u8[: E * max_m].view(fp4x2)
            grouped_scale = grouped_scale[: E * max_m]
        a_dtype = "fp4"
    elif is_fp8_recv:
        Kw = recv_hidden.shape[1]
        if _use_gather:
            grouped_a = recv_hidden[src_rows]
            grouped_scale = recv_scale[src_rows]
        else:
            grouped_a = torch.zeros((E * max_m + 1, Kw), dtype=recv_hidden.dtype, device=dev)
            grouped_scale = torch.zeros((E * max_m + 1, s_cols), dtype=recv_scale.dtype, device=dev)
            grouped_a[scatter_row] = recv_hidden[src_tok]
            grouped_scale[scatter_row] = recv_scale[src_tok]
            grouped_a = grouped_a[: E * max_m]
            grouped_scale = grouped_scale[: E * max_m]
        a_dtype = "fp8"
    else:  # bf16 recv: gather/scatter bf16 then a SINGLE per-1x32 fp8 quant (like DP)
        Kw = recv_hidden.shape[1]
        if _use_gather:
            grouped_bf16 = recv_hidden[src_rows]
        else:
            grouped_bf16 = torch.zeros((E * max_m + 1, Kw), dtype=recv_hidden.dtype, device=dev)
            grouped_bf16[scatter_row] = recv_hidden[src_tok]
            grouped_bf16 = grouped_bf16[: E * max_m]
        grouped_a, grouped_scale = _quant_per1x32_fp8(grouped_bf16)
        a_dtype = "fp8"

    _rec(_pacc, "scatter", _t); _t = _ck()
    a1_scale_shuf = e8m0_shuffle(grouped_scale)
    _rec(_pacc, "shuf1", _t); _t = _ck()

    # stage2 input (a2) dtype: "fp4" matches the shipping DSV4 a4w4 path.
    import os as _os3
    _a2dt = _os3.environ.get("SGLANG_MORI_MASKED_A2_DTYPE", "fp4").lower()
    # FUSED a2 quant (default): stage1 emits the quantized a2 + tiled scale in its
    # epilogue, ONLY for masked_m live rows -> no separate dense quant/shuffle over
    # all E*max_m rows (was ~48% of masked MoE time). Requires the masked
    # scale-write row fix in mixed_moe_gemm_2stage (global expert*max_m+row, not
    # local row). Unit logits_diff 5.9e-4. Disable with SGLANG_MORI_MASKED_FUSE_A2=0.
    _fuse_a2 = _os3.environ.get("SGLANG_MORI_MASKED_FUSE_A2", "1") in ("1", "true", "True")

    if _fuse_a2:
        s1_out = flydsl_masked_moe_stage1(
            a_grouped=grouped_a.view(E, max_m, Kw),
            w1=w1_shuf, masked_m=masked_m, max_m=max_m,
            w1_scale=w1_scale_shuf, a1_scale=a1_scale_shuf,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            a_dtype=a_dtype, act=activation, gate_mode=gate_mode,
            swiglu_limit=swiglu_limit, out_dtype=_a2dt,
        )
        _rec(_pacc, "stage1", _t); _t = _ck()
        a2q, a2_scale_shuf = s1_out  # scale already in tiled/shuffled layout
        if _a2dt == "fp4":
            a2q = a2q.view(E, max_m, inter_dim // 2)
        else:
            a2q = a2q.view(E, max_m, inter_dim)
        _s2_a_dtype = _a2dt
        _rec(_pacc, "quant2", _t); _t = _ck()  # ~0 now (fused into stage1)
    else:
        s1 = flydsl_masked_moe_stage1(
            a_grouped=grouped_a.view(E, max_m, Kw),
            w1=w1_shuf, masked_m=masked_m, max_m=max_m,
            w1_scale=w1_scale_shuf, a1_scale=a1_scale_shuf,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            a_dtype=a_dtype, act=activation, gate_mode=gate_mode,
            swiglu_limit=swiglu_limit,
        )
        _rec(_pacc, "stage1", _t); _t = _ck()
        if _a2dt == "fp4":
            from aiter.ops.quant import per_1x32_f4_quant

            a2q, a2_scale = per_1x32_f4_quant(
                s1.view(E * max_m, inter_dim), quant_dtype=_get_dtypes().fp4x2, shuffle=False
            )
            a2q = a2q.view(E, max_m, inter_dim // 2)
            a2_scale = a2_scale.view(E * max_m, inter_dim // 32)
            _s2_a_dtype = "fp4"
        else:
            a2q, a2_scale = _quant_per1x32_fp8(s1.view(E * max_m, inter_dim))
            a2q = a2q.view(E, max_m, inter_dim)
            _s2_a_dtype = "fp8"
        a2_scale_shuf = e8m0_shuffle(a2_scale)
        _rec(_pacc, "quant2", _t); _t = _ck()

    # combine weights per (token, slot): 0 for dropped/non-local/padding slots.
    import os as _os2
    _w = (
        torch.ones_like(recv_topk_weights)
        if _os2.environ.get("SGLANG_MORI_MASKED_NOWEIGHT", "0") in ("1", "true", "True")
        else recv_topk_weights
    )
    # ===== DEAD-END (kept opt-in OFF + diagnostics; do NOT enable) =====
    # Attempt to fuse the weighted un-permute (combine) into stage2's epilogue
    # (accumulate + token-major atomic un-permute), avoiding the separate
    # gather_reduce + s2 [E,max_m,model_dim] HBM round-trip.
    #
    # WHY DEAD-END:
    #  1. NON-CANONICAL: the shipping gfx1250 grouped MoE (the reference for this
    #     exact token-major->[E,max_m] design) ALSO uses a SEPARATE
    #     flydsl_moe_gather_reduce for combine and does NOT fuse it into stage2
    #     (see grouped_moe_gfx1250.py `_maybe_grouped_gfx1250_a8w4_moe` ~L1079).
    #     gather_reduce IS the canonical combine.
    #  2. KERNEL BUG (isolated, /tmp/dbg_combine.py): under
    #     grouped_masked_m + accumulate (e_vec=2), the mixed_moe_gemm_2stage
    #     epilogue misreads the masked MFMA acc -> out[r] = c[r%tile_m]*s2[r], a
    #     data-independent per-tile-row scalar (period=tile_m). Placement/weight
    #     are correct; the bug is in write_row_to_lds acc-extraction / e_vec=2
    #     column iteration vs the working grouped e_vec=8 plain store. Needs the
    #     flydsl/aiter kernel owner (or FLYDSL_DUMP_IR) to fix; not a Python-level
    #     issue. See MORI_EP_DECODE_ROOTCAUSE.md.
    #
    # Default OFF keeps the validated gather_reduce combine (gsm8k 0.92,
    # 2157 tok/s, 1.12x default -- the canonical grouped-MoE design performance).
    _fuse_combine = _use_gather and _os2.environ.get(
        "SGLANG_MORI_MASKED_FUSE_COMBINE", "0"
    ) in ("1", "true", "True")

    if _fuse_combine:
        gw_flat = torch.where(
            keep, _w.reshape(-1).to(torch.float32), torch.zeros(1, dtype=torch.float32, device=dev)
        )
        gr_flat = grouped_row.reshape(-1)
        trash = E * max_m
        scat = torch.where(keep, gr_flat, torch.full_like(gr_flat, trash))
        rw = torch.zeros(E * max_m + 1, dtype=torch.float32, device=dev)
        rw[scat] = gw_flat
        row_weight = rw[:E * max_m]
        if out is not None:
            out.zero_()
        out = flydsl_masked_moe_stage2(
            a_grouped=a2q, w2=w2_shuf, masked_m=masked_m, max_m=max_m,
            model_dim=model_dim, w2_scale=w2_scale_shuf, a2_scale=a2_scale_shuf,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, a_dtype=_s2_a_dtype,
            combine=True, row_to_token=src_rows, row_weight=row_weight,
            num_tokens=M, out=out,
        )
        _rec(_pacc, "stage2", _t)
        _rec(_pacc, "combine", _t)  # fused into stage2 (~0)
    else:
        s2 = flydsl_masked_moe_stage2(
            a_grouped=a2q, w2=w2_shuf, masked_m=masked_m, max_m=max_m,
            model_dim=model_dim, w2_scale=w2_scale_shuf, a2_scale=a2_scale_shuf,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, a_dtype=_s2_a_dtype,
        )
        _rec(_pacc, "stage2", _t); _t = _ck()
        topids_to_rows = grouped_row.view(M, topk).to(torch.int32)
        gather_w = torch.where(
            keep.view(M, topk), _w.to(s2.dtype), torch.zeros(1, dtype=s2.dtype, device=dev)
        ).contiguous()
        out = flydsl_moe_gather_reduce(
            s2.view(E, max_m, model_dim), topids_to_rows, gather_w, out=out
        )
        _rec(_pacc, "combine", _t)
    if _prof_on and _pn % 100 == 0:
        tot = sum(_pacc.values()) or 1.0
        line = "  ".join(
            f"{k}={_pacc[k] / _pn * 1e3:.3f}ms({_pacc[k] / tot * 100:.0f}%)"
            for k in ["route", "scatter", "shuf1", "stage1", "quant2", "stage2", "combine"]
            if k in _pacc
        )
        print(f"[masked-PROF] n={_pn} M={M} avg/call: {line}", flush=True)
    return out


def _maybe_grouped_gfx950_masked_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    E,
    model_dim,
    inter_dim,
    dtype,
    activation,
    quant_type,
    q_dtype_a,
    q_dtype_w,
    isG1U1,
    doweight_stage1,
    w1_scale,
    w2_scale,
    expert_mask,
    hidden_pad,
    intermediate_pad,
    bias1,
    bias2,
    gate_mode,
    swiglu_limit=None,
):
    """Opt-in gfx950 masked (deep_gemm-style) a8w4 MoE hook for ``fused_moe``.

    Default OFF: returns None unless ``AITER_GFX950_MASKED_MOE=1`` so the standard
    path is untouched. Expects raw (unshuffled) fp4 weights ``w1``/``w2`` and raw
    e8m0 ``w1_scale``/``w2_scale``; shuffles them here and runs the full masked
    grouped MoE. ``max_m`` = ``AITER_GFX950_MASKED_MAX_M`` if set, else derived from
    the routing (host sync -- eager only; the sglang mori-EP bridge instead passes
    the static recv-buffer cap = ``num_recv_tokens_per_expert`` capacity).

    On any unsupported condition it returns None (falls through to the default
    path), so enabling it can never break correctness -- only opt into the kernel.
    """
    import os

    if os.environ.get("AITER_GFX950_MASKED_MOE", "0") not in ("1", "true", "True"):
        return None
    try:
        from aiter.jit.core import get_gfx

        if get_gfx() != "gfx950":
            return None
    except Exception:
        return None

    from aiter import ActivationType, QuantType
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle
    from .moe_common import GateMode

    # v1 supported surface: a8w4 (fp4 weight), per_1x32, g1u1, silu/swiglu, GGUU,
    # no bias / expert_mask / pad. Anything else -> fall through.
    dtypes = _get_dtypes()
    if quant_type != QuantType.per_1x32:
        return None
    if q_dtype_w not in (dtypes.fp4x2, getattr(dtypes, "u8", torch.uint8)):
        return None
    if not isG1U1:
        return None
    if GateMode(gate_mode) != GateMode.SEPARATED:
        return None
    if activation not in (ActivationType.Silu, ActivationType.Swiglu):
        return None
    if w1_scale is None or w2_scale is None:
        return None
    if expert_mask is not None or bias1 is not None or bias2 is not None:
        return None
    if hidden_pad or intermediate_pad:
        return None
    if doweight_stage1:
        return None

    try:
        tile_m = 32
        scale_blk = 32
        act = "silu" if activation == ActivationType.Silu else "swiglu"
        w1p = w1.view(torch.uint8) if w1.dtype == dtypes.fp4x2 else w1
        w2p = w2.view(torch.uint8) if w2.dtype == dtypes.fp4x2 else w2
        w1_shuf = shuffle_weight(w1p.view(E, 2 * inter_dim, model_dim // 2), (16, 16))
        w2_shuf = shuffle_weight(w2p.view(E, model_dim, inter_dim // 2), (16, 16))
        w1s = e8m0_shuffle(
            w1_scale.reshape(E * 2 * inter_dim, model_dim // scale_blk)
        )
        w2s = e8m0_shuffle(w2_scale.reshape(E * model_dim, inter_dim // scale_blk))

        cap = int(os.environ.get("AITER_GFX950_MASKED_MAX_M", "0"))
        if cap <= 0:
            counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
            m_max = int(counts.max().item())
            cap = max(tile_m, ((m_max + tile_m - 1) // tile_m) * tile_m)
            # round up to a multiple of 32 (scale-block alignment)
            cap = ((cap + 31) // 32) * 32

        out = flydsl_masked_moe_gfx950(
            hidden_states=hidden_states,
            w1_shuf=w1_shuf,
            w2_shuf=w2_shuf,
            w1_scale_shuf=w1s,
            w2_scale_shuf=w2s,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            E=E,
            model_dim=model_dim,
            inter_dim=inter_dim,
            max_m=cap,
            activation=act,
            swiglu_limit=swiglu_limit,
            gate_mode="separated",
            tile_m=tile_m,
        )
        return out.to(dtype)
    except Exception:
        return None


def flydsl_masked_moe_stage1(
    a_grouped: torch.Tensor,  # [E, max_m, K] (fp8/fp4 packed) expert-major activations
    w1: torch.Tensor,  # [E, 2*inter_dim, K(/2 if fp4)] preshuffled weights
    masked_m: torch.Tensor,  # [E] int32, live token count per expert
    *,
    max_m: int,
    w1_scale: torch.Tensor,  # preshuffled B-scale (e8m0), same layout as e8m0_shuffle
    a1_scale: Optional[torch.Tensor] = None,  # preshuffled A-scale; None => a_scale_one
    out: Optional[torch.Tensor] = None,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    gate_mode: str = "separated",
    swiglu_limit: Optional[float] = None,
    persist_m: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
):
    """Masked expert-major stage1: ``act(x @ W_gate.T, x @ W_up.T)`` per expert.

    ``a_grouped`` / ``out`` are expert-major with a fixed per-expert stride
    ``max_m``; only the first ``masked_m[e]`` rows of expert ``e`` are computed
    (the rest are skipped, not zeroed). Returns ``out`` of shape
    ``[E, max_m, inter_dim]`` (bf16).
    """
    from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
    from .moe_common import GateMode

    # out_dtype "fp4"/"fp8" -> FUSED activation quant in the GEMM epilogue: the
    # kernel emits the quantized stage2 input (a2) + its e8m0 block scale directly,
    # and ONLY for the masked_m live rows (masked-aware). This replaces the separate
    # dense per_1x32 quant over all E*max_m rows (the old 48% cost). Returns
    # (out_packed, out_scale) where out_scale is already in the tiled/shuffled
    # layout stage2 consumes (== e8m0_shuffle output), so no extra shuffle.
    _need_fp4 = out_dtype == "fp4"
    _need_fp8_out = out_dtype == "fp8"
    _need_quant = _need_fp4 or _need_fp8_out
    if out_dtype not in ("bf16", "f16", "fp4", "fp8"):
        raise ValueError(
            f"flydsl_masked_moe_stage1 supports bf16/f16/fp4/fp8 output, got {out_dtype!r}"
        )

    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a_grouped.shape[-1]
    if a_dtype == "fp4":
        model_dim = model_dim * 2

    sort_block_m = max(32, tile_m)
    if max_m % sort_block_m != 0:
        raise ValueError(
            f"max_m ({max_m}) must be a multiple of sort_block_m ({sort_block_m})"
        )
    m_tiles_per_expert = max_m // sort_block_m
    size_expert_ids_in = m_tiles_per_expert * E
    tokens_in = E * max_m  # sizes the expert-major x buffer resource

    dtypes = _get_dtypes()
    dev = a_grouped.device
    out_scale = None
    if _need_quant:
        # packed fp4x2 [E,max_m,inter/2] or fp8 [E,max_m,inter]; + tiled e8m0 scale.
        if _need_fp4:
            out = torch.empty((E, max_m, inter_dim // 2), dtype=dtypes.fp4x2, device=dev)
        else:
            out = torch.empty((E, max_m, inter_dim), dtype=dtypes.fp8, device=dev)
        scale_cols = inter_dim // 32
        padded_rows = (tokens_in + 255) // 256 * 256
        padded_cols = (scale_cols + 7) // 8 * 8
        out_scale = torch.empty(
            padded_rows * padded_cols, dtype=torch.uint8, device=dev
        )
    else:
        torch_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        if out is None:
            out = torch.empty((E, max_m, inter_dim), dtype=torch_out_dtype, device=dev)

    a_scale_one = a1_scale is None
    flat_a_scale = (
        a1_scale.reshape(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = w1_scale.reshape(-1)

    # masked_m is carried in the repurposed arg_sorted_token_ids slot (int32[E]).
    masked_m_i32 = masked_m.to(torch.int32).contiguous()

    # unused-in-masked slots: expert_ids / sorted_weights get dummy buffers; the
    # num_valid_ids scalar is still loaded (line ~569) but not used by the masked
    # scheduler, so any 1-element int32 tensor is fine.
    dummy_i32 = torch.empty(0, device=dev, dtype=torch.int32)
    dummy_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    # graph-safe: torch.full does a device-side fill (no host->device memcpy that
    # torch.tensor([...], device=) would do, which is illegal during capture).
    # The masked scheduler loads but does not use this scalar.
    num_valid_ids = torch.full((1,), tokens_in, device=dev, dtype=torch.int32)
    out_scale_arg = out_scale if out_scale is not None else torch.empty(
        0, device=dev, dtype=torch.uint8
    )

    n_in = inter_dim * 2  # gate+up
    swiglu_limit_val = runtime_swiglu_limit(swiglu_limit, act)

    args = _s1_args_fp4(
        out.view(-1).view(torch.uint8) if _need_quant else out.view(-1),
        a_grouped.view(-1),
        w1.view(-1),
        flat_a_scale,
        flat_w_scale,
        masked_m_i32,
        dummy_i32,
        dummy_f32,
        num_valid_ids,
        out_scale_arg.view(-1),
        tokens_in,
        n_in,
        model_dim,
        size_expert_ids_in,
        dev,
        bias=torch.empty(0, device=dev),
        swiglu_limit=swiglu_limit_val,
    )

    exe = compile_mixed_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        persist_m=persist_m,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_mode=GateMode(gate_mode),
        enable_bias=False,
        a_scale_one=a_scale_one,
        grouped_masked_m=True,
        max_m=max_m,
    )
    _run_compiled(exe, args)
    if _need_quant:
        return out, out_scale
    return out


def flydsl_masked_moe_stage2(
    a_grouped: torch.Tensor,  # [E, max_m, inter] fp8 (stage1 output, quantized)
    w2: torch.Tensor,  # [E, model_dim, inter/2] preshuffled fp4 weights
    masked_m: torch.Tensor,  # [E] int32, live token count per expert
    *,
    max_m: int,
    model_dim: int,
    w2_scale: torch.Tensor,  # preshuffled B-scale (e8m0)
    a2_scale: torch.Tensor,  # preshuffled A-scale [E*max_m, inter//32]
    out: Optional[torch.Tensor] = None,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    persist_m: int = 4,
    waves_per_eu: int = 3,
    combine: bool = False,
    row_to_token: Optional[torch.Tensor] = None,  # [E*max_m] int32 (grouped->token)
    row_weight: Optional[torch.Tensor] = None,  # [E*max_m] f32 (per-grouped-row weight)
    num_tokens: int = 0,  # M (output token rows) when combine
):
    """Masked expert-major stage2: ``A2 @ W2.T`` per expert.

    ``combine=False`` -> grouped ``[E, max_m, model_dim]`` (weighted un-permute done
    later). ``combine=True`` -> FUSED weighted un-permute directly into token-major
    ``out[num_tokens, model_dim]`` via ``row_to_token``/``row_weight`` + atomic
    accumulate (``out`` MUST be pre-zeroed). Only ``masked_m[e]`` rows are computed.
    """
    from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

    if out_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"flydsl_masked_moe_stage2 v1 only supports bf16/f16 output, got {out_dtype!r}"
        )

    E = w2.shape[0]
    inter_dim = a_grouped.shape[-1]
    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    sort_block_m = max(32, tile_m)
    if max_m % sort_block_m != 0:
        raise ValueError(
            f"max_m ({max_m}) must be a multiple of sort_block_m ({sort_block_m})"
        )
    m_tiles_per_expert = max_m // sort_block_m
    size_expert_ids_in = m_tiles_per_expert * E
    tokens_in = E * max_m  # sizes x/out buffers with topk=1

    dtypes = _get_dtypes()
    dev = a_grouped.device
    torch_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    if out is None:
        if combine:
            out = torch.zeros((num_tokens, model_dim), dtype=torch_out_dtype, device=dev)
        else:
            out = torch.empty((E, max_m, model_dim), dtype=torch_out_dtype, device=dev)

    masked_m_i32 = masked_m.to(torch.int32).contiguous()
    dummy_i32 = torch.empty(0, device=dev, dtype=torch.int32)
    dummy_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    # masked_combine repurposes the (unused-in-masked) expert_ids / sorted_weights
    # arg slots to carry row_to_token / row_weight (both [E*max_m]).
    eid_arg = row_to_token.to(torch.int32).reshape(-1) if combine else dummy_i32
    sw_arg = row_weight.to(torch.float32).reshape(-1) if combine else dummy_f32
    # graph-safe: device-side fill (no host->device memcpy). See stage1 note.
    num_valid_ids = torch.full((1,), tokens_in, device=dev, dtype=torch.int32)

    args = _s2_args_fp4(
        out.view(-1),
        a_grouped.view(-1),
        w2.view(-1),
        a2_scale.reshape(-1),
        w2_scale.reshape(-1),
        masked_m_i32,
        eid_arg,
        sw_arg,
        num_valid_ids,
        tokens_in,
        model_dim,  # n_in (output dim)
        inter_dim,  # k_in (contraction dim)
        size_expert_ids_in,
        dev,
        bias=None,
    )

    exe = compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=combine,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=combine,
        enable_bias=False,
        persist_m=persist_m,
        waves_per_eu=waves_per_eu,
        grouped_masked_m=True,
        max_m=max_m,
        use_cshuffle_epilog=(False if combine else None),
    )
    _run_compiled(exe, args)
    return out
