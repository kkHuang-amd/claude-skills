# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AMD FlyDSL Mega-MoE backend for Deepseek V2/V4 (gfx950, a8w4).

This is the AMD counterpart to the DeepGEMM path in ``mega_moe.py``. It plugs
into the exact same three hooks that the DeepGEMM backend uses:

  * ``build_mega_moe_experts_weights(layer)`` -- called from the a8w4 MXFP4
    quant method's ``process_weights_after_loading`` to lay out the fused
    weights (and free the originals).
  * ``should_use_mega_moe(moe, hidden_states)`` -- routing gate consulted in
    ``DeepseekV2MoE.forward``.
  * ``forward_mega_moe(moe, hidden_states, ...)`` -- the fused forward.

Under the hood it drives FlyDSL's ``kernels.mega_moe.MegaMoE``, whose
``forward_bf16(x, topk_weights, topk_ids)`` runs dispatch + gemm1 + quant +
gemm2 + combine as one operator over mori's intranode all-to-all.

Everything here dispatches from ``mega_moe.py`` only when
``SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1``; on the default path this module is never
imported, so the FlyDSL/aiter/mori import chain stays optional.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

# One MegaMoE instance is shared across all layers: the kernel config (shapes,
# quant, mtpr, comm buffers) is identical layer-to-layer and only the weight
# pointers differ, so we build once and hot-swap w1/w2 per layer on each call.
# Keyed by the invariant kernel config.
_MEGA_MOE_INSTANCE: dict = {}
# Tracks whether the prefill+decode instance pair has been pre-built (once).
_MEGA_MOE_BUILT_MTPRS: set = set()
_FLYDSL_PATH_READY = False
# Diagnostic: last dual-mode selection, to detect a compact<->fixed-slot switch across forwards.
_MEGA_LAST_USE_DECODE = None


def _ensure_flydsl_on_path() -> None:
    """Make ``import kernels.mega_moe`` resolvable.

    Uses SGLANG_AMD_FLYDSL_KERNELS_PATH, falling back to $ATOM_FLYDSL_KERNELS_PATH
    (the ATOM guide's variable), so an operator can point at the FlyDSL workspace
    without a code change. The compiled ``flydsl`` lib itself is expected to be
    importable already (bundled in the container's site-packages).
    """
    global _FLYDSL_PATH_READY
    if _FLYDSL_PATH_READY:
        return
    path = envs.SGLANG_AMD_FLYDSL_KERNELS_PATH.get() or os.environ.get(
        "ATOM_FLYDSL_KERNELS_PATH", ""
    )
    if path and path not in sys.path:
        sys.path.insert(0, path)
    _FLYDSL_PATH_READY = True


def _import_flydsl():
    """Lazily import the FlyDSL MegaMoE kernel + weight-prep helpers.

    Single entry point for the whole FlyDSL-workspace dependency so the coupling
    is contained and documented in one place (rather than scattered imports):
      * ``kernels.mega_moe.MegaMoE`` -- the fused MegaMoE op (FlyDSL ``kernels/``).
      * ``per_1x32_fp4_quant`` / ``fp4_utils`` / ``shuffle_weight`` -- the MXFP4
        quant + weight-shuffle helpers used at weight build. These currently live
        under the FlyDSL repo's ``tests/`` tree; ROCm/FlyDSL does not yet expose
        them via the ``flydsl`` package, so we import them from the workspace made
        importable by _ensure_flydsl_on_path(). TODO: switch to a public
        ``flydsl.*`` API once FlyDSL promotes these out of ``tests/``.

    Imported here (not at module top) so a stock sglang install without the
    FlyDSL workspace stays unaffected; raises a clear error if unavailable.
    """
    _ensure_flydsl_on_path()
    try:
        # FlyDSL relocated MegaMoE under kernels/moe/ on the mega_moe_v1 branch;
        # older snapshots kept it top-level. Accept both so the integration is
        # robust to the exact FlyDSL checkout.
        try:
            from kernels.moe.mega_moe import MegaMoE
        except ImportError:
            from kernels.mega_moe import MegaMoE
        from tests.kernels.test_moe_gemm import _per_1x32_fp4_quant
        # The MXFP4 weight-prep helpers (mxfp4_to_f32/e8m0_*/shuffle_weight_w4/
        # shuffle_scale_w4) were renamed fp4_utils -> gemm_common_utils in the
        # "packaged" FlyDSL refactor. Accept both; alias to fp4_utils downstream.
        try:
            from tests.kernels.utils import gemm_common_utils as fp4_utils
        except ImportError:
            from tests.kernels.utils import fp4_utils
        from tests.utils import shuffle_weight
    except ImportError as exc:  # pragma: no cover - depends on external workspace
        raise ImportError(
            "FlyDSL MegaMoE backend requires the FlyDSL workspace on PYTHONPATH "
            "(kernels.mega_moe + the tests/ MXFP4 weight-prep helpers). Set "
            "SGLANG_AMD_FLYDSL_KERNELS_PATH (or $ATOM_FLYDSL_KERNELS_PATH) to the "
            f"FlyDSL checkout. Original import error: {exc}"
        ) from exc

    from types import SimpleNamespace

    return SimpleNamespace(
        MegaMoE=MegaMoE,
        per_1x32_fp4_quant=_per_1x32_fp4_quant,
        fp4_utils=fp4_utils,
        shuffle_weight=shuffle_weight,
    )


def _mtpr() -> int:
    mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.get())
    if mtpr & (mtpr - 1) != 0:
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr} must be a power of two "
            "(MegaMoE requirement)."
        )
    return mtpr


def _ep_rank_world():
    """This rank's (rank, world_size) within the MoE expert-parallel group."""
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group = get_moe_ep_group().device_group
    return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)


_MORI_SHMEM_READY = False


def _ensure_mori_shmem() -> None:
    """Initialize mori's (process-global) symmetric heap over the MoE EP group.

    FlyDSL's MegaMoE / dispatch_combine allocate from mori's shmem heap
    (``shmem_malloc``), which aborts unless ``shmem_torch_process_group_init``
    has run. sglang only does that inside the moriep token dispatcher (group
    "mori"), which the megamoe a2a backend bypasses -- so we bootstrap it here,
    once, mirroring moriep's register+init pattern. mori shmem is a per-process
    singleton, so a single init on the EP group is what all MegaMoE layers use.

    Must be reached collectively by every EP rank (it broadcasts a unique id);
    the first routed forward (incl. warmup) is such a point.
    """
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return
    import mori.shmem

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group_name = "megamoe_flydsl"
    cpu_group = get_moe_ep_group().cpu_group
    # Register the named c10d PG if absent. IMPORTANT: PG registration and mori's
    # symmetric-heap init are INDEPENDENT. In PD-disaggregation the PG can already
    # be registered (or mori.io, used for KV transfer, brought up the mori runtime)
    # while the MegaMoE symmetric heap was NEVER allocated -- observed as
    # "[shmem] Pointer ... not in symmetric heap [0x0, 0x0)" -> NULL-deref GPU fault
    # in FlyDSL stage1. The old code SKIPPED shmem init when the PG pre-existed,
    # which is wrong. So register-if-needed, then ALWAYS init the shmem heap.
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as e:  # noqa: BLE001
        if "already registered" not in str(e):
            raise
    mori.shmem.shmem_torch_process_group_init(group_name)
    try:
        logger.info(
            "FlyDSL MegaMoE mori shmem init: group=%s mype=%s npes=%s",
            group_name,
            mori.shmem.shmem_mype(),
            mori.shmem.shmem_npes(),
        )
    except Exception:  # noqa: BLE001
        pass
    _MORI_SHMEM_READY = True


# ---------------------------------------------------------------------------
# Weight layout (accuracy-critical). We do NOT feed the quark checkpoint's fp4
# bytes straight into shuffle_weight: quark's MXFP4 encode convention (e.g. the
# max->dtypeMax scale mapping) need not match what FlyDSL's kernel assumes, so a
# direct reuse silently corrupts magnitudes (gsm8k 0.96 -> ~0.5). Instead we
# dequantize with the standard OCP decode (convention-independent) and re-encode
# with FlyDSL's OWN quantizer -- exactly the prep the standalone bench
# (tests/kernels/test_mega_moe.py::_prepare) validates: raw fp4 (from
# _per_1x32_fp4_quant) -> shuffle_weight(16,16) + fp4_utils.e8m0_shuffle. In
# sglang EP each rank already holds ONLY its local experts, so no global-slice.
# ---------------------------------------------------------------------------
def build_mega_moe_experts_weights(layer) -> None:
    if getattr(layer, "_mega_moe_weights_built", False):
        return

    _fd = _import_flydsl()
    _per_1x32_fp4_quant = _fd.per_1x32_fp4_quant
    fp4_utils = _fd.fp4_utils
    shuffle_weight = _fd.shuffle_weight  # FlyDSL layout (NOT aiter)

    fp4_view = torch.float4_e2m1fn_x2

    # a8w4 drives MegaMoE's fused gate_mode=INTERLEAVE (g1u1), which consumes a
    # DIFFERENT w1 layout than the SEPARATED plain shuffle_weight: it needs
    # fp4_utils.shuffle_weight_w4(gate_up=True) + shuffle_scale_w4 (exactly what
    # tests/kernels/test_mega_moe.py::_prepare feeds an interleave MegaMoE).
    # Feeding plain-shuffled (separated) w1 to the interleave kernel silently
    # produces garbage. a4w4 uses the separated layout. w2 is always separated.
    mega_quant = getattr(layer, "_mega_quant", "a8w4")
    interleave = mega_quant == "a8w4"

    def _requant_shuffle(w_u8_3d, scale_u8_3d, *, gate_up):
        """quark (fp4 bytes + e8m0 scale) -> f32 -> FlyDSL fp4 -> shuffled bytes.

        w_u8_3d: [E, rows, K//2] uint8 (2 fp4/byte); scale_u8_3d: [E, rows, K//32]
        e8m0 uint8. Returns (w_shuffled_flat_u8, scale_shuffled_flat_u8).
        """
        e, rows, k_half = w_u8_3d.shape
        k = k_half * 2
        vals = fp4_utils.mxfp4_to_f32(w_u8_3d.view(fp4_view))  # [E, rows, K]
        sc = fp4_utils.e8m0_to_f32(scale_u8_3d)  # [E, rows, K//32]
        w_f32 = (vals.view(e, rows, k // 32, 32) * sc.view(e, rows, k // 32, 1)).view(
            e * rows, k
        )
        del vals, sc
        w_fp4, w_scale = _per_1x32_fp4_quant(w_f32)  # FlyDSL convention
        del w_f32
        if gate_up and interleave:
            w_out = (
                fp4_utils.shuffle_weight_w4(
                    w_fp4.view(e, rows, k // 2),
                    NLane=16,
                    gate_up=True,
                    moe_gemm=True,
                )
                .view(torch.uint8)
                .contiguous()
                .view(-1)
            )
            s_out = (
                fp4_utils.shuffle_scale_w4(
                    w_scale.view(e * rows, k // 32),
                    experts_cnt=e,
                    gate_up=True,
                )
                .view(torch.uint8)
                .contiguous()
                .view(-1)
            )
        else:
            w_out = shuffle_weight(w_fp4).view(torch.uint8).contiguous().view(-1)
            s_out = (
                fp4_utils.e8m0_shuffle(w_scale).view(torch.uint8).contiguous().view(-1)
            )
        return w_out, s_out

    e_local = layer.w13_weight.shape[0]
    # Padded dims (round_up 256 in the quark scheme) -- use them, per ATOM guide.
    hidden = layer.w13_weight.shape[2] * 2  # packed fp4 -> 2 values/byte
    inter = layer.w2_weight.shape[2] * 2

    # Scale attribute naming differs by quant method: quark (R1) registers
    # ``w13_weight_scale``; sglang's Fp8MoEMethod fp4-experts path (DeepSeek V4)
    # registers ``w13_weight_scale_inv``. Both hold the same e8m0 per-1x32 scale
    # (uint8, [E, rows, K//32]), so resolve whichever exists.
    def _scale_of(s_attr):
        s = getattr(layer, s_attr, None)
        if s is None:
            s = getattr(layer, s_attr + "_inv", None)
        if s is None:
            raise AttributeError(
                f"MegaMoE build: neither {s_attr} nor {s_attr}_inv on layer "
                f"(have: {[n for n in vars(layer) if 'scale' in n]})"
            )
        return s.data

    layer._mega_w1, layer._mega_w1_scale = _requant_shuffle(
        layer.w13_weight.data, _scale_of("w13_weight_scale"), gate_up=True
    )
    layer._mega_w2, layer._mega_w2_scale = _requant_shuffle(
        layer.w2_weight.data, _scale_of("w2_weight_scale"), gate_up=False
    )
    dev = layer.w13_weight.device

    # Pitfall 2: the original fp4 buffers are now dead fallback. Keeping them
    # doubles expert-weight memory and crushes the KV cache; drop them.
    layer.w13_weight.data = torch.empty(0, dtype=torch.uint8, device=dev)
    layer.w2_weight.data = torch.empty(0, dtype=torch.uint8, device=dev)

    layer._mega_moe_weights_built = True
    logger.info(
        "FlyDSL MegaMoE weights built: e_local=%d hidden=%d inter=%d "
        "w1=%d w1_scale=%d bytes/expert",
        e_local,
        hidden,
        inter,
        layer._mega_w1.numel() // e_local,
        layer._mega_w1_scale.numel() // e_local,
    )


def _get_or_build_mega_moe(
    layer,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str = "a8w4",
    mtpr: Optional[int] = None,
    decode_cap: int = 0,
):
    """Return the shared MegaMoE, building it once from ``layer``'s weights.

    ``mtpr`` sizes the symmetric buffers (prefill capacity). ``decode_cap`` > 0 builds a
    2nd fixed-slot (decode) stage-1 kernel INSIDE the same instance / same comb_op
    (approach A: one comb_op, dual-mode -> avoids the 2nd-comb_op shared-heap deadlock);
    it is selected per-forward via ``decode=``. Keyed by (mtpr, decode_cap); hot-swaps the
    same per-layer weights.
    """
    _ensure_mori_shmem()
    MegaMoE = _import_flydsl().MegaMoE

    rank, world = _ep_rank_world()
    if mtpr is None:
        mtpr = _mtpr()
    key = (rank, world, model_dim, inter_dim, experts, topk, quant, mtpr, int(decode_cap))
    mega = _MEGA_MOE_INSTANCE.get(key)
    if mega is None:
        mega = MegaMoE(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,  # GLOBAL count; MegaMoE derives epr = experts // world
            topk=topk,
            quant=quant,
            w1=layer._mega_w1,  # LOCAL (this rank's epr experts)
            w1_scale=layer._mega_w1_scale,
            w2=layer._mega_w2,
            w2_scale=layer._mega_w2_scale,
            max_tok_per_rank=mtpr,
            # stage1 GEMM tile is selected ONCE at build from a token bucket. It
            # defaults to the mtpr bucket (8192 -> prefill tile 64x256), which is
            # ~2x too large for decode (16-64 tok) -> slow moe_gemm1. tune_tokens
            # decouples the TILE-selection token from mtpr (buffers stay mtpr-sized)
            # so we can pick the decode tile (e.g. 64 -> 32x128) without shrinking
            # the prefill capacity. 0/unset = mtpr (original behavior).
            tune_tokens=(
                _tune_tok if (_tune_tok := int(
                    os.environ.get("SGLANG_AMD_FLYDSL_MEGA_TUNE_TOKENS", "0"))) > 0
                else None
            ),
            # gate_mode MUST match the w1 layout built in build_mega_moe_experts_
            # weights: a8w4 -> interleave (g1u1, shuffle_weight_w4), a4w4 ->
            # separated. Passed explicitly (mirrors the bench) rather than relying
            # on MegaMoE's gate_mode=None default.
            gate_mode="interleave" if quant == "a8w4" else "separated",
            # Explicit GEMM2 tile, matching the validated standalone bench
            # (tests/kernels/test_mega_moe.py: tm2/tn2/tk2 = 32/128/256). The
            # default (-1 -> MegaGemm2 auto tune-table) selects a tile whose
            # _k_shift_bits emits an invalid arith.constant (std::bad_cast) at
            # some token counts. Pinning the bench tile avoids that path.
            gemm2_tile_m=32,
            gemm2_tile_n=128,
            gemm2_tile_k=256,
            # approach A: also build the fixed-slot decode kernel INSIDE this instance
            # (one comb_op, dual-mode). 0 = compact-only (original single-mode).
            decode_cap=int(decode_cap),
        )
        _MEGA_MOE_INSTANCE[key] = mega
    return mega


def _swap_layer_weights(mega, layer) -> None:
    """Point the shared MegaMoE at this layer's weight buffers (no realloc).

    The fused stage-1 keeps the gate/up weights on the MegaMoE instance itself
    as ``_s1_w1`` / ``_s1_w1_scale`` (the mega_moe_v1 branch folded
    MegaMoeStage1/2 into MegaMoE); older snapshots exposed them via a
    ``stage1`` sub-object. Accept both. w2/w2_scale stayed on the instance.
    Forward reads all four attributes at call time (passed as kernel args), so
    re-pointing them hot-swaps this layer's weights with no realloc.
    """
    if hasattr(mega, "_s1_w1"):
        mega._s1_w1 = layer._mega_w1
        mega._s1_w1_scale = layer._mega_w1_scale
    else:
        mega.stage1.w1 = layer._mega_w1
        mega.stage1.w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale


# ---------------------------------------------------------------------------
# Routing gate + forward. Structure mirrors mega_moe.py (DeepGEMM) so the
# DeepseekV2MoE call site is backend-agnostic.
# ---------------------------------------------------------------------------
def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    max_tokens_per_rank = (
        max(global_num_tokens) if global_num_tokens else hidden_states.shape[0]
    )
    return max_tokens_per_rank <= _mtpr()


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )
    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(moe, hidden_states, forward_batch, input_ids_global)

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = moe.config.hidden_size
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    inter_dim = moe.config.moe_intermediate_size
    num_experts = moe.experts.num_experts

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        x_in = hidden_states
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
    else:
        # Idle DP rank (0 real tokens). The mega forward is a collective (mori
        # dispatch + combine), so every EP rank must still call it -- skipping
        # would deadlock the a2a. But FlyDSL's kernels need >=1 row (a 0-sized
        # grid raises "HIP error: invalid configuration argument"), so run one
        # dummy token (balanced routing, zero weight) and slice it back off.
        x_in = hidden_states.new_zeros((1, hidden_size))
        topk_ids = (
            torch.arange(top_k, device=hidden_states.device, dtype=torch.int32)
            % num_experts
        ).unsqueeze(0)
        topk_weights = hidden_states.new_zeros((1, top_k), dtype=torch.float32)

    prefill_mtpr = _mtpr()
    # Dual-mode (approach A): ONE MegaMoE built with decode_cap>0 holds BOTH a compact (prefill)
    # and a fixed-slot (decode) stage-1 kernel over the SAME comb_op buffers; the decode kernel is
    # ~18-21% faster at decode (skips compact's extra cross-PE round). We pick it per-forward via
    # decode=. Crucially this uses ONE comb_op: a 2nd comb_op's mere coexistence on the single
    # global mori heap deadlocks the scheduler (confirmed). decode_mtpr=0 -> compact-only (original).
    decode_mtpr = int(os.environ.get("SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR", "0"))
    if decode_mtpr > 0 and (decode_mtpr & (decode_mtpr - 1)) != 0:
        # round up to a power of two (MegaMoE requires it)
        decode_mtpr = 1 << (int(decode_mtpr) - 1).bit_length()
    # decode= MUST be identical on EVERY EP rank: the mega dispatch is a cross-PE collective (mori
    # a2a), so if ranks disagree on compact(prefill) vs fixed-slot(decode) they wait on different
    # shmem flags -> DEADLOCK. Decide by the DP-GLOBAL max token count (identical on all ranks via
    # get_dp_global_num_tokens), NOT the local num_tokens (which differs per rank under DP-attention).
    # decode= MUST be identical on EVERY EP rank (the mega dispatch is a collective). forward_mode
    # is a GLOBAL per-step property -- all DP/EP ranks run the SAME mode each step (idle ranks
    # included) -- and at a graphed decode step every rank is padded to the same bs, so num_tokens
    # is cross-rank-consistent too. get_dp_global_num_tokens() is NOT usable here: it returned a
    # per-rank / len-1 value (and None under cuda-graph capture), so ranks DISAGREED on decode-vs-
    # prefill -> some entered the fixed-slot dispatch while others entered compact -> DEADLOCK.
    _is_decode_step = bool(
        forward_batch is not None
        and getattr(forward_batch, "forward_mode", None) is not None
        and forward_batch.forward_mode.is_decode()
    )
    # Token bound for the decode gate MUST be cross-rank-consistent (local num_tokens is per-rank under
    # DP-attention -> ranks could disagree -> the fixed-slot vs compact dispatch rendezvous DEADLOCKS,
    # proven via the --hang-diverge repro). So:
    #   - under cuda-graph capture/replay: decode bs is bounded by cuda_graph_max_bs; gate on is_decode
    #     ALONE (requires decode_mtpr >= cuda_graph_max_bs so fixed-slot always fits) -> globally identical.
    #   - eager: use the DP-GLOBAL max token count (max(get_dp_global_num_tokens()), identical on all
    #     ranks), never the local num_tokens.
    if get_is_capture_mode():
        _tok_ok = True
    else:
        _gnt = get_dp_global_num_tokens()
        _tok_ok = (max(_gnt) <= decode_mtpr) if _gnt else (num_tokens <= decode_mtpr)
    use_decode = bool(decode_mtpr > 0 and _is_decode_step and _tok_ok)
    # Diagnostic (SGLANG_AMD_FLYDSL_MEGA_MODE_BARRIER=1, eager only): on a compact<->fixed-slot
    # switch, barrier all EP ranks so the previous mode's cross-PE mori state is fully drained
    # before the next mode starts -> isolates whether interleaving the two dispatch paths corrupts
    # mori's shared/global shmem state. use_decode is cross-rank-consistent (forward_mode), so all
    # ranks switch on the same step and reach this barrier together. NOT cuda-graph-safe.
    if decode_mtpr > 0 and os.environ.get("SGLANG_AMD_FLYDSL_MEGA_MODE_BARRIER", "") == "1":
        global _MEGA_LAST_USE_DECODE
        if _MEGA_LAST_USE_DECODE is not None and _MEGA_LAST_USE_DECODE != use_decode:
            import mori.shmem as _ms

            _ms.shmem_barrier_all()
        _MEGA_LAST_USE_DECODE = use_decode
    if decode_mtpr > 0 and os.environ.get("SGLANG_AMD_FLYDSL_MEGA_DEBUG", "") == "1":
        import torch.distributed as _dist

        _rk = _dist.get_rank() if _dist.is_initialized() else -1
        logger.info(
            "[flydsl-mega-dbg] rank=%s layer=%s capture=%s is_decode=%s tok_ok=%s local_tok=%d "
            "decode_mtpr=%d -> use_decode=%s",
            _rk, getattr(moe, "layer_id", "?"), get_is_capture_mode(), _is_decode_step, _tok_ok,
            num_tokens, decode_mtpr, use_decode,
        )
    # Definitive cross-rank DIVERGENCE detector (opt-in; eager only -- a collective can't run mid-capture).
    # All-reduce use_decode over the world group BEFORE the mega dispatch: if ranks disagree, the fixed-slot
    # vs compact rendezvous WOULD deadlock -- log it LOUDLY (rank, layer, values) instead of silently hanging.
    if (
        decode_mtpr > 0
        and os.environ.get("SGLANG_AMD_FLYDSL_MEGA_DIVERGE_CHECK", "") == "1"
        and not get_is_capture_mode()
    ):
        try:
            import torch.distributed as _dist

            if _dist.is_initialized():
                _v = torch.tensor(
                    [1 if use_decode else 0], device=hidden_states.device, dtype=torch.int32
                )
                _mx = _v.clone()
                _mn = _v.clone()
                _dist.all_reduce(_mx, op=_dist.ReduceOp.MAX)
                _dist.all_reduce(_mn, op=_dist.ReduceOp.MIN)
                if int(_mx.item()) != int(_mn.item()):
                    logger.error(
                        "[flydsl-mega-DIVERGE] rank=%s layer=%s use_decode=%s BUT RANKS DISAGREE "
                        "(max=%d min=%d) is_decode=%s tok_ok=%s local_tok=%d -> WOULD DEADLOCK the "
                        "cross-PE dispatch rendezvous",
                        _dist.get_rank(), getattr(moe, "layer_id", "?"), use_decode,
                        int(_mx.item()), int(_mn.item()), _is_decode_step, _tok_ok, num_tokens,
                    )
        except Exception as _e:  # never let the detector itself break the forward
            logger.warning("[flydsl-mega] diverge-check failed: %s", _e)
    assert num_tokens <= prefill_mtpr, (
        f"FlyDSL mega MoE: num_tokens={num_tokens} exceeds mtpr={prefill_mtpr}; raise "
        "SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR or shrink cuda_graph_max_bs / chunked_prefill_size."
    )

    _quant = (
        envs.SGLANG_AMD_FLYDSL_MEGA_QUANT.get()
        or getattr(moe.experts, "_mega_quant", "a8w4")
    )
    mega = _get_or_build_mega_moe(
        moe.experts,
        model_dim=hidden_size,
        inter_dim=inter_dim,
        experts=num_experts,
        topk=top_k,
        # Weight layout is identical for a4w4/a8w4 (only activation quant differs), so
        # SGLANG_AMD_FLYDSL_MEGA_QUANT can force a8w4 on an a4w4 checkpoint.
        quant=_quant,
        mtpr=prefill_mtpr,
        decode_cap=decode_mtpr,
    )
    _swap_layer_weights(mega, moe.experts)

    # bf16 in, single fused op, bf16 out. decode= selects the fixed-slot decode kernel (dual-mode).
    # Accept older snapshots without decode=/slice_output kwargs.
    _fwd = getattr(mega, "forward_bf16", None) or mega.forward
    try:
        y = _fwd(x_in, topk_weights, topk_ids, slice_output=False, decode=use_decode)
    except TypeError:
        try:
            y = _fwd(x_in, topk_weights, topk_ids, slice_output=False)
        except TypeError:
            y = _fwd(x_in, topk_weights, topk_ids)
    y = y[:num_tokens]

    # routed_scaling_factor must be applied EXACTLY once. On AMD the router uses
    # aiter_biased_grouped_topk, which folds routed_scaling_factor into each
    # routed topk weight; the reference DeepSeek forward paths therefore skip the
    # post-MoE multiply whenever _use_aiter (guard `not (should_fuse or
    # _use_aiter)`). should_fuse_routed_scaling_factor_in_topk is False for quark,
    # so guarding on it alone double-applied RSF here (routed came out RSF x too
    # large -> gsm8k 0.41 vs 0.97 fixed). Mirror the reference guard.
    try:
        from sglang.srt.models.deepseek_common.utils import _use_aiter
    except Exception:  # noqa: BLE001
        _use_aiter = True  # this FlyDSL MegaMoE path is AMD-only; aiter folds RSF
    if not (moe.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter):
        y.mul_(moe.routed_scaling_factor)
    return y
