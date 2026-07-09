"""E30: end-to-end bf16 EMULATION of the MoE with a selectable activation-quant scheme,
to test the quantization-matching thesis on gfx950 (where no a8w4/a16w4 MoE kernel exists,
E29). Env `SGLANG_MOE_EMUL` in {a4w4, a8w4, a16w4}:
  - a4w4 : activation mxfp4 q-dq  (VALIDATION: must reproduce native gfx950 ~0.93)
  - a8w4 : activation mxfp8 q-dq  (the TEST: gfx1250's scheme, on gfx950's a4w4-calibrated run)
  - a16w4: no activation quant    (upper precision bound)
Weights stay fp4 (dequanted per-forward per active expert -> ~fits; slow but memory-feasible).

Mechanism (no repo edits):
  1. Make the MoE weight/scale shuffles identity so weights keep the clean create_weights
     layout ([E, N, K//2] uint8 + [E, N, K//32] e8m0), dequantable by the E20 probe's
     dequant_mxfp4. (The real aiter kernel is bypassed, so its shuffled layout isn't needed.)
  2. Replace AiterRunnerCore.run with a bf16 grouped FFN: per active expert, dequant fp4->bf16,
     q-dq the activation per the scheme, silu(gate)*up @ down, topk-weighted sum. TP: experts
     replicated, inter-dim sharded -> per-rank partial down output (framework all-reduces).

Only the MoE math changes; routing/dispatch/combine/attention are the model's own code.
"""
import os

_MODE = os.environ.get("SGLANG_MOE_EMUL", "").lower()

if _MODE in ("a4w4", "a8w4", "a16w4"):
    import threading
    import time

    _MXFP4_VALUES = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    def _install():
        import torch

        for _ in range(600):
            try:
                import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe as QM
                from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerCore
                break
            except Exception:
                time.sleep(0.3)
        else:
            return

        # ---- 1. Neuter the weight/scale shuffles so the layout stays clean fp4 ----
        _ident = lambda w, *a, **k: w
        if hasattr(QM, "shuffle_weight"):
            QM.shuffle_weight = _ident
        if hasattr(QM, "e8m0_shuffle"):
            QM.e8m0_shuffle = _ident
        if hasattr(QM, "moe_shuffle_scale"):
            QM.moe_shuffle_scale = _ident

        _lut = None

        def _dequant_mxfp4(weight_u8, scale_e8m0):
            # weight_u8: [N, K//2] uint8 (two fp4/byte); scale_e8m0: [N, K//32] uint8 (e8m0)
            nonlocal _lut
            N, kp = weight_u8.shape
            K = kp * 2
            if _lut is None or _lut.device != weight_u8.device:
                _lut = torch.tensor(_MXFP4_VALUES, device=weight_u8.device, dtype=torch.float32)
            lo = (weight_u8 & 0xF).long()
            hi = (weight_u8 >> 4).long()
            vals = torch.empty(N, K, device=weight_u8.device, dtype=torch.float32)
            vals[:, 0::2] = _lut[lo]
            vals[:, 1::2] = _lut[hi]
            scale = torch.exp2(scale_e8m0.to(torch.float32) - 127.0)
            scale = torch.where(scale_e8m0 == 255, torch.zeros_like(scale), scale)
            scale = scale.view(N, K // 32, 1)
            return (vals.view(N, K // 32, 32) * scale).view(N, K)

        def _mxfp4_qdq(x):
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            xq, xs = dynamic_mxfp4_quant(x)
            return _dequant_mxfp4(xq.view(torch.uint8), xs)

        def _mxfp8_qdq(x):
            from aiter.ops.triton.quant import dynamic_mxfp8_quant
            from aiter import dtypes
            xq, xs = dynamic_mxfp8_quant(x, quant_dtype=dtypes.fp8)
            N, K = x.shape
            xf = xq.to(torch.float32).view(N, K // 32, 32)
            sc = torch.exp2(xs.to(torch.float32) - 127.0).view(N, K // 32, 1)
            return (xf * sc).view(N, K)

        if _MODE == "a4w4":
            _act_q = _mxfp4_qdq
        elif _MODE == "a8w4":
            _act_q = _mxfp8_qdq
        else:  # a16w4
            _act_q = lambda x: x.to(torch.float32)

        STATE = {"logged": False}

        def _run_emul(self, runner_input, quant_info, running_state, hooks=None):
            from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerOutput

            hs = runner_input.hidden_states
            if hs.shape[0] == 0:
                return AiterRunnerOutput(hidden_states=hs)

            topk_ids = runner_input.topk_ids.long()          # [T, topk]
            topk_w = runner_input.topk_weights.to(torch.float32)  # [T, topk]
            T = hs.shape[0]
            model_dim = hs.shape[-1]
            out_dtype = hs.dtype

            w13_u8 = quant_info.w13_weight.view(torch.uint8)  # [E, 2*inter, model//2]
            w2_u8 = quant_info.w2_weight.view(torch.uint8)    # [E, model, inter//2]
            w13_s = quant_info.w13_scale                       # [E, 2*inter, model//32]
            w2_s = quant_info.w2_scale                         # [E, model, inter//32]
            E = w13_u8.shape[0]

            xf = hs.to(torch.float32)
            out = torch.zeros(T, model_dim, device=hs.device, dtype=torch.float32)

            # experts actually referenced this forward
            active = torch.unique(topk_ids)
            for e in active.tolist():
                if e < 0 or e >= E:
                    continue
                sel = topk_ids == e                    # [T, topk]
                tok, slot = sel.nonzero(as_tuple=True)  # token idx, topk slot
                if tok.numel() == 0:
                    continue
                x_e = xf[tok]                          # [n, model]
                x_e = _act_q(x_e)                      # stage1 activation quant (fp32)
                w13 = _dequant_mxfp4(w13_u8[e], w13_s[e])  # [2*inter, model]
                gu = x_e @ w13.t()                     # [n, 2*inter]
                gate, up = gu.chunk(2, dim=-1)
                h = torch.nn.functional.silu(gate) * up  # [n, inter]
                h = _act_q(h)                          # stage2 activation quant
                w2 = _dequant_mxfp4(w2_u8[e], w2_s[e])     # [model, inter]
                y = h @ w2.t()                         # [n, model]
                out.index_add_(0, tok, y * topk_w[tok, slot].unsqueeze(-1))

            if not STATE["logged"]:
                print(f"[moe_emul] MODE={_MODE} active_experts={active.numel()} "
                      f"E={E} T={T} model={model_dim} inter={w13_u8.shape[1]//2}", flush=True)
                STATE["logged"] = True
            return AiterRunnerOutput(hidden_states=out.to(out_dtype))

        AiterRunnerCore.run = _run_emul
        print(f"[moe_emul] installed, MODE={_MODE} (shuffles neutered, run() replaced)", flush=True)

    threading.Thread(target=_install, daemon=True).start()
