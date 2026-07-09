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

        def _dequant_mxfp4_2d(weight_u8, scale_e8m0):
            # weight_u8: [M, K//2] uint8 (two fp4/byte); scale_e8m0: [M, K//32] uint8 (e8m0)
            # All-bf16 to keep peak memory low (fp32 [M,K] scratch would OOM for all experts).
            nonlocal _lut
            M, kp = weight_u8.shape
            K = kp * 2
            if _lut is None or _lut.device != weight_u8.device:
                _lut = torch.tensor(_MXFP4_VALUES, device=weight_u8.device, dtype=torch.bfloat16)
            lo = (weight_u8 & 0xF).long()
            hi = (weight_u8 >> 4).long()
            vals = torch.empty(M, K, device=weight_u8.device, dtype=torch.bfloat16)
            vals[:, 0::2] = _lut[lo]
            vals[:, 1::2] = _lut[hi]
            scale = torch.exp2(scale_e8m0.to(torch.float32) - 127.0)
            scale = torch.where(scale_e8m0 == 255, torch.zeros_like(scale), scale).to(torch.bfloat16)
            scale = scale.view(M, K // 32, 1)
            return (vals.view(M, K // 32, 32) * scale).view(M, K)

        def _dequant_batched(w_u8, s_e8m0):
            # [A, N, K//2] uint8 + [A, N, K//32] -> [A, N, K] bf16, one dequant call
            A, N, kp = w_u8.shape
            out = _dequant_mxfp4_2d(w_u8.reshape(A * N, kp), s_e8m0.reshape(A * N, -1))
            return out.view(A, N, kp * 2)

        def _mxfp4_qdq(x):
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            xq, xs = dynamic_mxfp4_quant(x)
            return _dequant_mxfp4_2d(xq.view(torch.uint8), xs)

        def _mxfp8_qdq(x):
            from aiter.ops.triton.quant import dynamic_mxfp8_quant
            from aiter import dtypes
            xq, xs = dynamic_mxfp8_quant(x, quant_dtype=dtypes.fp8)
            N, K = x.shape
            xf = xq.to(torch.float32).view(N, K // 32, 32)
            sc = torch.exp2(xs.to(torch.float32) - 127.0).view(N, K // 32, 1)
            return (xf * sc).view(N, K).to(torch.bfloat16)

        if _MODE == "a4w4":
            _act_q = _mxfp4_qdq
        elif _MODE == "a8w4":
            _act_q = _mxfp8_qdq
        else:  # a16w4
            _act_q = lambda x: x.to(torch.bfloat16)

        STATE = {"logged": False}

        def _run_emul(self, runner_input, quant_info, running_state, hooks=None):
            from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerOutput

            hs = runner_input.hidden_states
            if hs.shape[0] == 0:
                return AiterRunnerOutput(hidden_states=hs)

            topk_ids = runner_input.topk_ids.long()               # [T, topk]
            topk_w = runner_input.topk_weights.to(torch.float32)  # [T, topk]
            T = hs.shape[0]
            model_dim = hs.shape[-1]
            out_dtype = hs.dtype

            w13_u8 = quant_info.w13_weight.view(torch.uint8)  # [E, 2*inter, model//2]
            w2_u8 = quant_info.w2_weight.view(torch.uint8)    # [E, model, inter//2]
            w13_s = quant_info.w13_scale                       # [E, 2*inter, model//32]
            w2_s = quant_info.w2_scale                         # [E, model, inter//32]
            E = w13_u8.shape[0]

            # stage1 activation quant done ONCE for all tokens (per-row scale anyway)
            xq_all = _act_q(hs.to(torch.bfloat16))  # [T, model] bf16
            out = torch.zeros(T, model_dim, device=hs.device, dtype=torch.float32)

            active = torch.unique(topk_ids)
            active = active[(active >= 0) & (active < E)]
            # batched dequant of ONLY the active experts (2 calls, not 2*A)
            w13d = _dequant_batched(w13_u8[active], w13_s[active])  # [A, 2*inter, model] bf16
            w2d = _dequant_batched(w2_u8[active], w2_s[active])      # [A, model, inter] bf16

            for ai, e in enumerate(active.tolist()):
                sel = topk_ids == e
                tok, slot = sel.nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                x_e = xq_all[tok]                       # [n, model] bf16
                gu = x_e @ w13d[ai].t()                 # [n, 2*inter] bf16
                gate, up = gu.chunk(2, dim=-1)
                h = torch.nn.functional.silu(gate) * up  # [n, inter]
                h = _act_q(h)                           # stage2 activation quant
                y = h @ w2d[ai].t()                     # [n, model] bf16
                out.index_add_(
                    0, tok, (y.to(torch.float32) * topk_w[tok, slot].unsqueeze(-1))
                )

            if not STATE["logged"]:
                print(f"[moe_emul] MODE={_MODE} active={active.numel()} E={E} T={T} "
                      f"model={model_dim} inter={w13_u8.shape[1]//2}", flush=True)
                STATE["logged"] = True
            return AiterRunnerOutput(hidden_states=out.to(out_dtype))

        AiterRunnerCore.run = _run_emul
        print(f"[moe_emul] installed, MODE={_MODE} (shuffles neutered, run() replaced)", flush=True)

    threading.Thread(target=_install, daemon=True).start()
