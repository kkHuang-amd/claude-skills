"""E34 probe: per-op numerical check of every NON-MoE op vs a pure-torch fp32 reference,
same-node (no TP cross-node floor). Localizes which gfx1250 platform op carries the
0.95(gfx950)-vs-0.80(gfx1250) accuracy gap that E31-E33 proved is NON-MoE.

E22 only checked attention "vs torch". This extends to the ops E22 never covered:
  - rmsnorm  : RMSNorm.forward           vs RMSNorm.forward_native (fp32 var)
  - rope     : RotaryEmbedding.forward   vs .forward_native
  - linear   : every {Row,Column,QKV,Merged,Replicated}ParallelLinear.forward
               vs x.float() @ W.float().t() (+bias)  (covers o_proj, q/kv proj, dense MLP)
  - lm_head  : LogitsProcessor._get_logits result vs hidden.float() @ lm_head.float()
  - sampling : Sampler.forward greedy argmax(bf16/stored) vs argmax(fp32) + top1-top2 margin

Each hook: run the real op, recompute the fp32 reference from the SAME inputs, record
rel_l2 (mean/max) aggregated by op-type (+ a shape/out-dim key so o_proj vs gate_up vs
lm_head are separable). Robust: every hook is try/except'd; a probe failure never breaks
the model. Gated by SGLANG_NONMOE_PROBE=1. Dumps /tmp/nonmoe_probe.json (overwritten
every _DUMP_EVERY records so the file is always current).

Launch EAGER (Python hooks bypassed under cuda-graph replay), e.g. add to run_ds-r1.sh:
  PYTHONPATH=/tmp/nonmoe:$PYTHONPATH SGLANG_NONMOE_PROBE=1  ... --disable-cuda-graph
then send a normal prompt (a bit of prefill + some decode) and read /tmp/nonmoe_probe.json.
"""
import os

if os.environ.get("SGLANG_NONMOE_PROBE", "0") in ("1", "true", "True"):
    import threading
    import time
    import json

    OUT = os.environ.get("SGLANG_NONMOE_PROBE_OUT", "/tmp/nonmoe_probe.json")
    _DUMP_EVERY = int(os.environ.get("SGLANG_NONMOE_PROBE_DUMP_EVERY", "300"))
    _MAX_PER_KEY = int(os.environ.get("SGLANG_NONMOE_PROBE_MAX_PER_KEY", "400"))

    # agg[key] = [sum_rel, count, max_rel]
    AGG = {}
    STATE = {"records": 0}

    def _rec(key, rel):
        a = AGG.setdefault(key, [0.0, 0, 0.0])
        if a[1] >= _MAX_PER_KEY:
            return
        a[0] += rel
        a[1] += 1
        if rel > a[2]:
            a[2] = rel
        STATE["records"] += 1
        if STATE["records"] % _DUMP_EVERY == 0:
            _dump()

    def _dump():
        rows = []
        for k, (s, n, mx) in AGG.items():
            rows.append({"op": k, "mean_rel_l2": s / max(1, n), "max_rel_l2": mx, "n": n})
        rows.sort(key=lambda r: -r["mean_rel_l2"])
        try:
            json.dump({"rows": rows, "total_records": STATE["records"]},
                      open(OUT, "w"), indent=2)
        except Exception:
            pass

    def _install():
        import torch

        def _capturing():
            try:
                return torch.cuda.is_current_stream_capturing()
            except Exception:
                return False

        def _rel(real, ref):
            r = real.detach().float()
            f = ref.detach().float()
            if r.shape != f.shape:
                return None
            d = torch.norm(r - f).item()
            n = torch.norm(f).item()
            return d / (n + 1e-8)

        def _first_tensor(x):
            if torch.is_tensor(x):
                return x
            if isinstance(x, (tuple, list)):
                for e in x:
                    if torch.is_tensor(e):
                        return e
            return None

        # wait for the model modules to be importable
        for _ in range(600):
            try:
                from sglang.srt.layers.layernorm import RMSNorm
                from sglang.srt.layers import rotary_embedding as _re
                from sglang.srt.layers import linear as _lin
                from sglang.srt.layers.logits_processor import LogitsProcessor
                from sglang.srt.layers.sampler import Sampler
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("[nonmoe_probe] could not import model layers; abort", flush=True)
            return

        installed = []

        # ---- 1. RMSNorm: real forward vs forward_native (fp32 variance) ----
        try:
            _rms_orig = RMSNorm.forward

            def rms_hooked(self, *args, **kwargs):
                # snapshot inputs BEFORE the real op (fused add-rmsnorm mutates residual in place)
                do = not _capturing()
                if do:
                    try:
                        cargs = [a.clone() if torch.is_tensor(a) else a for a in args]
                        ckw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                    except Exception:
                        do = False
                out = _rms_orig(self, *args, **kwargs)
                if do:
                    try:
                        ref = self.forward_native(*cargs, **ckw)
                        ro, fo = _first_tensor(out), _first_tensor(ref)
                        if ro is not None and fo is not None:
                            rl = _rel(ro, fo)
                            if rl is not None:
                                d = int(getattr(self, "hidden_size", ro.shape[-1]))
                                _rec(f"rmsnorm[d={d}]", rl)
                    except Exception:
                        pass
                return out

            RMSNorm.forward = rms_hooked
            installed.append("rmsnorm")
        except Exception as e:
            print("[nonmoe_probe] rmsnorm hook failed:", repr(e), flush=True)

        # ---- 2. RoPE: real forward vs forward_native ----
        try:
            RE = _re.RotaryEmbedding
            _re_orig = RE.forward

            def rope_hooked(self, *args, **kwargs):
                # snapshot inputs BEFORE the real op (rope kernels write q/k in place)
                do = not _capturing()
                if do:
                    try:
                        cargs = [a.clone() if torch.is_tensor(a) else a for a in args]
                        ckw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                    except Exception:
                        do = False
                out = _re_orig(self, *args, **kwargs)
                if do:
                    try:
                        ref = self.forward_native(*cargs, **ckw)
                        if isinstance(out, (tuple, list)) and isinstance(ref, (tuple, list)):
                            for i, tag in enumerate(("q", "k")):
                                if i < len(out) and i < len(ref) and torch.is_tensor(out[i]):
                                    rl = _rel(out[i], ref[i])
                                    if rl is not None:
                                        _rec(f"rope[{tag}]", rl)
                        else:
                            ro, fo = _first_tensor(out), _first_tensor(ref)
                            if ro is not None and fo is not None:
                                rl = _rel(ro, fo)
                                if rl is not None:
                                    _rec("rope", rl)
                    except Exception:
                        pass
                return out

            RE.forward = rope_hooked
            installed.append("rope")
        except Exception as e:
            print("[nonmoe_probe] rope hook failed:", repr(e), flush=True)

        # ---- 3. Linear modules: real forward vs fp32 x@W.t()+bias ----
        _lin_classes = []
        for _cn in ("RowParallelLinear", "ColumnParallelLinear", "QKVParallelLinear",
                    "MergedColumnParallelLinear", "ReplicatedLinear"):
            c = getattr(_lin, _cn, None)
            if c is not None:
                _lin_classes.append((_cn, c))

        def _make_lin_hook(cname, orig):
            def lin_hooked(self, x, *args, **kwargs):
                out = orig(self, x, *args, **kwargs)
                if not _capturing():
                    try:
                        W = getattr(self, "weight", None)
                        o = _first_tensor(out)
                        if W is not None and torch.is_tensor(x) and o is not None:
                            # fp32 reference (TP1: no all-reduce needed)
                            ref = x.detach().float() @ W.detach().float().t()
                            b = getattr(self, "bias", None)
                            if b is not None and torch.is_tensor(b):
                                ref = ref + b.detach().float()
                            if ref.shape == o.shape:
                                rl = _rel(o, ref)
                                if rl is not None:
                                    _rec(f"linear[{cname}:out{W.shape[0]}]", rl)
                    except Exception:
                        pass
                return out
            return lin_hooked

        for cname, c in _lin_classes:
            try:
                if "forward" in c.__dict__:  # only hook classes that define their own forward
                    orig = c.forward
                    c.forward = _make_lin_hook(cname, orig)
                    installed.append(f"linear:{cname}")
            except Exception as e:
                print(f"[nonmoe_probe] linear hook {cname} failed:", repr(e), flush=True)

        # ---- 4. lm_head via LogitsProcessor._get_logits ----
        try:
            _gl_orig = LogitsProcessor._get_logits

            def gl_hooked(self, hidden_states, lm_head, logits_metadata, *a, **k):
                out = _gl_orig(self, hidden_states, lm_head, logits_metadata, *a, **k)
                if not _capturing():
                    try:
                        W = getattr(lm_head, "weight", None)
                        o = _first_tensor(out)
                        h = _first_tensor(hidden_states)
                        if W is not None and h is not None and o is not None:
                            ref = h.detach().float() @ W.detach().float().t()
                            # o may be padded to vocab_size or gathered; compare overlap
                            v = min(ref.shape[-1], o.shape[-1])
                            if ref.shape[0] == o.shape[0]:
                                rl = _rel(o[..., :v], ref[..., :v])
                                if rl is not None:
                                    _rec(f"lm_head[vocab~{o.shape[-1]}]", rl)
                    except Exception:
                        pass
                return out

            LogitsProcessor._get_logits = gl_hooked
            installed.append("lm_head")
        except Exception as e:
            print("[nonmoe_probe] lm_head hook failed:", repr(e), flush=True)

        # ---- 5. Sampler: greedy argmax stored-vs-fp32 agreement + top1-top2 margin ----
        try:
            _samp_orig = Sampler.forward

            def samp_hooked(self, logits_output, *a, **k):
                out = _samp_orig(self, logits_output, *a, **k)
                if not _capturing():
                    try:
                        lg = getattr(logits_output, "next_token_logits", None)
                        if torch.is_tensor(lg):
                            f = lg.detach().float()
                            # argmax on the tensor's native precision vs fp32 recast:
                            am_native = lg.detach().argmax(dim=-1)
                            am_fp32 = f.argmax(dim=-1)
                            agree = (am_native == am_fp32).float().mean().item()
                            _rec("sampling[argmax_disagree]", 1.0 - agree)
                            # top1-top2 margin (fragility of greedy): smaller => tie-prone
                            top2 = torch.topk(f, 2, dim=-1).values
                            margin = (top2[..., 0] - top2[..., 1]).mean().item()
                            m = AGG.setdefault("sampling[top1_top2_margin]", [0.0, 0, 0.0])
                            m[0] += margin
                            m[1] += 1
                            m[2] = max(m[2], margin)
                    except Exception:
                        pass
                return out

            Sampler.forward = samp_hooked
            installed.append("sampling")
        except Exception as e:
            print("[nonmoe_probe] sampler hook failed:", repr(e), flush=True)

        print(f"[nonmoe_probe] installed: {installed}", flush=True)

    threading.Thread(target=_install, daemon=True).start()
