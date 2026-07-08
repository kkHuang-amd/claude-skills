"""E22 probe: compare the gfx1250 Triton MLA attention kernels (decode + prefill)
against a pure-torch fp32 reference, per layer, to localize any numerical divergence.

Usage (no repo edits):
  place this file as `sitecustomize.py` in a dir, e.g. /tmp/attnhook/sitecustomize.py
  launch the server EAGER (cuda-graph replay bypasses the Python hook) with:
    PYTHONPATH=/tmp/attnhook:$PYTHONPATH SGLANG_ATTN_PROBE=1 bash run_ds-r1.sh  # + --disable-cuda-graph
  send a request that does decode (>~66 tokens) and/or several distinct-prefix
  prefills; results dumped to /tmp/attn_probe.json (decode) and
  /tmp/attn_probe_prefill.json (prefill MHA).

Result (2026-07-08, gfx1250): decode + prefill both rel_l2 ~0.0013-0.0020 across all
61 layers (bf16 noise, no outlier) => attention kernels are numerically CORRECT.
"""
import os

if os.environ.get("SGLANG_ATTN_PROBE", "0") in ("1", "true", "True"):
    import threading
    import time
    import json

    STATE = {"per_layer": {}, "calls": 0, "max_calls": 4000, "done": False,
             "dump": "/tmp/attn_probe.json"}

    def _install():
        import torch
        for _ in range(600):
            try:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
                break
            except Exception:
                time.sleep(0.5)
        else:
            return

        _orig = TritonAttnBackend.forward_decode

        def _ref_rel(self, q, layer, o):
            try:
                H = layer.tp_q_head_num
                Dq = layer.qk_head_dim
                Dv = layer.v_head_dim
                meta = self.forward_metadata
                kv_indptr = meta.kv_indptr
                kv_indices = meta.kv_indices
                if kv_indptr is None or kv_indices is None:
                    return None
                kbuf = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
                vbuf = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
                q3 = q.reshape(-1, H, Dq).float()
                o3 = o.reshape(-1, H, Dv).float()
                B = q3.shape[0]
                scale = layer.scaling
                kb = kbuf.reshape(kbuf.shape[0], -1).float()
                vb = vbuf.reshape(vbuf.shape[0], -1).float()
                rels = []
                for b in range(B):
                    lo = int(kv_indptr[b].item()); hi = int(kv_indptr[b + 1].item())
                    if hi <= lo:
                        continue
                    idx = kv_indices[lo:hi].long()
                    K = kb[idx]; V = vb[idx][:, :Dv]; qb = q3[b]
                    scores = (qb @ K.t()) * scale
                    attn = torch.softmax(scores, dim=-1)
                    ref = attn @ V
                    rels.append((torch.norm(o3[b] - ref) / (torch.norm(ref) + 1e-6)).item())
                return sum(rels) / len(rels) if rels else None
            except Exception:
                return None

        def hooked(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
            o = _orig(self, q, k, v, layer, forward_batch, save_kv_cache, sinks)
            if not STATE["done"]:
                try:
                    if torch.cuda.is_current_stream_capturing():
                        return o
                except Exception:
                    pass
                r = _ref_rel(self, q, layer, o)
                if r is not None:
                    acc = STATE["per_layer"].setdefault(int(layer.layer_id), [0.0, 0])
                    acc[0] += r; acc[1] += 1; STATE["calls"] += 1
                    if STATE["calls"] >= STATE["max_calls"]:
                        _dump(); STATE["done"] = True
            return o

        def _dump():
            rows = [{"layer": l, "mean_rel_l2": s / max(1, n), "n": n}
                    for l, (s, n) in sorted(STATE["per_layer"].items())]
            mm = [x["mean_rel_l2"] for x in rows]
            json.dump({"rows": rows, "overall_mean": sum(mm) / max(1, len(mm)),
                       "worst": sorted(rows, key=lambda x: -x["mean_rel_l2"])[:8],
                       "total_calls": STATE["calls"]}, open(STATE["dump"], "w"), indent=2)
            print(f"[attn_probe] dumped {STATE['dump']}", flush=True)

        TritonAttnBackend.forward_decode = hooked

        # ---- prefill (MHA cold-prefill) ----
        PRE = {"per_layer": {}, "calls": 0, "max": 200, "done": False,
               "dump": "/tmp/attn_probe_prefill.json"}
        _orig_ext = TritonAttnBackend.forward_extend

        def _ref_mha(q, k, v, layer, o):
            try:
                H = layer.tp_q_head_num; Dq = layer.qk_head_dim; Dv = layer.v_head_dim
                q3 = q.reshape(-1, H, Dq).float(); k3 = k.reshape(-1, H, Dq).float()
                v3 = v.reshape(-1, H, Dv).float(); o3 = o.reshape(-1, H, Dv).float()
                T = q3.shape[0]
                if k3.shape[0] != T or v3.shape[0] != T:
                    return None  # has prefix / not a clean single block
                scale = layer.scaling
                scores = torch.einsum("thd,shd->hts", q3, k3) * scale
                mask = torch.triu(torch.ones(T, T, device=q3.device, dtype=torch.bool), 1)
                scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                ref = torch.einsum("hts,shd->thd", attn, v3)
                return (torch.norm(o3 - ref) / (torch.norm(ref) + 1e-6)).item()
            except Exception:
                return None

        def hooked_ext(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
            o = _orig_ext(self, q, k, v, layer, forward_batch, save_kv_cache, sinks)
            if not PRE["done"] and layer.v_head_dim == 128:
                try:
                    if torch.cuda.is_current_stream_capturing():
                        return o
                except Exception:
                    pass
                r = _ref_mha(q, k, v, layer, o)
                if r is not None:
                    acc = PRE["per_layer"].setdefault(int(layer.layer_id), [0.0, 0])
                    acc[0] += r; acc[1] += 1; PRE["calls"] += 1
                    if PRE["calls"] >= PRE["max"]:
                        rows = [{"layer": l, "mean_rel_l2": s / max(1, n), "n": n}
                                for l, (s, n) in sorted(PRE["per_layer"].items())]
                        mm = [x["mean_rel_l2"] for x in rows]
                        json.dump({"rows": rows, "overall_mean": sum(mm) / max(1, len(mm)),
                                   "worst": sorted(rows, key=lambda x: -x["mean_rel_l2"])[:8]},
                                  open(PRE["dump"], "w"), indent=2)
                        print("[attn_probe] PREFILL dumped", flush=True)
                        PRE["done"] = True
            return o

        TritonAttnBackend.forward_extend = hooked_ext
        print("[attn_probe] installed forward_decode + forward_extend hooks", flush=True)

    threading.Thread(target=_install, daemon=True).start()
