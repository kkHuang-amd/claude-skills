import os, json, torch

# Sentinel-file gated (robust to env not propagating to worker procs).
OUT = "/workspace/router_dump"
ENABLE = OUT + "/.enable"

# engine -> {layer_id: {field: tensor/meta}}
_state = {}
_done = {}

def enabled():
    return os.path.exists(ENABLE)

def _cpu(t):
    return None if t is None else t.detach().float().cpu()

def record(engine, layer_id, *, hidden=None, router_logits=None,
           topk_ids=None, topk_weights=None, correction_bias=None, meta=None):
    """Merge-style recorder. Captures one full decode pass (T==64) of MoE routing.
    Flushes pass-1 to disk when a non-hash layer's `hidden` is recorded a 2nd time
    (i.e. the next decode step begins)."""
    d = OUT
    if not enabled() or _done.get(engine):
        return
    ref = hidden if hidden is not None else (
        topk_ids if topk_ids is not None else router_logits)
    try:
        if ref is None or ref.shape[0] != 64:
            return
    except Exception:
        return

    st = _state.setdefault(engine, {})
    # wrap detection: re-recording hidden for a layer we already have -> new step
    if hidden is not None and layer_id in st and st[layer_id].get("hidden") is not None:
        _flush(engine, d)
        return

    cur = st.setdefault(layer_id, {})
    if hidden is not None:
        cur["hidden"] = _cpu(hidden)
    if router_logits is not None:
        cur["router_logits"] = _cpu(router_logits)
    if topk_ids is not None:
        cur["topk_ids"] = topk_ids.detach().cpu()
    if topk_weights is not None:
        cur["topk_weights"] = _cpu(topk_weights)
    if correction_bias is not None:
        cur["correction_bias"] = _cpu(correction_bias)
    if meta is not None:
        cur.setdefault("meta", {}).update(meta)

_counts = {}        # (engine, layer_id) -> list[int active experts per step]
_cw = [0]
COUNT = OUT + "/.count"

def _rank0():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

def count_step(engine, layer_id, topk_ids, n_routed=384):
    """Lightweight: append #active experts for this decode step (T==64). Sentinel .count."""
    if not os.path.exists(COUNT) or not _rank0():
        return
    try:
        if topk_ids.shape[0] != 64:
            return
    except Exception:
        return
    f = topk_ids.reshape(-1).long()
    f = f[f < n_routed]
    n = torch.unique(f).numel()
    _counts.setdefault((engine, int(layer_id)), []).append(n)
    tot = sum(len(v) for v in _counts.values())
    if tot - _cw[0] >= 20:
        _cw[0] = tot
        json.dump({f"{e}|{l}": v for (e, l), v in _counts.items()},
                  open(os.path.join(OUT, f"counts_{engine}.json"), "w"))

def _flush(engine, d):
    os.makedirs(d, exist_ok=True)
    st = _state.get(engine, {})
    torch.save(st, os.path.join(d, f"router_{engine}.pt"))
    summary = {"engine": engine, "n_layers": len(st),
               "layers": sorted(int(k) for k in st.keys()),
               "fields": {int(k): sorted(v.keys()) for k, v in st.items()}}
    json.dump(summary, open(os.path.join(d, f"router_{engine}.json"), "w"),
              indent=2, default=str)
    print(f"[ROUTER_DUMP] {engine} flushed {len(st)} layers -> {d}", flush=True)
    _done[engine] = True
