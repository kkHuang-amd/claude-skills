import os, json, torch
_done = set()
def _m(t):
    if t is None: return None
    try: return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype), contig=bool(t.is_contiguous()))
    except Exception: return dict(repr=str(type(t)))
def maybe_dump(engine, hidden, w13, w13_scale, w2, w2_scale,
               a1_scale, a2_scale, topk_ids, topk_weights,
               bias1, bias2, expert_mask, kwargs):
    d = os.environ.get("DUMP_MOE1_DIR")
    if not d or engine in _done: return
    try:
        if hidden.shape[0] != 64: return
    except Exception:
        return
    flag = os.path.join(d, ".done_" + engine)
    if os.path.exists(flag): _done.add(engine); return
    os.makedirs(d, exist_ok=True)
    meta = dict(engine=engine, hidden=_m(hidden), w13=_m(w13), w13_scale=_m(w13_scale),
                w2=_m(w2), w2_scale=_m(w2_scale), a1_scale=_m(a1_scale), a2_scale=_m(a2_scale),
                topk_ids=_m(topk_ids), topk_weights=_m(topk_weights),
                bias1=_m(bias1), bias2=_m(bias2), expert_mask=_m(expert_mask), kwargs=kwargs)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2, default=str)
    def cpu(t): return None if t is None else t.detach().cpu()
    try:
        torch.save(dict(
            hidden=cpu(hidden), w13=cpu(w13), w13_scale=cpu(w13_scale),
            w2=cpu(w2), w2_scale=cpu(w2_scale), a1_scale=cpu(a1_scale), a2_scale=cpu(a2_scale),
            topk_ids=cpu(topk_ids), topk_weights=cpu(topk_weights),
            bias1=cpu(bias1), bias2=cpu(bias2), expert_mask=cpu(expert_mask), kwargs=kwargs,
        ), os.path.join(d, "full.pt"))
    except Exception as e:
        print("[DUMP_MOE1] save err", e, flush=True)
    print("[DUMP_MOE1]", json.dumps(meta, default=str), flush=True)
    open(flag, "w").close(); _done.add(engine)
