"""UG time vs union size at fixed M (1 GPU, graph): does the last partial round of pair-tasks cost?
FOLD_RQ pairs: nu * 16 pairs over 256 CTAs -> rounds = ceil(nu / 16)."""
import sys, torch
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
E.I, E.E_LOCAL = 512, 512
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import routed  # noqa: E402
m = int(sys.argv[1]); unions = [int(v) for v in sys.argv[2].split(",")]
dev = torch.device("cuda", 0)
w1, w2, w1s, w2s, _, cases = E.build([m])
x, wt, ids0 = cases[m]
scratch = torch.zeros(routed.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
out = torch.empty(m, routed.HIDDEN, dtype=torch.bfloat16, device=dev)
launch = routed.build_routed(m)
k = ids0.shape[1]
for nu in unions:
    pool = torch.randperm(512, generator=torch.Generator().manual_seed(nu))[:nu]
    flat = pool.repeat((m * k + nu - 1) // nu)[:m * k].view(m, k)  # every pool expert used, distinct a row
    ids = flat.to(ids0.dtype).to(dev).contiguous()
    assert int(torch.unique(ids).numel()) == nu and all(len(set(r.tolist())) == k for r in ids)
    t = {}
    for skip in (0, 1):
        args = (x.data_ptr(), ids.data_ptr(), wt.data_ptr(), w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
                w2s.data_ptr(), out.data_ptr(), scratch.data_ptr(), epoch.data_ptr(), 0, skip)
        launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1
        g.replay(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(30):
            g.replay()
        e.record(); torch.cuda.synchronize()
        t[skip] = s.elapsed_time(e) * 1e3 / 30
    ug = t[0] - t[1]
    print(f"M={m} union={nu:3d} rounds={(nu + 15) // 16}: full {t[0]:.1f} UG {ug:.1f} us "
          f"({nu * 8.65e6 / ug / 1e6:.2f} TB/s, {ug / nu:.2f} us/expert)", flush=True)
