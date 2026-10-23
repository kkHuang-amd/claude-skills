"""Probe the wave32 lane layout of gfx1250 v_wmma_f32_16x16x4_f32 via FlyDSL raw rocdl.

Each lane passes a[2] (A operand), b[2] (B operand) straight from memory; c starts at 0. Feeding
one-hot A/B patterns and reading back the 8 result VGPRs per lane reveals which (row, k) each lane's
A slot holds, which (k, col) each B slot holds, and which (row, col) each result VGPR holds.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import gpu, range_constexpr, rocdl


def _io():
    def load(addr, off, vec):
        pt = fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 4 * vec)
        view = rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(pt, fx.Int64(addr)), fx.make_layout((vec, 1), (1, 1))))
        )
        frag = fx.make_rmem_tensor(vec, fx.Float32)
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(32 * vec), fx.Float32), fx.slice(view, (None, off)), frag)
        return fx.Vector(frag.load())

    def store(addr, off, vec_val, vec):
        pt = fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 4 * vec)
        view = rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(pt, fx.Int64(addr)), fx.make_layout((vec, 1), (1, 1))))
        )
        frag = fx.make_rmem_tensor(vec, fx.Float32)
        frag.store(vec_val)
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(32 * vec), fx.Float32), frag, fx.slice(view, (None, off)))

    return load, store


@flyc.kernel(name="wmma_f32_16x16x4_probe", known_block_size=[32, 1, 1])
def probe(a_ptr: fx.Int64, b_ptr: fx.Int64, c_ptr: fx.Int64):
    load, store = _io()
    lane = fx.Int32(gpu.thread_id("x"))
    a = load(a_ptr, lane * fx.Int32(2), 2)
    b = load(b_ptr, lane * fx.Int32(2), 2)
    c0 = fx.Vector.from_elements([fx.Float32(0.0) for _ in range(8)], fx.Float32)
    d = rocdl_raw.wmma_f32_16x16x4_f32(fx.T.vec(8, fx.T.f32), a.ir_value(), b.ir_value(), c0.ir_value()).result
    dv = fx.Vector(d)
    for h in range_constexpr(2):  # buffer stores top out at 128 bits
        store(c_ptr, lane * fx.Int32(8) + fx.Int32(4 * h),
              fx.Vector.from_elements([dv[4 * h + j] for j in range(4)], fx.Float32), 4)


@flyc.jit
def launch(a_ptr: fx.Int64, b_ptr: fx.Int64, c_ptr: fx.Int64, stream: fx.Stream):
    probe(a_ptr, b_ptr, c_ptr).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def run(a_lane, b_lane):
    c = torch.zeros(32, 8, device="cuda")
    launch(a_lane.data_ptr(), b_lane.data_ptr(), c.data_ptr(), fx.Stream(torch.cuda.current_stream()))
    torch.cuda.synchronize()
    return c.cpu()


if __name__ == "__main__":
    # 1) B = all ones, A slot (lane, s) = unique id -> each result = sum over k of A[row, k];
    #    use powers so the k contributions of a row can be decoded: A value = 1000**s * (lane+1)
    #    is too wide for fp32, so instead probe one slot at a time.
    ones = torch.ones(32, 2, device="cuda")
    a_map = {}
    for lane in range(32):
        for s in range(2):
            a = torch.zeros(32, 2, device="cuda"); a[lane, s] = 1.0
            c = run(a, ones)
            rows = sorted({(l, v) for l in range(32) for v in range(8) if c[l, v] != 0})
            a_map[(lane, s)] = rows  # result slots touched -> identifies the row
    b_map = {}
    for lane in range(32):
        for s in range(2):
            b = torch.zeros(32, 2, device="cuda"); b[lane, s] = 1.0
            c = run(ones, b)
            b_map[(lane, s)] = sorted({(l, v) for l in range(32) for v in range(8) if c[l, v] != 0})
    # A slot -> row r touches result slots of row r across all 16 cols; B slot -> col n touches all rows of col n.
    # Recover k by pairing: A(lane,s) and B(lane',s') share k iff A one-hot x B one-hot gives a nonzero result.
    def first(x):
        return x[0] if x else None
    print("A lane,slot -> first result (lane,vgpr) touched, count:")
    for lane in (0, 1, 15, 16, 17, 31):
        print("  A", lane, [(s, first(a_map[(lane, s)]), len(a_map[(lane, s)])) for s in range(2)])
    print("B lane,slot -> first result (lane,vgpr) touched, count:")
    for lane in (0, 1, 15, 16, 17, 31):
        print("  B", lane, [(s, first(b_map[(lane, s)]), len(b_map[(lane, s)])) for s in range(2)])
    # k pairing: which B (lane,slot) combine with A(0,0) / A(0,1) / A(16,0) / A(16,1)
    for (al, as_) in ((0, 0), (0, 1), (16, 0), (16, 1)):
        a = torch.zeros(32, 2, device="cuda"); a[al, as_] = 1.0
        hits = []
        for bl in (0, 16):
            for bs in range(2):
                b = torch.zeros(32, 2, device="cuda"); b[bl, bs] = 1.0
                if run(a, b).abs().sum() != 0:
                    hits.append((bl, bs))
        print(f"  A({al},{as_}) shares k with B {hits}")
    # result layout: A row r (one-hot at the A slot that maps to row r) x B col n
    a = torch.zeros(32, 2, device="cuda"); a[3, 0] = 1.0
    b = torch.zeros(32, 2, device="cuda"); b[5, 0] = 1.0
    c = run(a, b)
    print("A(3,0) x B(5,0) -> result at", [(l, v) for l in range(32) for v in range(8) if c[l, v] != 0])
    a = torch.zeros(32, 2, device="cuda"); a[19, 0] = 1.0
    b = torch.zeros(32, 2, device="cuda"); b[21, 0] = 1.0
    c = run(a, b)
    print("A(19,0) x B(21,0) -> result at", [(l, v) for l in range(32) for v in range(8) if c[l, v] != 0])
    a = torch.zeros(32, 2, device="cuda"); a[11, 0] = 1.0
    b = torch.zeros(32, 2, device="cuda"); b[2, 0] = 1.0
    c = run(a, b)
    print("A(11,0) x B(2,0) -> result at", [(l, v) for l in range(32) for v in range(8) if c[l, v] != 0])
