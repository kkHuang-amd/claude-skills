import torch, triton, triton.language as tl

@triton.jit
def k(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M); n = tl.arange(0, N); kk = tl.arange(0, K)
    x = tl.load(a + m[:, None] * K + kk[None, :])
    y = tl.load(b + kk[:, None] * N + n[None, :])
    tl.store(c + m[:, None] * N + n[None, :], tl.dot(x, y))

print(torch.cuda.get_device_properties(0).gcnArchName, "triton", triton.__version__)
torch.manual_seed(0)
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    for K in (64, 128, 256, 512):
        M = N = 32
        a = torch.randn(M, K, device="cuda").to(dt)
        b = torch.randn(K, N, device="cuda").to(dt)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        k[(1,)](a, b, c, M, N, K)
        ref = a.float() @ b.float()
        err = ((c - ref).abs().max() / ref.abs().max()).item()
        print(f"{str(dt):22s} K={K:4d} max|c|={c.abs().max().item():.3e} rel_err={err:.2e} {'OK' if err < 1e-3 else 'BAD'}")
