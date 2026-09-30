# Engram host-table fault check (see SKILL.md History 09-29): after cudaHostRegister on an mmap, is the
# device VA (hipHostGetDevicePointer) == host VA on this node? If same=False, any kernel given the host VA
# (unpatched engram_gather) hits "Memory access fault"; fix = patches/sglang_local_engram_host_devptr_0001.patch.
# Env: WHICH=devptr (read via dev VA, must pass) | hostptr (read via host VA; faults iff same=False) | pinned (control).
# Run on a free GPU:  ulimit -c 0; HIP_VISIBLE_DEVICES=0 WHICH=devptr python3 hostreg_devptr_check.py
# Output: stdout only ("host 0x.. dev 0x.. same=..", "<WHICH> read ok sum N n N").
import ctypes, mmap, os, numpy as np, torch, triton, triton.language as tl
N = 1 << 30; WHICH = os.environ.get("WHICH", "devptr")  # devptr | hostptr | pinned
hip = ctypes.CDLL("libamdhip64.so")
@triton.jit
def k(p, ids, out):
    i = tl.program_id(0); pp = p.to(tl.int64).to(tl.pointer_type(tl.uint8)); tl.store(out + i, tl.load(pp + tl.load(ids + i)))
idx = torch.arange(0, N, 1 << 20, device="cuda"); out = torch.empty(idx.numel(), dtype=torch.uint8, device="cuda")
if WHICH == "pinned":
    t = torch.ones(N, dtype=torch.uint8).pin_memory(); ptr = t.data_ptr()
else:
    mm = mmap.mmap(-1, N, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    t = torch.frombuffer(mm, dtype=torch.uint8); np.frombuffer(mm, dtype=np.uint8)[:] = 1
    print("register err", int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), N, 0)))
    d = ctypes.c_void_p(); e = hip.hipHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(t.data_ptr()), 0)
    print(f"host 0x{t.data_ptr():x} dev 0x{(d.value or 0):x} err {e} same={d.value == t.data_ptr()}", flush=True)
    ptr = d.value if WHICH == "devptr" else t.data_ptr()
k[(idx.numel(),)](ptr, idx, out); torch.cuda.synchronize(); print(WHICH, "read ok sum", int(out.sum()), "n", idx.numel(), flush=True)
