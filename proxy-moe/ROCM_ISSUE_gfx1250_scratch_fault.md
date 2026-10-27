Owner node: ctheliosp-zuse-f19-14

# DRAFT (not filed): gfx1250 GPU page fault on scratch (register-spill) stores under concurrent dispatches

Status: draft for the user to review and file. Nothing below has been posted anywhere.

## Summary

On gfx1250 (MI455, one partition = 96 CUs, ROCm 10.1.0), a Triton/Gluon GEMM kernel that spills VGPRs to scratch
intermittently takes a GPU memory access fault ("Page not present or supervisor privilege") on its own
`scratch_store` instructions when it runs inside a multi-process serving workload. The same kernel never faults
when kernels are serialized (`AMD_SERIALIZE_KERNEL=3`), and an otherwise identical kernel compiled without
spills (0 scratch) never faults. The faulting address is not inside any allocation the process knows about
(all kernel arguments, model weights, KV cache, PyTorch allocator segments were logged and checked), which is
consistent with a scratch (private segment) backing problem rather than a bad pointer argument.

## Environment

- GPU: AMD MI455 (gfx1250), 8 partitions visible (4 sockets x 2); run on 2 partitions of one socket (TP2).
- ROCm 10.1.0 (TheRock wheels), amdgpu 7.1.0.31300009, kernel 6.16.1, AMDSMI 27.1.0.
- PyTorch 2.12.0+rocm10.1.0 (HIP 7.16.26385), Triton 3.8.0 (Gluon gfx1250 backend).
- aiter `kkHuang-amd/aiter@feat/proxy-model-gfx1250-1023` at `472f5cf39` (fault) / `2324075d0` (fixed config).
- Process env from the container image: `HSA_NO_SCRATCH_RECLAIM=1`, `HIP_FORCE_DEV_KERNARG=1`.
  (`HIP_FORCE_DEV_KERNARG=0` was tried: still faults. `HSA_NO_SCRATCH_RECLAIM=0` was NOT tried.)

## Faulting kernel

`_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_128_BLOCK_N_128_BLOCK_K_64_NUM_BUFFERS_4_LAYOUT_TN_USE_ACTIVATION_0_ADD_BIAS_0`
(aiter `aiter/ops/triton/_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16.py`, bandwidth-bound variant, num_warps=4,
`waves_per_eu=4`), bf16 x bf16 GEMM with M up to 8192 (e.g. M=8192, N=16384, K=8192; grid 8192 workgroups x 128 lanes).

Code-object stats from the Triton cache:

| config (bw, 128x128x64, 4 warps) | scratch (private_segment_fixed_size) | VGPRs | VGPR spills | faults? |
|---|---|---|---|---|
| NUM_BUFFERS=4, waves_per_eu=4 | 92 B | 256 | 22 | yes |
| NUM_BUFFERS=6, waves_per_eu=1 | 0 | 841 | 0 | never |
| NUM_BUFFERS=4, waves_per_eu=1 | 0 | 593 | 0 | never |
| compute_bound 128x128x64 NB2 | 0 | 262 | 0 | never |

HSA queue-error dump of the faulting dispatch: `cluster_count=[8192, 1, 1], workgroup=[128, 1, 1], private_seg_size=188,
group_seg_size=147424` (note: 188 in the packet vs 92 B in the cached code-object metadata; not explained yet).

## rocgdb at the fault

rocgdb attached (batch) to the serving process before the workload; on the fault:

```
Thread 283 "_gemm_a16w1-f364" received signal SIGSEGV, Segmentation fault.
* 283  AMDGPU Wave 3:3:6:3258 (1,0,0)/0 "_gemm_a16w1-f364"   0x00007fb36abaa6b8
* 6    AMDGPU Dispatch 3:3:6 (PKID 444) [1048576,1,1] [128,1,1] B|Aa|Ra  _gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_128...
```

All stopped waves are at the same PC. The instructions right before the reported PC (memory faults are reported
imprecisely, after the instruction):

```
... v_cmp_gt_i32 / v_add_lshl_u32 ...
s_wait_alu
scratch_store_b32 off, v0, off offset:80 nv
s_wait_xcnt 0
s_wait_alu
=> v_cndmask_b32 ...            <- reported PC
...
scratch_store_b32 off, v0, off offset:44 nv
```

Full dumps: `/shared_nfs/kk/results/proxy_4l/rocgdb_f19-14_1027/rocgdb_*.txt` (node-local NFS; attach when filing).

## What was ruled out

- Out-of-bounds tensor access: fault addresses (e.g. 0x7f5fe1fc6000, 0x7f6783047000, 0x7fa03a1a9000) are 88-370 GiB
  away from every A/B/C range of every launch of this kernel, and outside all model weights, KV-cache buffers and
  PyTorch caching-allocator segments of the faulting process.
- M-tail over-read at a 2 MiB mapping boundary (standalone test with `PYTORCH_NO_HIP_MEMORY_CACHING=1`): no fault.
- Overlap scheduler, custom all-reduce, `HIP_FORCE_DEV_KERNARG` 0/1: still faults.
- Standalone microbenchmark of the same kernel and shapes (incl. the exact M sequence of the workload, 5 repeats,
  no caching allocator): never faults. Only the multi-process serving workload reproduces it.

## Reproducer (serving workload, deterministic: 6/6 crashed before the fix)

SGLang server, TP2 on two partitions of one socket, dummy-weight proxy MoE model (4 layers), aiter backend kernels:

```bash
# aiter at 472f5cf39 (or any DEFAULT with M_LEQ_8192 = bw 128x128x64 NB4 waves_per_eu=4)
HIP_VISIBLE_DEVICES=2,3 SGLANG_USE_AITER=1 python3 -m sglang.launch_server --model-path <proxy_4l> \
  --tp-size 2 --ep-size 2 --kv-cache-dtype fp8_e4m3 --context-length 32768 --mem-fraction-static 0.90 \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 --chunked-prefill-size 8192 --attention-backend triton &
python3 <proxy_moe_sglang>/tools/smoke_test.py --port 30000          # short, 6000-token and 16x512 requests
python3 -m sglang.bench_serving --backend sglang --dataset-name random-ids --random-input-len 1024 \
  --random-output-len 1024 --random-range-ratio 1.0 --num-prompts 128 --max-concurrency 16
```

The fault hits during the bench prefill after the smoke test. `AMD_SERIALIZE_KERNEL=3` makes it disappear.

## Workaround (applied)

Make every kernel config scratch-free: aiter `2324075d0` changes the gfx1250 gluon a16w16 DEFAULT buckets that
spilled (M_LEQ_256, M_LEQ_2048, M_LEQ_8192, any). Validated: 3 runs of the reproducer, 0 faults; performance equal or
better (M=256 up to -37%).

## Questions for ROCm

1. Is there a known issue with scratch (private segment) backing on gfx1250 when many concurrent dispatches on
   different queues/processes need scratch (allocation growth / reclaim race)? Does `HSA_NO_SCRATCH_RECLAIM` matter?
2. Why does the dispatch packet carry `private_seg_size=188` while the code object reports 92 B?
3. Is a scratch-spilling kernel expected to work with `waves_per_eu=4` at 8192 workgroups x 128 lanes on one partition?
