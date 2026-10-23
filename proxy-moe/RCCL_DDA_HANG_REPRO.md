Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# RCCL DDA all-reduce hang on gfx1250 (SGLang proxy MoE, TP8, `NCCL_CUMEM_ENABLE=1`)

## Summary

On 8x MI455 (gfx1250), an SGLang server with TP8 EP8 and `NCCL_CUMEM_ENABLE=1` hangs about half the time at startup. It hangs in the **first warmup prefill**, never becomes ready, and the warmup request times out after 600 s.

- **Where it hangs:** every rank's GPU is stuck in RCCL `dda::common::ddaAllReduceFlatLL<__hip_bfloat16, 8>` (rocgdb).
- **Not a collective mismatch:** all 8 ranks enqueued the identical collective sequence (`NCCL_DEBUG_SUBSYS=COLL`).
- **RCCL versions:** it happens on the RCCL bundled in the image (2.30.7) and on RCCL 2.32.3 (rocm-systems `4ca8623`, sglang PR #45576).
- **Workaround:** `RCCL_DDA_ENABLE=0` (6/6 clean starts).
- **Why it matters:** without `NCCL_CUMEM_ENABLE=1`, SGLang sets `NCCL_CUMEM_ENABLE=0`. RCCL then uses `P2P/IPC`, and an all-reduce of 64 KB takes 55-129 us instead of 11 us.

| RCCL | Env (plus the standard flags below) | Hung starts |
|---|---|---|
| 2.30.7 (image) | `NCCL_CUMEM_ENABLE=0` (SGLang default) | 0 of 9+ |
| 2.30.7 (image) | `NCCL_CUMEM_ENABLE=1` | 3 of 5 |
| 2.32.3 `4ca8623` | `NCCL_CUMEM_ENABLE=1` | 4 of 7 |
| 2.32.3 `4ca8623` | `NCCL_CUMEM_ENABLE=1 RCCL_DDA_LL_ONESHOT_THRESHOLD=0` | 1 of 4 (hung on start 4) |
| 2.32.3 `4ca8623` | `NCCL_CUMEM_ENABLE=1 RCCL_DDA_ENABLE=0` | **0 of 6** |

A standalone torchrun replay of the same collective sequence (25x AllReduce plus 1x AllGather, eager, 300 rounds) did **not** hang. The trigger needs something else in the server.

## Environment

- **Node:** 8x MI455 (gfx1250, B0; amd-smi REV_ID 0x00, CHIP_REV_ID 0x11), 1400 W socket power limit.
- **Image:** `rocm/sgl-dev:v0.5.22-rocm101-mi45x-20261020-test`.
  - ROCm 10.1.0, torch 2.12.0+rocm10.1.0, triton 3.8.0.
  - sglang `bad03dde2b`, aiter `2bb549102`.
  - The image env includes `NCCL_MIN_NCHANNELS=112`, which RCCL ignores (> 96).
- **Model:** the proxy MoE (`LagunaForCausalLM`), 12 layers with dummy MXFP4 weights.
  - hidden 16384, 512 experts top-16, vocab 202048.
  - Generated with the `proxy_moe_sglang` package (step 3).

## 1. Start the container

```bash
docker run -d --name rccl_dda_hang \
  --network=host --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v <proxy_moe_sglang package>:/proxy_moe_sglang:ro -v <local nvme, >=1 TB free>:/work \
  rocm/sgl-dev:v0.5.22-rocm101-mi45x-20261020-test sleep infinity
docker exec -it rccl_dda_hang bash
```

`--cap-add=SYS_PTRACE` is needed for py-spy and rocgdb.

## 2. AITER cherry-picks (on top of the image's `2bb549102`)

These are needed to serve the model on gfx1250, not for the hang itself. Use only these three commits; do not take the whole PR branches, which are based on a newer main.

- **#6720 `ee56e61dd`:** `fused_dynamic_mx_quant_moe_sort` uses the split path above 8192 columns.
- **#6719 `5dd80d3a6`:** fall back to Triton MXFP4/MXFP8 quant when `cdna5.scaled_downcast` is missing.
- **#6719 `d02a8919f`:** add the gfx1250 `rmsnorm_large_m_small_n` config.
- Do **not** take `89dd89da9` (#6719's gemm_afp4wfp4 config replacement).

```bash
cd /sgl-workspace/aiter
git fetch origin pull/6720/head:pr6720 pull/6719/head:pr6719
for c in ee56e61dd 5dd80d3a6 d02a8919f; do git diff $c^ $c | git apply; done   # working tree; the image already has local edits
git status --short   # expect aiter/ops/quant.py, aiter/ops/triton/quant/quant.py and the new configs/gfx1250/triton/normalization/
```

Do not run the package's `patches/aiter_mx_quant_large_hidden.py`, and do not set `AITER_LARGE_HIDDEN_WORKAROUND`. #6720 already covers both.

## 3. Dummy 12-layer checkpoint (about 686 GB, a few minutes)

```bash
mkdir -p /work/tokenizer
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -fL -o /work/tokenizer/$f https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/$f
done
cd /tmp && python3 /proxy_moe_sglang/tools/make_hf_config.py --spec /proxy_moe_sglang/spec/proxy_spec.json \
  --num-layers 12 --out-dir /work/proxy_12l --tokenizer-dir /work/tokenizer --num-gpus 8
(cd /proxy_moe_sglang/tools && python3 gen_dummy_ckpt.py --model-dir /work/proxy_12l --workers 32)
```

## 4. RCCL 2.32.3 (optional: the hang also reproduces on the image's 2.30.7)

This follows sglang PR #45576: build rocm-systems `4ca8623` and replace the SDK's `librccl.so.1` in place. rocm_sdk preloads that file by absolute path, so loading a second copy breaks `torch.cuda.device_count()`.

```bash
SP=/opt/venv/lib/python3.12/site-packages
mkdir -p /work/rccl_backup && cp -a $SP/_rocm_sdk_libraries/lib/librccl* $SP/_rocm_sdk_devel/lib/librccl* /work/rccl_backup/  # restore from here
pip install patchelf
mkdir -p /tmp/rocm-systems && cd /tmp/rocm-systems && git init -q
git remote add origin https://github.com/ROCm/rocm-systems.git
git sparse-checkout init --cone && git sparse-checkout set projects/rccl
git fetch --depth 1 --filter=blob:none origin 4ca8623b699fe723929621d394af5144c9e643e0 && git checkout -q FETCH_HEAD
cd projects/rccl && CMAKE_PREFIX_PATH="$ROCM_HOME" ./install.sh --amdgpu_targets gfx1250 --prefix /tmp/rccl_install -i -j 128
RPATH="$(patchelf --print-rpath $SP/_rocm_sdk_libraries/lib/librccl.so.1)"
cp /tmp/rccl_install/lib/librccl.so.1.0 /tmp/librccl.so.1 && patchelf --force-rpath --set-rpath "$RPATH" /tmp/librccl.so.1
for d in $SP/_rocm_sdk_libraries/lib $SP/_rocm_sdk_devel/lib; do
  rm -f $d/librccl.so.1 $d/librccl.so.1.0; cp /tmp/librccl.so.1 $d/librccl.so.1; ln -sf librccl.so.1 $d/librccl.so.1.0
  [ -e $d/librccl.so ] && { rm -f $d/librccl.so; ln -sf librccl.so.1 $d/librccl.so; }
done
cd /tmp && python3 -c "import ctypes,torch; v=ctypes.c_int(); ctypes.CDLL('librccl.so.1').ncclGetVersion(ctypes.byref(v)); print(v.value, torch.cuda.device_count())"
# expect: 23203 8
```

## 5. Reproduce the hang

Run from `/tmp`. The image WORKDIR `/sgl-workspace` shadows `sglang.benchmark`.

```bash
cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1
export NCCL_CUMEM_ENABLE=1          # the trigger; SGLang otherwise forces 0
# optional evidence: export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL

for t in 1 2 3 4 5 6; do
  python3 -m sglang.launch_server --model-path /work/proxy_12l \
    --tp-size 8 --ep-size 8 --kv-cache-dtype fp8_e4m3 \
    --context-length 32768 --mem-fraction-static 0.80 \
    --max-running-requests 64 --cuda-graph-max-bs-decode 64 \
    --chunked-prefill-size 8192 --attention-backend triton \
    --watchdog-timeout 3600 --host 127.0.0.1 --port 18030 > /tmp/server_try$t.log 2>&1 &
  ready=0
  for i in $(seq 1 30); do curl -sf -m 5 http://127.0.0.1:18030/health_generate >/dev/null && { ready=1; break; }; sleep 10; done
  if [ $ready = 1 ]; then echo "try$t: ready (no hang)"; pkill -f sglang; sleep 15; continue; fi
  echo "try$t: HUNG (not ready after 300 s; a healthy start is ready in ~50 s)"; break   # leave it running for step 6
done
```

- **Healthy start:** `health_generate` answers in about 50 s with a warm JIT cache.
- **Hang:** the log stops after the model's warmup-forward messages (aiter "not found tuned config" lines). About 600 s later it shows `ReadTimeout ... read timeout=600` from the warmup request, and the server kills itself.

Variants:
- `--max-running-requests 2 --cuda-graph-max-bs-decode 2` hangs too (2/2 on 2.30.7).
- Once the server is up, serving and decode are fine. The hang is only seen in the first warmup prefill.

## 6. Collect evidence while it is hung

```bash
# host side: every scheduler is blocked in a stream sync after the warmup prefill
for p in $(pgrep -f 'sglang::scheduler'); do py-spy dump --native --pid $p > /tmp/pyspy_$p.txt; done
grep -A3 '(active)' /tmp/pyspy_*.txt | grep -E 'WaitRelaxed|synchronize|process_batch_result_prefill' | sort | uniq -c

# GPU side: which kernel's waves are resident
for p in $(pgrep -f 'sglang::scheduler'); do
  rocgdb -q -nx -batch -p $p -ex 'set pagination off' -ex 'info agents' -ex 'info queues' \
    -ex 'info dispatches' -ex 'info threads' > /tmp/rocgdb_$p.txt 2>&1
  echo "$p: $(grep 'AMDGPU Wave' /tmp/rocgdb_$p.txt | grep -oE 'in void [A-Za-z_:]+' | sort | uniq -c)"
done

# with NCCL_DEBUG_SUBSYS=COLL: the last collective per rank (all identical)
for r in 0 1 2 3 4 5 6 7; do grep -E "\[$r\] NCCL INFO [A-Za-z]+: opCount" /tmp/server_try*.log | tail -1 | sed -E 's/.*NCCL INFO //' | cut -c1-110; done
```

Observed (2026-10-23, RCCL 2.32.3):
- **py-spy:** all 8 `sglang::scheduler_TP*` main threads are in `rocr::core::BusyWaitSignal::WaitRelaxed` <- `torch.cuda` stream `synchronize` <- `process_batch_result_prefill` (`scheduler_components/batch_result_processor.py:268`).
- **rocgdb:** on every rank all resident waves are in `void dda::common::ddaAllReduceFlatLL<__hip_bfloat16, 8>(...)`. Seven ranks have 768 waves; one rank (TP1) has 349, meaning part of its grid retired while the rest spin. Example:
  `AMDGPU Wave 1:4:1:1 (2,0,0)/0 "ddaAllReduc-845d" 0x00007f0b828a332c in void dda::common::ddaAllReduceFlatLL<__hip_bfloat16, 8>(__hip_bfloat16* const*, __hip_bfloat16*, __hip_bfloat16 const*, unsigned long, int, int, unsigned int*, int, unsigned long) () from .../librccl.so.1`
- **COLL log:** on every rank the TP communicator saw 25x `AllReduce` bf16 count 98304 (a 6-token prefill x hidden 16384 = 192 KB), then `AllGather` bf16 count 25256/rank (1 token's logits, vocab shard) with opCount 0x19. Same sequence, sizes and order on all ranks.
  - Before that, a second communicator ran 937 AllReduces (65536-917504 elements, 75 each, a startup sweep); none were pending.
- **Path selection:** in 2.32.3 the 192 KB AllReduce is below `RCCL_DDA_LL_ONESHOT_THRESHOLD` (default 1 MiB, `src/rccl_wrap.cc:102`), so it takes DDA flat (one-shot) LL. Decode at 2 MB uses `ddaAllReduceTwoShotLL`.

## 7. Workarounds tested (RCCL 2.32.3 + `NCCL_CUMEM_ENABLE=1`)

Microbench: `torchrun --nproc-per-node 8 rccl_ar_bench.py`, CUDA-graph replay of `dist.all_reduce` on bf16 [M, 16384], median over ranks.

| Extra env | 64 KB | 192 KB | 512 KB | 2 MB | Server starts |
|---|---|---|---|---|---|
| none (DDA on) | 11.3 us | - | 27.7 us | 37.0 us | 4 of 7 hung |
| `RCCL_DDA_LL_ONESHOT_THRESHOLD=0` | 30.6 | 30.8 | 32.3 | 36.8 | hung on start 4 |
| `RCCL_DDA_ENABLE=0` | 34.4 | 36.7 | 41.3 | 68.8 | 0 of 6 hung |
| (`NCCL_CUMEM_ENABLE=0`, `P2P/IPC`) | 55 | - | 104-194 | 221 | 0 hung |

Avoiding only flat LL does not fix the hang, so DDA as a whole is implicated. Two microbench runs (`RCCL_DDA_ENABLE=0`, and the `P2P/IPC` path) printed results and then hung at process teardown (2/2 for IPC). That is a separate issue; the server never tears RCCL down mid-run.

Serving impact on 1K1K (ITL / output tok/s):

| Config | c=2 | c=64 |
|---|---|---|
| 2.30.7, CUMEM=0 (SGLang default) | 13.04 ms / 150.8 | 37.07 ms / 1596 |
| 2.32.3, CUMEM=1, DDA on (hangs) | 10.2 ms / 192 | 35.24 ms / 1686 |
| 2.32.3, CUMEM=1, `RCCL_DDA_ENABLE=0` | 11.12 ms / 178 | 36.41 ms / 1625 |

## Restore the image's RCCL

```bash
SP=/opt/venv/lib/python3.12/site-packages
rm -f $SP/_rocm_sdk_libraries/lib/librccl* $SP/_rocm_sdk_devel/lib/librccl*
cp -a /work/rccl_backup/librccl.so.1 /work/rccl_backup/librccl.so.1.0 $SP/_rocm_sdk_libraries/lib/
cp -a /work/rccl_backup/librccl.so.1 /work/rccl_backup/librccl.so.1.0 $SP/_rocm_sdk_devel/lib/
ln $SP/_rocm_sdk_devel/lib/librccl.so.1 $SP/_rocm_sdk_devel/lib/librccl.so   # stock devel .so is a hard link
md5sum $SP/_rocm_sdk_libraries/lib/librccl.so.1   # stock: a6bde8aae3f4f000ce2e0a6f14e74e05
```

## Raw data (h13-17)

All under `/shared_nfs/kk/results/proxy_12l/`:
- `hang_dump_rccl2323_h13-17_1023` (py-spy)
- `hang_coll_rccl2323_h13-17_1023` (COLL log)
- `hang_rocgdb_rccl2323_h13-17_1023` (rocgdb)
- `rccl_env_probe_h13-17_1023`, `rccl2323_probe_h13-17_1023` (microbench)
- `dda_ws_{a,b}_h13-17_1023` (workarounds)
- `hang_repro_rccl2323_h13-17_1023` (standalone replay, no hang)

Scripts are in `proxy-moe/scripts/`: `mi455_hang_dump.sh` (`ROCGDB=1`), `rccl_ar_bench.py` and `rccl_prefill_hang_repro.py`. Full investigation log: `MI455_RUN_h13-17.md`.
