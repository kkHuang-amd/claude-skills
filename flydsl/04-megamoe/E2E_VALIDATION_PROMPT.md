# MegaMoE e2e-validation prompt (paste into a fresh chat)

Ready-to-paste prompt for running the whole-network / gsm8k e2e validation in a separate chat box, aligned to
the CURRENT state (2026-07-21): dispatch work concluded -> ship compact-only (recv default-off, not shipped);
GEMM Track B done -> one robust win = `b_nt` per-bucket (env-gated `MEGA_S1_BNT`, default-off, not yet in tune
JSON); all working-tree changes are default-off, so the default path == baseline. Copy everything below the line.

---

繼續 MegaMoE FlyDSL 工作:kernel 優化階段已結案,現在跑 e2e 驗證。**先套用 skill「reduce-conversation-usage」**(狀態寫回文件、指令輸出一律 `rg` 過濾、窄讀不重讀、批次工具呼叫、少回合)。

**目前狀態(務必先讀,不要重建整段歷史):**
- `/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/KERNEL_OWNER_DECODE_PLAN.md` ← **主檔**,含 Track B 全部結論:occupancy 不是槓桿、K-loop 是 MFMA-bound、唯一穩健 win = **`b_nt` per-bucket**(decode `b_nt=2` bs64 −5.1% 已嚴格驗證;prefill `b_nt=1`)。
- `/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/README.md` 的「▶ CONTINUE HERE」(dispatch 結論:ship compact-only)。
- `/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/megamoe/MEGAMOE_HANDOFF.md`(baseline gsm8k ~0.937、conc256 tok/s A/B 口徑、driver `/workspace/ab_megamoe_driver.sh`)。
- `/dockerx/var/amdsgl/kk/workspace/claude-skills/dsv4/FLYDSL_KERNEL_OPT_PLAYBOOK.md`(profiling/gotchas 方法論;含「按數字 PID kill、清 VRAM、量測噪音」)。

**環境:** FlyDSL=`/sgl-workspace/FlyDSL` @ `mega_moe_v1`;sglang=`/sgl-workspace/sglang`;8× MI355X(gfx950);model=`/dockerx/data/deepseek-ai/DeepSeek-V4-Pro`。

**關鍵前提(決定驗證口徑):**
- `cd /sgl-workspace/FlyDSL && git status --short` 應看到 5 檔未 commit(dispatch.py/gemm1.py/mega_moe.py/utils.py/test_mega_moe.py)。**這些改動全部 default-off**:recv 預設關(不出貨)、`MEGA_S1_BNT` 是 env-gated 預設關、divzero 修復對 tile_k=256 是 no-op。
- 所以**不帶任何 env、不帶 `--recv` 的預設路徑 = 原本 compact-only,理論上跟 baseline 完全一致**。`--recv` 已結案(bit-correct 但 +5% 慢),**不需再跑**。

**任務:**

**1) 回歸 sanity — 確認 default-off 改動沒破壞出貨路徑**
清 cache 後掃 bs(**compact,不帶 `--recv`**):
```bash
rm -rf ~/.flydsl /tmp/flydsl*
for bs in 1 8 64 512 2048; do PYTHONPATH=/sgl-workspace/FlyDSL MORI_SHMEM_HEAP_SIZE=40G \
  torchrun --standalone --nproc_per_node=8 tests/kernels/test_mega_moe.py \
  --network v4_pro --quant a8w4 --tokens $bs --mtpr 8192 --iters 30 2>&1 \
  | rg "FULL-E2E|mega-vs-baseline|megav1="; done
```
判準:全 bs `FULL-E2E ... PASS (all 8 ranks)`,`megav1` 與 baseline 一致(default 未改)。

**2) 驗證 GEMM tuning 成果(`b_nt` win)的 e2e 效果**
micro-harness A/B(decode 區間):對每個 bs 跑 baseline(不帶 env)vs `MEGA_S1_BNT=2`,比 `megav1` ms;每次改 env 都要 `rm -rf ~/.flydsl /tmp/flydsl*` 重建。因噪音大,用 `--iters 100` 且**同一 bs 連跑 2-3 次取穩定值**(bs2048 噪音 ~±1.2%,小效果需重複確認;bs64 很緊 ~±0.05%):
```bash
for bs in 1 8 64 512; do rm -rf ~/.flydsl /tmp/flydsl*; MEGA_S1_BNT=2 PYTHONPATH=/sgl-workspace/FlyDSL \
  MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 tests/kernels/test_mega_moe.py \
  --network v4_pro --quant a8w4 --tokens $bs --mtpr 8192 --iters 100 2>&1 | rg "FULL-E2E|megav1="; done
```
判準:oracle 全 PASS(cache hint 不影響正確性);decode(bs8-64)`megav1` 應比 baseline 快 ~3-5%。

**3) server e2e(gsm8k 準確率 + 吞吐 A/B vs dp)**
- **先跑預設(不帶 env)當回歸**:gsm8k 應 ≈ **0.937**(baseline;改動 default-off,不該掉)。指令/口徑見 `MEGAMOE_HANDOFF.md`(gsm8k + `/workspace/ab_megamoe_driver.sh`,conc256 8k/1k NP8/WARM2)。
- **驗 `b_nt` serving 效果(注意口徑陷阱):** `MEGA_S1_BNT` env 會對**所有 bucket**強制同一值 → decode 快但 **prefill 會變慢**,直接用 env 跑 serving 吞吐不公平。要拿乾淨的 serving A/B,**先把 per-bucket `b_nt` 併進 tune JSON**(`kernels/comm/mega_moe_tuning_config/flydsl_gfx950_mi355x_MegaStage1_ep8.json` 的 v4_pro/a8w4:decode buckets 設 `b_nt=2`、prefill(≥1024 tok)保持 `b_nt=0/1`),再:
  - gsm8k(準確率不受 `b_nt` 影響,確認仍 ≈0.937)
  - conc256 8k/1k throughput A/B(megamoe vs dp),對比未改前的 29,482 tok/s。
- server log 一律 `2>&1 | rg "PASS|Error|throughput|token/s|gsm8k|Accuracy"`;跑完務必按**數字 PID** 清乾淨釋放 8 卡(`for p in $(rocm-smi --showpids | awk '/^[0-9]/{print $1}'); do kill -9 $p; done`,勿用會自撞的 `pkill -f`),並確認 `rocm-smi --showmeminfo vram` 回到 ~0.3GB/GPU。

**驗證結束把數據寫回文件(不要只留對話):**
- micro + b_nt e2e 結果 → `KERNEL_OWNER_DECODE_PLAN.md`(接在既有的 b_nt 段之後)。
- 一句話總結 + gsm8k/tok/s 數字 → `README.md` 的「▶ CONTINUE HERE」。
- (**不要**寫 `COMPACT_SINGLE_ROUND_DESIGN.md` §8.3 — 那是 recv,已結案。)
