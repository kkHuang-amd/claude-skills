Owner node: crsuse2-m2m-049

# sglang AgentX regression -- new image, c1 re-baseline (2026-10-02)

本目錄自成一套：runner、matrix、points、compare、CI baseline 都是從 `../` 複製來的，
`SKILL_DIR` 解析到本目錄，所以和 075 正在用 `../` 跑的 matrix 互不影響（凍結副本各自獨立）。
流程與陷阱見 `../REGRESSION_RUNBOOK.md`；075 的紀錄見 `../E2E_REBUILD.md`（owner 075，不要改）。

## CONTINUE HERE

**Status:** Phase A 在 crsuse2-m2m-049 新 image 上 c1/c4/c16 全部對齊（差距皆 ≤1.7 %，10:38 完成，GPU 已釋放）。新 image 的 aiter 差異在低併發三點均無可見影響。
driver pid 1231 @049）。先跑 300 s smoke，再跑 3600 s c1，約 80–90 分。
**Goal:** 新 image（aiter 不同）+ runbook 的 sglang `c73f7077eb`，c1 能否重現 075 的數字。
重現 => 之後的回歸只跟 sglang code 有關。
**Done (crsuse2-m2m-049):** c1 `TAG=sglc73f-049`（舊複製腳本）、c4/c16 同 TAG（`e2e/matrix.sh` 快照）。結果 `/workspace/results/e2e-{c1,c4,c16}-sglc73f-049/`。
**Next:** Phase B——使用者給新 sglang commit，開 worktree，`e2e/matrix.sh` 跑 `POINTS="c1 c4 c16"`，對比本表 049 的列（不是 CI）。
**Status file:** `/workspace/results/e2e-matrix-sglc73f-049.status`（用 wc -l 輪詢，NFS 不要 tail -F）
**Repro (resume; done points skipped):**
```bash
R=/workspace/claude-skills/agentx/regression; cd /workspace
TAG=sglc73f-049 POINTS=c1 PYTHONPATH=/sgl-workspace/sglang-ci/python SGL_DIR=/sgl-workspace/sglang-ci \
  setsid nohup bash $R/.agentx_e2e_matrix.frozen.sh < /dev/null \
  > /workspace/results/e2e-matrix-sglc73f-049.driver.log 2>&1 &
```
**Compare:**
```bash
R=/workspace/claude-skills/agentx/regression
python3 $R/compare_ci.py /workspace/results/e2e-c1-sglc73f/*_agentic.json /workspace/results/e2e-c1-sglc73f-049/*_agentic.json   # vs 075
python3 $R/compare_ci.py $R/ci_baseline_36401947630/conc1_kvnone_ep1-dpafalse/*.json /workspace/results/e2e-c1-sglc73f-049/*_agentic.json  # vs CI
```
**Pass criteria:** gates OK（errors 0、duration ≥95 %、ISL 差 ≤3 %）；P90 intvty 差 < 6.6 %
（c1 主看 intvty；c1 的 tok/s/GPU 對速度不敏感）。

**Restructure (user, 2026-10-02):** 已建 `../e2e/`（共用環境，matrix 改為自動快照到 `$RESULTS/e2e-matrix-<TAG>.code/`）與 `../legacy/`（空，舊檔先不搬）。
待 075 matrix 與本 c1 跑完：本目錄移到 `../tasks/regression/` 並刪掉複製的腳本；`E2E_REBUILD.md` 移到 `../tasks/rebuild-075/`；刪 `../` 的舊 e2e 腳本與 runbook 拆到 `../e2e/README.md`。

## Stack on crsuse2-m2m-049 (2026-10-02)

- sglang: worktree `/sgl-workspace/sglang-ci` @ `c73f7077eb`（detached），rust `_multimodal*.so`
  從 `/sgl-workspace/sglang` 複製。image 自帶 sglang 是 `0.5.21.dev20261001+g3b2ad1c6ae`
  （工作樹 HEAD `41cbe65de0`），本次不用。
- aiter: `e7d2453f25` + 6 個本地修改（075 是 `acf8fdf93` + 3 個）：
  - 與 075 相同的 3 個：`pa_mqa_logits_fp4_prefill.py`（`@cache`）、`csrc/cpp_itfs/torch_utils.py`、
    `hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co`（換過的 ASM）
  - 多出 3 個（dsv41 工作留下）：`dsv41_fp4_{tuned,untuned}_fmoe.csv`、
    `aiter/ops/triton/attention/pa_decode_sparse.py`（Triton <3.8 一律 unpeel）
- InferenceX `cb45b6da0b`、aiperf `754356e9a3`、venv `/workspace/agentx-runtime-e2e`（共用 NFS）。

## Results (append-only, node + date per row)

| node | date | point | stack | tok/s/GPU (ref) | P90 intvty (ref) | gates | verdict |
|---|---|---|---|---|---|---|---|
| crsuse2-m2m-075 | 2026-10-02 | c1 | sglang c73f7077eb, aiter acf8fdf93+3 | 2,079.0 | 220.0 | OK | 參考值（取自 ../E2E_REBUILD.md） |
| crsuse2-m2m-049 | 2026-10-02 | c1 | sglang c73f7077eb, aiter e7d2453f25+6（new image） | 2,093.2（075 +0.7 %，CI +0.7 %） | 219.3（075 -0.3 %，CI +1.2 %） | OK，244 succ、ISL +0.4 % | **aligned**：新 image/aiter 不影響 c1 |
| crsuse2-m2m-049 | 2026-10-02 | c4 | same（e2e/matrix.sh） | 3,640.3（CI 3,647.6，-0.2 %） | 191.7（CI 188.5，+1.7 %） | OK，589 succ（CI 592）、ISL +0.05 % | **aligned**；server 未掛（075 的 c4 掛過） |
| crsuse2-m2m-049 | 2026-10-02 | c16 | same（e2e/matrix.sh） | 11,187.8（CI 11,210.2，-0.2 %） | 95.9（CI 96.6，-0.7 %） | OK，2,304 succ（CI 2,303）、ISL -0.3 % | **aligned** |
