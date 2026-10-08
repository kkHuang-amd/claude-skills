# AgentX on the new InferenceX layout (inferencex-e2e) -- shared environment

本目錄是新版 InferenceX AgentX 的**共用**程式碼與資料，以後的 AgentX 工作都以這裡為主。
只放可執行的程式碼、point 定義、CI baseline；**不放任何 task 或節點的狀態**
（各 task 的紀錄放 `../tasks/<task>/README.md`，開頭帶 `Owner node:`）。

環境建置、stack 驗證、陷阱：暫見 `../REGRESSION_RUNBOOK.md` §1/§2/§7
（075 matrix 跑完後拆到本檔，腳本名稱換成下表）。

| 檔案 | 用途（舊名） |
|---|---|
| `env.sh` | 路徑與 venv 設定，全部可覆寫（`agentx_env_e2e.sh`） |
| `run_point.sh` | 單點 runner（`agentx_e2e.sh`） |
| `matrix.sh` | smoke 把關後依序跑多點，可續跑（`agentx_e2e_matrix.sh`） |
| `extract_point.sh` | CI job log -> `points/$POINT_SET/<pt>/`（需設 `POINT_SET`） |
| `stack_fingerprint.sh`、`compare_ci.py` | 同舊版 |
| `points/ci<run>/<pt>/` | point 定義，**依 CI run 分版本**；recipe 改版就新增一組，不改舊的 |
| `baselines/ci<run>/` | 該 CI run 的結果 JSON |

## 執行（一律透過 matrix.sh）

```bash
E=/workspace/claude-skills/agentx/e2e; cd /workspace     # 不可在 /sgl-workspace 下
TAG=<label> POINTS="c1" [POINT_SET=ci36401947630 | POINTS_DIR=<task 專用 point 目錄>] [SKIP_SMOKE=1] \
PYTHONPATH=<sglang>/python SGL_DIR=<sglang> \
  setsid nohup bash $E/matrix.sh < /dev/null > /shared_nfs/kk/results/DeepSeek-V4-Pro-0813/e2e-matrix-<label>.driver.log 2>&1 &
```

matrix 啟動時把腳本與 point set 快照到 `/shared_nfs/kk/results/DeepSeek-V4-Pro-0813/e2e-matrix-<TAG>.code/`
再從快照 re-exec，所以：不用再手動做 `.frozen.sh`；多個節點可同時從本目錄啟動；
run 進行中可以改本目錄的腳本；每個結果都留有實際執行的程式碼。

## 規則（本目錄在 NFS 上，多節點共用同一個工作樹）

- 改腳本用獨立小 commit；`git add <明確路徑>`，不用 `-A` / `.`。
- task 專用的 point 變體放在 task 目錄，用 `POINTS_DIR` 指過去，不改 `points/ci*/`。
