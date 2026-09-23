讀 /workspace/claude-skills/agentx/SKILL.md 的 CONTINUE HERE。不要重建對話歷史，
文件是唯一事實來源。同時遵守 cap-tool-output 和 reduce-conversation-usage。

任務：找出「新 main 的 GPU KV pool 佔用比舊 HEAD 多約 10 個百分點」的原因。
先不要動 GPU，讀完跟我討論方向。

已知（都在 SKILL.md，這裡只列座標）：

| arm | conc | GPU pool | tok/s/GPU |
|---|---|---|---|
| c64-chunk16384（舊 HEAD a1f9508dd4） | 64 | 42% | 18,724.0 |
| c64-chunk16384-newmain（main cdbfe90b4a） | 64 | 51% | 20,131.6 |
| c64-chunk16384-newmain-rep | 64 | 39% | 20,753.1 |
| c128-chunk16384（舊） | 128 | 70% | 25,968.4 |
| c128-chunk16384-newmain | 128 | 80% | 28,043.6 |

第一件要確認的事：c64 兩次新 main 量測是 51% 和 39%，跨度比「新舊差異」本身
還大。所以先判定這 10 個百分點是版本差異、還是這個指標本身不穩定。在這點
確認前，不要假設它是真的。

指標定義：GPU pool = summary_table.py:55 讀的 kv.gpu_usage_pct，來自各 arm
目錄的 dsv4_..._c<N>.json。請先確認它是「取樣時點的瞬時值」還是「整場的
平均／峰值」——若是瞬時值，跨 arm 比較可能沒有意義。

為什麼在意：c128 新 main 已到 80%，只剩 20 個百分點餘裕。另一台 node 在相同
組態（無 fusion）下 c128 會 OOR，這 10 個百分點是目前最合理的嫌疑。同一版本上
fusion 的三次探針（mem-fraction 0.90/0.85/0.80）全部在 VRAM 99% 撞牆，且降
mem-fraction 只延後不解決（見 SKILL.md「The real trap」與其後的階梯表）。
這兩件事可能同源。

素材（全在 /workspace/results/，/tmp 會被清空、不要放東西）：
- 各 arm 目錄：server.log（含啟動時 SGLANG_* env dump、max_total_num_tokens、
  available_gpu_mem）、sglang_command.txt、結果 json
- 逐 10 秒 VRAM 曲線：probe-fusion-mf085/ probe-fusion-mf080/
  probe2-fusion-emptycache/ 各自的 vram.csv
- 工具：arm_report.py、summary_table.py、warmup_check.sh、watch_arm.sh、
  vram_sampler.sh、empty_cache_probe.sh、run_arms.sh
- 版本比對：HEAD cdbfe90b4a，舊的是 a1f9508dd4。working tree 有別人未提交的
  HIP 改動，不要 revert 或 stash。

環境陷阱（都已經害過人，詳見 SKILL.md）：
- EP_SIZE 必須顯式傳 1，wrapper 預設是 8
- 監看要同時看 server.log，crash 不會出現在 launcher stdout
- errors=0 且進度凍結 = server 已死但 HTTP 仍回 200
- 跑完讓 launcher 自己 reclaim，不要強殺，否則 VRAM 會掛在無主狀態
