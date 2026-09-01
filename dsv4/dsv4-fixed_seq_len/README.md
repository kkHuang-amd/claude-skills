# dsv4 fixed_seq_len — server launch commands (b200 / mi355x)

Extracted 2026-08-31 from InferenceX `8fcfc62830` (2026-08-26), the pinned local
checkout at `/workspace/InferenceX`, which is the tree these two GitHub links
point at:

| file here | source |
|---|---|
| `serve_b200.sh` | `benchmarks/single_node/fixed_seq_len/dsv4_fp4_b200.sh` (148 lines) |
| `serve_mi355x.sh` | `benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh` (113 lines) |

Purpose: refresh the **1k/1k** and **8k/1k** numbers at
`CONC = 2 4 8 16 32 64 128 256 512 1024`.

## What was kept / dropped

Kept verbatim: every `export`, every server flag, the `DP_ATTENTION` branch, the
`EP_SIZE` branch, and the ISL-derived sizes. Values are now `${VAR:-upstream}`
so they can be overridden without editing — **defaults are byte-identical to
upstream**.

Dropped (client side, not server launch): `hf download`, `nvidia-smi`,
`start/stop_gpu_monitor`, `run_benchmark_serving`, `run_eval`.
`RANDOM_RANGE_RATIO` and `--num-prompts $((CONC*10))` belong to the client and
are gone with it.

Three deliberate changes, each forced by the scripts leaving their directory:

1. **No `source ../../benchmark_lib.sh`.** A copied launcher outside
   `fixed_seq_len/` cannot find it and exits 1 instantly. `PORT=${PORT:-8888}`
   (lib line 17) and `wait_for_server_ready` are inlined.
2. **`--chat-template` is absolute** in `serve_mi355x.sh`. Upstream resolves it
   as `$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja`; from a copy
   that silently resolves to nothing. Default now points into
   `/workspace/InferenceX/...`; the script aborts if unreadable.
3. **The readiness check verifies the listener is our PID**, not just that
   `/health` returns 200 — a stale router from an earlier arm answering 200 on
   8888 has burned a run before. Both scripts write `server.pid`; kill **by
   PID** (`pkill -f 'sglang::'` does not work, setproctitle rewrites argv).

## The sweep shape differs between the two platforms

**mi355x: one server per concurrency point.** Upstream sets
`--cuda-graph-max-bs $CONC` *and* `--max-running-requests $CONC`, so CONC is a
server-side variable. 10 concurrencies = 10 launches per (ISL,OSL). `CONC` is
required; the script refuses to start without it.

**b200: one server for the whole sweep.** Nothing in its launch reads `CONC` —
`--max-running-requests` and `--cuda-graph-max-bs` are constants per branch.
One server covers all 10 points. `CONC` is accepted for logging only.

Two b200 consequences worth deciding on before the run, both upstream behaviour:

- non-DP branch has `--max-running-requests 512`, so a **CONC=1024 point is
  capped at 512 in flight** — it measures 512, not 1024.
- DP branch has `--cuda-graph-max-bs 544`, so **CONC=1024 decodes outside CUDA
  graphs**. `MAX_RUNNING_REQUESTS=` / `CUDA_GRAPH_MAX_BS=` override both, but
  that departs from the recorded config.

`configs/nvidia-master.yaml` splits the CONC axis across recipes rather than
raising these: conc 1–32 → `DP_ATTENTION=false`; conc 64–1024 → `true`.
`configs/amd-master.yaml` (`dsv4-fp4-mi355x-sglang`): tp8 dp-attn conc 64–2048;
tp4 dp-attn conc 16–128; tp4 non-dp conc 1–32.
A single-recipe sweep across all 10 points is therefore **not** what CI runs.

## 1k/1k vs 8k/1k

| | b200 | mi355x |
|---|---|---|
| ISL=1024 | `swa-full-tokens-ratio` 0.5 … **but the DP branch overwrites it with 0.12**, so in DP mode 1k and 8k launch identically | `chunked-prefill-size` = 1024 (non-DP) / 8192 (DP, = ISL×TP) |
| ISL=8192 | 0.1, again overwritten to 0.12 in DP mode | 8192 (non-DP) / 65536 (DP) |

So on b200 the ISL branch only bites with `DP_ATTENTION=false`. This is upstream
behaviour, preserved; `SWA_FULL_TOKENS_RATIO_OVERRIDE=` exists if you want the
ISL-derived value to survive in DP mode.

Also note both config entries only list `isl: 8192` today — **1k/1k is not in
the current master config** (it lives in `configs/deprecated/nvidia-1k1k-master.yaml`),
though both scripts support `ISL=1024`.

`MAX_MODEL_LEN` is not in the config either; `benchmark_lib.sh:946` defaults it
to 16384, which is what `serve_mi355x.sh` uses. Fine for both 1k/1k and 8k/1k.

## Usage

```bash
# mi355x, 8k/1k, one launch per concurrency
for c in 2 4 8 16 32 64 128 256 512 1024; do
  ISL=8192 CONC=$c TP=8 DP_ATTENTION=true \
    RESULT_DIR=... bash serve_mi355x.sh
  # ... run client at --max-concurrency $c, then kill $(cat server.pid)
done

# b200, 8k/1k, one launch for all ten
ISL=8192 TP=8 DP_ATTENTION=true bash serve_b200.sh
```

Both scripts background the server, wait for ready, print the PID, and exit 0 —
they do not run a client and do not tear the server down.
