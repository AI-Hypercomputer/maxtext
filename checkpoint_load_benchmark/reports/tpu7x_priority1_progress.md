# Priority 1 (>200B models) progress on TPU v7x (`bodaborg-tpu7x-nap`)

> **Snapshot:** 2026-10-28 ~12:55 UTC · **Branch:** `docs/checkpoint-load-benchmark` · **Live queue:**
> `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/` (task ids below refer to
> [`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json))

**Hardware per slice:** `tpu7x-standard-4t`, topology `2x2x1` = 4 chips = 8 TPU cores, **813.8 GB HBM**,
**1,014.5 GB host RAM**, spot capacity. **1 slice** = 8 cores; **2 slices** = 16 cores (multi-slice over DCN,
`dcn_fsdp_parallelism=2`).

**Experiments**
- **1A**: `safetensors_dynamic` BEFORE (`origin/main`, `f596f5369`) vs WITH Lance's 3 PRs (#5706, #5828, #5705).
- **1B**: `to_maxtext` (S1 HF→GCS, S2 CPU conversion to Orbax, S3 Orbax load into HBM) vs `safetensors_dynamic`
  (S1 HF→GCS, S2 sharded load + in-memory conversion).

**Legend:** ✅ done · 🔄 running · ⏳ queued (task id) · ❌ failed (reason) · ⚠️ inconclusive · ➖ not applicable

## Summary by model

| Model | BF16 size / shards | S1 HF→GCS (cold) | 1A WITH PRs | 1A BEFORE (`origin/main`) | 1B `to_maxtext` |
| :--- | :--- | ---: | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 470.4 GB / 118 | 583.3 s | ✅ 4/4 | ⚠️ killed mid-load by the 360 s watchdog → ⏳ re-run at 1800 s | ⏳ 4 tasks (scanned conversion re-queued after the shard-thrash fix) |
| `deepseek4-284b` | 581.9 GB / 146 *(synthetic sparse files)* | – | ✅ 4/4 | ⚠️ killed at 360 s → ⏳ re-run at 1800 s | ⏳ 4 tasks |
| `qwen3.5-397b-a17b` | 795.5 GB / 94 | 814.5 s | 🔄 unscanned · 2 running; ❌ scanned (TPU HBM OOM) | ⚠️ killed at 360 s → ⏳ unscanned · 2 at 1800 s | ⏳ unscanned only (scanned conversion does not fit one host) |
| `qwen3-480b-a35b` | 961.0 GB / 241 | 1,046.1 s | ✅ 2/2 that fit (1 slice ❌ HBM) | ❌ `KeyError` (no mapping before #5706); conclusive | ⏳ unscanned only (scanned conversion does not fit one host) |
| `deepseek3-671b` | 1,342.4 GB / 336 *(synthetic sparse files)* | – | ❌ does not fit 1 or 2 slices | ➖ | ➖ |

## All tests

**WITH S2** is `safetensors_dynamic` Step 2 (sharded load + transform). **TM S2** is the `to_maxtext` CPU conversion,
run once per scan mode on one host and shared by the 1- and 2-slice loads. **TM S3** is the Orbax load into HBM.

| Model | Config | 1A BEFORE (`origin/main`) | 1A WITH PRs: WITH S2 (load + transform) | 1B TM S2 convert | 1B TM S3 load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | scanned · 1 | ⚠️ killed mid-load by the watchdog (deployed manifest: 360 s) → ⏳ `before_235b_s1` | ✅ **757.5 s** (674.8 + 82.7) | ⏳ `tm_235b_scanned_s1` | ⏳ `tm_235b_scanned_s1` |
| | unscanned · 1 | ➖ skipped after scanned → ⏳ `before_235b_s1` | ✅ **697.1 s** (661.8 + 35.2) | ⏳ `tm_235b_unscanned_s1` | ⏳ `tm_235b_unscanned_s1` |
| | scanned · 2 | ⚠️ killed mid-load (watchdog / slice teardown) → ⏳ `before_235b_s2` | ✅ **792.4 s** (708.0 + 83.6) | (shared) | ⏳ `tm_235b_scanned_s2` |
| | unscanned · 2 | ➖ skipped → ⏳ `before_235b_s2` | ✅ **749.0 s** (704.9 + 44.0) | (shared) | ⏳ `tm_235b_unscanned_s2` |
| `deepseek4-284b` | scanned · 1 | ⚠️ killed at 360 s → ⏳ `before_284b_s1` | ✅ **453.3 s** (366.3 + 86.8) | ⏳ `tm_284b_scanned_s1` | ⏳ `tm_284b_scanned_s1` |
| | unscanned · 1 | ➖ skipped → ⏳ `before_284b_s1` | ✅ **414.1 s** (377.0 + 37.0) | ⏳ `tm_284b_unscanned_s1` | ⏳ `tm_284b_unscanned_s1` |
| | scanned · 2 | ⚠️ killed at 360 s → ⏳ `before_284b_s2` | ✅ **440.7 s** (352.9 + 87.6) | (shared) | ⏳ `tm_284b_scanned_s2` |
| | unscanned · 2 | ➖ skipped → ⏳ `before_284b_s2` | ✅ **353.0 s** (306.3 + 46.5) | (shared) | ⏳ `tm_284b_unscanned_s2` |
| `qwen3.5-397b-a17b` | scanned · 1 | ⚠️ killed at 360 s | ❌ load finished (947.7 s), then TPU HBM OOM in the transform (795 GB ≈ 98% of HBM) → ⏳ clean record `st_397b_scanned_s1` | ➖ needs the full checkpoint staged on one host (795 GB > budget); recorded by ⏳ `tm_397b_scanned_s1` | ➖ |
| | unscanned · 1 | ➖ | ➖ does not fit 1 slice | ⏳ `tm_397b_unscanned_s1` (streams per layer) | ⏳ `tm_397b_unscanned_s1` (expected HBM OOM) |
| | scanned · 2 | ⚠️ killed at 360 s (WITH load alone takes 484 s) | ❌ load finished (483.8 s), then HBM OOM splitting `wi_0`/`wi_1` → ⏳ fix variant `st_397b_scanned_s2_hbmfix` | ➖ | ➖ |
| | unscanned · 2 | ⏳ `before_397b_unscanned_s2` (1800 s) | 🔄 `st_397b_unscanned_s2` (started 12:48 UTC) | (shared) | ⏳ `tm_397b_unscanned_s2` |
| `qwen3-480b-a35b` | scanned · 1 | ❌ `KeyError: 'qwen3-480b-a35b'` | ❌ TPU HBM OOM (961 GB > 813.8 GB) | ➖ does not fit one host's staging budget; recorded by ⏳ `tm_480b_scanned_conv` | ➖ |
| | unscanned · 1 | ❌ `KeyError` (skipped after scanned) | ➖ does not fit 1 slice | ⏳ `tm_480b_unscanned_conv` (streams per layer) | ➖ does not fit 1 slice |
| | scanned · 2 | ❌ `KeyError` | ✅ **1,067.9 s** (942.0 + 125.8) | ➖ | ➖ |
| | unscanned · 2 | ❌ `KeyError` (skipped) | ✅ **992.2 s** (907.7 + 84.3) | (shared) | ⏳ `tm_480b_unscanned_s2` |
| `deepseek3-671b` | scanned · 1 | ⚠️ killed at 360 s | ❌ exceeds 1-slice HBM (1,342 GB > 813.8 GB) | ➖ | ➖ |
| | scanned · 2 | ⚠️ killed at 360 s | ❌ host OOM-killed at 742.1 GB sampled peak; needs ≥ 4 slices | ➖ | ➖ |
| | unscanned · 1 / 2 | ➖ skipped | ➖ same capacity limits | ➖ | ➖ |

## Notes
- **Peak host RAM (WITH PRs)**:
  - 235B: 392 GB on 1 slice, 255 GB on 2 slices.
  - 284B: 963–980 GB on 1 slice, 823 GB on 2 slices (synthetic sparse files staged in local RAM).
  - 480B: 416 GB on 2 slices.
- **BEFORE watchdog**: the original BEFORE runs used `PHASE_TIMEOUT_S=360`, shorter than the WITH-PRs load times (306–942 s), so their kills don't show that `origin/main` fails. The 1800 s re-runs (`before_*` tasks) resolve this. Only `qwen3-480b-a35b`'s `KeyError` is a conclusive BEFORE failure.
- **DeepSeek numbers** use synthetic sparse BF16 files staged on the host (`STAGE_SPARSE_DEEPSEEK_OOM=1`). They measure load/transform mechanics and capacity, not GCS bandwidth.
- **`to_maxtext` S1** is the same HF→GCS download as `safetensors_dynamic` S1, measured once per model (480B's TM S1 was measured separately at 1,135.5 s). TM S2 conversions that re-stage the checkpoint locally now record `hf_to_local_restage_s`.
- **Not yet applicable**: there are no 1B comparisons yet, because no TM S2/S3 has finished for a >200B model. Two scanned 235B conversion attempts were lost: one to a pod eviction, one to shard thrashing (fixed). See [`../handoff.md`](../handoff.md) §7.14–7.15.

## How to refresh
```bash
# Per-phase results (any run dir) -> status matrix
gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b/logs /tmp/p1 --exclude='.*\.log$'
python3 checkpoint_load_benchmark/scripts/summarize_tpu7x_results.py /tmp/p1
# Queue: finished tasks / running tasks (with heartbeat)
gcloud storage ls gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/done/ gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/locks/
```
Run dirs: `p1_lance_prs/run_tpu7x_200b` (WITH + `to_maxtext`), `p1_origin_main/run_tpu7x_200b` (BEFORE, 360 s),
`p1_origin_main/run_tpu7x_200b_t1800` (BEFORE re-runs), `p1_lance_prs/run_tpu7x_200b_hbmfix` (fix variant).
