# Priority 1 (>200B models) on TPU v7x: Lance's PRs, and `to_maxtext` vs `safetensors_dynamic`

> Cluster `bodaborg-tpu7x-nap` (`tpu7x-standard-4t`, topology `2x2x1`). Branch `docs/checkpoint-load-benchmark`.
> Per-test status, including runs still in progress: [`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md).
> Issues and fixes: [`tpu7x_benchmark_ckpt_load_time_issues.md`](tpu7x_benchmark_ckpt_load_time_issues.md).

## 1. Summary

**Exp 1A: `safetensors_dynamic` BEFORE (`origin/main` `f596f5369`) vs WITH Lance's 3 PRs (#5706, #5828, #5705)**
- WITH the PRs, every >200B model loads once there is enough HBM:
  - `qwen3-235b-a22b`: 697–792 s, 1 or 2 slices
  - `deepseek4-284b`: 353–453 s, 1 or 2 slices (synthetic files)
  - `qwen3-480b-a35b`: 992–1,068 s, 2 slices
  - `qwen3.5-397b-a17b`: unscanned 1,312 s, 2 slices
  - `deepseek3-671b`: 531–649 s, 4 slices (synthetic files)
- BEFORE the PRs, no >200B load finished:
  - `qwen3-480b-a35b` has no parameter mapping (`KeyError`).
  - The other runs didn't finish their **load phase** within an 1800 s watchdog. The WITH-PRs load phases for the same configs take 353–708 s. So the PRs make the load at least 2.5–5× faster on every config where BEFORE got as far as loading: 235B, 284B, 397B and 671B. These are lower bounds.
- One remaining gap WITH the PRs: `qwen3.5-397b-a17b` scanned on 2 slices runs out of HBM while splitting the stacked
  `(wi_0, wi_1)` composite (issue U-1). A first fix (`patches/hbmfix`) didn't move that peak.

**Exp 1B: `to_maxtext` (S1 HF→GCS, S2 CPU conversion, S3 Orbax load) vs `safetensors_dynamic` (S1, S2 load + convert)**
- On a tpu7x host (1,014.5 GB RAM, ~100 GB boot disk), `to_maxtext` S2 **cannot convert most >200B configurations**:
  - It keeps the whole converted state in host RAM before the Orbax save.
  - Scanned conversions need the whole source staged locally, and the only local storage large enough is RAM.
  - Only unscanned 235B and 397B convert, by streaming shards.
- `safetensors_dynamic` loads straight from the HF shards, so it is the only pipeline that loads the scanned 235B,
  284B, 480B, and 671B configurations at all.

## 2. Setup

| Item | Value |
| :--- | :--- |
| Slice | 1 node, 4 chips = 8 TPU cores, 813.8 GB HBM, 1,014.5 GB host RAM (986.7 GB allocatable), spot |
| Configurations | {scanned, unscanned} × {1 slice (8 cores), 2 slices (16 cores, `dcn_fsdp_parallelism=2`)}; plus 4 slices (32 cores) for `deepseek3-671b` and scanned `qwen3.5-397b-a17b` |
| WITH PRs code | `src/` of this branch (= `origin/main` + #5706, #5705, #5828) plus the pod bootstrap's string patches ([`bootstrap_tpu7x.py`](../bootstrap_tpu7x.py)): single-GET HF→GCS streaming (Step 1 only), and freeing reshard/stack intermediates plus splitting composite leaves before the reshard (Step 2 HBM peak) |
| BEFORE code | `origin/main` `src/maxtext/checkpoint_conversion` (`f596f5369`) plus only the streaming patch. The 3 other files the PRs touch (`globals.py`, `max_utils.py`, `model_creation_utils.py`) are not on the measured path: the harness initializes `jax.distributed` itself and calls `checkpointing.load_state_if_possible`, which the PRs don't change |
| `hbmfix` variant | [`patches/hbmfix`](../patches/hbmfix.patch), run only as `st_397b_scanned_s2_hbmfix` |
| Step 1 (HF→GCS) | One cold download per model into a shared GCS cache, reused by every configuration and both pipelines: 235B 583.3 s, 397B 814.5 s (`results/p1_st_lance_b.csv`), 480B 1,046.1 s (`to_maxtext` measured its own: 1,135.5 s) |
| DeepSeek models | `STAGE_SPARSE_DEEPSEEK_OOM=1`: synthetic sparse BF16 safetensors written on the host (no HF→GCS step). They measure load/transform mechanics and capacity, not GCS bandwidth |
| Watchdog | `PHASE_TIMEOUT_S=1800` for BEFORE; the first BEFORE pass used 360 s and is not used for conclusions |
| Concurrency | Runs went through a GCS task queue on up to 20 spot nodes in parallel, each run on its own node(s). Runs of the same model sometimes read the same GCS cache at the same time |

## 3. Exp 1A: BEFORE vs WITH (`safetensors_dynamic` Step 2, load + transform)

| Model | Config | BEFORE (`origin/main`) | WITH Lance's PRs | Peak host RAM (BEFORE / WITH) |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | scanned · 1 | ❌ timeout at 1800 s, still loading | **757.5 s** (674.8 + 82.7) | 528.0 / 391.8 GB |
| | unscanned · 1 | (skipped after scanned, harness rule) | **697.1 s** (661.8 + 35.2) | – / 392.7 GB |
| | scanned · 2 | ❌ timeout at 1800 s, still loading | **792.4 s** (708.0 + 83.6) | 300.8 / 255.9 GB |
| | unscanned · 2 | (skipped after scanned) | **749.0 s** (704.9 + 44.0) | – / 254.6 GB |
| `deepseek4-284b` | scanned · 1 | ❌ timeout at 1800 s, still loading | **453.3 s** (366.3 + 86.8) | 501.4 / 963.4 GB |
| | unscanned · 1 | (skipped after scanned) | **414.1 s** (377.0 + 37.0) | – / 980.3 GB |
| | scanned · 2 | ❌ timeout at 1800 s, still loading | **440.7 s** (352.9 + 87.6) | 576.0 / 822.6 GB |
| | unscanned · 2 | (skipped after scanned) | **353.0 s** (306.3 + 46.5) | – / 822.6 GB |
| `qwen3.5-397b-a17b` | scanned · 1 | – | ❌ load 942.1 s, then TPU HBM OOM (795 GB ≈ 98% of HBM) | – / 48.3 GB |
| | scanned · 2 | – | ❌ load 483.8 s, then HBM OOM splitting `(wi_0, wi_1)` (U-1); `hbmfix`: load 469.7 s, same OOM | – / 161.2 GB |
| | unscanned · 2 | ❌ timeout at 1800 s, still loading | **1,311.9 s** (515.9 + 795.7) | 301.7 / 98.7 GB |
| | scanned · 4 | – | ⏳ `st_397b_scanned_s4` | – |
| `qwen3-480b-a35b` | scanned · 1 | ❌ `KeyError` | ❌ TPU HBM OOM (961 GB > 813.8 GB) | – |
| | scanned · 2 | ❌ `KeyError` | **1,067.9 s** (942.0 + 125.8) | – / 415.7 GB |
| | unscanned · 2 | ❌ `KeyError` | **992.2 s** (907.7 + 84.3) | – / 415.5 GB |
| `deepseek3-671b` | scanned · 1 / 2 | – | ❌ 1 slice: HBM OOM; 2 slices: host OOM-killed (742 GB sampled peak) | – |
| | scanned · 4 | ❌ timeout at 1800 s, still loading | **649.4 s** (468.9 + 176.8) | 344.3 / 689.9 GB |
| | unscanned · 4 | (skipped after scanned) | **531.0 s** (409.9 + 120.7) | – / 689.9 GB |

The PR changes behind the WITH results:
- **#5706** adds the 480B mapping, composite keys, legacy key renames, and multi-host hooks.
- **#5828** reads each tensor along its leading axis, so a device reads one contiguous byte range, and keeps inner MoE stacks in the read layout.
- **#5705** lets `safetensors_dynamic` run outside the training restore path.

## 4. Exp 1B: `to_maxtext` vs `safetensors_dynamic`

| Model (checkpoint) | `to_maxtext` S2 conversion on one host | `to_maxtext` S3 load | `safetensors_dynamic` S2 |
| :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` (470 GB) | scanned ❌: the staged source plus the converted state exceed host RAM (OOM-killed at leaf 4/15). Unscanned: ⏳ in progress (streams shards) | ⏳ (unscanned, 1 and 2 slices) | 697–792 s |
| `deepseek4-284b` (582 GB) | ❌ OOM-killed at 989.6 GB (scanned and unscanned) | ➖ no checkpoint | 353–453 s |
| `qwen3.5-397b-a17b` (807 GB on disk) | scanned ❌: the source must be staged in RAM. Unscanned: ⏳ in progress (streams shards) | ⏳ (unscanned) | unscanned · 2: 1,311.9 s |
| `qwen3-480b-a35b` (960 GB) | ❌ scanned: the source must be staged; unscanned: the converted state exceeds ~943 GB usable RAM | ➖ no checkpoint | 992–1,068 s (2 slices) |
| `deepseek3-671b` (1.34 TB) | ➖ not attempted | ➖ | 531–649 s (4 slices) |

The row-by-row table (a `to_maxtext` row and a `safetensor` row per configuration, as in the earlier small-model
summary) is in [`tpu7x_priority1_progress.md`](tpu7x_priority1_progress.md). It is regenerated from the result JSONs by
[`scripts/build_tpu7x_tables.py`](../scripts/build_tpu7x_tables.py).

## 5. What fits where

| Model | 1 slice (813.8 GB HBM) | 2 slices (1,627.6 GB) | 4 slices (3,255 GB) |
| :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | ✅ | ✅ | – |
| `deepseek4-284b` | ✅ | ✅ | – |
| `qwen3.5-397b-a17b` | ❌ transform OOM (98% of HBM) | ✅ unscanned · ❌ scanned (U-1) | ⏳ scanned |
| `qwen3-480b-a35b` | ❌ | ✅ | – |
| `deepseek3-671b` | ❌ | ❌ host OOM | ✅ |

## 6. Reproduce

- Every run is a task in [`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json); see the README
  section "Finishing Priority 1" and [`../handoff.md`](../handoff.md) §9 for the deployment and monitoring commands.
- Run dirs (per-phase `*_result.json` and logs under `logs/<model>/<config>/`):
  - `gs://mesa-maxtext/mesakh_tpu7x_bench/p1_lance_prs/run_tpu7x_200b`: WITH PRs and `to_maxtext`.
  - `.../p1_lance_prs/run_tpu7x_200b_hbmfix`: `hbmfix`.
  - `.../p1_origin_main/run_tpu7x_200b_t1800`: BEFORE.
  - `.../p1_origin_main/run_tpu7x_200b`: first BEFORE pass, 360 s watchdog, not used for conclusions.
