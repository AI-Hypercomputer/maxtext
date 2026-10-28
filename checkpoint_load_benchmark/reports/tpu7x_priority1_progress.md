# Priority 1 (>200B models) progress on TPU v7x (`bodaborg-tpu7x-nap`)

> **Snapshot:** 2026-10-28 ~15:00 UTC · **Branch:** `docs/checkpoint-load-benchmark` · **Live queue:**
> `gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1/` (task ids below refer to
> [`configs/p1_completion_tasks.json`](../configs/p1_completion_tasks.json))

**Hardware per slice:** `tpu7x-standard-4t`, topology `2x2x1` = 4 chips = 8 TPU cores, **813.8 GB HBM**,
**1,014.5 GB host RAM** (986.7 GB allocatable to pods), spot capacity. **1 slice** = 8 cores; **2 slices** = 16 cores
(multi-slice over DCN, `dcn_fsdp_parallelism=2`); **4 slices** = 32 cores, 3.26 TB HBM (`dcn_fsdp_parallelism=4`;
`deepseek3-671b`, plus `qwen3.5-397b-a17b` scanned).

**Experiments**
- **1A**: `safetensors_dynamic` BEFORE (`origin/main`, `f596f5369`) vs WITH Lance's 3 PRs (#5706, #5828, #5705).
- **1B**: `to_maxtext` (S1 HF→GCS, S2 CPU conversion to Orbax, S3 Orbax load into HBM) vs `safetensors_dynamic`
  (S1 HF→GCS, S2 sharded load + in-memory conversion).

**Legend:** ✅ done · 🔄 running · ⏳ queued (task id) · ❌ failed (reason) · ⚠️ inconclusive · ➖ not applicable

## Summary by model

| Model | BF16 size / shards | S1 HF→GCS (cold) | 1A WITH PRs | 1A BEFORE (`origin/main`) | 1B `to_maxtext` |
| :--- | :--- | ---: | :--- | :--- | :--- |
| `qwen3-235b-a22b` | 470.4 GB / 118 | 583.3 s | ✅ 4/4 | ❌ scanned · 1 and · 2: **timeout at 1800 s**, still loading (WITH: 757.5 s / 792.4 s) | ❌ scanned: S2 can't run on one host · 🔄 unscanned S2 (streams shards) |
| `deepseek4-284b` | 581.9 GB / 146 *(synthetic sparse files)* | – | ✅ 4/4 | ❌ scanned · 1 and · 2: **timeout at 1800 s**, still loading (WITH: 453.3 s / 440.7 s) | ❌ S2 OOM-killed at 989.6 GB host RAM (scanned and unscanned) |
| `qwen3.5-397b-a17b` | 795.5 GB / 94 | 814.5 s | ✅ unscanned · 2 (1,311.9 s) · ✅ scanned · 4 (1,399.7 s) · ❌ scanned · 1 and · 2 (HBM OOM; also with `hbmfix` on 2 slices) | ❌ unscanned · 2: **timeout at 1800 s**, still in the load phase (WITH: load 515.9 s, total 1,311.9 s) | ⏳ scanned (expected: does not fit one host) and unscanned |
| `qwen3-480b-a35b` | 961.0 GB / 241 | 1,046.1 s | ✅ 2/2 that fit (1 slice ❌ HBM) | ❌ `KeyError` (no mapping before #5706); conclusive | ❌ scanned: S2 can't run on one host · ⏳ unscanned (expected: the 960 GB converted state doesn't fit host RAM) |
| `deepseek3-671b` | 1,342.4 GB / 336 *(synthetic sparse files)* | – | ❌ doesn't fit 1 or 2 slices · ✅ **4 slices**: scanned 649.4 s, unscanned 531.0 s | ❌ scanned · 4: **timeout at 1800 s**, still loading (WITH: 649.4 s) | ➖ |

## `to_maxtext` vs `safetensor` (row-by-row, same layout as the earlier small-model summary)

For each `(Model, Scan Mode, Slices)` configuration, **`to_maxtext`** and **`safetensor`** are on consecutive rows:
- **Col 1 (`HF -> GCS`)**: `to_maxtext` Step 1 vs `safetensor` Step 1. Both are the same cold download of the
  `.safetensors` shards, measured once per model.
- **Col 2 (`Convert to Orbax`)**: `to_maxtext` Step 2 vs `—`. `safetensor` has no conversion step.
- **Col 3 (`Load into MaxText`)**: `to_maxtext` Step 3 (`Load Orbax`) vs `safetensor` Step 2 (`Safetensors+Convert`).

Slices (Devices): `1 (8)` = 1 slice, 8 TPU cores; `2 (16)` = 2 slices, 16 TPU cores; `4 (32)` = 4 slices, 32 TPU cores. The speedup is filled in once both rows have finished.

| Model | Scan Mode | Slices (Devices) | Pipeline | 1. `HF -> GCS` (s) | 2. `Convert to Orbax` (s) | 3. `Load into MaxText` (s) | **Total (No Step 1) (s)** | **Total (End-to-End) (s)** | **Speedup (E2E / No Step 1)** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-235b-a22b`** *(470.4 GB MoE)* | `scanned` | `1 (8)` | `to_maxtext` | 583.32 | **Does not fit one host** *(needs all 470 GB staged in RAM plus the 470 GB converted state; streaming thrashes)* | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **583.32** | **—** | **757.50** *(Safetensors+Convert)* | **757.50** | **1,340.81** | **Only `safetensor` loads this config** |
| **`qwen3-235b-a22b`** *(470.4 GB MoE)* | `scanned` | `2 (16)` | `to_maxtext` | 583.32 | **Does not fit one host** | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **583.32** | **—** | **792.38** *(Safetensors+Convert)* | **792.38** | **1,375.69** | **Only `safetensor` loads this config** |
| **`qwen3-235b-a22b`** *(470.4 GB MoE)* | `unscanned` | `1 (8)` | `to_maxtext` | 583.32 | 🔄 *(`tm_235b_unscanned_s1`, streaming shards)* | ⏳ *(Load Orbax)* | ⏳ | ⏳ | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **583.32** | **—** | **697.13** *(Safetensors+Convert)* | **697.13** | **1,280.44** | ⏳ *(awaiting `to_maxtext`)* |
| **`qwen3-235b-a22b`** *(470.4 GB MoE)* | `unscanned` | `2 (16)` | `to_maxtext` | 583.32 | 🔄 *(shared with 1 slice)* | ⏳ *(queued: `tm_235b_unscanned_s2`)* | ⏳ | ⏳ | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **583.32** | **—** | **749.00** *(Safetensors+Convert)* | **749.00** | **1,332.31** | ⏳ *(awaiting `to_maxtext`)* |
| **`deepseek4-284b`** *(581.9 GB MoE, synthetic sparse files)* | `scanned` | `1 (8)` | `to_maxtext` | 0.00 *(local stage)* | **OOM-killed** *(989.6 GB host RAM)* | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **453.27** *(Safetensors+Convert)* | **453.27** | **453.27** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** *(581.9 GB MoE, synthetic sparse files)* | `scanned` | `2 (16)` | `to_maxtext` | 0.00 *(local stage)* | **OOM-killed** *(shared with 1 slice)* | — | **N/A** | **N/A** | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **440.72** *(Safetensors+Convert)* | **440.72** | **440.72** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** *(581.9 GB MoE, synthetic sparse files)* | `unscanned` | `1 (8)` | `to_maxtext` | 0.00 *(local stage)* | **OOM-killed** *(989.6 GB host RAM)* | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **414.09** *(Safetensors+Convert)* | **414.09** | **414.09** | **Only `safetensor` loads this config** |
| **`deepseek4-284b`** *(581.9 GB MoE, synthetic sparse files)* | `unscanned` | `2 (16)` | `to_maxtext` | 0.00 *(local stage)* | **OOM-killed** *(shared with 1 slice)* | — | **N/A** | **N/A** | N/A |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **352.96** *(Safetensors+Convert)* | **352.96** | **352.96** | **Only `safetensor` loads this config** |
| **`qwen3.5-397b-a17b`** *(795.5 GB Hybrid MoE)* | `scanned` | `1 (8)` | `to_maxtext` | 814.53 *(shared)* | ⏳ *(expected: does not fit one host; `tm_397b_scanned_s1`)* | — | ⏳ | ⏳ | ⏳ |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **814.53** | **—** | **HBM OOM** *(load 942.10 s, then OOM in the transform)* | **OOM** | **OOM** | Failed (1-slice HBM) |
| **`qwen3.5-397b-a17b`** *(795.5 GB Hybrid MoE)* | `scanned` | `2 (16)` | `to_maxtext` | 814.53 *(shared)* | ⏳ *(shared with 1 slice)* | — | ⏳ | ⏳ | ⏳ |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **814.53** | **—** | **HBM OOM** *(load 483.84 s, then OOM splitting `wi_0`/`wi_1`; the `hbmfix` variant loads in 469.69 s and fails at the same 11.25 GiB allocation)* | **OOM** | **OOM** | Failed (transform HBM peak) |
| **`qwen3.5-397b-a17b`** *(795.5 GB Hybrid MoE)* | `scanned` | `4 (32)` | `to_maxtext` | 814.53 *(shared)* | ⏳ *(shared with 1 slice)* | — | **N/A** | **N/A** | Not run (no 4-slice `to_maxtext` task) |
| ↳ | `scanned` | `4 (32)` | **`safetensor`** | **814.53** | **—** | **1,399.68** *(Safetensors+Convert)* | **1,399.68** | **2,214.21** | **Only `safetensor` loads this config** |
| **`qwen3.5-397b-a17b`** *(795.5 GB Hybrid MoE)* | `unscanned` | `1 (8)` | `to_maxtext` | 814.53 *(shared)* | ⏳ *(queued: `tm_397b_unscanned_s1`)* | ⏳ *(Load Orbax; 795 GB ≈ 98% of 1-slice HBM)* | ⏳ | ⏳ | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **814.53** | **—** | **Not run** *(795 GB ≈ 98% of 1-slice HBM)* | **—** | **—** | — |
| **`qwen3.5-397b-a17b`** *(795.5 GB Hybrid MoE)* | `unscanned` | `2 (16)` | `to_maxtext` | 814.53 *(shared)* | ⏳ *(shared with 1 slice)* | ⏳ *(queued: `tm_397b_unscanned_s2`)* | ⏳ | ⏳ | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **814.53** | **—** | **1,311.85** *(Safetensors+Convert)* | **1,311.85** | **2,126.38** | ⏳ *(awaiting `to_maxtext`)* |
| **`qwen3-480b-a35b`** *(961.0 GB MoE)* | `scanned` | `1 (8)` | `to_maxtext` | 1,135.50 | **Does not fit one host** *(all 961 GB must be staged)* | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **1,046.14** | **—** | **HBM OOM** *(961 GB > 813.8 GB)* | **OOM** | **OOM** | Failed (1-slice HBM) |
| **`qwen3-480b-a35b`** *(961.0 GB MoE)* | `scanned` | `2 (16)` | `to_maxtext` | 1,135.50 | **Does not fit one host** | — | **N/A** | **N/A** | N/A (`to_maxtext` infeasible on 1 host) |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **1,046.14** | **—** | **1,067.86** *(Safetensors+Convert)* | **1,067.86** | **2,114.00** | **Only `safetensor` loads this config** |
| **`qwen3-480b-a35b`** *(961.0 GB MoE)* | `unscanned` | `1 (8)` | `to_maxtext` | 1,135.50 | ⏳ *(queued: `tm_480b_unscanned_conv`; expected: the 960 GB converted state doesn't fit host RAM)* | **Does not fit 1 slice** *(961 GB > 813.8 GB)* | **N/A** | **N/A** | N/A (1-slice HBM) |
| ↳ | `unscanned` | `1 (8)` | **`safetensor`** | **1,046.14** | **—** | **Not run** *(961 GB > 813.8 GB)* | **—** | **—** | — |
| **`qwen3-480b-a35b`** *(961.0 GB MoE)* | `unscanned` | `2 (16)` | `to_maxtext` | 1,135.50 | ⏳ *(shared with 1 slice)* | ⏳ *(queued: `tm_480b_unscanned_s2`)* | ⏳ | ⏳ | ⏳ |
| ↳ | `unscanned` | `2 (16)` | **`safetensor`** | **1,046.14** | **—** | **992.17** *(Safetensors+Convert)* | **992.17** | **2,038.31** | ⏳ *(awaiting `to_maxtext`)* |
| **`deepseek3-671b`** *(1,342.4 GB MoE, synthetic sparse files)* | `scanned` | `1 (8)` | `to_maxtext` | — | **Not run** *(does not fit 1 slice)* | — | **N/A** | **N/A** | N/A |
| ↳ | `scanned` | `1 (8)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **HBM OOM** *(1,342 GB > 813.8 GB)* | **OOM** | **OOM** | Failed (1-slice HBM) |
| **`deepseek3-671b`** *(1,342.4 GB MoE, synthetic sparse files)* | `scanned` | `2 (16)` | `to_maxtext` | — | **Not run** *(does not fit 2 slices)* | — | **N/A** | **N/A** | N/A |
| ↳ | `scanned` | `2 (16)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **OOMKilled** *(host, 742.1 GB sampled peak; needs ≥ 4 slices)* | **OOM** | **OOM** | Failed (2-slice capacity) |
| **`deepseek3-671b`** *(1,342.4 GB MoE, synthetic sparse files)* | `unscanned` | `1 (8)` / `2 (16)` | `to_maxtext` | — | **Not run** | — | **N/A** | **N/A** | N/A |
| ↳ | `unscanned` | `1 (8)` / `2 (16)` | **`safetensor`** | — | **—** | **Skipped** *(same capacity limits as scanned)* | **—** | **—** | — |
| **`deepseek3-671b`** *(1,342.4 GB MoE, synthetic sparse files)* | `scanned` | `4 (32)` | `to_maxtext` | — | **Not run** *(single-host conversion of 1.34 TB does not fit)* | — | **N/A** | **N/A** | N/A |
| ↳ | `scanned` | `4 (32)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **649.35** *(Safetensors+Convert)* | **649.35** | **649.35** | **Only `safetensor` loads this config** |
| **`deepseek3-671b`** *(1,342.4 GB MoE, synthetic sparse files)* | `unscanned` | `4 (32)` | `to_maxtext` | — | **Not run** | — | **N/A** | **N/A** | N/A |
| ↳ | `unscanned` | `4 (32)` | **`safetensor`** | **0.00** *(local stage)* | **—** | **530.96** *(Safetensors+Convert)* | **530.96** | **530.96** | **Only `safetensor` loads this config** |

## All tests

**WITH S2** is `safetensors_dynamic` Step 2 (sharded load + transform). **TM S2** is the `to_maxtext` CPU conversion,
run once per scan mode on one host and shared by the 1- and 2-slice loads. **TM S3** is the Orbax load into HBM.

| Model | Config | 1A BEFORE (`origin/main`) | 1A WITH PRs: WITH S2 (load + transform) | 1B TM S2 convert | 1B TM S3 load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen3-235b-a22b` | scanned · 1 | ❌ **timeout at 1800 s**, still loading (peak host RAM 528.0 GB) | ✅ **757.5 s** (674.8 + 82.7) | ❌ doesn't fit one host: 470 GB staged + 470 GB converted state (the full-staging attempt was OOM-killed at leaf 4/15); streaming thrashes | ➖ no checkpoint |
| | unscanned · 1 | ➖ skipped after the scanned timeout | ✅ **697.1 s** (661.8 + 35.2) | 🔄 `tm_235b_unscanned_s1` (streams shards; the full-staging attempt was evicted at 944 GB) | ⏳ `tm_235b_unscanned_s1` |
| | scanned · 2 | ❌ **timeout at 1800 s**, still loading (peak host RAM 300.8 GB) | ✅ **792.4 s** (708.0 + 83.6) | (shared) | ➖ no checkpoint |
| | unscanned · 2 | ➖ skipped after the scanned timeout | ✅ **749.0 s** (704.9 + 44.0) | (shared) | ⏳ `tm_235b_unscanned_s2` |
| `deepseek4-284b` | scanned · 1 | ❌ **timeout at 1800 s**, still loading (peak host RAM 501.4 GB) | ✅ **453.3 s** (366.3 + 86.8) | ❌ OOM-killed at 989.6 GB host RAM | ➖ no checkpoint |
| | unscanned · 1 | ➖ skipped after the scanned timeout | ✅ **414.1 s** (377.0 + 37.0) | ❌ OOM-killed at 989.6 GB host RAM | ➖ no checkpoint |
| | scanned · 2 | ❌ **timeout at 1800 s**, still loading (peak host RAM 576.0 GB) | ✅ **440.7 s** (352.9 + 87.6) | (shared) | ➖ no checkpoint |
| | unscanned · 2 | ➖ skipped after the scanned timeout | ✅ **353.0 s** (306.3 + 46.5) | (shared) | ⏳ `tm_284b_unscanned_s2` (no checkpoint expected) |
| `qwen3.5-397b-a17b` | scanned · 1 | ⚠️ killed at 360 s | ❌ load finished (942.1 s), then TPU HBM OOM in the transform (795 GB ≈ 98% of HBM) | ⏳ `tm_397b_scanned_s1` (expected: doesn't fit one host) | ➖ |
| | unscanned · 1 | ➖ | ➖ does not fit 1 slice | ⏳ `tm_397b_unscanned_s1` (streams per layer) | ⏳ `tm_397b_unscanned_s1` |
| | scanned · 2 | ⚠️ killed at 360 s (WITH load alone takes 484 s) | ❌ load finished (483.8 s), then HBM OOM splitting `wi_0`/`wi_1`. `hbmfix`: load 469.7 s, then the same 11.25 GiB allocation fails (2.77 GiB free) | ➖ | ➖ |
| | unscanned · 2 | ❌ **timeout at 1800 s** while still loading (WITH load: 515.9 s; peak host RAM 301.7 GB vs 98.7 GB WITH) | ✅ **1,311.9 s** (515.9 + 795.7; peak host RAM 98.7 GB) | (shared) | ⏳ `tm_397b_unscanned_s2` |
| | scanned · 4 | ➖ | ✅ **1,399.7 s** (260.0 + 1,139.1; peak host RAM 127.7 GB) | ➖ | ➖ |
| `qwen3-480b-a35b` | scanned · 1 | ❌ `KeyError: 'qwen3-480b-a35b'` | ❌ TPU HBM OOM (961 GB > 813.8 GB) | ❌ doesn't fit one host (all 961 GB must be staged) | ➖ |
| | unscanned · 1 | ❌ `KeyError` (skipped after scanned) | ➖ does not fit 1 slice | ⏳ `tm_480b_unscanned_conv` (expected: the converted state doesn't fit host RAM) | ➖ does not fit 1 slice |
| | scanned · 2 | ❌ `KeyError` | ✅ **1,067.9 s** (942.0 + 125.8) | ➖ | ➖ |
| | unscanned · 2 | ❌ `KeyError` (skipped) | ✅ **992.2 s** (907.7 + 84.3) | (shared) | ⏳ `tm_480b_unscanned_s2` |
| `deepseek3-671b` | scanned · 1 | ⚠️ killed at 360 s | ❌ exceeds 1-slice HBM (1,342 GB > 813.8 GB) | ➖ | ➖ |
| | scanned · 2 | ⚠️ killed at 360 s | ❌ host OOM-killed at 742.1 GB sampled peak; needs ≥ 4 slices | ➖ | ➖ |
| | unscanned · 1 / 2 | ➖ skipped | ➖ same capacity limits | ➖ | ➖ |
| | scanned · 4 | ❌ **timeout at 1800 s**, still loading (peak host RAM 344.3 GB; a first attempt was cut off by a spot preemption) | ✅ **649.4 s** (468.9 + 176.8; peak host RAM 689.9 GB) | ➖ | ➖ |
| | unscanned · 4 | ➖ skipped after the scanned timeout | ✅ **531.0 s** (409.9 + 120.7; peak host RAM 689.9 GB) | ➖ | ➖ |

## Notes
- **Peak host RAM (WITH PRs)**:
  - 235B: 392 GB on 1 slice, 255 GB on 2 slices.
  - 284B: 963–980 GB on 1 slice, 823 GB on 2 slices (synthetic sparse files staged in local RAM).
  - 397B: 99 GB on 2 slices (unscanned).
  - 480B: 416 GB on 2 slices.
  - 671B: 690 GB per host on 4 slices.
- **BEFORE watchdog**: the original BEFORE runs used `PHASE_TIMEOUT_S=360`, shorter than the WITH-PRs load times (306–942 s), so their kills don't show that `origin/main` fails. The 1800 s re-runs (`before_*` tasks) resolve this. Conclusive BEFORE failures so far: `qwen3-480b-a35b` (`KeyError`), and six runs that timed out at 1800 s while still loading: 235B scanned · 1 and · 2, 284B scanned · 1 and · 2, 397B unscanned · 2, 671B scanned · 4. With the PRs, the same load phases take 353–708 s. The harness skips unscanned after a scanned timeout at the same slice count.
- **DeepSeek numbers** use synthetic sparse BF16 files staged on the host (`STAGE_SPARSE_DEEPSEEK_OOM=1`). They measure load/transform mechanics and capacity, not GCS bandwidth.
- **`to_maxtext` S1** is the same HF→GCS download as `safetensors_dynamic` S1, measured once per model (480B's TM S1 was measured separately at 1,135.5 s). TM S2 conversions that re-stage the checkpoint locally now record `hf_to_local_restage_s`.
- **Why `to_maxtext` S2 fails for >200B models on one host**: `to_maxtext` keeps the whole converted state (about the checkpoint size) in host RAM before the Orbax save. A tpu7x host has 1,014.5 GB RAM, of which 986.7 GB is allocatable to pods, and a ~100 GB boot disk, so the source shards can only be staged in RAM.
  - Scanned conversions read every shard for every MoE expert, so they need the whole source staged: source plus converted state exceeds host RAM from 235B up, and streaming through a small shard cache thrashes.
  - Unscanned conversions read one layer at a time, so shards can be streamed. They fit when the converted state alone fits in RAM (235B; 397B expected), but not for 480B (960 GB).
  - 284B (synthetic sparse files) was OOM-killed at 989.6 GB in both scan modes.
- **`qwen3.5-397b-a17b` scanned on 2 slices** runs out of HBM while splitting the stacked `(wi_0, wi_1)` composite (11.25 GiB needed, 2.77 GiB free), both with Lance's PRs and with [`patches/hbmfix`](../patches/hbmfix.patch). The remaining fix is to split each layer before stacking (see [`../handoff.md`](../handoff.md) §7.6). With 4 slices of HBM headroom, scanned loads in 1,399.7 s, but the transform takes 1,139.1 s of that, vs 795.7 s for unscanned on 2 slices.

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
