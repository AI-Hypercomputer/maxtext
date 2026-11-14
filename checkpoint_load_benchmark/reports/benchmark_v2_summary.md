# Checkpoint Load Benchmark v2 — Multi-Platform Executive Summary (`v7x`, `v6e`, `v5e`)

- **Updated (UTC):** `2026-11-14 07:24 UTC`
- **Git Branch & Commit:** `docs/checkpoint-load-benchmark` (`a230891c8`)
- **Live Google Sheet v2:** [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit) (`11` tabs, `396` Summary rows = `352` v2 + `44` non-overlapping legacy v1; `1498` phase records in `Runs`)

---

## 1. Matrix Scope & Execution Outcome (`352 / 352` Planned `v2` Rows, `0` Errors)

All **21 HuggingFace models** (`0.6B` to `1600B`, real weights, `bfloat16` target load) were benchmarked end-to-end across **3 TPU generations** (`v7x` in `us-central1`, `v6e` in `southamerica-west1`, `v5e` in `europe-west4`), comparing **`safetensors_dynamic`** (direct sharded GCS `.safetensors` read + on-TPU transform) against **`to_maxtext`** (CPU conversion to Orbax + TPU Orbax restore) across `scanned` and `unscanned` layer modes:

| Platform | GKE Cluster & Region | Planned `v2` Rows | `OK` | `OOM-HOST` (`53` TM / `6` SD) | `OOM-HBM` | `SKIPPED` | `ERROR` / `TIMEOUT` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`v7x`** | `bodaborg-tpu7x-nap` (`us-central1`, `tpu7x-standard-4t`, `960 GiB` RAM, `95 GiB/dev` HBM) | `164` | **`142`** | `22` | `0` | `0` | **`0`** |
| **`v6e`** | `bodaborg-v6e-nap` (`southamerica-west1`, `ct6e-standard-8t/-4t`, `1,440/720 GiB` RAM, `32 GiB/dev` HBM) | `140` | **`127`** | `13` | `0` | `0` | **`0`** |
| **`v5e`** | `v5e-256-bodaborg-europe-west4` (`europe-west4`, `ct5lp-hightpu-4t`, `192 GiB` RAM, `16 GiB/dev` HBM) | `48` | **`22`** | `24` | `1` | `1` | **`0`** |
| **Total** | **3 Regions / 3 Clusters (`54 / 54` Cold Step 1 Downloads)** | **`352`** | **`291`** | **`59`** | **`1`** | **`1`** | **`0`** |

- **TPU Backend Verification:** 100% of `OK` load phases (`291` rows) passed pre-load `check_tpu_backend` (`jax_backend_platform = tpu`, exact expected global & per-host device count, `0` `NOT-TPU` fallbacks).
- **Cold Cache & Repetition Rigor:** Every measured phase ran in a fresh process with an empty JAX compilation cache (`jax_compilation_cache_state = empty`). Every Tier A `OK` configuration completed **3/3 repetitions** (`Summary` reports median and `min–max` range) and matched parameter checksums between `to_maxtext` and `safetensors_dynamic`.

---

## 2. Cold Step 1 (`HF -> GCS`) & DeepSeek Step 1b (`FP8/FP4 -> BF16`)

Each model's HuggingFace `.safetensors` shards were downloaded cold (`step1_files_already_in_gcs_count = 0`) into a dedicated regional bucket on a 64-vCPU VM:

| Platform | Regional GCS Bucket | CPU VM | Models | Total Downloaded | Cold Download Range | Median Throughput (Min–Max) |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| `v7x` | `us-central1` | `n2d-standard-64` | `21` | `5,398.6 GB` | `33.70s – 2846.21s` | `0.29 GB/s` (`0.05–0.40`) |
| `v6e` | `southamerica-west1` | `n4-standard-64` | `21` | `5,398.6 GB` | `91.42s – 4854.14s` | `0.14 GB/s` (`0.02–0.25`) |
| `v5e` | `europe-west4` | `n2-standard-64` | `12` | `5,093.3 GB` | `26.34s – 1774.83s` | `0.56 GB/s` (`0.06–0.60`) |

Because neither MaxText loader natively dequantizes DeepSeek's block-scaled `FP8` / `FP4` expert tensors on the fly, the 4 large DeepSeek models (`284B`, `671B`, `3.2-671B`, `1600B`) include **Step 1b** (`FP8/FP4 -> BF16` conversion or regional GCS transfer):

| Platform | Region | Model | BF16 Size | BF16 Source | CPU Convert (`s`) | Regional Copy (`s`) | Step 1b Total |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `v7x` | `us-central1` | `deepseek4-284b` | `581.9023 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `912.2054` | **`912.21s`** |
| `v7x` | `us-central1` | `deepseek3-671b` | `1368.9897 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `2118.6001` | **`2118.60s`** |
| `v7x` | `us-central1` | `deepseek3.2-671b` | `1370.7229 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `2399.2419` | **`2399.24s`** |
| `v7x` | `us-central1` | `deepseek4-1600b` | `3197.7139 GB` | `converted_by_this_benchmark` | `10415.6084` | `0.0` | **`10675.89s`** |
| `v6e` | `southamerica-west1` | `deepseek4-284b` | `581.9023 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `1051.7211` | **`1051.72s`** |
| `v6e` | `southamerica-west1` | `deepseek3-671b` | `1368.9897 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `3471.0423` | **`3471.04s`** |
| `v6e` | `southamerica-west1` | `deepseek3.2-671b` | `1370.7229 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `2826.7867` | **`2826.79s`** |
| `v6e` | `southamerica-west1` | `deepseek4-1600b` | `3197.7139 GB` | `converted_by_this_benchmark` | `N/A` | `5803.2998` | **`5803.30s`** |
| `v5e` | `europe-west4` | `deepseek4-284b` | `581.9023 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `967.5779` | **`967.58s`** |
| `v5e` | `europe-west4` | `deepseek3-671b` | `1368.9897 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `2777.4455` | **`2777.45s`** |
| `v5e` | `europe-west4` | `deepseek3.2-671b` | `1370.7229 GB` | `copied_from_maxtext_team_bucket` | `N/A` | `2799.973` | **`2799.97s`** |
| `v5e` | `europe-west4` | `deepseek4-1600b` | `3197.7139 GB` | `converted_by_this_benchmark` | `N/A` | `5300.9783` | **`5300.98s`** |

---

## 3. Pipeline Speedups (`to_maxtext ÷ safetensors_dynamic`) Across `121` Paired `OK` Configurations

Across the **`121`** configurations where both pipelines fit in Host RAM and TPU HBM:
- **Without HF Download (`(TM S1 + S2 + S3) ÷ (SD S1 + S2)`):** Median **`4.99x`** (`0.53x – 26.76x`) faster with `safetensors_dynamic`, because `safetensors_dynamic` eliminates `to_maxtext`'s single-host CPU conversion + Orbax GCS write (`Step 2`).
  - By platform: `v7x` median **`4.61x`**, `v6e` median **`5.53x`**, `v5e` median **`3.11x`**.
- **End-to-End Including Cold HF Download & Step 1b:** Median **`1.83x`** (`0.72x – 3.95x`) faster with `safetensors_dynamic`.
- **Repeated Warm Load Only (`TM Step 3 ÷ SD Step 2`):** Median **`0.49x`** (`0.08x – 10.42x`). Once an Orbax OCDBT checkpoint has already been converted and saved (`Step 2` amortized), restoring pre-sharded Orbax tensors in `Step 3` is typically ~2× faster than re-reading `.safetensors` and transforming weights on TPU (especially on 64-host `v5e 16x16`), except on smaller `unscanned` models where Orbax per-leaf metadata overhead exceeds direct `safetensors_dynamic` streaming (up to `10.42x`).

---

## 4. Representative Results Across Scales (`0.6B` to `1600B`)

| Platform | Model | Scan | Topology | BF16 Size | `to_maxtext` S2 (CPU) | `to_maxtext` S3 (Load) | `safetensors_dynamic` Load | Speedup (no HF) | Speedup (E2E) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v7x` | `qwen3-0.6b` | `scanned` | `1×2x2x1` | `1.5033 GB` | `13.96s` | `2.61s` | **`16.95s`** | **`2.09x`** | `1.36x` |
| `v7x` | `llama3.1-70b` | `scanned` | `1×2x2x1` | `141.1075 GB` | `867.80s` | `794.42s` | **`255.48s`** | **`6.87x`** | `3.30x` |
| `v7x` | `qwen3-235b-a22b` | `unscanned` | `1×2x2x1` | `470.1919 GB` | `3042.96s` | `685.46s` | **`733.03s`** | **`5.09x`** | `2.56x` |
| `v7x` | `qwen3.5-397b-a17b` | `unscanned` | `2×2x2x1` | `806.7962 GB` | `7930.76s` | `1952.53s` | **`1270.34s`** | **`7.78x`** | `3.09x` |
| `v7x` | `qwen3.5-397b-a17b` | `unscanned` | `1×2x2x2` | `806.7962 GB` | `7877.26s` | `2065.84s` | **`1120.93s`** | **`8.87x`** | `3.22x` |
| `v7x` | `deepseek3-671b` | `scanned` | `1×2x2x4` | `1369.0 GB` | `OOM-HOST` | `BLOCKED` | **`829.57s`** | **`—`** | `—` |
| `v7x` | `deepseek4-1600b` | `unscanned` | `1×2x4x4` | `3200.0 GB` | `OOM-HOST` | `BLOCKED` | **`873.83s`** | **`—`** | `—` |
| `v6e` | `llama3.1-70b` | `scanned` | `1×2x4` | `141.1075 GB` | `616.54s` | `316.06s` | **`153.39s`** | **`6.51x`** | `1.71x` |
| `v6e` | `qwen3-480b-a35b` | `scanned` | `1×8x8` | `961.0 GB` | `OOM-HOST` | `BLOCKED` | **`1173.55s`** | **`—`** | `—` |
| `v6e` | `deepseek3-671b` | `unscanned` | `1×16x16` | `1369.0 GB` | `OOM-HOST` | `BLOCKED` | **`188.70s`** | **`—`** | `—` |
| `v6e` | `deepseek4-1600b` | `unscanned` | `1×16x16` | `3200.0 GB` | `OOM-HOST` | `BLOCKED` | **`368.36s`** | **`—`** | `—` |
| `v5e` | `qwen3-30b-a3b` | `scanned` | `1×16x16` | `61.1 GB` | `157.55s` | `727.60s` | **`OOM-HOST`** | **`—`** | `—` |
| `v5e` | `llama3.1-70b` | `unscanned` | `1×16x16` | `141.1 GB` | `OOM-HOST` | `BLOCKED` | **`117.10s`** | **`—`** | `—` |
| `v5e` | `qwen3.5-397b-a17b` | `unscanned` | `1×16x16` | `795.5 GB` | `OOM-HOST` | `BLOCKED` | **`1029.50s`** | **`—`** | `—` |
| `v5e` | `deepseek3-671b` | `unscanned` | `1×16x16` | `1369.0 GB` | `OOM-HOST` | `BLOCKED` | **`234.22s`** | **`—`** | `—` |

---

## 5. Capacity Boundaries & Architectural Limits

1. **Host RAM Limits (`53` `OOM-HOST` rows on `to_maxtext` vs `6` on `v5e` `safetensors_dynamic` / `0` on `v7x` & `v6e` `safetensors_dynamic`):**
   - `to_maxtext.py` runs strictly on a **single host's CPU** (`jax_platforms=cpu`), requiring Host RAM ≈ `1.1×` checkpoint size (`unscanned`) and ≈ `2.0×` (`scanned` layer stacking).
   - Consequently, `to_maxtext` Step 2 fails with `OOM-HOST` (`53` rows total) on:
     - **`v5e` (`192 GiB` Host RAM):** Every model `>= 70B` (`141 GB+`) in `scanned` and `>= 120B` (`234 GB+`) in `unscanned` (`18` `to_maxtext` `OOM-HOST` rows). Note that `v5e` `1×16x16` (`64` hosts × `192 GiB` RAM) also hits `OOM-HOST` in `safetensors_dynamic` on the 3 unfused per-expert Qwen3 MoE checkpoints (`qwen3-30b-a3b`, `qwen3-235b-a22b`, `qwen3-480b-a35b` = `6` rows, tens of thousands of tiny 2D expert keys), whereas fused-expert models shard cleanly across the leading expert axis and **`safetensors_dynamic` loads up to `1,371 GB` (`deepseek3.2-671b`) in `239.5s` (`unscanned`) / `435.8s` (`scanned`)** on `v5e 1×16x16`!
     - **`v6e` (`720 GiB` on `4t` / `1,440 GiB` on `8t`):** `to_maxtext` `OOM-HOST` (`13` rows) on models `>= 397B` (`scanned`) / `>= 480B–671B` (`unscanned` on `4t`), vs **`0` `OOM-HOST` on `safetensors_dynamic`**, which loads `deepseek4-1600b` (`3,200 GB` BF16) on `1×16x16` in **`368.4s` (`unscanned`) / `1,137.9s` (`scanned`)**!
     - **`v7x` (`960 GiB` Host RAM):** `to_maxtext` `OOM-HOST` (`22` rows) on `scanned` `>= 235B` and `unscanned` `>= 480B` (`397B unscanned` fits at `~884 GB` peak Host RAM in `~7,877–7,931s`), vs **`0` `OOM-HOST` on `safetensors_dynamic`**, which loads `deepseek4-1600b` (`3,200 GB`) on `1×2x4x4` (`64` `TPU7x` cores) in **`873.8s` (`unscanned`) / `1,087.2s` (`scanned`)**!
2. **TPU HBM Headroom (`1` `OOM-HBM` + `1` `SKIPPED` row):**
   - On `v5e` `2×16x16` (`512` chips × `16 GiB` = `8,192 GiB` total HBM), `deepseek4-1600b` (`3,200 GB` weights, `6.25 GB/device` steady-state) hits `OOM-HBM` during `safetensors_dynamic` MoE expert resharding/stacking peak on `16 GiB` devices (`unscanned` skipped per identical per-layer expert HBM peak); it succeeds cleanly on `v6e` `1×16x16` (`32 GiB/dev`, peak `21.1–30.3 GB/dev`) and `v7x` `1×2x4x4` (`95 GiB/dev`, peak `58.8–66.3 GB/dev`).
3. **Log & Reproducibility Provenance:**
   - Every executed row in `Summary` and `Runs` links to a public `gpaste` log (`https://paste.googleplex.com/...`) plus GCS JSON/log URIs, package versions, hardware specs, and `CODE_VERSION.json` commit/SHA-256.
