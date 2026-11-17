# Checkpoint Load Benchmark v2 — 4-Platform Executive Summary (`v7x`, `v6e`, `v5p`, `v5e`)

- **Updated (UTC):** `2026-11-17 09:11 UTC`
- **Git Branch & Commit:** `docs/checkpoint-load-benchmark` (`bbe3128be`; `v5p` + PR `#6319` bundle `8990136d7`)
- **Live Google Sheet v2:** [`1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I`](https://docs.google.com/spreadsheets/d/1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I/edit) (`11` tabs, `615` Summary rows = `516` planned `v2|` + `44` non-overlapping `legacy|` + `17` `wl*` + `38` `v2real*`)

---

## 1. Matrix Scope & Execution Outcome (`516 / 516` Planned `v2` Rows, `0` Errors)

All **21 HuggingFace models** (`0.6B` to `1600B`, real weights, `bfloat16` target load) were benchmarked end-to-end across **4 TPU generations** (`v7x` in `us-central1`, `v6e` in `southamerica-west1`, `v5p` and `v5e` in `europe-west4`), comparing **`safetensors_dynamic`** (direct sharded GCS `.safetensors` read + on-TPU transform) against **`to_maxtext`** (CPU conversion to Orbax + TPU Orbax restore) across `scanned` and `unscanned` layer modes:

| Platform | GKE Cluster & Region | Planned `v2` Rows | `OK` | `OOM-HOST` (`81` TM / `6` SD) | `OOM-HBM` | `SKIPPED` | `ERROR` / `TIMEOUT` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`v7x`** | `bodaborg-tpu7x-nap` (`us-central1`, `tpu7x-standard-4t`, `960 GiB` RAM, `95 GiB/dev` HBM) | `164` | **`142`** | `22` | `0` | `0` | **`0`** |
| **`v6e`** | `bodaborg-v6e-nap` (`southamerica-west1`, `ct6e-standard-8t/-4t`, `1,440/720 GiB` RAM, `32 GiB/dev` HBM) | `140` | **`127`** | `13` | `0` | `0` | **`0`** |
| **`v5p`** | `bodaborg-v5p-nap` (`europe-west4`, `ct5p-hightpu-4t`, `448 GiB` RAM, `95 GiB/dev` HBM) | `164` | **`136`** | `28` | `0` | `0` | **`0`** |
| **`v5e`** | `v5e-256-bodaborg-europe-west4` (`europe-west4`, `ct5lp-hightpu-4t`, `192 GiB` RAM, `16 GiB/dev` HBM) | `48` | **`22`** | `24` | `1` | `1` | **`0`** |
| **Total** | **3 Regions / 4 Clusters (`63 / 63` Cold Step 1 Downloads)** | **`516`** | **`427`** | **`87`** | **`1`** | **`1`** | **`0`** |

- **TPU Backend Verification:** 100% of `OK` load phases (`427` rows) passed pre-load `check_tpu_backend` (`jax_backend_platform = tpu`, exact expected global & per-host device count, `0` `NOT-TPU` fallbacks).
- **Cold Cache & Repetition Rigor:** Every measured phase ran in a fresh process with an empty JAX compilation cache (`jax_compilation_cache_state = empty`). Every Tier A `OK` configuration completed **3/3 repetitions** (`Summary` reports median and `min–max` range) and matched parameter checksums between `to_maxtext` and `safetensors_dynamic`.

---

## 2. Cold Step 1 (`HF -> GCS`, `63 / 63`) & Real-Weight DeepSeek Step 1b (`FP8/FP4 -> BF16`)

Each model's HuggingFace `.safetensors` shards were downloaded cold (`step1_files_already_in_gcs_count = 0`) into a dedicated regional bucket on a 64-vCPU VM (`21` in `us-central1`, `21` in `southamerica-west1`, and `21` in `europe-west4` = `12` measured on `v5e` + `9` Tier B models measured on `v5p`):

| Platform | Regional GCS Bucket | CPU VM | Models Measured | Total Downloaded | Cold Download Range | Median Throughput (Min–Max) |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| `v7x` | `us-central1` | `n2d-standard-64` | `21` | `5,398.6 GB` | `33.70s – 2846.21s` | `0.29 GB/s` (`0.05–0.40`) |
| `v6e` | `southamerica-west1` | `n4-standard-64` | `21` | `5,398.6 GB` | `91.42s – 4854.14s` | `0.14 GB/s` (`0.02–0.25`) |
| `v5e` | `europe-west4` *(Tiers S/A/C)* | `n2-standard-64` | `12` | `5,093.3 GB` | `26.34s – 1774.83s` | `0.56 GB/s` (`0.06–0.60`) |
| `v5p` | `europe-west4` *(Tier B)* | `n2-standard-64` | `9` | `305.3 GB` | `61.99s – 323.76s` | `0.23 GB/s` (`0.10–0.37`) |

### Real-Weight Regional GCS `BF16` DeepSeek Loads (`b/586746548`)

Because official HuggingFace repos for `deepseek4-284b`, `deepseek3-671b`, `deepseek3.2-671b`, and `deepseek4-1600b` publish quantized `FP8` / `MXFP4` weights while MaxText expects `BF16` (tracked in http://b/586746548), Step 1b staged dequantized `BF16` checkpoints (`581.9 GB` to `3,197.7 GB`) into each regional `hf_cache_bf16/` bucket:
- **`v7x` (`v2real`, `14 / 14 OK`):** `284B` **`855.16–985.29s`**, `671B` **`1,209.69–1,543.10s`**, `3.2-671B` **`1,235.74–1,606.99s`**, `1600B` (`1×2x4x4`) **`2,493.02s`** (`scanned`) / **`2,203.11s`** (`unscanned`).
- **`v5p` (`v2` with PR `#6319`, `10 / 10 OK`, `0` OOMs):** `284B` `1×2x2x4` `scanned` **`1009.78s`**, `671B` `1×2x4x4` `scanned` **`1586.84s`**, `3.2-671B` `2×2x2x4` **`1,377.59s / 1,383.03s`**, `1600B` `1×4x4x4` **`2675.04s`** (`scanned`) / **`2540.00s`** (`unscanned`).
- **`v5e` (`v2real-gcsfs` with PR `#6319`):** `284B` **`390.59s / 313.77s`**, `671B` **`620.69s / 452.39s`**, `3.2-671B` **`638.18s / 474.81s`**, `1600B` `2×16x16` `unscanned` **`1,245.19s` `OK`** (`scanned` genuine `OOM-HBM` at `16.77 GB > 16 GiB/chip`).
- **`v6e`:** `1×8x8` `284B` **`1,227.60s / 1,148.10s` `OK`** (`1×16x16` `671B`/`3.2-671B`/`1600B` queued in `mesa-q-v6e-16x16-1s-g1` awaiting 256 `v6e` chips).

---

## 3. Pipeline Speedups (`to_maxtext ÷ safetensors_dynamic`) Across `175` Paired `OK` Configurations

Across the **`175`** real-weight configurations where both pipelines fit in Host RAM and TPU HBM:
- **Without HF Download (`(TM S1 + S2 + S3) ÷ (SD S1 + S2)`):** Faster in **`168 / 175`** pairs — median **`4.14x`** (`0.25x – 26.76x`) faster with `safetensors_dynamic`, because `safetensors_dynamic` eliminates `to_maxtext`'s single-host CPU conversion + Orbax GCS write (`Step 2`).
  - By platform: `v7x` median **`4.61x`**, `v6e` median **`5.53x`**, `v5p` median **`2.63x`**, `v5e` median **`3.11x`**.
- **End-to-End Including Cold HF Download & Step 1b:** Median **`1.81x`** (`0.30x – 4.88x`) faster with `safetensors_dynamic`.
- **Repeated Warm Load Only (`TM Step 3 ÷ SD Step 2`):** Median **`0.35x`** (`0.03x – 10.42x`). Pre-converted Orbax `Step 3` is faster on `unscanned` checkpoints and `<30B` `scanned` checkpoints, while `safetensors_dynamic` is `2x–10.4x` faster on larger `scanned` models (`30B–120B`).

---

## 4. Capacity Boundaries & Architectural Limits Across 4 TPU Generations

1. **Single-Host CPU `to_maxtext` RAM Wall (`81` `OOM-HOST` rows) vs Sharded `safetensors_dynamic` (`0` OOMs on `v7x`, `v6e`, `v5p`):**
   - **`v5p` (`448 GiB` / `473.3 GB` Host RAM, `95 GiB/dev` HBM):** `to_maxtext` fails with `OOM-HOST` on **`28` configurations** (`gpt-oss-120b` `scanned` + all 7 models `>= 235B`), whereas **`safetensors_dynamic` (with PR `#6319`) achieved `82 / 82` (`100%`) `OK` with `0` Host RAM or HBM OOMs** across every model from `0.6B` to `1600B` (`3.20 TB` BF16)!
   - **`v7x` (`960 GiB` Host RAM, `95 GiB/dev` HBM):** `to_maxtext` `OOM-HOST` on `22` rows (`scanned` `>= 235B` and `unscanned` `>= 480B`), vs `0` OOMs on `safetensors_dynamic` (`14/14` real-weight DeepSeek `OK` up to `1600B`).
   - **`v6e` (`720/1,440 GiB` Host RAM, `32 GiB/dev` HBM):** `to_maxtext` `OOM-HOST` on `13` rows, vs `0` OOMs on `safetensors_dynamic`.
   - **`v5e` (`192 GiB` Host RAM, `16 GiB/dev` HBM):** `to_maxtext` `OOM-HOST` on `18` rows (`>= 70B` `scanned`, `>= 120B` `unscanned`). Baseline `tf.io.gfile` hit `OOM-HOST` on unfused MoE models (`6` `v2|` + `8` `v2real|` rows), completely resolved by PR `#6319` (`gcsfs` reader) in `wl-pr6319|` and `v2real-gcsfs|` (`1600B` `unscanned` loads in `1,245.19s` at `102.3 GB` RAM; `1600B` `scanned` hits genuine `OOM-HBM` at `16.77 GB > 16 GiB/chip`).
2. **Log & Reproducibility Provenance:**
   - Every executed row in `Summary` and `Runs` links to a public `gpaste` log (`https://paste.googleplex.com/...`) plus GCS JSON/log URIs, package versions, hardware specs, and `CODE_VERSION.json` commit/SHA-256.
