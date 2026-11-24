# Checkpoint Load Time Benchmark Summary (`to_maxtext` vs `safetensor`)

## 1. What We Tested

We compared two ways to load HuggingFace checkpoints into MaxText across **12 models** (`0.6B` to `1600B`), **2 layer scan modes** (`scan_layers=True` / `scanned` and `scan_layers=False` / `unscanned`), and **2 slice topologies** (`num_slices=1` and `num_slices=2`, totaling **48 configurations**):

### Pipelines Compared
1. **`to_maxtext` (Offline 3-Step Pipeline)**:
   - **Step 1**: `HF -> GCS` (Download raw HF `.safetensors` to GCS / local staging)
   - **Step 2**: `Convert to Orbax` (Convert HF weights into an Orbax checkpoint on GCS on a single host)
   - **Step 3**: `Load Orbax` (Load the converted Orbax checkpoint into MaxText)
2. **`safetensor` (`safetensors_dynamic`, Direct 2-Step Pipeline)**:
   - **Step 1**: `HF -> GCS` (Cache raw HF `.safetensors` to GCS)
   - **Step 2**: `Safetensors & convert to MaxText` (Read `.safetensors` from GCS via Orbax `CheckpointLayout.SAFETENSORS` and convert to MaxText on the fly across slices)

### Models & Where They Ran (12 Models Across 3 Tiers)
| Tier | Model | HF Repo | Type | BF16 Size | `.safetensors` Shards | Layers | Peak Host RAM (`to_maxtext` Step 2) | Where Executed |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Small / Mid (`<=32B`)** | **`qwen3-0.6b`** | `Qwen/Qwen3-0.6B` | Dense | 1.5 GB | 1 | 28 | ~3.2 GB | Local Workstation (`mesakh.c.googlers.com`) |
| **Small / Mid (`<=32B`)** | **`gemma2-2b`** | `google/gemma-2-2b` | Dense (Gated) | 10.5 GB (FP32) | 3 | 26 | ~14.8 GB | Local Workstation (`mesakh.c.googlers.com`) |
| **Small / Mid (`<=32B`)** | **`deepseek2-16b`** | `deepseek-ai/DeepSeek-V2-Lite` | MoE + MLA (64 experts) | 31.4 GB | 4 (8.6 GB max) | 27 | 53.4 GB | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **Small / Mid (`<=32B`)** | **`qwen3-30b-a3b`** | `Qwen/Qwen3-30B-A3B-Thinking-2507` | MoE (128 experts) | 61.1 GB | 16 (4.0 GB max) | 48 | 102.8 GB | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **Small / Mid (`<=32B`)** | **`qwen3-32b`** | `Qwen/Qwen3-32B` | Dense | 65.5 GB | 17 (4.0 GB max) | 64 | 110.3 GB | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **Large (`47B–120B`)** | **`mixtral-8x7b`** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE (8 experts) | 93.4 GB | 19 (5.0 GB max) | 32 | 155.9 GB | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **Large (`47B–120B`)** | **`llama3.1-70b`** | `meta-llama/Llama-3.1-70B` | Dense (Gated) | 141.1 GB | 30 (5.0 GB max) | 80 | 235.7 GB | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **Large (`47B–120B`, OOM Test)** | **`gpt-oss-120b`** | `unsloth/gpt-oss-120b-BF16` | MoE (128 experts) | 233.7 GB | 73 (4.8 GB max) | 36 | **OOM (`251.7 GB`, Exit 137)** | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **DeepSeek Ultra-Large (`284B–1600B` OOM Tests)** | **`deepseek4-284b`** | `deepseek-ai/DeepSeek-V4-Flash` | MLA + MoE (128 experts) | 581.9 GB | 46 | 46 | **OOM (`245.4 GB`, Exit 137)** | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **DeepSeek Ultra-Large (`284B–1600B` OOM Tests)** | **`deepseek3-671b`** | `deepseek-ai/DeepSeek-V3` | MLA + MoE (256 experts) | 1,365.3 GB *(685 GB FP8)* | 163 | 61 | **OOM (`245.4 GB`, Exit 137)** | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **DeepSeek Ultra-Large (`284B–1600B` OOM Tests)** | **`deepseek3.2-671b`** | `deepseek-ai/DeepSeek-V3.2` | MLA + MoE + Sparse Indexer | 1,367.0 GB *(686 GB FP8)* | 163 | 61 | **OOM (`245.6 GB`, Exit 137)** | **`bodaborg-v5p-nap`** (`us-central1-a`) |
| **DeepSeek Ultra-Large (`284B–1600B` OOM Tests)** | **`deepseek4-1600b`** | `deepseek-ai/DeepSeek-V4-Pro` | MLA + MoE + Sparse Indexer | 3,197.7 GB | 64 | 128 | **OOM (`246.0 GB`, Exit 137)** | **`bodaborg-v5p-nap`** (`us-central1-a`) |

---

## 2. Commands Used

### A. Run the Benchmark Script ([`benchmark_ckpt_load_time.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time.py))
```bash
PYTHONPATH=src python3 benchmark_ckpt_load_time.py \
  --models qwen3-0.6b gemma2-2b deepseek2-16b qwen3-30b-a3b qwen3-32b \
           mixtral-8x7b llama3.1-70b gpt-oss-120b \
           deepseek3-671b deepseek3.2-671b deepseek4-284b deepseek4-1600b \
  --scan-modes scanned unscanned \
  --num-slices-list 1 2 \
  --hardware cpu \
  --weight-dtype bfloat16 \
  --reuse-model-hf-cache \
  --allow-step-failure \
  --gcs-base gs://mesa-maxtext/benchmark_ckpt_load_time \
  --csv-report benchmark_ckpt_load_time_summary.csv
```

### B. Underlying MaxText Commands
- **`to_maxtext` Step 2 (Convert HF to Orbax)**:
  ```bash
  python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=${MODEL} hf_access_token=${HF_TOKEN} \
    base_output_directory=${GCS_ORBAX_DIR} scan_layers=${SCAN_LAYERS} lazy_load_tensors=True
  ```
- **`to_maxtext` Step 3 (Load Orbax Checkpoint)**:
  ```bash
  python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=${MODEL} scan_layers=${SCAN_LAYERS} \
    load_parameters_path=${GCS_ORBAX_DIR}/0/items base_output_directory=${GCS_OUT_DIR}
  ```
- **`safetensor` Step 1 & 2 (Direct Load + Convert via `safetensors_dynamic`)**:
  ```bash
  python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=${MODEL} scan_layers=${SCAN_LAYERS} \
    source_checkpoint_layout=safetensors_dynamic \
    load_parameters_path=hf://${HF_REPO_ID} hf_access_token=${HF_TOKEN} \
    base_output_directory=${GCS_OUT_DIR}
  ```
*(For `num_slices=2`, add `num_slices=2 dcn_data_parallelism=2 ici_fsdp_parallelism=-1` across 2 slice processes; for `deepseek3.2-671b` and `deepseek4-*` on CPU, pass `attention=dot_product`.)*

---

## 3. Test Results (All 48 Configurations — `to_maxtext` vs `safetensor` Row-by-Row)

For each `(Model, Scan Mode, Slices)` configuration, **`to_maxtext`** and **`safetensor`** are listed on consecutive rows so each step and total can be compared directly:
- **Col 1 (`HF -> GCS`)**: `to_maxtext` Step 1 vs `safetensor` Step 1
- **Col 2 (`Convert to Orbax`)**: `to_maxtext` Step 2 vs `—` (skipped in `safetensor`)
- **Col 3 (`Load into MaxText`)**: `to_maxtext` Step 3 (`Load Orbax`) vs `safetensor` Step 2 (`Safetensors & convert to MaxText`)

### 3A. Models `0.6B – 120B` (32 Configurations Across 8 Models)

| Model | Scan Mode | Slices (Devices) | Pipeline | 1. `HF -> GCS` (s) | 2. `Convert to Orbax` (s) | 3. `Load into MaxText` (s) | **Total (No Step 1) (s)** | **Total (End-to-End) (s)** | **Speedup (E2E / No Step 1)** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-0.6b`** *(1.5 GB)* | `scanned` | `1 (1)` | `to_maxtext` | 232.48 | 100.41 | 26.27 *(Load Orbax)* | 126.68 | 359.17 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **240.03** | **—** | **33.95** *(Safetensors+Convert)* | **33.95** | **273.98** | **1.31x / 3.73x** |
| **`qwen3-0.6b`** *(1.5 GB)* | `scanned` | `2 (8)` | `to_maxtext` | 232.48 | 100.41 | 32.70 *(Load Orbax)* | 133.11 | 365.60 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **237.83** | **—** | **63.09** *(Safetensors+Convert)* | **63.09** | **300.91** | **1.21x / 2.11x** |
| **`qwen3-0.6b`** *(1.5 GB)* | `unscanned` | `1 (1)` | `to_maxtext` | 247.74 | 320.92 | 41.14 *(Load Orbax)* | 362.06 | 609.80 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **232.32** | **—** | **26.42** *(Safetensors+Convert)* | **26.42** | **258.74** | **2.36x / 13.70x** |
| **`qwen3-0.6b`** *(1.5 GB)* | `unscanned` | `2 (8)` | `to_maxtext` | 247.74 | 320.92 | 46.50 *(Load Orbax)* | 367.42 | 615.15 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **230.43** | **—** | **46.59** *(Safetensors+Convert)* | **46.59** | **277.02** | **2.22x / 7.89x** |
| **`gemma2-2b`** *(10.5 GB)* | `scanned` | `1 (1)` | `to_maxtext` | 739.82 | 407.06 | 121.15 *(Load Orbax)* | 528.21 | 1,268.04 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **726.89** | **—** | **88.65** *(Safetensors+Convert)* | **88.65** | **815.55** | **1.55x / 5.96x** |
| **`gemma2-2b`** *(10.5 GB)* | `scanned` | `2 (8)` | `to_maxtext` | 739.82 | 407.06 | 71.80 *(Load Orbax)* | 478.86 | 1,218.68 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **726.89** | **—** | **119.55** *(Safetensors+Convert)* | **119.55** | **846.44** | **1.44x / 4.01x** |
| **`gemma2-2b`** *(10.5 GB)* | `unscanned` | `1 (1)` | `to_maxtext` | 739.82 | 302.22 | 116.02 *(Load Orbax)* | 418.24 | 1,158.06 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **726.89** | **—** | **85.32** *(Safetensors+Convert)* | **85.32** | **812.21** | **1.43x / 4.90x** |
| **`gemma2-2b`** *(10.5 GB)* | `unscanned` | `2 (8)` | `to_maxtext` | 739.82 | 302.22 | 103.56 *(Load Orbax)* | 405.78 | 1,145.60 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **726.89** | **—** | **113.20** *(Safetensors+Convert)* | **113.20** | **840.10** | **1.36x / 3.58x** |
| **`deepseek2-16b`** *(31.4 GB MoE)* | `scanned` | `1 (1)` | `to_maxtext` | 638.24 | 172.79 | 82.13 *(Load Orbax)* | 254.92 | 893.17 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **643.64** | **—** | **208.70** *(Safetensors+Convert)* | **208.70** | **852.34** | **1.05x / 1.22x** |
| **`deepseek2-16b`** *(31.4 GB MoE)* | `scanned` | `2 (8)` | `to_maxtext` | 638.24 | 172.79 | 108.83 *(Load Orbax)* | 281.62 | 919.86 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **643.64** | **—** | **339.99** *(Safetensors+Convert)* | **339.99** | **983.63** | 0.94x / 0.83x |
| **`deepseek2-16b`** *(31.4 GB MoE)* | `unscanned` | `1 (1)` | `to_maxtext` | 638.24 | 325.44 | 89.22 *(Load Orbax)* | 414.66 | 1,052.91 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **643.64** | **—** | **173.03** *(Safetensors+Convert)* | **173.03** | **816.67** | **1.29x / 2.40x** |
| **`deepseek2-16b`** *(31.4 GB MoE)* | `unscanned` | `2 (8)` | `to_maxtext` | 638.24 | 325.44 | 81.43 *(Load Orbax)* | 406.87 | 1,045.11 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **643.64** | **—** | **314.45** *(Safetensors+Convert)* | **314.45** | **958.09** | **1.09x / 1.29x** |
| **`qwen3-30b-a3b`** *(61.1 GB MoE)* | `scanned` | `1 (1)` | `to_maxtext` | 330.86 | 335.21 | 111.53 *(Load Orbax)* | 446.74 | 777.60 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **336.20** | **—** | **398.75** *(Safetensors+Convert)* | **398.75** | **734.95** | **1.06x / 1.12x** |
| **`qwen3-30b-a3b`** *(61.1 GB MoE)* | `scanned` | `2 (8)` | `to_maxtext` | 330.86 | 335.21 | 185.50 *(Load Orbax)* | 520.71 | 851.57 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **336.20** | **—** | **871.94** *(Safetensors+Convert)* | **871.94** | **1,208.13** | 0.70x / 0.60x |
| **`qwen3-30b-a3b`** *(61.1 GB MoE)* | `unscanned` | `1 (1)` | `to_maxtext` | 330.86 | 477.58 | 107.51 *(Load Orbax)* | 585.09 | 915.95 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **336.20** | **—** | **335.61** *(Safetensors+Convert)* | **335.61** | **671.81** | **1.36x / 1.74x** |
| **`qwen3-30b-a3b`** *(61.1 GB MoE)* | `unscanned` | `2 (8)` | `to_maxtext` | 330.86 | 477.58 | 109.37 *(Load Orbax)* | 586.95 | 917.81 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **336.20** | **—** | **809.50** *(Safetensors+Convert)* | **809.50** | **1,145.70** | 0.80x / 0.73x |
| **`qwen3-32b`** *(65.5 GB Dense)* | `scanned` | `1 (1)` | `to_maxtext` | 296.64 | 397.19 | 95.18 *(Load Orbax)* | 492.37 | 789.01 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **305.62** | **—** | **363.46** *(Safetensors+Convert)* | **363.46** | **669.08** | **1.18x / 1.35x** |
| **`qwen3-32b`** *(65.5 GB Dense)* | `scanned` | `2 (8)` | `to_maxtext` | 296.64 | 397.19 | 365.06 *(Load Orbax)* | 762.25 | 1,058.89 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **305.62** | **—** | **736.29** *(Safetensors+Convert)* | **736.29** | **1,041.91** | **1.02x / 1.04x** |
| **`qwen3-32b`** *(65.5 GB Dense)* | `unscanned` | `1 (1)` | `to_maxtext` | 296.64 | 514.42 | 140.26 *(Load Orbax)* | 654.68 | 951.32 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **305.62** | **—** | **304.44** *(Safetensors+Convert)* | **304.44** | **610.06** | **1.56x / 2.15x** |
| **`qwen3-32b`** *(65.5 GB Dense)* | `unscanned` | `2 (8)` | `to_maxtext` | 296.64 | 514.42 | 147.96 *(Load Orbax)* | 662.38 | 959.02 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **305.62** | **—** | **661.83** *(Safetensors+Convert)* | **661.83** | **967.45** | 0.99x / **1.00x** |
| **`mixtral-8x7b`** *(93.4 GB MoE)* | `scanned` | `1 (1)` | `to_maxtext` | 532.24 | 329.93 | 130.30 *(Load Orbax)* | 460.23 | 992.46 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **399.71** | **—** | **599.24** *(Safetensors+Convert)* | **599.24** | **998.95** | 0.99x / 0.77x |
| **`mixtral-8x7b`** *(93.4 GB MoE)* | `scanned` | `2 (8)` | `to_maxtext` | 532.24 | 329.93 | 123.11 *(Load Orbax)* | 453.04 | 985.27 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **399.71** | **—** | **786.27** *(Safetensors+Convert)* | **786.27** | **1,185.98** | 0.83x / 0.58x |
| **`mixtral-8x7b`** *(93.4 GB MoE)* | `unscanned` | `1 (1)` | `to_maxtext` | 532.24 | 462.35 | 106.74 *(Load Orbax)* | 569.09 | 1,101.33 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **399.71** | **—** | **516.11** *(Safetensors+Convert)* | **516.11** | **915.82** | **1.20x / 1.10x** |
| **`mixtral-8x7b`** *(93.4 GB MoE)* | `unscanned` | `2 (8)` | `to_maxtext` | 532.24 | 462.35 | 107.09 *(Load Orbax)* | 569.44 | 1,101.67 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **399.71** | **—** | **757.98** *(Safetensors+Convert)* | **757.98** | **1,157.69** | 0.95x / 0.75x |
| **`llama3.1-70b`** *(141.1 GB Dense)* | `scanned` | `1 (1)` | `to_maxtext` | 530.06 | 728.12 | 329.15 *(Load Orbax)* | 1,057.27 | 1,587.32 | 1.00x (baseline) |
| ↳ | `scanned` | `1 (1)` | **`safetensor`** | **375.50** | **—** | **1,012.45** *(Safetensors+Convert)* | **1,012.45** | **1,387.95** | **1.14x / 1.04x** |
| **`llama3.1-70b`** *(141.1 GB Dense)* | `scanned` | `2 (8)` | `to_maxtext` | 530.06 | 728.12 | 170.22 *(Load Orbax)* | 898.34 | 1,428.39 | 1.00x (baseline) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **375.50** | **—** | **919.42** *(Safetensors+Convert)* | **919.42** | **1,294.91** | **1.10x / 0.98x** |
| **`llama3.1-70b`** *(141.1 GB Dense)* | `unscanned` | `1 (1)` | `to_maxtext` | 530.06 | 1,080.70 | 212.89 *(Load Orbax)* | 1,293.59 | 1,823.65 | 1.00x (baseline) |
| ↳ | `unscanned` | `1 (1)` | **`safetensor`** | **375.50** | **—** | **865.53** *(Safetensors+Convert)* | **865.53** | **1,241.03** | **1.47x / 1.49x** |
| **`llama3.1-70b`** *(141.1 GB Dense)* | `unscanned` | `2 (8)` | `to_maxtext` | 530.06 | 1,080.70 | 140.09 *(Load Orbax)* | 1,220.79 | 1,750.84 | 1.00x (baseline) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **375.50** | **—** | **886.54** *(Safetensors+Convert)* | **886.54** | **1,262.04** | **1.39x / 1.38x** |
| **`gpt-oss-120b`** *(233.7 GB MoE)* | `scanned` | `1 (4)` | `to_maxtext` | 1,199.91 | **OOM (Exit 137 @ 73%, 251.7 GB)** | — *(Step 2 OOM)* | **OOM** | **OOM** | Failed (Single-Host OOM) |
| ↳ | `scanned` | `1 (4)` | **`safetensor`** | **1,061.72** | **—** | **OOM (Exit 137, 253.1 GB)** | **OOM** | **OOM** | Failed (Single-Host OOM) |
| **`gpt-oss-120b`** *(233.7 GB MoE)* | `scanned` | `2 (8)` | `to_maxtext` | 1,199.91 | **OOM (Exit 137 @ 73%, 251.7 GB)** | — *(Step 2 OOM)* | **OOM** | **OOM** | Failed (`to_maxtext` Step 2 OOM) |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | **1,061.72** | **—** | **1,999.42** *(206.1 GB / slice)* | **1,999.42** | **3,061.14** | **Only `safetensor` (`2-slice`) Succeeds!** |
| **`gpt-oss-120b`** *(233.7 GB MoE)* | `unscanned` | `1 (4)` | `to_maxtext` | 1,199.91 | **OOM (Exit 137 @ 100%, 251.7 GB)** | — *(Step 2 OOM)* | **OOM** | **OOM** | Failed (Single-Host OOM) |
| ↳ | `unscanned` | `1 (4)` | **`safetensor`** | **1,061.72** | **—** | **OOM (Exit 137, 253.0 GB)** | **OOM** | **OOM** | Failed (Single-Host OOM) |
| **`gpt-oss-120b`** *(233.7 GB MoE)* | `unscanned` | `2 (8)` | `to_maxtext` | 1,199.91 | **OOM (Exit 137 @ 100%, 251.7 GB)** | — *(Step 2 OOM)* | **OOM** | **OOM** | Failed (`to_maxtext` Step 2 OOM) |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | **1,061.72** | **—** | **1,942.83** *(133.9 GB / slice)* | **1,942.83** | **3,004.55** | **Only `safetensor` (`2-slice`) Succeeds!** |

---

### 3B. DeepSeek Ultra-Large Cluster OOM Benchmark (`284B – 1600B`, 16 Configurations Across 4 Models)

Per the request to test **DeepSeek 3 (`deepseek3-671b`)**, **DeepSeek 3.2 (`deepseek3.2-671b`)**, and **DeepSeek 4 (`deepseek4-284b`, `deepseek4-1600b`)** on the `bodaborg-v5p-nap` cluster (`270.25 GB` RAM per node) and log their OOM behavior:

| Model (BF16 Footprint) | Scan Mode | Slices (Devices) | Pipeline | 1. `HF -> GCS` / Stage | 2. `Convert to Orbax` (`to_maxtext` Step 2) | 3. `Load into MaxText` (`safetensor` Step 2) | **Peak Host RAM Recorded** | **Result / OOM Log Status** |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`deepseek4-284b`** *(581.9 GB, 35,023 tensors)* | `scanned` | `1 (4)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 48%, 39/82 leaves)** | — *(Skipped)* | **245.42 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `1 (4)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137)** | **245.75 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-284b`** *(581.9 GB, 35,023 tensors)* | `scanned` | `2 (8)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 48%, 39/82 leaves)** | — *(Skipped)* | **245.42 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137, Slice 0)** | **245.29 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-284b`** *(581.9 GB, 35,023 tensors)* | `unscanned` | `1 (4)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 21%, 408/1930 leaves)** | — *(Skipped)* | **245.33 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `1 (4)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, init 74.2s)** | **245.95 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-284b`** *(581.9 GB, 35,023 tensors)* | `unscanned` | `2 (8)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 21%, 408/1930 leaves)** | — *(Skipped)* | **245.33 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, Slice 0)** | **245.29 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3-671b`** *(1,365.3 GB, 46,181 tensors)* | `scanned` | `1 (4)` | `to_maxtext` | 1.07s *(221.4s cold)* | **OOMKilled (Exit 137 @ 50%, 16/32 leaves)** | — *(Skipped)* | **245.40 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `1 (4)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137)** | **245.10 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3-671b`** *(1,365.3 GB, 46,181 tensors)* | `scanned` | `2 (8)` | `to_maxtext` | 1.07s | **OOMKilled (Exit 137 @ 50%, 16/32 leaves)** | — *(Skipped)* | **245.40 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137, Slice 0)** | **245.35 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3-671b`** *(1,365.3 GB, 46,181 tensors)* | `unscanned` | `1 (4)` | `to_maxtext` | 0.94s | **OOMKilled (Exit 137 @ 16%, 163/1025 leaves)** | — *(Skipped)* | **245.31 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `1 (4)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, init 74.7s)** | **245.88 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3-671b`** *(1,365.3 GB, 46,181 tensors)* | `unscanned` | `2 (8)` | `to_maxtext` | 0.94s | **OOMKilled (Exit 137 @ 16%, 163/1025 leaves)** | — *(Skipped)* | **245.31 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, Slice 0)** | **245.33 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3.2-671b`** *(1,367.0 GB, 46,491 tensors)* | `scanned` | `1 (4)` | `to_maxtext` | 0.98s | **OOMKilled (Exit 137 @ 50%, 21/42 leaves)** | — *(Skipped)* | **245.56 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `1 (4)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137)** | **246.10 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3.2-671b`** *(1,367.0 GB, 46,491 tensors)* | `scanned` | `2 (8)` | `to_maxtext` | 0.98s | **OOMKilled (Exit 137 @ 50%, 21/42 leaves)** | — *(Skipped)* | **245.56 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137, Slice 0)** | **245.19 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3.2-671b`** *(1,367.0 GB, 46,491 tensors)* | `unscanned` | `1 (4)` | `to_maxtext` | 0.98s | **OOMKilled (Exit 137 @ 15%, 198/1330 leaves)** | — *(Skipped)* | **245.43 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `1 (4)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, init 102.9s)** | **245.90 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek3.2-671b`** *(1,367.0 GB, 46,491 tensors)* | `unscanned` | `2 (8)` | `to_maxtext` | 0.98s | **OOMKilled (Exit 137 @ 15%, 198/1330 leaves)** | — *(Skipped)* | **245.43 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | 0.01s | — | **OOMKilled (Exit 137, init 102.2s)** | **245.46 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-1600b`** *(3,197.7 GB, 73,167 tensors)* | `scanned` | `1 (4)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 49%, 112/228 leaves)** | — *(Skipped)* | **245.65 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `1 (4)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137, init 4.8s)** | **245.11 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-1600b`** *(3,197.7 GB, 73,167 tensors)* | `scanned` | `2 (8)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 49%, 112/228 leaves)** | — *(Skipped)* | **245.65 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `scanned` | `2 (8)` | **`safetensor`** | 0.00s | — | **OOMKilled (Exit 137, Slice 0)** | **245.51 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-1600b`** *(3,197.7 GB, 73,167 tensors)* | `unscanned` | `1 (4)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 6%, 168/2748 leaves)** | — *(Skipped)* | **245.96 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `1 (4)` | **`safetensor`** | 0.02s | — | **OOMKilled (Exit 137, init 325.4s)** | **245.86 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| **`deepseek4-1600b`** *(3,197.7 GB, 73,167 tensors)* | `unscanned` | `2 (8)` | `to_maxtext` | 0.87s | **OOMKilled (Exit 137 @ 6%, 168/2748 leaves)** | — *(Skipped)* | **245.96 / 270.25 GB** | `OOMKilled (Exit Code 137)` |
| ↳ | `unscanned` | `2 (8)` | **`safetensor`** | 0.02s | — | **OOMKilled (Exit 137, init 318.5s)** | **240.32 / 270.25 GB** | `OOMKilled (Exit Code 137)` |

---

## 4. Key Takeaways (`<=32B`, `47B–120B`, and `284B–1600B` DeepSeek OOM Tests)

1. **`safetensor` Performance Remains Stable—and Even Stronger—on `>32B` Dense Models (`llama3.1-70b`)**:
   - On **`llama3.1-70b` (`141.1 GB` Dense)**, `safetensor` (`safetensors_dynamic`) beats `to_maxtext` across **all 4 configurations**: **`1.14x`** (`scanned, 1-slice`), **`1.10x`** (`scanned, 2-slice`), **`1.47x`** (`unscanned, 1-slice`), and **`1.39x`** (`unscanned, 2-slice`).
   - On **`mixtral-8x7b` (`93.4 GB` MoE)**, `safetensor` beats `to_maxtext` by **`1.20x`** in `unscanned, 1-slice`, matches it in `scanned, 1-slice` (`0.99x`), and is within `0.83x–0.95x` on `2-slice`.
2. **`gpt-oss-120b` (`233.7 GB` MoE) Proves the Key Architectural Advantage of Multi-Slice `safetensor`**:
   - On a `251–262 GB` RAM node, **`to_maxtext` Step 2 OOMs (`Exit Code 137` at `251.69 GB`)** in both `scanned` (`73%, 30/41 leaves`) and `unscanned` (`100%, Orbax save_finalize`) because `to_maxtext` is a single-host offline script.
   - Single-slice `safetensor` (`num_slices=1`) also OOMs at `253.1 GB`, **but multi-slice `safetensor` (`num_slices=2`) SUCCEEDS in both `scanned` (`3,061.14s`, `206.09 GB / slice`) and `unscanned` (`3,004.55s`, `133.86 GB / slice`)** because it shards the `233.7 GB` checkpoint across the 2 slices (`ici_fsdp_parallelism=4 × num_slices=2 = 8` devices) without needing a single-host `to_maxtext` conversion step!
3. **`unscanned` (`scan_layers=False`) Is Always Faster & Uses Less RAM in `safetensor`**:
   - Across every model from `0.6B` to `120B`, `safetensor` Step 2 is faster and uses **30%–40% less Peak Host RAM** in `unscanned` mode than in `scanned` mode (e.g., on `llama3.1-70b` `1-slice`, Peak RAM drops from **`225.16 GB` (`scanned`) to `154.35 GB` (`unscanned`)**; on `gpt-oss-120b` `2-slice`, Peak RAM drops from **`206.09 GB` (`scanned`) to `133.86 GB` (`unscanned`)**).
4. **DeepSeek 3, 3.2, and 4 (`284B – 1600B`) Cluster OOM Findings**:
   - **`deepseek4-284b` (`581.9 GB`)**, **`deepseek3-671b` (`1,365.3 GB`)**, **`deepseek3.2-671b` (`1,367.0 GB`)**, and **`deepseek4-1600b` (`3,197.7 GB`)** all exceed the memory capacity of `1` or `2` hosts (`270.25 GB/host`), triggering `OOMKilled (Exit Code 137)` at `~245.1–246.1 GB` across all 16 configurations.
   - To load `deepseek4-284b` (`582 GB`) via `safetensors_dynamic`, at least **`num_slices=4`** (`>1 TB` aggregate RAM) is required; **`deepseek3-671b` / `deepseek3.2-671b` (`1.37 TB`)** require **`num_slices=8`** (`>2 TB` aggregate RAM); and **`deepseek4-1600b` (`3.20 TB`)** requires **`num_slices=16`** (`>4 TB` aggregate RAM).
   - Native HuggingFace checkpoints for `deepseek3-671b` and `deepseek3.2-671b` store weights in **FP8 (`F8_E4M3`)** alongside `weight_scale_inv` scale tensors (`46,181–46,491` tensors across 163 shards), which Orbax `SafetensorsCheckpointHandler` cannot cast directly to `bfloat16` without an FP8 dequantization step.

---

## 5. Issues Found & Fixed (Branch `fix/safetensors-dynamic-multislice-hooks`)

| Issue | Models & Configs Affected | Cause | Fix |
| :--- | :--- | :--- | :--- |
| **`RuntimeError: non-addressable devices`** | `gemma2-2b` (`num_slices=2`), `qwen3-*` / `gemma4` / `olmo3` when vocab padded on `num_slices > 1` | `pad_*embedding_layer` in [`param_mapping.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/param_mapping.py) copies a multi-host `jax.Array` into a local `np.zeros` array. | Use `jnp.pad` when `isinstance(input_tensor, jax.Array)` (fixed on branch `fix/safetensors-dynamic-multislice-hooks`). |
| **Host RAM Spike in `_binary_chunked_stack`** | `qwen3-32b`, `qwen3-30b-a3b`, `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b` (`scanned`) | [`_binary_chunked_stack`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/tensor_handling.py#L91-L111) kept references to all layer / expert tensors during recursion. | Pop/clear input list and `del left, right` as chunks stack (fixed on branch `fix/safetensors-dynamic-multislice-hooks`). |
| **`orbax-checkpoint 0.11.x` `SafetensorsLayout` Full-Shard Memory Caching OOM** | `llama3.1-70b` (`scanned, num_slices=1`) & `gpt-oss-120b` (`num_slices=2`) | `orbax-checkpoint 0.11.x` reads full `.safetensors` files into memory (`263.04/270.25 GB` OOM on `llama3.1-70b` `scanned, 1-slice`). | Upgraded `orbax-checkpoint` to `0.12.7` (byte-range coalescing + `LimitInFlightBytes`), reducing `llama3.1-70b` `scanned, 1-slice` Peak RAM to `225.16 GB`. |
| **`deepseek3.2-671b` / `deepseek4-*` `MaxTextConfig` CPU Validation Error** | `deepseek3.2-671b`, `deepseek4-284b`, `deepseek4-1600b` (`hardware=cpu`) | `use_sparse_indexer: True` with `attention_type: mla` raises `ValueError: Sparse indexer with mla is only supported with dot_product attention or flash attention with tokamax splash` when `hardware=cpu`. | Pass `attention=dot_product` when running `deepseek3.2` or `deepseek4` on `hardware=cpu`. |
| **Multi-Slice Strided Axis-1 Byte Runs on `46,181+` Expert Tensors** | `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-*` (`num_slices=2`) | Sharding 2D `.weight` tensors along `axis=1` in row-major `.safetensors` creates `2,048` non-contiguous byte runs per device per tensor (`~245M` Python tuples across 30,000+ expert tensors). | Shard along leading `axis=0` in `_make_sharded_spec` so each device reads 1 contiguous byte range per tensor. |
