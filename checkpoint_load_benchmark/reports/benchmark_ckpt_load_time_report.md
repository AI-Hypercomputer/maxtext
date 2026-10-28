# MaxText Checkpoint Load Time Benchmark Report: `to_maxtext` vs `safetensor` (`safetensors_dynamic`)

## 1. Executive Summary

This report presents the complete **48-configuration benchmark** comparing MaxText's two HuggingFace checkpoint loading pipelines across **12 models** (`0.6B` to `1600B`) spanning three scale tiers:
- **Tier 0 (`<=32B`, 5 Models / 20 Configurations)**: `qwen3-0.6b` (`1.5 GB` Dense), `gemma2-2b` (`10.5 GB` FP32 Dense), `deepseek2-16b` (`31.4 GB` MLA+MoE), `qwen3-30b-a3b` (`61.1 GB` MoE), and `qwen3-32b` (`65.5 GB` Dense).
- **Tier 1 & 2 (`47B–120B`, 3 Models / 12 Configurations)**: `mixtral-8x7b` (`93.4 GB` MoE), `llama3.1-70b` (`141.1 GB` Dense), and `gpt-oss-120b` (`233.7 GB` MoE OOM & Multi-Slice Boundary Test).
- **Tier 3 (`284B–1600B` DeepSeek Ultra-Large Cluster OOM Benchmark, 4 Models / 16 Configurations)**: `deepseek4-284b` (`581.9 GB` MLA+MoE), `deepseek3-671b` (`1,365.3 GB` BF16 / `685 GB` FP8 MLA+MoE), `deepseek3.2-671b` (`1,367.0 GB` BF16 / `686 GB` FP8 MLA+MoE + Sparse Indexer), and `deepseek4-1600b` (`3,197.7 GB` MLA+MoE + Sparse Indexer).

Each model was evaluated across **both layer scan modes** (`scan_layers=True` / `scanned` and `scan_layers=False` / `unscanned`) and **both slice topologies** (`num_slices=1` single-slice and `num_slices=2` multi-slice):

1. **`to_maxtext` (Offline 3-Step Pipeline)**:
   - **Step 1 (`HF -> GCS`)**: Stream raw HuggingFace `.safetensors` shards to GCS (plus local staging `1b: HF -> Local` via `snapshot_download`).
   - **Step 2 (`Convert to an Orbax checkpoint`)**: Initialize the abstract MaxText model (`[2a]`), load and transform/shard HF weights into MaxText PyTree leaves (`[2b]`), and write the intermediate Zarr3/TensorStore Orbax checkpoint back to GCS (`[2c]`).
   - **Step 3 (`Load the Orbax checkpoint`)**: Restore the pre-converted Orbax checkpoint from GCS into the distributed MaxText `TrainState` / params PyTree.
2. **`safetensor` (`safetensors_dynamic`, Direct 2-Step Pipeline)**:
   - **Step 1 (`HF -> GCS`)**: Stream raw HuggingFace `.safetensors` shards to GCS via [`load_dynamic.prepare_gcs_SafeTensors`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/load_dynamic.py#L81-L157) (or verify the warm GCS cache in `0.46s–3.93s` on subsequent runs).
   - **Step 2 (`Safetensors and convert to MaxText`)**: Derive parameter mappings (`[1/3]`), read `.safetensors` byte ranges directly from GCS into sharded JAX arrays via Orbax's `SafetensorsCheckpointHandler` (`[2/3]`), and transform/shard into the target MaxText PyTree on the fly (`[3/3]`).

> [!IMPORTANT]
> **Key Takeaways Across All 48 Configurations (`12 Models × 2 Scan Modes × 2 Slice Topologies`)**:
> 1. **Performance Stability on `>32B` Models (`mixtral-8x7b`, `llama3.1-70b`)**:
>    - On **`llama3.1-70b` (`141.11 GB` Dense)**, `safetensor` (`safetensors_dynamic`) is **faster than `to_maxtext` across all 4 configurations**: **`1.14x`** (`scanned, num_slices=1`), **`1.10x`** (`scanned, num_slices=2`), **`1.47x`** (`unscanned, num_slices=1`), and **`1.39x`** (`unscanned, num_slices=2`).
>    - On **`mixtral-8x7b` (`93.41 GB` MoE)**, `safetensor` is **`1.20x` faster** in `unscanned, num_slices=1`, matches `to_maxtext` in `scanned, num_slices=1` (`0.99x`), and stays within `0.83x–0.95x` on `num_slices=2`.
> 2. **Multi-Slice `safetensors_dynamic` Solves the Single-Host OOM Bottleneck on `gpt-oss-120b` (`233.66 GB` MoE)**:
>    - Because `to_maxtext` Step 2 converts checkpoints on a single host, **`gpt-oss-120b` OOMs (`Exit Code 137` at `251.69/251.69 GB`)** in both `scanned` (`73%, 30/41 leaves`) and `unscanned` (`100%, Orbax save_finalize`).
>    - Single-slice `safetensor` (`num_slices=1`) also OOMs on a single `261.99 GB` host (`253.06 GB` / `253.01 GB`), **whereas multi-slice `safetensor` (`num_slices=2`) SUCCEEDS in both `scanned` (`3,061.14s`, `Peak RAM: 206.09/261.99 GB` per slice) and `unscanned` (`3,004.55s`, `Peak RAM: 133.86/261.99 GB` per slice)** because `safetensors_dynamic` shards the `233.66 GB` checkpoint across the 2 slices (`8` global devices).
> 3. **`scan_layers=False` (`unscanned`) Accelerates `safetensor` & Cuts Peak Host RAM by 30%–40%**:
>    - Across every model from `0.6B` to `120B`, `safetensor` Step 2 is faster and uses substantially less host RAM in `unscanned` mode than in `scanned` mode (`154.35 GB` vs `225.16 GB` on `llama3.1-70b` `1-slice`; `133.86 GB` vs `206.09 GB` on `gpt-oss-120b` `2-slice`) because it avoids stacking `L` layers (`_binary_chunked_stack` / `jnp.stack`) into massive 3D/4D scanned arrays.
>    - Conversely, in `to_maxtext`, `unscanned` mode is **1.3x–3.2x slower** (`1,080.70s` vs `728.12s` on `llama3.1-70b`) because `to_maxtext` iterates over `310–723` individual PyTree leaves sequentially and creates hundreds of separate Zarr metadata directories on GCS.
> 4. **DeepSeek 3, 3.2, and 4 (`284B – 1600B`) Cluster OOM Tests**:
>    - All 16 configurations across **`deepseek4-284b` (`581.89 GB`)**, **`deepseek3-671b` (`1,365.27 GB`)**, **`deepseek3.2-671b` (`1,367.00 GB`)**, and **`deepseek4-1600b` (`3,197.68 GB`)** exceed the `270.25 GB` per-node RAM on `1` or `2` slices and trigger **`OOMKilled (Exit Code 137)` at `240.32–246.10 GB`**.
>    - Furthermore, native HuggingFace checkpoints for `deepseek3-671b` and `deepseek3.2-671b` store weights in `F8_E4M3` with `weight_scale_inv` scale tensors (`46,181–46,491` tensors across 163 shards), requiring FP8-to-BF16 dequantization support in `safetensors_dynamic`, and `deepseek3.2-671b` / `deepseek4-1600b` require `attention=dot_product` when initialized on CPU (`use_sparse_indexer: True`).

---

## 2. Benchmark Environment & Model Specifications

### 2.1 Execution Environment & Mesh Topologies
- **Script**: [`benchmark_ckpt_load_time.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time.py)
- **CSV Summary**: [`benchmark_ckpt_load_time_summary.csv`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time_summary.csv)
- **GCS Buckets**:
  - `gs://mesa-maxtext/benchmark_ckpt_load_time/run_20261023_032224`
  - `gs://ml-auto-solutions/output/unowned/mesa_benchmark_ckpt_load_time/run_20261023_032224`
- **Compute Nodes**:
  - Small models (`qwen3-0.6b`, `gemma2-2b`): Local Linux workstation (`mesakh.c.googlers.com`)
  - Large & Ultra-Large models (`16B` to `1600B`): `bodaborg-v5p-nap` GKE cluster (`us-central1-a`, project `tpu-prod-env-multipod`, `251.69–270.25 GB RAM`, `112 vCPUs` per node)
- **Slice Topologies**:
  - **Single-Slice (`num_slices=1`)**: `1` process × `1` (or `4`) local devices (`dcn_data_parallelism=1, ici_fsdp_parallelism=1/4`).
  - **Multi-Slice (`num_slices=2`)**: `2` processes on 2 separate cluster nodes (`Slice 0` and `Slice 1` communicating via JAX distributed coordinator) × `4` devices/slice = **`8` global devices** (`dcn_data_parallelism=2, ici_fsdp_parallelism=4` for `<=32B`, and `dcn_fsdp_parallelism=2, ici_fsdp_parallelism=4` (`8-way FSDP`) for `>32B` models so weights are sharded across slices).

### 2.2 Benchmark Model Architecture Matrix (12 Models)

| Model Name | HuggingFace Repo ID | Architecture Type | Layers | Routed / Shared Experts | Checkpoint Size (BF16) | `.safetensors` Shards | Max Single Shard Size | Total HF Tensors | MaxText PyTree Leaves (`scanned` / `unscanned`) | Peak Host RAM (`to_maxtext` Step 2) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-0.6b`** | `Qwen/Qwen3-0.6B` | Dense (GQA + QK-Norm) | 28 | — | 1.50 GB | 1 | 1.50 GB | 310 | 13 / 310 | ~3.2 GB |
| **`gemma2-2b`** | `google/gemma-2-2b` | Dense (GQA, Gated) | 26 | — | 10.45 GB (FP32) / 5.22 GB (BF16) | 3 | 4.95 GB | 288 | 24 / 290 | ~14.8 GB |
| **`deepseek2-16b`** | `deepseek-ai/DeepSeek-V2-Lite` | **MLA + MoE** (1 Dense + 26 MoE) | 27 | 64 routed + 2 shared | 31.41 GB | 4 | **8.58 GB** | 5,335 | 27 / 377 | 53.44 GB |
| **`qwen3-30b-a3b`** | `Qwen/Qwen3-30B-A3B-Thinking-2507` | **MoE** (GQA + 48 MoE layers) | 48 | 128 routed | 61.07 GB | 16 | 4.00 GB | 19,009 | 15 / 579 | 102.77 GB |
| **`qwen3-32b`** | `Qwen/Qwen3-32B` | **Large Dense** (GQA + QK-Norm) | 64 | — | 65.52 GB | 17 | 4.00 GB | 707 | 14 / 707 | 110.30 GB |
| **`mixtral-8x7b`** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | **MoE** (32 MoE layers) | 32 | 8 routed | 93.41 GB | 19 | 5.00 GB | 602 | 14 / 386 | 155.92 GB |
| **`llama3.1-70b`** | `meta-llama/Llama-3.1-70B` | **Large Dense** (GQA, Gated) | 80 | — | 141.11 GB | 30 | 5.00 GB | 723 | 14 / 723 | 235.66 GB |
| **`gpt-oss-120b`** | `unsloth/gpt-oss-120b-BF16` | **Large MoE** (36 MoE layers) | 36 | 128 routed | **233.66 GB** | 73 | 4.83 GB | 795 | 41 / 687 | **OOM (`251.69 GB`, Exit 137)** |
| **`deepseek4-284b`** | `deepseek-ai/DeepSeek-V4-Flash` | **MLA + MoE** (2 Dense + 44 MoE) | 46 | 128 routed + 1 shared | **581.89 GB** | 46 | — | 35,023 | 82 / 1,930 | **OOM (`245.42 GB`, Exit 137)** |
| **`deepseek3-671b`** | `deepseek-ai/DeepSeek-V3` | **MLA + MoE** (3 Dense + 58 MoE) | 61 | 256 routed + 1 shared | **1,365.27 GB** *(685 GB FP8)* | 163 | 5.15 GB | 46,181 | 32 / 1,025 | **OOM (`245.40 GB`, Exit 137)** |
| **`deepseek3.2-671b`** | `deepseek-ai/DeepSeek-V3.2` | **MLA + MoE + Sparse Indexer** | 61 | 256 routed + 1 shared | **1,367.00 GB** *(686 GB FP8)* | 163 | 5.15 GB | 46,491 | 42 / 1,330 | **OOM (`245.56 GB`, Exit 137)** |
| **`deepseek4-1600b`** | `deepseek-ai/DeepSeek-V4-Pro` | **MLA + MoE + Sparse Indexer** | 128 | 128 routed + 1 shared | **3,197.68 GB** | 64 | — | **73,167** | 228 / 2,748 | **OOM (`245.96 GB`, Exit 137)** |

---

## 3. Master 48-Configuration Runtime Comparison

### 3.1 Models `0.6B – 120B` (32 Configurations Across 8 Models)

All values are exact wall-clock seconds recorded in [`benchmark_ckpt_load_time_summary.csv`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time_summary.csv):

| # | Model | Scan Mode | Slices (Devices) | `to_maxtext` 1. HF->GCS (s) | `to_maxtext` 2. Convert Orbax (s) | `to_maxtext` 3. Load Orbax (s) | **`to_maxtext` Total (s)** | `safetensor` 1. HF->GCS (s) *(Warm)* | `safetensor` 2. Load+Convert (s) | **`safetensor` Total (s)** | **Cold E2E Speedup** | **Pre-Cached GCS Speedup (No Step 1)** | **Peak Host RAM (`to_maxtext` Step 3 / `safetensor` Step 2)** |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **`qwen3-0.6b`** | `scanned` | `1 (1)` | 232.48 | 100.41 | 26.27 | **359.17** | 240.03 | 33.95 | **273.98** | **1.31x** | **3.73x** | ~3.1 GB |
| 2 | **`qwen3-0.6b`** | `scanned` | `2 (8)` | 232.48 | 100.41 | 32.70 | **365.60** | 237.83 | 63.09 | **300.91** | **1.21x** | **2.11x** | ~3.1 GB |
| 3 | **`qwen3-0.6b`** | `unscanned` | `1 (1)` | 247.74 | 320.92 | 41.14 | **609.80** | 232.32 | **26.42** | **258.74** | **2.36x** | **13.70x** | ~3.2 GB |
| 4 | **`qwen3-0.6b`** | `unscanned` | `2 (8)` | 247.74 | 320.92 | 46.50 | **615.15** | 230.43 | 46.59 | **277.02** | **2.22x** | **7.89x** | ~3.2 GB |
| 5 | **`gemma2-2b`** | `scanned` | `1 (1)` | 739.82 | 407.06 | 121.15 | **1,268.04** | 726.89 | **88.65** | **815.55** | **1.55x** | **5.96x** | ~14.8 GB |
| 6 | **`gemma2-2b`** | `scanned` | `2 (8)` | 739.82 | 407.06 | 71.80 | **1,218.68** | 726.89 *(3.36)* | 119.55 | **846.44** | **1.44x** | **4.01x** | ~14.8 GB |
| 7 | **`gemma2-2b`** | `unscanned` | `1 (1)` | 739.82 | 302.22 | 116.02 | **1,158.06** | 726.89 *(3.64)* | **85.32** | **812.21** | **1.43x** | **4.90x** | ~14.6 GB |
| 8 | **`gemma2-2b`** | `unscanned` | `2 (8)` | 739.82 | 302.22 | 103.56 | **1,145.60** | 726.89 *(3.12)* | 113.20 | **840.10** | **1.36x** | **3.58x** | ~14.6 GB |
| 9 | **`deepseek2-16b`** | `scanned` | `1 (1)` | 638.24 | 172.79 | 82.13 | **893.17** | 643.64 | 208.70 | **852.34** | **1.05x** | **1.22x** | 53.35 GB |
| 10 | **`deepseek2-16b`** | `scanned` | `2 (8)` | 638.24 | 172.79 | 108.83 | **919.86** | 643.64 *(0.47)* | 339.99 | **983.63** | 0.94x | 0.83x | 53.35 GB |
| 11 | **`deepseek2-16b`** | `unscanned` | `1 (1)` | 638.24 | 325.44 | 89.22 | **1,052.91** | 643.64 *(0.48)* | **173.03** | **816.67** | **1.29x** | **2.40x** | 53.44 GB |
| 12 | **`deepseek2-16b`** | `unscanned` | `2 (8)` | 638.24 | 325.44 | 81.43 | **1,045.11** | 643.64 *(0.49)* | 314.45 | **958.09** | **1.09x** | **1.29x** | 53.44 GB |
| 13 | **`qwen3-30b-a3b`** | `scanned` | `1 (1)` | 330.86 | 335.21 | 111.53 | **777.60** | 336.20 | 398.75 | **734.95** | **1.06x** | **1.12x** | 102.61 GB |
| 14 | **`qwen3-30b-a3b`** | `scanned` | `2 (8)` | 330.86 | 335.21 | 185.50 | **851.57** | 336.20 *(0.48)* | 871.94 | **1,208.13** | 0.70x | 0.60x | 102.61 GB |
| 15 | **`qwen3-30b-a3b`** | `unscanned` | `1 (1)` | 330.86 | 477.58 | 107.51 | **915.95** | 336.20 *(0.51)* | **335.61** | **671.81** | **1.36x** | **1.74x** | 102.77 GB |
| 16 | **`qwen3-30b-a3b`** | `unscanned` | `2 (8)` | 330.86 | 477.58 | 109.37 | **917.81** | 336.20 *(0.46)* | 809.50 | **1,145.70** | 0.80x | 0.73x | 102.77 GB |
| 17 | **`qwen3-32b`** | `scanned` | `1 (1)` | 296.64 | 397.19 | 95.18 | **789.01** | 305.62 | 363.46 | **669.08** | **1.18x** | **1.35x** | 110.14 GB |
| 18 | **`qwen3-32b`** | `scanned` | `2 (8)` | 296.64 | 397.19 | 365.06 | **1,058.89** | 305.62 *(1.08)* | 736.29 | **1,041.91** | **1.02x** | **1.04x** | 110.14 GB |
| 19 | **`qwen3-32b`** | `unscanned` | `1 (1)` | 296.64 | 514.42 | 140.26 | **951.32** | 305.62 *(0.66)* | **304.44** | **610.06** | **1.56x** | **2.15x** | 110.30 GB |
| 20 | **`qwen3-32b`** | `unscanned` | `2 (8)` | 296.64 | 514.42 | 147.96 | **959.02** | 305.62 *(0.59)* | 661.83 | **967.45** | 0.99x | **1.00x** | 110.30 GB |
| 21 | **`mixtral-8x7b`** | `scanned` | `1 (1)` | 532.24 | 329.93 | 130.30 | **992.46** | 399.71 | 599.24 | **998.95** | 0.99x | 0.77x | — / 170.72 GB |
| 22 | **`mixtral-8x7b`** | `scanned` | `2 (8)` | 532.24 | 329.93 | 123.11 | **985.27** | 399.71 *(0.55)* | 786.27 | **1,185.98** | 0.83x | 0.58x | — / 163.56 GB |
| 23 | **`mixtral-8x7b`** | `unscanned` | `1 (1)` | 532.24 | 462.35 | 106.74 | **1,101.33** | 399.71 *(0.91)* | **516.11** | **915.82** | **1.20x** | **1.10x** | — / 112.15 GB |
| 24 | **`mixtral-8x7b`** | `unscanned` | `2 (8)` | 532.24 | 462.35 | 107.09 | **1,101.67** | 399.71 *(0.55)* | 757.98 | **1,157.69** | 0.95x | 0.75x | — / 106.45 GB |
| 25 | **`llama3.1-70b`** | `scanned` | `1 (1)` | 530.06 | 728.12 | 329.15 | **1,587.32** | 375.50 | **1,012.45** | **1,387.95** | **1.14x** | **1.04x** | 219.16 / 225.16 GB |
| 26 | **`llama3.1-70b`** | `scanned` | `2 (8)` | 530.06 | 728.12 | 170.22 | **1,428.39** | 375.50 *(0.50)* | **919.42** | **1,294.91** | **1.10x** | 0.98x | 119.08 / 117.80 GB |
| 27 | **`llama3.1-70b`** | `unscanned` | `1 (1)` | 530.06 | 1,080.70 | 212.89 | **1,823.65** | 375.50 *(0.50)* | **865.53** | **1,241.03** | **1.47x** | **1.49x** | 156.36 / 154.35 GB |
| 28 | **`llama3.1-70b`** | `unscanned` | `2 (8)` | 530.06 | 1,080.70 | 140.09 | **1,750.84** | 375.50 *(3.93)* | **886.54** | **1,262.04** | **1.39x** | **1.38x** | 83.40 / 84.37 GB |
| 29 | **`gpt-oss-120b`** | `scanned` | `1 (4)` | 1,199.91 | **OOM (Exit 137 @ 73%)** | — | **OOM** | 1,061.72 | **OOM (Exit 137)** | **OOM** | Both OOM (1-Slice) | Both OOM (1-Slice) | **251.69 / 253.06 GB (OOM)** |
| 30 | **`gpt-oss-120b`** | `scanned` | `2 (8)` | 1,199.91 | **OOM (Exit 137 @ 73%)** | — | **OOM** | 1,061.72 *(0.00)* | **1,999.42** | **3,061.14** | **`safetensor` Only!** | **`safetensor` Only!** | **OOM / 206.09 GB (OK)** |
| 31 | **`gpt-oss-120b`** | `unscanned` | `1 (4)` | 1,199.91 | **OOM (Exit 137 @ 100%)** | — | **OOM** | 1,061.72 | **OOM (Exit 137)** | **OOM** | Both OOM (1-Slice) | Both OOM (1-Slice) | **251.69 / 253.01 GB (OOM)** |
| 32 | **`gpt-oss-120b`** | `unscanned` | `2 (8)` | 1,199.91 | **OOM (Exit 137 @ 100%)** | — | **OOM** | 1,061.72 *(0.70)* | **1,942.83** | **3,004.55** | **`safetensor` Only!** | **`safetensor` Only!** | **OOM / 133.86 GB (OK)** |

---

### 3.2 DeepSeek Ultra-Large Cluster OOM Benchmark (`284B – 1600B`, Configurations 33–48)

| # | Model (BF16 Footprint) | Scan Mode | Slices (Devices) | `to_maxtext` Step 1 (s) | `to_maxtext` Step 2 (`Convert to Orbax`) | `safetensor` Step 2 (`Safetensors & Convert`) | **`to_maxtext` Peak Host RAM** | **`safetensor` Peak Host RAM** | **Required Slices to Avoid OOM** |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 33 | **`deepseek4-284b`** *(581.89 GB)* | `scanned` | `1 (4)` | 0.87 | **OOMKilled (`Exit 137` @ 48%, 39/82 leaves)** | **OOMKilled (`Exit 137`, init `1.29s`)** | **245.42 / 270.25 GB** | **245.75 / 270.25 GB** | `>= 4 slices` (`>1 TB` RAM) |
| 34 | **`deepseek4-284b`** *(581.89 GB)* | `scanned` | `2 (8)` | 0.87 | **OOMKilled (`Exit 137` @ 48%, 39/82 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `1.34s`)** | **245.42 / 270.25 GB** | **245.29 / 270.25 GB** | `>= 4 slices` (`>1 TB` RAM) |
| 35 | **`deepseek4-284b`** *(581.89 GB)* | `unscanned` | `1 (4)` | 0.87 | **OOMKilled (`Exit 137` @ 21%, 408/1930 leaves)** | **OOMKilled (`Exit 137`, init `74.16s`)** | **245.33 / 270.25 GB** | **245.95 / 270.25 GB** | `>= 4 slices` (`>1 TB` RAM) |
| 36 | **`deepseek4-284b`** *(581.89 GB)* | `unscanned` | `2 (8)` | 0.87 | **OOMKilled (`Exit 137` @ 21%, 408/1930 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `74.02s`)** | **245.33 / 270.25 GB** | **245.29 / 270.25 GB** | `>= 4 slices` (`>1 TB` RAM) |
| 37 | **`deepseek3-671b`** *(1,365.27 GB)* | `scanned` | `1 (4)` | 1.07 *(221.37)* | **OOMKilled (`Exit 137` @ 50%, 16/32 leaves)** | **OOMKilled (`Exit 137`, init `1.88s`)** | **245.40 / 270.25 GB** | **245.10 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 38 | **`deepseek3-671b`** *(1,365.27 GB)* | `scanned` | `2 (8)` | 1.07 *(221.37)* | **OOMKilled (`Exit 137` @ 50%, 16/32 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `1.95s`)** | **245.40 / 270.25 GB** | **245.35 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 39 | **`deepseek3-671b`** *(1,365.27 GB)* | `unscanned` | `1 (4)` | 0.94 | **OOMKilled (`Exit 137` @ 16%, 163/1025 leaves)** | **OOMKilled (`Exit 137`, init `74.73s`)** | **245.31 / 270.25 GB** | **245.88 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 40 | **`deepseek3-671b`** *(1,365.27 GB)* | `unscanned` | `2 (8)` | 0.94 | **OOMKilled (`Exit 137` @ 16%, 163/1025 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `74.66s`)** | **245.31 / 270.25 GB** | **245.33 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 41 | **`deepseek3.2-671b`** *(1,367.00 GB)* | `scanned` | `1 (4)` | 0.98 | **OOMKilled (`Exit 137` @ 50%, 21/42 leaves)** | **OOMKilled (`Exit 137`, init `1.93s`)** | **245.56 / 270.25 GB** | **246.10 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 42 | **`deepseek3.2-671b`** *(1,367.00 GB)* | `scanned` | `2 (8)` | 0.98 | **OOMKilled (`Exit 137` @ 50%, 21/42 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `1.98s`)** | **245.56 / 270.25 GB** | **245.19 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 43 | **`deepseek3.2-671b`** *(1,367.00 GB)* | `unscanned` | `1 (4)` | 0.98 | **OOMKilled (`Exit 137` @ 15%, 198/1330 leaves)** | **OOMKilled (`Exit 137`, init `102.89s`)** | **245.43 / 270.25 GB** | **245.90 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 44 | **`deepseek3.2-671b`** *(1,367.00 GB)* | `unscanned` | `2 (8)` | 0.98 | **OOMKilled (`Exit 137` @ 15%, 198/1330 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `102.20s`)** | **245.43 / 270.25 GB** | **245.46 / 270.25 GB** | `>= 8 slices` (`>2 TB` RAM) |
| 45 | **`deepseek4-1600b`** *(3,197.68 GB)* | `scanned` | `1 (4)` | 0.87 | **OOMKilled (`Exit 137` @ 49%, 112/228 leaves)** | **OOMKilled (`Exit 137`, init `4.83s`)** | **245.65 / 270.25 GB** | **245.11 / 270.25 GB** | `>= 16 slices` (`>4 TB` RAM) |
| 46 | **`deepseek4-1600b`** *(3,197.68 GB)* | `scanned` | `2 (8)` | 0.87 | **OOMKilled (`Exit 137` @ 49%, 112/228 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `4.83s`)** | **245.65 / 270.25 GB** | **245.51 / 270.25 GB** | `>= 16 slices` (`>4 TB` RAM) |
| 47 | **`deepseek4-1600b`** *(3,197.68 GB)* | `unscanned` | `1 (4)` | 0.87 | **OOMKilled (`Exit 137` @ 6%, 168/2748 leaves)** | **OOMKilled (`Exit 137`, init `325.35s`)** | **245.96 / 270.25 GB** | **245.86 / 270.25 GB** | `>= 16 slices` (`>4 TB` RAM) |
| 48 | **`deepseek4-1600b`** *(3,197.68 GB)* | `unscanned` | `2 (8)` | 0.87 | **OOMKilled (`Exit 137` @ 6%, 168/2748 leaves)** | **OOMKilled (`Exit 137`, Slice 0, init `318.48s`)** | **245.96 / 270.25 GB** | **240.32 / 270.25 GB** | `>= 16 slices` (`>4 TB` RAM) |

---

## 4. Granular Sub-Phase Breakdown (`0.6B – 120B`)

### 4.1 `to_maxtext` Step 1 & Step 2 Internal Sub-Phases

| Model | Scan Mode | PyTree Leaves | Step 1: `HF -> GCS` Total (s) | Step 1a: Worker Upload (s) | Step 1b: `HF -> Local` Stage (s) | **Step 2: Convert to Orbax Total (s)** | Step 2a: Init Abstract Model (s) | Step 2b: Transform + Shard (s) | Step 2c: Orbax Save to GCS (s) | Peak Host RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-0.6b`** | `scanned` | 13 | 232.48 | 221.70 | 6.80 | **100.41** | 0.65 | 16.88 | 69.99 | ~3.1 GB |
| **`qwen3-0.6b`** | `unscanned` | 310 | 247.74 | 225.00 | 7.22 | **320.92** | 10.39 | 223.56 | 71.24 | ~3.2 GB |
| **`gemma2-2b`** | `scanned` | 24 | 739.82 | 721.20 | 34.48 | **407.06** | 0.91 | 221.71 | 173.72 | ~14.8 GB |
| **`gemma2-2b`** | `unscanned` | 290 | 739.82 | 721.20 | 34.48 | **302.22** | 4.63 | 134.61 | 151.95 | ~14.6 GB |
| **`deepseek2-16b`** | `scanned` | 27 | 638.24 | 630.20 | 30.35 | **172.79** | 0.39 | 52.08 | 113.63 | 53.35 GB |
| **`deepseek2-16b`** | `unscanned` | 377 | 638.24 | 630.20 | 30.35 | **325.44** | 5.01 | 158.69 | 153.46 | 53.44 GB |
| **`qwen3-30b-a3b`** | `scanned` | 15 | 330.86 | 320.30 | 49.30 | **335.21** | 0.18 | 107.08 | 218.93 | 102.61 GB |
| **`qwen3-30b-a3b`** | `unscanned` | 579 | 330.86 | 320.30 | 49.30 | **477.58** | 5.98 | 279.21 | 182.34 | 102.77 GB |
| **`qwen3-32b`** | `scanned` | 14 | 296.64 | 284.60 | 46.72 | **397.19** | 0.19 | 215.86 | 172.06 | 110.14 GB |
| **`qwen3-32b`** | `unscanned` | 707 | 296.64 | 284.60 | 46.72 | **514.42** | 11.91 | 296.67 | 196.65 | 110.30 GB |
| **`mixtral-8x7b`** | `scanned` | 14 | 532.24 | 521.70 | 201.74 | **329.93** | 0.20 | 110.06 | 210.16 | 155.87 GB |
| **`mixtral-8x7b`** | `unscanned` | 386 | 532.24 | 521.70 | 201.74 | **462.35** | 3.83 | 262.81 | 185.05 | 155.92 GB |
| **`llama3.1-70b`** | `scanned` | 14 | 530.06 | 520.30 | 286.40 | **728.12** | 0.21 | 489.48 | 223.96 | 235.55 GB |
| **`llama3.1-70b`** | `unscanned` | 723 | 530.06 | 520.30 | 286.40 | **1,080.70** | 21.88 | 796.99 | 247.16 | 235.66 GB |
| **`gpt-oss-120b`** | `scanned` | 41 | 1,199.91 | 797.50 | 503.08 | **OOM (Exit 137 @ 73%, 647s)** | — | — | — | **251.69 / 251.69 GB** |
| **`gpt-oss-120b`** | `unscanned` | 687 | 1,199.91 | 797.50 | 503.08 | **OOM (Exit 137 @ 100%, save_finalize)** | — | — | — | **251.69 / 251.69 GB** |

### 4.2 `safetensor` (`safetensors_dynamic`) Step 2 Internal Sub-Phases vs `to_maxtext` Step 3

| Model | Scan Mode | Slices (Devices) | `[1/3]` Derive Mappings (s) | `[2/3]` GCS Safetensors Load (s) | `[3/3]` Transform & Reshard (s) | **`safetensor` Step 2 Total (s)** | **`to_maxtext` Step 3 Load Orbax (s)** | **`to_maxtext` Step 2+3 Total (s)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3-0.6b`** | `scanned` | `1 (1)` | 0.00 | 25.86 | 8.07 | **33.95** | 26.27 | 126.68 |
| **`qwen3-0.6b`** | `scanned` | `2 (8)` | 0.00 | 49.07 | 13.97 | **63.09** | 32.70 | 133.11 |
| **`qwen3-0.6b`** | `unscanned` | `1 (1)` | 0.00 | 23.51 | **2.85** | **26.42** | 41.14 | 362.06 |
| **`qwen3-0.6b`** | `unscanned` | `2 (8)` | 0.00 | 38.69 | 7.69 | **46.59** | 46.50 | 367.42 |
| **`gemma2-2b`** | `scanned` | `1 (1)` | 0.00 | 77.40 | 11.18 | **88.65** | 121.15 | 528.21 |
| **`gemma2-2b`** | `scanned` | `2 (8)` | 0.00 | 83.46 | 35.99 | **119.55** | 71.80 | 478.86 |
| **`gemma2-2b`** | `unscanned` | `1 (1)` | 0.00 | 78.62 | **6.54** | **85.32** | 116.02 | 418.24 |
| **`gemma2-2b`** | `unscanned` | `2 (8)` | 0.00 | 83.50 | 28.86 | **113.20** | 103.56 | 405.78 |
| **`deepseek2-16b`** | `scanned` | `1 (1)` | 0.00 | 162.92 | 41.96 | **208.70** | 82.13 | 254.92 |
| **`deepseek2-16b`** | `scanned` | `2 (8)` | 0.00 | 257.17 | 82.77 | **339.99** | 108.83 | 281.62 |
| **`deepseek2-16b`** | `unscanned` | `1 (1)` | 0.00 | 146.77 | **26.09** | **173.03** | 89.22 | 414.66 |
| **`deepseek2-16b`** | `unscanned` | `2 (8)` | 0.00 | 255.30 | 59.05 | **314.45** | 81.43 | 406.87 |
| **`qwen3-30b-a3b`** | `scanned` | `1 (1)` | 0.01 | 302.59 | 91.93 | **398.75** | 111.53 | 446.74 |
| **`qwen3-30b-a3b`** | `scanned` | `2 (8)` | 0.01 | 689.16 | 182.49 | **871.94** | 185.50 | 520.71 |
| **`qwen3-30b-a3b`** | `unscanned` | `1 (1)` | 0.00 | 285.78 | **49.72** | **335.61** | 107.51 | 585.09 |
| **`qwen3-30b-a3b`** | `unscanned` | `2 (8)` | 0.00 | 674.92 | 134.47 | **809.50** | 109.37 | 586.95 |
| **`qwen3-32b`** | `scanned` | `1 (1)` | 0.00 | 298.11 | 59.13 | **363.46** | 95.18 | 492.37 |
| **`qwen3-32b`** | `scanned` | `2 (8)` | 0.00 | 276.34 | 452.56 | **736.29** | 365.06 | 762.25 |
| **`qwen3-32b`** | `unscanned` | `1 (1)` | 0.00 | 290.59 | **13.83** | **304.44** | 140.26 | 654.68 |
| **`qwen3-32b`** | `unscanned` | `2 (8)` | 0.00 | 266.72 | 383.91 | **661.83** | 147.96 | 662.38 |
| **`mixtral-8x7b`** | `scanned` | `1 (1)` | 0.00 | 449.00 | 134.01 | **599.24** | 130.30 | 460.23 |
| **`mixtral-8x7b`** | `scanned` | `2 (8)` | 0.00 | 550.90 | 235.34 | **786.27** | 123.11 | 453.04 |
| **`mixtral-8x7b`** | `unscanned` | `1 (1)` | 0.00 | 443.24 | **71.81** | **516.11** | 106.74 | 569.09 |
| **`mixtral-8x7b`** | `unscanned` | `2 (8)` | 0.00 | 547.25 | 210.68 | **757.98** | 107.09 | 569.44 |
| **`llama3.1-70b`** | `scanned` | `1 (1)` | 0.00 | 823.51 | 188.94 | **1,012.45** | 329.15 | 1,057.27 |
| **`llama3.1-70b`** | `scanned` | `2 (8)` | 0.00 | 827.65 | **91.72** | **919.42** | 170.22 | 898.34 |
| **`llama3.1-70b`** | `unscanned` | `1 (1)` | 0.00 | 820.64 | **44.75** | **865.53** | 212.89 | 1,293.59 |
| **`llama3.1-70b`** | `unscanned` | `2 (8)` | 0.00 | 809.91 | 76.57 | **886.54** | 140.09 | 1,220.79 |
| **`gpt-oss-120b`** | `scanned` | `1 (4)` | — | — | — | **OOM (Exit 137, 253.06 GB)** | — *(Step 2 OOM)* | **OOM** |
| **`gpt-oss-120b`** | `scanned` | `2 (8)` | 0.00 | 1,836.27 | 163.14 | **1,999.42 (206.09 GB/slice)** | — *(Step 2 OOM)* | **OOM** |
| **`gpt-oss-120b`** | `unscanned` | `1 (4)` | — | — | — | **OOM (Exit 137, 253.01 GB)** | — *(Step 2 OOM)* | **OOM** |
| **`gpt-oss-120b`** | `unscanned` | `2 (8)` | 0.00 | 1,850.54 | **92.26** | **1,942.83 (133.86 GB/slice)** | — *(Step 2 OOM)* | **OOM** |

---

## 5. Deep-Dive Architectural & Scaling Analysis

### 5.1 Finding 1: Performance Stability on Larger Models (`>32B`: `mixtral-8x7b`, `llama3.1-70b`, `gpt-oss-120b`)

Testing models larger than `32B` confirmed that **`safetensor` (`safetensors_dynamic`) remains as stable—and on dense models, even more advantageous—at `47B–120B` scale as on `<=32B` models**:
1. **`llama3.1-70b` (`141.11 GB` Dense)**:
   - Because `to_maxtext` Step 2 must sequentially transform and write `141.11 GB` to GCS (`728.12s` scanned, `1,080.70s` unscanned) and then read it back in Step 3 (`329.15s` / `212.89s`), `to_maxtext` takes **`1,428.39s–1,823.65s`** end-to-end.
   - `safetensor` completes end-to-end in **`1,241.03s–1,387.95s`**, winning **all 4 configurations (`1.10x–1.47x` speedup)**.
   - When using 2-slice FSDP sharding (`dcn_fsdp_parallelism=2, ici_fsdp_parallelism=4`), `llama3.1-70b` `scanned` Step 2 drops from `1,012.45s` (`1-slice`) to **`919.42s` (`2-slice`)**, and Peak Host RAM per slice drops by **nearly 50%** (`225.16 GB -> 117.80 GB` scanned; `154.35 GB -> 84.37 GB` unscanned).
2. **`gpt-oss-120b` (`233.66 GB` MoE) — Breaking the Single-Host Memory Ceiling**:
   - At `233.66 GB` BF16, a single host with `251.69–261.99 GB` RAM cannot hold the full checkpoint plus conversion buffers:
     - `to_maxtext` Step 2 (`scanned`) is killed by the Linux OOM killer (`Exit Code 137`) at **`73% (30/41 leaves, 647s, Peak RAM: 251.69/251.69 GB)`**.
     - `to_maxtext` Step 2 (`unscanned`) transforms all `687/687` leaves (`100%`) using `LazyTensor`, but is killed (`Exit Code 137`) during Orbax `async_save` (`save_finalize`, `Peak RAM: 251.69/251.69 GB`).
     - Single-slice `safetensor` (`num_slices=1`) also hits `OOMKilled` at `253.06/261.99 GB` (`scanned`) and `253.01/261.99 GB` (`unscanned`).
   - However, **2-slice `safetensor` (`num_slices=2`, 8 global devices across 2 nodes) succeeds in both `scanned` (`3,061.14s` total, `1,999.42s` Step 2, `206.09 GB` Peak RAM/slice) and `unscanned` (`3,004.55s` total, `1,942.83s` Step 2, `133.86 GB` Peak RAM/slice)** because each slice only materializes half (`~117 GB`) of the model parameters!

---

### 5.2 Finding 2: DeepSeek 3, 3.2, and 4 (`284B – 1600B`) Cluster OOM & Architecture Analysis

Running `deepseek4-284b` (`581.89 GB`), `deepseek3-671b` (`1,365.27 GB`), `deepseek3.2-671b` (`1,367.00 GB`), and `deepseek4-1600b` (`3,197.68 GB`) on the `bodaborg-v5p-nap` cluster (`270.25 GB` RAM per node) exposed four critical architectural behaviors at ultra-large MoE scale:

1. **Exact OOM Thresholds in `to_maxtext` Step 2 vs. `safetensors_dynamic` Step 2**:
   - In **`to_maxtext` Step 2 (`scanned`)**, conversion proceeds leaf-by-leaf until it reaches the stacked routed-expert MLP tensors (`params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_0`, shape `[58, 256, 2048, 7168]` = **`434.87 GB` for a single leaf** on `deepseek3-671b` / `deepseek3.2-671b`, and `[126, 128, 2048, 8192]` = **`541.17 GB` for a single leaf** on `deepseek4-1600b`). Trying to allocate a single `435–541 GB` leaf on a `270.25 GB` host immediately triggers **`OOMKilled (Exit Code 137)`** at:
     - `48% (39/82 leaves)` on `deepseek4-284b` (`245.42/270.25 GB`)
     - `50% (16/32 leaves)` on `deepseek3-671b` (`245.40/270.25 GB`)
     - `50% (21/42 leaves)` on `deepseek3.2-671b` (`245.56/270.25 GB`)
     - `49% (112/228 leaves)` on `deepseek4-1600b` (`245.65/270.25 GB`)
   - In **`to_maxtext` Step 2 (`unscanned`)**, each MoE layer's 256 (or 128) routed experts are stacked per layer (`~7.5 GB` per `wi_0`/`wi_1`/`wo` leaf per layer), so RAM climbs steadily across layers until hitting **`OOMKilled (Exit Code 137)`** at:
     - `21% (408/1,930 leaves)` on `deepseek4-284b` (`245.33/270.25 GB`)
     - `16% (163/1,025 leaves)` on `deepseek3-671b` (`245.31/270.25 GB`)
     - `15% (198/1,330 leaves)` on `deepseek3.2-671b` (`245.43/270.25 GB`)
     - `6% (168/2,748 leaves)` on `deepseek4-1600b` (`245.96/270.25 GB`)
   - In **`safetensors_dynamic` (`num_slices=1` and `num_slices=2`)**, even with 2 slices (`540.5 GB` combined RAM), the `581.89 GB – 3,197.68 GB` checkpoints exceed per-slice RAM (`291 GB – 1,599 GB` needed per slice on 2 slices) and hit **`OOMKilled (Exit Code 137)`** at `240.32–246.10 GB`.
2. **Unrolled Abstract State Initialization Overhead on Ultra-Deep Models (`deepseek4-1600b`)**:
   - Initializing the abstract MaxText state (`init_initial_state`) in `scanned` mode takes only **`1.29s–4.83s`** across all four DeepSeek models (`32–228` leaves).
   - In `unscanned` mode, unrolling 61–128 MoE + MLA layers increases abstract state initialization to **`74.16s`** (`deepseek4-284b`, `1,930` leaves), **`74.73s`** (`deepseek3-671b`, `1,025` leaves), **`102.89s`** (`deepseek3.2-671b`, `1,330` leaves), and **`325.35s`** (`deepseek4-1600b`, `2,748` leaves — **67.4x slower** than `scanned` mode's `4.83s`).
3. **Native FP8 (`F8_E4M3`) Weights in `deepseek3-671b` and `deepseek3.2-671b`**:
   - Official HuggingFace checkpoints (`deepseek-ai/DeepSeek-V3` and `deepseek-ai/DeepSeek-V3.2`) store linear and MoE expert weights in `F8_E4M3` (`1 byte/param`, `~685 GB` across 163 shards) alongside block-wise `weight_scale_inv` (`F32`) scale tensors (`75,879` total entries in `model.safetensors.index.json`, vs `46,181` weight keys in `HF_SHAPE["deepseek3-671b"]`).
   - Current `safetensors_dynamic` (`load_sharded_hf_state`) passes `bfloat16` target `ShapeDtypeStruct`s directly to Orbax `SafetensorsCheckpointHandler`, which does not multiply `F8_E4M3` weights by `weight_scale_inv`. Supporting native FP8 DeepSeek checkpoints in `safetensors_dynamic` requires either loading `F8_E4M3` + `weight_scale_inv` and dequantizing in `[3/3]` hooks or using a pre-dequantized BF16 checkpoint.

---

## 6. Runtime Failures Encountered, Upstream Bugs Uncovered & Engineering Recommendations

*(For the complete deep-dive root-cause analysis and stack traces of every issue, see [`benchmark_ckpt_load_time_issues.md`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/benchmark_ckpt_load_time_issues.md) / [`checkpoint_load_time_issues.md`](file:///usr/local/google/home/mesakh/.gemini/jetski/brain/c29bd623-4106-4942-aa52-ca670ebb5170/checkpoint_load_time_issues.md).)*

### 6.1 Summary Matrix of Experiment Failures & Fixes Applied

| Issue ID | Affected Model(s) & Config(s) | Pipeline & Step | Exact Error / Symptom | Root Cause | Fix Applied |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **F-1** | **`gemma2-2b`** (`num_slices=2`)<br>*(Also `qwen3-*` when HF vocab $\neq$ MaxText vocab)* | **`safetensor` Step 2** (`[3/3]` Transform) | `RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices is not possible.` | `pad_hf_embedding_layer` / `pad_embedding_layer` in [`param_mapping.py`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/param_mapping.py#L489-L510) copies a multi-host `jax.Array` into a local `np.zeros` array. | Replaced NumPy array slicing with pure JAX `jnp.pad` on branch `fix/safetensors-dynamic-multislice-hooks`. |
| **F-2 & F-3** | **`qwen3-32b`**, **`mixtral-8x7b`**, **`llama3.1-70b`** (`scanned`) | **`to_maxtext` Step 3** (Load Orbax) | `OOMKilled` (`concurrent_gb=96`) vs `ValueError: Requested more bytes than we reserved space for` (`concurrent_gb=8` or `20`) | Scanned MLP leaves are `16.78 GB` (`qwen3-32b`), `12.08 GB` (`mixtral-8x7b`), and `37.58 GB` (`llama3.1-70b` `[8192, 80, 28672]`). Orbax `LimitInFlightBytes` requires `concurrent_gb` $\ge$ largest single unsharded Zarr leaf on `1-slice` (`38 GB` for `llama3.1-70b`). | Sized `checkpoint_storage_concurrent_gb` per model and slice count (`38` for `llama3.1-70b` `1-slice`, `20` for `2-slice`, `16` for `unscanned`). |
| **F-4** | **`qwen3-32b`**, **`qwen3-30b-a3b`**, **`mixtral-8x7b`**, **`llama3.1-70b`**, **`gpt-oss-120b`** (`scanned`) | **`safetensor` Step 2** (`[3/3]` `_binary_chunked_stack`) | High host RAM spike during recursive layer/expert stacking | [`_binary_chunked_stack`](file:///usr/local/google/home/mesakh/Documents/mesakh/maxtext-project/maxtext-3/src/maxtext/checkpoint_conversion/utils/tensor_handling.py#L91-L111) retained references to intermediate chunk lists while recursively stacking 32–80 layers or 128 experts. | Pop and free intermediate tensors immediately as each chunk is stacked on branch `fix/safetensors-dynamic-multislice-hooks`. |
| **F-5** | **`llama3.1-70b`** (`scanned, num_slices=1`) & **`gpt-oss-120b`** (`num_slices=2`) | **`safetensor` Step 2** (`[2/3]` GCS Load) | **`OOMKilled (Exit Code 137)`** at `263.04/270.25 GB` on `llama3.1-70b` with `orbax-checkpoint 0.11.x` | Base image's `orbax-checkpoint 0.11.x` `SafetensorsLayout` reads entire `.safetensors` files into memory without byte-range coalescing or in-flight byte limiting. | Upgraded `orbax-checkpoint` to `0.12.7`, reducing `llama3.1-70b` `scanned, 1-slice` Peak RAM from `263.04 GB` (OOM) to `225.16 GB` (Succeeded in `1,012.45s`). |
| **F-6** | **`gpt-oss-120b`** (`233.66 GB` MoE) | **`to_maxtext` Step 2** & **`safetensor` (`num_slices=1`)** | **`OOMKilled (Exit Code 137)`** at `251.69–253.06 GB` | `233.66 GB` BF16 checkpoint exceeds single-host RAM (`251.69–261.99 GB`) when converting or loading on 1 slice. | Sharded across 2 slices (`num_slices=2`, `ici_fsdp_parallelism=4, dcn_fsdp_parallelism=2`) in `safetensors_dynamic`, succeeding at `206.09 GB` (`scanned`) and `133.86 GB` (`unscanned`) per slice. |
| **F-7** | **`deepseek3.2-671b`**, **`deepseek4-284b`**, **`deepseek4-1600b`** (`hardware=cpu`) | **`to_maxtext` Step 2** & **`safetensor` Step 2** | `ValueError: Sparse indexer with mla is only supported with dot_product attention or flash attention with tokamax splash.` | `deepseek3.2-671b.yml` and `deepseek4-*.yml` set `use_sparse_indexer: True` with `attention_type: mla`, which fails validation when `hardware=cpu` leaves `attention="autoselected"`. | Pass `attention=dot_product` when running `deepseek3.2` or `deepseek4` with `hardware=cpu`. |
| **F-8** | **`deepseek3-671b`**, **`deepseek3.2-671b`**, **`deepseek4-284b`**, **`deepseek4-1600b`** (`582 GB – 3,198 GB`) | **All `to_maxtext` Step 2 & `safetensor` (`num_slices=1, 2`)** | **`OOMKilled (Exit Code 137)`** at `240.32–246.10 / 270.25 GB` | Model BF16 footprints (`581.89 GB`, `1,365.27 GB`, `1,367.00 GB`, `3,197.68 GB`) exceed 1-host (`270 GB`) and 2-host (`540 GB`) memory capacity. | Logged exact OOM points across all 16 configurations; requires `num_slices >= 4` (`284B`), `>= 8` (`671B`), and `>= 16` (`1600B`) with `safetensors_dynamic`. |
