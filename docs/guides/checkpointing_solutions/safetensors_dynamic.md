<!--
 Copyright 2023-2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

(safetensors-dynamic)=

# Dynamic SafeTensors Loading (`safetensors_dynamic`)

`source_checkpoint_layout="safetensors_dynamic"` loads Hugging Face (HF)
`.safetensors` checkpoints **directly into sharded MaxText weights in host and
accelerator memory at startup**, without running offline `to_maxtext`
conversion or writing an intermediate Orbax checkpoint to GCS.

It is supported across both **training** (`train.py` ->
`load_state_if_possible`) and **inference / evaluation** (`from_pretrained`:
MaxEngine, vLLM, RL, and `forward_pass_logit_checker`), scaling from single-host
CPU/TPU runs up to **128+ hosts** and **1.6T+ parameter** checkpoints
(`3.2 TB`).

## Why Use `safetensors_dynamic`?

| Dimension                             | Offline `to_maxtext` + Orbax                                                        | `safetensors_dynamic` (On-the-Fly)                                                           |
| :------------------------------------ | :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **Workflow**                          | Download HF -> Run single-host CPU `to_maxtext` -> Write Orbax to GCS -> Load Orbax | Point `load_parameters_path` at `gs://`, local dir, or `hf://` -> Load directly into MaxText |
| **GCS Storage**                       | Requires 2x storage (HF SafeTensors + converted Orbax copy)                         | Requires **0x extra checkpoint storage** (`hf://` optionally caches raw SafeTensors once)    |
| **Single-Host RAM Limit**             | Fails with Host RAM OOM on scanned `>=470 GB` and unscanned `>=580 GB` models       | Shards reads across all hosts along axis 0; verified up to **3.2 TB (`deepseek4-1600b`)**    |
| **Startup Latency (Cold/Single Run)** | CPU conversion accounts for **76-95%** of unscanned time                            | **1.8x median speedup** (`<=120B`), **2.8-6.3x** (`>200B`) end-to-end on TPU v7x             |
| **Repeated Runs on Small/Mid Models** | Pre-converted Orbax restore alone is faster for repeated `<=120B` restarts          | Best for large models (`>200B`), cold HF experiments, evals, and storage-free workflows      |

## Quick Start

### 1. Load from a GCS SafeTensors Directory

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_dynamic_run \
    model_name=llama3.1-70b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Llama-3.1-70B/" \
    base_output_directory="gs://my-bucket/output/"
```

### 2. Load Directly from Hugging Face Hub (`hf://` or Auto-Lookup)

When loading from the Hugging Face Hub, **Host 0** lists (`HfFileSystem.glob`)
and downloads missing `.safetensors` files in parallel (`32` spawned workers,
`32 MB` chunks) into `<base_output_directory>/hf_cache/<repo>/`, then
synchronizes all hosts via
`jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")`.

```bash
# Explicit hf:// repo (must match model_name architecture)
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_hf_run \
    model_name=llama3.1-8b-Instruct \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="hf://meta-llama/Llama-3.1-8B-Instruct" \
    base_output_directory="gs://my-bucket/output/" \
    hf_access_token="${HF_TOKEN}"
```

If `load_parameters_path` is omitted (`""`), MaxText looks up the **exact**
`model_name` (preserving `-Instruct` suffixes) in `HF_IDS`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_auto_hf_run \
    model_name=qwen3-30b-a3b \
    source_checkpoint_layout="safetensors_dynamic" \
    base_output_directory="gs://my-bucket/output/"
```

### 3. Inference & Logit Verification (`from_pretrained`)

```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    model_name=gemma3-4b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/gemma-3-4b-it/" \
    weight_dtype=bfloat16 \
    enable_checkpointing=false \
    --run_hf_model=True \
    --hf_model_path="google/gemma-3-4b-it" \
    --max_kl_div=0.015
```

## Configuration Reference & Constraints

| Flag                                   | Requirement / Behavior                                                                                                                                           |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source_checkpoint_layout`             | Set to `"safetensors_dynamic"`.                                                                                                                                  |
| `load_parameters_path`                 | `gs://` directory, local directory, `hf://<org>/<repo>`, or `<org>/<repo>`. If empty, uses `HF_IDS[model_name]` (exact key match).                               |
| `base_output_directory`                | **Must start with `gs://`** whenever downloading from HF Hub (`hf://` or auto-lookup), as weights cache under `<base_output_directory>/hf_cache/`.               |
| `hf_access_token`                      | Required for gated Hugging Face repositories (e.g., Llama, Gemma).                                                                                               |
| `weight_dtype`                         | Loaded tensors are cast to each target leaf's `weight_dtype` (default `bfloat16`; supports `float32`).                                                           |
| `scan_layers`                          | Works in both `scan_layers=True` (scanned) and `scan_layers=False` (unscanned) modes, plus nested `-local_layers` scans.                                         |
| `enable_checkpointing`                 | Works with `enable_checkpointing=False` (`jax.distributed` is initialized automatically for multi-host runs).                                                    |
| `enable_single_controller`             | **Not supported.** Config validation rejects `source_checkpoint_layout="safetensors_dynamic"` with `enable_single_controller=True` (Pathways single-controller). |
| `enable_single_replica_ckpt_restoring` | Applies only to Orbax checkpoint restores; unused by `safetensors_dynamic`.                                                                                      |

## Architecture Overview

Implemented in `src/maxtext/checkpoint_conversion/utils/load_dynamic.py` and
`tensor_handling.py`, execution proceeds through four stages:

1. **Phase 0 - Host-0 HF Hub Cache (Optional):** Only `jax.process_index() == 0`
   queries the HF API (`HfFileSystem.glob`) and uploads missing `.safetensors`
   shards into `<base_output_directory>/hf_cache/` using `32` spawned workers
   (`32 MB` chunks), preventing cluster-wide HTTP `429` throttling on `128+`
   host jobs. All hosts wait on
   `sync_global_devices("dynamic_hf_download_complete")`.
2. **Phase 1 - Exact Mappings & Weights-Only Target Tree:**
   `get_hf_config_and_mappings` resolves `HF_MODEL_CONFIGS`, `PARAM_MAPPING`,
   and `HOOK_FNS` with exact `model_name` lookup. Only weights
   (`_abstract_params` / NNX `State`) are passed, avoiding optimizer-state key
   flattening issues.
3. **Phase 2 - Contiguous Leading-Axis Sharded Reads:** `load_sharded_hf_state`
   opens Orbax `Context(SAFETENSORS, SafetensorsOptions(max_over_read_ratio=1.0))`
   inside `_no_gcs_read_ahead()` (`gcsfs.GCSFileSystem.default_block_size = 0`).
   `_leading_axis_shardings` shards row-major HF tensors along axis `0` only on
   a 2D `("process", "local")` mesh so each host performs one contiguous
   byte-range read without strided over-reads or cross-host collectives during
   I/O. Legacy multimodal prefixes (e.g., Gemma-3) are normalized by
   `rename_legacy_hf_keys`.
4. **Phase 3 - Refcounted Transforms, Binary Stacking & Sliced Reshard:**
   `transform_hf_state_to_mt_state` tracks `remaining_uses` to pop HF tensors
   immediately after their last use (handling composite HF/MaxText keys, `None`
   keys, and `MoEBiasVar`/`Tid2EidVar` collections), stacks layers via
   recursive `_binary_chunked_stack` using shared `single_stack_axis` /
   `stacked_axes` rules, runs multi-host NumPy hooks via `1 GiB` sliced
   gather/scatter when needed, reshards arrays `>1 GiB`
   (`_MAX_RESHARD_BYTES = 1 << 30`) in slices to bound peak HBM, and validates
   completeness with `_raise_on_weight_mismatch`.

### Active Optimizations Under Review

- **Force `gcsfs` on `gs://`
  ([PR #6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319)):** Passes
  `FileOptions(path_class=_GcsfsPath)` to bypass `tf.io.gfile` `64 MiB`-per-file
  read buffers when TensorFlow is installed, cutting `qwen3-30b-a3b` per-host
  read RAM on `v5e-256` from `>170 GB` (OOM) to `8.3 GB` (`50 GB` full-load
  RSS), and capping `qwen3-480b-a35b` peak host RAM at `63 GB/host`.
- **Layer-by-Layer HF Streaming V2
  ([PR #6223](https://github.com/AI-Hypercomputer/maxtext/pull/6223)):** Builds
  on merged `SafetensorsReader` (`32 MiB` ranged reads across `64` threads,
  [PR #5592](https://github.com/AI-Hypercomputer/maxtext/pull/5592)) and
  multi-host `jnp` hooks
  ([PR #5593](https://github.com/AI-Hypercomputer/maxtext/pull/5593)) to stream
  a few decoder layers at a time via `dynamic_update_slice` into preallocated
  weight trees, targeting near-zero transient HBM overhead.

## Correctness & Parity Results

- **Two-Slice Bit-Exact Parity (`<=120B` Suite):** Across all **47** `<=120B`
  models registered in `PARAM_MAPPING`, 44 had valid `to_maxtext` baselines
  (excluding `weaver-mini`, `weaver-max`, where `to_maxtext` fails, and
  duplicate alias `qwen3.5-35b-fp8`):
  - **42 / 44 models** matched `to_maxtext` logits **bit-exact**
    (`forward_pass_logit_checker --atol=0 --rtol=0`).
  - **2 / 44 models** (`qwen3.5-35b-a3b-fp8`, `gemma2-27b`) passed vs HF
    (`max KL <= 0.017`) and differed from `to_maxtext` only by
    `float32`/`bfloat16` cast rounding.
- **Training Loss Parity:** **10-step `float32` training on C4** (`llama3.1-8b`
  on `TPU v6e-32`, 8 hosts) produced identical `learning/loss` at every step
  compared with training from an offline `to_maxtext` Orbax checkpoint.
- **Large Model (`>200B`) Validation:** Forward-pass token probes (`"Paris is the capital of"` -> `" France"`, `"Monday, Tuesday, Wednesday,"` -> `" Thursday"`) and 2-step C4 training validated across `mixtral-8x22b`,
  `qwen3-235b-a22b`, `qwen3.5-397b-a17b`, `llama3.1-405b`, `qwen3-480b-a35b`,
  `deepseek4-284b`, `deepseek3-671b`, `deepseek3.2-671b`, and
  `deepseek4-1600b`.

## Benchmarks by Accelerator Type

### Accelerator Coverage Summary

| Accelerator Type        | Hardware Profile Tested                                                                                      | Models & Scope Evaluated                                                                                 | Key Findings                                                                                                                                                                                                                                                                      |
| :---------------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TPU `v7x` (`tpu7x`)** | 1-8 hosts (`4` chips / `8` cores, `813.8 GB` HBM, `1,014.5 GB` host RAM per host); `tpu7x-128` (16-32 hosts) | **21 models (`0.6B-1.6T`)**, scanned & unscanned (`58-60` configs `<=120B`, `7` models `>200B`)          | Faster in **56/58** plain `<=120B` runs (median **1.8x** E2E, max **4.21x** E2E / **14.68x** excl. download). Loaded **7/7** models `>200B` (vs **2/7** for `to_maxtext`), **2.8-6.3x** faster E2E (**4.2-10.8x** excl. download).                                                |
| **TPU `v5p`**           | `1-8` hosts (`95 GB` HBM/chip, `~252-270 GB` host RAM); `v5p-8`, `1x`/`2x v5p-32`, `v5p-64`                  | **6 models (`0.6B-120B`)** `to_maxtext` comparison + Airflow E2E (`70B`) + Streaming V2 (`284B`, `397B`) | Faster E2E in **16/20** comparison runs (median **1.25x**, max **2.36x** E2E / **13.70x** excl. download). Only path that finishes `gpt-oss-120b` (2 slices) without single-host OOM. Streaming V2 loads **397B (`792.7 GB`) in 115 s (`6.89 GB/s`)** on `v5p-64`.                |
| **TPU `v5e`**           | `v5e-32` (16 hosts), `v5e-256` (64 hosts), `2x v5e-256` (128 hosts), `16 GB` HBM/chip, `~202 GB` RAM         | **9 large models (`30B-671B`)** + 2-slice `PARAM_MAPPING` parity suite                                   | Contiguous reads cut `mixtral-8x22b` read from `726 s` (`60.3x` over-read) to **14 s** (`64` hosts) and load `deepseek3-671b` (`1.3 TB`) in **193 s read + 120 s transform** (`128` hosts). `>=284B` models load into HBM but exceed `16 GB`/chip HBM during forward/train steps. |
| **TPU `v6e`**           | `v6e-32` (8 hosts, `32 GB` HBM/chip)                                                                         | Tier-1 logit parity (`C8-C16`), `from_pretrained`, `float32` training (`T1-T7`)                          | **Bit-exact (`--atol=0 --rtol=0`)** parity with `to_maxtext` and identical 10-step `float32` loss on `llama3.1-8b`.                                                                                                                                                               |
| **TPU `v4`**            | `v4-8` (1 host)                                                                                              | Streaming V2 (`SafetensorsReader`, PR #6223) baseline                                                    | `SafetensorsReader` ranged reads reach **~2.8 GB/s** vs `~1.1 GB/s` Orbax `0.12.x`; step-0 loss matches Orbax on `Qwen3-8B` and `DeepSeek-V2-Lite`.                                                                                                                               |
| **CPU**                 | Single-host CPU VM                                                                                           | Small/multimodal parity & logit checker (`T8-T10`, `C1-C7`: `0.6B-9B`)                                   | Bit-exact vs `to_maxtext` and `KL <= 8.6e-5` vs HF on `qwen3-0.6b`, `gemma2-2b`, `gemma3-4b`, `gemma4-e2b`, `qwen3.5-9b`, `qwen3-vl-2b`.                                                                                                                                          |
| **GPU (`H100`/`B200`)** | Not yet benchmarked                                                                                          | Out of scope in current TPU validation suite                                                             | Tracked for future validation; code uses standard JAX `Mesh`/`NamedSharding` abstractions.                                                                                                                                                                                        |

### 1. TPU `v7x` (`tpu7x`) Benchmarks

Host hardware: `4` chips (`8` TPU cores), `813.8 GB` HBM, `1,014.5 GB` host RAM
per host. All times below include Step 1 (`HF -> GCS` cold download) under
**End-to-End**, while **No Step 1** isolates `to_maxtext` (CPU convert + Orbax
load) vs `safetensors_dynamic` (`load_sharded_hf_state + transform_hf_state_to_mt_state`).

#### A. Representative Models `<=120B` (TPU `v7x`, 1-2 Hosts / 8-16 Cores)

Across 14 models (`qwen3-0.6b` through `gpt-oss-120b`, 60 paired runs total):

- **Speedup (End-to-End):** Median **1.77x** (range **0.65x-4.21x**; faster in
  **56/58** plain runs and **58/60** including the `gpt-oss-120b` HBM-peak fix
  branch).
- **Speedup (Excl. HF->GCS Download):** Median **3.05x** (range
  **0.59x-14.68x**).
- **Exceptions:** Slower in 2 plain runs (`qwen3-14b` scanned `1(8)` at `0.65x`
  E2E due to `289.65 s` transform; `qwen3-30b-a3b` unscanned `2(16)` at `0.78x`
  E2E); `gpt-oss-120b` scanned needed the HBM-peak fix branch (`2.07x-2.34x`
  E2E).

| Model           |   Size   | Config (Scan, Slices (Cores))      | HF->GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load+Transform (s) | E2E Speedup | No-Step-1 Speedup |
| :-------------- | :------: | :--------------------------------- | :---------: | :-----------------------------: | :--------------------------------------: | :---------: | :---------------: |
| `llama3.1-8b`   | 16.1 GB  | Scanned, `2 (16)`                  |    62.91    |          89.58 + 17.92          |       **28.52** (`10.89 + 17.61`)        |  **1.86x**  |     **3.77x**     |
| `qwen3-32b`     | 65.5 GB  | Scanned, `2 (16)`                  |   107.24    |         381.46 + 238.09         |       **88.92** (`35.44 + 53.42`)        |  **3.71x**  |     **6.97x**     |
| `qwen3-30b-a3b` | 61.1 GB  | Scanned, `1 (8)`                   |    78.82    |         151.57 + 386.06         |       **129.52** (`99.12 + 30.38`)       |  **2.96x**  |     **4.15x**     |
| `mixtral-8x7b`  | 93.4 GB  | Scanned, `2 (16)`                  |   239.75    |         134.07 + 792.49         |       **63.12** (`45.96 + 17.16`)        |  **3.85x**  |    **14.68x**     |
| `llama3.1-70b`  | 141.1 GB | Scanned, `2 (16)`                  |   197.80    |         780.27 + 295.79         |       **104.98** (`72.44 + 32.48`)       |  **4.21x**  |    **10.25x**     |
| `gpt-oss-120b`  | 233.7 GB | Scanned, `2 (16)` *(with HBM fix)* |   350.33    |         581.39 + 893.97         |      **429.98** (`127.52 + 302.46`)      |  **2.34x**  |     **3.43x**     |
| `gpt-oss-120b`  | 233.7 GB | Unscanned, `2 (16)`                |   350.33    |         617.84 + 115.05         |      **390.69** (`122.59 + 268.06`)      |  **1.46x**  |     **1.88x**     |

#### B. Large Models `>200B` (TPU `v7x`, 1-8 Hosts / 8-64 Cores)

Offline `to_maxtext` succeeded on only **2 of 7** models (`qwen3-235b-a22b`
unscanned and `qwen3.5-397b-a17b` unscanned), failing with single-host RAM OOM
on all scanned `>200B` runs and all runs `>=480B`. `safetensors_dynamic` loaded
**7 of 7** models.

| Model                | Ckpt Size | Config (Scan, Slices (Cores))      | `to_maxtext` E2E (Convert+Load)     | `safetensors_dynamic` Load+Transform (E2E) | Peak Host RAM |    Speedup (E2E / No-Step-1)     |
| :------------------- | :-------: | :--------------------------------- | :---------------------------------- | :----------------------------------------: | :-----------: | :------------------------------: |
| `qwen3-235b-a22b`    | 470.4 GB  | Scanned, `1 (8)`                   | Failed (Host RAM OOM)               |      **757.50 s** (`1,340.81 s` E2E)       |   391.8 GB    | Only `safetensors_dynamic` loads |
| `qwen3-235b-a22b`    | 470.4 GB  | Unscanned, `1 (8)`                 | 3,754.33 s (`2,625.50 + 545.51`)    |      **697.13 s** (`1,280.44 s` E2E)       |   392.7 GB    |        **2.93x / 4.55x**         |
| `deepseek4-284b`\*   | 581.9 GB  | Unscanned, `2 (32)`                | Failed (OOM-killed @ `989.6 GB`)    |      **258.40 s** (`200.93 + 57.23`)       |   318.3 GB    | Only `safetensors_dynamic` loads |
| `qwen3.5-397b-a17b`  | 795.5 GB  | Scanned, `2 (16)` *(with HBM fix)* | Failed (Host RAM OOM)               |     **1,608.32 s** (`2,422.85 s` E2E)      |    93.0 GB    | Only `safetensors_dynamic` loads |
| `qwen3.5-397b-a17b`  | 795.5 GB  | Unscanned, `1 (32)`                | 11,144.07 s (`8,096.53 + 2,233.01`) |      **954.34 s** (`1,768.87 s` E2E)       |    88.1 GB    |        **6.30x / 10.82x**        |
| `qwen3-480b-a35b`    | 961.0 GB  | Scanned, `2 (32)`                  | Failed (Host RAM OOM)               |      **757.70 s** (`1,803.84 s` E2E)       |   244.9 GB    | Only `safetensors_dynamic` loads |
| `qwen3-480b-a35b`    | 961.0 GB  | Unscanned, `2 (32)`                | Failed (Host RAM OOM)               |      **718.65 s** (`1,764.79 s` E2E)       |   244.6 GB    | Only `safetensors_dynamic` loads |
| `deepseek3-671b`\*   |  1.3 TB   | Scanned, `4 (32)`                  | Not run (`>1,014.5 GB` host RAM)    |      **649.35 s** (`468.92 + 176.84`)      |   689.9 GB    | Only `safetensors_dynamic` loads |
| `deepseek3.2-671b`\* |  1.37 TB  | Unscanned, `2 (32)`                | Not run (`>1,014.5 GB` host RAM)    |      **569.71 s** (`459.39 + 109.96`)      |   690.9 GB    | Only `safetensors_dynamic` loads |
| `deepseek4-1600b`\*  |  3.2 TB   | Unscanned, `1 (64)` (8 hosts)      | Not run (`>1,014.5 GB` host RAM)    |      **874.31 s** (`709.10 + 164.86`)      |   803.1 GB    | Only `safetensors_dynamic` loads |
| `deepseek4-1600b`\*  |  3.2 TB   | Scanned, `1 (64)` (8 hosts)        | Not run (`>1,014.5 GB` host RAM)    |      **971.49 s** (`698.05 + 273.22`)      |   615.5 GB    | Only `safetensors_dynamic` loads |

*\* `deepseek*` benchmark rows on `v7x` read locally staged synthetic sparse
SafeTensors files on each host (download time `0.00 s`) to validate multi-host
I/O, transform, and HBM/RAM scaling up to 3.2 TB. End-to-end GCS load + 2-step
training on real BF16 weights was additionally verified on `tpu7x-128` (`16-32`
hosts): `llama3.1-405b` (`36 s` read / `25 s` transform, step-0 loss `1.744`),
`qwen3.5-397b-a17b` (`51 s` / `587 s`, loss `3.602`), `qwen3-480b-a35b` (`191 s`
/ `68 s`, loss `2.443`), `deepseek4-284b` (`296 s` / `150 s`, loss `2.661`), and
`deepseek3-671b` (`252 s` / `48 s`, loss `2.153`).*

### 2. TPU `v5p` Benchmarks

TPU `v5p` hosts provide `95 GB` HBM per chip and `~252-270 GB` host RAM per
host.

#### A. `safetensors_dynamic` vs `to_maxtext` (6 Models, `1x`-`2x` Slices)

Across 20 configurations on `v5p` (`qwen3-0.6b`, `gemma2-2b`, `deepseek2-16b`,
`qwen3-30b-a3b`, `qwen3-32b`, `gpt-oss-120b`), `safetensors_dynamic` was faster
E2E in **16/20** runs (median **1.25x** E2E, max **2.36x** E2E / **13.70x**
excl. download):

| Model                       | Config (Scan, Slices (Devices)) |   `to_maxtext` E2E (Convert+Load)    |     `safetensors_dynamic` Load+Transform (E2E)      |            Speedup (E2E / No-Step-1)            |
| :-------------------------- | :------------------------------ | :----------------------------------: | :-------------------------------------------------: | :---------------------------------------------: |
| `qwen3-0.6b`                | Unscanned, `1 (1)`              |     609.80 s (`320.92 + 41.14`)      |            **26.42 s** (`258.74 s` E2E)             |               **2.36x / 13.70x**                |
| `gemma2-2b`                 | Scanned, `1 (1)`                |    1,268.04 s (`407.06 + 121.15`)    |            **88.65 s** (`815.55 s` E2E)             |                **1.55x / 5.96x**                |
| `qwen3-32b`                 | Unscanned, `1 (1)`              |     951.32 s (`514.42 + 140.26`)     |            **304.44 s** (`610.06 s` E2E)            |                **1.56x / 2.15x**                |
| `qwen3-30b-a3b`             | Unscanned, `1 (1)`              |     915.95 s (`477.58 + 107.51`)     |            **335.61 s** (`671.81 s` E2E)            |                **1.36x / 1.74x**                |
| `qwen3-30b-a3b`             | Scanned, `2 (8)`                |     851.57 s (`335.21 + 185.50`)     |             871.94 s (`1,208.13 s` E2E)             |    `0.70x / 0.60x` (cross-slice DCN reshard)    |
| `gpt-oss-120b` (`233.7 GB`) | Scanned, `1 (4)`                |        OOM (`251.7 GB` @ 73%)        |          OOM (`253.1 GB` single-host RAM)           |       Fails on single host (`252 GB` RAM)       |
| `gpt-oss-120b` (`233.7 GB`) | Scanned, `2 (8)`                | OOM (`251.7 GB` single-host convert) | **1,999.42 s** (`3,061.14 s` E2E, `206.1 GB`/slice) | **Only 2-slice `safetensors_dynamic` succeeds** |
| `gpt-oss-120b` (`233.7 GB`) | Unscanned, `2 (8)`              | OOM (`251.7 GB` single-host convert) | **1,942.83 s** (`3,004.55 s` E2E, `133.9 GB`/slice) | **Only 2-slice `safetensors_dynamic` succeeds** |

*Note on `>=284B` DeepSeek on 1-2 host `v5p` slices:* With `270.25 GB` RAM per
`v5p` host, `1 (4)` and `2 (8)` topologies hit host-RAM OOM (`~245 GB / 270 GB`,
Exit `137`) on `284B-1.6T` models in both `to_maxtext` and current Orbax-backed
`safetensors_dynamic`. Multi-host topologies (`4+` hosts) or PR #6319 / PR #6223
are required to stay within `270 GB` host RAM.

#### B. Airflow E2E DAG (`maxtext_e2e_tpu_safetensors_dynamic`) on `v5p`

- `qwen3-0.6b` on `1x v5p-8`: Bit-exact match (`max diff 0.0`) against
  `to_maxtext` weights, checkpoints, and losses.
- `llama3.1-70b` (`141 GB`, 30 shards): Loaded in **512 s** on `1x v5p-32` (`4`
  hosts) and **509 s** across both slices on `2x v5p-32` (`8` hosts,
  `dcn_data_parallelism=2`) + 2 training steps, with zero DCN hangs.

#### C. Streaming V2 Prototype (PR #6223 / `SafetensorsReader`) on `v5p`

- `DeepSeek V4 284B` (`568.7 GB`) on `v5p-32`: **163.5 s** (`3.48 GB/s`), peak
  HBM **40.8 GB/chip**; forward-pass logit check `max KL 0.340` (`mean 0.020`,
  top-1 match).
- `Qwen3.5-397B-A17B` (`792.7 GB`) on `v5p-64`: **115.0 s** (`6.89 GB/s`), peak
  HBM **28.6 GB/chip** (`149.8 s` / `5.29 GB/s` / `54.2 GB` HBM/chip on
  `v5p-32`); `max KL 0.00157`, top-10 overlap `9/10`.

### 3. TPU `v5e` (`v5e-256` 64 Hosts & `2x v5e-256` 128 Hosts)

TPU `v5e-256` hosts have `16 GB` HBM/chip (`202 GB` host RAM). Leading-axis
contiguous sharded reads
([PR #5828](https://github.com/AI-Hypercomputer/maxtext/pull/5828)) and
`_GcsfsPath` ([PR #6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319))
eliminate multi-host read amplification and host OOMs:

| Model                   |    Size    | Topology                   | Read (s)  | Transform (s) | Host RAM / Improvement Note                                                                                 |
| :---------------------- | :--------: | :------------------------- | :-------: | :-----------: | :---------------------------------------------------------------------------------------------------------- |
| `qwen3-30b-a3b`         |  61.1 GB   | `v5e-256` (`64` hosts)     |   112 s   |     27 s      | `1.04 GB` downloaded/host; PR #6319 drops read RAM from **`>170 GB` (OOM) to `8.3 GB`** (`50 GB` peak RSS). |
| `mixtral-8x22b`         |  ~282 GB   | `v5e-256` (`64` hosts)     | **14 s**  |   **29 s**    | Down from `726 s` read (`60.3x` over-read) before PR #5828; step-0 loss `1.952`.                            |
| `qwen3-235b-a22b`       |  470.4 GB  | `v5e-256` (`64` hosts)     | **166 s** |   **55 s**    | Down from `>43 min` load-barrier timeout before PR #5828; step-0 loss `2.356`.                              |
| `qwen3.5-397b-a17b`     |  795.5 GB  | `v5e-256` (`64` hosts)     |   104 s   |     622 s     | Top next-token probe passes (forward training step exceeds `16 GB`/chip HBM on single `v5e-256`).           |
| `llama3.1-405b`         | 405B Dense | `2x v5e-256` (`128` hosts) | **53 s**  |   **52 s**    | Top next-token probe passes; zero HF HTTP `429` throttling.                                                 |
| `qwen3-480b-a35b`       |  961.0 GB  | `v5e-256` / `2x v5e-256`   | 189-226 s |   77-247 s    | PR #6319 loads scanned on `1x v5e-256` in **426 s** total (`63 GB/host` peak RAM vs `>202 GB` OOM).         |
| `deepseek3-671b` (BF16) |   1.3 TB   | `2x v5e-256` (`128` hosts) | **193 s** |   **120 s**   | Weights load cleanly (`18.98 GB` forward HBM need exceeds `16 GB` `v5e` chip capacity; trains on `tpu7x`).  |

### 4. TPU `v6e`, TPU `v4`, CPU & GPU Summary

- **TPU `v6e-32` (`8` hosts, `32 GB` HBM/chip):** Used for Tier-1 parity and
  regression tests, confirming bit-exact (`--atol=0 --rtol=0`) logits vs
  `to_maxtext`, identical 10-step `float32` training loss (`llama3.1-8b`), and
  `qwen3.5-35b-a3b-fp8` `max KL 1.6e-2` vs HF.
- **TPU `v4-8`:** Used to validate `SafetensorsReader`
  ([PR #5592](https://github.com/AI-Hypercomputer/maxtext/pull/5592)) and
  streaming reads
  ([PR #6223](https://github.com/AI-Hypercomputer/maxtext/pull/6223)),
  achieving **~2.8 GB/s** vs `~1.1 GB/s` with Orbax `0.12.x` on
  `DeepSeek-V2-Lite` and matching step-0 Orbax loss on `Qwen3-8B` and
  `DeepSeek-V2-Lite`.
- **CPU:** Used for golden logit parity (`qwen3-0.6b`, `qwen2.5-1.5b`,
  `gemma2-2b`, `gemma3-4b`, `gemma4-e2b`, `qwen3.5-9b`, `qwen3-vl-2b`),
  confirming `KL <= 8.6e-5` on `gemma3-4b` and `1.4e-7` on `qwen3-0.6b`.
- **GPUs (`H100` / `B200` / CUDA):** Not benchmarked in the current TPU
  validation suite. Multi-process `safetensors_dynamic` relies on standard JAX
  `Mesh` / `NamedSharding` and `sync_global_devices`, so GPU functionality and
  throughput remain to be validated separately.

## Troubleshooting

- **`ValueError: base_output_directory with gs:// prefix is required for huggingface downloads`**: When `load_parameters_path` points to an `hf://`
  repo (or is omitted to use `HF_IDS`), set `base_output_directory="gs://..."`
  so Host 0 can cache shards under `/hf_cache/`.
- **`-Instruct` models load base weights on older revisions**: Fixed in
  [PR #5610](https://github.com/AI-Hypercomputer/maxtext/pull/5610) - pass the
  exact `model_name` (e.g., `llama3.1-8b-Instruct`) rather than stripping the
  suffix.
- **Host RAM OOM (`Exit 137`) on MoE models when TensorFlow is installed**:
  Caused by `tf.io.gfile`'s `64 MiB` per-file read buffer when reading GCS via
  `epath`. Fixed by forcing `FileOptions(path_class=_GcsfsPath)`
  ([PR #6319](https://github.com/AI-Hypercomputer/maxtext/pull/6319)) or
  streaming via `SafetensorsReader`
  ([PR #6223](https://github.com/AI-Hypercomputer/maxtext/pull/6223)).
- **`enable_single_controller=True` error**: Pathways single-controller mode is
  currently rejected at config validation (`types.py`); run
  `safetensors_dynamic` under McJAX or use pre-converted Orbax checkpoints for
  Pathways single-controller jobs.
