(safetensors-dynamic)=

# Dynamic SafeTensors loading

This guide explains how to use `source_checkpoint_layout="safetensors_dynamic"` to load Hugging Face (HF) `.safetensors` checkpoints directly into sharded MaxText weights at startup—transforming tensor layouts and stacking layers or experts on device without an offline [checkpoint conversion](checkpoint-conversion) step or an intermediate Orbax copy in Google Cloud Storage (GCS).

## Comparison with other checkpointing methods

| Method                                                                         | How it works                                                                                      | Extra GCS storage                                                                                                                           | Peak HBM during load                                                     |
| :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
| **Offline `to_maxtext` + Orbax**                                               | Convert `.safetensors` offline on CPU, write Orbax to GCS, then restore Orbax at startup          | 2x (HF SafeTensors + Orbax checkpoint)                                                                                                      | Target MaxText weights                                                   |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=False`, default)**     | Read `.safetensors` sharded directly onto devices, then transform and stack in memory             | None initially (0x duplicate Orbax copy until first save; `hf://` caches raw shards under `<base_output_directory>/hf_cache/<org>_<repo>/`) | Whole HF checkpoint (sharded) + MaxText weights (~2x weights)            |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=True`, experimental)** | Stream `.safetensors` in bounded layer chunks and place directly into preallocated target weights | None initially (0x duplicate Orbax copy until first save)                                                                                   | MaxText target weights + bounded streaming read buffer (~8 GiB per host) |

Use `safetensors_dynamic` for cold starts, fine-tuning, or evaluation directly from Hugging Face `.safetensors` weights. By eliminating single-host CPU `to_maxtext.py` conversion and the intermediate Orbax write/restore round-trip, it achieves **1.8x** median end-to-end speedup (**3.05x** excluding HF download) on `<=120B` models and **2.9x–6.3x** end-to-end speedup on `>200B` models on TPU `v7x`, while avoiding single-host CPU RAM limits and duplicate multi-terabyte Orbax copies. For repeated restarts of smaller models where a pre-converted Orbax checkpoint already exists, restoring with `source_checkpoint_layout="orbax"` remains faster.

## Prerequisites

- MaxText must be installed in a Python virtual environment (or container image). For installation instructions, refer to the [installation documentation](install-from-source).
- The target model architecture must be supported by MaxText checkpoint conversion (see [Checkpoint Conversion Utilities](checkpoint-conversion) and [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py) for `model_name` and `HF_IDS` entries).
- Multi-controller JAX (McJAX) must be used (`enable_single_controller=False`). Pathways single-controller workloads are not supported with `safetensors_dynamic`.
- Authenticate with a Hugging Face access token (`hf_access_token`) if loading gated models from the Hugging Face Hub, and provide a `gs://` `base_output_directory` for Hub shard caching.

## Loading from a GCS or local directory

### Setup environment

```bash
export MODEL="llama3.1-70b"
export HF_CKPT_PATH="gs://my-bucket/hf_checkpoints/Llama-3.1-70B/"
export BASE_OUTPUT_DIRECTORY="gs://my-bucket/output/"
export RUN_NAME="my_dynamic_run"
```

### Run training

Point `load_parameters_path` to a directory containing `.safetensors` files (on `gs://` or local disk) and set `source_checkpoint_layout="safetensors_dynamic"`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=${MODEL?} \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path=${HF_CKPT_PATH?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    weight_dtype=bfloat16
```

### What happens on restart

At startup, MaxText checks the run's own checkpoint directory (`${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/`) first and only uses `safetensors_dynamic` when no checkpoint exists for `run_name`. With `enable_checkpointing=True` and the default `save_checkpoint_on_start=True`, MaxText writes an Orbax checkpoint at step 0, so restarting with the same `run_name` resumes from that Orbax checkpoint and ignores `load_parameters_path` (pointing `load_parameters_path` at a new `.safetensors` directory while reusing `run_name` has no effect, and Orbax storage is created from step 0 onward).

## Loading directly from Hugging Face Hub

When `load_parameters_path` is a Hugging Face repository ID (`hf://<org>/<repo>` or `<org>/<repo>`) or is left unset, Host 0 downloads any missing `.safetensors` shards into `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` (for example, `hf_cache/meta-llama_Llama-3.1-8B-Instruct/`) using a parallel worker pool while all other hosts wait until caching finishes. Blobs already present in that GCS directory are skipped, so subsequent runs sharing the same `base_output_directory` and repository are a cache hit and do not re-download.

If `load_parameters_path` is left unset, MaxText resolves the exact `model_name` key (preserving suffixes such as `-Instruct`) via `HF_IDS` in [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py).

### Setup environment

```bash
export MODEL="llama3.1-8b-Instruct"
export HF_REPO="hf://meta-llama/Llama-3.1-8B-Instruct"
export BASE_OUTPUT_DIRECTORY="gs://my-bucket/output/" # Must use gs:// for HF Hub caching
export RUN_NAME="my_hf_run"
```

### Run training

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=${MODEL?} \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path=${HF_REPO?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    hf_access_token="${HF_TOKEN?}" \
    weight_dtype=bfloat16
```

> **Note:** To use the default `HF_IDS` repository for `model_name`, leave `load_parameters_path` unset on the CLI instead of passing `hf://...`. Passing `load_parameters_path=""` explicitly on the CLI fails config validation because the CLI parser converts empty values to `None`.

## Streaming mode (experimental)

On very large models where holding both the sharded Hugging Face tensors and the converted MaxText weights simultaneously would exceed device HBM, set `safetensors_dynamic_streaming=True`. MaxText preallocates target arrays and streams decoder layers incrementally in bounded per-host chunks.

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME?} \
    model_name=qwen3.5-397b-a17b \
    source_checkpoint_layout="safetensors_dynamic" \
    safetensors_dynamic_streaming=True \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Qwen3.5-397B-A17B/" \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    weight_dtype=bfloat16
```

## Inference and logit verification

Both `model_creation_utils.from_pretrained` and the [forward pass logit checker](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/utils/forward_pass_logit_checker.py) support `source_checkpoint_layout="safetensors_dynamic"`:

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

## How it works

Instead of downloading an entire checkpoint onto a single CPU host, converting it sequentially in host RAM, and writing a second Orbax copy to GCS, `safetensors_dynamic` splits loading across all hosts in a cluster and converts weights directly on accelerators:

### 1. Weights-only target tree and mapping resolution

- At startup, `checkpointing.load_state_if_possible` (or `model_creation_utils.from_pretrained`) passes only the parameter leaves (`_abstract_params` or NNX parameter state, excluding optimizer state) into `load_safetensors_dynamic_state`.
- MaxText resolves the exact `model_name` key (preserving `-Instruct` suffixes) against `HF_MODEL_CONFIGS`, `PARAM_MAPPING`, and `HOOK_FNS` to build the Hugging Face-to-MaxText key map and per-tensor transform hooks.
- Any older multimodal checkpoint prefixes (for example legacy Gemma 3 `language_model.model.` or `vision_tower.` keys) are normalized transparently to match current `PARAM_MAPPING` keys.

### 2. Optional Host-0 Hugging Face Hub → GCS caching

When `load_parameters_path` points to a Hub repo (`hf://<org>/<repo>`, `<org>/<repo>`, or is left unset to use `HF_IDS[model_name]`):

- **Single-host API discovery**: Only Host 0 (`jax.process_index() == 0`) lists remote files via `HfFileSystem` (`*.safetensors`, plus `config.json` when `safetensors_dynamic_streaming=True`). Avoiding per-host HF API calls prevents HTTP `429` rate limiting on `128+`-host jobs.
- **Parallel cache population**: Host 0 lists existing objects under `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/`, skips any blobs already present, and downloads only missing files using a `32`-worker spawned process pool (`32 MB` upload chunks, up to `5 s` random startup jitter, and up to `5` retries per file with `15 s` backoff).
- **Cluster barrier**: All other hosts wait on `jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")` before reading from GCS.

### 3. Default path (`safetensors_dynamic_streaming=False`): Distributed contiguous reads and in-memory transforms

- **Contiguous leading-axis sharded I/O**: SafeTensors stores tensors in row-major order. On multi-process runs, `_leading_axis_shardings` shards each tensor along **axis `0` only** over a 2D `("process", "local")` mesh so every host reads a single contiguous byte range per tensor. Smaller tensors that cannot divide across all devices are sharded across processes only (then replicated locally) or replicated.
- **Eliminating GCS over-reads**: Multi-process reads pass `SafetensorsOptions(max_over_read_ratio=1.0)` (preventing Orbax from falling back to full-span reads on MoE files with `>1,024` tensors) and temporarily set `gcsfs.GCSFileSystem.default_block_size = 0` to disable `gcsfs`'s `5 MiB` read-ahead buffer during small ranged reads.
- **Refcounted hooks, stacking, and chunked resharding**:
  - As `transform_hf_state_to_mt_state` processes each target key, a `remaining_uses` counter immediately pops each HF tensor from memory after its final use (handling HF tensors shared across multiple leaves, such as fused gate/up projections split into `wi_0` and `wi_1`).
  - Hooks run on local device shards whenever possible; if a NumPy-only hook hits a tensor split across hosts, MaxText falls back to `1 GiB` sliced cross-host gather/scatter.
  - Decoder layer and MoE expert lists are concatenated via balanced binary-tree stacking (`_binary_chunked_stack`) to avoid quadratic memory growth, preserving variable collections (`MoEBiasVar`, `Tid2EidVar`) in their expected dtype (for example FP32 router biases), and arrays larger than `1 GiB` are resharded in chunks to bound peak HBM.
  - Finally, `_raise_on_weight_mismatch` verifies every expected parameter leaf in `abstract_params` was materialized.

### 4. Streaming path (`safetensors_dynamic_streaming=True`, experimental)

- **Single-pass headers & `64`-thread `SafetensorsReader`**: Parses `.safetensors` headers once on initialization and reads slices directly onto local TPU chips via `32 MiB` ranged GCS requests across `64` threads per host. Very small sharded tensors (such as `16 MB` MoE experts where per-chip row slices would cause excessive tiny requests) are read whole per device and rearranged over ICI.
- **Double-buffered layer-by-layer writes**: Decoder layers are packed into `~4 GiB/host` read calls (`READ_BYTES_PER_HOST`). While the current chunk is transformed on device and written in place into preallocated target arrays via jitted `dynamic_update_slice`, a background thread prefetches the next chunk's bytes onto host/device memory, keeping transient overhead near two read chunks (`~8 GiB/host`).

## HF Hub-to-GCS cache download time

When loading from Hugging Face Hub for the first time, **Host 0** transfers shards from Hugging Face Hub into `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` before multi-host loading begins. On subsequent runs sharing the same `base_output_directory` and repo (or when pointing `load_parameters_path` directly at an existing `gs://` directory), existing blobs are skipped and download time is **0 s**.

| Model               | Checkpoint Size | Cold HF→GCS Download (s) | `safetensors_dynamic` Load + Transform (s) | Cold E2E Total (s)  | Download Share of Cold Startup |
| :------------------ | :-------------- | :----------------------- | :----------------------------------------- | :------------------ | :----------------------------- |
| `llama3.1-8b`       | 16.1 GB         | 62.91                    | 28.52                                      | 91.43               | 68.8%                          |
| `qwen3-30b-a3b`     | 61.1 GB         | 78.82                    | 129.52                                     | 208.34              | 37.8%                          |
| `qwen3-32b`         | 65.5 GB         | 107.24                   | 88.92                                      | 196.16              | 54.7%                          |
| `mixtral-8x7b`      | 93.4 GB         | 239.75                   | 63.12                                      | 302.87              | 79.2%                          |
| `llama3.1-70b`      | 141.1 GB        | 197.80                   | 104.98                                     | 302.78              | 65.3%                          |
| `gpt-oss-120b`      | 233.7 GB        | 350.33                   | 390.69                                     | 741.02              | 47.3%                          |
| `qwen3-235b-a22b`   | 470.4 GB        | 583.31                   | 697.13 (unscanned) / 757.50 (scanned)      | 1,280.44 / 1,340.81 | 45.6% / 43.5%                  |
| `qwen3.5-397b-a17b` | 795.5 GB        | 814.53                   | 954.34 (unscanned)                         | 1,768.87            | 46.0%                          |
| `qwen3-480b-a35b`   | 961.0 GB        | 1,046.14                 | 718.65 (unscanned) / 757.70 (scanned)      | 1,764.79 / 1,803.84 | 59.3% / 58.0%                  |

> **Tip:** On large multi-host TPU allocations, pre-populating `${BASE_OUTPUT_DIRECTORY}/hf_cache/<org>_<repo>/` (or a regional `gs://` SafeTensors directory) before launching multi-slice training avoids spending cluster-wide accelerator time waiting on Host 0's HF Hub download.

## Performance and tested models

Because all hosts read shards concurrently over GCS and transform weights on accelerators without serializing a full Orbax checkpoint first, `safetensors_dynamic` significantly reduces startup time—especially on warm GCS caches (no HF download step), where speedups reach **3.05x** median on `<=120B` models and **4.55x–10.82x** on `>200B` models.

Across 44 registered `<=120B` models with valid `to_maxtext` baselines on TPU `v6e`/`v5e`/`CPU`, **42/44** produced bit-exact logits (`--atol=0 --rtol=0`) against offline `to_maxtext` checkpoints (the remaining 2 differed only by `float32`/`bfloat16` cast rounding and passed HF logit checks), and 10-step `float32` training on `llama3.1-8b` (`v6e-32`) produced identical step-by-step losses.

### Representative models `<=120B` (TPU `v7x`, 1–2 hosts / 8–16 cores)

Across 14 models (`qwen3-0.6b` through `gpt-oss-120b`), `safetensors_dynamic` achieved a **1.77x** median end-to-end speedup (range `0.65x–4.21x`, faster in 56/58 runs) and a **3.05x** median speedup excluding HF-to-GCS download (up to **14.68x**):

| Model           | Size     | Config (Scan, Slices (Cores)) | HF→GCS (s) | `to_maxtext` Convert + Load (s) | `safetensors_dynamic` Load + Transform (s) | E2E Speedup | Excl. Download Speedup |
| :-------------- | :------- | :---------------------------- | :--------- | :------------------------------ | :----------------------------------------- | :---------- | :--------------------- |
| `llama3.1-8b`   | 16.1 GB  | Scanned, `2 (16)`             | 62.91      | 89.58 + 17.92                   | 28.52 (`10.89 + 17.61`)                    | 1.86x       | 3.77x                  |
| `qwen3-32b`     | 65.5 GB  | Scanned, `2 (16)`             | 107.24     | 381.46 + 238.09                 | 88.92 (`35.44 + 53.42`)                    | 3.71x       | 6.97x                  |
| `qwen3-30b-a3b` | 61.1 GB  | Scanned, `1 (8)`              | 78.82      | 151.57 + 386.06                 | 129.52 (`99.12 + 30.38`)                   | 2.96x       | 4.15x                  |
| `mixtral-8x7b`  | 93.4 GB  | Scanned, `2 (16)`             | 239.75     | 134.07 + 792.49                 | 63.12 (`45.96 + 17.16`)                    | 3.85x       | 14.68x                 |
| `llama3.1-70b`  | 141.1 GB | Scanned, `2 (16)`             | 197.80     | 780.27 + 295.79                 | 104.98 (`72.44 + 32.48`)                   | 4.21x       | 10.25x                 |
| `gpt-oss-120b`  | 233.7 GB | Unscanned, `2 (16)`           | 350.33     | 617.84 + 115.05                 | 390.69 (`122.59 + 268.06`)                 | 1.46x       | 1.88x                  |

### Large models `>200B` (TPU `v7x` & `v5e`)

On `>200B` models, single-host CPU `to_maxtext` ran out of host RAM (`>1,014 GB`) on all scanned runs and all models `>=480B`, whereas `safetensors_dynamic` loaded checkpoints up to **3.2 TB**:

| Model               | Size     | Platform & Config                       | `to_maxtext` Convert + Load (E2E)       | `safetensors_dynamic` Load + Transform (E2E) | Speedup (E2E / Excl. Download)   |
| :------------------ | :------- | :-------------------------------------- | :-------------------------------------- | :------------------------------------------- | :------------------------------- |
| `qwen3-235b-a22b`   | 470.4 GB | `v7x`, Scanned `1 (8)`                  | Failed (Host RAM OOM)                   | 757.50 s (`1,340.81 s` E2E)                  | Only `safetensors_dynamic` loads |
| `qwen3-235b-a22b`   | 470.4 GB | `v7x`, Unscanned `1 (8)`                | 3,754.33 s E2E (`2,625.50 + 545.51`)    | 697.13 s (`1,280.44 s` E2E)                  | 2.93x / 4.55x                    |
| `qwen3.5-397b-a17b` | 795.5 GB | `v7x`, Unscanned `1 (32)`               | 11,144.07 s E2E (`8,096.53 + 2,233.01`) | 954.34 s (`1,768.87 s` E2E)                  | 6.30x / 10.82x                   |
| `qwen3-480b-a35b`   | 961.0 GB | `v7x`, Unscanned `2 (32)`               | Failed (Host RAM OOM)                   | 718.65 s (`1,764.79 s` E2E)                  | Only `safetensors_dynamic` loads |
| `qwen3-480b-a35b`   | 961.0 GB | `v7x`, Scanned `2 (32)`                 | Failed (Host RAM OOM)                   | 757.70 s (`1,803.84 s` E2E)                  | Only `safetensors_dynamic` loads |
| `deepseek3-671b`    | 1.3 TB   | `2x v5e-256` (`128` hosts), Scanned     | Exceeds single-host RAM                 | 313 s (`193 s` read + `120 s` transform)     | Only `safetensors_dynamic` loads |
| `deepseek4-1600b`\* | 3.2 TB   | `v7x` (`8` hosts / `64` cores), Scanned | Exceeds single-host RAM                 | 971.49 s (`698.05 + 273.22`)                 | Only `safetensors_dynamic` loads |

*\* `deepseek4-1600b` was benchmarked using locally staged synthetic BF16 `.safetensors` files (`0 s` download) to validate multi-host I/O and HBM scaling up to `3.2 TB`. Real BF16 GCS loads + 2 training steps were also verified on `tpu7x-128` (`16–32` hosts) for `llama3.1-405b` (`36 s` read / `25 s` transform), `qwen3.5-397b-a17b` (`51 s` / `587 s`), `qwen3-480b-a35b` (`191 s` / `68 s`), `deepseek4-284b` (`296 s` / `150 s`), and `deepseek3-671b` (`252 s` / `48 s`).*

## MaxText configuration

| Flag                                   | Description                                                                                                                                                                                          | Type      | Default     |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------- | :---------- |
| `source_checkpoint_layout`             | Specifies the source checkpoint format to load. Set to `"safetensors_dynamic"` to load and convert Hugging Face `.safetensors` weights directly at startup.                                          | `string`  | `"orbax"`   |
| `safetensors_dynamic_streaming`        | Experimental. When `True` with `source_checkpoint_layout="safetensors_dynamic"`, streams decoder layers sequentially to limit transient HBM usage.                                                   | `boolean` | `False`     |
| `load_parameters_path`                 | Path to a `gs://` or local `.safetensors` directory, `hf://<org>/<repo>`, or `<org>/<repo>`. Leave unset to use `HF_IDS[model_name]`; passing `load_parameters_path=""` on the CLI fails validation. | `string`  | `""`        |
| `base_output_directory`                | Output directory for training artifacts. Must use a `gs://` prefix when caching weights from Hugging Face Hub (`hf_cache/`).                                                                         | `string`  | `""`        |
| `hf_access_token`                      | Hugging Face authentication token used by Host 0 when downloading from private or gated Hugging Face repositories.                                                                                   | `string`  | `""`        |
| `weight_dtype`                         | Target weight dtype in MaxText (for example `"bfloat16"`). We recommend `"bfloat16"` when loading BF16 Hugging Face checkpoints to reduce memory usage.                                              | `string`  | `"float32"` |
| `scan_layers`                          | Controls scanned (`True`) or unscanned (`False`) decoder layer layout. Works with both modes when using `safetensors_dynamic`.                                                                       | `boolean` | `True`      |
| `enable_single_controller`             | Enables Pathways single-controller execution. Must remain `False` for `safetensors_dynamic`.                                                                                                         | `boolean` | `False`     |
| `enable_single_replica_ckpt_restoring` | Single-replica restore optimization for Orbax checkpoints; not used by `safetensors_dynamic`.                                                                                                        | `boolean` | `False`     |

## Troubleshooting

- **Pathways error (`enable_single_controller=True`)**: `source_checkpoint_layout="safetensors_dynamic"` raises a `ValueError` on Pathways. Either run on McJAX (`enable_single_controller=False`) or convert the checkpoint offline with `to_maxtext.py` and load it with `source_checkpoint_layout="orbax"`.
- **Missing `gs://` prefix for Hugging Face Hub downloads**: When loading via `hf://<repo>` or leaving `load_parameters_path` unset, set `base_output_directory="gs://<bucket>/..."` so Host 0 can stage shards to shared GCS storage, or download the `.safetensors` files to a local/GCS directory beforehand.
- **Changes to `load_parameters_path` ignored on restart**: If an Orbax checkpoint already exists under `${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/` (created at step 0 when `enable_checkpointing=True` and `save_checkpoint_on_start=True`), restarts resume from Orbax and ignore `load_parameters_path`. Use a new `run_name` or clear the run's checkpoint directory to reload from `.safetensors`.
- **Weight-mismatch errors on startup**: Verify that `model_name` (including suffixes such as `-Instruct`), `scan_layers`, and any model overrides match the Hugging Face checkpoint being loaded.
