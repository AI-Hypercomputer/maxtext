(safetensors-dynamic)=

# Dynamic SafeTensors loading

This guide explains how to use `source_checkpoint_layout="safetensors_dynamic"` to load Hugging Face (HF) `.safetensors` checkpoints directly into sharded MaxText weights at startup—transforming tensor layouts and stacking layers or experts on device without an offline [checkpoint conversion](checkpoint-conversion) step or an intermediate Orbax copy in Google Cloud Storage (GCS).

## Comparison with other checkpointing methods

| Method                                                                         | How it works                                                                                      | Extra GCS storage                                                                                                                           | Peak HBM during load                                                     |
| :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------- |
| **Offline `to_maxtext` + Orbax**                                               | Convert `.safetensors` offline on CPU, write Orbax to GCS, then restore Orbax at startup          | 2x (HF SafeTensors + Orbax checkpoint)                                                                                                      | Target MaxText weights                                                   |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=False`, default)**     | Read `.safetensors` sharded directly onto devices, then transform and stack in memory             | None initially (0x duplicate Orbax copy until first save; `hf://` caches raw shards under `<base_output_directory>/hf_cache/<org>_<repo>/`) | Whole HF checkpoint (sharded) + MaxText weights (~2x weights)            |
| **`safetensors_dynamic` (`safetensors_dynamic_streaming=True`, experimental)** | Stream `.safetensors` in bounded layer chunks and place directly into preallocated target weights | None initially (0x duplicate Orbax copy until first save)                                                                                   | MaxText target weights + bounded streaming read buffer (~8 GiB per host) |

Use `safetensors_dynamic` for cold starts, fine-tuning, or evaluation directly from Hugging Face `.safetensors` weights—especially for large models where single-host CPU `to_maxtext.py` hits host RAM limits or duplicate multi-terabyte Orbax copies are undesirable. For repeated restarts of smaller models where a pre-converted Orbax checkpoint already exists, restoring with `source_checkpoint_layout="orbax"` remains faster.

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
