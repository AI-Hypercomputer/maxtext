# Checkpoint Conversion Utilities

This guide provides instructions to use [checkpoint conversion scripts](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/checkpoint_conversion) to convert model checkpoints bidirectionally between Hugging Face and MaxText formats.

## Supported models

The following models are supported:

| Model Family            | Sizes                  | HF $\to$ Orbax (scan) | HF $\to$ Orbax (unscan) | Orbax (scan) $\to$ HF | Orbax (unscan) $\to$ HF |
| :---------------------- | :--------------------- | :-------------------: | :---------------------: | :-------------------: | :---------------------: |
| **Gemma2**              | 2B, 9B, 27B            |           √           |            √            |           √           |            √            |
| **Gemma3** (Multimodal) | 4B, 12B, 27B           |           √           |            √            |           √           |            √            |
| **Llama3.1**            | 8B, 70B, 450B          |           √           |            √            |           √           |            √            |
| **Qwen2.5**             | 1.5B, 7B, 14B          |           √           |            √            |           √           |            √            |
| **Qwen3**               | 0.6B, 4B, 8B, 14B, 32B |           √           |            √            |           √           |            √            |
| **Qwen3 MoE**           | 30B, 235B, 480B        |           √           |            √            |           √           |            √            |
| **Mixtral**             | 8x7B, 8x22B            |           √           |            √            |           √           |            √            |
| **GPT-OSS**             | 20B, 120B              |           √           |            √            |           √           |            √            |
| **DeepSeek2**           | 16B                    |           √           |            √            |           √           |            √            |
| **DeepSeek3**           | 671B                   |           √           |            √            |           √           |            √            |
| **DeepSeek3.2**         | 671B                   |           √           |            √            |           -           |            -            |
| **Qwen3 Next**          | 80B                    |           √           |            √            |           √           |            √            |

## Prerequisites

- MaxText must be installed in a Python virtual environment using the `maxtext[tpu]` option. For instructions on installing MaxText on your VM, please refer to the official [installation documentation](../../install_maxtext.md).
- Hugging Face model checkpoints are cached locally at `$HOME/.cache/huggingface/hub` before conversion. Ensure you have sufficient disk space.
- Authenticate via the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/v0.21.2/guides/cli) if using private or gated models.

## Hugging Face to MaxText

Use the `to_maxtext.py` script to convert a Hugging Face model checkpoint into a MaxText checkpoint. The script will automatically download the specified model from the Hugging Face Hub, perform conversion, and save converted checkpoints to the given output directory.

> **Note:** For more information, checkout [qwen3-4b example script](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/qwen3/4b/test_qwen3_to_mt.sh) and [gemma3-4b example script](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh).

### Setup Environment

```bash
# Setup environment variables
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<HF_MODEL> # e.g. 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# The directory where the converted Orbax checkpoint will be stored (GCS bucket or local directory).
export BASE_OUTPUT_DIRECTORY=<CKPT_PATH> # e.g., gs://my-bucket/my-checkpoint-directory

# Set to 1 if you intend to use Pathways for training, 0 for McJAX.
export USE_PATHWAYS=0
```

### Run Conversion

```bash
# Optional: If you run out of disk space when downloading Hugging Face safetensors,
# customize your "HF_HOME" to redirect the cache to a larger or mounted disk (e.g., on a TPU VM).
# export HF_HOME="/dev/shm/huggingface_tmp"

python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=${MODEL?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    scan_layers=True \
    use_multimodal=false \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS)) \
    checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS)) \
    --save_dtype=bfloat16
```

You can find your converted checkpoint files under `${BASE_OUTPUT_DIRECTORY}/0/items`.

### Key Parameters

- `model_name`: The specific model identifier. It must match a supported entry in the MaxText [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/16b684840db9b96b19e24e84ac49f06af7204ae3/src/maxtext/utils/globals.py#L46C1-L46C7).
- `scan_layers`: Controls whether the output uses a scanned (`scan_layers=true`) or unscanned (`scan_layers=false`) checkpoint format. Refer [here](../../reference/core_concepts/checkpoints.md) for more information. **IMPORTANT:** This setting *must* match the `scan_layers` value used during model training or loading. A mismatch will cause PyTree loading errors (though MaxText will intercept these and raise a descriptive `ValueError` explaining the mismatch).
- `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
- `base_output_directory`: The path where the converted Orbax checkpoint will be stored; it can be Google Cloud Storage (GCS) or local.
- `hardware=cpu`: The conversion script runs on a CPU machine.
- `checkpoint_storage_use_zarr3` and `checkpoint_storage_use_ocdbt`: These storage flags enable McJAX compatibility when set to True (the default). For Pathways, these should be False.
- `--hf_model_path` (Optional): Specifies a customized remote directory or local directory containing the model weights. If unspecified, we use the [default Hugging Face repository ID](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py) (e.g., openai/gpt-oss-20b). This is necessary for locally dequantized models like GPT-OSS or DeepSeek.
- `--save_dtype` (Optional): Specifies the data type of saved model weights. Default to `bfloat16` to save memory.

## MaxText to Hugging Face

Use the `to_huggingface.py` script to convert a MaxText checkpoint into the Hugging Face format. This is useful for sharing your models or integrating them with the Hugging Face ecosystem.

> **Note:** For more information, checkout [qwen3-4b example script](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh).

### Setup Environment

```bash
# Setup environment variables
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g. 'qwen3-4b'

# The path to the MaxText compatible model checkpoint (Orbax format).
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items

# -- MaxText configuration --
# The directory where the converted Hugging Face checkpoint will be stored (GCS bucket,
# local directory, or Hugging Face Hub path).
export BASE_OUTPUT_DIRECTORY=<HF_CKPT_PATH> # e.g., gs://my-bucket/my-checkpoint-directory
```

### Run Conversion

The following command converts a MaxText checkpoint and saves it locally, to GCS (`gs://`), or uploads it directly to the Hugging Face Hub (`hf://`).

```bash
python3 -m maxtext.checkpoint_conversion.to_huggingface \
    model_name=${MODEL?} \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    hardware=cpu \
    skip_jax_distributed_system=true \
    scan_layers=false \
    use_multimodal=false \
    weight_dtype=bfloat16
```

### Key Parameters

- `model_name`: The specific model identifier. It must match a supported entry in the MaxText [globals.py](https://github.com/AI-Hypercomputer/maxtext/blob/16b684840db9b96b19e24e84ac49f06af7204ae3/src/maxtext/utils/globals.py#L46C1-L46C7).
- `load_parameters_path`: The path to the MaxText Orbax checkpoint.
- `scan_layers`: Controls whether the output uses a scanned (`scan_layers=true`) or unscanned (`scan_layers=false`) checkpoint format. Refer [here](../../reference/core_concepts/checkpoints.md) for more information.
- `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
- `hardware=cpu`: The conversion script runs on a CPU machine.
- `base_output_directory`: The path where the converted checkpoint will be stored; it can be Google Cloud Storage (GCS), Hugging Face Hub or local.
- `weight_dtype`: It affects the resulting Hugging Face weight dtype. Default value is `float32`. We recommend using `bfloat16` to save memory and speed up conversion.

## Verifying conversion correctness

To ensure the conversion was successful, you can use the [test script](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/utils/forward_pass_logit_checker.py). It runs a forward pass on both the original and converted models and compares the output logits to verify conversion. It is used to verify the bidirectional conversion.

> **Note:** This correctness test will only work when MaxText is installed from source by following the installation instructions [here](../../install_maxtext.md#from-source).

### Setup Environment

```bash
# Setup environment variables
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g. 'qwen3-4b'

# The path to the MaxText compatible model checkpoint (Orbax format).
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items

# The path to the Hugging Face model checkpoint directory.
export HF_CKPT_PATH=<HF_CKPT_PATH> # e.g., gs://my-bucket/my-checkpoint-directory

# Install PyTorch (in MaxText virtual environment)
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run Correctness Test

```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    load_parameters_path=${MAXTEXT_CKPT_PATH?} \
    model_name=${MODEL?} \
    skip_jax_distributed_system=true \
    scan_layers=false \
    max_prefill_predict_length=4 \
    max_target_length=8 \
    use_multimodal=false \
    --run_hf_model=True \
    --hf_model_path=${HF_CKPT_PATH?} \
    --max_kl_div=0.015
```

### Key Parameters

- `load_parameters_path`: The path to the MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
- `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
- `scan_layers`: Controls whether the output uses a scanned (`scan_layers=true`) or unscanned (`scan_layers=false`) checkpoint format. Refer [here](../../reference/core_concepts/checkpoints.md) for more information.
- `use_multimodal`: Indicates if multimodality is used.
- `--run_hf_model` (Optional): Indicates if loading Hugging Face model from the hf_model_path. If not set, it will compare the maxtext logits with pre-saved golden logits.
- `--hf_model_path` (Optional): The path to the Hugging Face checkpoint (if `--run_hf_model=True`).
- `--golden_logits_path` (Optional): The pre-saved golden logits. (if `--run_hf_model` is not set).
- `--max_kl_div`: Max KL divergence tolerance during comparisons.

### Example of Successful Conversion Verification

Here is part of the output of `forward_pass_logit_checker` for the gemma2-2b.

```
--- Prompt: What is the ---

--- MaxText model top 10 tokens ---
| Token ID   | Token                | Score      |
|------------|----------------------|------------|
| 5830       | difference           | 27.2500    |
| 1963       | best                 | 26.6250    |
| 5316       | average              | 26.3750    |
| 2669       | change               | 26.1250    |
| 12070      | percentage           | 26.1250    |
| 1618       | value                | 25.8750    |
| 1546       | most                 | 25.7500    |
| 66202      | molar                | 25.5000    |
| 3051       | total                | 25.5000    |
| 1503       | name                 | 25.3750    |


--- HF model top 10 tokens ---
| Token ID   | Token                | Score      |
|------------|----------------------|------------|
| 5830       | difference           | 27.2500    |
| 1963       | best                 | 26.6250    |
| 5316       | average              | 26.3750    |
| 12070      | percentage           | 26.1250    |
| 2669       | change               | 26.1250    |
| 1618       | value                | 25.8750    |
| 1546       | most                 | 25.7500    |
| 66202      | molar                | 25.5000    |
| 3051       | total                | 25.5000    |
| 6187       | purpose              | 25.3750    |


--- Similarity Metrics of Top Tokens ---
| Metric                         | Value                |
|--------------------------------|----------------------|
| overlap_count                  | 9/10                 |
| jaccard_similarity             | 0.8181818181818182   |
| rank_agreement_percentage      | 70.0                 |


Average KL divergence per token (D_KL(P_golden || Q_model)): 0.000409

Max KL divergence for a single token in the set: 0.003497
```

______________________________________________________________________

## Development and Troubleshooting

### Inspection Tools

We provide a unified inspection tool In [`inspect_checkpoint.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/inspect_checkpoint.py) to help you view model structures. This is highly useful for both development and troubleshooting.

- **Memory-efficient:** Operates without requiring heavy compute or massive RAM allocation.
- **Detailed outputs:** Prints tensor keys and shapes. Optional: data types with `--check_dtype=True` flag.

**Tool 1 (HF Inspector)**. To inspect the HuggingFace checkpoint structure, from either `.safetensors` or `.pth` files:

```
python src/maxtext/checkpoint_conversion/inspect_checkpoint.py hf --path <local_hf_path> --format <safetensors | pth>
```

**Tool 2 (MaxText Inspector)**. To inspect the MaxText model structure:

```
python src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext model_name=<maxtext_model_name> scan_layers=<True | False>
```

**Tool 3 (Orbax Inspector)**. To inspect the Orbax checkpoint:

```
python src/maxtext/checkpoint_conversion/inspect_checkpoint.py orbax --path <local_orbax_path | gcs_orbax_path>
```

### Adding New Models

To extend conversion support to a new model architecture, you must define its specific parameter and configuration mappings. The conversion logic is decoupled, so you only need to modify the mapping files.

1. **Add parameter mappings**:

- As the first step, we inspect the checkpoint structures. To see the HuggingFace checkpoint structure, use **Tool 1 (HF Inspector)**. To see the MaxText model structure, use **Tool 2 (MaxText Inspector)**.

- In [`utils/param_mapping.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/param_mapping.py), add the parameter name mappings(`def {MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING`). This is the 1-to-1 mappings of parameters names per layer.

- In [`utils/param_mapping.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/param_mapping.py), add the `hook_fn` logic (`def {MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN`). This is the transformation needed per layer.

2. **Add Hugging Face weights Shape**: In [`utils/globals.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/hf_shape.py), define the tensor shape of Hugging Face format (`def {MODEL}_HF_WEIGHTS_TO_SHAPE`). This is used to ensure the tensor shape is matched after to_huggingface conversion.

3. **Register model key**: In [`utils/utils.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/globals.py), add the new model key in `HF_IDS`.

4. **Add transformer config**: In [`utils/hf_model_configs.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/hf_model_configs.py), add the `transformers.Config` object, describing the Hugging Face model configuration (defined in [`src/maxtext/configs/models`](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/configs/models)). This configuration must precisely match the MaxText model's architecture.

Here is an example [PR to add support for gemma3 multi-modal model](https://github.com/AI-Hypercomputer/maxtext/pull/1983).

### Common Errors

- **Error:** When loading a converted checkpoint, `Type ShapeDtypeStruct is not a valid JAX type`.

  - **Cause (most common): Structure mismatch** between the converted checkpoint and the MaxText model.

  - **Solution:** To see the MaxText model structure, use **Tool 2 (MaxText Inspector)**. To inspect the checkpoint, use **Tool 3 (Orbax Inspector)**.

- **Error:** `Type ShapeDtypeStruct is not a valid JAX type` or generic **PyTree structure/shape mismatches** (e.g., Orbax reporting `"X/Y paths matched"`, such as `143/145 paths`).

  - **Cause: Configuration mismatch** (e.g., `scan_layers`) between the checkpoint conversion script (e.g., `to_maxtext.py` or `to_huggingface.py`) and the trainer/inference runner (e.g., `train.py`).

  - **Solution:** Ensure the `scan_layers` flag is set to the exact same value (`True` or `False`) in both the conversion command and your training/execution command.

- **Error:** The converted checkpoint loads without errors but produces nonsensical output.

  - **Cause:** There is likely an error in the Q/K/V weight reshaping logic during conversion.

- **Error:** The model generates repetitive text sequences.

  - **Cause:** Layer normalization parameters were likely not mapped correctly.
