# Checkpoint conversion utilities

This guide provides instructions for using the [scripts](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/utils/ckpt_conversion) that convert model checkpoints bidirectionally between Hugging Face and MaxText formats.

## Supported models

The following models are supported:

| Model Family            | Sizes                  | HF $\\to$ Orbax (scan) | HF $\\to$ Orbax (unscan) | Orbax (scan) $\\to$ HF | Orbax (unscan) $\\to$ HF |
| :---------------------- | :--------------------- | :--------------------: | :----------------------: | :--------------------: | :----------------------: |
| **Gemma2**              | 2B, 9B, 27B            |           √            |            √             |           √            |            √             |
| **Gemma3** (Multimodal) | 4B, 12B, 27B           |           -            |            √             |           -            |            √             |
| **Llama3.1**            | 8B, 70B, 450B          |           √            |            √             |           √            |            √             |
| **Qwen3**               | 0.6B, 4B, 8B, 14B, 32B |           √            |            √             |           √            |            √             |
| **Qwen3 MoE**           | 30B, 235B, 480B        |           √            |            √             |           √            |            √             |
| **Mixtral**             | 8x7B, 8x22B            |           √            |            √             |           √            |            √             |
| **GPT-OSS**             | 20B, 120B              |           √            |            √             |           √            |            √             |
| **DeepSeek3**           | 671B                   |           -            |            -             |           √            |            -             |

## Prerequisites

- Hugging Face requires Pytorch.
- Hugging Face model checkpoints require local disk space.
  - The model files are always downloaded to a disk cache first before being loaded into memory (for more info, please consult Hugging Face [docs](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)). The default local storage path for Hugging Face models is \$HOME/.cache/huggingface/hub

## Hugging Face to MaxText

Use the `to_maxtext.py` script to convert a Hugging Face model into a MaxText checkpoint. The script will automatically download the specified model from the Hugging Face Hub, perform conversion, and save converted checkpoints to given output directory.

\*\**For a complete example, see the test script at [`end_to_end/tpu/qwen3/4b/test_qwen3_to_mt.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/qwen3/4b/test_qwen3_to_mt.sh) and [`end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh).*

### Usage

First, make sure python3 virtual environment for MaxText is set up and enabled.

```bash
export VENV_NAME=<your virtual env name> # e.g., maxtext_venv
pip install uv
uv venv --python 3.12 --seed $VENV_NAME
source $VENV_NAME/bin/activate
```

Second, ensure you have the necessary dependencies installed (PyTorch for the conversion script).

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Third, setup following environment variables for conversion script

```bash
# -- Model configuration --
export HF_MODEL=<Hugging Face Model to be converted to MaxText> # e.g. 'llama3.1-8b-Instruct'
export HF_TOKEN=<Hugging Face access token> # your token to access gated HF repos

# -- MaxText configuration --
export MODEL_CHECKPOINT_DIRECTORY=<output directory to store output of checking point> # e.g., gs://my-bucket/my-checkpoint-directory

# -- storage and format options
export USE_ZARR3=<Flag to use zarr3> # Set to True to use zarr3 format (recommended for McJAX); set to False for Pathways.
export USE_OCDBT=<Flag to use ocdbt> # Set to True to use OCDBT format (recommended for McJAX); set to False for Pathways.

export LAZY_LOAD_TENSORS=<Flag to lazy load> # True to use lazy load, False to use eager load.
```

Finally, run below command to complete the conversion

```bash
python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=${HF_MODEL} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_CHECKPOINT_DIRECTORY} \
    scan_layers=True \
    use_multimodal=false \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=${USE_ZARR3} \
    checkpoint_storage_use_ocdbt=${USE_OCDBT} \
    --lazy_load_tensors=${LAZY_LOAD_TENSORS}
```

**Key arguments:**

- `model_name`: The model identifier, which should be defined in `src/MaxText/utils/utils.py`.
- `scan_layers`: Indicates if the output checkpoint is [scanned](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/checkpoints.md) (scan_layers=true) or unscanned (scan_layers=false).
- `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
- `hf_access_token`: Your Hugging Face token.
- `base_output_directory`: The path where the converted Orbax checkpoint will be stored; it can be Googld Cloud Storage (GCS) or local. If not set, the default output directory is `Maxtext/tmp`.
- `hardware=cpu`: run the conversion script on a CPU machine.
- `checkpoint_storage_use_zarr3`: # Set to True to use zarr3 format (recommended for McJAX); set to False for Pathways.
- `checkpoint_storage_use_ocdbt`: # Set to True to use OCDBT format (recommended for McJAX); set to False for Pathways.
- `--lazy_load_tensors` (optional): If `true`, loads Hugging Face weights on-demand to minimize RAM usage. For large models, it is recommended to use the `--lazy_load_tensors=true` flag to reduce memory usage during conversion. For example, converting a Llama3.1-70B model with `--lazy_load_tensors=true` uses around 200GB of RAM and completes in ~10 minutes.
- `--hf_model_path` (optional): Specifies a local directory containing the model weights. If unspecified, we use the [default Hugging Face repository ID](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/utils.py#L58-L85) (e.g., openai/gpt-oss-20b). This is necessary for locally dequantized models like GPT-OSS or DeepSeek.

Above command will download the Hugging Face model to local machine, convert it to the MaxText format and save it to `${MODEL_CHECKPOINT_DIRECTORY}/0/items`.

## MaxText to Hugging Face

Use the `to_huggingface.py` script to convert a MaxText checkpoint into the Hugging Face format. This is useful for sharing your models or integrating them with the Hugging Face ecosystem.
\*\**For a complete example, see the test script at [`end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh).*

### Usage

The following command converts a MaxText checkpoint and saves it locally, to GCS, or uploads it directly to the Hugging Face Hub.

```bash
python3 -m MaxText.utils.ckpt_conversion.to_huggingface src/MaxText/configs/base.yml \
    model_name=<MODEL_NAME> \
    load_parameters_path=<path-to-maxtext-checkpoint> \
    base_output_directory=<path-to-save-converted-checkpoint> \
    scan_layers=false \
    use_multimodal=false \
    hf_access_token=<your-hf-token> \
    weight_dtype=bfloat16
```

**Key arguments:**

- `load_parameters_path`: The path to the source MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
- `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
- `scan_layers`: Indicates if the output checkpoint is [scanned](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/checkpoints.md) (scan_layers=true) or unscanned (scan_layers=false).
- `hf_access_token`: Your Hugging Face token.
- `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
- `base_output_directory`: The path where the converted Orbax checkpoint will be stored; it can be Googld Cloud Storage (GCS), Hugging Face Hub or local. If not set, the default output directory is `Maxtext/tmp`.
- `weight_dtype`: dtype for MaxText weights. It affects the resulting HF weight dtype. Default value is `float32`. We recommend using `bfloat16` to save memory and speed up conversion.

## Verifying conversion correctness

To ensure the conversion was successful, you can use the `tests/utils/forward_pass_logit_checker.py` script. It runs a forward pass on both the original and converted models and compares the output logits to verify conversion. It is used to verify the bidirectional conversion.

### Usage

```bash
python3 -m tests.utils.forward_pass_logit_checker src/MaxText/configs/base.yml \
    tokenizer_path=assets/<tokenizer> \
    load_parameters_path=<path-to-maxtext-checkpoint> \
    model_name=<MODEL_NAME> \
    scan_layers=false \
    max_prefill_predict_length=4 \
    max_target_length=8 \
    use_multimodal=false \
    --run_hf_model=True \
    --hf_model_path=<path-to-HF-checkpoint> \
    --max_kl_div=0.015
```

**Key arguments:**

- `load_parameters_path`: The path to the source MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
- `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
- `scan_layers`: Indicates if the output checkpoint is scanned (scan_layers=true) or unscanned (scan_layers=false).
- `use_multimodal`: Indicates if multimodality is used.
- `--run_hf_model`: Indicates if loading Hugging Face model from the hf_model_path. If not set, it will compare the maxtext logits with pre-saved golden logits.
- `--hf_model_path`: The path to the Hugging Face checkpoint.
- `--max_kl_div`: Max KL divergence tolerance during comparisons.

**Example successful conversion verification:**

Here is part of the output of forward_pass_logit_checker for the gemma2-2b.

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

## Adding support for new models

To extend conversion support to a new model architecture, you must define its specific parameter and configuration mappings. The conversion logic is decoupled, so you only need to modify the mapping files.

1. **Add parameter mappings**:

- In [`utils/param_mapping.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/param_mapping.py), add the parameter name mappings(`def {MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING`). This is the 1-to-1 mappings of parameters names per layer.
- In [`utils/param_mapping.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/param_mapping.py), add the `hook_fn` logic (`def {MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN`). This is the transformation needed per layer.

2. **Add Hugging Face weights Shape**: In [`utils/hf_shape.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/hf_shape.py), define the tensor shape of Hugging Face format (`def {MODEL}_HF_WEIGHTS_TO_SHAPE`). This is used to ensure the tensor shape is matched after to_huggingface conversion.
1. **Register model key**: In [`utils/utils.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/utils.py), add the new model key in `HF_IDS`.
1. **Add transformer config**: In [`utils/hf_model_configs.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_conversion/utils/hf_model_configs.py), add the `transformers.Config` object, describing the Hugging Face model configuration (defined in ['src/MaxText/configs/models'](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/configs/models)). **Note**: This configuration must precisely match the MaxText model's architecture.

Here is an example [PR to add support for gemma3 multi-modal model](https://github.com/AI-Hypercomputer/maxtext/pull/1983)

## Debugging tips

If the converted checkpoint can not get loaded and got error like: "type \<class 'jax.\_src.core.ShapeDtypeStruct'> is not a valid JAX type."

- **Potential Cause**: The scan_layers flag is set wrong.

If a converted checkpoint loads without errors but produces incorrect output, consider these common issues:

- **Symptom**: The model generates garbage or nonsensical tokens.

  - **Potential Cause**: The query/key/value (Q/K/V) or Out vectors weights were likely reshaped incorrectly during conversion.

- **Symptom**: The model generates repetitive text sequences.

  - **Potential Cause**: The layer normalization parameters may have been converted incorrectly.
