# Checkpoint conversion utilities

This guide provides instructions for using the scripts that convert model checkpoints bidirectionally between Hugging Face and MaxText formats.

## Supported models

The following models are supported:

- Gemma2 (2B, 9B, 27B).
- Gemma3 multimodal (4B, 12B, 27B).
- Qwen3 (0.6B, 4B, 8B, 14B, 32B).
- Mixtral (8x7B, 8x22B).

## Prerequisites
- Hugging Face requires Pytorch.
- Hugging Face model checkpoints require local disk space.
  - The model files are always downloaded to a disk cache first before being loaded into memory (for more info, please consult Hugging Face [docs](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)). The default local storage path for Hugging Face models is $HOME/.cache/huggingface/hub

## Hugging Face to MaxText

Use the `to_maxtext.py` script to convert a Hugging Face model into a MaxText checkpoint. The script will automatically download the specified model from the Hugging Face Hub, perform conversion, and save converted checkpoints to given output directory.

\*\**For a complete example, see the test script at [`end_to_end/tpu/qwen3/4b/test_qwen3.sh`](../../../end_to_end/tpu/qwen3/4b/test_qwen3.sh) and [`end_to_end/tpu/gemma3/4b/test_gemma3_unified.sh`](../../../end_to_end/tpu/gemma3/4b/test_gemma3_unified.sh).*

### Usage

The following command demonstrates how to run the conversion. You must provide your Hugging Face token in the `src/MaxText/configs/base.yml` file (hf_access_token).

```bash
python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/MaxText/configs/base.yml \
    model_name=<model-name> \
    base_output_directory=<gcs-path-to-save-checkpoint> \
    hf_access_token=<your-hf-token> \
    use_multimodal=false \
    scan_layers=false
```

**Key arguments:**

  * `model_name`: The model identifier, which should be defined in `src/MaxText/utils/utils.py`.
  * `scan_layers`: Indicates if the output checkpoint is [scanned](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/checkpoints.md) (scan_layers=true) or unscanned (scan_layers=false).
  * `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
  * `hf_access_token`: Your Hugging Face token.
  * `base_output_directory`: The path where the converted Orbax checkpoint will be stored; it can be Googld Cloud Storage (GCS) or local. If not set, the default output directory is `Maxtext/tmp`.

\*\**It only converts the official version of Hugging Face model. You can refer the supported official version in HF_IDS in `src/MaxText/utils/ckpt_conversion/utils/utils.py`*

## MaxText to Hugging Face

Use the `to_huggingface.py` script to convert a MaxText checkpoint into the Hugging Face format. This is useful for sharing your models or integrating them with the Hugging Face ecosystem.
\*\**For a complete example, see the test script at [`end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh`](../../../end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh).*

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
```

**Key arguments:**

  * `load_parameters_path`: The path to the source MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
  * `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
  * `scan_layers`: Indicates if the output checkpoint is [scanned](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/checkpoints.md)  (scan_layers=true) or unscanned (scan_layers=false).
  * `hf_access_token`: Your Hugging Face token.
  * `use_multimodal`: Indicates if multimodality is used, important for Gemma3.
  * `base_output_directory`: The path where the converted Orbax checkpoint will be stored; it can be Googld Cloud Storage (GCS), Hugging Face Hub or local. If not set, the default output directory is `Maxtext/tmp`.


## Verifying conversion correctness

To ensure the conversion was successful, you can use the `tests/forward_pass_logit_checker.py` script. It runs a forward pass on both the original and converted models and compares the output logits to verify conversion. It is used to verify the bidirectional conversion. 

### Usage

```bash
python3 -m tests.forward_pass_logit_checker src/MaxText/configs/base.yml \
    tokenizer_path=assets/<tokenizer> \
    load_parameters_path=<path-to-maxtext-checkpoint> \
    model_name=<MODEL_NAME> \
    scan_layers=false \
    max_prefill_predict_length=4 \
     max_target_length=8 \
    use_multimodal=false \
    --run_hf_model=True \
    --hf_model_path=<path-to-HF-checkpoint> \
    --max_kl_div=0.015 \
```

**Key arguments:**

  * `load_parameters_path`: The path to the source MaxText Orbax checkpoint (e.g., `gs://your-bucket/maxtext-checkpoint/0/items`).
  * `model_name`: The corresponding model name in the MaxText configuration (e.g., `qwen3-4b`).
  * `scan_layers`: Indicates if the output checkpoint is scanned (scan_layers=true) or unscanned (scan_layers=false).
  * `use_multimodal`: Indicates if multimodality is used.
  * `--run_hf_model`: Indicates if loading Hugging Face model from the hf_model_path. If not set, it will compare the maxtext logits with pre-saved golden logits. 
  * `--hf_model_path`: The path to the Hugging Face checkpoint.
  * `--max_kl_div`: Max KL divergence tolerance during comparisons.

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
-----

## Adding support for new models
To extend conversion support to a new model architecture, you must define its specific parameter and configuration mappings. The conversion logic is decoupled, so you only need to modify the mapping files.

1.  **Add parameter mappings**: 
- In [`utils/param_mapping.py`](./utils/param_mapping.py), add the parameter name mappings(`def {MODEL}_MAXTEXT_TO_HF_PARAM_MAPPING`). This is the 1-to-1 mappings of parameters names per layer. 
- In [`utils/param_mapping.py`](./utils/param_mapping.py), add the `hook_fn` logic (`def {MODEL}_MAXTEXT_TO_HF_PARAM_HOOK_FN`). This is the transformation needed per layer. 
2.  **Add Hugging Face weights Shape**: In [`utils/hf_shape.py`](./utils/hf_shape.py), define the tensor shape of Hugging Face format (`def {MODEL}_HF_WEIGHTS_TO_SHAPE`). This is used to ensure the tensor shape is matched after to_huggingface conversion. 
3.  **Register model key**: In [`utils/utils.py`](./utils/utils.py), add the new model key in `HF_IDS`.
4.  **Add transformer config**: In [`utils/hf_model_configs.py`](./utils/hf_model_configs.py), add the `transformers.Config` object, describing the Hugging Face model configuration (defined in ['src/MaxText/configs/models'](../configs/models)). **Note**: This configuration must precisely match the MaxText model's architecture.

Here is an example [PR to add support for gemma3 multi-modal model](https://github.com/AI-Hypercomputer/maxtext/pull/1983)

## Debugging tips

If the converted checkpoint can not get loaded and got error like: "type <class 'jax._src.core.ShapeDtypeStruct'> is not a valid JAX type."
* **Potential Cause**: The scan_layers flag is set wrong. 

If a converted checkpoint loads without errors but produces incorrect output, consider these common issues:

  * **Symptom**: The model generates garbage or nonsensical tokens.

      * **Potential Cause**: The query/key/value (Q/K/V) or Out vectors weights were likely reshaped incorrectly during conversion.

  * **Symptom**: The model generates repetitive text sequences.

      * **Potential Cause**: The layer normalization parameters may have been converted incorrectly.