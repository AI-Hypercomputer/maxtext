Qwen
=====

Qwen3 is a family of open-source large language models from the Qwen team at Alibaba. This documentation covers the integration of the following Qwen Mixture-of-Experts (MoE) models into MaxText:

-   **Qwen3-30B-A3B**

-   **Qwen3-235B-A22B**

-   **Qwen3-480B-A35B**

-   **Qwen3-Omni-30B-A3B**

-   **Qwen3.5-397B-A17B**

-   **Qwen3.5-35B-A3B**

For more details on Qwen3 architecture, see the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

For more details on Qwen3-Omni architecture, see the [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765).

For more details on Qwen3.5 architecture, see the [Qwen3.5 Blog](https://qwen.ai/blog?id=qwen3.5)

For multimodal functionality (image, video, and audio input), see the [Multimodal Support guide](../../../../../docs/tutorials/posttraining/multimodal.md).

* * * * *

Checkpoint Conversion
---------------------

To get started, you first need a MaxText-compatible checkpoint.

1.  **Download the Model**: Download the official model from Hugging Face. You can use a tool like `hf_transfer` for a fast download.

    ```bash
    # Example for Qwen3.5-35B-A3B
    hf_transfer download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/qwen3.5_35b_hf_checkpoint
    ```

    ```bash
    # Example for Qwen3-235B-A22B
    hf_transfer download Qwen/Qwen3-235B-A22B-Thinking-2507 --local-dir /path/to/qwen3_hf_checkpoint
    ```

2.  **Convert the Checkpoint**: 

    **Checkpointing Util**: Supports bidirection conversion from MaxText <-> HF. For more info look at [convert_checkpoint](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/checkpointing_solutions/convert_checkpoint.md) doc.
  - Ex. Qwen3.5-35B

    ```bash
    JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
        model_name=qwen3.5-35b-a3b \
        base_output_directory=gs://your-gcs-bucket/qwen3.5_35b_maxtext_ckpt \
        hf_access_token=${HF_TOKEN} \
        scan_layers=true \ # Set to false for unscanned checkpoint
        use_multimodal=false
    ```

  - Ex. Qwen3-MoE
    
    ```bash
    JAX_PLATFORMS=cpu python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
        model_name=<qwen3-30b-a3b|qwen3-235b-a22b|qwen3-480b-a35b> \
        base_output_directory=gs://your-gcs-bucket/qwen3_maxtext_ckpt \
        hf_access_token=${HF_TOKEN} \
        scan_layers=true \ # Set to false for unscanned checkpoint
        hf_model_path=${HF_MODEL_PATH} # Optional
    ```

* * * * *

Pre-training and Fine-tuning
----------------------------

After converting the checkpoint, you can use it for fine-tuning or start a pre-training run from scratch. The command below is an example for fine-tuning on a v5p-512 slice. To pre-train, simply remove the `load_parameters_path` argument.

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml\
    base_output_directory=${BASE_OUTPUT_DIRECTORY?}\
    dataset_path=${DATASET_PATH?}\
    load_parameters_path=gs://your-gcs-bucket/qwen3_maxtext_ckpt/0/items\
    run_name=qwen3_finetuning\
    per_device_batch_size=1\
    model_name=<qwen3-30b-a3b|qwen3-235b-a22b|qwen3-480b-a35b>\
    steps=500\
    max_target_length=8192\
    ici_fsdp_parallelism=256\
    tokenizer_type=huggingface\
    tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer
```

* * * * *

Decoding
--------

To generate text with a trained model, use the `decode` command. The command below is an example for decoding on a v5p-512 slice.

```bash
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml\
    load_parameters_path=gs://your-gcs-bucket/qwen3_maxtext_ckpt/0/items\
    tokenizer_type=huggingface\
    tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer\
    prompt="Today is a beautiful day to"\
    model_name=<qwen3-30b-a3b|qwen3-235b-a22b|qwen3-480b-a35b>\
    per_device_batch_size=1\
    max_target_length=128\
    ici_fsdp_parallelism=256\

```

* * * * *

Correctness Validation
----------------------

To verify that the MaxText implementation is numerically equivalent to the original Hugging Face model, you can run the end-to-end test scripts. These scripts automate the logit comparison test for each model.

Before running, you must set the `MAXTEXT_CHECKPOINT_PATH` environment variable. You can also optionally set `HF_MODEL_PATH` to point to a local copy of the Hugging Face model.

### Qwen3-30B-A3B

```bash
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-30b-a3b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-30b-a3b_hf_checkpoint

# Execute the validation script
bash tests/end_to_end/tpu/qwen/moe/qwen3-30b-a3b/1_test_qwen3_30b_a3b.sh

```

### Qwen3-235B-A22B

```bash
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-235b-a22b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-235b-a22b_hf_checkpoint

# Execute the validation script
bash tests/end_to_end/tpu/qwen/moe/qwen3-235b-a22b/1_test_qwen3_235b_a22b.sh

```

### Qwen3-480B-A35B

```bash
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-480b-a35b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-480b-a35b_hf_checkpoint

# Execute the validation script
bash tests/end_to_end/tpu/qwen/moe/qwen3-480b-a35b/1_test_qwen3_480b_a35b.sh
```

### Qwen3-Omni-30B-A3B

```bash
# 1. Export your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# 2. Set the base path for conversion and SFT outputs
export BASE_OUTPUT_PATH=gs://<YOUR-GCS-BUCKET>/qwen3-omni-30b-a3b_maxtext_ckpt

# (Optional) Set the path if you are using a local Hugging Face checkpoint instead of downloading
# export HF_MODEL_PATH=/path/to/local/qwen3-omni-30b-a3b_hf_checkpoint

# 3. Execute the conversion and multimodal decode verification
bash tests/end_to_end/tpu/qwen/moe/qwen3-omni-30b-a3b/1_test_qwen3_omni_30b_a3b.sh
```

### Qwen3.5-35B-A3B

```bash
# 1. Export your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# 2. Set the base path for conversion outputs
export BASE_OUTPUT_PATH=gs://<YOUR-GCS-BUCKET>/qwen3.5-35b-a3b_maxtext_ckpt

# (Optional) Set the path if you are using a local Hugging Face checkpoint instead of downloading
# export HF_MODEL_PATH=/path/to/local/qwen3.5-35b-a3b_hf_checkpoint

# 3. Execute the conversion and validation script
bash tests/end_to_end/tpu/qwen/moe/qwen3.5-35b-a3b/1_test_qwen3.5_35b_a3b.sh
```

### Qwen3.5-397B-A17B

```bash
# 1. Export your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# 2. Set the base path for conversion outputs
export BASE_OUTPUT_PATH=gs://<YOUR-GCS-BUCKET>/qwen3.5-397b-a17b_maxtext_ckpt

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3.5-397b-a17b_hf_checkpoint

# 3. Execute the conversion and validation script
bash tests/end_to_end/tpu/qwen/moe/qwen3.5-397b-a17b/1_test_qwen3.5_397b_a17b.sh
```