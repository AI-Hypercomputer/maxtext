Qwen3 Next
=========

Qwen3-Next is Alibaba 80B Mixture-of-Experts (MoE) model (activating only 3B parameters) that features a novel **hybrid attention** architecture combining Gated DeltaNet (linear attention) and Gated Attention (full attention) for massive context scaling. This documentation covers the integration of **Qwen3-Next-80B-A3B** into MaxText:

For more details on the architecture, see the [Qwen3 Technical Blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list).

* * * * *

Checkpoint Conversion
---------------------

To get started, you first need a MaxText-compatible checkpoint.

1.  **Download the Model**: Download the official model from Hugging Face. You can use a tool like `hf_transfer` for a fast download.

    ```
    # Example for Qwen3-Next-80B-A3B-Instruct
    hf_transfer download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir /path/to/qwen3_next_hf_checkpoint
    ```

2.  **Convert the Checkpoint**: Run the `convert_qwen3_next_scanned.py` script to convert the downloaded Hugging Face weights into the Orbax format required by MaxText.

    ```
    python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_qwen3_next_scanned \
      --base_model_path /path/to/qwen3_next_hf_checkpoint \
      --maxtext_model_path gs://your-gcs-bucket/qwen3_next_maxtext_ckpt \
      --model_size qwen3-next-80b-a3b
    ```

* * * * *

Pre-training and Fine-tuning
----------------------------

After converting the checkpoint, you can use it for fine-tuning or start a pre-training run from scratch. The command below is an example for fine-tuning on a v5p-512 slice. To pre-train, simply remove the `load_parameters_path` argument.

```
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    dataset_path=${DATASET_PATH} \
    load_parameters_path=gs://your-gcs-bucket/qwen3_next_maxtext_ckpt/0/items \
    run_name=qwen3_next_finetuning \
    per_device_batch_size=1 \
    model_name=qwen3-next-80b-a3b \
    steps=500 \
    max_target_length=8192 \
    ici_fsdp_parallelism=256 \
    tokenizer_type=huggingface \
    tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer

```

* * * * *

Correctness Validation
----------------------

To verify that the MaxText implementation is numerically equivalent to the original Hugging Face model, you can run the end-to-end test scripts. These scripts automate the logit comparison test for each model.

Before running, you must set the `MAXTEXT_CHECKPOINT_PATH` environment variable. You can also optionally set `HF_MODEL_PATH` to point to a local copy of the Hugging Face model.

### Qwen3-Next-80B-A3B

Bash

```
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-next-80b-a3b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-next-80b-a3b_hf_checkpoint

# Execute the validation script
bash tests/end_to_end/tpu/qwen/next/qwen3-next-80b-a3b/1_test_qwen3_next_80b_a3b.sh

```

## Supported MoE Strategies

This model implementation supports both **Token Dropping** and **Dropless** strategies for Mixture of Experts routing. Take a look at the MaxText [documentation](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md) on MoE configs and flags to set based on desired strategy.

