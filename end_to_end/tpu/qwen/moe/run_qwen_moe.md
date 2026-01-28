Qwen3
=====

Qwen3 is a family of open-source large language models from the Qwen team at Alibaba. This documentation covers the integration of the following Qwen3 Mixture-of-Experts (MoE) models into MaxText:

-   **Qwen3-30B-A3B**

-   **Qwen3-235B-A22B**

-   **Qwen3-480B-A35B**

For more details on the architecture, see the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

* * * * *

Checkpoint Conversion
---------------------

To get started, you first need a MaxText-compatible checkpoint.

1.  **Download the Model**: Download the official model from Hugging Face. You can use a tool like `hf_transfer` for a fast download.

    ```
    # Example for Qwen3-235B-A22B
    hf_transfer download Qwen/Qwen3-235B-A22B-Thinking-2507 --local-dir /path/to/qwen3_hf_checkpoint

    ```

2.  **Convert the Checkpoint**: Run the `convert_qwen3_moe.py` script to convert the downloaded Hugging Face weights into the Orbax format required by MaxText.

    ```
    python3 -m MaxText.utils.ckpt_scripts.convert_qwen3_moe\
      --base_model_path /path/to/qwen3_hf_checkpoint\
      --maxtext_model_path gs://your-gcs-bucket/qwen3_maxtext_ckpt\
      --model_size <qwen3-30b-a3b|qwen3-235b-a22b|qwen3-480b-a35b>

    ```

* * * * *

Pre-training and Fine-tuning
----------------------------

After converting the checkpoint, you can use it for fine-tuning or start a pre-training run from scratch. The command below is an example for fine-tuning on a v5p-512 slice. To pre-train, simply remove the `load_parameters_path` argument.

```
python3 -m MaxText.train src/MaxText/configs/base.yml\
    base_output_directory=${BASE_OUTPUT_DIRECTORY}\
    dataset_path=${DATASET_PATH}\
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

```
python3 -m MaxText.decode src/MaxText/configs/base.yml\
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

Bash

```
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-30b-a3b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-30b-a3b_hf_checkpoint

# Execute the validation script
bash end_to_end/tpu/qwen/moe/qwen3-30b-a3b/1_test_qwen3_30b_a3b.sh

```

### Qwen3-235B-A22B

Bash

```
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-235b-a22b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-235b-a22b_hf_checkpoint

# Execute the validation script
bash end_to_end/tpu/qwen/moe/qwen3-235b-a22b/1_test_qwen3_235b_a22b.sh

```

### Qwen3-480B-A35B

Bash

```
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3-480b-a35b_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3-480b-a35b_hf_checkpoint

# Execute the validation script
bash src/MaxText/end_to_end/tpu/qwen/moe/qwen3-480b-a35b/1_test_qwen3_480b_a35b.sh
```