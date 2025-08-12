Qwen3
=====

Qwen3 is a family of powerful, open-source large language models from the Qwen team at Alibaba Cloud. This documentation covers the integration of the **Qwen3-235B-A22B** Mixture-of-Experts (MoE) model into MaxText. For more details on the architecture, see the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

* * * * *

Checkpoint Conversion
---------------------

To get started, you first need a MaxText-compatible checkpoint.

1.  **Download the Model**: Download the official model from Hugging Face: [Qwen/Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507). You can use a tool like `hf_transfer` for a fast download.


    ```
    hf_transfer download Qwen/Qwen3-235B-A22B-Thinking-2507 --local-dir /path/to/qwen3_hf_checkpoint

    ```

2.  **Convert the Checkpoint**: Run the `convert_qwen3_moe.py` script to convert the downloaded Hugging Face weights into the scanned Orbax format required by MaxText.

    ```
    python3 -m MaxText.convert_qwen3_moe\
      --base_model_path /path/to/qwen3_hf_checkpoint\
      --maxtext_model_path gs://your-gcs-bucket/qwen3_maxtext_ckpt\
      --model_size qwen3-235b-a22b

    ```

* * * * *

Pre-training and Fine-tuning
----------------------------

After converting the checkpoint, you can use it for fine-tuning or start a pre-training run from scratch. The command below is an example for fine-tuning on a v5p-256 slice. To pre-train, simply remove the `load_parameters_path` argument.

```
python3 -m MaxText.train MaxText/configs/base.yml\
    base_output_directory=${BASE_OUTPUT_DIRECTORY}\
    dataset_path=${DATASET_PATH}\
    load_parameters_path=gs://your-gcs-bucket/qwen3_maxtext_ckpt/0/items\
    run_name=qwen3_finetuning\
    per_device_batch_size=4\
    model_name=qwen3-235b-a22b\
    steps=500\
    max_target_length=2048\
    ici_fsdp_parallelism=128\
    attention=flash\
    tokenizer_path=assets/qwen3-tokenizer\
    # Qwen3-specific flags
    decoder_block="qwen3_moe"\
    use_qk_norm=True\
    norm_topk_prob=True

```

* * * * *

Decoding
--------

To generate text with a trained model, use the `decode` command.

```
python3 -m MaxText.decode MaxText/configs/base.yml\
    load_parameters_path=gs://your-gcs-bucket/qwen3_maxtext_ckpt/0/items\
    tokenizer_path=assets/qwen3-tokenizer\
    prompt="Today is a beautiful day to"\
    model_name=qwen3-235b-a22b\
    per_device_batch_size=1\
    max_target_length=128

```

* * * * *

Correctness Validation
----------------------

To verify that the MaxText implementation is numerically equivalent to the original Hugging Face model, run the end-to-end test script. This script automates the logit comparison test.

Before running, you must set the `MAXTEXT_CHECKPOINT_PATH` environment variable. You can also optionally set `HF_MODEL_PATH` to point to a local copy of the Hugging Face model.

```
# Set the required path to your converted MaxText checkpoint
export MAXTEXT_CHECKPOINT_PATH=gs://your-gcs-bucket/qwen3_maxtext_ckpt/0/items/

# (Optional) Set the path to your local Hugging Face checkpoint
# export HF_MODEL_PATH=/path/to/local/qwen3_hf_checkpoint

# Execute the validation script
bash MaxText/end_to_end/tpu/qwen/moe/qwen3-235b-a22b/1_test_qwen3_235b_a22b.sh

```