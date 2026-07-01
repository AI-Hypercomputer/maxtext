<!--
 Copyright 2023–2025 Google LLC

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

# SFT on single-host TPUs

Supervised fine-tuning (SFT) is a process where a pre-trained large language model is fine-tuned on a labeled dataset to adapt the model to perform better on specific tasks.

This tutorial demonstrates step-by-step instructions for setting up the environment and then training the model on a Hugging Face dataset using SFT.

We use [Tunix](https://github.com/google/tunix), a JAX-based library designed for post-training tasks, to perform SFT.

In this tutorial we run SFT on a GKE cluster using XPK. Let's get started!

## Environment Setup

Before you begin, ensure you have set up your GKE cluster and installed XPK. Refer to the [At scale with XPK](../../run_maxtext/run_maxtext_via_xpk.md) guide for prerequisites and environment setup.

You also need to build and upload a MaxText Docker image with post-training dependencies. Refer to the [Build and upload MaxText Docker images](../build_maxtext.md) tutorial and specifically build the **TPU post-training Docker image**.

## Setup environment variables

Set up the following environment variables on your development machine to configure your training run. Replace placeholders with your actual values.

```bash
# -- GKE & Docker configuration --
export GKE_CLUSTER=<your-cluster-name>
export ZONE=<your-cluster-zone> # e.g., us-central1-a
export CLOUD_IMAGE_NAME=<your-uploaded-docker-image-path> # e.g., gcr.io/my-project/maxtext-post-train

# -- Hugging Face configuration --
# Your Hugging Face personal access token with 'read' permissions.
export HF_TOKEN=<your-huggingface-token>

# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MODEL_NAME> # e.g., 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
export BASE_OUTPUT_DIRECTORY=<GCS_BUCKET> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<RUN_NAME>

export STEPS=<STEPS> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<BATCH_SIZE_PER_DEVICE> # e.g., 1

# -- Dataset configuration --
export DATASET_NAME=<DATASET_NAME> # e.g., HuggingFaceH4/ultrachat_200k
export TRAIN_SPLIT=<TRAIN_SPLIT> # e.g., train_sft
export TRAIN_DATA_COLUMNS=<DATA_COLUMNS> # e.g., ['messages']
```

## Get your model checkpoint

This section explains how to prepare your model checkpoint for use with MaxText. You have two options: using an existing MaxText checkpoint or converting a Hugging Face checkpoint.

### Option 1: Using an existing MaxText checkpoint

If you already have a MaxText-compatible model checkpoint, simply set the following environment variable and move on to the next section.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

### Option 2: Converting a Hugging Face checkpoint

Refer the steps in [Hugging Face to MaxText](hf-to-maxtext) to convert a hugging face checkpoint to MaxText. Make sure you have correct checkpoint files converted and saved. Similar as Option 1, you can set the following environment and move on.

```sh
export MAXTEXT_CKPT_PATH=<CKPT_PATH> # e.g., gs://my-bucket/my-model-checkpoint/0/items
```

> [!IMPORTANT]
> **Matching the `scan_layers` Parameter:**
> The `scan_layers` setting during your fine-tuning run **must match** the setting used when creating the checkpoint at `MAXTEXT_CKPT_PATH`.
>
> - If the checkpoint was converted or saved with `scan_layers=False` (which is common for Hugging Face conversions and inference-ready models), you **must also provide `scan_layers=False` in the MaxText command.**
> - If `scan_layers` does not match, MaxText will raise a `ValueError`.
>   See the [Checkpoints concept guide](../../reference/core_concepts/checkpoints.md) for more details.

## Run SFT on Hugging Face Dataset

Submit your SFT training job to the GKE cluster using XPK:

```sh
xpk workload create \
    --cluster ${GKE_CLUSTER?} \
    --workload ${RUN_NAME?} \
    --docker-image ${CLOUD_IMAGE_NAME?} \
    --tpu-type v5p-8 \
    --num-slices 1 \
    --command "export HF_TOKEN=${HF_TOKEN?} && \
        python3 -m maxtext.trainers.post_train.sft.train_sft \
        run_name=${RUN_NAME?} \
        base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
        model_name=${MODEL?} \
        load_parameters_path=${MAXTEXT_CKPT_PATH?} \
        per_device_batch_size=${PER_DEVICE_BATCH_SIZE?} \
        steps=${STEPS?} \
        hf_path=${DATASET_NAME?} \
        train_split=${TRAIN_SPLIT?} \
        train_data_columns=${TRAIN_DATA_COLUMNS?} \
        profiler=xplane"
```

*Note: Adjust `--tpu-type` and `--num-slices` as appropriate for your GKE cluster.*

Your fine-tuned model checkpoints will be saved to your GCS bucket at: `$BASE_OUTPUT_DIRECTORY/$RUN_NAME/checkpoints`.

## Dataset Customization & Chat Templates

Supervised Fine-Tuning in MaxText relies on tokenizing conversational datasets using chat templates. This requires the dataset structure and templates to be aligned.

### Supported Dataset Schemas

By default, MaxText SFT expects one of three conversational dataset structures:

- `["messages"]`: A single column containing a list of dictionaries with `role` and `content` (recommended).
- `["prompt", "completion"]`: Separated prompt and completion columns.
- `["question", "answer"]`: Question and answer columns (e.g., math datasets).

During data processing, MaxText converts these into a unified `messages` schema (OpenAI-like format) before feeding it to the tokenizer:

```json
[
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]
```

### Custom Tokenizer Chat Templates

To customize the tokenizer's chat formatting (e.g., adding special tokens like `<start_of_turn>`, `<end_of_turn>`, etc.), you can provide a custom chat template using the `chat_template` or `chat_template_path` configs:

- **`chat_template`**: Use this config to specify a custom Jinja2 template string directly.
- **`chat_template_path`**: Path to a custom Jinja2 template file (e.g., `.jinja`) or a JSON file containing the template.
- **`use_chat_template=True`**: Enables chat template formatting.

### Advanced: Custom Dataset Formatter (e.g., ShareGPT)

If your dataset is in a format not natively supported—such as **ShareGPT** (which uses a `conversations` column with `from` and `value` keys)—you can write a custom Python formatting function to convert it on-the-fly.

#### 1. Write a custom formatting function

Create a Python file in your workspace (e.g., `src/maxtext/input_pipeline/custom_formatters.py`):

```python
def format_sharegpt(example):
    """Converts ShareGPT format (from/value) to standard messages (role/content)."""
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    messages = []
    for turn in example["conversations"]:
        role = role_map.get(turn["from"], "user")
        messages.append(
            {
                "role": role,
                "content": turn["value"],
            }
        )

    example["messages"] = messages
    return example
```

#### 2. Configure MaxText to use your formatter

When starting your SFT training, append the following parameters to the python command inside your XPK `--command` flag:

- `train_data_columns`: Point to the original column name in the raw dataset (`"['conversations']"`).
- `formatting_func_path`: Point to the python import path of your formatting function (`"maxtext.input_pipeline.custom_formatters.format_sharegpt"`).

```sh
xpk workload create \
    ... \
    --command "... && python3 -m maxtext.trainers.post_train.sft.train_sft \
        ... \
        train_data_columns=\"['conversations']\" \
        formatting_func_path=\"maxtext.input_pipeline.custom_formatters.format_sharegpt\""
```

### Runnable Example in the Codebase

For a complete, runnable SFT workflow that demonstrates how to configure the training loop and use a custom dataset formatter (`formatting_func_path` and `formatting_func_kwargs`), check out the [sft_qwen3_demo.ipynb](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/examples/sft_qwen3_demo.ipynb) Jupyter notebook.
