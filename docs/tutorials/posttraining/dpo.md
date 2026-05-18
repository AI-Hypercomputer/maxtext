<!--
 Copyright 2026 Google LLC

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

# Preference Optimization (DPO & ORPO) on Single-Host TPUs

MaxText supports two primary methods for aligning models with human preferences: **Direct Preference Optimization (DPO)** and **Odds Ratio Preference Optimization (ORPO)**. Both methods avoid the complexity of traditional Reinforcement Learning from Human Feedback (RLHF) by optimizing directly on preference data.

## DPO vs. ORPO

- **Direct Preference Optimization (DPO):** Optimizes the policy by maximizing the relative log-probability of preferred responses over rejected ones. DPO requires a **reference model** (a frozen copy of the base model) to regularize the training and ensure the policy does not drift too far from the original model's distribution.
- **Odds Ratio Preference Optimization (ORPO):** A newer, reference-free alignment method that integrates the preference loss directly into the supervised fine-tuning objective using an odds ratio. Because it **does not require a reference model**, ORPO is more memory-efficient and faster than DPO.

## Data Requirements

Both methods consume preference data in a **triplet format** consisting of a Prompt, a Chosen response, and a Rejected response. MaxText supports two ways to provide this data via the `train_data_columns` configuration:

1. **Explicit Triplets (3 Columns):** The dataset provides three distinct columns for the prompt, chosen response, and rejected response.
2. **Shared Prefix (2 Columns):** For datasets like `Anthropic/hh-rlhf`, where the prompt is embedded at the beginning of the responses, you can provide just two columns (e.g., `chosen` and `rejected`). MaxText will automatically extract the shared common prefix as the **Prompt** and treat the differing suffixes as the responses.

During the input pipeline, prompts are left-padded and responses are right-padded to maintain optimal context for the model.

## Prerequisites

For instructions on installing MaxText with post-training dependencies on your VM, please refer to the [official documentation](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) and use the `maxtext[tpu-post-train]` installation path.

## Local run on a single-host TPU VM

### Setup environment variables

Login to Hugging Face:

```bash
hf auth login
```

Set up your training environment:

```bash
# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
export MODEL=<MaxText Model>  # e.g., "qwen3-0.6b"

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
export BASE_OUTPUT_DIRECTORY=<gcs bucket path> # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
export RUN_NAME=<Name for this run>

export STEPS=<number of DPO steps to run> # e.g., 1000
export PER_DEVICE_BATCH_SIZE=<batch size per device> # e.g., 1

export ALGORITHM=<"dpo" or "orpo">  # Set to either "orpo" or "dpo"

# -- Dataset configuration --
export DATASET_NAME=<Hugging Face dataset name> # e.g., "argilla/distilabel-intel-orca-dpo-pairs"
export TRAIN_SPLIT=<data split for train> # e.g., train

# Map your dataset columns to [Prompt, Chosen, Rejected]
# For 3-column datasets:
export TRAIN_DATA_COLUMNS="['input', 'chosen', 'rejected']"

# For 2-column datasets (Prefix Extraction):
# export TRAIN_DATA_COLUMNS="['chosen', 'rejected']"
```

## Running DPO Training

You can run the DPO training using the specialized post-training script:

```{note}
The script below uses `eval_interval=0` because the default "argilla/distilabel-intel-orca-dpo-pairs" dataset only has a "train" split.
To use the same split for eval you can set a non-zero value and add `hf_eval_split=train`.
```

```bash
python3 -m maxtext.trainers.post_train.dpo.train_dpo \
    run_name=${RUN_NAME?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    dataset_type=hf \
    hf_path=${DATASET_NAME?} \
    train_split=${TRAIN_SPLIT?} \
    train_data_columns="${TRAIN_DATA_COLUMNS?}" \
    steps=${STEPS?} \
    eval_interval=0 \
    per_device_batch_size=1 \
    max_target_length=1024 \
    use_dpo=1 \
    dpo.algo=${ALGORITHM?}
```
