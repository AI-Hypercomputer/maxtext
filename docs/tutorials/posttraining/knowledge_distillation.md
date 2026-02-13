<!--
 Copyright 2024 Google LLC

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

# Knowledge distillation

## Overview

Knowledge Distillation is a compression technique that transfers knowledge from a larger (teacher) model to a smaller (student) model. This allows the smaller model to achieve performance levels closer to the larger one, but with significantly fewer parameters and computational resources.

This tutorial focuses on **response-based knowledge distillation**, a technique where the student model is trained to replicate the outputs and behaviors of the teacher model. Within response-based knowledge distillation, two primary methods are often employed:

1. **Offline Distillation (Dataset Generation):**

   - The pre-trained teacher model (running in vLLM) generates a new dataset of input-output pairs.
   - The student model is then trained on this teacher-generated dataset using standard fine-tuning techniques in MaxText.

2. **Online Distillation (Logit Matching):**

   - During the training process, both the teacher model (which is typically frozen) and the student model process the same input data simultaneously.
   - The student model is trained by minimizing a loss function that encourages its output logits to match the logits produced by the teacher model for the same inputs.

## Running Offline Distillation with MaxText

The following recipe demonstrates the process of offline distillation using **Qwen/Qwen3-32B** as the teacher model and **Llama-3.1-8B** as the student model. Since this recipe fine-tunes the student model using Supervised Fine-Tuning (SFT), it's crucial to use the conversational variant for both the teacher and student models. Here's a step-by-step tutorial:

### Prerequisites

#### a. Setup environment variables

```bash
export HF_TOKEN=<your-hf-token> # e.g., hf_BA6...
export RUN_NAME=<your-run-name> # e.g., distill-20260115
```

#### b. Install dependencies

To install MaxText and its dependencies for post-training (including vLLM for the teacher), run the following:

1. Follow the [MaxText installation instructions](https://maxtext.readthedocs.io/en/latest/install_maxtext.html#install-maxtext).

2. Install the additional dependencies for post-training:

```bash
bash tools/setup/setup_post_training_requirements.sh
```

#### c. Setup storage with Hyperdisk

To store large models and datasets, attach a Hyperdisk to your TPU VM. Refer to the [Google Cloud Hyperdisk documentation](https://cloud.google.com/compute/docs/disks/add-hyperdisk) and [TPU VM management](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm) for detailed instructions.

First, create a Hyperdisk:

```bash
export ZONE=<your-tpu-zone>  # e.g., us-central1-a
export TPU_VM_NAME=<your-tpu-vm-name>
export DISK_NAME=<your-disk-name>  # e.g., my-hyperdisk
export DISK_SIZE=<disk-size>  # e.g., 500GB

gcloud compute disks create ${DISK_NAME} \
  --size=${DISK_SIZE} \
  --type=hyperdisk-balanced \
  --zone=${ZONE}
```

Then, attach the disk to your TPU VM:

```bash
gcloud compute instances attach-disk ${TPU_VM_NAME} \
  --disk=${DISK_NAME} \
  --zone=${ZONE}
```

Inside the TPU VM, format and mount the disk (if not already mounted):

```bash
# Assuming the disk is /dev/sdb, check with lsblk
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/hyperdisk
sudo mount /dev/sdb /mnt/hyperdisk
```

Update the BASE_DIRECTORY to point to the mounted disk and create the directory:

```bash
export BASE_NAME=<your-base-directory>  # e.g., knowledge-distillation
export BASE_DIRECTORY=/mnt/hyperdisk/${BASE_NAME}
mkdir -p ${BASE_DIRECTORY}
```

> **Note:** This tutorial uses a mounted Hyperdisk for performance and reproducibility, because writing large model files and many small I/O operations directly to `gs://` can be significantly slower.

### Obtain and prepare the teacher model

For the teacher model, we will use **vLLM** to run inference. vLLM can load Hugging Face checkpoints directly, so **no conversion to MaxText format is needed** for the teacher. Ensure the teacher model is supported on TPU vLLM (refer to the [vLLM TPU recommended models](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/#text-only-models) for the latest list).

You can simply download the model from Hugging Face to your local directory:

```bash
huggingface-cli login --token $HF_TOKEN
huggingface-cli download Qwen/Qwen3-32B --repo-type model --local-dir ${BASE_DIRECTORY}/qwen3-32b
```

### Obtain and prepare the student model

The student model will be trained in MaxText, which uses the Orbax checkpointing format. You will download the Hugging Face weights to your mounted bucket and convert them for training.

#### Convert checkpoint to MaxText format

The following command downloads the Hugging Face weights and converts them to the MaxText format.

**Note:** This conversion script requires PyTorch.

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
# Set the checkpoint directory
export PRE_TRAINED_MODEL_CKPT_DIRECTORY=${BASE_DIRECTORY}/llama3.1-8b-ckpt

# Convert to MaxText format
python3 -m MaxText.utils.ckpt_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=llama3.1-8b \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${PRE_TRAINED_MODEL_CKPT_DIRECTORY} \
    scan_layers=True skip_jax_distributed_system=True
```

### Generate dataset using vLLM (Teacher Step)

Use the provided script `generate_distillation_data_vllm.py` to generate the dataset from the teacher model. This script writes a Parquet dataset compatible with MaxText SFT.

Run the generation script:

```bash
export OUTPUT_DATASET=${BASE_DIRECTORY}/datasets/distillation_data.parquet

python3 -m tools.data_generation.generate_distillation_data_vllm \
  --dataset-path HuggingFaceH4/ultrachat_200k \
  --data-split train_sft \
  --data-columns messages \
  --hf-access-token $HF_TOKEN \
  --teacher-model ${BASE_DIRECTORY}/qwen3-32b \
  --use-chat-template \
  --num-prompts 5120 \
  --num-generations 2 \
  --output-file ${OUTPUT_DATASET}

```

### Fine-tune the student model using Supervised Fine Tuning (SFT)

You can now fine-tune your smaller student model using supervised fine-tuning technique in MaxText.

#### Fine-tune the student model using the generated dataset

Example command to run fine-tuning on a TPU v6e-8:

```bash
python3 -m MaxText.sft_trainer src/maxtext/configs/post_train/sft.yml \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_DIRECTORY}/distillation/qwen3-32b-distill-llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path=parquet \
  hf_train_files=${OUTPUT_DATASET} \
  train_split='train' \
  train_data_columns=['messages'] \
  load_parameters_path=${PRE_TRAINED_MODEL_CKPT_DIRECTORY}/0/items \
  model_name=llama3.1-8b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN \
  profiler=xplane
```

#### **[OPTIONAL]** Fine-tune the student model using the original dataset

The checkpoint from the student model's fine-tuning (on the teacher-generated dataset) can be used for a subsequent fine-tuning stage. In this step, the student model is fine-tuned on the original dataset that was initially provided to the teacher model for generating the dataset.

```bash
# Get the latest checkpoint for fine-tuned student model
CHECKPOINTS_PATH=${BASE_DIRECTORY}/distillation/qwen3-32b-distill-llama3.1-8b/${RUN_NAME}/checkpoints
checkpoints=$(ls $CHECKPOINTS_PATH)
integer_dirs=()
for dir in $checkpoints; do
  dir_name=$(basename "$dir")
  if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
    integer_dirs+=("$dir_name")
  fi
done
sorted_dirs=($(printf '%s\n' "${integer_dirs[@]}" | sort -n))
largest_dir="${sorted_dirs[-1]}"
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH}/${largest_dir}/model_params

# Fine-tune student model on original dataset
python3 -m MaxText.sft.sft_trainer src/maxtext/configs/post_train/sft.yml \
  run_name=${RUN_NAME}_stage2 \
  base_output_directory=${BASE_DIRECTORY}/distillation/qwen3-32b-distill-llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path='HuggingFaceH4/ultrachat_200k' \
  train_split='train_sft' \
  train_data_columns=['messages'] \
  load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH} \
  model_name=llama3.1-8b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN \
  profiler=xplane
```
