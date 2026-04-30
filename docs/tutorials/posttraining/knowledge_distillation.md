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

Knowledge Distillation transfers knowledge from a teacher model to a student model. This allows the student to achieve performance levels closer to the teacher's, typically with significantly fewer parameters and computational resources.

This tutorial focuses on **response-based knowledge distillation**, a technique where the student model is trained to replicate the outputs and behaviors of the teacher model. Within response-based knowledge distillation, two primary methods are often employed, both covered below:

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

gcloud compute disks create ${DISK_NAME?} \
  --size=${DISK_SIZE?} \
  --type=hyperdisk-balanced \
  --zone=${ZONE?}
```

Then, attach the disk to your TPU VM:

```bash
gcloud compute instances attach-disk ${TPU_VM_NAME?} \
  --disk=${DISK_NAME?} \
  --zone=${ZONE?}
```

Inside the TPU VM, format and mount the disk (if not already mounted):

```bash
# Assuming the disk is /dev/sdb, check with lsblk
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/hyperdisk
sudo mount /dev/sdb /mnt/hyperdisk
```

Update the BASE_OUTPUT_DIRECTORY to point to the mounted disk and create the directory:

```bash
export BASE_NAME=<your-base-directory>  # e.g., knowledge-distillation
export BASE_OUTPUT_DIRECTORY=/mnt/hyperdisk/${BASE_NAME?}
mkdir -p ${BASE_OUTPUT_DIRECTORY?}
```

> **Note:** This tutorial uses a mounted Hyperdisk for performance and reproducibility, because writing large model files and many small I/O operations directly to `gs://` can be significantly slower.

### Obtain and prepare the teacher model

For the teacher model, we will use **vLLM** to run inference. vLLM can load Hugging Face checkpoints directly, so **no conversion to MaxText format is needed** for the teacher. Ensure the teacher model is supported on TPU vLLM (refer to the [vLLM TPU recommended models](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features) for the latest list).

You can simply download the model from Hugging Face to your local directory:

```bash
huggingface-cli login --token ${HF_TOKEN?}
huggingface-cli download Qwen/Qwen3-32B --repo-type model --local-dir ${BASE_OUTPUT_DIRECTORY?}/qwen3-32b
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
export MAXTEXT_CKPT_PATH=${BASE_OUTPUT_DIRECTORY?}/llama3.1-8b-ckpt

# Convert to MaxText format
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=llama3.1-8b \
    hf_access_token=${HF_TOKEN?} \
    base_output_directory=${MAXTEXT_CKPT_PATH?} \
    scan_layers=True skip_jax_distributed_system=True
```

### Generate dataset using vLLM (Teacher Step)

Use the provided script `generate_distillation_data_vllm.py` to generate the dataset from the teacher model. This script writes a Parquet dataset compatible with MaxText SFT.

Run the generation script:

```bash
export OUTPUT_DATASET=${BASE_OUTPUT_DIRECTORY?}/datasets/distillation_data.parquet

python3 -m tools.data_generation.generate_distillation_data_vllm \
  --dataset-path HuggingFaceH4/ultrachat_200k \
  --data-split train_sft \
  --data-columns messages \
  --hf-access-token ${HF_TOKEN?} \
  --teacher-model ${BASE_OUTPUT_DIRECTORY?}/qwen3-32b \
  --use-chat-template \
  --num-prompts 5120 \
  --num-generations 2 \
  --output-file ${OUTPUT_DATASET?}

```

### Fine-tune the student model using Supervised Fine Tuning (SFT)

You can now fine-tune your smaller student model using supervised fine-tuning technique in MaxText.

#### Fine-tune the student model using the generated dataset

Example command to run fine-tuning on a TPU v6e-8:

```bash
python3 -m maxtext.trainers.post_train.sft.train_sft_deprecated \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?}/distillation/qwen3-32b-distill-llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path=parquet \
  hf_train_files=${OUTPUT_DATASET?} \
  train_split='train' \
  train_data_columns=['messages'] \
  load_parameters_path=${MAXTEXT_CKPT_PATH?}/0/items \
  model_name=llama3.1-8b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=${HF_TOKEN?} \
  profiler=xplane
```

#### **[OPTIONAL]** Fine-tune the student model using the original dataset

The checkpoint from the student model's fine-tuning (on the teacher-generated dataset) can be used for a subsequent fine-tuning stage. In this step, the student model is fine-tuned on the original dataset that was initially provided to the teacher model for generating the dataset.

```bash
# Get the latest checkpoint for fine-tuned student model
CHECKPOINTS_PATH=${BASE_OUTPUT_DIRECTORY?}/distillation/qwen3-32b-distill-llama3.1-8b/${RUN_NAME?}/checkpoints
checkpoints=$(ls ${CHECKPOINTS_PATH?})
integer_dirs=()
for dir in $checkpoints; do
  dir_name=$(basename "$dir")
  if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
    integer_dirs+=("$dir_name")
  fi
done
sorted_dirs=($(printf '%s\n' "${integer_dirs[@]}" | sort -n))
largest_dir="${sorted_dirs[-1]}"
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH?}/${largest_dir}/model_params

# Fine-tune student model on original dataset
python3 -m maxtext.trainers.post_train.sft.train_sft \
  run_name=${RUN_NAME?}_stage2 \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?}/distillation/qwen3-32b-distill-llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path='HuggingFaceH4/ultrachat_200k' \
  train_split='train_sft' \
  train_data_columns=['messages'] \
  load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH?} \
  model_name=llama3.1-8b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=${HF_TOKEN?} \
  profiler=xplane
```

## Running Online Distillation with MaxText

Online distillation runs the teacher and student in the same training process. Each step, both models do a forward pass on the same batch and the student is updated to match a combination of:

1. The teacher's softened logit distribution (KL-divergence soft loss).
2. The ground-truth labels (cross-entropy hard loss).
3. (Optional) The teacher's intermediate attention activations (feature loss).

The trainer entry point is [`maxtext.trainers.post_train.distillation.train_distill`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/train_distill.py), built on [Tunix](https://github.com/google/tunix). For a deeper reference on how the trainer is structured, the loss anatomy, and how to tune the α / β / temperature schedules for different scenarios, see the dedicated [Distillation guide](../../guides/distillation.md).

### Prerequisites

#### a. Convert both teacher and student to MaxText format

Online distillation runs the teacher inside MaxText (not vLLM), so both checkpoints must be in MaxText/Orbax format. Convert them with the same script used for the student in the offline section:

```bash
# Student
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=llama3.1-8b \
    hf_access_token=${HF_TOKEN?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?}/llama3.1-8b-ckpt \
    scan_layers=True skip_jax_distributed_system=True

# Teacher (example: same family, larger)
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    model_name=llama3.1-70b \
    hf_access_token=${HF_TOKEN?} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?}/llama3.1-70b-ckpt \
    scan_layers=True skip_jax_distributed_system=True
```

> **Note:** Student and teacher must share the same vocabulary. The trainer asserts `student_config.vocab_size == teacher_config.vocab_size` at startup.

#### b. Install Tunix

The online distillation trainer depends on Tunix. The XPK launcher script ([`scripts/run_distill_xpk.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh)) contains a `prep_image` step that layers Tunix on top of the MaxText base image. For local runs, install the same pin used by the launcher — the default `TUNIX_SOURCE` in `run_distill_xpk.sh` is the source of truth. As of this writing:

```bash
pip install "git+https://github.com/google/tunix@110932a8395086511228483312131841521695c1"
```

> **Note:** The commit pin above will drift as the launcher is updated. Before installing, check the `TUNIX_SOURCE` default in [`run_distill_xpk.sh`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh) and use that spec. Once a Tunix PyPI release ships, this will become a versioned `google-tunix==<ver>` install.

### Configuration

The starter config is [`src/maxtext/configs/post_train/distillation.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/post_train/distillation.yml). The trainer initializes **two MaxText models** with isolated configurations applied via `student_overrides` and `teacher_overrides`. CLI overrides only apply to the student by default — the teacher is initialized from the YAML + `teacher_overrides` only.

Key knobs (see the [Distillation guide](../../guides/distillation.md) for the full configuration surface and tuning advice):

```yaml
distill_alpha: 0.5             # weight on KL(teacher||student)
distill_temperature: 1.0
distill_beta: 0.0              # >0 enables feature distillation; requires scan_layers=True, enable_nnx=True
distill_layer_indices: None
```

The student and teacher are configured separately via `student_overrides` and `teacher_overrides`. Two patterns cover most use cases:

#### Pattern A — Compression (large teacher, smaller student)

The headline use case: distill a larger teacher into a smaller student that shares its tokenizer. The trainer asserts `student_config.vocab_size == teacher_config.vocab_size` at startup, so the simplest path is to stay within a single family (Llama-3.1-70B → Llama-3.1-8B, Qwen → Qwen, Gemma → Gemma) where the vocabulary is guaranteed to match.

```yaml
student_overrides:
  model_name: "llama3.1-8b"
  load_parameters_path: "/path/to/llama3.1-8b-ckpt/0/items"

teacher_overrides:
  model_name: "llama3.1-70b"
  load_parameters_path: "/path/to/llama3.1-70b-ckpt/0/items"
```

#### Pattern B — Pruning recovery (same model name, student is structurally smaller)

What `train_distill.py` was originally built for: recover quality after structural pruning by aligning the pruned student to the unpruned teacher. The student overrides architectural keys (e.g. `base_num_decoder_layers`) to match its pruned shape.

```yaml
student_overrides:
  model_name: "llama3.1-8b"
  base_num_decoder_layers: 24    # 8b has 32 layers; this student is pruned down to 24
  load_parameters_path: "/path/to/pruned-llama3.1-8b/0/items"

teacher_overrides:
  model_name: "llama3.1-8b"
  load_parameters_path: "/path/to/full-llama3.1-8b/0/items"
```

> **Note:** Producing the pruned checkpoint is out of scope. This trainer recovers quality from any pruned student you bring; the pruning step itself lives in your own pipeline.

### Launching online distillation

#### Single-host (TPU v6e-8)

The example below demonstrates **Pattern B** (pruning recovery): the student is a depth-pruned Llama-3.1-8B (24 of 32 layers) being aligned to the unpruned 32-layer teacher, with feature loss enabled. For **Pattern A** (compression) substitute the `teacher_overrides` with a larger model and set `distill_beta=0` — see the note on layer indices below.

```bash
python3 -m maxtext.trainers.post_train.distillation.train_distill \
  src/maxtext/configs/post_train/distillation.yml \
  run_name=${RUN_NAME?} \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?}/distillation/online \
  tokenizer_path=meta-llama/Llama-3.1-8B tokenizer_type=huggingface \
  hf_access_token=${HF_TOKEN?} \
  student_overrides.model_name=llama3.1-8b \
  student_overrides.base_num_decoder_layers=24 \
  student_overrides.load_parameters_path=${BASE_OUTPUT_DIRECTORY?}/pruned-llama3.1-8b-24L/0/items \
  teacher_overrides.model_name=llama3.1-8b \
  teacher_overrides.load_parameters_path=${BASE_OUTPUT_DIRECTORY?}/llama3.1-8b-ckpt/0/items \
  per_device_batch_size=2 \
  gradient_accumulation_steps=8 \
  ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  steps=10000 \
  distill_alpha=0.9 distill_alpha_end=0.5 distill_alpha_schedule=cosine \
  distill_temperature=2.0 \
  distill_beta=1.0 distill_beta_end=0.1 distill_beta_schedule=cosine \
  distill_layer_indices=[2,5,8,11,14,17,20,23] \
  scan_layers=True enable_nnx=True \
  profiler=xplane
```

The schedule values above are a strong default for same-size pruning recovery. See [α and β schedule guide](../../guides/distillation.md#alpha-schedule-guide) for other scenarios (large teacher → small student, logit-only, aggressive recovery, etc.).

> **Note:** `distill_layer_indices` is applied to **both** student and teacher activations identically. When the two have different depths (Pattern A or a depth-pruned Pattern B), every index must be valid on the *smaller* side, and same-numbered layers are aligned across the two models. The trainer cannot map student layer *i* to teacher layer *f(i)* for arbitrary *f*. If the depths differ significantly, prefer logit-only distillation (`distill_beta=0`).

#### Multi-host on GKE via XPK

A reference launcher is provided at `src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh`. It handles image preparation (`prep_image` layers Tunix on top of the MaxText base image), workload submission, log streaming, and an auto-resume loop for long-running jobs.

Minimum environment variables:

```bash
export XPK_CLUSTER=<your-gke-cluster>
export XPK_PROJECT=<your-gcp-project>
export XPK_ZONE=<cluster-zone>             # e.g. us-central1-a
export XPK_DEVICE_TYPE=<tpu-type>          # e.g. tpu7x-4x4x4, v5p-128
export XPK_BASE_OUTPUT_DIR=gs://<bucket>/distill-runs

# Distillation hyperparameters (always passed; override yml values)
export DISTILL_ALPHA=0.9
export DISTILL_TEMPERATURE=2.0
export DISTILL_BETA=1.0
# Layer indices for feature loss. Every index must be valid on the smaller side
# (student for Pattern A, both for Pattern B). Values below assume a 32-layer
# student; adjust for other depths — see the Distillation guide's layer-index table.
export DISTILL_LAYER_INDICES=[3,7,11,15,19,23,27,31]   # no spaces inside brackets
```

Then:

```bash
# One-time: layer Tunix on top of the MaxText base image
bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh prep_image

# Submit a workload
bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh submit

# Stream logs
bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh monitor

# Auto-resume on failure (uses the same workload + base output dir, so checkpoint resume works)
bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh resume_until_done
```

The script's header comment lists every supported environment variable.

### Offline top-k logits variant

If running the teacher every step is too expensive (for very large teachers), you can cache its top-k logits once and stream them to the trainer. **Feature loss is unavailable in this mode** — only `distill_alpha` / `distill_temperature` are active.

1. Save top-k teacher logits to ArrayRecord files:

   ```bash
   python3 src/maxtext/trainers/post_train/distillation/save_top_k_teacher_logits.py \
     src/maxtext/configs/post_train/distillation.yml \
     --local_tmp_dir=/tmp/teacher_topk \
     --steps_per_file=10 \
     --top_k=64
   ```

2. Run the trainer with `offline_data_dir` pointing at the cached files:

   ```bash
   python3 -m maxtext.trainers.post_train.distillation.train_distill \
     src/maxtext/configs/post_train/distillation.yml \
     offline_data_dir=/tmp/teacher_topk \
     ...other-args...
   ```

Bias α slightly lower (e.g. `0.7 → 0.3`) since the reconstructed distribution has narrow, hard-cut support — see the schedule guide for details.

### Next steps

- [Distillation guide](../../guides/distillation.md) — loss anatomy, α / β / temperature schedule tuning, layer-index selection, monitoring metrics, troubleshooting, and ablation priority.
