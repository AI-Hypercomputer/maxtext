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

# Knowledge Distillation

## Overview
Knowledge Distillation is a compression technique that transfers knowledge from a larger (teacher) model to a smaller (student) model. This allows the smaller model to achieve performance levels closer to the larger one, but with significantly fewer parameters and computational resources.

This guide focuses on **response-based knowledge distillation**, a technique where the student model is trained to replicate the outputs and behaviors of the teacher model. Within response-based knowledge distillation, two primary methods are often employed:

1.  **Offline Distillation (Dataset Generation):**
    *   The pre-trained teacher model first generates a new dataset of input-output pairs.
    *   The student model is then trained on this teacher-generated dataset using standard fine-tuning techniques.

2.  **Online Distillation (Logit Matching):**
    *   During the training process, both the teacher model (which is typically frozen) and the student model process the same input data simultaneously.
    *   The student model is trained by minimizing a loss function that encourages its output logits to match the logits produced by the teacher model for the same inputs.

## Running Offline Distillation with MaxText

The following recipe demonstrates the process of offline distillation using **Deepseek2-16b** as the teacher model and **Llama2-7b** as the student model. Since this recipe fine-tunes the student model using Supervised Fine-Tuning (SFT), it's crucial to use the conversational variant for both the teacher and student models. Here’s a step-by-step guide:

### Prerequisites

#### a. Setup environment variables

```bash
export HF_TOKEN = <Hugging Face access token>
export BASE_DIRECTORY = <Directory to store distillation results>
export HF_REPO_NAME = <Hugging Face repository name to store teacher-generated dataset>
export USERNAME_OR_ORG = <Owner of Hugging Face repository>
export RUN_NAME = <unique name for the run>
```

#### b. Install dependencies

```sh
git clone https://github.com/AI-Hypercomputer/maxtext.git
python3 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate
python3 -m pip install uv
cd maxtext
uv pip install -r dependencies/requirements/requirements.txt
```

### 1. Obtain and prepare the teacher model

#### a. Download model from Hugging Face

```bash
huggingface-cli login  # Provide your Hugging Face token
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat --repo-type model --local-dir ~/deepseek2-16b-chat
```

#### b. Convert checkpoint to MaxText format
MaxText requires checkpoints to be in a specific format. You'll need to convert the downloaded Hugging Face checkpoints to a MaxText-compatible checkpoint.

```bash
# Get unscanned checkpoint for efficient decoding
JAX_PLATFORMS=cpu \
python3 -m MaxText.utils.ckpt_scripts.convert_deepseek_family_unscanned_ckpt \
  --base_model_path ~/deepseek2-16b-chat \
  --maxtext_model_path ${BASE_DIRECTORY}/deepseek2-16-chat/unscanned \
  --model_size deepseek2-16b
```

### 2. Obtain and prepare the student model

#### a. Download model from Hugging Face

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --repo-type model --local-dir ~/llama2-7b-chat
```

#### b. Convert checkpoint to MaxText format
MaxText requires checkpoints to be in a specific format. You'll need to convert the downloaded Hugging Face checkpoints to a MaxText-compatible checkpoint.

```bash
# Get scanned checkpoint for fine-tuning
JAX_PLATFORMS=cpu \
python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt \
  --base-model-path ~/llama2-7b-chat \
  --maxtext-model-path ${BASE_DIRECTORY}/llama2-7b-chat/scanned \
  --model-size llama2-7b
```

### 3. Generate dataset using the teacher model
Once the teacher model's checkpoint is in the MaxText format, you can run inference to generate the dataset that will be used to fine-tune the student model.

### 3.a. Run the JetStream server

Example command to run JetStream server on `v4-8`:

```bash
python3 -m MaxText.maxengine_server src/MaxText/configs/base.yml \
  tokenizer_path=deepseek-ai/DeepSeek-V2-Lite-chat tokenizer_type=huggingface \
  load_parameters_path=${BASE_DIRECTORY}/deepseek2-16-chat/unscanned/0/items \
  model_name=deepseek2-16b \
  per_device_batch_size=10 ici_tensor_parallelism=4 \
  max_target_length=2048 max_prefill_predict_length=64 \
  hf_access_token=$HF_TOKEN \
  scan_layers=False \
  multi_sampling=True decode_sampling_strategy=weighted
```

Set `multi_sampling` to `True` to generate multiple independent completions per prompt.


### 3.b. Generate dataset using JetStream server
In a new tab in your terminal, run the following command to generate dataset from teacher model. Note that this is an example command to run on `v4-8`:

```bash
python3 -m MaxText.generate_distillation_data \
  --tokenizer-path deepseek-ai/DeepSeek-V2-Lite-chat \
  --dataset-path HuggingFaceH4/ultrachat_200k --data-split train_sft \
  --data-columns messages \
  --max-prefill-length 64 --max-target-length 2048 \
  --hf-access-token $HF_TOKEN \
  --use-chat-template --remove-local-dataset-files \
  --num-generations 2 --batch-size 1024 --num-batches 200 \
  upload-to-hf --hf-repo-id ${HF_REPO_NAME}
```

When `multi_sampling=True` (Step 3.a), the `--num-generations` parameter specifies the number of distinct completions to generate per prompt. The `--batch-size` parameter controls how many prompts are processed per batch, and `--num-batches` defines how many such batches to run. The total number of prompt-completion pairs generated is approximately `num_batches * batch_size * num_generations`.

For example, with `--batch-size 1024`, `--num-generations 2`, and `--num-batches 200`, this would yield `200 * 1024 * 2 = 409,600` prompt-completion pairs.

It's important to note that some prompts may be filtered out by pre-processing logic before inference. If the prompt sequences are longer than `max-prefill-length`, then those prompts will be filtered out in pre-processing stage.

Additionally, the generated dataset can be uploaded to either Hugging Face or Google Cloud Storage (GCS). To upload to Hugging Face, use the `upload-to-hf --hf-repo-id <hf_repo_name>` flags. To upload to GCS, use the `upload-to-gcs --gcs-bucket <gcs bucket name> --gcs-data-path <path in gcs bucket>` flags.

### 4. Fine-tune the student model using Supervised Fine Tuning (SFT)
You can now fine-tune your smaller student model using supervised fine-tuning technique in MaxText.

### 4.a. Fine-tune the student model using dataset generated in Step 3

Example command to run fine-tuning on v4-8:

```bash
python3 -m MaxText.sft_trainer src/MaxText/configs/sft.yml \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_DIRECTORY}/distillation/deepseek2-16b-distill-llama2-7b \
  tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
  hf_path=${USERNAME_OR_ORG}/${HF_REPO_NAME} \
  train_split='train' train_data_columns=['prompt','completion'] \
  load_parameters_path=${BASE_DIRECTORY}/llama2-7b-chat/scanned/0/items \
  model_name=llama2-7b \
  per_device_batch_size=2 ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN
```

### 4.b. **[OPTIONAL]** Fine-tune the student model using the original dataset

The checkpoint from the student model's fine-tuning (on the teacher-generated dataset) can be used for a subsequent fine-tuning stage. In this step, the student model is fine-tuned on the original dataset that was initially provided to the teacher model for generating the dataset.

```bash
# Get the latest checkpoint for fine-tuned student model
CHECKPOINTS_PATH=${BASE_DIRECTORY}/distillation/deepseek2-16b-distill-llama2-7b/${RUN_NAME}/checkpoints
checkpoints=$(gcloud storage ls $CHECKPOINTS_PATH)
integer_dirs=()
for dir in $checkpoints; do
  dir_name=$(basename "$dir")
  if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
    integer_dirs+=("$dir_name")
  fi
done
sorted_dirs=($(printf '%s\n' "${integer_dirs[@]}" | sort -n))
largest_dir="${sorted_dirs[-1]}"
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH}/${largest_dir}/items

# Fine-tune student model on original dataset
python3 -m MaxText.sft_trainer src/MaxText/configs/sft.yml \
  run_name=${RUN_NAME} \
  base_output_directory=${BASE_DIRECTORY}/distillation/deepseek2-16b-distill-llama2-7b \
  tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
  hf_path='HuggingFaceH4/ultrachat_200k' \
  train_split='train_sft' train_data_columns=['messages'] \
  load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH} \
  model_name=llama2-7b \
  per_device_batch_size=2 ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN
```
