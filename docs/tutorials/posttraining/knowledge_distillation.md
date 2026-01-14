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

This guide focuses on **response-based knowledge distillation**, a technique where the student model is trained to replicate the outputs and behaviors of the teacher model. Within response-based knowledge distillation, two primary methods are often employed:

1.  **Offline Distillation (Dataset Generation):**
    *   The pre-trained teacher model (running in vLLM) generates a new dataset of input-output pairs.
    *   The student model is then trained on this teacher-generated dataset using standard fine-tuning techniques in MaxText.

2.  **Online Distillation (Logit Matching):**
    *   During the training process, both the teacher model (which is typically frozen) and the student model process the same input data simultaneously.
    *   The student model is trained by minimizing a loss function that encourages its output logits to match the logits produced by the teacher model for the same inputs.

## Running Offline Distillation with MaxText

The following recipe demonstrates the process of offline distillation using **Deepseek2-16b** as the teacher model and **Llama2-7b** as the student model. Since this recipe fine-tunes the student model using Supervised Fine-Tuning (SFT), it's crucial to use the conversational variant for both the teacher and student models. Here’s a step-by-step guide:

### Prerequisites

#### a. Setup environment variables

```bash
export HF_TOKEN=<your-hf-token> # hf_BA6...
export BUCKET_NAME=<your-bucket-name> # distill-bucket
export MOUNT_PATH=<your-mount-path> # ~/gcs-bucket
export RUN_NAME=<your-run-name> # distill-20260115
```

#### b. Install dependencies

To install MaxText and its dependencies for post-training (including vLLM for the teacher), run the following:

1. Follow the [MaxText installation instructions](https://maxtext.readthedocs.io/en/latest/install_maxtext.html#install-maxtext):

2. Install the additional dependencies for post-training:
```bash
bash tools/setup/setup_post_training_requirements.sh
```

#### c. Mount GCS bucket

Since TPU VM local disks are often limited in size, you must mount your GCS bucket as a local directory using `gcsfuse` to store large model weights during download and conversion.

1. **Create a mount point**:
   ```bash
   mkdir -p ${MOUNT_PATH}
   ```

2. **Mount the bucket**:
   ```bash
   gcsfuse --implicit-dirs ${BUCKET_NAME} ${MOUNT_PATH}
   ```

> **Note on Permissions:** Ensure your TPU VM's service account has the **Storage Object Admin** or **Storage Object Creator** role on the bucket. Without these permissions, the mount will fail or be read-only.

### 1. Obtain and prepare the teacher model

For the teacher model, we will use **vLLM** to run inference. vLLM can load Hugging Face checkpoints directly, so **no conversion to MaxText format is needed** for the teacher.

You can simply download the model from Hugging Face to your mounted bucket:

```bash
huggingface-cli login --token $HF_TOKEN
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat --repo-type model --local-dir ${MOUNT_PATH}/deepseek2-16b-chat
```

### 2. Obtain and prepare the student model

The student model will be trained in MaxText, which uses the Orbax checkpointing format. You will download the Hugging Face weights to your mounted bucket and convert them for training.

#### a. Download model from Hugging Face
Download the student model to the mounted bucket.

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --repo-type model --local-dir ${MOUNT_PATH}/llama2-7b-chat
```

#### b. Convert checkpoint to MaxText format
The following command takes the Hugging Face weights from the mount and converts them to the MaxText format, saving the output back into your mounted bucket.

**Note:** This conversion script requires PyTorch.
```bash
uv pip install torch
```

```bash
# Convert to MaxText format and save to GCS
JAX_PLATFORMS=cpu \
python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt \
  --base-model-path ${MOUNT_PATH}/llama2-7b-chat \
  --maxtext-model-path ${MOUNT_PATH}/llama2-7b-chat/scanned \
  --model-size llama2-7b \
  --huggingface-checkpoint true
```

### 3. Generate dataset using vLLM (Teacher Step)

We will use vLLM to generate the dataset from the teacher model.

Create a python script named `generate_distillation_data_vllm.py` with the following content:

```python
from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd
import transformers

# --- Configuration ---
TEACHER_MODEL = "deepseek-ai/DeepSeek-V2-Lite-chat" # Path to teacher model or HF repo
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
PROMPT_COLUMN = "messages"
import os
OUTPUT_FILE = os.path.join(os.environ.get("MOUNT_PATH"), "datasets", "distillation_data.jsonl")
TP_SIZE = 8 # Number of TPU chips
MAX_MODEL_LEN = 2048
MAX_NEW_TOKENS = 512
# ---------------------

def apply_chat_template(example, tokenizer, prompt_column):
    messages = example[prompt_column]
    # Apply chat template (e.g., Llama-3, DeepSeek-V2)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"formatted_prompt": prompt}

def main():
    print(f"Loading dataset {DATASET_NAME} ({DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
    # Optional: limit dataset for quick testing
    # dataset = dataset.select(range(100))

    print(f"Loading tokenizer {TEACHER_MODEL}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TEACHER_MODEL)
    
    print("Formatting prompts...")
    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer, PROMPT_COLUMN),
        desc="Applying chat template"
    )
    prompts = dataset["formatted_prompt"]

    print(f"Initializing vLLM with model {TEACHER_MODEL}...")
    llm = LLM(
        model=TEACHER_MODEL,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True # Often helpful for TPU stability
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output, original_item in zip(outputs, dataset):
        results.append({
            "prompt": output.prompt,
            "completion": output.outputs[0].text,
            "original_messages": original_item[PROMPT_COLUMN]
        })

    print(f"Saving results to {OUTPUT_FILE}...")
    df = pd.DataFrame(results)
    df.to_json(OUTPUT_FILE, orient="records", lines=True)

if __name__ == "__main__":
    main()
```

Now, run the generation script:

```bash
# Ensure the output directory exists on the mount
mkdir -p ${MOUNT_PATH}/datasets

python3 generate_distillation_data_vllm.py

# After generation, the dataset is stored directly in your mounted bucket.
export OUTPUT_DATASET=${MOUNT_PATH}/datasets/distillation_data.jsonl
```

### 4. Fine-tune the student model using Supervised Fine Tuning (SFT)
You can now fine-tune your smaller student model using supervised fine-tuning technique in MaxText.

### 4.a. Fine-tune the student model using dataset generated in Step 3

Example command to run fine-tuning on a TPU v4-8:

```bash
# Using the variables set in Step a and Step 3
python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml \
  run_name=${RUN_NAME} \
  base_output_directory=${MOUNT_PATH}/distillation/deepseek2-16b-distill-llama2-7b \
  tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path=json \
  hf_train_files=${OUTPUT_DATASET} \
  train_split='train' \
  train_data_columns=['prompt','completion'] \
  load_parameters_path=${MOUNT_PATH}/llama2-7b-chat/scanned/0/items \
  model_name=llama2-7b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN \
  profiler=xplane
```

### 4.b. **[OPTIONAL]** Fine-tune the student model using the original dataset

The checkpoint from the student model's fine-tuning (on the teacher-generated dataset) can be used for a subsequent fine-tuning stage. In this step, the student model is fine-tuned on the original dataset that was initially provided to the teacher model for generating the dataset.

```bash
# Get the latest checkpoint for fine-tuned student model
CHECKPOINTS_PATH=${MOUNT_PATH}/distillation/deepseek2-16b-distill-llama2-7b/${RUN_NAME}/checkpoints
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
FINE_TUNED_MODEL_CKPT_PATH=${CHECKPOINTS_PATH}/${largest_dir}/items

# Fine-tune student model on original dataset
python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml \
  run_name=${RUN_NAME}_stage2 \
  base_output_directory=${MOUNT_PATH}/distillation/deepseek2-16b-distill-llama2-7b \
  tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
  dataset_type=hf \
  hf_path='HuggingFaceH4/ultrachat_200k' \
  train_split='train_sft' \
  train_data_columns=['messages'] \
  load_parameters_path=${FINE_TUNED_MODEL_CKPT_PATH} \
  model_name=llama2-7b \
  per_device_batch_size=2 \
  steps=200 \
  ici_expert_parallelism=-1 ici_fsdp_parallelism=4 \
  max_target_length=2048 \
  hf_access_token=$HF_TOKEN \
  profiler=xplane
```