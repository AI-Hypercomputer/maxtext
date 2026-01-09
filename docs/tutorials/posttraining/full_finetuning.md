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

(full-finetuning)=
# Full fine-tuning on single-host TPUs

MaxText can perform pre-training and full finetuning. To perform full fine
tuning, you need to pass the checkpoint to the training script.

Following is the parameter to assign a checkpoint to the training script.

- `load_parameters_path`: Path to the checkpoint directory

The high level steps involve:
- Converting the model checkpoints to MaxText formatted checkpoints
- Preparing the dataset so that data can be fed into the training script.
  MaxText provides sample pipelines to load the data via tf.data or Pygrain from
  a disk or gcs bucket. Or it can also input data directly from the hugging face
  dataset.
- Running the training script with the checkpoint
- Note: Training parameters may require adjustment to align the model with the specific TPU or GPU topology and achieve optimal performance.

## MaxText checkpoints

MaxText checkpoints are in their own format. You can see the format in the script for llama conversion script.

### Meta's PyTorch checkpoint to Maxtext (Orbax) checkpoint

The conversion scripts for LLama work with Meta’s original checkpoints and not with HuggingFace Checkpoint.

#### Pre-requist
- Download the Meta format checkpoints 

  Option 1: Download the checkpoint from Meta (https://llama.meta.com/llama-downloads/) in your local directory.
  
  Option 2: Download the checkpoint from a GCS Bucket to a local directoty with command ```gcloud storage cp -r <GCS path for META format checkpoint> <local/path>``` .

- Install Torch CPU because TPU or GPU is not required in this convertion script.

  ```python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu```

- Setup Environment Variables

  ```bash
  export CONVERTED_CHECKPOINT_PATH=<GCS path for saving converted checkpoint> # e.g., gs://my-bucket/my-model-checkpoint
  export LOCAL_META_CHECKPOINT_PATH=<local path for META checkpoint> # e.g., /local/meta-ckpt
  ```
#### Running the weight conversion script

Using 11ama-7b as an example:

```bash
python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt \
--base-model-path ${LOCAL_META_CHECKPOINT_PATH} \
--model-size llama2-7b \
--maxtext-model-path ${CONVERTED_CHECKPOINT_PATH}
```
Note:

The conversion scripts do not use accelerators but need large host memory to perform the conversion.

- The base model checkpoints should be in the format `{name}.{chkpt_idx}.pth` 
    - For example: `mistral-7b.00.pth`
- For large size model (e.g. 70B model), this script requires large memory VM.
- The script load and save weights in a single pass.

### MaxText checkpoint to Hugging Face

Post finetuning or pre-training, MaxText also provides scripts to convert MaxText format weights back to [Hugging Face](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/utils/ckpt_scripts/llama_mistral_mixtral_orbax_to_hf.py).

#### Sample for coverting Maxtext format weight to Hugging Face format

- Setup Environment Variables

  ```bash
  export BASE_OUTPUT_DIRECTORY=<output directory to store run logs> # e.g., gs://my-bucket/my-output-directory
  export PATH_TO_CHECKPOINT=<GCS path for saving converted checkpoint>/0/items # e.g., ${CONVERTED_CHECKPOINT_PATH}/0/items
  export HF_MODLE_PATH=<local path for hf> # e.g., /local/convert_ckp
  ```
- Running the conversion script

  The following example is executing a v6e-8 TPU VM with llama2-7b.

  ```bash
  python3 -m MaxText.utils.ckpt_scripts.llama_mistral_mixtral_orbax_to_hf \
    src/MaxText/configs/base.yml \ 
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${PATH_TO_CHECKPOINT} \
    run_name="mxt-2-hf" \
    model_name='llama2-7b' \
    hardware=tpu \
    hf_model_path=${HF_MODLE_PATH}
  
  ```
### Dataset

MaxText provides examples to work with [Common Crawl](https://commoncrawl.org/). The dataset is available in TFRecords format in a cloud bucket. MaxText provides scripts to copy the dataset to a Google Cloud Storage Bucket.

##### Common Crawl (c4) dataset setup

Run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for downloading and retrieving the dataset and the other for storing the logs.
2. Download the dataset in your gcs bucket

MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them:

```bash
export PROJECT=<Google Cloud Project ID>
export DATASET_GCS_BUCKET=<GCS for dataset> # e.g., gs://my-bucket/my-dataset

bash tools/data_generation/download_dataset.sh ${PROJECT} ${DATASET_GCS_BUCKET}
```

The above will download the c4 dataset to the GCS BUCKET.

### Sample full fine tuning script

Below is a sample training script for LLama2-7b on v6e-8 TPU VM.

```bash
python3 -m MaxText.train \
  src/MaxText/configs/base.yml \
  run_name="llama2-finetune-maxtext" \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${PATH_TO_CHECKPOINT} \
  model_name='llama2-7b' \
  dataset_path=${DATASET_GCS_BUCKET} \
  async_checkpointing=False  \
  steps=10 per_device_batch_size=1
```

You can find some [end to end scripts here](https://github.com/AI-Hypercomputer/maxtext/tree/main/end_to_end/tpu).
These scripts can provide a reference point for various scripts.

## Parameters to achieve high MFU

This content is in progress.
