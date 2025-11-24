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
# Full Fine-Tuning (Llama3-7B)

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
- Note: You may need to change the training parameters to fit the model to the
  TPU or GPU shape and to obtain an optimized performance.

## MaxText checkpoints

MaxText checkpoints are in their own format. You can see the format in the script for llama conversion script.

The conversion scripts for LLama work with Meta’s original checkpoints and not with HuggingFace Checkpoint.

E.g.

```bash
python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b
```

The conversion scripts do not use accelerators but need large host memory to perform the conversion.

- The base model checkpoints should be in the format `{name}.{chkpt_idx}.pth` 
    - For example: `mistral-7b.00.pth`
- For large size model (e.g. 70B model), this script requires large memory VM.
- The script load and save weights in a single pass.

### Sample full fine tuning script

Below is a sample training script for LLama2-7b.

```bash
python3 -m MaxText.train \
  src/MaxText/configs/base.yml \
  run_name="llama2-finetune-maxtext" \
  base_output_directory=${output_directory} \
  load_parameters_path=${path_to_checkpoint} \
  model_name='llama2-7b' \
  dataset_path=${dataset_path} \
  async_checkpointing=False  \
  model_name='llama2-7b' \
  steps=10 per_device_batch_size=.25
```

You can find some [end to end scripts here](https://github.com/AI-Hypercomputer/maxtext/tree/main/end_to_end/tpu).
These scripts can provide a reference point for various scripts.

### MaxText checkpoint to Hugging Face

Post finetuning or pre-training, MaxText also provides scripts to convert MaxText format weights back to [Hugging Face](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/llama_mistral_mixtral_orbax_to_hf.py).

#### Dataset

MaxText provides examples to work with [Common Crawl](https://commoncrawl.org/). The dataset is available in TFRecords format in a cloud bucket. MaxText provides scripts to copy the dataset to a Google Cloud Storage Bucket.

##### Common Crawl (c4) dataset setup

You need to run these steps once per project prior to any local development or cluster experiments.

1. Create two gcs buckets in your project, one for downloading and retrieving the dataset and the other for storing the logs.
2. Download the dataset in your gcs bucket

MaxText assumes these GCS buckets are created in the same project and that it has permissions to read and write from them:

```bash
bash tools/data_generation/download_dataset.sh ${GCS_PROJECT?} ${GCS_BUCKET_NAME?}
```

The above will download the c4 dataset to your GCS BUCKET.

## Parameters to achieve high MFU

This content is in progress.
