<!--
 # Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# Gemma3

[Gemma3](https://ai.google.dev/gemma) is an iteration of the Gemma family, designed for enhanced performance and efficiency which is capable of running on a single-accelerator ([Developer Blog](https://blog.google/technology/developers/gemma-3/)).

We provide examples for checkpoint conversion and decoding/training/finetuning Gemma3 in test scripts at [tests/end_to_end/tpu/gemma3](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu/gemma3). 


## Pre-training
You can train from scratch to generate a new checkpoint. One example command to run pretraining Gemma3-4B model is as follows:

```sh
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma3-4b base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.gemma3 per_device_batch_size=1 run_name=runner_pretrain_gemma3_4b steps=10 enable_checkpointing=false sharding_tolerance=0.03
```

## Checkpoint Conversion
To obtain the Gemma3 model weights, follow the instructions provided on [Kaggle](https://www.kaggle.com/models/google/gemma-3/flax/). You will need to accept the Gemma3 license through your Kaggle account and utilize your Kaggle [API credentials](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials) for authentication. Once the weights are downloaded to your GCS bucket, use the [checkpoint conversion utils](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/utils/ckpt_conversion#usage) to transform the checkpoint into a format compatible with MaxText. This script will also upload the converted checkpoints to a Google Cloud Storage (GCS) bucket.

## Fine-tuning
After the conversion, you will have a MaxText compatible checkpoint which allows you to fine-tune it with different datasets. One example command to fine-tune a Gemma3-4B model is as follows:

```
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml model_name=gemma3-4b base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} load_parameters_path=${CONVERTED_CHECKPOINT} tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.gemma3 per_device_batch_size=1 run_name=runner_finetune_gemma3_4b steps=10 enable_checkpointing=true sharding_tolerance=0.03
```

## Decoding
One example to use a converted checkpoint to decode with prompt "I love to":

```
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml model_name=gemma3-4b tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer.gemma3 load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=runner_decode_gemma3_4b max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false scan_layers=false prompt="I love to"
```