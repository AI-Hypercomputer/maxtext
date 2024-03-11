<!--
 Copyright 2023 Google LLC

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

## About Gemma

Gemma is a family of lightweight, state-of-the art open models built from research and technology that we used to create the Gemini models. To get started on decoding and finetuning of Gemma, you will first need to download weights from [kaggle](https://www.kaggle.com/models/google/gemma?rvi=1)

Following commands will let you download Gemma-2B model weights along with its tokenizer, convert the orbax checkpoints to be compatible with MaxText and upload it to a GCS bucket. \
Values for environment variables $KAGGLE_USERNAME and $KAGGLE_KEY can be set using your kaggle account's [API credentials](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#api-credentials). \
Please use seperate GCS buckets for uploading model weights from kaggle ($MODEL_BUCKET) and MaxText compatible weights ($CHKPT_BUCKET).
```
wget https://www.kaggle.com/api/v1/models/google/gemma/maxtext/2b/1/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge
# Extract downloaded model
tar -xf download
# export variables $CHKPT_BUCKET and $MODEL_BUCKET which are google cloud buckets to store weights
gsutil -m cp -r 2b/* $CHKPT_BUCKET/2b
gsutil -m cp tokenizer.model $CHKPT_BUCKET/tokenizer.model

python MaxText/convert_gemma_chkpt.py --base_model_path $CHKPT_BUCKET/2b --maxtext_model_path $MODEL_BUCKET/2b --model_size 2b
```

### Run `decode.py`.

```
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=$CHKPT_BUCKET/tokenizer.model load_parameters_path=$MODEL_BUCKET/{MODEL_VARIATION}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=64 dataset_type=synthetic steps=10 async_checkpointing=false attention=dot_product model_name=gemma-2b prompt="Kaggle is good for"
```

### MaxText supports fine-tuning with high performance.

Command for training Gemma-2b from scratch on 1 slice of v5e-256.
```
python MaxText/train.py MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIR model_name=gemma-2b dataset_path=$DATASET_PATH enable_checkpointing=false tokenizer_path=$CHKPT_BUCKET/tokenizer.model steps=10 ici_fsdp_transpose_parallelism=16 per_device_batch_size=2 remat_policy=minimal max_target_length=8192
```

### Performance

Model Flop utilization for training on v5e and v5p TPUs.


| Model    | v5e-256 (bf16) | v5p-128 (bf16) | v5e-256 (int8) | v5p-128 (int8) |
| -------- | -------------- | -------------- | -------------- | -------------- |
| Gemma-2b | 58.21%         | 55.36%         | 64.68%         | 67.80%         |
| Gemma-7b | 57.70%         | 60.16%         | 70.31%         | 70.12%         |