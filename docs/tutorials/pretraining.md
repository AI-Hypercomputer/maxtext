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

(pretraining)=

# Pre-training

In this tutorial, we introduce how to run pretraining with real datasets. While synthetic data is commonly used for benchmarking, we rely on real datasets to obtain meaningful weights. Currently, MaxText supports three dataset input pipelines: HuggingFace, Grain, and TensorFlow Datasets (TFDS). We will walk you through: setting up dataset, modifying the [dataset configs](https://github.com/AI-Hypercomputer/maxtext/blob/08d9f20329ab55b9b928543fedd28ad173e1cd97/src/maxtext/configs/base.yml#L486-L514) and [tokenizer configs](https://github.com/AI-Hypercomputer/maxtext/blob/08d9f20329ab55b9b928543fedd28ad173e1cd97/src/maxtext/configs/base.yml#L452-L455) for training, and optionally enabling evaluation.

To start with, we focus on HuggingFace datasets for convenience.

- Later on, we will give brief examples for Grain and TFDS. For a comprehensive guide, see the [Data Input Pipeline](../guides/data_input_pipeline.md) topic.
- For demonstration, we use Deepseek-V2-Lite model and C4 dataset. C4 stands for "Colossal Clean Crawled Corpus", a high-quality pretraining dataset first introduced by Google's [T5](https://arxiv.org/pdf/1910.10683) work. Feel free to try other models and datasets.

## 1. HuggingFace pipeline

We use the HuggingFace dataset [allenai/c4](https://huggingface.co/datasets/allenai/c4), a processed version of Google's C4. This dataset is organized into subsets (e.g., `en`, `es`), and each subset contains data splits (e.g., `train`, `validation`).

**Data preparation**: You don't need to download data, as the pipeline can stream data directly from the HuggingFace Hub. Alternatively, it can stream from a Cloud Storage bucket; see the [HuggingFace Pipeline](../guides/data_input_pipeline/data_input_hf.md) page.

We can use this **command** for pretraining:

```bash
# replace base_output_directory with your bucket
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo \
model_name=deepseek2-16b per_device_batch_size=1 steps=10 max_target_length=2048 enable_checkpointing=false \
dataset_type=hf hf_path=allenai/c4 hf_data_dir=en train_split=train \
tokenizer_type=huggingface tokenizer_path=deepseek-ai/DeepSeek-V2-Lite
```

**Dataset config**:

- `dataset_type`: `hf`
- `hf_path`: the HuggingFace dataset repository is `allenai/c4`
- `hf_data_dir`: the subset is `en`, corresponding to English data.
- `train_split`: `train`. Training will use the `train` split.

The above command runs training only: `steps=10` on the `train` split, for `en` subset of `allenai/c4`. The log shows:

```
completed step: 1, seconds: 0.287, TFLOP/s/device: 110.951, Tokens/s/device: 7131.788, total_weights: 7517, loss: 12.021
...
completed step: 9, seconds: 1.010, TFLOP/s/device: 31.541, Tokens/s/device: 2027.424, total_weights: 7979, loss: 9.436
```

The total weights is the number of real tokens processed in each step. More explanation can be found in [Understand Logs and Metrics](understand-logs-and-metrics) page.

**Evaluation config (optional)**:

To add evaluation steps, we can specify a positive evaluation interval and the dataset split, for instance `eval_interval=5 eval_steps=10 hf_eval_split=validation`. For every 5 training step, we run evaluation for 10 steps, using the `validation` split. In the log, you will additionally see:

```
Completed eval step 0
...
Completed eval step 9
eval metrics after step: 4, loss=9.855, total_weights=75264.0
Completed eval step 0
...
Completed eval step 9
eval metrics after step: 9, loss=9.420, total_weights=75264.0
```

**Tokenizer config**:

- `tokenizer_type`: `huggingface`. Note HuggingFace input pipeline only supports HuggingFace tokenizer.
- `tokenizer_path`: `deepseek-ai/DeepSeek-V2-Lite`, corresponding to the HuggingFace [model repository](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main).

**HuggingFace access token (optional)**:

- For a [gated dataset](https://huggingface.co/docs/hub/en/datasets-gated) or a tokenizer from a [gated model](https://huggingface.co/docs/hub/en/models-gated), you need to request access on HuggingFace and provide `hf_access_token=<YOUR_TOKEN>` in the command. For instance, [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) is a gated model.

## 2. Grain pipeline

Grain is a library for reading data for training and evaluating JAX models. It is the recommended input pipeline for determinism and resilience! It supports data formats like ArrayRecord and Parquet. You can check [Grain pipeline](../guides/data_input_pipeline/data_input_grain.md) for more details.

**Data preparation**: You need to download data to a Cloud Storage bucket, and read data via Cloud Storage Fuse with [setup_gcsfuse.sh](https://github.com/AI-Hypercomputer/maxtext/blob/0baff00ac27bb7996c62057f235cc1d2f43d734e/setup_gcsfuse.sh#L18).

- For example, we can mount the bucket `gs://maxtext-dataset` on the local path `/tmp/gcsfuse` before training
  ```bash
  bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=/tmp/gcsfuse
  ```
- After training, we unmount the local path
  ```bash
  fusermount -u /tmp/gcsfuse
  ```

This **command** shows pretraining with Grain pipeline, along with evaluation:

```bash
# replace DATASET_GCS_BUCKET and base_output_directory with your buckets
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo \
model_name=deepseek2-16b per_device_batch_size=1 steps=10 max_target_length=2048 enable_checkpointing=false \
dataset_type=grain grain_file_type=arrayrecord grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* grain_worker_count=2 \
eval_interval=5 eval_steps=10 grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record* \
tokenizer_type=huggingface tokenizer_path=deepseek-ai/DeepSeek-V2-Lite
```

**Dataset config**:

- `dataset_type`: `grain`
- `grain_file_type`: `arrayrecord`. We also support `parquet`.
- `grain_train_files`: `/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record*`, which is a regex pattern.
- `grain_worker_count`: `2`. This parameter controls the number of child processes used by Grain, which should be tuned for performance.

**Evaluation config (optional)**:

- `eval_interval=5 eval_steps=10`: after every 5 train steps, perform 10 evaluation steps
- `grain_eval_files`: `/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record*`, which is a regex pattern.

**Tokenizer config**:

- The Grain pipeline supports tokenizer_type: `sentencepiece, huggingface`
- Here we use the same `huggingface` tokenizer as in Section 1. If you use a HuggingFace tokenizer from a gated model, you will need to provide `hf_access_token`.

## 3. TFDS pipeline

The TensorFlow Datasets (TFDS) pipeline uses dataset in the TFRecord format. You can check [TFDS Pipeline](../guides/data_input_pipeline/data_input_tfds.md) for more details.

**Data preparation**: You need to download data to a [Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets), and the pipeline streams data from the bucket.

- To download the AllenAI C4 dataset to your bucket, you can use [download_dataset.sh](https://github.com/AI-Hypercomputer/maxtext/blob/08d9f20329ab55b9b928543fedd28ad173e1cd97/download_dataset.sh#L19): `bash download_dataset.sh <GCS_PROJECT> <GCS_BUCKET_FOR_DATASET>`

This **command** shows pretraining with TFDS pipeline, along with evaluation:

```bash
# replace base_output_directory and dataset_path with your buckets
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo \
model_name=deepseek2-16b per_device_batch_size=1 steps=10 max_target_length=2048 enable_checkpointing=false \
dataset_type=tfds dataset_path=gs://maxtext-dataset dataset_name='c4/en:3.0.1' train_split=train \
eval_interval=5 eval_steps=10 eval_dataset_name='c4/en:3.0.1' eval_split=validation \
tokenizer_type=huggingface tokenizer_path=deepseek-ai/DeepSeek-V2-Lite
```

**Dataset config**:

- `dataset_type`: `tfds`
- `dataset_path`: the cloud storage bucket is `gs://maxtext-dataset`
- `dataset_name`: `c4/en:3.0.1` corresponds to the subdirectory inside dataset_path `gs://maxtext-dataset/c4/en/3.0.1`
- `train_split`: `train`, corresponds to `*-train.tfrecord-*` files
- Putting together, we are training on files like `gs://maxtext-dataset/c4/en/3.0.1/c4-train.tfrecord-0000-of-01024`

**Evaluation config (optional)**:

- `eval_interval=5 eval_steps=10`: after every 5 train steps, perform 10 evaluation steps
- `eval_dataset_name`: `c4/en:3.0.1`, corresponds to the subdirectory inside dataset_path `gs://maxtext-dataset/c4/en/3.0.1`. It can be different from `dataset_name`.
- `eval_split`: `validation`, corresponds to `*-validation.tfrecord-*` files
- Putting together, we are evaluating on files like `gs://maxtext-dataset/c4/en/3.0.1/c4-validation.tfrecord-00000-of-00008`

**Tokenizer config**:

- TFDS pipeline supports tokenizer_type: `sentencepiece, huggingface, tiktoken`
- Here we use the same `huggingface` tokenizer as in Section 1. If you use a HuggingFace tokenizer from a gated model, you will need to provide `hf_access_token`.
