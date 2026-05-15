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

(run-pretraining)=

# Pre-training

Pre-training is the process of training a language model from scratch (or from randomly initialized weights) on large-scale datasets to learn general language understanding. This guide covers running pre-training workloads with MaxText, including model selection, hyperparameter configuration, dataset setup, deployment options, and monitoring.

## Prerequisites

Before starting, ensure you have:

1. **Completed the installation** - See [Install MaxText](../install_maxtext.md)
2. **Set up a Cloud Storage bucket** - For storing logs and checkpoints. See [First Run](../tutorials/first_run.md#prerequisites-set-up-storage-and-configure-maxtext) for detailed instructions.
3. **Configured your environment** - Set `BASE_OUTPUT_DIRECTORY` environment variable:
   ```bash
   export BASE_OUTPUT_DIRECTORY=gs://<your-bucket>/<your-folder>
   ```

## 1. Model selection

MaxText provides pre-configured models for popular architectures. To use one,
specify the `model_name` parameter when using `maxtext.trainers.pre_train.train`.
MaxText will load the corresponding configuration from `src/maxtext/configs/models/`
(for TPU defaults) or `src/maxtext/configs/models/gpu/` (for GPU defaults).

MaxText supports many open-source models. For a complete list, see the [configs/models](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/configs/models) directory and [our README](https://github.com/AI-Hypercomputer/maxtext/blob/main/README.md).

### Example: Training with a specific model

```bash
python3 -m maxtext.trainers.pre_train.train \
  model_name=llama3-8b \
  run_name=my_pretraining_run \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=synthetic \
  steps=100
```

### Custom model configurations

You can also create custom model configurations or override specific architecture parameters:

```bash
# Override specific parameters
python3 -m maxtext.trainers.pre_train.train \
  model_name=llama3-8b \
  base_emb_dim=4096 \
  base_num_decoder_layers=32 \
  run_name=custom_model \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=synthetic \
  steps=100
```

**Note:** You cannot override parameters that are already defined in the model config file from the command line. To fully customize a model, create a new YAML config file similar to the ones in the [MaxText repository](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/configs) and pass it as a parameter:

```bash
# Override specific parameters
python3 -m maxtext.trainers.pre_train.train \
  /path/to/model_config.yml \
  run_name=custom_model \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=synthetic \
  steps=100
```

## 2. Hyperparameter configuration

Key hyperparameters control the training process. Here are the most important ones.

- **Training duration:**
  - `steps`: Total number of training steps (required)
  - `max_target_length`: Maximum sequence length in tokens (default: 1024)
  - `per_device_batch_size`: Batch size per device/chip (default: 12.0)
- **Learning rate and optimizer:**
  - `learning_rate`: Peak learning rate (default: 3e-5)
  - `opt_type`: Optimizer type - `adamw`, `adam_pax`, `sgd` or `muon` (default: `adamw`)
- **Checkpointing:**
  - `enable_checkpointing`: Save checkpoints during training (default: `True`)
  - `checkpoint_period`: (default: 10000)
- **Logging and monitoring:**
  - `log_period`: The frequency of Tensorboard flush, gcs metrics writing, and managed profiler metrics updating (default: 100)
  - `enable_tensorboard`: Enable TensorBoard logging (default: `True`)

### Example with common hyperparameters

```bash
python3 -m maxtext.trainers.pre_train.train \
  model_name=qwen3-4b \
  run_name=qwen_pretrain \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=hf \
  hf_path=allenai/c4 \
  hf_data_dir=en \
  train_split=train \
  tokenizer_type=huggingface \
  tokenizer_path=Qwen/Qwen2.5-4B \
  steps=10000 \
  per_device_batch_size=16 \
  max_target_length=4096 \
  learning_rate=1e-4 \
  checkpoint_period=1000 \
  log_period=50
```

For a complete list of configurable parameters, see
[src/maxtext/configs/base.yml](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml).

## 3. Dataset configuration

MaxText supports three dataset input pipelines: Grain, HuggingFace and TFDS
(TensorFlow Datasets). We will briefly describe how to get started with each of
them in this document. See [Data Input Pipeline](../guides/data_input_pipeline.md) and the individual guides listed in that document for more details.

### Grain pipeline

Grain is the **recommended input pipeline** for production training due to its determinism and resilience to preemption. It supports ArrayRecord (random access) and Parquet (sequential access) formats.

To get started, you need to:

1. **Download data** to a Cloud Storage bucket
2. **Mount the bucket** using [Cloud Storage FUSE (GCSFuse)](https://cloud.google.com/storage/docs/gcs-fuse)

```bash
bash src/dependencies/scripts/setup_gcsfuse.sh \
  DATASET_GCS_BUCKET=gs://<your-dataset> \
  MOUNT_PATH=/tmp/gcsfuse
```

After training, unmount the bucket:

```bash
fusermount -u /tmp/gcsfuse
```

#### Example: using GCSFuse and ArrayRecord

```bash
# Replace DATASET_GCS_BUCKET and base_output_directory with your buckets; replace run-name with your run name
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://<your-bucket> \
  run_name=<run-name> \
  model_name=deepseek2-16b \
  per_device_batch_size=1 \
  steps=10 \
  max_target_length=2048 \
  enable_checkpointing=false \
  dataset_type=grain \
  grain_file_type=arrayrecord \
  grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* \
  grain_worker_count=2 \
```

#### Dataset configuration parameters

- `dataset_type`: Set to `grain`
- `grain_file_type`: Format type - `arrayrecord` (recommended) or `parquet`
- `grain_train_files`: Path pattern to training files (supports regex patterns like `*`)
- `grain_worker_count`: Number of child processes for data loading (tune for performance)

#### Evaluation during training (optional)

To add periodic evaluation during training, specify an evaluation interval:

```bash
# Add to your command
eval_interval=5 \
eval_steps=10 \
grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record*
```

For comprehensive Grain configuration and best practices, see [Grain Pipeline Guide](../guides/data_input_pipeline/data_input_grain.md).

### HuggingFace pipeline

The HuggingFace pipeline provides the easiest way to get started with real datasets, streaming data directly from the HuggingFace Hub without requiring downloads. Alternatively, you can stream from a Cloud Storage bucket; see [HuggingFace Pipeline](../guides/data_input_pipeline/data_input_hf.md) for details.

#### Example: [allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset

We'll use the [allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset from HuggingFace, a processed version of Google's C4 (Colossal Clean Crawled Corpus). This dataset is organized into subsets (e.g., `en`, `es`), each containing data splits (e.g., `train`, `validation`).

```bash
# Replace base_output_directory with your bucket and run-name with your run name
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://<your-bucket> \
  run_name=<run-name> \
  model_name=deepseek2-16b \
  per_device_batch_size=1 \
  steps=10 \
  max_target_length=2048 \
  enable_checkpointing=false \
  dataset_type=hf \
  hf_path=allenai/c4 \
  hf_data_dir=en \
  train_split=train \
  tokenizer_type=huggingface \
  tokenizer_path=deepseek-ai/DeepSeek-V2-Lite
```

#### Dataset configuration parameters

- `dataset_type`: set to `hf`
- `hf_path`: set to the HF dataset you want to use
- `hf_data_dir`: path to the data on the HuggingFace repository

#### Evaluation during training (optional)

To add periodic evaluation during training, specify an evaluation interval and split:

```bash
eval_interval=5 \
eval_steps=10 \
hf_eval_split=validation
```

This runs 10 evaluation steps every 5 training steps using the `validation` split.

For a more comprehensive description of the available configuration parameters, see [HuggingFace Pipeline](../guides/data_input_pipeline/data_input_hf.md).

### TFDS pipeline

The TensorFlow Datasets (TFDS) pipeline uses datasets in TFRecord format, which is performant and widely supported in the TensorFlow ecosystem.

To get started, you need to:

1. **Create a Cloud Storage bucket** for your dataset
2. **Download dataset** to your bucket

You can use [the `download_dataset.sh` script provided by MaxText](https://github.com/AI-Hypercomputer/maxtext/blob/main/tools/data_generation/download_dataset.sh) to download the AllenAI C4 dataset:

```bash
bash download_dataset.sh <GCS_PROJECT> <GCS_BUCKET_FOR_DATASET>
```

#### Example: training with TFDS

```bash
# Replace base_output_directory and dataset_path with your buckets
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://<your-bucket> \
  run_name=demo \
  model_name=deepseek2-16b \
  per_device_batch_size=1 \
  steps=10 \
  max_target_length=2048 \
  enable_checkpointing=false \
  dataset_type=tfds \
  dataset_path=gs://<your-dataset> \
  dataset_name='c4/en:3.0.1' \
  train_split=train \
  tokenizer_type=huggingface \
  tokenizer_path=deepseek-ai/DeepSeek-V2-Lite
```

#### Dataset configuration parameters

- `dataset_type`: Set to `tfds`
- `dataset_path`: Cloud Storage bucket containing the dataset (e.g., `gs://<your-dataset>`)
- `dataset_name`: Subdirectory path within the bucket (e.g., `c4/en:3.0.1`)
- `train_split`: Split name for training (typically `train`, corresponds to `*-train.tfrecord-*` files)

The pipeline reads from files matching the pattern:

```
gs://<your-dataset>/c4/en/3.0.1/c4-train.tfrecord-0000-of-01024
```

#### Evaluation during training (optional)

To add periodic evaluation during training, specify an evaluation interval:

```bash
# Add to your command
eval_interval=5 \
eval_steps=10 \
eval_dataset_name='c4/en:3.0.1' \
eval_split=validation
```

- `eval_dataset_name`: Can be different from `dataset_name` if evaluating on a different dataset
- `eval_split`: Split name for evaluation (corresponds to `*-validation.tfrecord-*` files)

For comprehensive TFDS configuration, see [TFDS Pipeline Guide](../guides/data_input_pipeline/data_input_tfds.md).

## 4. Deployment options

Choose your deployment method based on your scale and infrastructure.

- **Localhost / Single VM:** Best for getting started and testing configurations on a single machine with a single TPU or GPU. See how to [run MaxText via localhost](./run_maxtext_localhost) or get specific instructions for [running on a single-host GPU](./run_maxtext_single_host_gpu).
- **XPK (Google Kubernetes Engine):** Best for large-scale training on TPU or GPU clusters. See how to [run MaxText via XPK](./run_maxtext_via_xpk).
- **Pathways:** Best for large-scale multi-host JAX jobs on TPUs. See how to [run MaxText via Pathways](./run_maxtext_via_pathways).
- **Decoupled Mode:** Best for local testing and development without Google Cloud dependencies. See how to [run MaxText in Decoupled Mode](./decoupled_mode).

## 5. Monitoring training progress

### Understanding logs

MaxText produces detailed logs during training. Here's what to look for:

```
completed step: 100, seconds: 1.234, TFLOP/s/device: 156.789, Tokens/s/device: 10234.567, total_weights: 8192, loss: 3.456
```

- `step`: Current training step
- `seconds`: Time taken for this step
- `TFLOP/s/device`: Compute throughput per device
- `Tokens/s/device`: Token processing rate per device
- `total_weights`: Number of actual tokens processed (excluding padding)
- `loss`: Training loss value

For detailed log interpretation, see [Understand Logs and Metrics](../guides/monitoring_and_debugging/understand_logs_and_metrics).

## 6. Complete pre-training example

Here's a complete example that combines all the concepts, using the HuggingFace pipeline to pre-train a Llama3-8B model on the C4 dataset:

```bash
# Pre-training Llama3-8B on C4 dataset using HuggingFace pipeline
python3 -m maxtext.trainers.pre_train.train \
  model_name=llama3-8b \
  run_name=llama3_c4_pretrain \
  base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
  dataset_type=hf \
  hf_path=allenai/c4 \
  hf_data_dir=en \
  train_split=train \
  tokenizer_type=huggingface \
  tokenizer_path=meta-llama/Meta-Llama-3-8B \
  steps=100000 \
  per_device_batch_size=16 \
  max_target_length=4096 \
  learning_rate=3e-4 \
  warmup_steps=2000 \
  checkpoint_period=5000 \
  log_period=100 \
  enable_tensorboard=true \
  eval_interval=1000 \
  eval_steps=100 \
  hf_eval_split=validation \
  enable_goodput_recording=true
```
