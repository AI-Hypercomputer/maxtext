<!--
 Copyright 2026 Google LLC

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

# Batch Size

This document explains the different concepts of "batch size" within MaxText and how to configure them to tune performance and manage memory.

## Per-Device Batch Size

`per_device_batch_size` is the number of training examples processed by a single device in one forward and backward pass. This value impacts the memory usage on each device and is a configuration parameter in `configs/base.yml`

## Global Batch Size

`global_batch_to_train` is the total number of training examples processed before the optimizer performs a single weight update. It is the effective batch size for training, calculated as:

`global_batch_to_train = per_device_batch_size x number_of_devices x gradient_accumulation_steps`

You can set `per_device_batch_size` and `gradient_accumulation_steps` in `configs/base.yml`.

`global_batch_to_load` is the total number of examples the data input pipeline loads from storage at once. It can be larger than the training batch size to optimize I/O performance, and is calculated as:

`global_batch_to_load` = `global_batch_size_to_train_on x expansion_factor_real_data`

When `expansion_factor_real_data > 1`, only a subset of hosts read data from the source (e.g., a GCS bucket). These "loading hosts" read more data than they need for their own devices and distribute the surplus to other "non-loading" hosts. This reduces the number of concurrent connections to the data source, which can significantly improve I/O throughput. When set to between 0 and 1, it's for grain pipeline to use a smaller chip count to read checkpoint from a larger chip count job. Details in https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/data_input_grain.md#using-grain.

## Gradient Accumulation Steps

`gradient_accumulation_steps` defines how many forward/backward passes are performed before the optimizer updates the model weights. The gradients from each pass are accumulated (summed). It is discussed in more detail [here](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/tiling.html#gradient-accumulation).

For example, if `gradient_accumulation_steps` is set to `4`, the model will execute four forward and backward passes, sum the gradients, and then apply a single optimizer step. This achieves the same effective global batch size as quadrupling the `per_device_batch_size` with significantly less memory, but can potentially lead to lower MFU.

## Pipeline Microbatches

When pipeline parallelism is enabled, the global batch is split into smaller chunks called **microbatches**. These are fed into the pipeline sequentially, allowing different stages of the model to work on different microbatches simultaneously.

The `num_pipeline_microbatches` parameter in `configs/base.yml` configures how many of these smaller chunks the global batch is divided into. It must be a multiple of the total number of pipeline stages (`ici_pipeline_parallelism` x `dcn_pipeline_parallelism`).

The choice of `num_pipeline_microbatches` is a trade-off between reducing pipeline idle time and the computational efficiency within each stage. More microbatches reduces the "Pipeline Bubble" but leads to smaller matrix multiplications within each stage. Very small operations may not fully saturate the compute units of the hardware, potentially lowering arithmetic intensity.

## Batch Size Ramp-up

MaxText supports gradually increasing the batch size during the initial phase of training to improve stability, a technique also used in frameworks like [NVIDIA's NeMo Megatron](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/nemo_megatron/rampup_batch_size.html). This can be configured in `configs/base.yml`:

- Setting `enable_rampup_batch_size=True` activates the ramp-up process.
- `per_device_batch_size_start`: The minimum batch size to start training on.
- `per_device_batch_size`: The target batch size to stabilize on at the end of the ramp-up process.
- `per_device_batch_size_increment`: How much batch size increases for each ramp-up stage.
- `global_rampup_samples`: The total number of samples to process across all ramp-up stages.

The ramp-up is based on the number of samples processed, not the number of training steps. Each stage processes an equal number of samples before batch size is increased.

The number of stages is determined by:

`num_increments = (per_device_batch_size - per_device_batch_size_start) / per_device_batch_size_increment`

The total number of ramp-up samples (`global_rampup_samples`) is then distributed equally across these stages. The number of samples processed in each stage is determined by:

`samples_per_increment = global_rampup_samples / num_increments`

During training, the model processes `samples_per_increment` samples at the current batch size. Once this threshold is reached, the batch size is increased by `per_device_batch_size_increment` until the target `per_device_batch_size` is reached. This entire process is managed by the `RampupBatchManager` class.

## Reinforcement Learning (RL) Batch Size

The batch size parameters for RL training are defined in `configs/post_train/rl.yml`:

- `batch_size` refers to the number of unique prompts loaded from the dataset in a single batch. For instance, `batch_size=1` means one prompt is processed at a time by the data loader.

- `num_generations` is the number of times the policy generates multiple responses for a given prompt within a single training step.

- The effective training batch is the total number of prompt-response pairs used in a training step, calculated as `batch_size x num_generations`. It is determined by the number of responses generated for each prompt, which is configured by `num_generations`.

- `micro_batch_size` is used to split the batch of prompt-response pairs into smaller chunks for memory management. This enables overlapping the rollout phase (generating responses) of one micro-batch with the training phase (updating model weights) of the previous micro-batch, which can improve hardware utilization. A value of `-1` means no micro-batching is enabled.
