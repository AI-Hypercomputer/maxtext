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

(understand-logs-and-metrics)=

# Understand logs and metrics

When you run a training job, MaxText produces detailed output logs. This guide shows you how to interpret these logs to understand your configuration and monitor performance.

To start, run a simple pretraining job on a single-host TPU. For instance, we can run the following command on TPU v5p-8. The resulting log is used as an example throughout this guide.

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo \
model_name=deepseek2-16b \
per_device_batch_size=24 max_target_length=2048 steps=10 dataset_type=synthetic enable_checkpointing=false
```

## 1. Configuration information

The first section of the log details the configuration of your run. This is crucial for debugging, as it shows you exactly which parameters were used.

MaxText builds its configuration in layers.

- It starts with the **default configuration** from a YAML file. In our example, the file is [`src/maxtext/configs/base.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/maxtext/configs/base.yml).

- Then, it overwrites any of these values with the arguments you provide in the **command line**.

  ```none
  Updating keys from env and command line: ['run_name', 'model_name', 'enable_checkpointing', 'base_output_directory', 'per_device_batch_size', 'dataset_type', 'steps', 'max_target_length']
  ```

- It updates keys based on the **model-specific configuration** file. When you specify a model, like `deepseek2-16b`, MaxText reads the corresponding parameters from the [deepseek2-16b.yml](https://github.com/AI-Hypercomputer/maxtext/blob/fafdeaa14183a8f5ca7b9f7b7542ce1655237574/src/maxtext/configs/models/deepseek2-16b.yml) file.

  ```none
  Running Model: deepseek2-16b
  Updating following parameters in config

  base_emb_dim: 2048
  base_num_query_heads: 16
  ...
  Updating keys from model: ['base_emb_dim', 'base_num_query_heads', 'base_num_kv_heads', 'base_mlp_dim', 'base_moe_mlp_dim', 'base_num_decoder_layers', 'first_num_dense_layers', 'mlp_activations', 'vocab_size', 'enable_dropout', 'logits_via_embedding', 'normalization_layer_epsilon', 'num_experts', 'num_experts_per_tok', 'shared_experts', 'routed_scaling_factor', 'routed_score_func', 'routed_bias', 'decoder_block', 'attention_type', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'rope_type', 'rope_max_timescale', 'max_position_embeddings', 'original_max_position_embeddings', 'rope_factor', 'beta_fast', 'mscale']
  ```

  Note that you cannot modify a key from both model config and command line.

The final, consolidated configuration is printed last.

```none
# From base.yml default
Config param opt_type: adamw
...
# From model config
Config param base_emb_dim: 2048
...
# From command line
Config param dataset_type: synthetic
Config param steps: 10
Config param per_device_batch_size: 24.0
Config param max_target_length: 2048
...
# Other config behind the scene
Config param data_sharding: (('data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'context_autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive'),)
...
```

This also includes the **output paths** for your run artifacts.

```
Config param base_output_directory: gs://runner-maxtext-logs
Config param run_name: demo
Config param metrics_dir: gs://runner-maxtext-logs/demo/metrics/
Config param tensorboard_dir: gs://runner-maxtext-logs/demo/tensorboard/
Config param checkpoint_dir: gs://runner-maxtext-logs/demo/checkpoints/
```

### Understanding output paths

MaxText organizes all of your run's artifacts into a main output directory. The primary location for your run is constructed by combining the `base_output_directory` and the `run_name` you specify in your command. Based on the logs above, the base path for this specific run is `gs://runner-maxtext-logs/demo`.

Within this base path, MaxText creates several subdirectories for different types of artifacts. Many of these are optional and only created if you enable them with a specific flag.

- **TensorBoard logs (`tensorboard/`)**

  - Flag: `enable_tensorboard=True` (default)
  - Path: `gs://runner-maxtext-logs/demo/tensorboard/`

- **Profiler traces (`tensorboard/plugins/profile/`)**

  - Flag: `profiler=xplane`
  - Path: The profiler output is saved within the TensorBoard directory.

- **Metrics in plain text (`metrics/`)**

  - Flag: `gcs_metrics=True`
  - Path: `gs://runner-maxtext-logs/demo/metrics/`

- **Configuration file (`config.yml`)**

  - Flag: `save_config_to_gcs=True`
  - Path: `gs://runner-maxtext-logs/demo/config.yml`

- **Checkpoints (`checkpoints/`)**

  - Flag: `enable_checkpointing=True`
  - Path: `gs://runner-maxtext-logs/demo/checkpoints/`

To generate all optional artifacts in one run, you can set the corresponding flags in the command line, like in the example below.

This command enables tensorboard, profiler, text metrics, config saving, and checkpointing:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo2 \
model_name=deepseek2-16b \
per_device_batch_size=24 max_target_length=2048 steps=10 dataset_type=synthetic \
enable_tensorboard=True \
profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 \
gcs_metrics=True \
save_config_to_gcs=True \
enable_checkpointing=True
```

## 2. Environment information

Next, the log displays the software and hardware environment for your run. This is useful for verifying your setup and understanding how parallelism is being applied.

```none
System Information: Jax Version: 0.7.2.dev20250826
System Information: Jaxlib Version: 0.7.2.dev20250826
System Information: Jax Backend: PJRT C API
TFRT TPU v5
Num_devices: 4, shape (1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1)
```

- **Software**: You can confirm the versions of `Jax` and `Jaxlib`, which are core frameworks for the MaxText library.
- **Hardware**: You are running on the `TPU v5` accelerator with `4` total devices.
- **Parallelism strategy**: The `shape` tuple `(1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1)` shows how your devices are arranged for parallelism. Recall from Section 1, `Config param data_sharding: (('data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'context_autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive'),)`. This confirms that all 4 devices are being used for Fully Sharded Data Parallelism (FSDP), which is the default behavior.

## 3. Resource accounting

Before executing training, the program analyzes the resource requirements for your training job, specifically memory and compute (FLOPs).

### 3.1. Memory analysis

We first perform a "dry run" compilation of a training step to [analyze its memory requirement](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/train.py#L380-L382). This static analysis is performed by the XLA compiler. The log outputs [memory sizes](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/max_utils.py#L672-L690):

```none
Total memory size: 100.4 GB, Output size: 44.5 GB, Temp size: 55.9 GB, Argument size: 44.5 GB, Host temp size: 0.0 GB.
```

The most important number is `Total memory size: 100.4 GB`. This is the total High Bandwidth Memory (HBM) the TPU device needs to execute the program. Here is a breakdown:

- `Argument size: 44.5 GB`: This is the memory needed to hold the inputs for your function. This typically includes the batch of data, parameter (master copy), and optimizer state (e.g., moment).
- `Output size: 44.5 GB`: This is the space required to store the results of the computation, such as the updated model weights and updated optimizer states.
- `Temp size: 55.9 GB`: This is the "scratch space" memory. It's used for all the intermediate values created during the forward and backward passes that are discarded once the step is complete. This includes activation (forward pass), gradient (backward pass), and parameter (working copy, if mixed precision).
- Memory aliasing: You might notice that the sum of the parts is greater than `100.4 GB (total)`: `44.5 GB (Argument) + 44.5 GB (Output) + 55.9 GB (Temp) = 144.9 GB`. The difference is due to a compiler optimization called memory aliasing. The compiler reuses memory blocks. The true calculation is `Total = Argument + Output + Temp - Aliased`. In our case, the compiler identified `44.5 GB (144.9 GB - 100.4 GB)` of memory that could be safely reused. Most likely, it reuses memory for `Argument` and `Output`.

In addition, it shows temporary memory used on the host CPU. In this case, `Host temp size: 0.0 GB`, indicating that all the significant memory allocation happens on the accelerator device.

### 3.2. Memory snapshot

The previous section is a forecast of memory usage for entire training step, based on static analysis of the compiled code from the XLA compiler. To see the actual memory usage, we now turn to a real-time snapshot from the JAX runtime, captured right after the training state is initialized.

To set the stage for training, we first initialize the training state, which include parameter and optimizer states. At the [beginning](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/train.py#L445), the log shows a real-time snapshot of the [memory statistics](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/max_utils.py#L645-L654) on your TPU devices.

```none
number parameters: 15.933 billion

Memstats: After params initialized:
	Using (GB) 44.63 / 95.74 (46.615835%) on TPU_0(process=0,(0,0,0,0))
	Using (GB) 44.63 / 95.74 (46.615835%) on TPU_1(process=0,(1,0,0,0))
	Using (GB) 44.63 / 95.74 (46.615835%) on TPU_2(process=0,(0,1,0,0))
	Using (GB) 44.63 / 95.74 (46.615835%) on TPU_3(process=0,(1,1,0,0))
```

This log shows that each of the four TPUs has `95.74 GB` of available High Bandwidth Memory (HBM). The initial training state is evenly distributed across devices, with each using the same amount of `44.63 GB`.

### 3.3. Model TFLOP per device

The **model FLOPs** are the floating point operations to perform model computation. For training, the computation includes a single forward and backward pass.

- In MaxText, we estimate model FLOPs by summing operations in matrix multiplications (matmuls); see [calculate_tflops_training_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/maxtext_utils.py#L480).
- The number of model FLOPs is dependent on model architecture, input size (batch size, sequence length), and gradient accumulation steps. It does not include optimization operations.
- We break down the FLOPs into two parts:
  - "Learnable weight FLOPs" are matmuls between activations and learnable weights. Specifically, this occurs in embedding, feed forward networks, attention-related projections, and unembedding.
  - "Attention FLOPs" are matmuls in attention score computation like $\mathrm{softmax}{\left(\frac{QK^\top}{\sqrt{d}}\right)} V$.

One **TFLOP** (TeraFLOP) is equal to $10^{12}$ FLOPs. The log shows the theoretical estimate of **model TFLOP per device**:

```none
Per train step:
 Total TFLOPs: 764.67
 split as 94.54% learnable weight flops and 5.46% attention flops
```

In this example, given `model=deepseek2-16b`, `per_device_batch_size=24`, `max_target_length=2048`, and no gradient accumulation, we have $\text{model tflop per device} \approx 764.67$.

- 94.54% of the TFLOPs are attributed to learnable weight and 5.46% are attributed to attention.
- As you will see next, this number is important for calculating performance metrics, such as TFLOP/s/device and Model FLOPs Utilization (MFU).

You can find more information about model FLOPs and MFU in the [Performance Metrics](performance-metrics) topic.

## 4. Training metrics

Finally, we are getting to the training steps! In this section, we introduce performance metrics including TFLOP/s/device, MFU, and Tokens/s/device (throughput). We briefly cover learning metrics including loss and total weights.

```none
completed step: 0, seconds: 44.923, TFLOP/s/device: 17.022, Tokens/s/device: 1094.129, total_weights: 196608, loss: 12.038
completed step: 1, seconds: 0.319, TFLOP/s/device: 2400.734, Tokens/s/device: 154316.608, total_weights: 196608, loss: 12.038
completed step: 2, seconds: 5.658, TFLOP/s/device: 135.158, Tokens/s/device: 8687.815, total_weights: 196608, loss: 11.689
completed step: 3, seconds: 5.402, TFLOP/s/device: 141.542, Tokens/s/device: 9098.189, total_weights: 196608, loss: 11.379
completed step: 4, seconds: 5.669, TFLOP/s/device: 134.884, Tokens/s/device: 8670.207, total_weights: 196608, loss: 11.110
completed step: 5, seconds: 5.668, TFLOP/s/device: 134.909, Tokens/s/device: 8671.794, total_weights: 196608, loss: 10.879
completed step: 6, seconds: 5.668, TFLOP/s/device: 134.914, Tokens/s/device: 8672.153, total_weights: 196608, loss: 10.688
completed step: 7, seconds: 5.669, TFLOP/s/device: 134.882, Tokens/s/device: 8670.101, total_weights: 196608, loss: 10.542
completed step: 8, seconds: 5.668, TFLOP/s/device: 134.911, Tokens/s/device: 8671.946, total_weights: 196608, loss: 10.440
completed step: 9, seconds: 5.667, TFLOP/s/device: 134.924, Tokens/s/device: 8672.758, total_weights: 196608, loss: 10.374
```

Before we dive deep here, recall a few numbers from previous sections:

- $\text{max target length} = 2048$, $\text{per device batch size} = 24$
- $\text{model tflop per device} \approx 764.67$ (rounded), $\text{number of devices} = 4$

### 4.1. Performance metrics

The performance metrics fluctuate at the beginning, and become stable towards the end. Therefore, we usually read them from the last step. Let's take a closer look at Step 9.

```none
completed step: 9, seconds: 5.667, TFLOP/s/device: 134.924, Tokens/s/device: 8672.758, total_weights: 196608, loss: 10.374
```

As shown in `seconds: 5.667`, $\text{measured step time in seconds} \approx 5.667$ (rounded).

**TFLOP per second per device**

- It is [computed](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/metric_logger.py#L211-L213) as

$$\text{tflop/s/device} = \frac{\text{model tflop per device}}{\text{measured step time in seconds}}$$

- Here we have `TFLOP/s/device: 134.924`. Let's try to verify manually: $764.67 / 5.667 = 134.934$. Not exactly the same but close, since the both tflop and time are rounded in the log.
- Further, we can calculate **Model FLOPs Utilization (MFU)** from this:

$$\text{MFU} = \frac{\text{tflop/s/device}}{\text{peak hardware tflop/s}}$$

For TPU v5p, $\text{peak hardware tflop/s}=459$. Thus, $134.924 / 459 = 29.40$%. Note this is an example for explanation with small batch size and sequence length, so the MFU is not optimal.

**Tokens per second per device (throughput)**

- It is [computed](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/metric_logger.py#L215-L217) as

$$\text{token/s/device} = \frac{\text{number of tokens per device}}{\text{measured step time in seconds}}$$

- The numerator is from [calculate_tokens_training_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/maxtext_utils.py#L148)

$$\text{number of tokens per device} = \text{per device batch size} \times \text{max target length}$$

- Here we have `Tokens/s/device: 8672.758`. Let's try to verify manually: $24 \times 2048 / 5.667 = 8673.372$. Not exactly the same but close, since the time is rounded in the log.

### 4.2. Learning metrics

**Loss**. The loss is the key indicator of learning progress, which should decrease over training steps. In this example, the loss is `12.038` at Step 0 and decreases to `10.374` at Step 9. Ideally, we want the loss to converge to a small value with sufficiently large training steps.

**Total weights**. When discussing the throughput, we have $\text{number of tokens} = \text{per device batch size} \times \text{max target length} \times \text{number of device}$. In this example, $\text{number of tokens} = 24 \times 2048 \times 4 = 196608$. There are two types of tokens: real tokens and pad tokens. The pad tokens are placeholders introduced by data preprocessing: We truncate or pad each sentence to max target length. Only real tokens contribute to the learning signal (i.e., loss). Therefore, we monitor $\text{number of real tokens}$, which is shown as [total weights](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/train.py#L151).

- Here we see `total_weights: 196608` for all steps. This is because we are using `dataset_type=synthetic`, where all sentences are generated with a length of `max_target_length=2048`. As a result, there are no pad tokens and total weights = number of tokens.
- However, in real datasets, sentences can have variable lengths and total weights < number of tokens. For example, we can set `dataset_type=tfds dataset_path=gs://maxtext-dataset dataset_name='c4/en:3.0.1'`, and will see total weights smaller than `196608`:
  ```none
  completed step: 8, seconds: 5.670, TFLOP/s/device: 134.856, Tokens/s/device: 8668.393, total_weights: 163259, loss: 9.596
  completed step: 9, seconds: 5.669, TFLOP/s/device: 134.884, Tokens/s/device: 8670.184, total_weights: 155934, loss: 9.580
  ```
- For better convergence, we want to have large total weights. Towards this end, MaxText supports [packing](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/MaxText/sequence_packing.py#L37) multiple short sequences into one. This is enabled by default with `packing=True` in [base.yml](https://github.com/AI-Hypercomputer/maxtext/blob/28e5097ac467ed8b1d17676d68aa5acc50f9d60d/src/maxtext/configs/base.yml#L465).
