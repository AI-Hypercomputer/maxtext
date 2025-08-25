# Understand Logs and Metrics

## Introduction

When you run a training job, MaxText produces detailed output logs. This guide shows you how to interpret these logs to understand your configuration and monitor performance.

To start, run a simple pretraining job on a single-host TPU. The logs from this command will be used as an example throughout this guide.

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo \
model_name=deepseek2-16b \
per_device_batch_size=1 max_target_length=2048 steps=10 dataset_type=synthetic enable_checkpointing=false
```

## 1 Configuration Info

The first section of the log details the final configuration of your run. This is crucial for debugging, as it shows you exactly which parameters were used. 

MaxText builds its configuration in layers. 
- It starts with the **default configuration** from the YAML file [base.yml](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/base.yml), as you provide `MaxText/configs/base.yml`. 

- Then, it overwrites any of these values with the arguments you provide in the **command line**.
  ```
  Updating keys from env and command line: ['run_name', 'model_name', 'enable_checkpointing', 'base_output_directory', 'per_device_batch_size', 'dataset_type', 'steps', 'max_target_length']
  ```
- It updates parameters based on the **model-specific configuration** file. As you specify `model_name=deepseek2-16b`, it read from [deepseek2-16b.yml](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/models/deepseek2-16b.yml).

  ```
  Running Model: deepseek2-16b
  Updating following parameters in config

  base_emb_dim: 2048
  base_num_query_heads: 16
  ...
  Updating keys from model: ['base_emb_dim', 'base_num_query_heads', 'base_num_kv_heads', 'base_mlp_dim', 'base_moe_mlp_dim', 'base_num_decoder_layers', 'first_num_dense_layers', 'mlp_activations', 'vocab_size', 'enable_dropout', 'logits_via_embedding', 'normalization_layer_epsilon', 'num_experts', 'num_experts_per_tok', 'shared_experts', 'routed_scaling_factor', 'routed_score_func', 'routed_bias', 'decoder_block', 'attention_type', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'rope_type', 'rope_max_timescale', 'max_position_embeddings', 'original_max_position_embeddings', 'rope_factor', 'beta_fast', 'mscale']
  ```

The final, consolidated configuration is printed last.
```
# From base.yml default
Config param opt_type: adamw 
...
# From model config
Config param base_emb_dim: 2048 
...
# From command line
Config param dataset_type: synthetic
Config param steps: 10
Config param per_device_batch_size: 1.0
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

### Understanding Output Paths

MaxText organizes all of your run's artifacts into a main output directory. The primary location for your run is constructed by combining the `base_output_directory` and the `run_name` you specify in your command. Based on the logs above, the base path for this specific run is `gs://runner-maxtext-logs/demo`.

Within this base path, MaxText creates several subdirectories for different types of artifacts. Many of these are optional and only created if you enable them with a specific flag.
* **TensorBoard Logs (`tensorboard/`)**
    * This is enabled by default for logging metrics.
    * Path: `gs://runner-maxtext-logs/demo/tensorboard/`

* **Profiler Traces (`tensorboard/plugins/profile/`)**
    * Flag: `profiler=xplane`
    * Path: The profiler output is saved within the TensorBoard directory.

* **Metrics in Plain Text (`metrics/`)**
    * Flag: `gcs_metrics=True`
    * Path: `gs://runner-maxtext-logs/demo/metrics/`

* **Configuration File (`config.yml`)**
    * Flag: `save_config_to_gcs=True`
    * Path: `gs://runner-maxtext-logs/demo/config.yml`

* **Checkpoints (`checkpoints/`)**
    * Flag: `enable_checkpointing=True`
    * Path: `gs://runner-maxtext-logs/demo/checkpoints/`

To generate all optional artifacts in one run, you can set the corresponding flags in the command line, like in the example below.
```bash
# This command enables the profiler, text metrics, config saving, and checkpointing
python3 -m MaxText.train MaxText/configs/base.yml \
base_output_directory=gs://runner-maxtext-logs run_name=demo2 \
model_name=deepseek2-16b \
per_device_batch_size=1 max_target_length=2048 steps=10 dataset_type=synthetic \
profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 \
gcs_metrics=True \
save_config_to_gcs=True \
enable_checkpointing=True
```

## 2 Environment Info

Next, the log displays the software and hardware environment for your run. This is useful for verifying your setup and understanding how parallelism is being applied. 

```
System Information: Jax Version: 0.7.0
System Information: Jaxlib Version: 0.7.0
System Information: Jax Backend: PJRT C API
TFRT TPU v5
Num_devices: 4, shape (1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1)
```

- **Software**: You can confirm the versions of `Jax` and `Jaxlib`, which are core frameworks for MaxText library.
- **Hardware**: You are running on the `TPU v5` accelerator with `4` total devices.
- **Parallelism Strategy**: The `shape` tuple `(1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1)` shows how your devices are arranged for parallelism. 
	- Recall from Section 1, `Config param data_sharding: (('data', 'stage', 'fsdp', 'fsdp_transpose', 'sequence', 'context', 'context_autoregressive', 'tensor', 'tensor_transpose', 'tensor_sequence', 'expert', 'autoregressive'),)`. This confirms that all 4 devices are being used for Fully Sharded Data Parallelism (FSDP), which is the default behavior.


## 3 Memory and TFLOP

### 3.1 Memory Analysis

Before executing training, we first perform a "dry run" compilation of a training step to [analyze its memory requirement](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/train.py#L630-L632
). The log outputs [memory sizes](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/max_utils.py#L735-L753):
```
Total memory size: 70.0 GB, Output size: 44.5 GB, Temp size: 25.5 GB, Argument size: 44.5 GB, Host temp size: 0.0 GB.
```

The most important number is `Total memory size: 70.0 GB`. This is the total HBM the TPU device needs to execute the program. Here is a breakdown:
- `Argument size: 44.5 GB`: This is the memory needed to hold the inputs for your function. This typically includes the batch of data, parameter (master copy), and optimizer state (e.g., momentum).
- `Output size: 44.5 GB`: This is the space required to store the results of the computation, such as the updated model weights and updated optimizer states.
- `Temp size: 25.5 GB`: This is the "scratch space" memory. It's used for all the intermediate values created during the forward and backward passes that are discarded once the step is complete. This includes activation (forward pas), gradient (backward pass), and parameter (working copy, if mixed precision).
- Q: Why it does not sum up? A: Some memory are shared (usually between argument and output).
  - You might notice that the sum of the parts is greater than `70.0 GB (total)`: `44.5 GB (Argument) + 44.5 GB (Output) + 25.5 GB (Temp) = 114.5 GB`. The difference is due to a compiler optimization called memory aliasing. The compiler is smart enough to reuse memory blocks. The true calculation is `Total = Argument + Output + Temp - Aliased`. In our case, the compiler identified `44.5 GB (114.5 GB - 70.0 GB)` of memory that could be safely reused. Mostly likely, it reuses memory for `Argument` and `Output`.

In addition, it shows temporary memory used on the host CPU. In this case, `Host temp size: 0.0 GB`, indicating that all the significant memory allocation happens on the accelerator device.


### 3.2 Memory Snapshot 

The previous section is a forecast of memory usage for entire training step, from static analysis of the compiled code. What about the actual memory usage? We now turn to the  snapshot of current usage right after the training state initialization, from a real-time check during execution. 

To set stage for training, we first initialize the training state, which include parameter and optimizer states. At the [beginning](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/train.py#L695), the log shows a real-time snapshot of the [memory statistics](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/max_utils.py#L708-L717) on your TPU devices.

```
number parameters: 15.933 billion

Memstats: After params initialized:
	Using (GB) 44.6 / 95.74 (46.584500%) on TPU_0(process=0,(0,0,0,0))
	Using (GB) 44.6 / 95.74 (46.584500%) on TPU_1(process=0,(1,0,0,0))
	Using (GB) 44.6 / 95.74 (46.584500%) on TPU_2(process=0,(0,1,0,0))
	Using (GB) 44.6 / 95.74 (46.584500%) on TPU_3(process=0,(1,1,0,0))
```
This log shows that each of the four TPUs has `95.74 GB` of available High Bandwidth Memory (HBM). The initial training state is evenly distributed devices, with each using the same amount of `44.6 GB`.

We also note that `Argument size=44.5GB` from previous analysis is a close prediction of memory usage here.


### 3.3 Model TFLOP Per Device

As a background, **model FLOPs** are the floating point operations to perform model computations. For training, the computation includes a single forward and backward pass. 
- In MaxText, we estimate model FLOPs by summing operations in matrix multiplications (matmuls); see [calculate_tflops_training_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/maxtext_utils.py#L297).
- It depends on model architecture, input size (batch size, sequence length), and gradient accumulation steps. It does not include optimization operations.
- We breakdown the FLOPs into two parts:
  - "Learnable weight FLOPs" are matmuls between activations and learnable weights. Specifically, this occurs in embedding, feed forward networks, attention-related projections, and unembedding.
  - "Attention FLOPs" are matmuls in attention score computation like $\mathrm{softmax}{\left(\frac{QK^\top}{\sqrt{d}}\right)} V$. 
- More information can be found in the [Performance Metrics](https://github.com/AI-Hypercomputer/maxtext/blob/main/getting_started/Performance_Metrics.md) page.

One **TFLOP** (TeraFLOP) is equal to $10^{12}$ FLOPs. The log shows the theoretical estimate of **model TFLOP per device**:
```
Per train step:
Total TFLOPs: 31.86 
split as 94.54% learnable weight flops and 5.46% attention flops

number parameters: 15.933 billion
```

In this example, given `model=deepseek2-16b`, `per_device_batch_size=1`, `max_target_length=2048` and no gradient accumulation, we have $\text{model tflop per device} \approx 31.86$. 
- 94.54% of the TFLOPs are attributed to learnable weight and 5.46% are attributed to attention. 
- As you will see next, this number is important for calculating performace metrics, such as TFLOP/s/device and model flop utilization (MFU).


## 4 Training Metrics

Finally, we are getting to the training steps! In this section, we will introduce performance metrics including TFLOP/s/device, model flop utilization (MFU), and Tokens/s/device (throughput). We will briefly cover learning metrics including loss and total weights.
```
completed step: 0, seconds: 19.015, TFLOP/s/device: 1.676, Tokens/s/device: 107.703, total_weights: 8192, loss: 12.047
completed step: 1, seconds: 0.323, TFLOP/s/device: 98.718, Tokens/s/device: 6345.508, total_weights: 8192, loss: 12.047
completed step: 2, seconds: 1.022, TFLOP/s/device: 31.170, Tokens/s/device: 2003.577, total_weights: 8192, loss: 10.876
completed step: 3, seconds: 0.699, TFLOP/s/device: 45.558, Tokens/s/device: 2928.446, total_weights: 8192, loss: 9.823
completed step: 4, seconds: 1.012, TFLOP/s/device: 31.497, Tokens/s/device: 2024.600, total_weights: 8192, loss: 8.767
completed step: 5, seconds: 1.012, TFLOP/s/device: 31.497, Tokens/s/device: 2024.564, total_weights: 8192, loss: 7.867
completed step: 6, seconds: 1.012, TFLOP/s/device: 31.494, Tokens/s/device: 2024.384, total_weights: 8192, loss: 7.154
completed step: 7, seconds: 1.011, TFLOP/s/device: 31.502, Tokens/s/device: 2024.926, total_weights: 8192, loss: 6.616
completed step: 8, seconds: 1.011, TFLOP/s/device: 31.505, Tokens/s/device: 2025.138, total_weights: 8192, loss: 6.237
completed step: 9, seconds: 1.012, TFLOP/s/device: 31.496, Tokens/s/device: 2024.558, total_weights: 8192, loss: 5.989
```

Before we dive deep here, recall a few things from previous sections:
- $\text{max target length} = 2048$, $\text{per device batch size} = 1$
- $\text{model tflop per device} \approx 31.86$ (rounded), $\text{number of devices} = 4$


### 4.1 Performance Metrics

For the performance indicators, they fluctuate at the beginning, and become stable towards the end. Therefore, we usually read those from the last step. Let's take a closer look at Step 9.
```
completed step: 9, seconds: 1.012, TFLOP/s/device: 31.496, Tokens/s/device: 2024.558, total_weights: 8192, loss: 5.989
```
As shown in `seconds: 1.012`, $\text{measured step time in seconds} \approx 1.012$ (rounded).

**TFLOP Per Second Per device**

- It is [computed](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/metric_logger.py#L193-L194) as 

$$\text{tflop/s/device} = \frac{\text{model tflop per device}}{\text{measured step time in seconds}}$$

- Here we have `TFLOP/s/device: 31.496`. Let's try to verify manually: $31.86 /1.012 = 31.482$. Not exactly same but close, since the both tflop and time are rounded in log.
- Further, we can calculate **Model Flop Utilization (MFU)** from this:
  
$$\text{MFU} = \frac{\text{tflop/s/device}}{\text{peak hardware tflop/s}}$$
  
  For TPU v5p, $\text{peak hardware tflop/s}=459$. Thus, $31.496 / 459 = 6.86\%$.

**Tokens Per Second Per Device (throughput)**

-  It is [computed](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/metric_logger.py#L197-L199) as

$$\text{token/s/device} = \frac{\text{number of tokens per device}}{\text{measured step time in seconds}}$$

  - The numerator is from [calculate_tokens_training_per_device](https://github.com/AI-Hypercomputer/maxtext/blob/e969faabbb571285a51545530f34d8f0a9f237e9/MaxText/maxtext_utils.py#L151)

$$\text{number of tokens per device} = \text{per device batch size} \times \text{max target length}$$

  - Here we have `Tokens/s/device: 2024.558`. Let's try to verify manually: $1 \times 2048 /1.012 = 2023.715$. Not exactly same but close, since the time is rounded in log.


### 4.2 Learning Metrics

**Loss**. The loss is the key indicator of learning progress, which should decrease over time steps. Ideally, we want it to converge to a small value.

**Total Weights**. When discussing the throughput, we have $\text{number of tokens} = \text{per device batch size} \times \text{max target length} \times \text{number of device}$. In data preprocessing, for each sentence, we truncate or pad to max target length. The pad tokens are meaningless and the loss are calculated based on the nonpad tokens. Thus, we monitor $\text{number of nonpad tokens}$, which is shown as [total weights](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/train.py#L307).
- Here we see `total_weights: 8192` for all steps. This is because we are using `dataset_type=synthetic`, where all sentences are generated with length of `max_target_length=2048`. As a result, there are no padded tokens and total weights = number of tokens. 
- However, in real dataset, sentences can have variable lengths and total weights < number of tokens. For example, we can set `dataset_type=tfds dataset_path=gs://maxtext-dataset dataset_name='c4/en:3.0.1'`, and will see total weights is smaller than 8192:
  ```
  completed step: 8, seconds: 0.983, TFLOP/s/device: 32.418, Tokens/s/device: 2083.764, total_weights: 7805, loss: 9.607
  completed step: 9, seconds: 0.983, TFLOP/s/device: 32.397, Tokens/s/device: 2082.441, total_weights: 7100, loss: 9.794
  ```
- For better convergence, we want to have large total weights. For example, MaxText allows for [packing](https://github.com/AI-Hypercomputer/maxtext/blob/f82ce194c490d668b14574a072a0a630c27bbd6e/MaxText/sequence_packing.py#L39) multiple short sequences into one.
