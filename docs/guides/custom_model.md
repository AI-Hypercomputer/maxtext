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

# Customize your model configs on TPU

## Introduction

This document provides a guide to optimize and customize your LLM model configurations for higher performance (i.e. MFU) on Cloud TPU. Note that this document focuses exclusively on performance tuning. The analysis of model quality and convergence behavior is outside of scope.

## Step 1. Identify initial configs

To begin, identify your model's size, review open-source model configs, and establish the initial configurations for each block. You can use our [reference calculator (on Colab)](https://colab.research.google.com/github/AI-Hypercomputer/maxtext/blob/main/docs/explanations/llm_calculator.ipynb) to estimate parameters and FLOPs for dense, Mixtral-like Mixture of Experts (MoE), and DeepSeek-like MoE models to help you estimate the parameter count and FLOPs.

Based on resources like [Language Modeling from Scratch](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf), we observe common architectural ratios for dense models, as shown below:

*   `mlp_dim / emb_dim`: 2.5-4
*   `head_dim * num_query_heads / emb_dim`: 1-2
*   `emb_dim / num_decoder_layers`: 100-200

For MoE models,

*   sparsity (`num_experts / num_experts_per_tok`): 4-32
*   `moe_mlp_dim / emb_dim`: 0.3-3

## Step 2. Consider TPU best practices

### Model configs

To unlock peak performance on [TPUs](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm), it is critical to keep the Matrix Multiply Unit (MXU) fully utilized. The MXU is the primary computational engine, with the Trillium chip specifically optimized for 256×256 matrix multiplications (earlier TPU versions, like v4/v5e/v5p, are optimized for 128×128 operations). Processing smaller matrix multiplications (e.g., two 128×128 operations on Trillium) will halve the efficiency compared to a single, fully-utilized 256×256 operation.

Therefore, for optimal efficiency:

*   Model and MLP Dimensions: Design your model's emb_dim and mlp_dim to be multiples of 256 (for Trillium) or 128 (for older TPUs).
*   Self-Attention Head Dimension: Ensure your attention head_dim are also multiples of 256 (for Trillium) or 128 (for older TPUs).

Generally, larger multiples are more efficient. If achieving these specific multiples isn't possible, prioritize dimensions to a multiple of either 8 or 128 to help the XLA compiler optimize memory and computation.

To achieve efficient memory usage on a TPU, configure your training with the largest batch size that fits within its memory limits (configure a rematerialization policy with offloading to achieve the best MFU). Each TPU core leverages internal 8×128 vector registers for highly optimized matrix multiplications. Therefore, for peak performance and to minimize padding, your batch size should ideally be a multiple of 128. If a multiple of 128 is not feasible, try a multiple of 8. For more detailed explanations, see this [performance guide](https://cloud.google.com/tpu/docs/performance-guide).

### Performance configs

Use these general runtime configurations to improve your model's performance.

* **Multi-Head Attention (MHA)**. If you are using MHA, we recommend to set `fused_qkv=True` to fuse the query, key, and value computations into a single, more efficient operation.

* **Flash Attention**. Use the largest possible block size to maximize throughput.

* **Memory usage**. To free up memory with large models, use custom remat policy to offload layer activations (including inputs, query, key, value, out projections, etc) to the host CPU.

* **Compiler flags**. XLA is the backend compiler for TPUs. Many critical performance settings can be controlled directly through XLA flags. We suggest beginning with the proven flags we have tested and provided [here](https://github.com/AI-Hypercomputer/maxtext/blob/02b6b8d2558f7dab7d2be024783977bdbb3ed251/benchmarks/xla_flags_library.py).

* **Benchmark**. For consistent speed tests, set `reuse_example_batch=1` to repeatedly use the same data batch, isolating computation speed from data loading. Or use on-the-fly generated data by setting `dataset_type=synthetic`.

(roofline-sharding)=
## Step 3. Choose efficient sharding strategies using Roofline Analysis

To achieve good performance, it's often necessary to co-design the model's dimensions (like the MLP dimension) along with the sharding strategy. We have included examples for Trillium that demonstrate which sharding approaches work well for specific models. We recommend reading [](sharding) and Jax’s [scaling book](https://jax-ml.github.io/scaling-book/sharding/).

For the calculation below on Trillium, we will use Arithmetic Intensity (AI) of 5100 for 2 ICI links bandwidth bandwidth (1D with wrapound or 2D without wraparound) and 2500 for 4 ICI links bandwidth (2D with wraparound on both dimensions) over the ICI. The later bandwidth is particularly for Trillium v6e-256 (16x16) with wraparound connection.

### Fully Sharded Data Parallelism (FSDP)

#### Pure FSDP

For pure FSDP to be effective, it must have enough memory to hold both a large data batch and a full, single layer of weights at the same time. 

FSPD AI: `global batch / sparsity` (`sparsity = num_experts / num_experts_per_tok`).

**Example with a sparsity of 16**:
  * `global batch / sparsity > hardware AI`
  * `global batch / 16 > 2500` (16x16 with wraparound)
  * `global batch > 40k` (in tokens)

We also need a single layer of weights to fit into memory which can be an issue for medium/large MoE models, e.g. DeepSeek has roughly 10B params per layer, which corresponds to 40GiB of bf16 weights and gradients, which will not fit into Trillium’s 32GiB of HBM. So the use of pure FSDP on Trillium is feasible for models with layers not exceeding roughly 5B parameters. For these larger models need Expert or Tensor Parallelism.

#### Mix FSDP

For sparse models, large models, or when scaling to a large number of chips FSDP can be used in conjunction with other sharding strategies, such as Expert Parallelism (EP), Tensor Parallelism (TP), and Pipeline Parallelism (PP).

The same AI as derived in the Pure FSDP section above still hold, we need `global batch / sparsity * FSDP > hardware AI` which is equivalently to `per device batch (pdb) / sparsity * TP * EP * PP > hardware AI`.

**Example with EP=16, FSDP=16, and sparsity=32**:
  * `pdb * EP / sparsity > hardware AI`
  * `pdb * 16 / 32 > 5100`
  * `pdb > 5100 * 32 / 16 = 10200` (in tokens)

We need a per device batch of at least 10200 in this case.

### Expert Parallelism (EP)

If pure FSDP doesn’t work either due to AI or to fit in layer weights, EP is generally the way to go for sparse models (large dense models should use TP).

AI of 1D EP on ICI rings `= 4 * mlp_dim / EP`. Communication cost of all-to-all is roughly 1/4 of all-gather and reduce-scatter.

**Example with EP=4**
* `4 * M > 5100 * 4`
* `M > 5,100 * 4 = 5,100`

**Example with EP=16**
* `4 * M > 5,100 * 16`
* `M > 5,100 * 4 = 20,400`

These examples show that to use EP, we need a large enough mlp dimension.

It's important to note that this is only a roofline analysis. A nocap strategy with a high degree of EP introduces additional overhead - load balancing across expert groups becomes more challenging.

### Tensor Parallelism (TP)

Tensor parallelism can be used for large dense models or super large sparse models, particularly helpful when a small per device batch is needed and to be used with PP.

AI of TP: M / TP

**Example with TP=4**
* `M / TP > hardware AI`
* `M / 4 > 5100`
* `M > 20400`

We have seen in practice M should be even larger- ideally 40k+. This is what we use for Llama-405B (M=53k), and was used for a custom sparse 10T model (M=40k, 64 experts).

TP=4 corresponds to a custom Trillium mesh, an 8x8 ring of 2x2 subrings (the TP communication operates on the 2x2 ring). This 2x2 ring performs well (near roofline), but the 8x8 rings perform poorly (0.5 x 1 axis). E.g. if we use FSDP=64, TP=4, the FSDP=64 communications will be slower than the hardware ICI roofline, so we prefer to use the full 16 axis when M is large enough.

**Example with TP=16**
* `M / TP > hardware AI`
* `M / 16 > 5100`
* `M > 81600`

To use TP=16, we need M > 80k (ideally larger, 100k+). We have used this in a custom dense model (900B, M=131k), which performs very well even at 1k per device tokens (scaling to 25k+ with a reasonable global batch).

## Step 4. Analyze experiments

With your configs, begin experimenting to evaluate the model's performance. We strongly recommend capturing a profile by following these [instructions](https://docs.jax.dev/en/latest/profiling.html#). If you are using MaxText, this can be done by simply setting `profiler=xplane` in your configuration.

After generating the profile, use a tool, like [xprof](https://github.com/openxla/xprof), [xprofiler](https://github.com/AI-Hypercomputer/cloud-diagnostics-xprof), or [tensorboard](https://github.com/tensorflow/tensorboard) to analyze the results. This example ([Profile TPU Programs](https://jax-ml.github.io/scaling-book/profiling/) can serve as your guide. A key principle for maximizing training throughput is to ensure you are fully utilizing the available HBM. Once you achieve satisfactory performance, you can proceed with full training runs. Continue to analyze your model and refine your configurations as needed.

## Example of dense model

### 900B dense model on Trillium

To use Trillium's 16x16 mesh efficiently for a large dense model, we would like to use TP=16. This requires a huge MLP dimension, of at least 5k * 16 = 80k. With a per-device batch size of 4k tokens, this model achieved 39.8% MFU. The model demonstrated excellent scalability, maintaining 37% MFU even when the batch size was reduced to just 1k tokens per device.

| | Final Configs |
|---|---|
| emb_dim | 16384 |
| mlp_dim | 131072 |
| head_dim | 256 |
| num_query_head | 64 |
| num_kv_head | 16 |
| num_decoder_layers | 128 |
| **Total Params** | 9.15E+11 |
| **MFU (1 pod Trillium)** | 39.8% |

## Example of MoE model

### 700B Mixtral-like MoE on Trillium

Our objective was to develop a custom Mixtral-like MoE model capable of high MFU on Trillium TPUs, targeting a 1.5 capacity factor (The **capacity factor** is a multiplier used to determine the processing capacity of each expert. it is used as Expert Capacity = (Tokens in Batch / Number of Experts) * Capacity Factor). We established an initial baseline of 43.1% MFU with a 1.0 capacity factor. Profiling revealed this configuration utilized approximately 20GiB HBM. To better leverage Trillium's 32GiB HBM and avoid potential convergence issues with large global batch sizes during scaling (maintaining a per device batch size of 8k), we made the following architectural adjustments:

*   Increased the MLP dimension from 3x to 4x of the model dimension (32,768 : 8,192).
*   Increased query heads from 32 to 128 for each layer, while reducing the number of layers from 72 to 56 to preserve overall model size around 700B.

These changes, without updating sharding strategies, initially yielded nearly 50% MFU. Upon increasing the capacity factor to 1.5 (adding a buffer to allow experts to handle imbalance in token routing), MFU slightly decreased to 38.1% and scaling to 4 pods to get 35.3% MFU, which still exceeded our target of 35%. More detailed configs can be found [here](https://github.com/AI-Hypercomputer/maxtext/blob/3662540ee852d0d8f8333a36c04ddc0f1316ebfb/benchmarks/maxtext_trillium_model_configs.py#L1743) in the repo.

| | Initial Configs | Experimental Config | Final Configs |
|---|---|---|---|
| emb_dim | 8192 | 8192 | 8192 |
| mlp_dim | **24576** | **32768** | **32768** |
| num_experts | 16 | 16 | 16 |
| num_experts_per_tok | 2 | 2 | 2 |
| sparsity | 8 | 8 | 8 |
| head_dim | 256 | 256 | 256 |
| num_query_head | **32** | **128** | **128** |
| num_kv_head | 8 | 8 | 8 |
| num_decoder_layers | **72** | **56** | **56** |
| capacity_factor | **1.0** | **1.0** | **1.5** |
| **Total Params** | 7.08E+11 | 7.54E+11 | 7.54E+11 |
| **Active Params** | 9.96E+10 | 1.23E+11 | 1.23E+11 |
| **MFU (1 pod Trillium)** | 43.1% | 49.8% | 38.1% |
| **MFU (4 pod Trillium)** | n/a | n/a | 35.3% |

### 10T Mixtral-like MoE on Trillium

Objective was to demonstrate achieving reasonable MFU on a low batch setting (2k tokens per device) for a highly sparse (sparsity=32) model on Trillium. This requires using pipeline parallelism over DCN, which in turn calls for EP+TP over ICI (EP=64, TP=4). This model achieved 26% MFU on 16 pods (PP=16), and degrades only by a few percent when adding in more DP replicas (24% MFU with PP=8 and DP=2), even at a small per device batch size of only 2k (scaling to 25k+ chips and maintaining a reasonable global batch size).

| | Final Configs |
|---|---|
| emb_dim | 10240 |
| mlp_dim | 40960 |
| num_experts | 64 |
| num_experts_per_tok | 2 |
| sparsity | 32 |
| head_dim | 256 |
| num_query_head | 64 |
| num_kv_head | 16 |
| num_decoder_layers | 128 |
| capacity_factor | 1.0 |
| **Total Params** | 1.04E+13 |
| **Active Params** | 3.76E+11 |
| **MFU (1 pod Trillium)** | 34.5% |
| **MFU(16 pods Trillium)** | 26.2% |
