# Supported models list

> **Purpose**: This page provides detailed, reference-style information about model families supported in MaxText. This page is a technical dictionary for quick lookup, reproducibility, and customization.

## Overview

MaxText is an open-source, high-performance LLM framework written in Python/JAX. It targets Google Cloud TPUs and NVIDIA GPUs for training. MaxText prioritizes scalability (from a single host to recent runs with tens of thousands of chips), high Model FLOPs Utilization (MFU), and simplicity by leveraging JAX with the XLA compiler and optimized JAX Pallas kernels.

**Key capabilities and features**:

* **Supported Precisions**: FP32, BF16, INT8, and FP8.
* **Ahead-of-Time Compilation (AOT)**: For faster model development/prototyping and earlier OOM detection.
* **Quantization**: Via **Qwix** (recommended) and AQT. See Quantization [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/explanations/quantization.md).
* **Diagnostics**: Structured error context via **`cloud_tpu_diagnostics`** (filters stack traces to user code), simple logging via `max_logging`, profiling in **XProf**, and visualization in **TensorBoard**.
* **Multi-Token Prediction (MTP)**: Enables token efficient training with mutli-token prediction.
* **Elastic Training**: Fault-tolorent and dynamic scale-up/scale-down on Cloud TPUs with Pathways.
* **Flexible Remat Policy**: Provides fine-grained control over memory-compute trade-offs. Users can select pre-defined policies (like 'full' or 'minimal') or set the policy to **'custom'**. 


## Supported model families

> _**Note on GPU Coverage**: Support and tested configurations for NVIDIA GPUs can vary by model family. Please see the specific model guides for details._

**Primary Platforms**: All model families listed below target **TPU** and **NVIDIA GPUs**.


### Llama

* **Variants**: Llama 2; **Llama 3 / 3.1 / 3.3**; Llama 4 (**Scout**, **Maverick**; text & multimodal)
* **Notes**: RoPE, RMSNorm, SwiGLU; GQA; routed experts (Llama 4); **QK-Norm** (Llama 4); multimodal projector & vision encoder.

### Mistral / Mixtral

* **Variants**: Mistral (dense); Mixtral 8×7B, 8×22B (MoE)
* **Notes**: Sliding-Window Attention (SWA), GQA; MoE top-k with load-balancing loss.

### Gemma

* **Variants**: Gemma 1 (2B/7B), Gemma 2 (2B/9B/27B), **Gemma 3 (4B/12B/27B)** (text & multimodal)
* **Notes**: RMSNorm; RoPE; GELU/SwiGLU; **QK-Norm** (Gemma 3); Local–Global interleaved attention; long-context scaling.

### DeepSeek

* **Variants**: V2 (16B, 236B), V3 (671B), R1
* **Notes**: MLA; shared/finer-grained experts; MTP; YaRN-style scaling.

### Qwen3

* **Variants**: Dense (0.6B–32B); MoE (30B-A3B, 235B-A22B, 480B Coder)
* **Notes**: **QK-Norm**, GQA, SwiGLU, RMSNorm, RoPE.

### GPT-OSS

* **Variants**: 20B, 120B
* **Notes**: Local–Global interleaved attention, GQA, attention sink; YaRN-style scaling; MoE.

## Parallelism building blocks

MaxText supports a wide range of parallelism strategies for scaling training and inference across TPUs and GPUs:

* **FSDP (Fully Sharded Data Parallel)**: Reduces memory footprint by sharding parameters and optimizer states across devices.
* **TP (Tensor Parallelism)**: Splits tensor computations (e.g., matrix multiplications, attention heads) across devices for intra-layer speedups.
* **EP (Expert Parallelism)**: Distributes MoE experts across devices, supporting dropless routing and load balancing to ensure efficient utilization.
* **DP (Data Parallelism)**: Replicates the model across devices while splitting the input data batches.
* **PP (Pipeline Parallelism)**: Splits layers across device stages to support extremely large models by managing inter-stage communication.
* **CP (Context Parallelism)**: Splits sequence tokens across devices, complementing tensor parallelism for long-context workloads.
* **Hybrid Parallelism**: Allows for flexible combinations of FSDP, TP, EP, DP, PP, and CP to maximize hardware utilization based on model size and topology.

## Performance characteristics

The following summarizes observed runtime efficiency and scaling behaviors of MaxText across different hardware and model types, based on published benchmarks and large-scale runs.

* **High MFU**: MaxText targets high Model FLOPs Utilization across scales; exact numbers vary by model, hardware and config. See [**Performance Metrics → MFU**](../performance_metrics.md#performance-metrics) for the definition and how we calculate it.
* **Quantization**: MaxText supports quantization via both the AQT and Qwix libraries. Qwix is the recommended approach, providing a non-intrusive way to apply various quantization techniques, including Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
 * **MoE**: The Mixture-of-Experts implementation features dropless routing with Megablox and `jax.lax.ragged_dot` kernels for enhanced performance.
* **Multi-Token Prediction (MTP)**: This feature improves training efficiency on DeepSeek-style models by adding an auxiliary loss based on predicting multiple future tokens.
* **Long-Context Optimizations**: Implements various efficient attention mechanisms, including: Grouped-Query Attention (GQA), Sliding-Window Attention (SWA), Local–Global interleaved attention, Multi-Head Latent Attention (MLA). They reduce the KV-cache size, making it possible to handle long contexts efficiently.
 

## References

* [MaxText Repo](https://github.com/AI-Hypercomputer/maxtext)

* **Model Implementation Guides & Source Code:**
    * **Llama**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama2/run_llama2.md) | [Llama2 and Llama3 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/llama2.py) | [Llama4 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/llama4.py)
    * **Gemma**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gemma/Run_Gemma.md) | [Gemma Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/gemma.py) | [Gemma2 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/gemma2.py) | [Gemma3 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/gemma3.py)
    * **Mixtral**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/Run_Mixtral.md) | [Mixtral Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/mixtral.py) | [Mistral Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/mistral.py)
    * **DeepSeek**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/deepseek/Run_DeepSeek.md) | [DeepSeek Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/deepseek.py)
    * **Qwen3**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/qwen/moe/run_qwen_moe.md) | [Qwen3 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/qwen3.py)
    * **GPT-OSS**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gpt_oss/run_gpt_oss.md) | [GPT-OSS Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/layers/gpt_oss.py)


* **Technical Explanations:**
    * [Parallelism & Sharding](../../guides/optimization/sharding.md)
    * [Quantization Documentation](../core_concepts/quantization.md)
    * [AOT Compilation Instructions](../../guides/monitoring_and_debugging/features_and_diagnostics.md#ahead-of-time-compilation-aot)
