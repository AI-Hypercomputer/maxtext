# Supported Models & Architectures

> **Purpose**: This section provides detailed and reference-style information about model families supported in MaxText. It is designed to be a technical dictionary for quick lookup, reproducibility, and customization.

## Overview

MaxText is an open-source, high-performance LLM framework written in Python/JAX. It targets Google Cloud TPUs and NVIDIA GPUs for both training and inference. MaxText prioritizes scalability (from single host to ~51K chips), high Model FLOPs Utilization (MFU), and simplicity by leveraging JAX and the XLA compiler instead of hand-optimized kernels.

**Key capabilities**:

* **Supported Precisions**: BF16, INT8, and FP8.
* **Ahead-of-Time Compilation (AOT)**: For faster startup and early OOM detection.
* **Quantization**: Support for Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
* **Diagnostics**: Stack trace collection, profiling, and automatic uploads to Vertex AI TensorBoard.
* **Features**: Multi-Token Prediction (MTP), efficient key-value cache, and modular imports.

> **Legend**: EP = Expert Parallelism; TP = Tensor/Context Parallelism; FSDP = Fully Sharded Data Parallel.

## Supported Models — At a Glance

| Family | Variants (as implemented) | Platforms | Train | Inference | Distinctive Implementation Details | Key Differentiators |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama** | Llama 2, Llama 4 (Scout & Maverick model sizes; text & multimodal WIP) | TPU v4/v5e/v5p, NVIDIA GPUs | ✓ | ✓ | RoPE (Rotary Position Embedding); RMSNorm; SwiGLU MLP; GQA (Grouped-Query Attention); routed experts (Llama 4); **QK-Norm** (Llama 4); multimodal projector & vision encoder (Llama 4 multimodal in-progress). | Standard, robust architecture with optional MoE, QK-Norm, and multimodal capabilities in Llama 4. |
| **Mistral / Mixtral** | Mistral (dense); Mixtral 8x7B (MoE) | TPU, NVIDIA GPUs | ✓ | ✓ | Sliding-Window Attention (SWA); GQA; MoE top-k experts; load-balancing loss. | Efficient long-context handling via Sliding-Window Attention; parameter-efficient MoE in Mixtral. |
| **Gemma** | Gemma 1/2; Gemma 3 (text & multimodal) | TPU, NVIDIA GPUs | ✓ | ✓ | RMSNorm; RoPE; GELU/SwiGLU; **QK-Norm** (Gemma 3); Local–Global interleaved attention (Gemma 3); text–vision fusion; long-context scaling. | Hybrid attention pattern in Gemma 3, balancing local and global context. |
| **DeepSeek** | V2-Lite (16B), V3 (671B) (MoE) | TPU, NVIDIA GPUs | ✓ | ✓ | MLA (Multi-Head Latent Attention); shared/finer-grained experts; MTP (Multi-Token Prediction); auxiliary-loss-free balancing; RoPE with YaRN-style scaling. | Unique MLA attention mechanism and auxiliary MTP loss for enhanced training and inference efficiency. |
| **Qwen3**| Dense (0.6B–32B) & MoE (30B-A3B, 235B-A22B, 480B Coder) | TPU, NVIDIA GPUs | ✓ | ✓ | **QK-Norm** (Query-Key Normalization); GQA; SwiGLU; RMSNorm; RoPE. | QK-Norm for improved training stability. |

## Parallelism Building Blocks

MaxText supports a wide range of parallelism strategies for scaling training and inference across TPUs and GPUs:

* **FSDP (Fully Sharded Data Parallel)**: Reduces memory footprint by sharding parameters and optimizer states across devices.
* **TP (Tensor Parallelism)**: Splits tensor computations (e.g., matrix multiplications, attention heads) across devices for intra-layer speedups.
* **EP (Expert Parallelism)**: Distributes MoE experts across devices, supporting dropless routing and load balancing to ensure efficient utilization.
* **DP (Data Parallelism)**: Replicates the model across devices while splitting the input data batches.
* **PP (Pipeline Parallelism)**: Splits layers across device stages to support extremely large models by managing inter-stage communication.
* **CP (Context Parallelism)**: Splits sequence tokens across devices, complementing tensor parallelism for long-context workloads.
* **Hybrid Parallelism**: Allows for flexible combinations of FSDP, TP, EP, DP, PP, and CP to maximize hardware utilization based on model size and topology.

## Performance Characteristics

The following summarizes observed runtime efficiency and scaling behaviors of MaxText across different hardware and model types, based on published benchmarks and large-scale runs.

* **Reported MFU**: MaxText achieves high Model FLOPs Utilization, with reported figures of ~64–70% on TPU v5p and ~46–67% on TPU v5e.
* **Quantization**: MaxText supports quantization via both the AQT and Qwix libraries. Qwix is the recommended approach, providing a non-intrusive way to apply various quantization techniques, including Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ).
* **MoE**: The Mixture-of-Experts implementation is production-ready, featuring dropless routing with Megablox and `jax.lax.ragged_dot` kernels for enhanced performance.
* **Multi-Token Prediction (MTP)**: This feature improves training efficiency on DeepSeek-style models by adding an auxiliary loss based on predicting multiple future tokens.
* **Long-context optimizations**: Attention mechanisms such as GQA, SWA, Local-Global Attention, and MLA significantly reduce the cost of the KV-cache, making it possible to handle long contexts efficiently.

## References

* [MaxText Repo](https://github.com/AI-Hypercomputer/maxtext)

* **Model Implementation Guides & Source Code:**
    * **Llama**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/run_llama2.md) | [Llama2 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/llama2.py) | [Llama4 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/llama4.py)
    * **Gemma**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/gemma/Run_Gemma.md) | [Gemma Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/gemma.py) | [Gemma2 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/gemma2.py) | [Gemma3 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/gemma3.py)
    * **Mixtral**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/Run_Mixtral.md) | [Mixtral Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/mixtral.py) | [Mistral Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/mistral.py)
    * **DeepSeek**: [Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/deepseek/Run_DeepSeek.md) | [DeepSeek Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/deepseek.py)
    * **Qwen3**: [Qwen3 Source](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/layers/qwen3.py)

* **Technical Explanations:**
    * [Parallelism & Sharding](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/explanations/sharding.md)
    * [Quantization Documentation](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/explanations/quantization.md)
    * [AOT Compilation Instructions](https://github.com/AI-Hypercomputer/maxtext/blob/main/README.md#ahead-of-time-compilation-aot)
    * [MoE Configuration Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/mixtral/Run_Mixtral.md)