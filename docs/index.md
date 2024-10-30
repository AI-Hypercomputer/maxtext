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

# MaxText

## Overview

MaxText is a a Google initiated open source project for **high performance**, **highly scalable**, **open-source** LLM written in pure Python/[JAX](https://jax.readthedocs.io/en/latest/index.html) and targeting Google Cloud TPUs and GPUs for **training** and **inference**. MaxText achieves [high MFUs](#runtime-performance-results) and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

MaxText achieves very high MFUs (Model Flop Utilization) and scales from single host to very large clusters while staying simple and "optimization-free".

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs.

We have used MaxText to [demonstrate high-performance, well-converging training in int8](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e) and [scale training to ~51K chips](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e).

Key supported features:
- TPUs and GPUs (in preview)
- Training and Inference (in preview)

MaxText additionally provides an highly optimized reference implementations for popular Open Source models like:

- Llama 2, 3 and 3.1
- Mistral and Mixtral
- Gemma and Gemma2
- GPT

These reference implementations support pre-training and full fine tuning. Maxtext also allows you to create various sized models for benchmarking purposes.

The key value proposition of using MaxText for pre-training or full fine tuning is:

- Very high performance of average of 50% MFU
- Open code base
- Easy to understand: MaxText is purely written in JAX and Python, which makes it accessible to ML developers interested in inspecting the implementation or stepping through it. It is written at the block-by-block level, with code for Embeddings, Attention, Normalization etc. Different Attention mechanisms like MQA and GQA are all present. For quantization, it uses the JAX AQT library. The implementation is suitable for both GPUs and TPUs.

```{note}
Maxtext today only supports Pre-training and Full Fine Tuning of the models. It does not support PEFT/LoRA, Supervised Fine Tuning or RLHF.
```

## Who are the target users of MaxText?

- Any individual or a company that is interested in forking maxtext and seeing it as a reference implementation of a high performance Large Language Models and wants to build their own LLMs on TPU and GPU.
- Any individual or a company that is interested in performing a pre-training or Full Fine Tuning of the supported open source models, can use Maxtext as a blackbox to perform full fine tuning. Maxtext attains an extremely high MFU, resulting in large savings in training costs.

## Runtime Performance Results

More details on reproducing these results can be found in [MaxText/configs/README.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/README.md).

### TPU v5p

| No. of params | Accelerator Type | TFLOP/chip/sec | Model flops utilization (MFU) |
|---|---|---|---|
| 32B | v5p-128 | 3.28e+02 | 71.47% |
| 64B | v5p-128 | 3.23e+02 | 70.31% |
| 128B | v5p-256 | 3.15e+02 | 68.68% |
| 128B | v5p-512 | 3.15e+02 | 68.53% |
| 256B | v5p-1024 | 3.16e+02 | 68.82% |
| 512B | v5p-1024 | 2.94e+02 | 63.99% |
| 1024B | v5p-2048 | 2.49e+02 | 64.05% |
| 1024B | v5p-4096 | 2.97e+02 | 64.80% |
| 1160B | v5p-7680 | 2.95e+02 | 64.27% |
| 1160B | v5p-12288 | 3.04e+02 | 66.23% |

### TPU v5e

For 16B, 32B, 64B, and 128B models. See full run configs in [MaxText/configs/v5e/](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/v5e/) as `16b.sh`, `32b.sh`, `64b.sh`, `128b.sh`.

| Hardware    | 16B TFLOP/sec/chip | 16B MFU | 32B TFLOP/sec/chip | 32B MFU | 64B TFLOP/sec/chip | 64B MFU | 128B TFLOP/sec/chip | 128B MFU |
| ----------- | -----------------: | ------- | -----------------: | ------- | -----------------: | ------- | ------------------: | -------- |
| 1x v5e-256  | 120                | 61.10%  | 132                | 66.86%  | 118                | 59.90%  | 110                 | 56.06%   |
| 2x v5e-256  | 117                | 59.37%  | 128                | 64.81%  | 112                | 56.66%  | 110                 | 55.82%   |
| 4x v5e-256  | 117                | 59.14%  | 126                | 64.10%  | 110                | 55.85%  | 108                 | 54.93%   |
| 8x v5e-256  | 115                | 58.27%  | 125                | 63.67%  | 108                | 54.96%  | 104                 | 52.93%   |
| 16x v5e-256 | 111                | 56.56%  | 123                | 62.26%  | 105                | 53.29%  | 100                 | 50.86%   |
| 32x v5e-256 | 108                | 54.65%  | 119                | 60.40%  | 99                 | 50.18%  | 91                  | 46.25%   |




```{toctree}
:maxdepth: 1
:hidden:

getting_started/index.md
code_organization.md
concepts.md
guides.md
terminologies.md
```
