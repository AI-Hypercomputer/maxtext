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

MaxText is a a Google initiated open source project for **high performance**, **highly scalable**, **open-source** LLM written in pure Python/[JAX](https://jax.readthedocs.io/en/latest/index.html) and targeting Google Cloud TPUs and GPUs for **training** and **inference**. MaxText achieves [high MFUs](https://github.com/AI-Hypercomputer/src/MaxText/blob/main/README.md#runtime-performance-results) and scales from single host to very large clusters while staying simple and "optimization-free" thanks to the power of Jax and the XLA compiler.

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
- [Open code base](https://github.com/AI-Hypercomputer/src/MaxText)
- Easy to understand: MaxText is purely written in JAX and Python, which makes it accessible to ML developers interested in inspecting the implementation or stepping through it. It is written at the [block-by-block](https://github.com/AI-Hypercomputer/src/MaxText/tree/main/MaxText/layers) level, with code for Embeddings, Attention, Normalization etc. Different Attention mechanisms like MQA and GQA are all present. For quantization, it uses the [JAX AQT](https://github.com/google/aqt) library. The implementation is suitable for both GPUs and TPUs.

```{note}
Maxtext today only supports Pre-training and Full Fine Tuning of the models. It does not support PEFT/LoRA, Supervised Fine Tuning or RLHF.
```

## Who are the target users of MaxText?

- Any individual or a company that is interested in forking src/MaxText and seeing it as a reference implementation of a high performance Large Language Models and wants to build their own LLMs on TPUs or GPUs.
- Any individual or a company that is interested in performing a pre-training or Full Fine Tuning of the supported open source models, can use Maxtext as a blackbox to perform full fine tuning. Maxtext attains an extremely high MFU, resulting in large savings in training costs.

## Learn more

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card}
:link: full-finetuning
:link-type: ref
:class-card: sd-text-black sd-bg-light

{material-regular}`settings;2em` Full finetuning and training with Llama3
:::

:::{grid-item-card}
:link: first-run
:link-type: ref
:class-card: sd-text-black sd-bg-light

{material-regular}`rocket_launch;2em` First run
:::
::::

## Code repository

You can find the latest version of MaxText at https://github.com/AI-Hypercomputer/src/MaxText

## In-depth documentation

You can find in-depth documentation at [the MaxText GitHub repository](https://github.com/AI-Hypercomputer/src/MaxText/blob/main/docs/advanced_docs/).


```{toctree}
:maxdepth: 2
:hidden:

tutorials.md
guides.md
explanations.md
reference.md
```
