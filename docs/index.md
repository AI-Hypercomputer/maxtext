<!--
 # Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# MaxText

MaxText is a high performance, highly scalable, open-source LLM library and reference implementation written in pure Python/[JAX](https://docs.jax.dev/en/latest/jax-101.html) and targeting Google Cloud TPUs and GPUs for training.

MaxText provides a library of high performance models to choose from, including Gemma, Llama, DeepSeek, Qwen, and Mistral. For each of these models, MaxText supports pre-training (up to tens of thousands of chips) and scalable post-training, with popular techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO, a type of Reinforcement Learning).

MaxText achieves high Model FLOPs Utilization (MFU) and tokens/second from single host to very large clusters while staying simple and largely "optimization-free" thanks to the power of JAX and the XLA compiler.

MaxText is the launching point for ambitious LLM projects both in research and production. We encourage you to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet your needs.

Check out our [Read The Docs site](https://maxtext.readthedocs.io/en/latest/) or directly [Get Started](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/tutorials/first_run.md) with your first MaxText run. If you’re interested in Diffusion models (Wan 2.1, Flux, etc), see the [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) repository in our AI Hypercomputer GitHub organization.

## 🔥 Latest news 🔥

* [September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md]
* [August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
* [July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)
* [July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.
* [July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
* [June 25, 2025] DeepSeek R1-0528 variant is now supported
* [April 24, 2025] Llama 4 Maverick models are now supported

## Use cases

MaxText provides a library of models and demonstrates how to perform pre-training or post-training with high performance and scale. MaxText leverages [JAX AI libraries](https://docs.jaxstack.ai/en/latest/getting_started.html) and presents a cohesive and comprehensive demonstration of training at scale by using [Flax](https://flax.readthedocs.io/en/latest/) (neural networks), [Tunix](https://github.com/google/tunix) (post-training), [Orbax](https://orbax.readthedocs.io/en/latest/) (checkpointing), [Optax](https://optax.readthedocs.io/en/latest/) (optimization), and [Grain](https://google-grain.readthedocs.io/en/latest/) (dataloading). In addition to pure text-based LLMs, we also support multi-modal training with Gemma 3 and Llama 4 VLMs.

### Pre-training

If you’re building models from scratch, MaxText can serve as a reference implementation for experimentation, ideation, and inspiration \- just fork and modify MaxText to train your model, whether it’s a small dense model like Llama 8B, or a large MoE like DeepSeek-V3. Experiment with configs and model design to build the most efficient model on TPU or GPU.

MaxText provides opinionated implementations for how to achieve optimal performance across a wide variety of dimensions like sharding, quantization, and checkpointing.

### Post-training

If you are post-training a model, whether it is proprietary or open source, MaxText provides a scalable framework using Tunix. For RL (like GRPO), we leverage vLLM for sampling and Pathways (soon) for multi-host.

Our goal is to provide a variety of models (dimension “a”) and techniques (dimension “b”), so you can easily explore (a) \* (b) combinations and efficiently train the perfect model for your use case.

Check out these getting started guides:

* [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh) (Supervised Fine Tuning)
* [GRPO](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/grpo.md) (Group Relative Policy Optimization)

### Model library

MaxText aims to provide you with the best OSS models, whether as a reference implementation, or to post-train and then serve with vLLM.

We “tier” each model for how optimized it is in the MaxText framework for a given hardware platform. When a new model is initially added, it’s not yet optimized and then, as we see demand in the OSS ecosystem, we invest the resources to optimize performance (i.e., MFU & tokens/sec, etc).

**Supported JAX models in MaxText**

* Google
  * Gemma 3 (4B, 12B, 27B)
  * Gemma 2 (2B, 9B, 27B)
  * Gemma 1 (2B, 7B)
* Alibaba
  * Qwen 3 MoE 2507 (235B, 480B)
  * Qwen 3 MoE (30B, 235B)
  * Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
* DeepSeek
  * DeepSeek-V2 (16B, 236B)
  * DeepSeek-V3 0528 (671B)
* Meta
  * Llama 4 Scout (109B) & Maverick (400B)
  * Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
  * Llama 2 (7B, 13B, 70B)
* Open AI
  * GPT3 (52k, 6B, 22B, 175B)
* Mistral
  * Mixtral (8x7B, 8x22B)
  * Mistral (7B)
* Diffusion Models
  * See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Get involved

Please join our [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and if you have feedback, you can file a feature request, documentation request, or bug report [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).

```{toctree}
:maxdepth: 2
:hidden:

tutorials.md
guides.md
explanations.md
reference.md
development.md
```
