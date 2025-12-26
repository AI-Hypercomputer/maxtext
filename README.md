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

[![MaxText Tests](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml)

> **_NOTE:_** We recommend running MaxText with Python 3.12, as it is our primary supported version. Other Python versions may encounter compatibility issues.

MaxText is a high performance, highly scalable, open-source LLM library and reference implementation written in pure Python/[JAX](https://docs.jax.dev/en/latest/jax-101.html) and targeting Google Cloud TPUs and GPUs for training.

MaxText provides a library of high performance models to choose from, including Gemma, Llama, DeepSeek, Qwen, and Mistral. For each of these models, MaxText supports pre-training (up to tens of thousands of chips) and scalable post-training, with popular techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO, a type of Reinforcement Learning) and Group Sequence Policy Optimization (GSPO, a type of Reinforcement Learning).

MaxText achieves high Model FLOPs Utilization (MFU) and tokens/second from single host to very large clusters while staying simple and largely "optimization-free" thanks to the power of JAX and the XLA compiler.

MaxText is the launching point for ambitious LLM projects both in research and production. We encourage you to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet your needs.

Check out our [Read The Docs site](https://maxtext.readthedocs.io/en/latest/) or directly [Get Started](https://maxtext.readthedocs.io/en/latest/tutorials/first_run.html) with your first MaxText run. If you’re interested in Diffusion models (Wan 2.1, Flux, etc), see the [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) repository in our AI Hypercomputer GitHub organization.

## Installation

See our installation guide to [install MaxText with pip from PyPI](https://maxtext.readthedocs.io/en/latest/install_maxtext.html#from-pypi-recommended).

## Decoupled mode
See our guide on running MaxText in decoupled mode, without any GCP dependencies in [Decoupled Mode Guide](https://maxtext.readthedocs.io/en/latest/run_maxtext/decoupled_mode.html).

<!-- NEWS START -->

## 🔥 Latest news 🔥

* \[December 22, 2025\] [Muon optimizer](https://kellerjordan.github.io/posts/muon) is now supported.
* \[December 10, 2025\] DeepSeek V3.1 is now supported. Use existing configs for [DeepSeek V3 671B](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/models/deepseek3-671b.yml) and load in V3.1 checkpoint to use model.
* \[December 9, 2025\] [New RL and SFT Notebook tutorials](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/examples) are available.
* \[December 4, 2025\] The [ReadTheDocs documentation site](https://maxtext.readthedocs.io/en/latest/index.html) has been reorganized.
* \[December 3, 2025\] Multi-host support for GSPO and GRPO is now available via [new RL tutorials](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_on_multi_host.html).
* \[November 20, 2025\] A new guide, [What is Post Training in MaxText?](https://maxtext.readthedocs.io/en/latest/tutorials/post_training_index.html), is now available.
* \[November 6, 2025\] Ironwood TPU co-designed AI stack announced. Read the [blog post on its co-design with MaxText](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack?e=48754805).
* \[October 29, 2025\] [Optimized models tiering documentation](https://maxtext.readthedocs.io/en/latest/reference/models/tiering.html) has been refreshed.
* \[October 12, 2025\] Added Versioning. Check out our [first set of release notes](https://maxtext.readthedocs.io/en/latest/release_notes.html)!
* \[October 10, 2025\] Post-Training (SFT, RL) via [Tunix](https://github.com/google/tunix) is now available.
* \[September 26, 2025\] Vocabulary tiling ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/2242)) is now supported in MaxText! Adjust config `num_vocab_tiling` to unlock more efficient memory usage.
* \[September 24, 2025\] The GPT-OSS family of models (20B, 120B) is now supported.
* \[September 15, 2025\] MaxText is now available as a [PyPI package](https://pypi.org/project/maxtext). Users can now [install maxtext through pip](https://maxtext.readthedocs.io/en/latest/guides/install_maxtext.html).
* \[September 5, 2025\] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/aca5b24931ebcbadb55a82e56ebffe8024874028/RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
* \[August 13, 2025\] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 480B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
* \[July 27, 2025\] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://maxtext.readthedocs.io/en/latest/reference/performance_metrics.html)
* \[July 16, 2025\] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/aca5b24931ebcbadb55a82e56ebffe8024874028/RESTRUCTURE.md) and provide feedback.
* \[July 11, 2025\] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
* \[June 25, 2025\] DeepSeek R1-0528 variant is now supported.
* \[April 24, 2025\] Llama 4 Maverick models are now supported.
<!-- NEWS END -->

## Use cases

MaxText provides a library of models and demonstrates how to perform pre-training or post-training with high performance and scale.

MaxText leverages [JAX AI libraries](https://docs.jaxstack.ai/en/latest/getting_started.html) and presents a cohesive and comprehensive demonstration of training at scale by using [Flax](https://flax.readthedocs.io/en/latest/) (neural networks), [Tunix](https://github.com/google/tunix) (post-training), [Orbax](https://orbax.readthedocs.io/en/latest/) (checkpointing), [Optax](https://optax.readthedocs.io/en/latest/) (optimization), and [Grain](https://google-grain.readthedocs.io/en/latest/) (dataloading).

In addition to pure text-based LLMs, we also support multi-modal training with Gemma 3 and Llama 4 VLMs.

### Pre-training

If you’re building models from scratch, MaxText can serve as a reference implementation for experimentation, ideation, and inspiration \- just fork and modify MaxText to train your model, whether it’s a small dense model like Llama 8B, or a large MoE like DeepSeek-V3. Experiment with configs and model design to build the most efficient model on TPU or GPU.

MaxText provides opinionated implementations for how to achieve optimal performance across a wide variety of dimensions like sharding, quantization, and checkpointing.

### Post-training

If you are post-training a model, whether it is proprietary or open source, MaxText provides a scalable framework using Tunix. For RL (like GRPO), we leverage vLLM for sampling and Pathways (soon) for multi-host.

Our goal is to provide a variety of models (dimension “a”) and techniques (dimension “b”), so you can easily explore (a) \* (b) combinations and efficiently train the perfect model for your use case.

Check out these getting started guides:

* Supervised Fine Tuning (SFT)
  * [SFT on Single-Host TPUs](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft.html)
  * [SFT on Multi-Host TPUs](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft_on_multi_host.html)
* Reinforcement Learning (RL)
  * [RL on Single-Host TPUs](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html)
  * [RL on Multi-Host TPUs](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_on_multi_host.html)

### Model library

MaxText aims to provide you with the best OSS models, whether as a reference implementation, or to post-train and then serve with vLLM.

**Supported JAX models in MaxText**

* Google
  * Gemma 3 (4B, 12B, 27B)
  * Gemma 2 (2B, 9B, 27B)
  * Gemma 1 (2B, 7B)
* Alibaba
  * Qwen 2.5 (7B, 14B)
  * Qwen 3 MoE 2507 (235B, 480B)
  * Qwen 3 MoE (30B, 235B)
  * Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
* DeepSeek
  * DeepSeek V3.1 (671B)
  * DeepSeek V3 0324 (671B) & DeepSeek R1 0528 (671B)
  * DeepSeek V2 (16B, 236B)
* Meta
  * Llama 4 Scout (109B) & Maverick (400B)
  * Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
  * Llama 2 (7B, 13B, 70B)
* Open AI
  * GPT-OSS (20B, 120B)
  * GPT3 (52K, 6B, 22B, 175B)
* Mistral
  * Mixtral (8x7B, 8x22B)
  * Mistral (7B)
* Diffusion Models
  * See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (LTXV, Wan 2.1, Flux, SDXL, etc)

## Get involved

Please join our [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and if you have feedback, you can file a feature request, documentation request, or bug report [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).

## License

[Apache License 2.0](LICENSE)
