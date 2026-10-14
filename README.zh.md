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

[![MaxText Tests](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/ci_pipeline.yml/badge.svg?event=schedule)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/ci_pipeline.yml)
[![PyPI version](https://img.shields.io/pypi/v/maxtext)](https://pypi.org/project/maxtext/)

[**官方文档 (Documentation)**](https://maxtext.readthedocs.io/en)
| [**版本说明 (Release Notes)**](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/release_notes.md)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

> **_注意：_** 我们建议用户优先使用最新的 [PyPI 发布版本](https://pypi.org/project/maxtext/)（请参阅[安装指南](docs/install_maxtext.md)）或对应的[预构建容器镜像](docs/tutorials/build_maxtext.md#pre-built-maxtext-docker-images)；代码库中的 `main` 分支主要面向持续集成与开发，不保证处于生产就绪状态。
>
> 推荐在 Python 3.12 环境下运行 MaxText，这是我们官方主要支持的版本。其他 Python 版本可能会遇到环境兼容性问题。

MaxText 是一个采用纯 Python/[JAX](https://docs.jax.dev/en/latest/jax-101.html) 编写的高性能、高可扩展性开源大语言模型（LLM）库与参考实现，专为 Google Cloud TPU 和 GPU 大规模训练深度打造。

MaxText 提供了丰富的开箱即用高性能模型库，涵盖 Gemma、Llama、DeepSeek、Qwen 和 Mistral 等主流架构。针对这些模型，MaxText 全面支持超大规模预训练（最高可扩展至数万颗加速芯片集群）与可扩展后训练（Post-Training），深度集成了业界前沿的对齐算法，包括监督微调（SFT）、组相对策略优化（GRPO，一种先进强化学习对齐算法）以及组序列策略优化（GSPO，一种先进强化学习对齐算法）。

得益于 JAX 与 XLA 编译器的深度优化能力，MaxText 在从单机单卡到超大规模异构计算集群的全场景下均能保持极高的模型浮点运算利用率（MFU）与吞吐量（tokens/second），同时代码逻辑清晰，大幅免去了繁重的底层手工优化。

无论是前沿科研探索还是工业级生产落地，MaxText 都是构建雄心勃勃的 LLM 项目的理想起点。我们鼓励您首先体验开箱即用的 MaxText，随后根据具体的业务或研究需求自由 Fork 并定制扩展。

欢迎查阅我们的 [Read The Docs 文档站点](https://maxtext.readthedocs.io/en) 或直接按照[快速入门指南](https://maxtext.readthedocs.io/page/getting_started.html)开启您的第一次 MaxText 训练任务。如果您对扩散生成模型（如 Wan 2.1、Flux 等）感兴趣，请访问我们 AI Hypercomputer 组织下的 [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) 代码库。

## 安装指南

请参阅我们的安装文档，了解如何[通过 PyPI 使用 pip 安装 MaxText](https://maxtext.readthedocs.io/page/install_maxtext.html#from-pypi-recommended)。

## 解耦模式 (Decoupled Mode)

有关在脱离任何 GCP 依赖的环境下以完全解耦模式运行 MaxText 的详细指导，请参阅[解耦模式指南](https://maxtext.readthedocs.io/page/run_maxtext/decoupled_mode.html)。

<!-- NEWS START -->

## 🔥 最新动态 🔥

- [2026年10月8日] MaxText 现已通过 vLLM 支持 TPU 上的 JEV 风格决策打分；详情请查阅[决策模型教程 (Decision Models tutorial)](https://maxtext.readthedocs.io/en/latest/tutorials/decision_models.html)，包含基于 Bespoke-Nimble LoRA 权重的 Qwen3.5-9B 示例。
- [2026年8月19日] MaxText 正式支持基于 **Streaming DiLoCo** 的分布式低通信开销训练！可跨越多种切片 TPU 集群以及低带宽 DCN/WAN 网络高效扩展 LLM 与 MoE 模型。详情请参阅 [DiLoCo 训练教程](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/diloco_pretraining.html) 与 [DiLoCo 理论参考](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/diloco.html)。
- [2026年8月5日] 正式支持 DeepSeek V4 Flash (284B)。权重转换与评测详情请参阅[用户指南](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md)。
- [2026年8月3日] 面向 MoE 算子的 Tokamax GMM v2 现已发布。开启方式请参阅 [MoE 配置指南](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md)。
- [2026年7月15日] 现已支持针对 Qwen3 30B 与 GPT-OSS 20B 的强化学习（RL）训练。详情请参阅 [Qwen3 30B RL 教程](https://maxtext.readthedocs.io/en/maxtext-v0.2.4/tutorials/posttraining/rl_qwen3_30b.html) 与 [GPT-OSS 20B RL 教程](https://maxtext.readthedocs.io/en/maxtext-v0.2.4/tutorials/posttraining/rl_gptoss_20b.html)。
- [2026年6月29日] MaxText 已全面从 Linen 迁移至 [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html)。现有训练任务可照常无缝运行。如遇任何问题，请[向我们反馈](https://github.com/AI-Hypercomputer/maxtext/issues/new?template=bug_report.yml)。
- [2026年6月9日] 正式[支持](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/qwen/moe/run_qwen_moe.md) Qwen3.5 35B 与 397B 模型。
- [2026年5月29日] 发布全新评测框架，支持基于 MaxText 检查点直接运行 lm-eval、evalchemy 以及自定义基准评测。请查阅[评测指南](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/guides/eval_framework.html)快速上手。
- [2026年5月6日] 正式支持 Kimi-K2-Thinking、Kimi-K2.5（文本）以及 Kimi-K2.6（文本）。详情请参阅 [Run_Kimi.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md#quantized-variants-k2-thinking-k25-k26)。
- [2026年4月18日] 新增 Notebook 教程 [maxtext_with_gepa.ipynb](https://github.com/AI-Hypercomputer/maxtext/blob/3c7d8d27864fc12cccac07786f02bd0e5262c982/src/maxtext/examples/maxtext_with_gepa.ipynb)，展示如何结合 MaxText 与 GEPA 框架优化 AIME 竞赛提示词。
- [2026年4月14日] 已移除历史遗留的 `MaxText.*` 后训练 shim 适配层。有关新命令路径与迁移指南，请参阅 [src/MaxText/README.md](https://github.com/AI-Hypercomputer/maxtext/blob/0536605a8ca116087ed93178433a67e905be566c/src/MaxText/README.md)。
- [2026年4月13日] 正式支持 Kimi-K2 模型以及 MuonClip 优化器。欢迎尝试 [kimi-k2-1t](https://github.com/AI-Hypercomputer/maxtext/blob/fa5b5ebf9a8e4f7a33bd88eae051dc21f3147791/src/maxtext/configs/models/kimi-k2-1t.yml) 配置并查阅[用户指南](https://github.com/AI-Hypercomputer/maxtext/blob/fa5b5ebf9a8e4f7a33bd88eae051dc21f3147791/tests/end_to_end/tpu/kimi/Run_Kimi.md)。

<details>
<summary><b>点击展开完整历史更新归档</b></summary>

- [2026年4月10日] 正式支持 [DeepSeek-V3.2](https://arxiv.org/pdf/2512.02556)，引入 DeepSeek 稀疏注意力（DSA）以支持超长上下文。可使用 [deepseek3.2-671b](https://github.com/AI-Hypercomputer/maxtext/blob/20d93f62a91899dbbb8f23562973d75104411d3a/src/maxtext/configs/models/deepseek3.2-671b.yml) 配置进行体验。更多细节请参阅[用户指南](https://github.com/AI-Hypercomputer/maxtext/blob/20d93f62a91899dbbb8f23562973d75104411d3a/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md)。
- [2026年4月2日] 正式支持 Gemma 4 多模态模型（26B MoE 与 31B Dense）！可通过 [gemma4-26b](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/src/maxtext/configs/models/gemma4-26b.yml) 与 [gemma4-31b](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/src/maxtext/configs/models/gemma4-31b.yml) 配置运行。详情请参阅 [Run_Gemma4.md](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/tests/end_to_end/tpu/gemma4/Run_Gemma4.md)。
- [2026年3月6日] 支持来自 DeepSeek-AI 的全新特性：基于可扩展查找的条件记忆 ([Engram](https://arxiv.org/abs/2601.07372)) 与流形约束超连接 ([mHC](https://arxiv.org/abs/2512.24880))。欢迎尝试 [deepseek-custom](https://github.com/AI-Hypercomputer/maxtext/blob/92f55e3d8123e225c2714a9bac40dd0a7f89efb0/src/maxtext/configs/models/deepseek-custom.yml) 快速入门配置。
- [2026年3月5日] PyPI 上新增 `tpu-post-train` [安装目标](https://pypi.org/project/maxtext)。运行 vllm_decode 时亦建议采用该选项。详情请查阅 [MaxText 安装说明](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/install_maxtext.html)。
- [2026年3月5日] 正式支持 [Qwen3-Next](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/tests/end_to_end/tpu/qwen/next/run_qwen3_next.md)。
- [2026年2月27日] 全新 MaxText 代码架构！根据 [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/1b9e38aa0a19b6018feb3aed757406126b6953a1/RESTRUCTURE.md) 完成项目结构重构，欢迎分享您的体验与反馈。
- [2025年12月22日] 正式支持 [Muon 优化器](https://kellerjordan.github.io/posts/muon)。
- [2025年12月10日] 正式支持 DeepSeek V3.1。可直接使用 [DeepSeek V3 671B](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/configs/models/deepseek3-671b.yml) 现有配置并加载 V3.1 检查点使用。
- [2025年12月9日] 发布[全新强化学习（RL）与监督微调（SFT）Notebook 教程](https://github.com/AI-Hypercomputer/maxtext/tree/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/examples)。
- [2025年12月4日] [ReadTheDocs 文档站点](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/index.html)结构全面优化调整。
- [2025年12月3日] 通过[全新 RL 教程](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/tutorials/posttraining/rl_on_multi_host.html)支持针对 GSPO 与 GRPO 的多机多主机（Multi-host）分布式训练。
- [2025年11月20日] 全新指导手册[什么是 MaxText 中的后训练？](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/tutorials/post_training_index.html)正式上线。
- [2025年11月6日] Google 发布与 Ironwood TPU 联合设计的 AI 软件栈。欢迎阅读[关于其与 MaxText 联合设计的深度博客](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack?e=48754805)。
- [2025年10月29日] [优化模型分级文档](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/reference/models/tiering.html)已完成全面更新。
- [2025年10月12日] 引入正式版本控制。欢迎查看我们的[首份发布说明 (Release Notes)](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/release_notes.html)！
- [2025年10月10日] 正式支持通过 [Tunix](https://github.com/google/tunix) 进行后训练（SFT、RL）。
- [2025年9月26日] MaxText 正式支持词表分块优化（Vocabulary Tiling, [PR #2242](https://github.com/AI-Hypercomputer/maxtext/pull/2242)）！调节配置项 `num_vocab_tiling` 可显著提升显存利用效率。
- [2025年9月24日] 正式支持 GPT-OSS 系列模型（20B 与 120B）。
- [2025年9月15日] MaxText 正式以 [PyPI 软件包](https://pypi.org/project/maxtext)形式发布。用户现可直接[通过 pip 安装 MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/install_maxtext.html)。
- [2025年9月5日] 作为架构重构的一部分，MaxText 全面切换至 `src` 代码布局。对于现有开发环境，请在根目录下执行 `pip install -e .`。
- [2025年8月13日] 正式支持 Qwen3 2507 MoE 系列模型：MoE（235B Thinking 与 480B Coder）以及全系列稠密模型（0.6B、4B、8B、14B 与 32B）。
- [2025年7月27日] 更新 TFLOPS/s 计算逻辑（[PR #1988](https://github.com/AI-Hypercomputer/maxtext/pull/1988)），计入因果注意力机制并将注意力浮点运算减半；并在 [PR #2009](https://github.com/AI-Hypercomputer/maxtext/pull/2009) 与 [PR #2030](https://github.com/AI-Hypercomputer/maxtext/pull/2030) 中考虑滑动窗口与分块注意力带来的计算量缩减。详情可参阅[性能指标文档](https://maxtext.readthedocs.io/en/maxtext-v0.2.0/reference/performance_metrics.html)。
- [2025年7月16日] 启动 MaxText 代码库目录优化调整，详情请参阅[重构架构规划](https://github.com/AI-Hypercomputer/maxtext/blob/aca5b24931ebcbadb55a82e56ebffe8024874028/RESTRUCTURE.md)。
- [2025年7月11日] 多 Token 预测（Multi-Token Prediction, MTP）训练支持！受 [DeepSeek-V3 论文](https://arxiv.org/html/2412.19437v1)启发增加辅助损失函数以预测未来多个 token，大幅增强训练效率。
- [2025年6月25日] 正式支持 DeepSeek R1-0528 变体模型。
- [2025年4月24日] 正式支持 Llama 4 Maverick 系列模型。

</details>
<!-- NEWS END -->

## 应用场景 (Use cases)

MaxText 提供了丰富的模型库，并演示了如何在大规模集群上以极致性能执行预训练或后训练任务。

MaxText 深度依托 [JAX AI 软件生态库](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)，并通过无缝整合 [Flax](https://flax.readthedocs.io/en/latest/)（神经网络构建）、[Tunix](https://github.com/google/tunix)（后训练）、[Orbax](https://orbax.readthedocs.io/en/latest/)（高效检查点保存与恢复）、[Optax](https://optax.readthedocs.io/en/latest/)（优化器与调度）以及 [Grain](https://google-grain.readthedocs.io/en/latest/)（高性能数据加载），构建起一套完备的大规模模型训练系统。

除纯文本 LLM 外，我们还全面支持基于 Gemma 3、Gemma 4 以及 Llama 4 VLM 的多模态（Vision-Language）训练。

### 预训练 (Pre-training)

如果您需要从零开始训练基座模型，MaxText 可作为极佳的参考基准——您可以直接 Fork 并针对性修改 MaxText 代码，无论是类似 Llama 8B 的轻量稠密模型，还是类似 DeepSeek-V3 的大规模 MoE 模型均可得心应手。您可以灵活探索模型架构与系统配置，在 TPU 或 GPU 上构建出最具计算效率的专属模型。

MaxText 针对切分（Sharding）、量化（Quantization）以及检查点（Checkpointing）等多个系统核心维度提供了经过大量生产环境检验的最佳实践实现方案。

### 后训练 (Post-training)

无论是私有商业模型还是开源社区权重，MaxText 均可通过 Tunix 提供高效可扩展的后训练框架支持。对于强化学习（如 GRPO），我们借助 vLLM 实现高吞吐采样，并即将接入 Pathways 实现大规模多主机扩展。

我们的目标是提供丰富的模型维度（维度 “a”）与算法技术（维度 “b”），让开发者能够轻松探索 (a) * (b) 的丰富组合，高效训练契合实际落地场景的理想模型。

欢迎参阅以下快速入门指南：

- 监督微调 (SFT)
  - [单机 TPU SFT 指南](https://maxtext.readthedocs.io/page/tutorials/posttraining/sft.html)
  - [多机 TPU SFT 指南](https://maxtext.readthedocs.io/page/tutorials/posttraining/sft_on_multi_host.html)
- 强化学习 (RL)
  - [单机 TPU RL 指南](https://maxtext.readthedocs.io/page/tutorials/posttraining/rl.html)
  - [多机 TPU RL 指南](https://maxtext.readthedocs.io/page/tutorials/posttraining/rl_on_multi_host.html)

### 支持的 JAX 模型库 (Model library)

MaxText 致力于为您带来业界前沿的开源大模型，无论是作为科研参考实现，还是用于后训练并通过 vLLM 提供推理服务：

- **Google**
  - Gemma 4 (26B MoE, 31B Dense)
  - Gemma 3 (4B, 12B, 27B)
  - Gemma 2 (2B, 9B, 27B)
  - Gemma 1 (2B, 7B)
- **Alibaba**
  - Qwen 3.5 MoE (35B, 397B)
  - Qwen 3 Next (80B)
  - Qwen 3 MoE (30B, 235B), Qwen 3 MoE 2507 (235B, 480B)
  - Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
  - Qwen 2.5 (1.5B, 7B, 14B)
- **DeepSeek AI**
  - DeepSeek V3.2 (671B)
  - DeepSeek V3.1 (671B)
  - DeepSeek V3 0324 (671B), DeepSeek R1 0528 (671B)
  - DeepSeek V2 (16B, 236B)
- **Moonshot AI (月之暗面)**
  - Kimi K2 (1T), Kimi K2-Thinking (1T)
  - Kimi K2.5 (文本), Kimi K2.6 (文本)
- **Meta**
  - Llama 4 Scout (109B) & Maverick (400B)
  - Llama 3.3 (70B), 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
  - Llama 2 (7B, 13B, 70B)
- **OpenAI**
  - GPT-OSS (20B, 120B)
  - GPT3 (52K, 6B, 22B, 175B)
- **Mistral AI**
  - Mixtral (8x7B, 8x22B)
  - Mistral (7B)
- **扩散生成模型 (Diffusion Models)**
  - 请参阅 [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)（LTXV、Wan 2.1、Flux、SDXL 等）

## 参与贡献与社区交流

欢迎加入我们的 [Discord 社区频道](https://discord.com/invite/2H9PhvTcDU)。如果您有任何需求反馈、文档建议或 Bug 报告，欢迎随时[在此提交 Issue](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)。

## 开源许可证

本项目基于 [Apache License 2.0](LICENSE) 开源许可证授权。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月14日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
