<!--
 Copyright 2023-2025 Google LLC

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

# MaxText release notes

## PyPI Package

MaxText is [available in PyPI](https://pypi.org/project/maxtext/) and can be installed through pip. Please see our [MaxText Installation Guide](install_maxtext.md) for setup instructions.

## Releases

### v0.2.2

#### Changes

- Upgraded JAX to version 0.9.2, improving support for both pre-training and post-training.
- Introduced simplified APIs for accessing MaxText models.
- Included [maxtext_with_gepa.ipynb](https://github.com/AI-Hypercomputer/maxtext/blob/3c7d8d27864fc12cccac07786f02bd0e5262c982/src/maxtext/examples/maxtext_with_gepa.ipynb), a new notebook demonstrating AIME prompt optimization using the GEPA framework within MaxText.
- Added support for Kimi-K2 models and the MuonClip optimizer. Users can explore this with the [kimi-k2-1t](https://github.com/AI-Hypercomputer/maxtext/blob/fa5b5ebf9a8e4f7a33bd88eae051dc21f3147791/src/maxtext/configs/models/kimi-k2-1t.yml) config (see [user guide](https://github.com/AI-Hypercomputer/maxtext/blob/fa5b5ebf9a8e4f7a33bd88eae051dc21f3147791/tests/end_to_end/tpu/kimi/Run_Kimi.md) for details).
- Kimi-K2-Thinking, Kimi-K2.5 (text), and Kimi-K2.6 (text) are now supported. See [Run_Kimi.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md#quantized-variants-k2-thinking-k25-k26) for details.
- [DeepSeek-V3.2](https://arxiv.org/pdf/2512.02556) is now supported, including DeepSeek Sparse Attention for handling long contexts. Use the [deepseek3.2-671b](https://github.com/AI-Hypercomputer/maxtext/blob/20d93f62a91899dbbb8f23562973d75104411d3a/src/maxtext/configs/models/deepseek3.2-671b.yml) config to try it out (refer to the [user guide](https://github.com/AI-Hypercomputer/maxtext/blob/20d93f62a91899dbbb8f23562973d75104411d3a/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md) for more information).
- Support has been added for Gemma 4 multi-modal models (26B MoE and 31B dense). These can be used with the [gemma4-26b](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/src/maxtext/configs/models/gemma4-26b.yml) and [gemma4-31b](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/src/maxtext/configs/models/gemma4-31b.yml) configs. See [Run_Gemma4.md](https://github.com/AI-Hypercomputer/maxtext/blob/cdc587f0935a5e2d6f8287b96669cf2e87a0acdc/tests/end_to_end/tpu/gemma4/Run_Gemma4.md) for further details.
- Support has been added for Gemma 4 inference using [MaxText on vLLM plugin](tutorials/inference.md).
- Enhanced RL capabilities with support for the `open-r1/OpenR1-Math-220k` dataset and `nvidia/OpenMathReasoning`.
- Added more evaluation modes for RL like majority voting and pass@1 estimation.
- Sync weights to vllm prior to pre RL evaluation.
- More robust usage of math-verify in RL.
- MaxText's Supervised Fine-Tuning (SFT) now supports non-instruct models.
- Added support for tensor parallelism using the Fused MoE kernel for MaxText on vLLM inference.
- Added support for MaxText to vllm converters for Qwen3 and Gemma4 family of models.
- [validate_converter.py](https://github.com/AI-Hypercomputer/maxtext/blob/472f53b70089e661be399ad3905c05a53a172ec5/src/maxtext/integration/vllm/torchax_converter/validate_converter.py#L108) now runs on multislice environment to test larger models with utilities to compare maxtext and vllm weights.

#### Deprecations

- Legacy `MaxText.*` shims have been removed. Please refer to [src/MaxText/README.md](https://github.com/AI-Hypercomputer/maxtext/blob/0536605a8ca116087ed93178433a67e905be566c/src/MaxText/README.md) for details on the new command locations and how to migrate.
- Sequence parallelism has been deprecated, please use context parallelism instead.
- The flag `expert_shard_attention_option` is deprecated, use `custom_mesh_and_rule=ep-as-cp` for the same functionality.

### v0.2.1

#### Changes

- Use the new `maxtext[runner]` installation option to build Docker images without cloning the repository. This can be used for scheduling jobs through XPK. See the [MaxText installation instructions](build_maxtext.md) for more info.
- Config can now be inferred for most MaxText commands. If you choose not to provide a config, MaxText will now [select an appropriate one](https://github.com/AI-Hypercomputer/maxtext/blob/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/pyconfig.py#L51-L67).
- Configs in MaxText PyPI will now be picked up without storing them locally.
- New features from DeepSeek-AI are now supported: Conditional Memory via Scalable Lookup ([Engram](https://arxiv.org/abs/2601.07372)) and Manifold-Constrained Hyper-Connections ([mHC](https://arxiv.org/abs/2512.24880)). Try them out with our [deepseek-custom](https://github.com/AI-Hypercomputer/maxtext/blob/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/models/deepseek-custom.yml) starter config.
- MaxText now supports customizing your own mesh and logical rules. Two examples guiding how to use your own mesh and rules for sharding are provided in the [custom_mesh_and_rule](https://github.com/AI-Hypercomputer/maxtext/tree/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/custom_mesh_and_rule) directory.

### v0.2.0

#### Changes

- New `tpu-post-train` target in PyPI. Please also use this installation option for running vllm_decode. See the [MaxText installation instructions](install_maxtext.md) for more info.
- [Qwen3-Next](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/tests/end_to_end/tpu/qwen/next/run_qwen3_next.md) is now supported.
- New MaxText structure! MaxText has been restructured according to [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/1b9e38aa0a19b6018feb3aed757406126b6953a1/RESTRUCTURE.md). Please feel free to share your thoughts and feedback.
- [Muon optimizer](https://kellerjordan.github.io/posts/muon) is now supported.
- DeepSeek V3.1 is now supported. Use existing configs for [DeepSeek V3 671B](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/configs/models/deepseek3-671b.yml) and load in V3.1 checkpoint to use model.
- [New RL and SFT Notebook tutorials](https://github.com/AI-Hypercomputer/maxtext/tree/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/examples) are available.
- The [ReadTheDocs documentation site](index.md) has been reorganized.
- Multi-host support for GSPO and GRPO is now available via [new RL tutorials](tutorials/posttraining/rl_on_multi_host.md).
- A new guide, [What is Post Training in MaxText?](tutorials/post_training_index.md), is now available.
- Ironwood TPU co-designed AI stack announced. Read the [blog post on its co-design with MaxText](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack?e=48754805).
- [Optimized models tiering documentation](reference/models/tiering.md) has been refreshed.
- Added Versioning. Check out our [first set of release notes](release_notes.md)!
- Post-Training (SFT, RL) via [Tunix](https://github.com/google/tunix) is now available.
- Vocabulary tiling ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/2242)) is now supported in MaxText! Adjust config `num_vocab_tiling` to unlock more efficient memory usage.
- The GPT-OSS family of models (20B, 120B) is now supported.

#### Deprecations

- Many MaxText modules have changed locations. Core commands like train, decode, sft, etc. will still work as expected temporarily. Please update your commands to the latest file locations
- install_maxtext_github_deps installation script replaced with install_maxtext_tpu_github_deps
- `tools/setup/setup_post_training_requirements.sh` for post training dependency installation is deprecated in favor of [pip installation](install_maxtext.md)

### v0.1.0

Our first MaxText PyPI package is here! MaxText is a high performance, highly scalable, open-source LLM library and reference implementation written in pure Python/JAX and targeting Google Cloud TPUs and GPUs for training. We are excited to make it easier than ever to get started.

Users can now install MaxText through pip, both for local development and through stable PyPI builds. Please see our [MaxText Installation Guide](install_maxtext.md) for more setup details.

Going forward, this page will document notable changes as we release new versions of MaxText.
