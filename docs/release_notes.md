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

MaxText is [available in PyPI](https://pypi.org/project/maxtext/) and can be installed through pip. Please see our [MaxText Installation Guide](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) for setup instructions.

## Releases

### v0.2.1

- Use the new `maxtext[runner]` installation option to build Docker images without cloning the repository. This can be used for scheduling jobs through XPK. See the [MaxText installation instructions](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/build_maxtext.html) for more info.
- Config can now be inferred for most MaxText commands. If you choose not to provide a config, MaxText will now [select an appropriate one](https://github.com/AI-Hypercomputer/maxtext/blob/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/pyconfig.py#L51-L67).
- Configs in MaxText PyPI will now be picked up without storing them locally.
- New features from DeepSeek-AI are now supported: Conditional Memory via Scalable Lookup ([Engram](https://arxiv.org/abs/2601.07372)) and Manifold-Constrained Hyper-Connections ([mHC](https://arxiv.org/abs/2512.24880)). Try them out with our [deepseek-custom](https://github.com/AI-Hypercomputer/maxtext/blob/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/models/deepseek-custom.yml) starter config.
- MaxText now supports customizing your own mesh and logical rules. Two examples guiding how to use your own mesh and rules for sharding are provided in the [custom_mesh_and_rule](https://github.com/AI-Hypercomputer/maxtext/tree/9e786c888cc7acdfc00a8f73064e285017e80b86/src/maxtext/configs/custom_mesh_and_rule) directory.

### v0.2.0

# Changes

- New `tpu-post-train` target in PyPI. Please also use this installation option for running vllm_decode. See the [MaxText installation instructions](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) for more info.
- [Qwen3-Next](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/tests/end_to_end/tpu/qwen/next/run_qwen3_next.md) is now supported.
- New MaxText structure! MaxText has been restructured according to [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/1b9e38aa0a19b6018feb3aed757406126b6953a1/RESTRUCTURE.md). Please feel free to share your thoughts and feedback.
- [Muon optimizer](https://kellerjordan.github.io/posts/muon) is now supported.
- DeepSeek V3.1 is now supported. Use existing configs for [DeepSeek V3 671B](https://github.com/AI-Hypercomputer/maxtext/blob/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/configs/models/deepseek3-671b.yml) and load in V3.1 checkpoint to use model.
- [New RL and SFT Notebook tutorials](https://github.com/AI-Hypercomputer/maxtext/tree/7656eb8d1c9eb0dd91e617a6fdf6ad805221221a/src/maxtext/examples) are available.
- The [ReadTheDocs documentation site](https://maxtext.readthedocs.io/en/latest/index.html) has been reorganized.
- Multi-host support for GSPO and GRPO is now available via [new RL tutorials](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_on_multi_host.html).
- A new guide, [What is Post Training in MaxText?](https://maxtext.readthedocs.io/en/latest/tutorials/post_training_index.html), is now available.
- Ironwood TPU co-designed AI stack announced. Read the [blog post on its co-design with MaxText](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack?e=48754805).
- [Optimized models tiering documentation](https://maxtext.readthedocs.io/en/latest/reference/models/tiering.html) has been refreshed.
- Added Versioning. Check out our [first set of release notes](https://maxtext.readthedocs.io/en/latest/release_notes.html)!
- Post-Training (SFT, RL) via [Tunix](https://github.com/google/tunix) is now available.
- Vocabulary tiling ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/2242)) is now supported in MaxText! Adjust config `num_vocab_tiling` to unlock more efficient memory usage.
- The GPT-OSS family of models (20B, 120B) is now supported.

# Deprecations

- Many MaxText modules have changed locations. Core commands like train, decode, sft, etc. will still work as expected temporarily. Please update your commands to the latest file locations
- install_maxtext_github_deps installation script replaced with install_maxtext_tpu_github_deps
- `tools/setup/setup_post_training_requirements.sh` for post training dependency installation is deprecated in favor of [pip installation](https://maxtext.readthedocs.io/en/latest/install_maxtext.html)

### v0.1.0

Our first MaxText PyPI package is here! MaxText is a high performance, highly scalable, open-source LLM library and reference implementation written in pure Python/JAX and targeting Google Cloud TPUs and GPUs for training. We are excited to make it easier than ever to get started.

Users can now install MaxText through pip, both for local development and through stable PyPI builds. Please see our [MaxText Installation Guide](https://maxtext.readthedocs.io/en/latest/install_maxtext.html) for more setup details.

Going forward, this page will document notable changes as we release new versions of MaxText.
