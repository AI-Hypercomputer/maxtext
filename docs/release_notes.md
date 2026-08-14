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

### v0.2.4

#### Changes

- **Flax NNX Transition**:
  - Flipped `pure_nnx`, `enable_nnx`, and `pure_nnx_decoder` configurations to True by default ([PR #3526](https://github.com/AI-Hypercomputer/maxtext/pull/3526)).
  - MaxText now executes primarily on Flax NNX with full support for NNX TrainState and NNX-native pipeline parallelism ([PR #2885](https://github.com/AI-Hypercomputer/maxtext/pull/2885)).
  - Fixed Linen-parity gaps and added unit tests ([PR #4255](https://github.com/AI-Hypercomputer/maxtext/pull/4255)), and implemented pure-NNX path for post-training correctness tests ([PR #4186](https://github.com/AI-Hypercomputer/maxtext/pull/4186)).
- **Model Support & Architecture**:
  - **Gemma 3**: Integrated weight mappings, vLLM adapter, and E2E pipelines ([PR #4338](https://github.com/AI-Hypercomputer/maxtext/pull/4338)).
  - **DeepSeek-V4**: Full model integration, decoders, and configuration stack ([PR #4153](https://github.com/AI-Hypercomputer/maxtext/pull/4153)), added HyperHead, aligned Sinkhorn implementation ([PR #4337](https://github.com/AI-Hypercomputer/maxtext/pull/4337)), and added checkpoint conversion support ([PR #4336](https://github.com/AI-Hypercomputer/maxtext/pull/4336)).
  - **Qwen3-VL**: Added support for Qwen3-VL models ([PR #4293](https://github.com/AI-Hypercomputer/maxtext/pull/4293), [PR #4517](https://github.com/AI-Hypercomputer/maxtext/pull/4517)) and Qwen3-VL-4B ([PR #4263](https://github.com/AI-Hypercomputer/maxtext/pull/4263)).
  - **Apple Envy MoE**: Added model configurations and support for Apple Envy Switch architectures.
  - **Chunked MoE**: Added chunked MoE support via `num_moe_token_chunks` to reduce memory footprint ([PR #4499](https://github.com/AI-Hypercomputer/maxtext/pull/4499)).
  - **Block Diffusion**: Added block-diffusion pre-training support ([PR #4776](https://github.com/AI-Hypercomputer/maxtext/pull/4776)), model-independent block corruption utilities ([PR #4737](https://github.com/AI-Hypercomputer/maxtext/pull/4737)), and causal-block attention across Dense, Splash, and Tokamax kernels ([PR #4743](https://github.com/AI-Hypercomputer/maxtext/pull/4743)).
- **LoRA & QLoRA**:
  - Added native LoRA Integration for Gemma 4, Gemma 3, Qwen3, and LLaMA 3 ([PR #3969](https://github.com/AI-Hypercomputer/maxtext/pull/3969), [PR #4265](https://github.com/AI-Hypercomputer/maxtext/pull/4265), [PR #4068](https://github.com/AI-Hypercomputer/maxtext/pull/4068)).
  - Added QLoRA support ([PR #3968](https://github.com/AI-Hypercomputer/maxtext/pull/3968)) and notebooks ([PR #3970](https://github.com/AI-Hypercomputer/maxtext/pull/3970)).
  - Added native LoRA interactive tutorial notebooks ([PR #4417](https://github.com/AI-Hypercomputer/maxtext/pull/4417)).
- **Attention & Context Parallelism**:
  - **TPU Ulysses CP**: Added TPU Ulysses context parallelism support via `context_parallel_strategy="ulysses"` ([PR #4687](https://github.com/AI-Hypercomputer/maxtext/pull/4687)) and Ulysses packing ([PR #4825](https://github.com/AI-Hypercomputer/maxtext/pull/4825)).
  - **TPU Ring Attention**: Added Tokamax ring MHA attention ([PR #4266](https://github.com/AI-Hypercomputer/maxtext/pull/4266)), load-balanced TPU Tokamax ring attention ([PR #4537](https://github.com/AI-Hypercomputer/maxtext/pull/4537)), and support packing for TPU Tokamax ring attention ([PR #4622](https://github.com/AI-Hypercomputer/maxtext/pull/4622)).
  - **TPU USP CP**: Supported packing in TPU USP attention ([PR #4887](https://github.com/AI-Hypercomputer/maxtext/pull/4887)).
  - **All-Gather CP**: Enabled packing for all-gather CP ([PR #4230](https://github.com/AI-Hypercomputer/maxtext/pull/4230)).
  - **DeepSeek MoE**: Added Ring Attention with DeepSeek DSA Sparse Indexer ([PR #4767](https://github.com/AI-Hypercomputer/maxtext/pull/4767)) and enabled auxiliary loss-free and sequence-wise load balancing ([PR #4753](https://github.com/AI-Hypercomputer/maxtext/pull/4753)).
  - **MLA Optimizations**: Implemented QK attention head chunking in MLA ([PR #4564](https://github.com/AI-Hypercomputer/maxtext/pull/4564)), optimized `generate_mask` ([PR #4437](https://github.com/AI-Hypercomputer/maxtext/pull/4437)), and added Approximate Top-K configuration ([PR #4243](https://github.com/AI-Hypercomputer/maxtext/pull/4243)).
  - **YaRN RoPE**: Added YaRN RoPE configuration support ([PR #4238](https://github.com/AI-Hypercomputer/maxtext/pull/4238)).
  - **MRoPE**: Standardized MRoPE to BS3 convention, enabled in multimodal training ([PR #4709](https://github.com/AI-Hypercomputer/maxtext/pull/4709)), and fixed partial rotary factor handling for Qwen3.5.
  - **ViT Attention**: Made ViT attention kernel configurable via `attention_for_vit` ([PR #4232](https://github.com/AI-Hypercomputer/maxtext/pull/4232)).
  - **Megacore**: Enabled megacore for splash attention dkv backward ([PR #4755](https://github.com/AI-Hypercomputer/maxtext/pull/4755)).
- **Quantization & Performance**:
  - **FP4 Support**: Added FP4 [E2M1] support to MaxText ([PR #4495](https://github.com/AI-Hypercomputer/maxtext/pull/4495)).
  - **Attention Quantization**: Added experimental attention quantization flags ([PR #4487](https://github.com/AI-Hypercomputer/maxtext/pull/4487)).
  - **TE Collective GEMM**: Supported TE Collective GEMMs ([PR #4470](https://github.com/AI-Hypercomputer/maxtext/pull/4470)) and overlap via `use_te_comm_gemm_overlap` ([PR #4307](https://github.com/AI-Hypercomputer/maxtext/pull/4307)).
  - **MoE Optimization**: Overlapped MoE comms with collective matmul ([PR #4295](https://github.com/AI-Hypercomputer/maxtext/pull/4295)) and added support for Tokamax GMM v2 ([PR #4197](https://github.com/AI-Hypercomputer/maxtext/pull/4197)).
  - **Double-buffering**: Double-buffered inner scans during gradient accumulation ([PR #4316](https://github.com/AI-Hypercomputer/maxtext/pull/4316)).
- **Checkpointing & Resiliency**:
  - **Intra-step Checkpointing**: Supported saving and restoring intra-step checkpoints in `MaxTextTrainingEngine` ([PR #4679](https://github.com/AI-Hypercomputer/maxtext/pull/4679)).
  - **Orbax v1 Migration**: Migrated Tunix's Checkpoint Manager to Orbax V1 ([PR #4498](https://github.com/AI-Hypercomputer/maxtext/pull/4498), [PR #4471](https://github.com/AI-Hypercomputer/maxtext/pull/4471)) and added Orbax v1 `StatefulCheckpointable` for Grain iterators ([PR #4380](https://github.com/AI-Hypercomputer/maxtext/pull/4380)).
  - **NNX Checkpoint**: Persisted RNGs/dropout across resumes ([PR #4343](https://github.com/AI-Hypercomputer/maxtext/pull/4343)) and added emergency checkpoint layout abstract ([PR #4372](https://github.com/AI-Hypercomputer/maxtext/pull/4372)).
  - **Dynamic scan_layers**: Dynamically set `scan_layers` from checkpoint metadata ([PR #4344](https://github.com/AI-Hypercomputer/maxtext/pull/4344)) and verified compatibility ([PR #4304](https://github.com/AI-Hypercomputer/maxtext/pull/4304)).
  - **LoRA Config**: Saved/restored LoRA config in checkpoint metadata ([PR #4269](https://github.com/AI-Hypercomputer/maxtext/pull/4269)).
- **Goodput & Elasticity**:
  - Added Goodput support for Pathways Elasticity & Slice Efficiency, including `record_slice_state()` to query live slice counts ([PR #4840](https://github.com/AI-Hypercomputer/maxtext/pull/4840)).
  - Added elasticity and goodput integration tests and debugging logs ([PR #4846](https://github.com/AI-Hypercomputer/maxtext/pull/4846)).
  - Implemented checkpoint-based elasticity using set-based slice tracking ([PR #4245](https://github.com/AI-Hypercomputer/maxtext/pull/4245)).
- **Reinforcement Learning (RL)**:
  - Migrated single-step SPMD RL trainer to `src/maxtext/training_engine/` ([PR #4574](https://github.com/AI-Hypercomputer/maxtext/pull/4574)).
  - Added `reward_functions_path` and `reward_functions` CLI knobs for custom rewards ([PR #4149](https://github.com/AI-Hypercomputer/maxtext/pull/4149)).
  - Updated tutorials with `AgenticGRPOLearner` for async RL training ([PR #4181](https://github.com/AI-Hypercomputer/maxtext/pull/4181)) and added GRPO Gemma4-e4b tutorial ([PR #4427](https://github.com/AI-Hypercomputer/maxtext/pull/4427)).
  - Added DPO notebooks ([PR #4362](https://github.com/AI-Hypercomputer/maxtext/pull/4362)).
- **Usability & Infrastructure**:
  - Provided a top-level maxtext package facade and simplified Tunix imports ([PR #4810](https://github.com/AI-Hypercomputer/maxtext/pull/4810)), and lazy-loaded top-level package exports ([PR #4850](https://github.com/AI-Hypercomputer/maxtext/pull/4850)).
  - Made TensorFlow optional for running `train.py` ([PR #4757](https://github.com/AI-Hypercomputer/maxtext/pull/4757)).
  - Added wandb logging support ([PR #3053](https://github.com/AI-Hypercomputer/maxtext/pull/3053)).
  - Added Hugging Face Grain streaming integration and onboarding guide ([PR #4486](https://github.com/AI-Hypercomputer/maxtext/pull/4486)).
  - Added Simple-evals runner support for gpt-oss model family ([PR #4644](https://github.com/AI-Hypercomputer/maxtext/pull/4644)).
  - Added scripts to run vanilla DiLoCo on MaxText ([PR #4095](https://github.com/AI-Hypercomputer/maxtext/pull/4095)).
  - Introduced an inflight computation throttler to rate-limit TPU computations in training engine ([PR #4629](https://github.com/AI-Hypercomputer/maxtext/pull/4629)).
  - Added option to enable on-demand profiling server in ML Diagnostics ([PR #4131](https://github.com/AI-Hypercomputer/maxtext/pull/4131)).
- **Dependency Upgrades**:
  - Upgraded JAX / jaxlib to version 0.10.2 for pre-training and 0.11.0 for post-training.
  - Upgraded Flax to 0.12.8 and Orbax Checkpoint to >=0.12.2.
  - Upgraded Orbax Checkpoint to 0.12.4 ([PR #4863](https://github.com/AI-Hypercomputer/maxtext/pull/4863)).
  - Added support for Qwix >=0.1.8, Pathwaysutils >=0.1.11, and NVIDIA Transformer Engine v2.16.

#### Bug Fixes

- **Gemma 3/4 RL Rollout**: Resolved Gemma 3/4 RL rollout gibberish by unrolling scanned weights for vLLM adapter ([PR #4536](https://github.com/AI-Hypercomputer/maxtext/pull/4536), [PR #4519](https://github.com/AI-Hypercomputer/maxtext/pull/4519), [PR #4404](https://github.com/AI-Hypercomputer/maxtext/pull/4404)).
- **Double-compilation Fix**: Fixed double-compilation in `train_step` by matching input sharding ([PR #4174](https://github.com/AI-Hypercomputer/maxtext/pull/4174)).
- **MoE Sharding Fix**: Made MoE dispatch/MLP expert-axis batch sharding configurable to fix Mixtral EP throughput ([PR #4179](https://github.com/AI-Hypercomputer/maxtext/pull/4179)).
- **Qwix Quantization NNX**: Relanded support for Qwix quantization on NNX with cross-environment import compatibility ([PR #4198](https://github.com/AI-Hypercomputer/maxtext/pull/4198)).
- **RemoteIteratorWrapper Multi-Device**: Resolved ValueError when unpacking partitioned arrays during save/restore on multi-device topologies, corrected device resolution to use global_mesh.devices, and enforced Pathways validation for colocated python data input ([PR #4212](https://github.com/AI-Hypercomputer/maxtext/pull/4212)).
- **Qwen3.5 & Hybrid Qwen Fixes**: Fixed MRoPE dimension expansion in vLLM adapter ([PR #4177](https://github.com/AI-Hypercomputer/maxtext/pull/4177)), fixed KV cache dictionary indexing in vLLM rollouts, and corrected partial rotary factor in Qwen3.5 attention.
- **Gemma 4 Multimodal Chat-Template**: Fixed special token handling in processor_gemma4 to prevent token sequence explosion.
- **NNX Decoder Memory Optimization**: Excluded read-only parameters from jax.lax.scan carry/output states in NNX decoders, saving ~12.7 GiB of HBM memory and preventing RESOURCE_EXHAUSTED OOM errors during large model execution ([PR #4405](https://github.com/AI-Hypercomputer/maxtext/pull/4405)).
- **NNX MTP Fix**: Resolved silent zero-loss issue in NNX Multi-Token Prediction ([PR #4525](https://github.com/AI-Hypercomputer/maxtext/pull/4525)).
- **RL LR Schedule Fix**: Fixed RL LR schedule regression by defaulting schedule length to `max_train_steps` ([PR #4225](https://github.com/AI-Hypercomputer/maxtext/pull/4225)).
- **RL Tail Batch Handling**: Added drop_remainder=True support to prevent shape mismatches on tail batches during GRPO training ([PR #4252](https://github.com/AI-Hypercomputer/maxtext/pull/4252)).
- **Model Fixes (Qwen, GPT-OSS, Gemma4)**:
  - Fixed Qwen3-30B training failure caused by sharding mismatch ([PR #4883](https://github.com/AI-Hypercomputer/maxtext/pull/4883)).
  - Fixed GPT-OSS-20B checkpoint conversion failures ([PR #4882](https://github.com/AI-Hypercomputer/maxtext/pull/4882)).
  - Fixed double application of per-expert-scale in Gemma 4 MoE ([PR #4880](https://github.com/AI-Hypercomputer/maxtext/pull/4880)).
  - Applied partial MRoPE to Qwen3.5 attention ([PR #4764](https://github.com/AI-Hypercomputer/maxtext/pull/4764)).
  - Fixed Qwix LoRA mesh sharding and scaled Qwen/LLaMA parallelism in post-training ([PR #4866](https://github.com/AI-Hypercomputer/maxtext/pull/4866)).
- **NNX / MoE**: Preserved intermediates in scanned layers for MoE load balance loss in NNX ([PR #4829](https://github.com/AI-Hypercomputer/maxtext/pull/4829)).
- **Sharding**: Truncated out_sharding to match tensor ndim when pspec has extra dimensions ([PR #4769](https://github.com/AI-Hypercomputer/maxtext/pull/4769)).
- **Other Fixes**:
  - Fix load-balanced TPU Splash local sliding masks when CP load balancing is enabled ([PR #4298](https://github.com/AI-Hypercomputer/maxtext/pull/4298)).
  - Restricted GMM quantization to fp8_full ([PR #4842](https://github.com/AI-Hypercomputer/maxtext/pull/4842)).
  - Fixed the use of targets_segmentation in Multi-Token Prediction (MTP) ([PR #4756](https://github.com/AI-Hypercomputer/maxtext/pull/4756)).

### v0.2.3

#### Changes

- Upgraded JAX to version 0.10.0 for pre-training and 0.10.1 for post-training.
- **New vLLM-Powered Evaluation Framework**: Introduced an eval framework for running lm-eval, evalchemy, and custom benchmarking against MaxText checkpoints. See the [evaluation guide](https://maxtext.readthedocs.io/en/latest/guides/eval_framework.html) for details.
- Added support for pre-training new models:
  - **Qwen3.5**: Qwen3.5 35B & 397B is now [supported](https://github.com/AI-Hypercomputer/maxtext/blob/d938b91acaa3baaaf32956e21677bd29e14549a1/tests/end_to_end/tpu/qwen/moe/run_qwen_moe.md).
  - **Qwen3-Omni**: Support for multimodal SFT ([PR #3863](https://github.com/AI-Hypercomputer/maxtext/pull/3863)).
- **Direct Preference Optimization (DPO/ORPO) Support**: Full support for DPO and ORPO alignment pipelines. See the [DPO tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/dpo.html) for details.
- **Reinforcement Learning (RL) Recipe**: Added a pre-configured [RL recipe for Qwen3-30b-a3b](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_qwen3_30b.html).
- **Iterative Quality Monitoring (RL)**: Added intermediate evaluation hooks to automatically run quality benchmarks during RL training (every `eval_interval` steps), optimized with a new `eval_batch_size` configuration knob.
- **Developer Extensibility**: Added `dataset_processor_path` CLI knob for custom dataset integration, and refactored shared post-training hooks to simplify custom SFT, DPO, and RL workflow development.
- **Generalized Learn-to-Init (LTI) for Distillation**: Enhanced post-training distillation capabilities with generalized LTI support.
- Added support for recording elastic goodput events during training to track efficiency ([PR #3901](https://github.com/AI-Hypercomputer/maxtext/pull/3901)).
- **Installation Updates**: Updated the `[tpu-post-train]` installation command to require `UV_TORCH_BACKEND=cpu`(see [Installation Guide](install_maxtext.md)).
- **Zero1 AOT Compilation**: Added zero1 support to Ahead-Of-Time (AOT) compilation in train compile, improving compilation capabilities for zero1 config.
- **MoE Performance Optimization**: Integrated ragged gather reduce into Mixture of Experts (MoE) layers to optimize memory and performance by replacing ragged scatter and supporting backward pass.
- Added [E2E scripts](https://github.com/AI-Hypercomputer/maxtext/tree/main/tests/end_to_end/tpu/gemma3/4b) to run checkpoint conversion, pre-training and post-training (SFT, RL) with Gemma3-4B model.
- **Bug Fixes and Usability Enhancements**:
  - **Attention Masking Fix in RL**: Fixed an issue in `TunixMaxTextAdapter` where queries at non-pad positions could attend to pad-position keys during training, which was corrupting log-probabilities and affecting GRPO training reward trajectories ([PR #4016](https://github.com/AI-Hypercomputer/maxtext/pull/4016)).
  - **JAX/NNX Gradient Mutation Fix**: Refactored post-training loops (`train_distill`, `train_sft`, `train_rl`) to use `jax.value_and_grad` with explicit NNX state split/merge instead of nesting `nnx.value_and_grad` inside `nnx.jit` ([PR #3652](https://github.com/AI-Hypercomputer/maxtext/pull/3652)).
  - **Qwen3-MoE Checkpoint Conversion**: Fixed checkpoint conversion issues for Qwen3-MoE models ([PR #3868](https://github.com/AI-Hypercomputer/maxtext/pull/3868)).
  - **Duplicate Configuration Failures Fix**: Allowed identical config overrides and handled configuration exceptions cleanly ([PR #3933](https://github.com/AI-Hypercomputer/maxtext/pull/3933)).
- **Documentation Improvements**: Updated [Getting started](https://maxtext.readthedocs.io/en/latest/getting_started.html) guide, including new guides for the [evaluation framework](https://maxtext.readthedocs.io/en/latest/guides/eval_framework.html) and the [DPO tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/dpo.html).

#### Deprecations

- Deleted [legacy DPO implementation](https://github.com/AI-Hypercomputer/maxtext/pull/3997) in favor of the integrated [DPO trainer](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/dpo.html).
- Removed stack trace collection feature.

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

- Use the new `maxtext[runner]` installation option to build Docker images without cloning the repository. This can be used for scheduling jobs through XPK. See the [MaxText installation instructions](build-docker) for more info.
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
