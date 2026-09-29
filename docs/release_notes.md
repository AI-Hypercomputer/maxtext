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

## Unreleased

**Last Updated**: 8154b7d97

<!-- Add new unreleased changes below this line -->

#### Changes

##### Models

- **Weaver Architecture**: Added the Weaver Mixture-of-Transformers (MoT) block and standardized Weaver model naming across configurations.
- **DeepSeek-V3 / Lineage**: Integrated the Lineage DeepSeek-V3 model (`dsv3.py`) into MaxText, including routed expert output weight layout storage under `use_lineage`, router bias update collection, and a custom `dsv3-mlperf-4k` device mesh and sharding rule.
- **DeepSeek-V4**: Added checkpoint conversion support and parameter mapping for DeepSeek-V4 models.
- **Cosmos3**: Added model bring-up configurations and architecture support for `Cosmos3-super-reasoner` and `Cosmos3-nano`.
- **Qwen3 / Qwen3.5**: Added model configurations and Hugging Face parameter mappings for `qwen3.5-35b-a3b-fp8` and `qwen3.5-397b-a17b-fp8`, verified `qwen3.5-35b-a3b` weight conversion, and optimized Qwen3-Next to only instantiate shared experts when `shared_experts > 0`.
- **M3 Core**: Introduced the `m3` core package scaffolding with architectural design rules, layout testing, mesh creation, sharding rules, and RoPE positional embeddings.
- **MoE Dropless Fallback & Ragged Buffers**: Added `moe_dropless_fallback` (`None|step|layer`) with memory-neutral step fallback, host-driven step replay retry for ragged buffer overflow, `eval_ragged_buffer_factor` for evaluation-specific ragged buffer sizing, and precompiled replay/eval programs ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html)).
- **MoE Routing & Load Balancing**: Implemented dense custom VJP for router top-k selection (eliminating backward scatter), optimized top-2 expert group scoring and routing performance, aligned sigmoid-router auxiliary loss and full-batch expert bias updates with Megatron-LM, added support for extracting and replaying routing decisions across inference and training, and integrated TE `MoEBlock` ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html)).
- **Decoders & Normalization**: Added RMSNorm support for Apple ENVY and OLMo3 decoder blocks, introduced GPT-OSS attention sinks, added 2D transposed shape alignment in `model_creation_utils`, split embedding logical axis names in attention modules, and established default sharding rule constants in `types.py` with base configuration parity testing.

##### Pre-Training

- **DiLoCo Pre-Training**: Added comprehensive documentation and tutorials for DiLoCo distributed pre-training, and migrated DiLoCo launch workflows and scripts from XPK to Cluster Toolkit ([core concepts](https://maxtext.readthedocs.io/en/latest/reference/core_concepts.html), [diloco](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/diloco.html), [diloco pretraining](https://maxtext.readthedocs.io/en/latest/tutorials/diloco_pretraining.html), [tutorials](https://maxtext.readthedocs.io/en/latest/tutorials.html)).
- **Attention Kernels & Flash Attention**: Optimized JAX Flash Attention with a specialized path for sequences longer than 4096 tokens, added HCA static compilation for Splash Attention in DeepSeek-V4, and refactored the mHC kernel to use `emit_pipeline`.
- **DeepSeek-V4 Compressed Structured Attention (CSA)**: Added a fused Pallas TPU StreamIndex score kernel, implemented QK attention head chunking to lower peak memory usage, added indexer loss support, and enabled the native block-sparse path for indexer masks under Context Parallelism.
- **Multi-Token Prediction (MTP)**: Enabled quantized Multi-Token Prediction and added separate final layer normalization for MTP blocks in DeepSeek models.
- **Context Parallelism & Positional Embeddings**: Eliminated XLA backward adjoint interior padding during sequence restoration in `reorder_sequence` for context parallelism, sharded GatedDeltaNet sequences under `ici_context_parallelism`, removed all-to-all communication from TSP/CP segmentation masking, avoided JAX recompilations in MRoPE position calculation via inline NumPy, and normalized MRoPE shapes across vLLM hybrid cache utilities.

##### Post-Training

- **Raiden Weight Synchronization**: Integrated high-performance Raiden and Raiden-FFI weight synchronization into `MaxTextTrainingEngine` conforming to Tunix's trainer contract, featuring streaming and target-free weight conversion for vLLM rollout, support for inhomogeneous layer cycle intervals in `raiden_unscan`, and unified synchronizer management.
- **Reinforcement Learning (RL) & GRPO**: Plumbed GRPO flags to the agentic learner, added RL sequence-packing and vLLM block-size knobs, added `free_kv_cache_during_weight_sync` and `gc_collect_after_weight_sync` options, optimized `extract_answer` to linear time complexity, exposed `reward_num_workers` with picklable reward wrappers, added a Gemma 4 26B GRPO tutorial, and integrated the MaxText trainer backend into the distributed GSM8K example ([post training index](https://maxtext.readthedocs.io/en/latest/tutorials/post_training_index.html), [rl gemma4 26b](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_gemma4_26b.html), [rl gemma4 e4b](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_gemma4_e4b.html)).
- **Distillation & LoRA**: Added end-to-end launch scripts and configs for Distillation Training, and refactored LoRA parameter resharding with modernized Flax NNX variable access and streamlined graph traversal.
- **vLLM Serving**: Added vLLM Mamba prefix caching in Qwen3 GatedDeltaNet layers, implemented MoE 128-lane chunking and padding for GMM layouts, and indexed KV caches via `layer_name_to_kvcache_index` in the vLLM adapter.

##### Multimodal

- **Multimodal Pipelines & SFT**: Added Grain data pipeline support for multimodal SFT, enabled end-to-end video SFT with variable frame resolutions and durations, added ChartNet SFT configurations with string response support, and updated Omni multimodal processing pipelines ([multimodal](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/multimodal.html)).
- **Qwen3-VL Serving**: Added `vllm_decode` support for Qwen3-VL vision-language inference ([inference](https://maxtext.readthedocs.io/en/latest/tutorials/inference.html)).

##### Performance

- **FP8 Training & Quantization Infrastructure**: Added native FP8 inference (`serve_fp8_weight`) for `DenseGeneral` and GMM v2, added dynamic weight-only FP8 dequantization types and direct FP8 scale tensor ingestion in `to_maxtext`, implemented `drhs_grad_quantization_calibration_method` cotangent calibration for weight gradients, enabled quantized logits projection, and allowed keeping MoE router projections unquantized for numerical stability ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html), [quantization](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/quantization.html)).
- **MoE Communication Quantization**: Enabled token-quantized all-gather pipelines with Ring of Experts, added quantized all-gather for the Ring of Experts combine backward, surfaced TPU inference MoE kernel knobs and padding, and pre-quantized weights for fused MoE kernels ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html), [quantization](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/quantization.html)).
- **Explicit Sharding & ZeRO-1**: Onboarded Qwen3.5, Qwen3-Next, Qwen3, Qwen2, Kimi-K2, Mistral, Mixtral, and Gemma model families to explicit sharding and ZeRO-1 with gradient accumulation; deferred data-parallel gradient all-reduces to `update()` under GA; scaled backward cotangents at GA=1 scale; supported partitioned optimizer state sharding propagation with `MaskedNode` filtering in ZeRO-1; and explicitly tracked `shard_embed_moe_on_fsdp` ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html)).
- **Sharded Muon Optimizer**: Integrated a sharding-aware, distributed version of the Muon optimizer into MaxText, generalized weight dimension extraction, added support for Qwen3 and GPT-OSS MoE blocks in Muon, and added configuration flags to selectively apply Muon to MoE routers.
- **SparseCore & Pallas Kernels**: Added `moe_pin_sparse_core_all_gathers` to pin FSDP and EP all-gathers to SparseCores, lowered `plsc.bitcast` to `tpu.bitcast` during layout passes, and tuned Pallas SparseCore compiler parameters across `sc_ragged_gather` and `sc_ragged_gather_reduce`.
- **Execution & Tiling Optimizations**: Added Tokamax GMM v2 heuristic tiling, evaluation-stage tile sizes and QKV layouts, nested scans over hybrid attention with per-layer remat for Qwen3-Next, `moe_x_sorted` remat locations across expert boundaries, and collective `psum` reductions for expert bias calculation ([moe configuration](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/moe_configuration.html)).

##### Checkpointing / Goodput

- **Orbax v1 Checkpointing**: Refactored MaxText checkpointing to use Orbax v1 APIs as the primary backend, enabled dequantize-on-load parameter restoration, supported restoring variables outside the standard `params` collection, registered Pathways persistence array handlers with bounded host staging, and introduced standalone checkpointer benchmarking configurations.
- **Goodput & Checkpointing Guides**: Added `goodput_job_name` configuration for Goodput telemetry logging, and migrated multi-tier and emergency checkpointing guides from XPK to Cluster Toolkit ([emergency checkpointing](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/emergency_checkpointing.html), [multi tier checkpointing](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/multi_tier_checkpointing.html)).

##### Usability

- **Data Input & Grain**: Added Megatron MMap data format support for Grain data loaders with integrated mmap index building, migrated post-training scripts from TFDS to Grain, enabled Grain to read TFDS configurations, optimized input pipeline throughput during evaluation, and set default `dataset_type` to synthetic ([data input pipeline](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline.html), [data input grain](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_grain.html), [data input megatron mmap](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_megatron_mmap.html), [full finetuning](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/full_finetuning.html)).
- **MLPerf Compliance & Performance**: Added official MLPerf training logging compliance, reduced step-0 overhead via per-file `c4_mlperf` train sharding and data-free pre-run start setup, implemented continuous stream chunking for C4 MLPerf, and cached eval batches per host.
- **Documentation & Cluster Toolkit Migration**: Migrated core navigation, getting started, and architecture guides from XPK to Cluster Toolkit (CTK), added pre-training and post-training Docker images for release 0.2.4 to tutorials, and added Gemini agent skills for MaxText development ([getting started](https://maxtext.readthedocs.io/en/latest/getting_started.html), [install maxtext](https://maxtext.readthedocs.io/en/latest/install_maxtext.html), [architecture overview](https://maxtext.readthedocs.io/en/latest/reference/architecture/architecture_overview.html), [build maxtext](https://maxtext.readthedocs.io/en/latest/tutorials/build_maxtext.html)).
- **Artifact Registry Migration**: Updated container image registry paths across tutorials, scripts, and documentation from Google Container Registry (GCR) to Google Artifact Registry ([run maxtext elastic training](https://maxtext.readthedocs.io/en/latest/run_maxtext/run_maxtext_elastic_training.html), [run maxtext via pathways](https://maxtext.readthedocs.io/en/latest/run_maxtext/run_maxtext_via_pathways.html), [lora on multi host](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/lora_on_multi_host.html)).
- **Dependency Upgrades**: Upgraded JAX to version 0.11.1, made TensorFlow an optional dependency in requirements, and upgraded Tokamax to 0.0.14 and DrJAX to 0.2.1 ([update dependencies](https://maxtext.readthedocs.io/en/latest/development/update_dependencies.html), [data input tfds](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_tfds.html), [install maxtext](https://maxtext.readthedocs.io/en/latest/install_maxtext.html)).
- **Diagnostics & Telemetry**: Added support for custom GCS destination paths for ML Diagnostic telemetry logging, and optimized multi-worker XLA dumping to avoid redundant CNS I/O with `--dump_xla_all_hosts`.

#### Bug Fixes

- Fixed attention mask generation edge cases for `AttentionType.COMPRESSED` (DeepSeek-V4 attention) across chunked prefill, packed sequences, and standalone environments.
- Fixed Context Parallelism (CP) size derivation from active logical axis rules and computed local block padding for dynamic Splash indexer masks under CP.
- Fixed vocab tiling hidden-state cotangent scaling.
- Fixed inference `query_heads` rule mapping in `AttentionOp` and inference configuration.
- Fixed dropless-replay remat override to match the Flax NNX decoder.
- Fixed dropped routing-weight gradients in `ring_ragged_unsort`.
- Fixed FP8 MoE on the `sparse_matmul` path and quantized MoE on the `dense_matmul` path.
- Fixed Tunix adapter defects, engine packing path, and MoE weight tracking / zero-pad key sets in RL training.
- Fixed `RLConfig` device mesh initialization to prevent `NoneType` `AttributeError`.
- Fixed vLLM sampler initialization failures, `cache_heads`/`paged_kv_heads` axis ordering, and hybrid decode for Qwen3.5.
- Fixed MoE data-parallel reduce-scatter and logical axis sharding in the vLLM serving path.
- Fixed GatedDeltaNet (GDN) Mamba block table indexing and enabled align cache mode under prefix caching.
- Fixed standalone Torchax converter for current TPU inference and wired `use_standalone_converter` into rollout.
- Fixed standalone checkpointer restore loop using Orbax v1 load, and raised `RuntimeError` on fatal checkpointing errors.
- Fixed GCS path handling by skipping unnecessary `exists()` checks, and allowed warm starts from `load_parameters_path` without requiring checkpointing.
- Fixed Grain dataloader starvation and process leak on slice recovery, resolved `c4_mlperf` fractional eval batch size handling, and fixed `tensorflow_text` `ImportError` in post-training.
- Fixed tokenizer type mismatch and Grain `IterDataset` crash for Qwen3-VL-2B.
- Fixed EOS token detection and Qwen vision model routing in `multimodal_eval`.
- Fixed chat template prompt/completion formatting and SFT masking for Gemma 4 reasoning.
- Fixed XLA fusion breakage and numerical drift in Qwen3 under AUTO shard mode, passed `weight_dtype` in `Qwen3NextSparseMoeBlock` shared expert gate, and fixed argument mismatch in `_maybe_set_compact_mamba_num_blocks_override`.
- Fixed `AttributeError` in MHC-lite during shape evaluation and sharding construction.
- Fixed duplicate XLA compilation of `jit(train_step)` and resolved Flax NNX throughput regression when `scan_layers=False`.
- Fixed FP8 custom-gradient state overwrite during Flax NNX training.
- Fixed Ahead-Of-Time (AOT) compilation for ragged kernels and supported internal compile and orchestration.
- Fixed DiLoCo test state loading by explicitly setting `enable_checkpointing`.
- Suppressed Pyrefly type checking errors on JAX scalar types to improve compatibility with upcoming JAX releases.

#### Deprecations

- **Flax Linen Removal**: Completely removed legacy Flax Linen modules, layers, and `*_as_linen` model wrappers, deleted the `pure_nnx`, `enable_nnx`, and `pure_nnx_decoder` configuration flags, and collapsed dispatch logic across trainers (pre-train, DiLoCo, GRPO), inference (MaxEngine, KVCache, vLLM, LoRA), quantization, and model creation ([distillation](https://maxtext.readthedocs.io/en/latest/guides/distillation.html), [knowledge distillation](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/knowledge_distillation.html), [lora](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/lora.html)).
- Removed the deprecated `--parallel_threads` CLI flag from `to_huggingface.py`.
- Replaced deprecated `google-cloud-sdk` package with `google-cloud-cli` in Dockerfiles ([run maxtext via xpk](https://maxtext.readthedocs.io/en/latest/run_maxtext/run_maxtext_via_xpk.html), [rl on multi host](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_on_multi_host.html), [sft on multi host](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/sft_on_multi_host.html)).

## Releases

### v0.2.4

#### Changes

- **Flax NNX Migration**: Enabled `pure_nnx`, `enable_nnx`, and `pure_nnx_decoder` configurations by default ([PR #3526](https://github.com/AI-Hypercomputer/maxtext/pull/3526)), migrating MaxText primarily on Flax NNX ([PR #2885](https://github.com/AI-Hypercomputer/maxtext/pull/2885)).

- **Dependency Upgrades**: Upgraded JAX to version 0.10.2 for pre-training and 0.11.0 for post-training.

- **Model Support & Architecture**:

  - **DeepSeek-V4**: Full model integration, decoders, and configuration stack ([PR #4153](https://github.com/AI-Hypercomputer/maxtext/pull/4153)), added HyperHead, aligned Sinkhorn implementation ([PR #4337](https://github.com/AI-Hypercomputer/maxtext/pull/4337)), and added checkpoint conversion support ([PR #4336](https://github.com/AI-Hypercomputer/maxtext/pull/4336)). See the [user guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/deepseek/Run_DeepSeek.md) for more details.
  - **Qwen3-VL**: Added support for Qwen3-VL models ([PR #4293](https://github.com/AI-Hypercomputer/maxtext/pull/4293), [PR #4517](https://github.com/AI-Hypercomputer/maxtext/pull/4517)) and Qwen3-VL-4B ([PR #4263](https://github.com/AI-Hypercomputer/maxtext/pull/4263)).
  - **Apple Envy MoE**: Added model configurations and support for Apple Envy Switch architectures.
  - **Chunked MoE**: Added chunked MoE support via `num_moe_token_chunks` to reduce memory footprint ([PR #4499](https://github.com/AI-Hypercomputer/maxtext/pull/4499)).
  - **Block Diffusion**: Added block-diffusion pre-training support ([PR #4776](https://github.com/AI-Hypercomputer/maxtext/pull/4776)), model-independent block corruption utilities ([PR #4737](https://github.com/AI-Hypercomputer/maxtext/pull/4737)), and causal-block attention across Dense, Splash, and Tokamax kernels ([PR #4743](https://github.com/AI-Hypercomputer/maxtext/pull/4743)).

- **LoRA & QLoRA**: Added native LoRA and QLoRA support for Gemma4, Gemma3, Qwen3, and Llama3, along with interactive tutorials ([PR #3969](https://github.com/AI-Hypercomputer/maxtext/pull/3969), [PR #4265](https://github.com/AI-Hypercomputer/maxtext/pull/4265), [PR #4068](https://github.com/AI-Hypercomputer/maxtext/pull/4068), [PR #3968](https://github.com/AI-Hypercomputer/maxtext/pull/3968), [PR #3970](https://github.com/AI-Hypercomputer/maxtext/pull/3970), [PR #4417](https://github.com/AI-Hypercomputer/maxtext/pull/4417)).

- **Context Parallelism (CP), Ring Attention**:

  - Added Ulysses and USP CP strategy and packing ([PR #4687](https://github.com/AI-Hypercomputer/maxtext/pull/4687), [PR #4825](https://github.com/AI-Hypercomputer/maxtext/pull/4825), [PR #4836](https://github.com/AI-Hypercomputer/maxtext/pull/4836)), Tokamax load-balanced Ring Attention ([PR #4266](https://github.com/AI-Hypercomputer/maxtext/pull/4266), [PR #4537](https://github.com/AI-Hypercomputer/maxtext/pull/4537), [PR #4622](https://github.com/AI-Hypercomputer/maxtext/pull/4622)), and sequence packing for USP and All-Gather CP ([PR #4230](https://github.com/AI-Hypercomputer/maxtext/pull/4230), [PR #4887](https://github.com/AI-Hypercomputer/maxtext/pull/4887)).
  - DeepSeek MoE & MLA: Added Ring Attention with DSA Sparse Indexer [PR #4767](https://github.com/AI-Hypercomputer/maxtext/pull/4767), auxiliary loss-free and sequence-wise load balancing [PR #4753](https://github.com/AI-Hypercomputer/maxtext/pull/4753), MLA QK head chunking [PR #4564](https://github.com/AI-Hypercomputer/maxtext/pull/4564), optimized generate_mask [PR #4437](https://github.com/AI-Hypercomputer/maxtext/pull/4437), and Approximate Top-K [PR #4243](https://github.com/AI-Hypercomputer/maxtext/pull/4243).
  - Positional Embeddings: Added YaRN RoPE config [PR #4238](https://github.com/AI-Hypercomputer/maxtext/pull/4238), standardized MRoPE to BS3 convention for multimodal training [PR #4709](https://github.com/AI-Hypercomputer/maxtext/pull/4709), and fixed Qwen3.5 partial rotary factor handling.
  - Kernels & Megacore: Added configurable attention_for_vit kernels [PR #4232](https://github.com/AI-Hypercomputer/maxtext/pull/4232) and enabled Megacore for Splash Attention dkv backward [PR #4755](https://github.com/AI-Hypercomputer/maxtext/pull/4755).

- **Quantization & Performance**: Added FP4 [E2M1] ([PR #4495](https://github.com/AI-Hypercomputer/maxtext/pull/4495)) and experimental attention quantization ([PR #4487](https://github.com/AI-Hypercomputer/maxtext/pull/4487)); enabled TE Collective GEMMs ([PR #4470](https://github.com/AI-Hypercomputer/maxtext/pull/4470)) and overlap ([PR #4307](https://github.com/AI-Hypercomputer/maxtext/pull/4307)), MoE comms with collective matmul ([PR #4295](https://github.com/AI-Hypercomputer/maxtext/pull/4295)), Tokamax GMM v2 ([MoE configuration guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md)), and double-buffered inner scans during gradient accumulation ([PR #4316](https://github.com/AI-Hypercomputer/maxtext/pull/4316)).

- **Checkpointing**: Added support for Multi-tier checkpointing in Pathways.

- **Goodput & Elasticity**:

  - Added Goodput support for Pathways Elasticity & Slice Efficiency, including `record_slice_state()` to query live slice counts ([PR #4840](https://github.com/AI-Hypercomputer/maxtext/pull/4840)).
  - Implemented checkpoint-based elasticity using set-based slice tracking ([PR #4245](https://github.com/AI-Hypercomputer/maxtext/pull/4245)).

- **Post Training**:

  - Added `reward_functions_path` and `reward_functions` CLI knobs for custom rewards ([PR #4149](https://github.com/AI-Hypercomputer/maxtext/pull/4149)) to RL training.
  - Updated tutorials with `AgenticGRPOLearner` for async RL training ([PR #4181](https://github.com/AI-Hypercomputer/maxtext/pull/4181)) and added GRPO Gemma4-e4b tutorial ([PR #4427](https://github.com/AI-Hypercomputer/maxtext/pull/4427)).
  - Added RL support for Qwen3 30B and GPT-OSS 20B. See the [Qwen3 30B RL tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_qwen3_30b.html) and [GPT-OSS 20B RL tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_gptoss_20b.html) for recipes.
  - Added support for DPO along with tutorials ([PR #4362](https://github.com/AI-Hypercomputer/maxtext/pull/4362)).

- **Usability & Infrastructure**:

  - Added wandb logging support ([PR #3053](https://github.com/AI-Hypercomputer/maxtext/pull/3053)).
  - Added Hugging Face Grain streaming integration and onboarding guide ([PR #4486](https://github.com/AI-Hypercomputer/maxtext/pull/4486)).
  - Added Simple-evals runner support for gpt-oss model family ([PR #4644](https://github.com/AI-Hypercomputer/maxtext/pull/4644)).
  - Added scripts to run vanilla DiLoCo on MaxText ([PR #4095](https://github.com/AI-Hypercomputer/maxtext/pull/4095)).
  - Added option to enable on-demand profiling server in ML Diagnostics ([PR #4131](https://github.com/AI-Hypercomputer/maxtext/pull/4131)).

#### Bug Fixes

- **Post-Training**:

  - Resolved Gemma 3/4 RL rollout gibberish issue by unrolling scanned weights for vLLM adapter ([PR #4536](https://github.com/AI-Hypercomputer/maxtext/pull/4536), [PR #4519](https://github.com/AI-Hypercomputer/maxtext/pull/4519), [PR #4404](https://github.com/AI-Hypercomputer/maxtext/pull/4404)).
  - Fixed RL LR schedule defaults ([PR #4225](https://github.com/AI-Hypercomputer/maxtext/pull/4225)), added `drop_remainder=True` to prevent shape mismatches on tail batches during GRPO training ([PR #4252](https://github.com/AI-Hypercomputer/maxtext/pull/4252)) and resolved Qwen3.5 MRoPE/Kv-cache rollout issues ([PR #4177](https://github.com/AI-Hypercomputer/maxtext/pull/4177)).

- **Compilation**:

  - Fixed double-compilation in `train_step` by matching input sharding ([PR #4174](https://github.com/AI-Hypercomputer/maxtext/pull/4174)).
  - Truncated out_sharding on extra pspec dimensions ([PR #4769](https://github.com/AI-Hypercomputer/maxtext/pull/4769)) and restricted GMM quantization to fp8_full ([PR #4842](https://github.com/AI-Hypercomputer/maxtext/pull/4842)).

- **Model-Specific Fixes**:

  - Qwen3.5: Applied partial MRoPE for Qwen3.5 ([PR #4764](https://github.com/AI-Hypercomputer/maxtext/pull/4764)).
  - Mixtral: Fixed EP throughput via configurable expert-axis batch sharding ([PR #4179](https://github.com/AI-Hypercomputer/maxtext/pull/4179)).

- **NNX, MoE & MTP**:

  - Resolved silent zero-loss ([PR #4525](https://github.com/AI-Hypercomputer/maxtext/pull/4525)) and targets_segmentation bugs ([PR #4756](https://github.com/AI-Hypercomputer/maxtext/pull/4756)) in Multi-Token Prediction (MTP).
  - Preserved scanned layer intermediates for MoE load-balancing loss in NNX ([PR #4829](https://github.com/AI-Hypercomputer/maxtext/pull/4829)).
  - Relanded Qwix quantization on NNX ([PR #4198](https://github.com/AI-Hypercomputer/maxtext/pull/4198)) and fixed Qwix LoRA mesh sharding ([PR #4866](https://github.com/AI-Hypercomputer/maxtext/pull/4866)).

#### Deprecations

- **Tensor Transpose Parallelism Removed**: Completely removed the `tensor_transpose` physical mesh axis and deleted `ici_tensor_transpose_parallelism` and `dcn_tensor_transpose_parallelism` configuration options.
- **Flax Linen Deprecation Warning**: Flax Linen is now deprecated in favor of Flax NNX; running with `pure_nnx=False` or `enable_nnx=False` will issue a deprecation warning.

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

- Use the new `maxtext[runner]` installation option to build Docker images without cloning the repository. This can be used for scheduling jobs through Cluster Toolkit on GKE. See the [MaxText installation instructions](build-docker) for more info.
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
