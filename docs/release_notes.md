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

**Last Updated**: 0bcf71871

<!-- Add new unreleased changes below this line -->

#### Changes

##### Models

- **Qwen3.8 Architecture & Parameter Mappings**: Added model configuration and HuggingFace parameter mapping for Qwen3.8-2.4T-A95B, parameterizing `hf_prefix` in Qwen3.5 mapping and shape builders to support Qwen3.8 variants.
- **M3 Default Sharding**: Added `DefaultSharding` for m3 (Modern MaxText Models) and aligned mesh axis names with legacy MaxText conventions.
- **DeepSeek-V4 Layer Rematerialization**: Honored `remat_policy` in DeepSeek-V4 train-mode layers by running unscanned prefix layers under `jax.checkpoint` within a length-1 scan barrier and applying layer-level rematerialization inside `DeepSeek4ScannableBlock`, avoiding block-level rematerialization to fit large configurations into device HBM.
- **Lineage DeepSeek-V3 Modules & Mesh Sharding**: Synced Lineage DeepSeek-V3 modules into `maxtext/experimental/lineage`, including the branched Splash Attention kernel and 1D expert-parallel sharding; looked up Lineage DeepSeek-V3 weight shardings dynamically by mesh topology; aligned the package with the code that produced the DeepSeek-V3 671B MLPerf Training v6.1 8k-chip result so it runs on released `jax 0.11.1` / `libtpu 0.0.48`; and fed the decoder's final-norm output into the Lineage MTP block, mirroring `mtp_norm_main_hidden_state` (always on for the Lineage path).
- **DeepSeek-V3 Lineage Architecture**: Added Lineage DeepSeek-V3 modules under `maxtext/experimental/lineage` with the exported `deepseek3-671b-lineage` configuration and `embed_router` rule. Lineage is a very efficient implementation of DeepSeek-V3 targeting `pdb=1` using a batch split strategy where the EP comms are fully overlapped (overlapped with splash). Also imported Lineage modules lazily in `lineage_adapter` when `use_lineage=False`, added optional model hooks for alternative DeepSeek-V3 implementations, and added `lineage_expert_permutation` for per-step expert load balancing in the Lineage MoE path.
- **DeepSeek-V4 Multi-Token Prediction**: Added Multi-Token Prediction (MTP) support for DeepSeek-V4 decoder layers along with HuggingFace parameter mapping and shape alignment utilities.
- **WeaverOmni Architecture**: Added the full 36-layer `WeaverOmniTransformer` backbone architecture and tests, supporting the `weaver` and `weaver-mini-diffuser` model configurations.
- **Qwen3.5-9B Model Support**: Added base model onboarding for Qwen3.5-9B, including model configurations, HuggingFace checkpoint conversion, decode workflows, and logit verification.
- **Flax NNX Migration**: Ported the remaining Linen sharding helpers into native MaxText implementations (`sharding.logical_to_spec` and `sharding.with_logical_constraint`), replacing Flax Linen dependencies across KV cache, attention ops, and decoder architectures. Additionally, converted FP8 ops (`Fp8DirectDotGeneralOp`, `NANOOFp8DotGeneralOp`, `Fp8Einsum`, `QwixDotGeneral`, `QwixEinsum`) and HuggingFace checkpoint parameter conversion (`to_maxtext.py`) to native Flax NNX modules, and deleted legacy `ToLinen` layer wrappers and factories.

##### Pre-Training

- **Splash Attention FP8 Scaling**: Added the experimental `sa_q_prescaled_log2e` (folds `log2(e)` into the float32 Q softmax scale before the FP8 cast, so Q is rounded to FP8 once and in-kernel Q scaling is skipped) and `sa_dkv_reuse_scaled_q` (the fused dKV kernel reuses the `log2(e)`-scaled Q for dK) Splash Attention options, both off by default.
- **Splash Attention Stream Chunking Optimization**: Added `sa_ignore_segment_ids` to skip uniform segment mask loading and comparisons in Splash Attention when running continuous stream-chunked `c4_mlperf` datasets, eliminating redundant VPU overhead while preserving position IDs and loss masking.
- **YaRN Cis Optimization**: Built YaRN rotary position embedding `freqs_cis` values directly from `complex(cos(theta), sin(theta))` instead of `jnp.exp(1j * theta)`, avoiding XLA complex exponential expansion and redundant exponential evaluations under explicit sharding.
- **Multi-Token Prediction Hidden State Normalization**: Added `mtp_norm_main_hidden_state` configuration flag to apply decoder normalization before the output head and feed the normalized hidden state directly into the Multi-Token Prediction (MTP) block, aligning with Megatron-LM DeepSeek-V3 behavior.
- **JAX Flash & Splash Attention**: Added the JAX Flash and Splash Attention backward pass supporting Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA) with block sparsity, causal masking, soft capping, and packed-sequence segment IDs, and exposed `sa_qk_diag_skip`, `sa_sv_diag_skip`, and `sa_qk_diag_grid` configuration flags in Tokamax Splash Attention.
- **YaRN RoPE Optimizations**: Applied YaRN RoPE rescaling to DeepSeek-V4 compressed attention (CSA/HCA) layers matching the HuggingFace configuration. Also added `rope_direct_position_freqs` to compute YaRN RoPE angles directly from position frequencies in 4D float32 without lookup tables, and added `rope_dot_pairwise` using a 4D block-skew contraction matrix to preserve rank-4 activations during pairwise rotation and eliminate layout copy overhead.
- **DiLoCo Distributed Training**: Added `split_mesh_along_axis` and `remove_mesh_axis_from_rules` to derive per-replica views by splitting meshes along the `diloco` axis and stripping the axis from logical rules, and introduced `diloco_bucketize_non_scanned` to optionally bucketize non-scanned parameter matrices across layer fragments along their longest unsharded axis to balance data transfer.

##### Post-Training

- **RL Profiling & Managed ML Diagnostics**: Integrated the Tunix RL trainer with Tunix's per-role, invocation-counted profiler (`RLProfileConfig`) through the new `rl.profiler_start_invocation`, `rl.profiler_num_invocations`, `rl.profiler_timeout_secs`, and `rl.profiler_max_num_hosts` keys (the last also traces disaggregated sampler hosts on Pathways), producing one `<role>_invocation_<n>` trace per role. With `managed_mldiagnostics=true`, the trainer creates the Managed ML Diagnostics run before the `RLCluster`, uploads the trainer config (plus differing `sampler.*` keys on multi-slice topologies), and installs `MLDiagScalarBackend` as a metrics backend. Added `managed_mldiagnostics_metric_only_run` to create metric-only runs.
- **Decode Context Parallelism for vLLM**: Added Decode Context Parallelism (DCP) support for Qwen3.5 in `MaxTextForCausalLM` via the vLLM adapter.
- **Block-Diffusion Fine-Tuning**: Extended the block-diffusion objective to Continued Fine-Tuning (CFT) and completion-only Supervised Fine-Tuning (SFT), integrating with Tunix diffusion training and corrupting only configured supervision scopes while preventing future-prompt leakage.
- **Continuous Profiling**: Added the `enable_continuous_profiling` configuration flag across profilers and post-training pipelines (SFT, DPO, RL, and Distillation) to capture profiles for long-running workloads.

##### Multimodal

- **Qwen3.5-9B Multimodal Support**: Enabled `use_multimodal=true` for Qwen3.5-9B by adding its Qwen3.5 vision-encoder (`vision_encoder_block: qwen3_5`) and MRoPE settings to `qwen3.5-9b.yml` and registering the model with the multimodal processor and decoder alongside the other Qwen3.5 variants.
- **Qwen3-VL Integration**: Added M3-style `qwen3-vl-2b` modeling implementation and layer correctness comparison tests.
- **Wan 2.1 Video Diffusion Scaffolding**: Added M3 architecture skeleton and module stubs for diffusion models and Wan 2.1 text-to-video (`wan2.1-t2v-14b`), including Diffusers schedulers and pipeline utilities.

##### Performance

- **FSDP Weight Prefetching & Gathered Weight Caching**: Added `prefetch_fsdp_weights` for scanned NNX decoder layers using a `custom_vjp` double-buffering pipeline that overlaps layer k+1 weight all-gathers with layer k computation in forward and layer k-1 all-gathers during backward recomputation ([fsdp prefetch](https://maxtext.readthedocs.io/en/latest/guides/optimization/fsdp_prefetch.html)); supported symmetric `jax.lax.scan(..., unroll=2)`; extended prefetching to honor layer-level rematerialization policies; and introduced `save_fsdp_gathered_weights` to retain gathered FSDP kernels in compute dtype across forward and backward passes.
- **Kernel-Order Weight Gradients & LM Head Optimization**: Introduced `weight_grad_in_kernel_order` in `DenseGeneral`, `lm_head_weight_grad_in_kernel_order`, and `dense_weight_grad_in_kernel_order` to route matmuls through `jax.sharding.auto_axes` under explicit sharding, allowing XLA to fold activation transposes into gradient matmuls and recover all-reduce to reduce-scatter fusion; contracted tied LM head logits directly over embedding feature axes (`einsum("...e,ve->...v")`) to eliminate redundant type casts and transposes; and added opt-in `lm_head_fsdp_as_vocab_tp` to shard untied LM heads along vocabulary dimensions using FSDP axes for small-batch efficiency.
- **Explicit Sharding Expansion & Performance Guide**: Onboarded GPT-3, GPT-OSS, Llama4, and Gemma 4 decoders to explicit sharding with loss parity against auto sharding (bit-for-bit at step 0 and under tensor parallelism, within bf16 reassociation tolerance thereafter), and published a comprehensive performance guide detailing HLO differences, step-time comparisons, and explicit sharding tuning flags ([optimization](https://maxtext.readthedocs.io/en/latest/guides/optimization.html), [shard mode performance](https://maxtext.readthedocs.io/en/latest/guides/optimization/shard_mode_performance.html)).
- **TransformerEngine MoE Sharding & Quantization**: Added TransformerEngine MoE checkpoint policies, exposed optional quantization before the TE MoE FSDP weight all-gather (`te_moe_quantize_before_fsdp_all_gather`), passed explicit mesh resources, and enabled configurable dispatch rematerialization (`te_moe_dispatch_remat_location`).
- **MoE Logical Sharding & 1D Expert Parallelism**: Expressed joint expert and FSDP placement with existing logical axes via the `expert-on-fsdp-ep` custom mesh and rules, and updated sharding configurations to support 1D expert parallelism.
- **Lineage DeepSeek-V3 Quantization & MLPerf Performance**: Introduced `fp8_full_splash_qk_e3_combine_per_hop` Lineage quantization combining FP8 Splash Attention Q/K with per-hop FP8 combine reduce-scatter; fused per-hop token absmax into SparseCore FP8 combine reduce-scatter; exposed FP8 Q/K Splash and Splash-skip flags; enabled async TensorCore ragged all-to-all in Lineage DeepSeek-V3; supported gradient accumulation (`gradient_accumulation_steps > 1`) with explicit `ragged_buffer_factor` overrides; enabled `warm_eval_input_reshard_before_run_start` and `eval_cache_prefill_in_background` by default in the `deepseek3-671b-lineage` recipe and set `use_stream_chunking: true` so Lineage Splash Attention skips the no-op segment mask on stream-chunked `c4_mlperf` training rows; supported fractional eval sharding (`0 < eval_per_device_batch_size < 1`) using expert axes as context parallelism; capped `mllog` `eval_samples` at the 1024-sequence validation set size and reported `fp8` for `lowest_numerical_precision_in_attn` when Splash Q/K FP8 is enabled; and removed `lineage_expert_permutation: "lpt"` from the default Lineage DeepSeek-V3 recipe.
- **TensorCore MoE Ragged Sort & Backward Quantization**: Enhanced TensorCore ragged sort with `[start, start+cap)` windowed routing, uint32 packed sorting, and `unique_indices` weight gathering, and fused MoE combine backward quantization prior to TC unsort-gather while skipping 3D DLHS zero-initialization.
- **Quantization & Qwix Optimizations**: Disabled Qwix weight-gradient matmul tiling by default via `qwix_tile_weight_grad_matmuls`, supported `use_sliced_mla_proj` with Qwix interception quantization by preserving dot-general interception across sliced projection kernels, and added a configuration flag to keep backward passes in BF16 for Q projections during quantized training.
- **Pre-Training Initialization Warmup**: Added `warmup_programs_in_init` to execute precompiled train and eval programs on synthetic batches during setup to warm JIT dispatch caches before timed execution, with options to skip dropless training warmup, and added `log_step_diagnostics` for step-level timing breakdown.
- **TensorCore Ragged Sort & 3D Layout Pipeline for Ring-of-Experts MoE**: Added a TensorCore ragged sort implementation for Ring-of-Experts MoE, applied MoE routing weights to activations alongside the ragged sort, maintained the sorted MoE token buffer in 3D layout through GMM v2 kernels, and enabled token all-gather and combine in the TensorCore kernels' native 3D layout.
- **In-Place MoE Gradient Accumulation**: Added `moe_accumulate_wi_dlhs` to accumulate `wi_0` and `wi_1` backward DLHS gradients in-place via `gmm_v2` `partial_sum` with fused DLHS scale, and introduced `moe_accumulate_chunk_wgrad` to accumulate weight gradients across token chunks in-place via `tgmm_v2` `partial_sum`.
- **GMM v2 Native Transpose RHS**: Added native `transpose_rhs` support to Tokamax GMM v2 block specs and tiled matmuls to consume expert weights in their existing layout and eliminate RHS layout copies.
- **FP8 Quantization**: Added `logits_proj_bwd_quant_calibration_method` allowing custom backward quantization calibration overrides for logits projections ([quantization](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/quantization.html)), and added the `lineage_quantization` (`none` or `fp8_full`) option to support quantization for Lineage DeepSeek-V3 routed experts.
- **MLPerf DeepSeek-V3 Optimization**: Added the `fsdp-as-dp-for-attn-rf` custom mesh and sharding rule to shard MoE router weights on FSDP and overlap gradient communication with GMM backward execution ([custom mesh and rule](https://maxtext.readthedocs.io/en/latest/guides/optimization/custom_mesh_and_rule.html)), and added `warm_eval_input_reshard_before_run_start` and `eval_cache_prefill_in_background` flags to overlap eval input caching with initial training steps.

##### Checkpointing / Goodput

- **In-Memory Snapshots for Elastic Training**: Wired up `elastic_backup_kind: snapshot` (the default) so elastic runs keep a copy of the train state and data-iterator position in pinned host memory every `elastic_snapshot_interval` steps (new, default 10) and restore from it when it is at least as recent as the latest checkpoint. Snapshot mode requires `dataset_type=grain` with `grain_use_elastic_iterator=True` (or synthetic data) and is rejected at config time with DiLoCo, `enable_rampup_batch_size`, or `colocated_python_data_input`; such runs must set `elastic_backup_kind: checkpoint`.
- **NNX & Flat MoE Checkpoint Resharding**: Updated `reshard_checkpoint.py` to build the abstract train-mode NNX model for the target mesh (restoring every weight training saves, including `MoEBiasVar`), support flat MoE layouts (`moe_flat_fsdp_weights=True`), read each stored chunk once instead of once per device to cut host memory on large meshes, and preserve each weight's stored dtype. `convert_deepseek_family_ckpt.py` gained `--weight_dtype keep` to retain source dtypes (e.g., DeepSeek-V3's float32 `e_score_correction_bias`).
- **Goodput Monitoring for RL & Distillation**: Instrumented the RL and Distillation trainers with ML Goodput recording and monitoring hooks (job start/end, TPU init, training preparation, and step start events via `GoodputTrainingHooks`), honoring the existing `enable_goodput_recording`, `monitor_goodput`, `goodput_job_name`, and `enable_pathways_goodput` flags, which are now also listed in `post_train/rl.yml`.
- **Multi-Tier Checkpointing Restore Documentation**: Documented procedures for restoring Multi-Tier Checkpointing jobs from specific backup checkpoints on Google Cloud Storage ([multi tier checkpointing](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/multi_tier_checkpointing.html)).
- **Goodput Monitoring & Logging**: Integrated the Orbax v1 logger into Goodput monitoring, and added sliding window tracking for goodput metrics to accurately evaluate sustained throughput and performance trends over time.
- **Pathways Checkpointing**: Added opt-in colocated-Python checkpoint reads for Pathways single-controller runs across all parameter restoration paths (including metadata read) and introduced the `train_lib_tpu` build target.
- **Elastic Training**: Unified all elastic training modes to use `elastic_retry` as the single entry point.

##### Usability

- **Cluster Toolkit Benchmark Migration**: Migrated the MaxText benchmark runner and configurations from XPK to Cluster Toolkit (`gcluster`), adding `ClusterConfig` in `benchmarks/ctk_configs.py`, `gcluster job submit` orchestration with kubectl completion polling in `benchmarks/maxtext_ctk_runner.py`, and retaining backward-compatible XPK re-export shims; `benchmark_runner.py` gained `ctk`/`gcluster` subcommands with `--location`, `--compute_type`, `--topology`, and `--mount` flags.
- **Dependency Upgrades**: Upgraded `tokamax` to 0.0.16 for TPU environments (the GPU extra stays pinned at 0.0.14) ([install maxtext](https://maxtext.readthedocs.io/en/latest/install_maxtext.html)), raised the `orbax-checkpoint` floor to `>=0.12.8`, and raised `google-cloud-mldiagnostics` to `>=1.0.9`.
- **Decision Models & Scoring Harness**: Added `vllm_score` for JEV-style decision-making LLMs, enabling next-token probability distribution extraction, candidate continuation scoring, and batch evaluation ([tutorials](https://maxtext.readthedocs.io/en/latest/tutorials.html), [decision models](https://maxtext.readthedocs.io/en/latest/tutorials/decision_models.html), [inference](https://maxtext.readthedocs.io/en/latest/tutorials/inference.html)).
- **Configuration Support**: Updated `pyconfig` to accept configuration files with `.yaml` extensions alongside `.yml`.
- **Documentation & Guides**: Updated ReadTheDocs links across documentation and tutorial notebooks to point to default versioned builds via redirects.
- **Dependencies & Compatibility**: Updated pre-training dependency requirements to require `flax>=0.12.10` to ensure compatibility with JAX 0.11.2 ([update dependencies](https://maxtext.readthedocs.io/en/latest/development/update_dependencies.html)).

#### Bug Fixes

- Fixed DeepSeek-V4 decode drifting from train-mode forward passes after padded prefill by deriving prefill true length from segment IDs in compressed attention, properly sizing compressed AR cache, and correcting sliding-window decode masks measured from token positions across DeepSeek-V4, Gemma2, and GPT-OSS.
- Fixed DeepSeek router numerics by applying SwiGLU input clamping (`mlp_activations_limit`) to shared experts, retaining float32 precision for routed gate bias (`bias_dtype`) and source non-bf16 tensors in `deepseek_dequantize.py`, and zero-initializing dropped token rows in combine ragged all-to-all when `ragged_buffer_factor > 0`.
- Fixed XLA arithmetic fusion breakage and numerical drift in Gemma4 under `ShardMode.AUTO` by strictly restricting physical `out_sharding` and explicit sharding constraints to `ShardMode.EXPLICIT`.
- Fixed `native_fp8_dot_general` quantization by using per-token activation scaling in `per_tensor` mode and properly skipping quantization for modules specified in `unquantized_modules`.
- Fixed `safetensors_dynamic` checkpoint loading keeping single-key weights (embedding, final norm, LM head) in the source checkpoint dtype instead of casting them to the target leaf's `weight_dtype`, and fixed `-Instruct` model names being stripped before the `HF_IDS` lookup (which silently loaded the base model's weights); exact model names are now required.
- Fixed TPU SparseCore collective offload stalls during gradient accumulation by applying the `_xla_loop_unroll_strategy="double-buffer"` attribute only on GPU hardware, preserving TensorCore `compute_on` collective placement on TPU.
- Fixed duplicated evaluation sequences when multiple eval hosts share a single C4 evaluation file in `tfds_data_processing_c4_mlperf.py`.
- Fixed `Profiler.deactivate()` raising `RuntimeError: No profile started` on early `StopTraining` teardown when profiling windows have already closed.
- Fixed distillation training by aligning `_train_step` signatures, slicing steps properly, and propagating `distill_beta` to the teacher model.
- Fixed `to_maxtext` conversion of scanned DeepSeek checkpoints (MoE gate-bias stacking and `MoEBiasVar`/`Tid2EidVar` key detection), added DeepSeek MTP-layer parameter mappings, and made Grain iterator worker cleanup robust in `multihost_dataloading.py`; migrated the DeepSeek end-to-end tests to `to_maxtext`.
- Fixed scheduled execution failures in the `dpo_qwen3_demo_no_eval.ipynb` example notebook.
- Fixed loss calculation zeroing the language model loss when indexer loss is disabled or unscaled (`indexer_loss_scaling_factor == 0`).
- Fixed DeepSeek-V4 logit checker forward pass execution under mesh JIT, token space decode assertion, and Linen side collection restoration (`Tid2EidVar`).
- Fixed post-training hooks swallowing non-`StopIteration` data loading exceptions, ensuring unexpected loader errors surface immediately ([#5523](https://github.com/AI-Hypercomputer/maxtext/issues/5523)).
- Fixed scanned decoders with inhomogeneous layer cycles silently dropping remainder layers or crashing when `num_decoder_layers` is not an exact multiple of `inhomogeneous_layer_cycle_interval`, raising an explicit `ValueError`.
- Fixed `gmm_v2` partial K tail handling when `tile_k` does not divide `size_k` by masking stale VMEM in the LHS operand, preventing NaN/Inf poisoning of the accumulator and block absmax quantization.
- Fixed MLPerf parallelism, precision, and config filename MLLog disclosure logging for Lineage DeepSeek-V3 runs.
- Fixed SFT evaluation invoking the loss function with `is_train=True`, ensuring dropout, MTP loss, and sparsity `batch_stats` updates are disabled during eval.

#### Deprecations

- Removed experimental Lineage hooks from decoder, MTP, and MoE layers, and removed the `lineage_integration` configuration flag.
- Deprecated calling `from_config()` without explicit `rngs` and deprecated `models.transformer_as_linen` in favor of native Flax NNX module initialization.

## Releases

### v0.2.5

#### Changes

##### Models

- **DeepSeek-V3 Lineage Integration**: Integrated Lineage DeepSeek-V3 (`dsv3.py`) into MaxText, storing routed expert output weights in Lineage's layout and sowing router bias updates along the `use_lineage` path.
- **DeepSeek-V4 & Multi-Token Prediction**: Supported DeepSeek-V4 checkpoint conversion and parameter mapping with 2D transposed shape alignment, added DeepSeek-V4 indexer loss, and implemented separate final normalization for Multi-Token Prediction (MTP).
- **Weaver Architecture**: Added Weaver Mixture-of-Transformers block and standardized Weaver model naming across configs.
- **Qwen3.5 & Qwen3-Next**: Added `qwen3.5-35b-a3b-fp8` and `qwen3.5-397b-a17b-fp8` model configs with HuggingFace parameter mapping and weight conversion, and optimized Qwen3-Next shared expert construction to only instantiate when `shared_experts > 0`.
- **Cosmos 3 Bring-Up**: Added model definitions and bring-up support for Cosmos 3 Nano and Cosmos 3 Super Reasoner.
- **Normalization Enhancements**: Added RMSNorm normalization layer support for Apple Envy decoder blocks and OLMo 3 MLP blocks (`DecoderBlockType.OLMO3`).
- **GPT-OSS Attention Sink**: Added attention-sink support for GPT-OSS models.
- **M3 Core Architecture**: Added m3 package scaffolding with architecture rules and layout verification tests, introducing m3 core RoPE implementation, mesh creation, and sharding rules.

##### Pre-Training

- **JAX Flash Attention Optimization**: Enhanced JAX Flash Attention kernel with a specialized high-performance execution path for long sequences (splitting at a 4,096 sequence length threshold).
- **DeepSeek-V4 CSA & Splash Attention**: Added fused Pallas TPU StreamIndex score kernel, QK attention head chunking for Compressed Self-Attention (CSA) memory reduction, and HCA static compilation for Splash Attention on DeepSeek-V4.
- **Context Parallelism (CP)**: Sharded GatedDeltaNet sequences under `ici_context_parallelism`, enabled native block-sparse execution for indexer masks under CP, removed all-to-all communication overhead from TSP/CP segmentation masks, and split embedding logical names in attention modules.
- **Multimodal RoPE (MRoPE)**: Eliminated redundant JAX recompilations in `get_mrope_input_positions` using inline NumPy operations, and normalized MRoPE position shapes for vLLM across hybrid cache utilities and embeddings.
- **DiLoCo Distributed Training**: Added comprehensive documentation and tutorials for DiLoCo pre-training ([core concepts](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts.html), [diloco](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/diloco.html), [tutorials](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials.html)), and migrated DiLoCo launch scripts and tutorials from XPK to Cluster Toolkit ([diloco pretraining](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/diloco_pretraining.html)).
- **Quantized Multi-Token Prediction**: Enabled quantized Multi-Token Prediction (MTP) for accelerated pre-training.

##### Post-Training

- **Reinforcement Learning (GRPO) Infrastructure**: Plumbed GRPO flags to `AgenticGRPOLearner`, added RL sequence packing and vLLM block-size configuration knobs, added `free_kv_cache_during_weight_sync`, `reward_num_workers`, and `gc_collect_after_weight_sync` settings, made `extract_answer` linear in response length, and added a Gemma 4 26B GRPO tutorial ([post training index](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/post_training_index.html), [rl gemma4 26b](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/rl_gemma4_26b.html), [rl gemma4 e4b](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/rl_gemma4_e4b.html)).
- **LoRA & Distillation Workflows**: Refactored LoRA resharding to modernize NNX variable access and graph traversal, migrated full fine-tuning scripts from TFDS to Grain ([full finetuning](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/full_finetuning.html)), added launch scripts and configs for Distillation Training, and integrated MaxText's trainer backend into a distributed GSM8K example.
- **vLLM Rollout Enhancements**: Added Mamba prefix caching for Qwen3 GDN layers in vLLM, indexed KV caches via `layer_name_to_kvcache_index` in the vLLM adapter, and implemented MoE 128-lane chunking and padding for GMM layout.

##### Multimodal

- **Multimodal SFT Pipeline**: Added Grain-based multimodal Supervised Fine-Tuning (SFT) data processing pipeline ([multimodal](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/multimodal.html)), end-to-end video SFT supporting variable frame sizes and durations, and ChartNet SFT configuration with string response handling in multimodal input pipelines.
- **Omni Pipeline & Qwen3-VL Inference**: Added custom multimodal processor and end-to-end pipeline for Omni multimodal models, and enabled `vllm_decode` inference support for Qwen3-VL ([inference](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/inference.html)).

##### Performance

- **Explicit Sharding & ZeRO-1 Expansion**: Onboarded explicit sharding and ZeRO-1 gradient accumulation across Qwen3.5, Qwen3-Next, Qwen3, Qwen2, Kimi-K2, Mistral, Mixtral, and Gemma model families; supported partitioned optimizer state sharding propagation and MaskedNode filtering in ZeRO-1; deferred data-parallel gradient all-reduce to `update()` under GA with backward cotangents scaled to GA=1.
- **Sharded Muon Optimizer**: Integrated a sharding-aware, distributed Muon optimizer into MaxText with generalized weight dimension extraction, support for Qwen3 and GPT-OSS MoE blocks, and a configuration flag to govern applying Muon to MoE routers.
- **FP8 Quantization & Dequantization**: Implemented an FP8 weight-only dynamic dequantization engine with direct FP8 and scale tensor ingestion in `to_maxtext`, native FP8 inference (`serve_fp8_weight`) for DenseGeneral and GMM v2, weight-gradient-arm cotangent calibration (`drhs_grad_quantization_calibration_method`), quantized logits projection ([quantization](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/quantization.html)), pre-quantized weights for TPU inference fused MoE kernels, and options to keep MoE router projections unquantized for stability ([moe configuration](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/moe_configuration.html), [quantization](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/quantization.html)).
- **Quantized Ring-of-Experts Pipeline**: Enabled token-quantized all-gather pipelines with Ring of Experts and added quantized all-gather for the Ring of Experts combine backward ([moe configuration](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/moe_configuration.html), [quantization](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/quantization.html)).
- **MoE Dropless Fallback & Buffer Resilience**: Implemented configurable MoE dropless fallback (`moe_dropless_fallback: None|step|layer`) with memory-neutral step fallback, first-phase ragged buffers, host-driven step replay retry mechanisms for buffer overflows, evaluation-specific buffer scaling (`eval_ragged_buffer_factor`), and evaluation tile sizes and QKV layouts ([moe configuration](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/moe_configuration.html)).
- **MoE Routing & Kernel Optimizations**: Added dense custom VJPs for router top-k selection (eliminating backward scatters), optimized top-2 expert group score computation, aligned sigmoid-router auxiliary losses and expert bias updates with Megatron-LM, added collective psum reductions for expert bias load balancing, and supported extracting and replaying router expert decisions between inference and training ([moe configuration](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/moe_configuration.html)).
- **MoE Kernels & Sharding**: Integrated TransformerEngine (TE) MoEBlock, supported Tokamax GMM v2 heuristic tiling functions, forwarded TPU inference MoE kernel knobs, saved routed expert inputs across remat boundaries via `moe_x_sorted`, added nested scanning over hybrid attention with per-layer remat in Qwen3-Next, and added explicit tracking for `shard_embed_moe_on_fsdp` ([moe configuration](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/core_concepts/moe_configuration.html)).
- **SparseCore & Pallas Accelerations**: Added `moe_pin_sparse_core_all_gathers` to offload FSDP and Expert Parallel all-gathers to SparseCores, lowered `plsc.bitcast` to `tpu.bitcast` when layout passes are enabled, and refactored the MaxText mHC kernel to utilize `emit_pipeline`.

##### Checkpointing / Goodput

- **Orbax v1 Checkpointing**: Refactored MaxText's checkpointing architecture to natively utilize the Orbax v1 API (retiring legacy v0 implementations), added dequantize-on-load parameter restoration, and added support for restoring checkpoint weights stored outside the standard `"params"` collection.
- **Goodput & Pathways Persistence**: Added `goodput_job_name` configuration option for Goodput monitoring, registered Pathways persistence array handlers with bounded host staging, and introduced standalone checkpointer benchmarking loop features and configuration flags.
- **Checkpointing Documentation**: Migrated multi-tier and emergency checkpointing guides from XPK to Cluster Toolkit ([emergency checkpointing](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/checkpointing_solutions/emergency_checkpointing.html), [multi tier checkpointing](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/checkpointing_solutions/multi_tier_checkpointing.html)).

##### Usability

- **MLPerf Training Compliance & Optimizations**: Added full MLPerf training logging compliance, accelerated step-0 initialization time via per-file `c4_mlperf` train sharding and data-free setup, added host-local eval batch caching, implemented continuous stream chunking for C4 MLPerf datasets, and added the `dsv3-mlperf-4k` custom mesh and sharding rule.
- **Data Pipeline Enhancements**: Added Megatron MMap data format support for Grain data loaders with an integrated mmap index builder ([data input pipeline](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/data_input_pipeline.html), [data input grain](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/data_input_pipeline/data_input_grain.html), [data input megatron mmap](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/data_input_pipeline/data_input_megatron_mmap.html)), allowed Grain to ingest TFDS configurations, optimized data input pipelines for evaluation, and set default `dataset_type` to `synthetic`.
- **Documentation & Cluster Toolkit Migration**: Migrated core navigation, getting started, and execution guides from XPK to Cluster Toolkit ([getting started](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/getting_started.html), [install maxtext](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/install_maxtext.html), [architecture overview](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/reference/architecture/architecture_overview.html)), updated container image registry references from GCR to Artifact Registry ([run maxtext elastic training](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/run_maxtext/run_maxtext_elastic_training.html), [run maxtext via pathways](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/run_maxtext/run_maxtext_via_pathways.html), [lora on multi host](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/lora_on_multi_host.html)), and published TPU pre-training and post-training Docker images for MaxText in the build tutorial ([build maxtext](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/build_maxtext.html)).
- **Dependencies & Developer Tooling**: Upgraded JAX dependencies to 0.11.1 ([update dependencies](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/development/update_dependencies.html)), Tokamax to 0.0.14, and DrJAX to 0.2.1; added custom GCS path support for ML Diagnostics telemetry; added Gemini CLI development skills; streamlined multi-worker XLA dumping (`--dump_xla_all_hosts`); and defined default rule constants in `types.py` with base configuration parity testing.

#### Bug Fixes

- Fixed FP8 and quantized MoE numerical paths across dense and sparse matmul implementations.
- Fixed MoE weight tracking, vLLM HBM memory allocation, and restored `wo` in Tunix's zero-pad key set during RL rollouts.
- Fixed RLConfig mesh initialization AttributeError and resolved defects on the Tunix adapter and engine packing path.
- Fixed vLLM sampler initialization failures and corrected `cache_heads` and `paged_kv_heads` logical axis order in `vllm.yml`.
- Unblocked Qwen3.5 hybrid decode through the vLLM adapter and fixed the standalone torchax converter for TPU inference rollouts.
- Fixed standalone checkpointer restore loop to use Orbax v1 and ensured fatal checkpointing errors raise `RuntimeError`.
- Resolved Pyrefly typing errors related to JAX scalar types across trainers and Megablox kernels.
- Derived `cp_size` dynamically from active logical axis rules and computed local block padding for dynamic Splash indexer masks under Context Parallelism.
- Fixed GDN Mamba block table indexing under prefix caching and resolved argument mismatches in compact Mamba block override calls.
- Fixed dropped routing-weight gradients in `ring_ragged_unsort` and aligned dropless-replay remat overrides with the NNX decoder.
- Configured layout passes on SparseCore Pallas ragged gather and ragged gather-reduce kernels.
- Eliminated XLA backward adjoint interior padding in `reorder_sequence`.
- Fixed duplicate XLA compilation of `jit(train_step)`.
- Fixed vocab tiling hidden-state cotangent scaling.
- Fixed FP8 custom-gradient state overwrite during NNX training.
- Fixed NNX throughput regression when running with `scan_layers=False`.
- Fixed XLA fusion breakage and numerical drift in Qwen3 under AUTO sharding mode.
- Fixed edge cases in `AttentionOp.generate_attention_mask` for DeepSeek-V4 compressed attention across chunked prefill, packed sequences, and standalone environments.
- Fixed Qwen3-VL tokenizer type mismatch and resolved Grain `IterDataset` crashes on multimodal data.
- Ensured `weight_dtype` is propagated correctly in Qwen3-Next sparse MoE shared expert gate.
- Fixed EOS token detection and model routing for Qwen vision models in multimodal evaluation.
- Fixed GCS path resolution by skipping redundant `exists()` checks on Google Cloud Storage paths.
- Allowed warm starts from `load_parameters_path` without requiring checkpointing enabled.
- Fixed MoE Data Parallel reduce-scatter and logical axis sharding in the vLLM serving path.
- Fixed Ahead-Of-Time (AOT) compilation for ragged kernels and supported internal compilation and orchestration workflows.
- Fixed C4 MLPerf evaluation input pipeline handling for fractional evaluation batch sizes.
- Resolved Grain dataloader starvation and process leak on slice recovery.
- Fixed chat template formatting and SFT prompt/completion masking for Gemma 4 reasoning.
- Fixed `ImportError` on optional `tensorflow_text` in post-training workflows.
- Explicitly set `enable_checkpointing` when loading state in DiLoCo tests.
- Fixed `AttributeError` in MHC-lite during shape evaluation and sharding construction.
- Fixed inference query head rule mapping and aligned QKV batch dimensions for cuDNN flash attention.

#### Deprecations

- **Flax Linen Removal**: Removed legacy Flax Linen module code, dead code, and shared-primitive imports, deprecating and removing `pure_nnx`, `enable_nnx`, and `pure_nnx_decoder` configuration flags, Linen decoder/attention layers, and `*_as_linen` model wrappers across trainers, inference dispatch, quantization, and sharding ([distillation](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/distillation.html), [knowledge distillation](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/knowledge_distillation.html), [lora](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/lora.html)).
- Replaced deprecated `google-cloud-sdk` package with `google-cloud-cli` in Dockerfiles ([run maxtext via xpk](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/run_maxtext/run_maxtext_via_xpk.html), [rl on multi host](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/rl_on_multi_host.html), [sft on multi host](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/tutorials/posttraining/sft_on_multi_host.html)).
- Made TensorFlow an optional dependency in requirements ([update dependencies](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/development/update_dependencies.html), [data input tfds](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/data_input_pipeline/data_input_tfds.html), [install maxtext](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/install_maxtext.html)).

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
  - Added RL support for Qwen3 30B and GPT-OSS 20B. See the [Qwen3 30B RL tutorial](https://maxtext.readthedocs.io/en/maxtext-v0.2.4/tutorials/posttraining/rl_qwen3_30b.html) and [GPT-OSS 20B RL tutorial](https://maxtext.readthedocs.io/en/maxtext-v0.2.4/tutorials/posttraining/rl_gptoss_20b.html) for recipes.
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
- **New vLLM-Powered Evaluation Framework**: Introduced an eval framework for running lm-eval, evalchemy, and custom benchmarking against MaxText checkpoints. See the [evaluation guide](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/guides/eval_framework.html) for details.
- Added support for pre-training new models:
  - **Qwen3.5**: Qwen3.5 35B & 397B is now [supported](https://github.com/AI-Hypercomputer/maxtext/blob/d938b91acaa3baaaf32956e21677bd29e14549a1/tests/end_to_end/tpu/qwen/moe/run_qwen_moe.md).
  - **Qwen3-Omni**: Support for multimodal SFT ([PR #3863](https://github.com/AI-Hypercomputer/maxtext/pull/3863)).
- **Direct Preference Optimization (DPO/ORPO) Support**: Full support for DPO and ORPO alignment pipelines. See the [DPO tutorial](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/tutorials/posttraining/dpo.html) for details.
- **Reinforcement Learning (RL) Recipe**: Added a pre-configured [RL recipe for Qwen3-30b-a3b](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/tutorials/posttraining/rl_qwen3_30b.html).
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
- **Documentation Improvements**: Updated [Getting started](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/getting_started.html) guide, including new guides for the [evaluation framework](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/guides/eval_framework.html) and the [DPO tutorial](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/tutorials/posttraining/dpo.html).

#### Deprecations

- Deleted [legacy DPO implementation](https://github.com/AI-Hypercomputer/maxtext/pull/3997) in favor of the integrated [DPO trainer](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/tutorials/posttraining/dpo.html).
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
