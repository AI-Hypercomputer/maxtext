## Verdict
PASS

## Critical issues (must fix)
None found.

## Suggestions (sources to add)
- The claim about `LatencyHidingScheduler` on GPU (line 249, 402) could benefit from a reference to the XLA source or documentation. It is a well-known XLA pass, but the draft does not link to any source. This is a minor suggestion, not a factual error.
- The claim about NCCL buffer allocation for collectives (lines 589-590) -- "allocated by NCCL and must persist until the collective completes" -- is standard knowledge but unsourced. Consider citing NCCL docs or XLA's GpuCollectivePermute implementation for completeness.

## Verified claims (summary)

### Named entities
- **DeepSeek-V2 (60 layers)** and **DeepSeek-V3 (61 layers)**: VERIFIED via DeepSeek-V2 paper (arXiv:2405.04434) and DeepSeek-V3 technical report (arXiv:2412.19437).
- **Multi-head Latent Attention (MLA)**: VERIFIED. Correct full name (prior draft had dropped "head"; this is fixed).
- **VMA = Varying Manual Axes**: VERIFIED via JAX documentation and Edward Yang's blog post on JAX sharding type system.

### Dates and numbers
- **29.9 GB -> 20.4 GB tmem reduction**: Matches the config file comment at ds-proxy-se2-e256-h4096.yml line 84: "reduces pipeline tmem from ~29.9 GB to ~20.4 GB". VERIFIED against source.
- **32% reduction**: (29.9 - 20.4) / 29.9 = 31.8%, rounded to 32%. VERIFIED.
- **9.5 GB total reduction**: 29.9 - 20.4 = 9.5. VERIFIED.
- **num_pipeline_repeats = 64 / (8 * 4) = 2**: Correctly derived from config values (base_num_decoder_layers=64, ici_pipeline_parallelism=8, num_layers_per_pipeline_stage=4). VERIFIED.
- **microbatches_per_stage = 64 / 8 = 8**: Correctly derived. VERIFIED.
- **micro_batch_size_to_train_on = 8 * 24 = 192**: VERIFIED (8 devices * 24 per_device_batch_size).
- **pipeline_microbatch_size = 192 / 64 = 3**: VERIFIED.
- **activation_size = 3 * 4096 * 4096 * 2 = ~96 MB**: 3 * 4096 * 4096 * 2 = 100,663,296 bytes = 96 MB. VERIFIED.
- **per_stage = 4 * 96 MB = ~384 MB**: VERIFIED.

### Config parameters
- All config values quoted in the draft (base_emb_dim: 4096, base_num_query_heads: 32, base_num_kv_heads: 8, base_num_decoder_layers: 64, head_dim: 128, mlp_activations, vocab_size: 102400, num_experts: 16, num_experts_per_tok: 4, shared_experts: 2, base_mlp_dim: 2048, base_moe_mlp_dim: 2048, ici_pipeline_parallelism: 8, num_layers_per_pipeline_stage: 4, num_pipeline_microbatches: 64, per_device_batch_size: 24, max_target_length: 4096, remat_policy: full, pipeline_save_decoder_layer_input: false): All VERIFIED against ds-proxy-se2-e256-h4096.yml.

### Code snippets
- **Pipeline._rotate_right** (slice_in_dim + concatenate): Matches pipeline.py lines 504-508. VERIFIED.
- **Pipeline._shift_right** (pad + lax.slice): Matches pipeline.py lines 510-513. VERIFIED.
- **Pipeline._update_state_io** (pad + slice_in_dim + where): Matches pipeline.py lines 557-565. VERIFIED.
- **CircularPipeline._rotate_right** (shard_map + ppermute): Matches pipeline.py lines 994-998. VERIFIED.
- **CircularPipeline._shift_right** (shard_map + ppermute + where): Matches pipeline.py lines 1000-1006. VERIFIED.
- **CircularPipeline._update_state_io** (shard_map + _shift_left): Matches pipeline.py lines 1045-1055. VERIFIED.
- **PipelineBase.get_pipeline_remat_policy()**: Matches pipeline.py lines 182-195. VERIFIED.
- **PipelineBase._maybe_shard_with_logical** with skip_trivial_specs=True: Matches pipeline.py lines 111-122. VERIFIED.
- **Pipeline.vmap_parallel_gather** using dynamic_slice_in_dim: Matches pipeline.py lines 447-448. VERIFIED.
- **Pipeline.vmap_gather** using dynamic_slice_in_dim: Matches pipeline.py lines 479-480. VERIFIED.
- **MixtralDecoderLayer shard() function** with skip_trivial_specs=True: Matches mixtral.py lines 142-150. VERIFIED.
- **checkpoint_name(inputs, "decoder_layer_input")**: Matches mixtral.py line 156. VERIFIED.
- **maybe_shard_with_logical skip_trivial_specs logic**: Matches sharding.py lines 158-159. VERIFIED.

### Field definitions in types.py
- **pipeline_save_decoder_layer_input**: Field(True, description=...): Matches types.py lines 993-999. Default True, description matches. VERIFIED.
- **float32_weight_sum**: Field(False, description=...): Matches types.py lines 748-751. Default False, description matches. VERIFIED.

### base.yml defaults
- **float32_weight_sum: false** in base.yml: Matches line 189. VERIFIED.
- **pipeline_save_decoder_layer_input: true** in base.yml: Matches line 322. VERIFIED.

### Technical claims
- **CompiledMemoryStats fields** (argument_size_in_bytes, output_size_in_bytes, temp_size_in_bytes, alias_size_in_bytes, host_temp_size_in_bytes): VERIFIED via JAX documentation.
- **Total memory formula**: argument + output + temp - alias: VERIFIED via JAX documentation.
- **SPMD partitioner runs during .compile()** (not .lower()): VERIFIED. The draft correctly places it in the compilation phase.
- **Pipeline class used when pipeline_fsdp_ag_per_repeat=false; CircularPipeline when true**: Matches create_pipeline() at pipeline.py lines 1380-1387. VERIFIED.
- **CircularPipeline still uses shard_map + ppermute**: VERIFIED by reading CircularPipeline.advance_circular_buffers() at pipeline.py lines 983-1066.

## Unverifiable
- **Exact tmem numbers (29.9 GB and 20.4 GB)**: These are empirical measurements from the config comment. I verified they match the source config file comment, but I cannot independently reproduce the measurement. The draft correctly treats them as measured values, not derived ones.
- **The claim that lax.slice guarantees a pad gradient rather than a scatter gradient**: This is a JAX implementation detail. The draft's reasoning is consistent with known JAX behavior (slice has a pad-like adjoint), but I did not trace through the JAX source to verify the exact adjoint implementation. The code comment in pipeline.py line 505 ("Use lax.slice to avoid generating a gather") and line 512 ("Use lax.slice to guarantee the gradient is a pad") corroborates this intent.
- **"~2 GB f32 temp per device" for float32_weight_sum**: This figure appears in the types.py description string but is not independently verified.

## Notes
- This draft has successfully corrected all critical issues identified in the v1 fact-check: "VMA" is now correctly expanded as "Varying Manual Axes"; DeepSeek layer counts are correct (60/61); MLA is correctly named "Multi-head Latent Attention"; the SPMD partitioner is correctly placed in the .compile() phase.
- All code snippets were verified against the actual source files. No discrepancies found.
- The draft is careful to qualify uncertainty: the savings breakdown table uses qualitative labels ("largest single contributor", "moderate", "small") rather than specific GB figures for individual optimizations, and explicitly notes that standalone impacts are not additive. This is factually responsible.
- No fabricated acronym expansions, no misattributed numbers, no incorrect code snippets detected.
