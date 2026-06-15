## Verdict
NEEDS_REVISION

## Critical issues (must fix)

1. **"check_vma (Verified Memory Annotation)"** (Section 6.1, paragraph about `shard_map` with `check_vma=True`): The draft says "`check_vma` (Verified Memory Annotation) flag forces the compiler to verify that the output sharding matches the declared `out_specs`." The acronym expansion is wrong. VMA stands for **Varying Manual Axes**, not "Verified Memory Annotation." VMA is a type system that tracks whether values are varying or invariant across mesh dimensions. While the general description of its effect (forcing verification and potentially requiring materialization) is approximately correct in spirit, the expansion of the acronym is fabricated. Must fix.

2. **"The real DeepSeek-V2/V3 uses MLA (Multi-Latent Attention), a specialized decoder block, and 160+ layers."** (Section 1, callout box): Two errors here:
   - DeepSeek V2 has **60 layers** and DeepSeek V3 has **61 layers**, not "160+ layers." The number 160 appears to be confused with the number of routed experts per MoE layer in DeepSeek V2 (which is 160).
   - MLA stands for **Multi-head Latent Attention**, not "Multi-Latent Attention." This is a named architectural component introduced in DeepSeek V2 and the full name matters for technical credibility.

3. **"The SPMD partitioner runs during this phase"** (Section 3.3, describing the lowering phase `.lower()`): The SPMD partitioner does **not** run during lowering. Lowering (`.lower()`) converts Jaxpr to StableHLO with sharding annotations embedded as metadata. The SPMD partitioner runs during **compilation** (`.compile()`), as part of XLA's `RunHloPasses()` stage. The entire section 3.3 heading is "Lowering: Jaxpr to HLO" and the claim that the SPMD partitioner runs "during this phase" is incorrect. It should be moved to section 3.4 or clarified that the SPMD partitioner runs at compile time.

## Suggestions (sources to add)

- The savings breakdown table (Section 12) provides per-optimization estimates that sum to roughly 16-20.5 GB, while the measured total is 9.5 GB. The text correctly notes these don't add up due to XLA's global buffer sharing, but it might be clearer to explicitly state "the sum of individual estimates exceeds the measured reduction because optimizations interact."

- The claim about `LatencyHidingScheduler` being used "on GPU" (Section 2) is technically correct for this config (`hardware: gpu`), but could note that LHS also runs on TPU, to avoid implying it is GPU-specific.

## Verified claims (summary)

- **Config parameters**: All values from `ds-proxy-se2-e256-h4096.yml` quoted in Section 1 match the actual YAML file exactly: `base_emb_dim: 4096`, `base_num_query_heads: 32`, `base_num_kv_heads: 8`, `base_num_decoder_layers: 64`, `head_dim: 128`, `vocab_size: 102400`, `decoder_block: mixtral`, `num_experts: 16`, `num_experts_per_tok: 4`, `shared_experts: 2`, `base_mlp_dim: 2048`, `base_moe_mlp_dim: 2048`, `ici_pipeline_parallelism: 8`, `num_layers_per_pipeline_stage: 4`, `num_pipeline_microbatches: 64`, `per_device_batch_size: 24`, `max_target_length: 4096`, `remat_policy: full`, `pipeline_save_decoder_layer_input: false`. VERIFIED.

- **Pipeline class selection**: "The `Pipeline` class ... is instantiated because `pipeline_fsdp_ag_per_repeat` is `false`." Confirmed by `create_pipeline()` in pipeline.py line 1380-1386. VERIFIED.

- **Pipeline repeat calculation**: `num_pipeline_repeats = 64 / (8 * 4) = 2`. Matches the formula (decoder_layers / (stages * layers_per_stage)). VERIFIED.

- **`microbatches_per_stage = 64 / 8 = 8`**: Confirmed by pipeline.py line 56. VERIFIED.

- **`get_pipeline_remat_policy()` code**: The draft's quoted code for `names_to_save = ["iteration_input"]` and the conditional append of `"decoder_layer_input"` matches pipeline.py lines 187-190 exactly. VERIFIED.

- **`checkpoint_name` usage in mixtral.py**: `inputs = checkpoint_name(inputs, "decoder_layer_input")` matches mixtral.py line 156. VERIFIED.

- **`_rotate_right` implementation**: The `slice_in_dim` + `concatenate` code matches pipeline.py lines 506-508 exactly. VERIFIED.

- **`_shift_right` implementation**: The `pad` + `lax.slice` code matches pipeline.py lines 511-513 exactly. VERIFIED.

- **`_update_state_io` implementation**: Matches pipeline.py lines 557-565. VERIFIED.

- **`vmap_parallel_gather` using `dynamic_slice_in_dim`**: Confirmed at pipeline.py line 448: `jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)`. VERIFIED.

- **`vmap_gather` using `dynamic_slice_in_dim`**: Confirmed at pipeline.py line 480. VERIFIED.

- **`skip_trivial_specs` in `PipelineBase._maybe_shard_with_logical`**: Confirmed at pipeline.py line 122: `skip_trivial_specs=True`. VERIFIED.

- **`skip_trivial_specs` in `MixtralDecoderLayer.shard()`**: Confirmed at mixtral.py line 149: `skip_trivial_specs=True`. VERIFIED.

- **`skip_trivial_specs` implementation in sharding.py**: The check `all(ax is None or ax == () for ax in named_sharding.spec)` matches sharding.py line 158. VERIFIED.

- **`pipeline_save_decoder_layer_input` field definition in types.py**: Default `True`, description matches. Confirmed at types.py lines 993-1000. VERIFIED.

- **`float32_weight_sum` field in types.py**: Default `False`, matches types.py line 748-751. VERIFIED.

- **`float32_weight_sum` in base.yml**: Value is `false`, matches base.yml line 189. VERIFIED.

- **`pipeline_save_decoder_layer_input` default in base.yml**: Value is `true`, matches base.yml line 322. VERIFIED.

- **`CompiledMemoryStats` fields**: The 5 fields (`argument_size_in_bytes`, `output_size_in_bytes`, `temp_size_in_bytes`, `alias_size_in_bytes`, `host_temp_size_in_bytes`) are confirmed by JAX documentation. VERIFIED.

- **Total memory formula**: `argument + output + temp - alias` is confirmed by JAX documentation. VERIFIED.

- **CircularPipeline still uses `shard_map` + `ppermute`**: Confirmed by `advance_circular_buffers()` at pipeline.py lines 994-998. VERIFIED.

- **32% reduction**: (29.9 - 20.4) / 29.9 = 31.77%, rounds to 32%. VERIFIED.

- **`shard` function in `MixtralDecoderLayer.__call__`**: The shard function is defined as a local function using `maybe_shard_with_logical` with `skip_trivial_specs=True`. Matches mixtral.py lines 142-150. VERIFIED.

- **`CircularPipeline` uses `check_vma=True` in its `shard_map` calls**: Confirmed at pipeline.py lines 994, 1000, 1045. VERIFIED.

## Unverifiable

- **Tmem values (~29.9 GB before, ~20.4 GB after)**: These are empirical measurements from compiling the specific config. Cannot independently verify without running the compilation, but the config file's comment at line 84 (`# reduces pipeline tmem from ~29.9 GB to ~20.4 GB`) corroborates these numbers.

- **Per-optimization savings estimates** (e.g., "~5-7 GB" for ppermute replacement, "~7-9 GB" for decoder_layer_input): The draft correctly labels these as estimates. Cannot verify without controlled experiments.

- **Claim that `lax.slice` gradient is always a `pad`**: This is a well-known property of JAX's autodiff for slicing operations, but I did not independently verify the specific JAX source code. The reasoning is sound.

## Notes

- The draft's description of the `pipeline_save_decoder_layer_input` flag says the default is `true` which "preserves upstream behavior for all existing users" -- this is consistent with both types.py and base.yml.

- The draft's code snippets for the CircularPipeline's `_rotate_right`, `_shift_right`, and `_update_state_io` (in the "Before" sections) accurately reflect the CircularPipeline's `advance_circular_buffers()` method.

- The draft describes `meta.remove_axis` usage in the `Pipeline` class for circular pipeline weights. Checking the actual code, `meta.remove_axis` IS still used in the `Pipeline` class at line 622 (within `get_current_repeat_from_stages`) and at line 684 (within `run_one_iteration` for `num_pipeline_repeats > 1`). The draft's claim that "meta.remove_axis" was removed appears to be about a different code path or partially inaccurate -- the weight metadata strip is still present. However, the draft frames it as "Removed meta.remove_axis on Weights" referring specifically to the weight preparation in the main iteration path. Since the code still has `meta.remove_axis` calls in both `get_current_repeat_from_stages` and `run_one_iteration`, the "removed" framing may be misleading. The draft may be describing changes relative to an older version not present in this codebase.
