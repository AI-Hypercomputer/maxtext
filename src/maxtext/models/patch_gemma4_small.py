import re

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/models/gemma4_small.py', 'r') as f:
    content = f.read()

block_code = """
class Gemma4SmallScannableBlock(nnx.Module):
  \"\"\"A scannable block that encapsulates all Gemma 4 Small decoder layers to prevent XLA OOMs.\"\"\"

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    self.layer_types = build_layer_types(config.num_decoder_layers, config.model_name)
    self.num_kv_shared = config.num_kv_shared_layers
    self.cache_index_of = kv_cache_slot_map(self.layer_types, self.num_kv_shared)

    for lyr in range(config.num_decoder_layers):
      attention_type = self.layer_types[lyr]
      layer = Gemma4SmallDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          rngs=rngs,
          quant=quant,
          attention_type=attention_type,
          layer_idx=lyr,
      )
      setattr(self, f"layers_{lyr}", layer)

  def __call__(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      per_layer_inputs=None,
      kv_caches=None,
      attention_metadata=None,
      previous_chunk=None,
      slot=None,
      bidirectional_mask_value=None,
      remat_policy=None,
  ):
    from jax.experimental import xla_metadata
    from maxtext.utils import maxtext_utils
    cfg = self.config

    if remat_policy is not None:
      save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg)
      if offload_names[0] or offload_names[1]:
        remat_policy = jax.checkpoint_policies.save_only_these_names(*(save_names + offload_names[0] + offload_names[1]))

    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}

    for lyr in range(cfg.num_decoder_layers):
      layer = getattr(self, f"layers_{lyr}")
      donor_idx = kv_donor_layer_idx(lyr, self.layer_types, self.num_kv_shared)
      is_donor = is_kv_donor_layer(lyr, self.layer_types, self.num_kv_shared)

      shared_key, shared_value = None, None
      if donor_idx is not None:
        if donor_idx not in shared_kv_states:
          raise RuntimeError(f"KV-shared layer {lyr} references missing donor {donor_idx}.")
        shared_key, shared_value = shared_kv_states[donor_idx]

      ple_slice = per_layer_inputs[..., lyr, :] if per_layer_inputs is not None else None
      cache_idx = self.cache_index_of[lyr]
      kv_cache = kv_caches[cache_idx] if kv_caches is not None else None

      graphdef, intermediates, other_state = nnx.split(layer, nnx.Intermediate, ...)
      intermediate_xs = jax.tree.map(lambda x: x[None], intermediates)

      def run_layer_fn(carry, intermediate_slice):
        hidden_carry, state_carry = carry
        merged_layer = nnx.merge(graphdef, intermediate_slice, state_carry)

        cur_shared_key, cur_shared_value = shared_key, shared_value
        donor_k, donor_v = None, None
        if is_donor:
          donor_k, donor_v = merged_layer.compute_shared_kv(hidden_carry, decoder_positions)
          cur_shared_key, cur_shared_value = donor_k, donor_v

        out_y, out_kv = merged_layer(
            hidden_carry,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            previous_chunk=previous_chunk,
            slot=slot,
            bidirectional_mask=bidirectional_mask_value,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
            per_layer_input=ple_slice,
            shared_key=cur_shared_key,
            shared_value=cur_shared_value,
        )

        _, new_intermediates, new_other = nnx.split(merged_layer, nnx.Intermediate, ...)
        return (out_y, new_other), (new_intermediates, out_kv, donor_k, donor_v)

      if remat_policy is not None and cfg.remat_policy != "none":
        prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)
        run_layer_fn = jax.checkpoint(run_layer_fn, policy=remat_policy, prevent_cse=prevent_cse)

      with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
        (y, final_other_state), (stacked_intermeds, out_kv_cache, donor_k, donor_v) = jax.lax.scan(
            run_layer_fn,
            (y, other_state),
            intermediate_xs,
            length=1,
        )

      final_intermeds = jax.tree.map(lambda x: x[0], stacked_intermeds)
      nnx.update(layer, final_other_state, final_intermeds)

      if is_donor:
        shared_kv_states[lyr] = (donor_k[0], donor_v[0])
      
      if kv_caches is not None and out_kv_cache is not None:
        kv_caches[cache_idx] = jax.tree.map(lambda x: x[0], out_kv_cache)

    return y, kv_caches

Gemma4SmallScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Gemma4SmallScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
"""

if "Gemma4SmallScannableBlock" not in content:
    content += "\n" + block_code
    with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/models/gemma4_small.py', 'w') as f:
        f.write(content)
    print("Added block successfully")
else:
    print("Block already exists")
