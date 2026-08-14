import re

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/nnx_decoders.py', 'r') as f:
    content = f.read()

old_func = """  def _apply_gemma4_small_layers(
      self,
      y,
      decoder_input_tokens,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      multimodal_input=None,
      kv_caches=None,
      attention_metadata=None,
      previous_chunk=None,
      slot=None,
  ):
    \"\"\"Apply Gemma 4 small (E2B/E4B) decoder layers (pure-NNX).\"\"\"
    cfg = self.config
    bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None

    per_layer_inputs = None
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      per_layer_inputs = self.per_layer_embedder(decoder_input_tokens, y)

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    num_kv_shared = cfg.num_kv_shared_layers
    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}
    # tpu-inference allocates one kv_caches slot per non-shared layer; KV-shared layers reuse the donor's slot.
    cache_index_of = gemma4_small.kv_cache_slot_map(layer_types, num_kv_shared)

    for lyr in range(cfg.num_decoder_layers):
      layer = getattr(self, f"layers_{lyr}")
      donor_idx = gemma4_small.kv_donor_layer_idx(lyr, layer_types, num_kv_shared)
      is_donor = gemma4_small.is_kv_donor_layer(lyr, layer_types, num_kv_shared)

      shared_key = None
      shared_value = None
      if donor_idx is not None:
        if donor_idx not in shared_kv_states:
          raise RuntimeError(
              f"KV-shared layer {lyr} references donor {donor_idx} but no donor K/V "
              f"have been recorded yet. This indicates the layer iteration order is wrong."
          )
        shared_key, shared_value = shared_kv_states[donor_idx]

      # Donor layers expose their rotated, normed K/V to downstream shared layers, and reuse the
      # just-computed K/V in their own forward to avoid double-computing the K/V projection.
      if is_donor:
        donor_k, donor_v = layer.compute_shared_kv(y, decoder_positions)
        shared_kv_states[lyr] = (donor_k, donor_v)
        shared_key, shared_value = donor_k, donor_v

      ple_slice = per_layer_inputs[..., lyr, :] if per_layer_inputs is not None else None

      cache_idx = cache_index_of[lyr]
      kv_cache = kv_caches[cache_idx] if kv_caches is not None else None
      y, kv_cache = layer(
          y,
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
          shared_key=shared_key,
          shared_value=shared_value,
      )
      if kv_caches is not None and kv_cache is not None:
        kv_caches[cache_idx] = kv_cache

    return y, kv_caches"""

new_func = """  def _apply_gemma4_small_layers(
      self,
      y,
      decoder_input_tokens,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      multimodal_input=None,
      kv_caches=None,
      attention_metadata=None,
      previous_chunk=None,
      slot=None,
  ):
    \"\"\"Apply Gemma 4 small (E2B/E4B) decoder layers (pure-NNX).\"\"\"
    from jax.experimental import xla_metadata
    cfg = self.config
    bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None

    per_layer_inputs = None
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      per_layer_inputs = self.per_layer_embedder(decoder_input_tokens, y)

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    num_kv_shared = cfg.num_kv_shared_layers
    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}
    
    cache_index_of = gemma4_small.kv_cache_slot_map(layer_types, num_kv_shared)

    remat_policy = self.get_remat_policy()
    if remat_policy is not None:
      save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg)
      if offload_names[0] or offload_names[1]:
        remat_policy = jax.checkpoint_policies.save_only_these_names(*(save_names + offload_names[0] + offload_names[1]))

    for lyr in range(cfg.num_decoder_layers):
      layer = getattr(self, f"layers_{lyr}")
      donor_idx = gemma4_small.kv_donor_layer_idx(lyr, layer_types, num_kv_shared)
      is_donor = gemma4_small.is_kv_donor_layer(lyr, layer_types, num_kv_shared)

      shared_key = None
      shared_value = None
      if donor_idx is not None:
        if donor_idx not in shared_kv_states:
          raise RuntimeError(
              f"KV-shared layer {lyr} references donor {donor_idx} but no donor K/V "
              f"have been recorded yet. This indicates the layer iteration order is wrong."
          )
        shared_key, shared_value = shared_kv_states[donor_idx]

      ple_slice = per_layer_inputs[..., lyr, :] if per_layer_inputs is not None else None
      cache_idx = cache_index_of[lyr]
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

    return y, kv_caches"""

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/nnx_decoders.py', 'w') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old function not found")
