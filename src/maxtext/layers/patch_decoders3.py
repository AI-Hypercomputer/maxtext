import re

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/decoders.py', 'r') as f:
    content = f.read()

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
    \"\"\"Apply Gemma 4 small (E2B / E4B) decoder layers.\"\"\"
    cfg = self.config
    mesh = self.mesh
    bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None

    per_layer_inputs = None
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      per_layer_inputs = gemma4_small.PLEToLinen(
          config=cfg,
          mesh=mesh,
          name="per_layer_embedder",
      )(decoder_input_tokens, y)

    policy = self.get_remat_policy()
    
    y, kv_caches = self.decoder_layer[0](
        config=cfg,
        mesh=mesh,
        name="scanned_blocks",
        quant=self.quant,
        model_mode=self.model_mode,
    )(
        y,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        per_layer_inputs=per_layer_inputs,
        kv_caches=kv_caches,
        attention_metadata=attention_metadata,
        previous_chunk=previous_chunk,
        slot=slot,
        bidirectional_mask_value=bidirectional_mask_value,
        remat_policy=policy,
    )

    return y, kv_caches"""

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
    \"\"\"Apply Gemma 4 small (E2B / E4B) decoder layers.

    Threads per-call state through the layer loop:
      * ``per_layer_inputs`` from PLE, sliced per layer.
      * ``shared_kv_states``: donor-layer-index → (key, value) for
        downstream KV-shared layers to consume.
      * ``kv_caches``: when running via the vLLM RPA path, the per-layer
        cache buffer threaded back from the kernel. KV-shared layers
        redirect to the donor's cache slot via ``cache_index_of``.

    Returns ``(y, kv_caches)``. Scan-over-layers and pipeline
    parallelism are not supported.
    \"\"\"
    cfg = self.config
    mesh = self.mesh
    bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None
  
    policy = self.get_remat_policy()
    RemattedGemma4SmallDecoderLayer = self.set_remat_policy([gemma4_small.Gemma4SmallDecoderLayerToLinen], policy)[0]

    per_layer_inputs = None
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      per_layer_inputs = gemma4_small.PLEToLinen(
          config=cfg,
          mesh=mesh,
          name="per_layer_embedder",
      )(decoder_input_tokens, y)

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    num_kv_shared = cfg.num_kv_shared_layers
    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}

    # tpu-inference allocates one `kv_caches` slot per non-shared layer;
    # KV-shared layers reuse the donor's slot.
    cache_index_of = gemma4_small.kv_cache_slot_map(layer_types, num_kv_shared)

    for lyr in range(cfg.num_decoder_layers):
      attention_type = layer_types[lyr]
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

      layer = RemattedGemma4SmallDecoderLayer(
          config=cfg,
          mesh=mesh,
          name=f"layers_{lyr}",
          quant=self.quant,
          model_mode=self.model_mode,
          attention_type=attention_type,
          layer_idx=lyr,
      )

      # Donor layers expose their rotated, normed K / V to downstream
      # shared layers via the decoder layer's compute_shared_kv method.
      if is_donor:
        donor_k, donor_v = layer(y, decoder_positions, nnx_method="compute_shared_kv")
        shared_kv_states[lyr] = (donor_k, donor_v)
        # Reuse the just-computed K / V in the layer's own forward pass to
        # avoid double-computing the K / V projection / norm / RoPE.
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

if old_func in content:
    content = content.replace(old_func, new_func)
    
    # Also update the decoder_block mapping
    old_mapping = "case DecoderBlockType.GEMMA4_SMALL:\n        # PLE input + KV-share donor threading requires per-layer-index state,\n        # which is not expressible inside ``nn.scan``.\n        return [gemma4_small.Gemma4SmallDecoderLayerToLinen]"
    new_mapping = "case DecoderBlockType.GEMMA4_SMALL:\n        return [gemma4_small.Gemma4SmallScannableBlockToLinen]"
    content = content.replace(old_mapping, new_mapping)

    with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/decoders.py', 'w') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old function not found")
