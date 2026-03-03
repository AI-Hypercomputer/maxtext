# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strategy pattern for model-specific layer application in Decoder."""

from abc import ABC, abstractmethod
from typing import Any, List

from flax import linen as nn
import jax
import jax.numpy as jnp


class ModelStrategy(ABC):
  """Abstract base class for model-specific layer application strategies."""

  def get_pipeline_layer(self, block_layers: List[Any]) -> Any:
    """Returns the layer class to be wrapped in the pipeline stage."""
    return block_layers[0]

  @abstractmethod
  def apply_layers(
      self,
      decoder_instance,
      y: jax.Array,
      block_layers: List[Any],
      broadcast_args: tuple,
      **kwargs
  ) -> jax.Array:
    """Applies the model's layers to the input `y`.

    Args:
      decoder_instance: The Decoder instance applying the layers (provides access to config, mesh, scan_decoder_layers, etc.).
      y: The input hidden states.
      block_layers: The list of layer classes / modules for this model.
      broadcast_args: Arguments to be broadcasted to all layers (e.g. positions, segment_ids, deterministic flag, mode).
      **kwargs: Additional model-specific kwargs (kv_caches, page_state, etc.).

    Returns:
      The output hidden states.
    """
    pass


class DefaultStrategy(ModelStrategy):
  """Standard strategy that applies layers sequentially directly or via scan."""

  def apply_layers(
      self,
      decoder_instance,
      y: jax.Array,
      block_layers: List[Any],
      broadcast_args: tuple,
      **kwargs
  ) -> jax.Array:
    cfg = decoder_instance.config
    mesh = decoder_instance.mesh
    model_mode = kwargs.get("model_mode")
    kv_caches = kwargs.get("kv_caches")
    attention_metadata = kwargs.get("attention_metadata")
    previous_chunk = kwargs.get("previous_chunk")
    page_state = kwargs.get("page_state")
    slot = kwargs.get("slot")
    pipeline_module = kwargs.get("pipeline_module")
    bidirectional_mask = kwargs.get("bidirectional_mask")
    deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")

    if cfg.using_pipeline_parallelism and pipeline_module is not None:
      logical_partition_spec = None
      if cfg.pipeline_fsdp_ag_once:
        logical_partition_spec = pipeline_module.get_weight_sharding(
            y, *broadcast_args
        )
      y = pipeline_module(y, *broadcast_args, logical_partition_spec=logical_partition_spec)

      remaining_layers = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
      if remaining_layers > 0:
        # Import sharding locally to avoid circular import if necessary, though it seems safe here
        from maxtext.utils import sharding
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
        with mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          y, _ = decoder_instance.scan_decoder_layers(
              cfg,
              block_layers[0],
              remaining_layers,
              "layers_outside_pipeline",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
          )(y, *broadcast_args)
      return y

    if cfg.scan_layers:
      scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
      RemattedBlockLayer = block_layers[0]
      y, _ = decoder_instance.scan_decoder_layers(
          cfg,
          RemattedBlockLayer,
          scan_length,
          "layers",
          mesh,
          in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
          model_mode=model_mode,
      )(y, *broadcast_args)
      return y

    # Unscanned fallback
    from maxtext.layers import decoders  # Import locally to avoid circular dependency
    for lyr in range(cfg.num_decoder_layers):
      RemattedBlockLayer = block_layers[0]
      layer_kwargs = {}
      layer_call_kwargs = {}

      # Model-specific injections could go via specialized strategies
      layer = RemattedBlockLayer(
          config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=decoder_instance.quant, model_mode=model_mode, **layer_kwargs
      )
      kv_cache = kv_caches[lyr] if kv_caches is not None else None
      y, kv_cache = layer(
          y,
          *broadcast_args, # contains segment_ids, positions, deterministic, model_mode
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          bidirectional_mask=bidirectional_mask,
          **layer_call_kwargs,
      )
      if kv_caches is not None and kv_cache is not None:
        kv_caches[lyr] = kv_cache

      if deepstack_visual_embeds is not None and lyr < len(deepstack_visual_embeds):
        visual_embeds = deepstack_visual_embeds[lyr]
        # Use bidirectional_mask to identify visual token positions
        if bidirectional_mask is not None and visual_embeds is not None:
          y = decoders.deepstack_process(y, bidirectional_mask, visual_embeds)

    return y
