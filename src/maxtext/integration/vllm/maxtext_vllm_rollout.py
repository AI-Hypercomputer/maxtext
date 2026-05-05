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

"""MaxText-specific VllmSampler and VllmRollout subclasses.

These replace the Tunix built-in key-mapping path with model-specific
MaxText to vLLM converters, which handle:
  - QKV fusion with GQA interleaving (attention)
  - MoE expert gate+up fusion (w13_weight chunk-interleaved for TP)
  - MoE gate / down transpose
  - Layer-norm and LM-head transposes
"""

from typing import Any, Optional, Tuple

import gc
import logging
import time
import jax
from flax import nnx
from pathwaysutils.experimental import reshard as _experimental_reshard
from tunix.generate import mappings
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout, vllm_rollout

from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter


def _create_model_converter(model_name: str, config: Any, mesh: jax.sharding.Mesh):
  """Instantiate the converter for a MaxText model name."""
  if model_name in {"qwen3-30b-a3b", "qwen3-30b-a3b-base", "qwen3-235b-a22b"}:
    return Qwen3MaxTextToVLLMConverter(config=config, mesh=mesh)

  raise ValueError(f"No MaxText->vLLM converter registered for model {model_name!r}.")


class MaxTextVllmSampler(VllmSampler):
  """VllmSampler that delegates weight updates to a MaxText to vLLM converter.

  When a converter is supplied, update_params bypasses transfer_state_with_mappings
  entirely and instead runs converter.convert() followed by a direct device_put
  into the vLLM model-runner state dict.  If no converter is supplied the base-class
  behaviour is preserved, so this class is safe to use as a drop-in replacement.
  """

  def __init__(
      self,
      tokenizer: Any,
      config: VllmConfig,
      converter: Any = None,
  ):
    super().__init__(tokenizer=tokenizer, config=config)
    self._converter = converter

  def update_params(
      self,
      updated_weights,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    """Update the vLLM runner weights from a MaxText state tree."""
    if self._converter is None:
      super().update_params(updated_weights, filter_types)
      return None

    del filter_types

    # Delete KV cache to free HBM before weight conversion begins.
    # effects_barrier after the RPC ensures the deallocation is complete on
    # device before we start dispatching conversion ops.
    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache")  # will free hbm
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")

    # Perform explicit garbage collection and synchronization to free up HBM memory before loading new weights
    gc.collect()
    jax.effects_barrier()

    logging.info("MaxTextVllmSampler.update_params: starting converter.convert()...")
    vllm_state = self._converter.convert(updated_weights)
    jax.block_until_ready(vllm_state)

    logging.info("MaxTextVllmSampler.update_params: converter.convert() done, %d weights to assign", len(vllm_state))
    model_runner_state = self.transformer_state
    assigned = 0
    start_time = time.time()

    # Stream weights one-at-a-time: convert a chunk → reshard → assign → free.
    # This avoids holding all converted weights in HBM simultaneously, which
    # was the cause of OOMs at hbm_utilization > 0.5.  Peak HBM is now bounded
    # to one w13 chunk (8 layers) + attention weights, not the full model.
    for _layer_idx, weight_dict in self._converter.convert_streaming_per_layer(updated_weights):
      jax.block_until_ready(weight_dict)
      target_sharding_tree = {}
      for key, weight_array in weight_dict.items():
        weight_array = weight_array.value if hasattr(weight_array, "value") else weight_array
        assert weight_array.shape == model_runner_state[key].shape, (
            f"Shape mismatch for {key}: converter produced {weight_array.shape}, "
            f"expected {model_runner_state[key].shape}"
        )
        target_sharding = model_runner_state[key].sharding
        if weight_array.ndim == 1 and len(target_sharding.spec) > 1:
          target_sharding = jax.sharding.NamedSharding(target_sharding.mesh, jax.sharding.PartitionSpec())
        target_sharding_tree[key] = target_sharding

      for key in weight_dict:
        if key in model_runner_state:
          model_runner_state[key].delete()
      model_runner_state.update(reshard_pytree(
          weight_dict, target_sharding_tree, donate_input=False, cache_plan=True,
          use_experimental_pre_reshard=True
      ))
      for key in weight_dict:
        jax.block_until_ready(model_runner_state[key])
        assigned += 1
      del weight_dict, target_sharding_tree
      if assigned % 4 == 0:
        jax.effects_barrier()
        gc.collect()
        _malloc_trim()

    jax.effects_barrier()
    gc.collect()
    _malloc_trim()
    # Release the converter's internal state so it doesn't hold HBM across steps.
    self._converter.vllm_state = {}
    logging.info("MaxTextVllmSampler.update_params: %d weights assigned in %.4f s",
                 assigned, time.time() - start_time)
    # _log_mem_stats("sampler:post_assign_all")

    # Reinitialize KV cache.
    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

  def load_checkpoint(self, state):
    """Override load_checkpoint to use streaming update if converter is available."""
    if self._converter is not None:
      logging.info("MaxTextVllmSampler: Using streaming update_params for load_checkpoint")
      return self.update_params(state)
    else:
      return super().load_checkpoint(state)

    return None


class MaxTextVllmRollout(vllm_rollout.VllmRollout):
  """VllmRollout that uses MaxTextVllmSampler for weight synchronisation.

  The extra `maxtext_config` argument is forwarded to the model-specific converter
  together with `mesh`.  All other arguments mirror VllmRollout.__init__.

  Usage (direct):
      rollout = MaxTextVllmRollout(
          rollout_actor=tunix_model,
          tokenizer=tokenizer,
          mesh=mesh,
          rollout_config=rollout_config,
          maxtext_config=maxtext_config,   # <-- new
      )

  Usage via RLCluster (recommended):
      cluster_config = ClusterConfig(
          ...
          rollout_engine=functools.partial(MaxTextVllmRollout, maxtext_config=maxtext_config),
          ...
      )
  """

  def __init__(
      self,
      rollout_actor: Any,
      tokenizer: Any,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
      maxtext_config: Any,
      cache_config_or_size: base_rollout.CacheConfig | int = None,
  ):  # pylint: disable=super-init-not-called,too-many-positional-arguments
    # RLCluster's custom-class path doesn't pass cache_config_or_size; fall
    # back to the value embedded in rollout_config.
    if cache_config_or_size is None:
      cache_config_or_size = rollout_config.kv_cache_size

    converter = _create_model_converter(maxtext_config.model_name, config=maxtext_config, mesh=mesh)

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=rollout_actor,
        backend="vllm_jax",
    )
    self._sampler = MaxTextVllmSampler(
        tokenizer=tokenizer,
        config=VllmConfig(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            mesh=mesh,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            mapping_config=mapping_config,
            lora_config=rollout_config.rollout_vllm_lora_config,
            server_mode=rollout_config.rollout_vllm_server_mode,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            engine_kwargs={
                "max_model_len": cache_config_or_size,
                "model": rollout_config.rollout_vllm_model_version,
                "swap_space": rollout_config.rollout_vllm_swap_space_size_gb,
                # Async scheduling causes KeyError in dp_scheduler on slow models
                # (30B+) where inference latency exceeds the scheduler's window.
                "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
            },
        ),
        converter=converter,
    )

    # Initial weight sync: run the converter so vLLM starts with real weights.
    state = nnx.state(rollout_actor)
    self._sampler.load_checkpoint(state)
