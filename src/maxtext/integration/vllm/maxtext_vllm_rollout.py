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

  def load_checkpoint(self, path_or_weights):
    """Override to route through MaxTextVllmSampler.update_params, not the base class."""
    self.update_params(updated_weights=path_or_weights, filter_types=None)

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

    # delete kv_cache
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

    logging.info("Weight sync: using pathwaysutils experimental_reshard")

    keys = list(vllm_state.keys())
    start_time = time.time()
    for i, key in enumerate(keys):
      weight = vllm_state.pop(key)  # free immediately to avoid accumulating all weights in RAM
      weight_array = (
          weight.value if hasattr(weight, "value") else weight
      )  # handle both jnp arrays and ShardedDeviceArrays
      weight_shape_matches = weight_array.shape == model_runner_state[key].shape
      assert (
          weight_shape_matches
      ), f"Shape mismatch for {key}: converter produced {weight_array.shape}, expected {model_runner_state[key].shape}"
      # logging.info(f"{key}: mt {weight_array.shape} -> vllm {model_runner_state[key].shape}: {weight_shape_matches}")
      target_sharding = model_runner_state[key].sharding
      model_runner_state[key] = _experimental_reshard.reshard(
          weight_array,
          target_sharding,
          donate=True,
          may_alias=None,
          cache_resharding_plans=True,
      )
      del weight, weight_array  # release TPU buffer before pushing back to device
      # Periodically flush async ops and GC to prevent host RAM accumulation.
      if i % 16 == 15:
        jax.effects_barrier()
        gc.collect()
    jax.effects_barrier()
    gc.collect()
    end_time = time.time()
    logging.info("MaxTextVllmSampler.update_params: all weights assigned in %.4f seconds", end_time - start_time)

    # reinitialize kv_cache
    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

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
