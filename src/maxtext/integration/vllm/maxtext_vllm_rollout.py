"""MaxText-specific VllmSampler and VllmRollout subclasses.

These replace the Tunix built-in key-mapping path with the battle-tested
MaxTextToVLLMConverter from bench_weight_sync.py, which handles:
  - QKV fusion with GQA interleaving (attention)
  - MoE expert gate+up fusion (w13_weight chunk-interleaved for TP)
  - MoE gate / down transpose
  - Layer-norm and LM-head transposes
"""

from typing import Any, Optional, Tuple

import functools
import gc
import logging
import time
import jax
import jax.numpy as jnp
from flax import nnx
from pathwaysutils.experimental import reshard as _experimental_reshard  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
from tunix.generate import mappings, vllm_sampler
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout, vllm_rollout


def _patch_tpu_inference_group_offset():
  """Fix tensor_parallel_gmm for rollout_data_parallelism > 1.

  Bug in tpu_inference: group_offset is hardcoded to jnp.array([0]) (shape
  [1]), but shard_map with in_specs=P('data') requires its size to be
  divisible by the mesh 'data' axis size.  When rollout_data_parallelism > 1
  the data axis size is > 1, causing a ValueError during decode:
    "4 does not evenly divide 1"

  Fix: allocate group_offset with size == data axis size so each shard
  receives exactly one zero element, preserving the original semantics.
  """
  try:
    import tpu_inference.layers.common.fused_moe_gmm as _fmgm  # pylint: disable=g-import-not-at-top
    from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top
    from tpu_inference.layers.common.sharding import ShardingAxisName  # pylint: disable=g-import-not-at-top
    from tpu_inference.utils import get_mesh_shape_product  # pylint: disable=g-import-not-at-top
  except ImportError:
    return  # tpu_inference not installed; nothing to patch.

  def _fixed_tensor_parallel_gmm(
      x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias,
      group_sizes, topk_argsort_revert_indices, topk_weights, *,
      activation, topk, mesh, **kwargs,
  ):
    data_p_spec = P(ShardingAxisName.MLP_DATA)
    # Fix: size must equal the data axis product so each shard gets 1 element.
    data_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_DATA)
    group_offset = jnp.zeros(data_size, dtype=jnp.int32)

    w1_spec = P(None, None, ShardingAxisName.MLP_TENSOR)
    w2_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
    w1_scale_spec = (None if w1_scale is None else
                     P(None, None, None, ShardingAxisName.MLP_TENSOR))
    w1_bias_spec = (None if w1_bias is None else
                    P(None, None, ShardingAxisName.MLP_TENSOR))
    num_blocks = 1 if w2_scale is None else w2_scale.shape[1]
    w2_scale_spec = (None if num_blocks == 1 else
                     P(None, ShardingAxisName.MLP_TENSOR, None, None))
    w2_bias_spec = None if w2_bias is None else P(None, None, None)

    return jax.shard_map(
        functools.partial(
            _fmgm.moe_gmm_local,
            activation=activation,
            topk=topk,
            parallelism="tp",
            **kwargs,
        ),
        mesh=mesh,
        in_specs=(
            data_p_spec, w1_spec, w1_scale_spec, w1_bias_spec,
            w2_spec, w2_scale_spec, w2_bias_spec,
            data_p_spec, data_p_spec, data_p_spec, data_p_spec,
        ),
        out_specs=(data_p_spec),
        check_vma=False,
    )(
        x, w1, w1_scale, w1_bias, w2, w2_scale, w2_bias,
        group_sizes, group_offset, topk_argsort_revert_indices, topk_weights,
    )

  _fmgm.tensor_parallel_gmm = _fixed_tensor_parallel_gmm
  logging.info(
      "Applied tpu_inference group_offset patch: tensor_parallel_gmm now "
      "allocates group_offset with size == data axis size (DP-decode fix)."
  )


_patch_tpu_inference_group_offset()


class MaxTextVllmSampler(VllmSampler):
  """VllmSampler that delegates weight updates to a MaxTextToVLLMConverter.

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
    if self._converter is None:
      return super().update_params(updated_weights, filter_types)

    del filter_types

    # delete kv_cache
    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache") # will free hbm
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")

    # Perform explicit garbage collection and synchronization to free up HBM memory before loading new weights
    gc.collect()
    jax.clear_caches()
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
      weight_array = weight.value if hasattr(weight, "value") else weight  # handle both jnp arrays and ShardedDeviceArrays
      weight_shape_matches = weight_array.shape == model_runner_state[key].shape
      assert weight_shape_matches, f"Shape mismatch for {key}: converter produced {weight_array.shape}, expected {model_runner_state[key].shape}"
      logging.info(f"{key}: mt {weight_array.shape} -> vllm {model_runner_state[key].shape}: {weight_shape_matches}")
      target_sharding = model_runner_state[key].sharding
      model_runner_state[key] = _experimental_reshard.reshard(
          weight_array, target_sharding, donate=False, may_alias=None,
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



class MaxTextVllmRollout(vllm_rollout.VllmRollout):
  """VllmRollout that uses MaxTextVllmSampler for weight synchronisation.

  The extra `maxtext_config` argument is forwarded to MaxTextToVLLMConverter
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
  ):
    # Import here to avoid a circular dependency at module load time.
    # from maxtext.bench_weight_sync import MaxTextToVLLMConverter  # pylint: disable=g-import-not-at-top
    from maxtext.integration.tunix.weight_mapping.bench_weight_sync import MaxTextToVLLMConverter  # pylint: disable=g-import-not-at-top

    # RLCluster's custom-class path doesn't pass cache_config_or_size; fall
    # back to the value embedded in rollout_config.
    if cache_config_or_size is None:
      cache_config_or_size = rollout_config.kv_cache_size

    converter = MaxTextToVLLMConverter(config=maxtext_config, mesh=mesh)

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=rollout_actor,
        backend="vllm_jax",
    )
    self._sampler = MaxTextVllmSampler(
        tokenizer=tokenizer,
        config=VllmConfig(
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
