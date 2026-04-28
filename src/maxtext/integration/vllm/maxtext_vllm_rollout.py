"""MaxText-specific VllmSampler and VllmRollout subclasses.

These replace the Tunix built-in key-mapping path with the battle-tested
MaxTextToVLLMConverter from bench_weight_sync.py, which handles:
  - QKV fusion with GQA interleaving (attention)
  - MoE expert gate+up fusion (w13_weight chunk-interleaved for TP)
  - MoE gate / down transpose
  - Layer-norm and LM-head transposes
"""

from typing import Any, Optional, Tuple

import ctypes
import functools
import gc
import logging
import time
import jax
import jax.numpy as jnp
from flax import nnx
from tunix.rl.reshard import reshard_pytree
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


def _malloc_trim():
  """Return freed Python heap pages to the OS. Counters host RSS creep."""
  try:
    ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
  except Exception:  # pylint: disable=broad-except
    pass


def _log_mem_stats(tag: str) -> None:
  """Log JAX live-array count/bytes and process RSS.

  Filter in cloud logging with:  textPayload =~ "\\[ROLLOUT_MEM\\]"
  """
  live = jax.live_arrays()
  num_arrays = len(live)
  total_bytes = sum(a.nbytes for a in live if hasattr(a, "nbytes"))
  rss_gb = 0.0
  try:
    with open("/proc/self/status") as _f:
      for _line in _f:
        if _line.startswith("VmRSS:"):
          rss_gb = int(_line.split()[1]) / 1e6
          break
  except OSError:
    pass
  logging.info(
      "[ROLLOUT_MEM] %s | live_arrays=%d jax_bytes=%.3f GB rss=%.3f GB",
      tag, num_arrays, total_bytes / 1e9, rss_gb,
  )


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
    # _log_mem_stats("sampler:update_params_start")

    # Delete KV cache to free HBM before weight conversion begins.
    # effects_barrier after the RPC ensures the deallocation is complete on
    # device before we start dispatching conversion ops.
    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")
    jax.effects_barrier()
    gc.collect()
    # _log_mem_stats("sampler:post_kv_delete")

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
    # _log_mem_stats("sampler:post_kv_reinit")
  def load_checkpoint(self, state):
    """Override load_checkpoint to use streaming update if converter is available."""
    if self._converter is not None:
      logging.info("MaxTextVllmSampler: Using streaming update_params for load_checkpoint")
      return self.update_params(state)
    else:
      return super().load_checkpoint(state)



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

    use_ep = rollout_config.expert_parallel_size > 1
    converter = MaxTextToVLLMConverter(config=maxtext_config, mesh=mesh, use_ep=use_ep)

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
