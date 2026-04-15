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
import io
import logging
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from pathwaysutils.experimental import reshard as _experimental_reshard  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
from tunix.generate import mappings, vllm_sampler
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout, vllm_rollout


def _save_arrays_to_gcs(
    mt_array: np.ndarray,
    vllm_array: np.ndarray,
    gcs_dir: str = "gs://hengtaoguo-maxtext-logs/weights",
    flat: bool = True,
) -> None:
  """Save mt_array and vllm_array as .npy files to a GCS bucket.

  Args:
    mt_array: The MaxText weight array to save.
    vllm_array: The vLLM weight array to save.
    gcs_dir: GCS directory prefix (no trailing slash).
    flat: If True, iterate over the first (expert) dimension and save each
      expert slice as a separate file.  If False, save each array whole.
  """
  try:
    import gcsfs  # pylint: disable=g-import-not-at-top
    fs = gcsfs.GCSFileSystem()
    if flat:
      num_experts = mt_array.shape[0]
      for expert_idx in range(num_experts):
        for name, arr in (
            ("tmp_mt_array", mt_array),
            ("tmp_vllm_array", vllm_array),
        ):
          path = f"{gcs_dir}/{name}_{expert_idx}.npy"
          buf = io.BytesIO()
          np.save(buf, arr[expert_idx])
          buf.seek(0)
          with fs.open(path, "wb") as f:
            f.write(buf.read())
      logging.info(
          "Saved %d experts for tmp_mt_array and tmp_vllm_array to %s",
          num_experts, gcs_dir,
      )
    else:
      for name, arr in (
          ("tmp_mt_array", mt_array),
          ("tmp_vllm_array", vllm_array),
      ):
        path = f"{gcs_dir}/{name}.npy"
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        with fs.open(path, "wb") as f:
          f.write(buf.read())
      logging.info(
          "Saved tmp_mt_array and tmp_vllm_array to %s", gcs_dir,
      )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("_save_arrays_to_gcs failed: %s", e)


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
    _log_mem_stats("sampler:update_params_start")

    # Perform explicit garbage collection and synchronization to free up HBM memory before loading new weights.
    # NOTE: jax.clear_caches() must NOT be called here. It evicts all compiled XLA executables, forcing
    # every @jax.jit function in the converter (including the expensive vmap+jit over all MoE layers) to
    # recompile from scratch on every update_params call. HBM arrays are freed by dropping Python references
    # + gc.collect() + jax.effects_barrier() -- clear_caches() adds no benefit and causes growing sync time.
    gc.collect()
    jax.effects_barrier()
    _log_mem_stats("sampler:post_gc_clear")

    # Use stream_convert(): yields (key, weight) pairs one at a time so each weight is resharded
    # and assigned immediately, keeping peak HBM at ~2x model size (actor params + vLLM in-place
    # update) instead of the old 3x peak (actor params + full converted dict + vLLM state all live
    # simultaneously). No KV cache teardown needed as this matches the native path's memory profile.
    logging.info("MaxTextVllmSampler.update_params: starting streaming weight conversion and assignment...")
    model_runner_state = self.transformer_state
    start_time = time.time()
    i = 0
    for key, weight_array in self._converter.stream_convert(updated_weights):
      assert weight_array.shape == model_runner_state[key].shape, (
          f"Shape mismatch for {key}: converter produced {weight_array.shape}, "
          f"expected {model_runner_state[key].shape}"
      )
      # Logging for correctness check
      if "layers.0" in key:
        tmp_mt_array = np.asarray(weight_array)
        tmp_vllm_array = np.asarray(model_runner_state[key])
        allclose = np.allclose(tmp_mt_array, tmp_vllm_array, rtol=1e-1, atol=1e-1)
        logging.info(f"{key}: allclose = {allclose}")
        if "qkv_proj" in key and not allclose:
          _save_arrays_to_gcs(tmp_mt_array, tmp_vllm_array, "gs://hengtaoguo-maxtext-logs/weights_qkv_proj", flat=False)
        # if "w13_weight" in key and not allclose:
        #   _save_arrays_to_gcs(tmp_mt_array[:5,:,:], tmp_vllm_array[:5,:,:])
      target_sharding = model_runner_state[key].sharding
      model_runner_state[key] = _experimental_reshard.reshard(
          weight_array, target_sharding, donate=True, may_alias=None,
          cache_resharding_plans=True,
      )
      del weight_array
      if i % 16 == 15:
        jax.effects_barrier()
        gc.collect()
        _log_mem_stats(f"sampler:assign_weights_after_{i+1}")
      i += 1
    jax.effects_barrier()
    gc.collect()
    end_time = time.time()
    logging.info("MaxTextVllmSampler.update_params: %d weights assigned in %.4f seconds", i, end_time - start_time)
    _log_mem_stats("sampler:post_assign_all")



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
            expert_parallel_size=rollout_config.expert_parallel_size,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            engine_kwargs={
                "max_model_len": cache_config_or_size,
                "model": rollout_config.rollout_vllm_model_version,
                "swap_space": rollout_config.rollout_vllm_swap_space_size_gb,
                # Async scheduling causes KeyError in dp_scheduler on slow models
                # (30B+) where inference latency exceeds the scheduler's window.
                "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
                **rollout_config.rollout_vllm_kwargs,
            },
        ),
        converter=converter,
    )

    # Initial weight sync: run the converter so vLLM starts with real weights.
    state = nnx.state(rollout_actor)
    self._sampler.load_checkpoint(state)
