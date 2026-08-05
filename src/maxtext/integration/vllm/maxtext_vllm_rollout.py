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
import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from tunix.rl import reshard
from tunix.generate import mappings
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl.rollout import base_rollout, vllm_rollout

from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES
from maxtext.integration.vllm.convert_utils import _reshard_in_chunks

from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
def _create_model_converter(model_name: str, config: Any, mesh: jax.sharding.Mesh, use_hf_mapping: bool = False):
  """Instantiate the converter for a MaxText model name."""
  tp = config.rollout_tensor_parallelism
  if model_name in {"qwen3-0.6b"}:
    rules = _MODEL_TO_CONVERSION_RULES.get("qwen3", []) if use_hf_mapping else []
    return WeightConverter(rules=rules, tp=tp)
  if model_name in {"qwen3-30b-a3b", "qwen3-30b-a3b-base", "qwen3-235b-a22b"} or (model_name.startswith("qwen3-") and not "3.5" in model_name):
    # Target state HuggingFace mapping
    rules = _MODEL_TO_CONVERSION_RULES.get("qwen3_moe", []) if use_hf_mapping else []
    return WeightConverter(rules=rules, tp=tp)
  if model_name in {"qwen3.5-35b-a3b"} or model_name.startswith("qwen3.5-"):
    # Target state HuggingFace mapping
    rules = _MODEL_TO_CONVERSION_RULES.get("qwen35_moe", []) if use_hf_mapping else []
    return WeightConverter(rules=rules, tp=tp)
  elif model_name.startswith("gemma4"):
    return Gemma4MaxTextToVLLMConverter(config=config, mesh=mesh)

  # For all other models, return None to fallback to transfer_state_with_mappings()

  return None


def _find_scanned_layer_idx(key_tuple, container_names=("layers", "scanned_blocks", "layers_remainder")):
  """Returns (container_idx, container_name) if a scanned layer structure is found, else (-1, None)."""
  for name in container_names:
    for i in range(len(key_tuple) - 1):
      if key_tuple[i] == name and isinstance(key_tuple[i + 1], str) and key_tuple[i + 1].startswith("layers_"):
        return i, name
  return -1, None


def unroll_gemma_scanned_weights(weights):
  """Workaround for tunix unstacking bug with Gemma 3/4 scanned blocks.

  tunix fails to map nested layers like `layers.layers_0` to `layers_X`
  if the target expects integer keys (as in nnx.List).
  We manually unroll them here, keeping the keys as tuples with integers.
  """
  if hasattr(weights, "to_pure_dict"):
    pure_dict = weights.to_pure_dict()
  elif hasattr(weights, "to_dict"):
    pure_dict = weights.to_dict()
  elif isinstance(weights, dict):
    pure_dict = weights
  else:
    return weights

  flat_w = flatten_dict(pure_dict)
  new_flat_w = {}

  logging.debug("MaxTextVllmSampler: First 5 keys in flat_w: %s", list(flat_w.keys())[:5])

  # Check if this is actually a scanned Gemma 3/4 checkpoint
  is_gemma_scanned = any(_find_scanned_layer_idx(k)[0] != -1 for k in flat_w)

  if not is_gemma_scanned:
    return weights

  logging.info("MaxTextVllmSampler: Detected Gemma scanned weights structure. Unrolling along axis 1...")

  # Determine attention pattern length and scan length
  pattern_keys = set()
  scan_length = 0
  for k, v in flat_w.items():
    container_idx, name = _find_scanned_layer_idx(k)
    if container_idx != -1 and name != "layers_remainder":
      layer_sub_idx = int(k[container_idx + 1].split("layers_")[1])
      pattern_keys.add(layer_sub_idx)
      if hasattr(v, "shape") and len(v.shape) >= 2:
        if "mlp" in k and "wi_0" in k:
          scan_length = max(scan_length, v.shape[1])

  pattern_length = max(pattern_keys) + 1 if pattern_keys else 0
  logging.info("MaxTextVllmSampler: Discovered scan_length=%d, pattern_length=%d", scan_length, pattern_length)

  unrolled_count = 0
  for k, v in flat_w.items():
    if "dropout" in k or "rngs" in k:
      continue

    container_idx, container_name = _find_scanned_layer_idx(k)

    if container_idx != -1 and container_name in ("layers", "scanned_blocks"):
      layer_sub_idx = int(k[container_idx + 1].split("layers_")[1])
      prefix = k[:container_idx]
      suffix = k[container_idx + 2 :]

      if hasattr(v, "shape") and len(v.shape) >= 2 and v.shape[1] == scan_length:
        v_swapped = jnp.swapaxes(v, 1, 0)
        unstacked = [v_swapped[i] for i in range(scan_length)]
      else:
        unstacked = [v] * scan_length

      for i in range(scan_length):
        global_idx = i * pattern_length + layer_sub_idx
        new_k = prefix + (f"layers_{global_idx}",) + suffix
        new_flat_w[new_k] = unstacked[i]
        unrolled_count += 1

    elif container_idx != -1 and container_name == "layers_remainder":
      layer_sub_idx = int(k[container_idx + 1].split("layers_")[1])
      prefix = k[:container_idx]
      suffix = k[container_idx + 2 :]

      global_idx = scan_length * pattern_length + layer_sub_idx
      new_k = prefix + (f"layers_{global_idx}",) + suffix
      new_flat_w[new_k] = v
      unrolled_count += 1
    else:
      new_flat_w[k] = v

  assert unrolled_count > 0, "MaxTextVllmSampler: Detected scanned structure, but failed to unroll any layers!"

  logging.info(
      "MaxTextVllmSampler: Successfully unrolled %d scanned tensor components into vLLM-compatible nnx.List format.",
      unrolled_count,
  )
  return unflatten_dict(new_flat_w)


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
      updated_weights = unroll_gemma_scanned_weights(updated_weights)
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
    start_time = time.time()
    vllm_state = self._converter.convert(
        src_pytree=updated_weights, target_state=self.transformer_state
    )
    if isinstance(self.transformer_state, nnx.State):
      state_dict = (
          self.transformer_state.to_pure_dict()
          if hasattr(self.transformer_state, "to_pure_dict")
          else dict(self.transformer_state)
      )
    else:
      state_dict = self.transformer_state

    if getattr(self.config, "reshard_chunk_size", None) is not None:
      src_flat = flatten_dict(vllm_state)
      spec_flat = flatten_dict(state_dict)

      resharded_flat = _reshard_in_chunks(
          src_flat,
          spec_flat,
          reshard.reshard_pytree,
          getattr(self.config, "reshard_chunk_size", None),
          getattr(self.config, "delete_dst_buffers", False),
      )
      resharded_weights = unflatten_dict(resharded_flat)
    else:
      resharded_weights = reshard.reshard_pytree(
          source=vllm_state,
          target=state_dict,
      )

    if isinstance(self.transformer_state, nnx.State):
      nnx.update(self.transformer_state, resharded_weights)
    elif hasattr(self.transformer_state, "update"):
      self.transformer_state.update(resharded_weights)
    elif isinstance(self.transformer_state, dict):
      self.transformer_state.update(resharded_weights)
    else:
      self._model_runner.state = resharded_weights
    jax.block_until_ready(self.transformer_state)
    end_time = time.time()
    logging.info("MaxTextVllmSampler.update_params: all weights assigned in %.4f seconds", end_time - start_time)

    # reinitialize kv_cache
    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

    return None


class MaxTextVllmRollout(vllm_rollout.VllmRollout):
  """VllmRollout that uses VllmSampler with WeightConverter for weight sync.

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

    model_version = bool(getattr(rollout_config, "rollout_vllm_model_version", ""))
    force_maxtext = "MaxTextForCausalLM" in str(getattr(rollout_config, "rollout_vllm_hf_overrides", ""))
    use_hf = bool(model_version and not force_maxtext) 
    converter = _create_model_converter(maxtext_config.model_name, config=maxtext_config, mesh=mesh, use_hf_mapping=use_hf)

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=rollout_actor,
        backend="vllm_jax",
    )
    engine_kwargs = {
        "max_model_len": cache_config_or_size,
        "model": rollout_config.rollout_vllm_model_version,
        "swap_space": getattr(rollout_config, "rollout_vllm_swap_space_size_gb", maxtext_config.swap_space_vllm_gb),
        # Async scheduling causes KeyError in dp_scheduler on slow models
        # (30B+) where inference latency exceeds the scheduler's window.
        "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
    }

    # Merge additional kwargs like dtype and hf_overrides provided by train_rl.py
    if hasattr(rollout_config, "rollout_vllm_kwargs") and rollout_config.rollout_vllm_kwargs:
      engine_kwargs.update(rollout_config.rollout_vllm_kwargs)

    # Safely extract and parse vllm_additional_config from maxtext_config
    additional_config = rollout_config.rollout_vllm_additional_config
    if not additional_config and hasattr(maxtext_config, 'vllm_additional_config'):
        if type(maxtext_config.vllm_additional_config).__name__ == "DictConfig":
            from omegaconf import OmegaConf
            additional_config = OmegaConf.to_container(maxtext_config.vllm_additional_config, resolve=True)
        elif isinstance(maxtext_config.vllm_additional_config, dict):
            additional_config = maxtext_config.vllm_additional_config

    self._sampler = VllmSampler(
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
            additional_config=additional_config,  # <-- Fix: Pass the safely parsed config!
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
                "max_num_batched_tokens": getattr(rollout_config, "rollout_vllm_max_num_batched_tokens", None),
                "max_num_seqs": getattr(rollout_config, "rollout_vllm_max_num_seqs", None),
                "hf_config_path": getattr(rollout_config, "rollout_vllm_hf_config_path", None),
                "hf_overrides": getattr(rollout_config, "rollout_vllm_hf_overrides", None),
                "max_logprobs": 1,
                "logprobs_mode": getattr(rollout_config, "rollout_vllm_logprobs_mode", "processed_logprobs"),
                **getattr(rollout_config, "rollout_vllm_kwargs", {}),
            },
        ),
        converter=converter,
    )

    # Initial weight sync: run the converter so vLLM starts with real weights.
    state = nnx.state(rollout_actor)
    self._sampler.load_checkpoint(state)

  def update_params(
      self,
      params: Any,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates rollout parameters with logging of L2 norm for weight propagation verification."""
    leaves = jax.tree_util.tree_leaves(params)
    param_leaves = [p.value if hasattr(p, "value") else p for p in leaves if hasattr(p, "shape")]
    if param_leaves:
      import jax.numpy as jnp
      sq_sums = [jnp.sum(jnp.square(p.astype(jnp.float32))) for p in param_leaves]
      l2_norm = float(jnp.sqrt(sum(sq_sums)))
      logging.info("STEP WEIGHT SYNC - MaxText Actor Model Parameters L2 Norm: %.6f", l2_norm)
      print(f"\n[STEP WEIGHT SYNC] MaxText Actor Model Parameters L2 Norm: {l2_norm:.6f}\n", flush=True)
    super().update_params(params, filter_types)

