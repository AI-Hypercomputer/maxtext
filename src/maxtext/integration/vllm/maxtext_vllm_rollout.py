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

from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES

def _create_model_converter(model_name: str, config: Any, mesh: jax.sharding.Mesh, use_hf_mapping: bool = False):
  """Instantiate the converter for a MaxText model name."""
  tp = config.rollout_tensor_parallelism
  if model_name in {"qwen3-0.6b"}:
    rules = _MODEL_TO_CONVERSION_RULES.get("qwen3", []) if use_hf_mapping else []
    return WeightConverter(rules=rules, tp=tp)
  if model_name in {"qwen3-30b-a3b", "qwen3-30b-a3b-base", "qwen3-235b-a22b", "qwen3.5-35b-a3b"} or model_name.startswith("qwen3-"):
    # Target state HuggingFace mapping
    rules = _MODEL_TO_CONVERSION_RULES.get("qwen3_moe", []) if use_hf_mapping else []
    return WeightConverter(rules=rules, tp=tp)
  
  # For all other models, return None to fallback to transfer_state_with_mappings()
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

    use_hf = bool(getattr(rollout_config, "rollout_mapping_config", None))
    converter = _create_model_converter(maxtext_config.model_name, config=maxtext_config, mesh=mesh, use_hf_mapping=use_hf)

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=rollout_actor,
        backend="vllm_jax",
    )
    
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
                "max_logprobs": 1,
                "logprobs_mode": getattr(rollout_config, "rollout_vllm_logprobs_mode", "processed_logprobs"),
                **rollout_config.rollout_vllm_kwargs,
            },
        ),
        converter=converter,
    )

    # Initial weight sync: run the converter so vLLM starts with real weights.
    state = nnx.state(rollout_actor)
    self._sampler.load_checkpoint(state)
