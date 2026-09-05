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

These extend Tunix weight synchronization with both model-specific native-vLLM
converters and scanned MaxText-to-MaxText state unrolling. The converters handle:
  - QKV fusion with GQA interleaving (attention)
  - MoE expert gate+up fusion (w13_weight chunk-interleaved for TP)
  - MoE gate / down transpose
  - Layer-norm and LM-head transposes
"""

import ast
import contextlib
import copy
import json
import logging
import re
import time
import traceback
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from tunix.generate import mappings
from tunix.generate import utils as tunix_gen_utils
from tunix.generate.vllm_sampler import VllmConfig, VllmSampler
from tunix.rl import reshard as tunix_reshard
from tunix.rl.rollout import base_rollout, vllm_rollout

from maxtext.integration.vllm.convert_utils import _sharding_summary
from maxtext.integration.vllm.weight_converter import (
    WeightConverter,
    MODEL_TO_CONVERSION_RULES,
)
from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter

# Sentinel distinguishing "this model has no entry" from "this model has an
# entry whose value is None", which means direct-sync-only.
_NO_RULE_TABLE = object()


def _rule_table_for(model_name: str):
  """Maps a MaxText model name to its torchax rule table.

  Returns `None` for models supported only on the direct path, and
  `_NO_RULE_TABLE` for models this converter does not handle at all.
  """
  if model_name == "qwen3-0.6b":
    return MODEL_TO_CONVERSION_RULES["qwen3"]
  return _NO_RULE_TABLE


def _create_model_converter(
    model_name: str,
    config: Any,
    mesh: jax.sharding.Mesh,
    use_hf_mapping: bool = False,
    use_weight_converter: bool = False,
    use_standalone_converter: bool = False,
    sharding_hints: Optional[dict] = None,
    debug: bool = False,
):
  """Instantiate the converter for a MaxText model name."""
  tp = config.rollout_tensor_parallelism
  if use_standalone_converter:
    # Standalone torchax converters emit the tpu-inference runner's internal
    # layout keyed by its state names; MaxTextVllmSampler syncs them with
    # `_sync_standalone_converted`. Requires vLLM to run its *native* model
    # (no MaxTextForCausalLM overrides).
    if model_name.startswith("qwen3.5"):
      return Qwen35MaxTextToVLLMConverter(
          config=config,
          mesh=mesh,
          vllm_attn_dp=sharding_hints.get("attn_dp_size", 1) if sharding_hints else 1,
          vllm_use_ep=sharding_hints.get("enable_expert_parallel", False) if sharding_hints else False,
      )
    if model_name.startswith("gemma4"):
      return Gemma4MaxTextToVLLMConverter(config=config, mesh=mesh)
    raise NotImplementedError(f"use_standalone_converter: no standalone torchax converter for {model_name}")
  if not use_hf_mapping and not use_weight_converter:
    # Default MaxText-to-MaxText sync uses direct transfer_state_directly with unroll
    return None

  rule_table = _rule_table_for(model_name)
  if rule_table is not _NO_RULE_TABLE:
    if not use_hf_mapping:
      # Direct path: vLLM runs the MaxText model itself, so the differences are
      # purely structural (scanned vs unrolled layers, fused vs split MoE).
      return WeightConverter(rules=None, config=config, tp=tp, debug=debug)
    if rule_table is None:
      raise NotImplementedError(
          f"{model_name} has no HuggingFace-target conversion rules. It is only "
          "supported on the direct MaxText-to-MaxText sync path, which requires "
          "'maxtext_config' in vllm_additional_config so vLLM runs "
          "MaxTextForCausalLM."
      )
    return WeightConverter(rules=rule_table, tp=tp)

  if model_name.startswith("gemma4"):
    return Gemma4MaxTextToVLLMConverter(config=config, mesh=mesh)
  if model_name.startswith("qwen3.5"):
    return Qwen35MaxTextToVLLMConverter(config=config, mesh=mesh)
  if model_name.startswith("qwen3-30"):
    return Qwen3MaxTextToVLLMConverter(config=config, mesh=mesh)

  # For all other models, return None to fallback to transfer_state_with_mappings()

  return None


def _as_dict(value: Any) -> dict:
  """Coerces a config blob to a dict; it arrives as JSON, a repr, or a DictConfig."""
  if isinstance(value, dict):
    return value
  if isinstance(value, str):
    for parse in (json.loads, ast.literal_eval):
      try:
        parsed = parse(value)
      except (ValueError, SyntaxError):
        continue
      if isinstance(parsed, dict):
        return parsed
    logging.warning("Could not parse vllm_additional_config: %.120s", value)
    return {}
  if type(value).__name__ == "DictConfig":
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

    return OmegaConf.to_container(value, resolve=True)
  return {}


def uses_maxtext_vllm_adapter(config: Any) -> bool:
  """Returns whether vLLM is configured to instantiate MaxTextForCausalLM."""
  overrides = getattr(config, "vllm_hf_overrides", None)
  if isinstance(overrides, str):
    return "MaxTextForCausalLM" in overrides
  if isinstance(overrides, dict):
    architectures = overrides.get("architectures", ())
    if isinstance(architectures, str):
      architectures = (architectures,)
    return "MaxTextForCausalLM" in architectures
  return False


def requires_maxtext_scanned_weight_unroll(config: Any) -> bool:
  """Returns whether direct MaxText-to-MaxText sync needs a custom unroll."""
  return bool(getattr(config, "scan_layers", False) and uses_maxtext_vllm_adapter(config))


def prepare_direct_sync_additional_config(
    additional_config: Optional[dict[str, Any]],
    *,
    direct_maxtext_sync: bool,
    num_experts: int,
    tensor_parallel_size: int,
) -> Optional[dict[str, Any]]:
  """Makes the direct MaxText MoE target use TPU-safe prefused weights.

  TPU inference shards the fused gate/up dimension across tensor-parallel
  devices. Each shard must therefore contain its local gate chunk followed by
  its local up chunk. The unfused MaxText inference path concatenates the two
  complete tensors globally, which gives incorrect local shards when TP > 1.
  Tunix's direct-sync MoE fusion builds the required per-shard layout once at
  weight-load time when the target exposes a prefused ``wi`` parameter.
  """
  if not direct_maxtext_sync or num_experts <= 1 or tensor_parallel_size <= 1:
    return additional_config

  prepared = copy.deepcopy(additional_config) if additional_config is not None else {}
  maxtext_overrides = prepared.setdefault("maxtext_config", {})
  if not isinstance(maxtext_overrides, dict):
    raise ValueError("vLLM additional_config.maxtext_config must be a dictionary for direct MaxText sync.")

  if not maxtext_overrides.get("prefuse_moe_weights", False):
    logging.info(
        "MaxTextVllmRollout: enabling prefuse_moe_weights for correct MoE gate/up layout with TP=%d.",
        tensor_parallel_size,
    )
  maxtext_overrides["prefuse_moe_weights"] = True
  return prepared


def _find_scanned_layer_idx(key_tuple, container_names=("layers", "scanned_blocks", "layers_remainder")):
  """Returns (container_idx, container_name) if a scanned layer structure is found, else (-1, None)."""
  for name in container_names:
    for i in range(len(key_tuple) - 1):
      if key_tuple[i] == name and isinstance(key_tuple[i + 1], str) and key_tuple[i + 1].startswith("layers_"):
        return i, name
  return -1, None


def _find_qwen_scanned_layer_idx(key_tuple):
  """Finds a Qwen scanned block path like `layers.layer_0` or `layers.moe_block`."""
  for i in range(len(key_tuple) - 1):
    if key_tuple[i] != "layers" or not isinstance(key_tuple[i + 1], str):
      continue
    if key_tuple[i + 1].startswith("layers_"):
      continue
    match = re.fullmatch(r"layer_(\d+)", key_tuple[i + 1])
    if match:
      return i, int(match.group(1)), 2
    return i, 0, 1
  return -1, -1, 0


def unroll_qwen_scanned_weights(weights, scan_axis: int = 1, pattern_length: Optional[int] = None):
  """Unroll Qwen's heterogeneous or homogeneous scanned blocks for an unscanned MaxText target.

  Qwen 3 Next/3.5 training stores a repeating layer cycle as
  `decoder.layers.layer_{slot}`, with repetitions stacked on `scan_axis`.
  Qwen 3 base training stores homogeneous layers as `decoder.layers.*` stacked on `scan_axis`.
  The inference model stores every layer as a direct decoder attribute named
  `layers_{global_index}`. Tunix's generic direct-sync mapper cannot bridge
  these two structures and otherwise silently leaves all destination layers at
  their random initialization. `max_utils.unscan_train_state_params` is not
  applicable here because it expects a Linen params/sharding pair with
  homogeneous layer groups; this path receives an NNX state and must interleave
  heterogeneous layer slots.
  """
  if hasattr(weights, "filter") and hasattr(weights, "to_pure_dict"):
    # NNX stacks non-parameter state (notably RNG state) on axis 0 even when
    # parameters use param_scan_axis=1. Only parameters belong in weight sync.
    pure_dict = weights.filter(nnx.Param).to_pure_dict()
  elif hasattr(weights, "to_pure_dict"):
    pure_dict = weights.to_pure_dict()
  elif hasattr(weights, "to_dict"):
    pure_dict = weights.to_dict()
  elif isinstance(weights, dict):
    pure_dict = weights
  else:
    return weights

  flat_w = flatten_dict(pure_dict)
  scanned_keys = []
  slot_indices = set()
  scan_lengths = set()
  for key, value in flat_w.items():
    container_idx, slot_idx, consumed = _find_qwen_scanned_layer_idx(key)
    if container_idx == -1 or "dropout" in key or "rngs" in key:
      continue
    if not hasattr(value, "shape") or len(value.shape) <= scan_axis:
      raise ValueError(f"Qwen scanned parameter {'.'.join(map(str, key))} has no scan axis {scan_axis}: {value!r}")
    scanned_keys.append((key, value, container_idx, slot_idx, consumed))
    slot_indices.add(slot_idx)
    scan_lengths.add(value.shape[scan_axis])

  if not scanned_keys:
    return weights

  if pattern_length is None:
    expected_slots = set(range(max(slot_indices) + 1))
    if slot_indices != expected_slots:
      raise ValueError(
          f"Qwen scanned layer slots must be contiguous when pattern_length is omitted; found {sorted(slot_indices)}"
      )
    pattern_length = len(slot_indices)
  elif pattern_length <= max(slot_indices):
    raise ValueError(f"Qwen scanned layer slot {max(slot_indices)} is outside configured pattern length {pattern_length}")
  if len(scan_lengths) != 1:
    raise ValueError(f"Qwen scanned parameters disagree on scan length: {sorted(scan_lengths)}")

  scan_length = scan_lengths.pop()
  scanned_key_paths = {key for key, _, _, _, _ in scanned_keys}
  new_flat_w = {key: value for key, value in flat_w.items() if key not in scanned_key_paths}

  for key, value, container_idx, slot_idx, consumed in scanned_keys:
    prefix = key[:container_idx]
    suffix = key[container_idx + consumed :]
    for repetition in range(scan_length):
      global_idx = repetition * pattern_length + slot_idx
      new_key = prefix + (f"layers_{global_idx}",) + suffix
      new_flat_w[new_key] = jnp.take(value, repetition, axis=scan_axis)

  logging.info(
      "MaxTextVllmSampler: unrolled %d Qwen tensor components across %d layers for direct MaxText weight sync.",
      len(scanned_key_paths),
      scan_length * pattern_length,
  )
  return unflatten_dict(new_flat_w)


def validate_direct_sync_layer_coverage(source, target) -> int:
  """Fail if an unrolled source would leave MaxText target layers untouched.

  Tunix intentionally intersects direct-sync trees. For heterogeneous Qwen
  scans, a schema error can therefore skip every transformer layer without an
  exception. This check runs on the initial full-parameter load and requires
  every unscanned target-layer parameter path to exist in the source.
  """

  def to_pure_params(state):
    if hasattr(state, "filter") and hasattr(state, "to_pure_dict"):
      return state.filter(nnx.Param).to_pure_dict()
    if hasattr(state, "to_pure_dict"):
      return state.to_pure_dict()
    if hasattr(state, "to_dict"):
      return state.to_dict()
    return state

  def unwrap(state, wrapper):
    while isinstance(state, dict) and wrapper in state:
      state = state[wrapper]
    return state

  source = unwrap(to_pure_params(source), "base")
  target = unwrap(to_pure_params(target), "model")
  if not isinstance(source, dict) or not isinstance(target, dict):
    return 0

  source_flat = flatten_dict(source)
  target_flat = flatten_dict(target)

  def is_unscanned_layer_path(path):
    return any(isinstance(part, str) and re.fullmatch(r"layers_\d+", part) for part in path)

  source_layer_keys = {key for key in source_flat if is_unscanned_layer_path(key)}
  target_layer_keys = {key for key in target_flat if is_unscanned_layer_path(key)}

  def source_covers(target_key):
    if target_key in source_layer_keys:
      return True
    # Tunix fuses split training weights into the inference-only prefused
    # parameter before transfer. Treat the pair as coverage for target `wi`.
    if target_key and target_key[-1] == "wi":
      prefix = target_key[:-1]
      return prefix + ("wi_0",) in source_layer_keys and prefix + ("wi_1",) in source_layer_keys
    return False

  missing = {key for key in target_layer_keys if not source_covers(key)}
  if not target_layer_keys or missing:
    examples = [".".join(map(str, key)) for key in sorted(missing)[:5]]
    raise ValueError(
        "Direct MaxText weight sync would leave rollout transformer parameters at random initialization: "
        f"matched {len(target_layer_keys) - len(missing)}/{len(target_layer_keys)} target layer parameters; "
        f"missing examples: {examples}"
    )

  logging.info(
      "MaxTextVllmSampler: verified direct-sync coverage for all %d rollout layer parameters.",
      len(target_layer_keys),
  )
  return len(target_layer_keys)


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
  """VllmSampler that hands MaxText weights to a converter before the sync.

  The weight-sync implementation itself lives in `VllmSampler.update_params`
  (Tunix), which owns the KV-cache teardown/rebuild and the `state_leaves`
  refresh that vLLM needs to actually observe new weights. This subclass only
  supplies the converter and applies scanned-weight pre-unrolls for the
  legacy / direct-sync paths.
  """

  def __init__(
      self,
      tokenizer: Any,
      config: VllmConfig,
      direct_maxtext_sync: bool = False,
      scan_axis: int = 1,
      layer_pattern_length: Optional[int] = None,
  ):
    super().__init__(tokenizer=tokenizer, config=config)
    self._direct_maxtext_sync = direct_maxtext_sync
    self._scan_axis = scan_axis
    self._layer_pattern_length = layer_pattern_length
    model_config = getattr(config, "model_config", None)
    model_name = getattr(model_config, "model", "") or ""
    architectures = getattr(model_config, "architectures", []) or []
    hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", "") or ""
    arch_str = " ".join(str(a) for a in architectures)
    self._is_gemma = (
        "gemma" in str(model_name).lower()
        or "gemma" in str(model_type).lower()
        or "gemma" in str(arch_str).lower()
    )

  def update_params(
      self,
      updated_weights,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    """Update the vLLM runner weights from a MaxText state tree."""
    if isinstance(self._converter, BaseMaxTextToVLLMConverter):
      try:
        return self._sync_standalone_converted(updated_weights)
      except BaseException:
        logging.error("MaxTextVllmSampler standalone sync failed:\n%s", traceback.format_exc())
        for handler in logging.getLogger().handlers:
          try:
            handler.flush()
          except Exception:  # pylint: disable=broad-except
            pass
        raise
    if self._converter is None:
      if self._direct_maxtext_sync and self._is_gemma:
        updated_weights = unroll_gemma_scanned_weights(updated_weights)
    try:
      return super().update_params(updated_weights, filter_types)
    except BaseException:
      # A device-addressability failure here takes the whole Pathways JobSet
      # down, and the teardown races the normal exception propagation -- the
      # Python traceback is routinely truncated or lost entirely in the worker
      # logs. Force it out before re-raising.
      logging.error("MaxTextVllmSampler.update_params failed:\n%s", traceback.format_exc())
      for handler in logging.getLogger().handlers:
        try:
          handler.flush()
        except Exception:  # pylint: disable=broad-except
          pass
      raise

  def _sync_standalone_converted(self, updated_weights):
    """Standalone torchax-converter sync path.

    The converter emits tensors in the tpu-inference runner's *internal* layout,
    keyed by its state names, so this bypasses Tunix's mapped/direct transfers:
    tear down the KV cache, convert, reshard each tensor onto its existing
    sharding (chunked, Pathways-aware) and assign into the runner's flat state
    dict in place.
    """
    runner = self._model_runner
    state = runner.state
    if not isinstance(state, dict):
      raise TypeError(
          "Standalone torchax converters target the vLLM (torchax) model "
          "implementation, whose runner state is a flat dict; got "
          f"{type(state).__name__}. Remove MaxTextForCausalLM overrides so "
          "vLLM runs its native model."
      )

    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")
    jax.effects_barrier()

    start = time.time()
    pure = updated_weights.to_pure_dict() if hasattr(updated_weights, "to_pure_dict") else updated_weights
    converted = self._converter.convert(pure)

    src = {k: v for k, v in converted.items() if k in state}
    version_aliases = sorted(set(converted) - set(src))
    if version_aliases:
      logging.info(
          "Standalone sync: %d converted tensors have no runner target (vLLM version aliases), e.g. %s",
          len(version_aliases),
          version_aliases[:3],
      )
    uncovered = [k for k in state if k not in src and not k.rsplit(".", 1)[-1].startswith("_") and "rotary_emb" not in k]
    if uncovered:
      logging.warning(
          "Standalone sync: %d runner tensors NOT covered by the converter (stale weights!), e.g. %s",
          len(uncovered),
          uncovered[:5],
      )

    spec = {k: state[k] for k in src}
    expected = {k: (tuple(v.shape), v.dtype) for k, v in spec.items()}
    chunk = getattr(self.config, "reshard_chunk_size", None)
    delete_dst = getattr(self.config, "delete_dst_buffers", True)
    reshard_in_chunks = getattr(tunix_gen_utils, "_reshard_in_chunks", None)
    if chunk and reshard_in_chunks is None:
      logging.warning("Standalone sync: this Tunix has no _reshard_in_chunks; falling back to one reshard call.")
      chunk = None
    if chunk:
      resharded = reshard_in_chunks(
          src_flat=dict(src),
          spec_flat=spec,
          reshard_fn=tunix_reshard.reshard_pytree,
          chunk_size=chunk,
          delete_spec_buffers=delete_dst,
      )
    else:
      shardings = {k: v.sharding for k, v in spec.items()}
      if delete_dst:
        tunix_gen_utils._delete_target_buffers(spec, src)  # pylint: disable=protected-access
      resharded = tunix_reshard.reshard_pytree(src, shardings)

    for k in src:
      new = resharded[k]
      shape, dtype = expected[k]
      if tuple(new.shape) != shape or new.dtype != dtype:
        raise ValueError(
            f"{k}: converter produced {tuple(new.shape)}/{new.dtype}, the runner expects {shape}/{dtype}; "
            "the converter's layout is out of date with tpu-inference."
        )
      state[k] = new
    runner.state_leaves = state
    logging.info("Standalone sync: updated %d/%d runner tensors in %.1fs", len(src), len(state), time.time() - start)

    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")


class MaxTextVllmRollout(vllm_rollout.VllmRollout):
  """VllmRollout that uses MaxTextVllmSampler for weight synchronization.

  The extra `maxtext_config` selects either a native-vLLM converter or direct
  MaxText adapter synchronization. All other arguments mirror VllmRollout.__init__.

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

    vllm_additional_config = _as_dict(
        getattr(rollout_config, "rollout_vllm_additional_config", None)
        or getattr(maxtext_config, "vllm_additional_config", None)
        or {}
    )

    # The presence of "maxtext_config" is what makes the direct MaxText-to-MaxText
    # sync legal: it is the key that makes vLLM instantiate MaxTextForCausalLM
    # rather than its own HuggingFace-shaped model. Deriving this from
    # rollout_vllm_model_version (always set -- train_rl.py points it at
    # tokenizer_path) or from string-matching hf_overrides guessed at the same
    # fact indirectly and got it wrong when either field was reformatted.
    use_hf = "maxtext_config" not in vllm_additional_config and not uses_maxtext_vllm_adapter(maxtext_config)
    direct_maxtext_sync = not use_hf
    use_weight_converter = bool(
        getattr(maxtext_config, "use_weight_converter", False)
        or vllm_additional_config.get("use_weight_converter", False)
    )
    use_standalone_converter = bool(
        getattr(maxtext_config, "use_standalone_converter", False)
        or vllm_additional_config.get("use_standalone_converter", False)
    )
    # Sampler sharding the standalone converter must mirror: attention DP from
    # the sharding_strategy blob, expert parallelism from the vLLM engine kwargs.
    strategy = {}
    sharding_blob = vllm_additional_config.get("sharding") if isinstance(vllm_additional_config, dict) else None
    if isinstance(sharding_blob, dict):
      strategy = sharding_blob.get("sharding_strategy") or {}
    rollout_vllm_kwargs = getattr(rollout_config, "rollout_vllm_kwargs", None) or {}
    sharding_hints = {
        "attn_dp_size": (int(strategy.get("attn_dp_size") or 1) if strategy.get("enable_dp_attention", False) else 1),
        "enable_expert_parallel": bool(rollout_vllm_kwargs.get("enable_expert_parallel", False)),
    }
    # Accepted from either spelling, matching use_weight_converter above, so a
    # debug run can be triggered by editing the same JSON blob.
    self._weight_sync_debug = bool(
        getattr(maxtext_config, "weight_sync_debug", False) or vllm_additional_config.get("weight_sync_debug", False)
    )
    converter = _create_model_converter(
        maxtext_config.model_name,
        config=maxtext_config,
        mesh=mesh,
        use_hf_mapping=use_hf,
        use_weight_converter=use_weight_converter,
        use_standalone_converter=use_standalone_converter,
        sharding_hints=sharding_hints,
        debug=self._weight_sync_debug,
    )

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=rollout_actor,
        backend="vllm_jax",
    )
    engine_kwargs = {
        "max_model_len": cache_config_or_size,
        "model": rollout_config.rollout_vllm_model_version,
        # Async scheduling causes KeyError in dp_scheduler on slow models
        # (30B+) where inference latency exceeds the scheduler's window.
        "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
        "max_num_batched_tokens": rollout_config.rollout_vllm_max_num_batched_tokens,
        "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
        "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
        "max_logprobs": 1,
        "logprobs_mode": rollout_config.rollout_vllm_logprobs_mode,
    }

    # Merge additional kwargs like dtype and hf_overrides provided by train_rl.py
    if hasattr(rollout_config, "rollout_vllm_kwargs") and rollout_config.rollout_vllm_kwargs:
      engine_kwargs.update(rollout_config.rollout_vllm_kwargs)

    rollout_additional_config = prepare_direct_sync_additional_config(
        vllm_additional_config,
        direct_maxtext_sync=direct_maxtext_sync,
        num_experts=getattr(maxtext_config, "num_experts", 1),
        tensor_parallel_size=rollout_config.tensor_parallel_size,
    )
    self._maxtext_config = maxtext_config

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
            server_mode_submission_threshold=rollout_config.rollout_vllm_server_mode_submission_threshold,
            server_mode_submission_timeout_s=rollout_config.rollout_vllm_server_mode_submission_timeout_s,
            return_logprobs=rollout_config.return_logprobs,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            delete_dst_buffers=rollout_config.rollout_vllm_delete_dst_buffers,
            reshard_chunk_size=rollout_config.rollout_vllm_reshard_chunk_size,
            engine_kwargs=engine_kwargs,
            additional_config=rollout_additional_config,
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
        ),
        direct_maxtext_sync=direct_maxtext_sync,
        scan_axis=getattr(maxtext_config, "param_scan_axis", 1),
        layer_pattern_length=getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", None),
    )

    # Counts every weight sync, including the initial one below. See
    # `_timed_weight_sync` for why sync 0 is the interesting one.
    self._weight_sync_count = 0

    # Initial weight sync: run the converter so vLLM starts with real weights.
    # Filter to Params: NNX stacks non-parameter state (notably RNG state) on
    # axis 0 even when parameters use param_scan_axis=1, so an unfiltered state
    # carries leaves the converter has no target for.
    state = nnx.state(rollout_actor, nnx.Param)
    with self._timed_weight_sync("initial load_checkpoint"):
      self._sampler.load_checkpoint(state)

  def update_params(
      self,
      params: Any,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates rollout parameters, optionally logging an actor-weight checksum.

    The L2 norm is a full reduction over every actor parameter (~35B for
    qwen3.5-35b-a3b) and runs on every sync, so it is opt-in rather than
    always on. Set `log_weight_sync_norm` to re-enable it when debugging
    whether new weights are actually reaching the rollout.
    """
    if getattr(self._maxtext_config, "log_weight_sync_norm", False):
      param_leaves = [
          p.value if hasattr(p, "value") else p for p in jax.tree_util.tree_leaves(params) if hasattr(p, "shape")
      ]
      if param_leaves:
        l2_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(p.astype(jnp.float32))) for p in param_leaves)))
        logging.info("Weight sync: actor parameter L2 norm = %.6f", l2_norm)

    if getattr(self, "_weight_sync_debug", False):
      self._log_sync_boundary(params)

    with self._timed_weight_sync("sync_weights"):
      super().update_params(params, filter_types)

  def _sync_path_name(self) -> str:
    """Which of the three sync implementations this rollout actually uses."""
    converter = getattr(self._sampler, "converter", getattr(self._sampler, "_converter", None))
    if converter is not None:
      return type(converter).__name__
    if getattr(self._sampler, "to_hf_key_mappings", None):
      return "tunix transfer_state_with_mappings"
    return "tunix transfer_state_directly"

  @contextlib.contextmanager
  def _timed_weight_sync(self, label: str):
    """Times one whole weight sync -- convert, reshard and assign.

    Blocks on the rollout state afterwards. JAX dispatch is asynchronous, so
    without the barrier this measures how long it took to *enqueue* the sync,
    and enqueue time is exactly what conversion optimizations shrink -- an
    unblocked number would show a large speedup whether or not one occurred.

    Sync 0 is the initial `load_checkpoint` from `__init__`, which pays XLA
    compilation for every conversion program. Syncs 1+ reuse those executables.
    The gap between them is therefore a direct read of how much of a sync is
    compilation rather than data movement -- a question a single-sync harness
    like `validate_converter.py` cannot answer at all, since
    every one of its numbers includes a cold compile.

    Off by default: the barrier removes any overlap between the sync and the
    training work that follows, so the reported time is slightly pessimistic
    against real throughput. That is the right trade for an A/B, not for a
    production run.
    """
    index = self._weight_sync_count
    self._weight_sync_count += 1
    if not getattr(self._maxtext_config, "log_weight_sync_time", False):
      yield
      return

    start = time.perf_counter()
    try:
      yield
      self._block_on_rollout_state(index)
    except BaseException:
      # Reporting a duration for a sync that raised would print a
      # normal-looking timing line next to the traceback and invite someone to
      # record it as a result.
      logging.info(
          "weight_sync_time: sync %d (%s) FAILED after %.4f s (no measurement)",
          index,
          label,
          time.perf_counter() - start,
      )
      raise
    logging.info(
        "weight_sync_time: sync %d (%s) via %s: %.4f s",
        index,
        label,
        self._sync_path_name(),
        time.perf_counter() - start,
    )

  def _block_on_rollout_state(self, index: int) -> None:
    """Waits for the synced weights to actually land on the rollout devices."""
    try:
      jax.block_until_ready(jax.tree_util.tree_leaves(self._sampler.transformer_state))
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning(
          "weight_sync_time: could not block on rollout state (%s); sync %d " "excludes any work still in flight.",
          exc,
          index,
      )

  def _log_sync_boundary(self, params: Any) -> None:
    """Logs which devices the trainer and rollout trees actually sit on.

    On a split Pathways cluster these must be disjoint halves of the device
    list (`setup_configs_and_devices` gives the trainer the low ids and the
    sampler the high ones). Printing both spans at the boundary turns an
    "ExecuteShard ... device id N is not addressable" panic from a mystery into
    a question of which side N belongs to.
    """

    def _first_leaf(tree):
      for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "shape") or hasattr(leaf, "value"):
          return leaf
      return None

    try:
      src_leaf = _first_leaf(params)
      logging.info(
          "weight_sync_debug: incoming trainer params | %s",
          _sharding_summary(src_leaf) if src_leaf is not None else "<empty tree>",
      )
      tgt_leaf = _first_leaf(self._sampler.transformer_state)
      logging.info(
          "weight_sync_debug: rollout target state | %s",
          _sharding_summary(tgt_leaf) if tgt_leaf is not None else "<empty tree>",
      )
    except Exception:  # pylint: disable=broad-except
      logging.exception("weight_sync_debug: could not summarize sync boundary")
