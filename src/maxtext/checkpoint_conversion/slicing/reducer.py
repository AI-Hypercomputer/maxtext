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

"""Legal reductions for onboarded MaxText models.

MaxText's config is the architecture source of truth. This module only knows how
much of that architecture may be removed without breaking it:

- `get_depth_plan(config)`: legal depths are `prefix + K * cycle + tail`.
- `reduce_depth / reduce_experts / reduce_width`: ordered lists of override dicts,
  largest (highest fidelity) first. Each dict only touches its own axis, so the
  greedy search can merge them on top of each other.

Anything without a known rule raises `UnsupportedSliceError` rather than guessing.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from maxtext.checkpoint_conversion.slicing.slice_utils import align_down
from maxtext.checkpoint_conversion.slicing.slice_utils import align_up
from maxtext.checkpoint_conversion.slicing.slice_utils import decoder_block
from maxtext.checkpoint_conversion.slicing.slice_utils import get_config_value
from maxtext.checkpoint_conversion.slicing.slice_utils import lcm

# ----------------------------------------------------------------------------- constants

AXES: tuple[str, ...] = ("depth", "expert", "width")

# Decoder blocks with a known depth rule.
_DEPTH_BLOCKS = frozenset({
    "default", "llama2", "mistral", "mixtral", "deepseek", "deepseek4", "gemma", "gemma2", "gemma3",
    "gemma4", "qwen2", "qwen3", "qwen3_moe", "qwen3_custom_moe", "qwen3_next", "qwen3_5", "gpt3",
    "gpt_oss", "simple", "simple_mlp", "llama4", "olmo3", "envy",
})  # fmt: skip

# Fixed per-block layer patterns that are not expressed through config fields
# (see *_ATTENTION_PATTERN in maxtext/models and the DeepSeek4 2-layer scan block).
# Gemma2 merges [local, global] into one MaxText layer, so its cycle is 1.
_BLOCK_PATTERN_CYCLE = {"gemma3": 6, "gemma4": 6, "gpt_oss": 2, "olmo3": 4, "deepseek4": 2}

# Blocks whose width can be reduced by scaling emb / mlp / head counts alone
# (standard MHA/GQA attention, no MLA, GatedDeltaNet, or vision coupling).
_WIDTH_BLOCKS = frozenset({
    "default", "llama2", "mistral", "mixtral", "gemma", "gemma2", "gemma3", "qwen2", "qwen3",
    "qwen3_moe", "olmo3",
})  # fmt: skip

DEFAULT_WIDTH_RATIOS: tuple[float, ...] = (0.75, 0.5, 0.25)
_DIM_ALIGNMENT = 128


class UnsupportedSliceError(ValueError):
  """The slicer has no legal reduction rule for this model / axis."""


def _int(config: Any, name: str, default: int = 0) -> int:
  return int(get_config_value(config, name, default))


# ----------------------------------------------------------------------------- types


@dataclasses.dataclass(frozen=True)
class DepthPlan:
  """Legal depths are `prefix + K * cycle + tail` for `min_cycles <= K <= num_cycles`."""

  prefix: int
  cycle: int
  tail: int
  num_cycles: int
  min_cycles: int = 1

  def layers(self, cycles: int) -> int:
    return self.prefix + cycles * self.cycle + self.tail


# ----------------------------------------------------------------------------- validation


def _uses_pipeline_parallelism(config: Any) -> bool:
  return _int(config, "ici_pipeline_parallelism", 1) > 1 or _int(config, "dcn_pipeline_parallelism", 1) > 1


def _has_cross_layer_kv_sharing(config: Any) -> bool:
  return _int(config, "num_kv_shared_layers") > 0


def _has_absolute_layer_dependencies(config: Any) -> bool:
  return bool(list(get_config_value(config, "engram_layers", [])))


def _is_multimodal(config: Any) -> bool:
  return bool(get_config_value(config, "use_multimodal", False))


def _validate_common(config: Any) -> None:
  if _int(config, "global_parameter_scale", 1) != 1:
    raise UnsupportedSliceError("global_parameter_scale != 1 is not supported by the slicer.")


def _validate_depth(config: Any) -> None:
  """Reject configs whose layers are not a plain `prefix + K * cycle + tail` stack."""
  _validate_common(config)
  block = decoder_block(config)
  if block not in _DEPTH_BLOCKS:
    raise UnsupportedSliceError(f"No depth slicing rule for decoder_block={block!r}.")
  if _uses_pipeline_parallelism(config):
    raise UnsupportedSliceError("Pipeline parallelism is not supported by depth slicing.")
  if _has_absolute_layer_dependencies(config):
    raise UnsupportedSliceError("engram_layers pins absolute layer indices; no depth slicing rule.")
  if _has_cross_layer_kv_sharing(config):
    raise UnsupportedSliceError("Cross-layer KV sharing (num_kv_shared_layers > 0) has no depth slicing rule.")
  num_layers = _int(config, "num_decoder_layers")
  compress_ratios = list(get_config_value(config, "compress_ratios", []))
  if compress_ratios and len(compress_ratios) != num_layers:
    raise UnsupportedSliceError(f"len(compress_ratios)={len(compress_ratios)} != num_decoder_layers={num_layers}.")


def _validate_expert(config: Any) -> None:
  _validate_common(config)
  if _int(config, "num_experts", 1) <= 1:
    raise UnsupportedSliceError("model has no routed experts.")
  if decoder_block(config) == "deepseek4" and _int(config, "first_num_hash_layers") > 0:
    raise UnsupportedSliceError("hash-routed layers map tokens to fixed expert ids; no expert slicing rule.")


def _validate_width(config: Any) -> None:
  _validate_common(config)
  block = decoder_block(config)
  if block not in _WIDTH_BLOCKS:
    raise UnsupportedSliceError(f"no width rule for decoder_block={block!r}.")
  if _is_multimodal(config):
    raise UnsupportedSliceError("multimodal models couple decoder width to the vision encoder.")
  q_heads, kv_heads = _int(config, "base_num_query_heads"), _int(config, "base_num_kv_heads")
  if kv_heads < 1 or q_heads % kv_heads != 0:
    raise UnsupportedSliceError(f"num_query_heads={q_heads} is not a multiple of num_kv_heads={kv_heads}.")


# ----------------------------------------------------------------------------- depth


def get_depth_plan(config: Any) -> DepthPlan:
  """Derive the legal depth unit of `config` from MaxText's own config fields."""
  _validate_depth(config)

  block = decoder_block(config)
  num_layers = _int(config, "num_decoder_layers")
  prefix = max(_int(config, "first_num_dense_layers"), _int(config, "first_num_hash_layers"))
  cycle = lcm(
      _int(config, "inhomogeneous_layer_cycle_interval", 1),
      _int(config, "interleave_moe_layer_step", 1),
      _int(config, "nope_layer_interval", -1),
      _BLOCK_PATTERN_CYCLE.get(block, 1),
  )
  body = num_layers - prefix
  if body < cycle:
    raise UnsupportedSliceError(
        f"{num_layers} decoder layers with prefix={prefix} do not contain a full cycle of {cycle} layers."
    )
  num_cycles, tail = divmod(body, cycle)

  # Multimodal DeepStack injects vision features into these decoder layers; keep them all.
  min_layers = 1
  deepstack = list(get_config_value(config, "deepstack_visual_indexes_for_vit", []))
  if _is_multimodal(config) and deepstack:
    min_layers = max(int(i) for i in deepstack) + 1
  min_cycles = next((k for k in range(1, num_cycles + 1) if prefix + k * cycle + tail >= min_layers), num_cycles)
  return DepthPlan(prefix=prefix, cycle=cycle, tail=tail, num_cycles=num_cycles, min_cycles=min_cycles)


def depth_schedule(num_cycles: int, min_cycles: int) -> list[int]:
  """Roughly halving cycle counts below `num_cycles`: ceil(K/2), ceil(K/4), ..., min_cycles."""
  schedule: list[int] = []
  divisor = 2
  while True:
    k = max(min_cycles, -(-num_cycles // divisor))
    if k < num_cycles and k not in schedule:
      schedule.append(k)
    if k <= min_cycles:
      return schedule
    divisor *= 2


def reduce_depth(config: Any, plan: DepthPlan) -> list[dict[str, Any]]:
  """Depth overrides, largest first, excluding the unmodified depth."""
  compress_ratios = list(get_config_value(config, "compress_ratios", []))
  steps: list[dict[str, Any]] = []
  for cycles in depth_schedule(plan.num_cycles, plan.min_cycles):
    layers = plan.layers(cycles)
    step: dict[str, Any] = {"base_num_decoder_layers": layers}
    if compress_ratios:
      step["compress_ratios"] = compress_ratios[:layers]
    steps.append(step)
  return steps


# ----------------------------------------------------------------------------- expert


def reduce_experts(config: Any, mesh_shape: Mapping[str, int], min_experts: int | None = None) -> list[dict[str, Any]]:
  """Halving routed-expert counts that preserve top-k, expert parallelism, and routing groups."""
  _validate_expert(config)

  num_experts = _int(config, "num_experts", 1)
  top_k = _int(config, "num_experts_per_tok", 1)
  groups = max(1, _int(config, "n_routing_groups", -1))
  topk_groups = _int(config, "topk_routing_group", -1)
  quantum = lcm(int(mesh_shape.get("expert", 1)), groups)

  floor = max(2, top_k, int(min_experts or 0))
  if groups > 1 and topk_groups > 0:
    # Grouped routing picks top-k experts from `topk_groups` groups of `num_experts / groups` experts.
    floor = max(floor, groups * -(-top_k // topk_groups))
  floor = align_up(floor, quantum)

  steps: list[dict[str, Any]] = []
  current = num_experts
  while current > floor:
    current = max(floor, align_down(current // 2, quantum))
    steps.append({"num_experts": current})
  return steps


# ----------------------------------------------------------------------------- width


def _width_step(config: Any, ratio: float, tp: int) -> dict[str, Any] | None:
  """Scale emb / mlp / heads by `ratio`, keeping head_dim, GQA ratio, and TP divisibility."""
  q_heads, kv_heads = _int(config, "base_num_query_heads"), _int(config, "base_num_kv_heads")
  group = q_heads // kv_heads
  new_kv = max(1, round(kv_heads * ratio))
  while new_kv >= 1 and (group * new_kv) % tp != 0:
    new_kv -= 1
  if new_kv < 1:
    return None

  mlp_quantum = lcm(_DIM_ALIGNMENT, tp)

  def scaled(name: str, quantum: int) -> int:
    return max(quantum, align_down(round(_int(config, name) * ratio), quantum))

  step = {
      "base_emb_dim": scaled("base_emb_dim", _DIM_ALIGNMENT),
      "base_mlp_dim": scaled("base_mlp_dim", mlp_quantum),
      "base_num_query_heads": group * new_kv,
      "base_num_kv_heads": new_kv,
  }
  if _int(config, "num_experts", 1) > 1 and _int(config, "base_moe_mlp_dim", -1) > 0:
    step["base_moe_mlp_dim"] = scaled("base_moe_mlp_dim", mlp_quantum)
  return step


def reduce_width(
    config: Any,
    mesh_shape: Mapping[str, int],
    min_width_ratio: float = min(DEFAULT_WIDTH_RATIOS),
    ratios: tuple[float, ...] = DEFAULT_WIDTH_RATIOS,
) -> list[dict[str, Any]]:
  """Width overrides for each ratio >= `min_width_ratio`, largest first."""
  _validate_width(config)

  tp = int(mesh_shape.get("tensor", 1))
  original = {k: _int(config, k) for k in ("base_emb_dim", "base_mlp_dim", "base_num_query_heads")}
  steps: list[dict[str, Any]] = []
  for ratio in sorted((r for r in ratios if min_width_ratio <= r < 1.0), reverse=True):
    step = _width_step(config, ratio, tp)
    if step is None or all(step[k] == v for k, v in original.items()) or (steps and step == steps[-1]):
      continue
    steps.append(step)
  return steps


# ----------------------------------------------------------------------------- reporting


def describe(config: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
  """Summarize the model shape that `overrides` produce on top of `config`."""
  emb = _int(config, "base_emb_dim")
  return {
      "layers": int(overrides.get("base_num_decoder_layers", _int(config, "num_decoder_layers"))),
      "experts": int(overrides.get("num_experts", _int(config, "num_experts", 1))),
      "width_ratio": round(int(overrides.get("base_emb_dim", emb)) / emb, 4) if emb else 1.0,
  }
