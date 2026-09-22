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

"""Pure-planner model reduction and structural slicing tool for MaxText configurations.

This module implements a pure, non-mutating planning pipeline (`plan_model_reduction`)
and an atomic applicator (`apply_reduction_plan`) that reduce MaxText model
configurations under explicit preservation contracts:

1. `structural` (or `depth`):
   Preserves per-layer dimensions (`emb_dim`, `mlp_dim`, `moe_mlp_dim`), `num_experts`,
   `num_experts_per_tok`, and the true architectural cycle pattern (`C_arch`). Reduces
   only the number of repeating architectural cycles (`K -> K'`). Supports direct cycle
   selection via `target_num_cycles` (e.g., `target_num_cycles=1` or `2`).
2. `experts`:
   Preserves `emb_dim`, `mlp_dim`, `moe_mlp_dim`, and multi-expert routing (`top-k >= 2`),
   reducing only repeating cycles `K` and `num_experts`.
3. `compact` (or `balanced` / `auto`):
   Preserves all computational layer types and multi-expert combination (`top-k` is never
   collapsed to 1), while allowing bounded discrete reductions in cycles `K`, `num_experts`,
   and hardware-aligned MLP/embedding widths.
4. `width`:
   Preserves `base_num_decoder_layers` (`K` unchanged) and reduces per-layer experts/widths
   within the feasible aligned interval.
5. `budget`:
   Fits a parameter budget (`target_model_size` or `model_reduce_factor`, respecting
   `hard_budget=True` as an upper bound) over the discrete candidate space and records all
   relaxed invariants in `ReductionPlan`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import functools
import math
import os
import re
import sys
from typing import Any
import weakref


import absl
import jax
import omegaconf
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging

absl.logging.set_verbosity(absl.logging.INFO)

_MAX_ALLOWED_RELATIVE_DEVIATION = 0.15

# Sentinel distinguishing "this value cannot be serialized to YAML" from a legitimate `None`.
_UNSET = object()

# Grouped MoE routers (DeepSeek-style) score each routing group by the sum of its top-2 expert
# scores, so a group holding a single expert cannot be scored.
_MIN_EXPERTS_PER_ROUTING_GROUP = 2

# Fields recomputed by `derive_dimensions` from the strategy-mutable `base_*` fields. Changes here
# are consequences of a legal change, not independent modifications.
_DERIVED_FIELDS = frozenset({
    "emb_dim",
    "num_query_heads",
    "num_kv_heads",
    "mlp_dim",
    "moe_mlp_dim",
    "num_decoder_layers",
    # `derive_dimensions` re-establishes the fully-MoE invariant `base_mlp_dim == base_moe_mlp_dim`
    # (types.py cross-field validation), which can touch `base_mlp_dim` as a consequence.
    "base_mlp_dim",
    "mlp_dim",
})

# Request/response plumbing rather than architecture; excluded from contract comparisons.
_REDUCTION_CONTROL_FIELDS = frozenset({
    "model_reduce_factor",
    "target_model_size",
    "target_num_cycles",
    "hard_budget",
    "model_reduce_strategy",
})

# ---------------------------------------------------------------------------
# Exported model-specification schema
# ---------------------------------------------------------------------------
# `save_reduced_model_yaml` must emit a SPECIFICATION of the resolved architecture, not a dump of
# the job configuration: a full dump would carry credentials, dataset paths, output directories and
# run identity into a shared artifact. The schema below is therefore declared explicitly as the set
# of pydantic groups in `types.py` that describe (a) the model architecture and (b) the execution
# layout that architecture is only valid under.
#
# Groups deliberately EXCLUDED (job state, not model spec): RunInfo, Checkpointing, OrbaxStorage,
# EmergencyCheckpointing, Tokenizer, *Dataset, DPO, FineTuning, LoRA, Distillation, TrainingLoop,
# Optimizer/AdamW/Muon, Inference*, Profiling, HloDump, Metrics, Goodput, GcpMonitoring,
# Tensorboard, RL*, DevelopmentAndDebugging, DerivedValues, RematAndOffload.
_EXPORTED_CONFIG_GROUPS: tuple[str, ...] = (
    # --- Architecture ---
    "ModelArchitecture",
    "MTP",
    "LogitsAndLoss",
    "DataTypes",
    "Quantization",
    # --- Attention ---
    "Attention",
    "MoBa",
    "MlaAttention",
    "CompressedAttention",
    "AttentionIndexer",
    "Llama4Attention",
    "SplashAttention",
    # --- Mixture of experts / routing ---
    "MoEGeneral",
    "MoEKernels",
    "DeepSeekMoE",
    "Qwen3Next",
    # --- Positional embeddings ---
    "PositionalEmbedding",
    "Rope",
    "YarnRope",
    # --- Multimodal ---
    "MultimodalGeneral",
    "VisionTower",
    "VisionProjector",
    "AudioEncoder",
    "Multimodal",
    # --- Other architectural blocks ---
    "Engram",
    # --- Execution layout the architecture is validated against ---
    "LayoutAndSharding",
    "DcnParallelism",
    "IciParallelism",
    "PipelineParallelism",
    "HardwareAndMesh",
)

# Identity keys are written explicitly by the exporter and must never be driven by the schema loop.
_EXPORT_IDENTITY_KEYS = ("base_config", "model_name", "override_model_config")

# Marks a YAML as an already-resolved architecture snapshot produced by this tool. `pyconfig` uses
# it to recognize exported slices as user intent regardless of which directory they live in.
SNAPSHOT_MARKER_FIELD = "is_resolved_architecture_snapshot"


@functools.lru_cache(maxsize=1)
def get_exported_model_spec_fields() -> frozenset[str]:
  """Returns the field names belonging to the exported model-specification schema.

  Built by unioning `model_fields` across `_EXPORTED_CONFIG_GROUPS`, then removing reduction control
  flags and identity keys (which the exporter writes itself).
  """
  from maxtext.configs import types as _types  # pylint: disable=import-outside-toplevel

  fields: set[str] = set()
  for group_name in _EXPORTED_CONFIG_GROUPS:
    group = getattr(_types, group_name, None)
    if group is None:  # tolerate group renames across MaxText versions
      max_logging.log(f"[ModelReducer] Export schema: group '{group_name}' not found in types.py; skipping.")
      continue
    fields.update(getattr(group, "model_fields", {}) or {})
  return frozenset(fields - _REDUCTION_CONTROL_FIELDS - set(_EXPORT_IDENTITY_KEYS) - {SNAPSHOT_MARKER_FIELD})


@functools.lru_cache(maxsize=1)
def _exported_field_defaults() -> dict[str, Any]:
  """Returns `{field_name: declared pydantic default}` for the exported schema.

  A key whose resolved value equals its declared default does not need to be written: omitting it
  means reload resolves it back to that same default. Keeping the artifact minimal also preserves
  mesh portability, since inferred (`-1`) parallelism equals the default and so is never pinned.
  Fields whose default cannot be determined are represented by `_UNSET`, which never compares equal
  to a resolved value and therefore forces the key to be exported (fail-safe).
  """
  from maxtext.configs import types as _types  # pylint: disable=import-outside-toplevel

  defaults: dict[str, Any] = {}
  for group_name in _EXPORTED_CONFIG_GROUPS:
    group = getattr(_types, group_name, None)
    if group is None:
      continue
    for name, info in (getattr(group, "model_fields", {}) or {}).items():
      try:
        defaults[name] = info.get_default(call_default_factory=True)
      except Exception:  # pylint: disable=broad-except
        defaults[name] = getattr(info, "default", _UNSET)
  return defaults

# Strategy contract: explicit frozenset of base fields each strategy is allowed to modify.
# This is a whitelist of INTENT. It is enforced against the configuration that is actually applied
# by `_validate_plan_contract`, not merely against the plan's own field list - a plan can omit a
# field it nonetheless changed, and `planned_fields <= allowed_fields` would not detect that.
STRATEGY_MUTABLE_FIELDS: dict[str, frozenset[str]] = {
    "structural": frozenset({"base_num_decoder_layers"}),
    "depth": frozenset({"base_num_decoder_layers"}),
    "experts": frozenset({"base_num_decoder_layers", "num_experts"}),
    "width": frozenset({
        "num_experts",
        "base_moe_mlp_dim",
        "base_mlp_dim",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }),
    "compact": frozenset({
        "base_num_decoder_layers",
        "num_experts",
        "base_moe_mlp_dim",
        "base_mlp_dim",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }),
    "balanced": frozenset({
        "base_num_decoder_layers",
        "num_experts",
        "base_moe_mlp_dim",
        "base_mlp_dim",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }),
    "auto": frozenset({
        "base_num_decoder_layers",
        "num_experts",
        "base_moe_mlp_dim",
        "base_mlp_dim",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }),
    "budget": frozenset({
        "base_num_decoder_layers",
        "num_experts",
        "num_experts_per_tok",
        "base_moe_mlp_dim",
        "base_mlp_dim",
        "base_emb_dim",
        "base_num_query_heads",
        "base_num_kv_heads",
    }),
}


@dataclass(frozen=True)
class ArchitectureSpec:
  """Describes the true structural layout and constraints of a MaxText model.

  All layer counts are RESOLVED (post-`global_parameter_scale`) counts.
  """

  model_name: str
  decoder_block: str
  prefix_layers: int
  arch_cycle_length: int
  full_cycles: int
  tail_layers: int
  pp_divisor: int
  min_cycles: int
  layer_scale_multiplier: int
  # Minimum DECODER depth required so every DeepStack visual feature still has a consumer layer.
  # Zero when the model has no DeepStack injection.
  min_layers_for_modality_coverage: int
  has_vision_encoder: bool
  has_per_layer_input_emb: bool
  has_kv_sharing: bool


@dataclass
class ReductionPlan:
  """Immutable-style manifest produced by `plan_model_reduction` before any config mutation."""

  source_model_name: str
  strategy: str
  requested_factor: float
  actual_factor: float
  target_params: int
  hard_budget: bool
  orig_stats: dict[str, int]
  final_stats: dict[str, int]
  min_total_params: int
  max_reduce_factor: float
  arch_spec: ArchitectureSpec
  retained_cycles: int
  retained_tail_layers: int
  # Whether `target_params` came from the user (`target_model_size` / `model_reduce_factor`) or was
  # defined by the requested slice itself (`target_num_cycles`). A user budget is never rewritten.
  explicit_budget: bool = True
  budget_source: str = "model_reduce_factor"
  retained_resolved_layers: int = 0
  # Provenance of the reported parameter count. "estimated" means the numbers come from the
  # analytical estimator only; "abstract" means they were checked against the real parameter tree
  # via `verify_plan_with_abstract_params`. A budget guarantee is only a guarantee about the
  # CONSTRUCTED model in the "abstract" case.
  verification_mode: str = "estimated"
  verified_params: int | None = None
  estimator_relative_error: float | None = None
  verification_error: str = ""
  # Provenance of the SHARDING feasibility claim, which is separate from the parameter-count claim.
  # Divisibility floors are computed against the parallelism degrees declared in the configuration.
  # An axis left at `-1` is inferred at mesh-construction time from the real topology, so treating
  # it as 1 during planning is a planning ASSUMPTION, not proof of placement validity.
  #   "passed"   - every relevant axis was explicitly declared and the floors were checked against it
  #   "deferred" - at least one relevant axis is unresolved; placement must be re-checked on device
  mesh_validation: str = "passed"
  unresolved_parallelism_axes: list[str] = field(default_factory=list)
  planned_fields: dict[str, Any] = field(default_factory=dict)
  remapped_metadata: dict[str, Any] = field(default_factory=dict)
  preserved_invariants: list[str] = field(default_factory=list)
  changed_fields: list[str] = field(default_factory=list)
  relaxed_invariants: list[str] = field(default_factory=list)
  not_established: list[str] = field(
      default_factory=lambda: [
          "Equivalence of large-model XLA/Pallas kernel selection under reduced tensor shapes",
          "Full-depth numerical error accumulation across omitted cycles",
          "Multi-slice DCN communication overlap at full cluster scale",
      ]
  )

  def format_fidelity_report(self) -> str:
    """Formats a human-readable fidelity and parameter report."""
    lines = [
        f"=== Reduction Manifest: '{self.source_model_name}' (strategy='{self.strategy}') ===",
        f"Parameters: {_format_params(self.orig_stats['total_params'])} -> "
        f"{_format_params(self.final_stats['total_params'])} "
        f"(target: {_format_params(self.target_params)} [from {self.budget_source}], "
        f"factor: {self.actual_factor:.3f}x, "
        f"min under policy: {_format_params(self.min_total_params)} [max {self.max_reduce_factor:.2f}x])",
        f"Depth: {self.retained_cycles} cycle(s) x {self.arch_spec.arch_cycle_length} + "
        f"{self.retained_tail_layers} tail + {self.arch_spec.prefix_layers} prefix "
        f"= {self.retained_resolved_layers} resolved layers",
        f"Active/tok: {_format_params(self.orig_stats['active_params'])} -> "
        f"{_format_params(self.final_stats['active_params'])}",
        self._format_verification_line(),
        "architecture_validation: passed",
        self._format_mesh_validation_line(),
        "Preserved:",
    ]
    for item in self.preserved_invariants:
      lines.append(f"  - {item}")
    lines.append("Changed:")
    for item in self.changed_fields or ["  (none)"]:
      lines.append(f"  - {item}")
    if self.relaxed_invariants:
      lines.append("Relaxed:")
      for item in self.relaxed_invariants:
        lines.append(f"  - {item}")
    lines.append("Not established:")
    for item in self.not_established:
      lines.append(f"  - {item}")
    return "\n".join(lines)

  def _format_verification_line(self) -> str:
    """States plainly whether the reported size is verified against the constructed parameter tree."""
    if self.verification_mode == "abstract" and self.verified_params is not None:
      error = f"{self.estimator_relative_error * 100:+.2f}%" if self.estimator_relative_error is not None else "n/a"
      return (
          f"Verification: ABSTRACT - constructed parameter tree has "
          f"{_format_params(self.verified_params)} (estimator error {error})"
      )
    suffix = f" ({self.verification_error})" if self.verification_error else ""
    return (
        "Verification: ESTIMATED ONLY - sizes are analytical estimates and have NOT been checked "
        f"against a constructed parameter tree; any budget guarantee applies to the estimate{suffix}"
    )

  def _format_mesh_validation_line(self) -> str:
    """States whether sharding feasibility was checked against declared degrees or merely assumed."""
    if self.mesh_validation == "passed":
      return "mesh_validation: passed - divisibility checked against explicitly declared parallelism degrees"
    axes = ", ".join(self.unresolved_parallelism_axes) or "unknown"
    return (
        f"mesh_validation: deferred - parallelism remains unresolved ({axes} = -1, assumed 1 for planning); "
        "sharding placement must be re-validated once the device mesh is built"
    )


def parse_size_string(size_val: str | float | int) -> float:
  """Parses a human-readable parameter count string (e.g., '500B', '0.5T', '125M') into a float."""
  if isinstance(size_val, (int, float)):
    return float(size_val)
  s = str(size_val).strip().upper()
  if not s:
    return 0.0
  match = re.match(r"^([0-9.]+)\s*([TBMK]?)$", s)
  if not match:
    raise ValueError(
        f"Invalid target_model_size format '{size_val}'. Expected format like '500B', '0.5T', '70B', or a number."
    )
  val = float(match.group(1))
  unit = match.group(2)
  multipliers = {
      "T": 1e12,
      "B": 1e9,
      "M": 1e6,
      "K": 1e3,
      "": 1.0,
  }
  return val * multipliers[unit]


def _format_params(num_params: float) -> str:
  """Formats a parameter count into human-readable B or M units."""
  if num_params >= 1e9:
    return f"{num_params / 1e9:.2f}B"
  return f"{num_params / 1e6:.2f}M"


def _round_to_multiple(
    val: float,
    multiple: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
  """Rounds `val` to the nearest multiple of `multiple` strictly within the aligned interval `[min_val, max_val]`.

  Searches inside:
    { m * ceil(x_min / m), ..., m * floor(x_max / m) }
  Raises `ValueError` if the feasible aligned interval is empty.
  """
  m = max(1, int(multiple))
  low_bound = max(m, int(min_val)) if min_val is not None else m
  aligned_low = int(math.ceil(low_bound / m)) * m

  if max_val is not None:
    aligned_high = int(math.floor(int(max_val) / m)) * m
    if aligned_low > aligned_high:
      raise ValueError(
          f"Empty feasible aligned interval for multiple={m} within [min_val={min_val}, max_val={max_val}] "
          f"(aligned_low={aligned_low} > aligned_high={aligned_high})."
      )
  else:
    aligned_high = None

  candidate = int(round(val / m)) * m
  candidate = max(aligned_low, candidate)
  if aligned_high is not None:
    candidate = min(aligned_high, candidate)
  return candidate


def derive_dimensions(config: Any) -> None:
  """Pure, idempotent computation of resolved dimensions (`emb_dim`, `mlp_dim`, etc.) from `base_*` and `global_parameter_scale`.

  Does NOT touch layer metadata (`num_kv_shared_layers`, `compress_ratios`, `engram_layers`, etc.).
  """
  scale = int(getattr(config, "global_parameter_scale", 1))
  log_2_scale = math.floor(math.log2(scale)) if scale > 0 else 0
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale

  config.emb_dim = int((2**emb_scale) * config.base_emb_dim)
  config.num_query_heads = int((2**num_head_scale) * config.base_num_query_heads)
  config.num_kv_heads = int((2**num_head_scale) * config.base_num_kv_heads)
  config.mlp_dim = int((2**mlp_dim_scale) * config.base_mlp_dim)
  config.moe_mlp_dim = int((2**mlp_dim_scale) * getattr(config, "base_moe_mlp_dim", 0))
  config.num_decoder_layers = int((2**layer_scale) * config.base_num_decoder_layers)

  # Keep fully-MoE invariant (types.py:4660): base_mlp_dim == base_moe_mlp_dim
  if int(getattr(config, "num_experts", 1)) > 1:
    is_fully_moe = (
        int(getattr(config, "interleave_moe_layer_step", 1)) == 1
        and int(getattr(config, "first_num_dense_layers", 0)) == 0
        and int(getattr(config, "inhomogeneous_layer_cycle_interval", 1)) == 1
    )
    decoder_block = str(getattr(config, "decoder_block", "")).lower()
    if is_fully_moe and "gemma4" not in decoder_block:
      config.base_mlp_dim = config.base_moe_mlp_dim
      config.mlp_dim = config.moe_mlp_dim


def compute_remapped_metadata(source_config: Any, new_base_layers: int, new_base_emb_dim: int) -> dict[str, Any]:
  """Idempotently computes remapped layer/modality metadata from an immutable `source_config`."""
  src = getattr(source_config, "_pydantic_config", source_config)
  scale = int(getattr(src, "global_parameter_scale", 1))
  log_2_scale = math.floor(math.log2(scale)) if scale > 0 else 0
  base_scale, rem = divmod(log_2_scale, 3)
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale

  orig_layers = int((2**layer_scale) * int(src.base_num_decoder_layers))
  new_layers = int((2**layer_scale) * int(new_base_layers))
  orig_emb_dim = int((2**emb_scale) * int(src.base_emb_dim))
  new_emb_dim = int((2**emb_scale) * int(new_base_emb_dim))

  metadata: dict[str, Any] = {}

  # 1. Multimodal projection output dimension (couples ViT projector output to decoder emb_dim)
  if getattr(src, "out_hidden_size_for_vit", None) == orig_emb_dim:
    metadata["out_hidden_size_for_vit"] = new_emb_dim

  # 2. DeepSeek-V4 per-decoder-layer compression schedule (`compress_ratios` domain: decoder layers)
  compress_ratios = getattr(src, "compress_ratios", None)
  if isinstance(compress_ratios, list) and compress_ratios:
    if new_layers <= len(compress_ratios):
      metadata["compress_ratios"] = list(compress_ratios[:new_layers])

  # 3. Gemma4-Small KV-shared layer count (`num_kv_shared_layers` domain: decoder layers, stepped by cycle_len)
  orig_kv_shared = int(getattr(src, "num_kv_shared_layers", 0))
  if orig_kv_shared > 0 and orig_layers > 0:
    if new_layers == orig_layers:
      metadata["num_kv_shared_layers"] = orig_kv_shared
    else:
      spec = get_architecture_spec(src)
      cycle_len = spec.arch_cycle_length
      max_allowed = max(cycle_len, new_layers - cycle_len)
      metadata["num_kv_shared_layers"] = _round_to_multiple(
          orig_kv_shared * new_layers / orig_layers,
          cycle_len,
          min_val=min(orig_kv_shared, cycle_len),
          max_val=min(orig_kv_shared, max_allowed),
      )

  # 4. Decoder-domain layer index list (`engram_layers` domain: decoder layers [0, new_layers - 1])
  engram_layers = getattr(src, "engram_layers", None)
  if isinstance(engram_layers, list) and engram_layers and orig_layers > 0:
    metadata["engram_layers"] = sorted(
        {
            min(new_layers - 1, max(0, int(round(idx * new_layers / orig_layers))))
            for idx in engram_layers
        }
    )

  # 5. Vision-domain layer index list (`deepstack_visual_indexes_for_vit` domain: `num_hidden_layers_for_vit`)
  # Decoder reduction does NOT alter `num_hidden_layers_for_vit`, so vision extraction indices stay intact.
  deepstack_indices = getattr(src, "deepstack_visual_indexes_for_vit", None)
  if isinstance(deepstack_indices, list) and deepstack_indices:
    vit_layers = int(getattr(src, "num_hidden_layers_for_vit", 0))
    if vit_layers > 0:
      metadata["deepstack_visual_indexes_for_vit"] = [idx for idx in deepstack_indices if 0 <= idx < vit_layers]
    else:
      metadata["deepstack_visual_indexes_for_vit"] = list(deepstack_indices)

  return metadata


def _positive_degree(config: Any, *fields: str) -> int:
  """Returns the product of parallelism degrees, treating inferred/unset values as 1.

  MaxText allows a parallelism axis to be `-1` ("infer from the remaining device count") or `0`.
  Multiplying those raw values produces nonsense - two `-1` axes multiply to `+1`, and a single
  `-1` axis flips the sign - so each degree is clamped to at least 1 BEFORE multiplying.
  """
  product = 1
  for name in fields:
    try:
      value = int(getattr(config, name, 1) or 1)
    except (TypeError, ValueError):
      value = 1
    product *= max(1, value)
  return product


# Axes whose degree participates in the divisibility floors computed by `_get_width_and_expert_floors`.
# Only these matter for sharding feasibility of the sliced architecture.
_SHARDING_RELEVANT_AXES: tuple[str, ...] = (
    "ici_tensor_parallelism",
    "dcn_tensor_parallelism",
    "ici_tensor_transpose_parallelism",
    "dcn_tensor_transpose_parallelism",
    "ici_expert_parallelism",
    "dcn_expert_parallelism",
    "ici_fsdp_parallelism",
    "dcn_fsdp_parallelism",
)


def _unresolved_parallelism_axes(config: Any) -> list[str]:
  """Returns the sharding-relevant axes left at `-1` (inferred at mesh-construction time).

  `_positive_degree` clamps such axes to 1 so planning can proceed. That is a sound planning
  assumption but it is NOT proof of placement validity: the real degree is only known once the
  device mesh exists, and it may exceed the head/expert counts the slice retained. Reporting the
  unresolved axes lets the manifest say `mesh_validation: deferred` instead of overclaiming.
  """
  unresolved: list[str] = []
  for name in _SHARDING_RELEVANT_AXES:
    raw = getattr(config, name, None)
    if raw is None:
      continue
    try:
      if int(raw) < 0:
        unresolved.append(name)
    except (TypeError, ValueError):
      continue
  return unresolved


def get_architecture_spec(config: Any) -> ArchitectureSpec:
  """Extracts the true architectural cycle periodicity, prefix, tail, and placement divisor.

  All layer counts on the returned spec are RESOLVED layer counts (i.e. after
  `global_parameter_scale` has been applied), because the architectural cycle structure - attention
  pattern, MoE interleave, NoPE interval - is defined over actual layers. `layer_scale_multiplier`
  records the factor needed to convert a resolved depth back into the `base_num_decoder_layers`
  unit the configuration stores; see `resolved_depth_to_base`.
  """
  src = getattr(config, "_pydantic_config", config)
  scale = int(getattr(src, "global_parameter_scale", 1))
  log_2_scale = math.floor(math.log2(scale)) if scale > 0 else 0
  layer_scale = log_2_scale // 3
  layer_scale_multiplier = 2**layer_scale
  total_layers = int(layer_scale_multiplier * int(getattr(src, "base_num_decoder_layers", 16)))

  first_dense = int(getattr(src, "first_num_dense_layers", 0))
  first_hash = int(getattr(src, "first_num_hash_layers", 0))
  prefix_layers = max(first_dense, first_hash)
  if prefix_layers >= total_layers:
    prefix_layers = 0

  remaining_layers = total_layers - prefix_layers

  decoder_block = str(getattr(src, "decoder_block", "")).lower()
  if "." in decoder_block:
    decoder_block = decoder_block.split(".")[-1]
  model_name = str(getattr(src, "model_name", "default")).lower()

  # True architectural attention cycle period (NEVER collapsed via gcd with remaining_layers)
  if decoder_block == "gemma4_small":
    attn_cycle = 5 if model_name == "gemma4-e2b" else 6
  elif decoder_block in ("gemma3", "gemma4"):
    attn_cycle = 6
  elif decoder_block in ("gemma2", "gpt_oss", "deepseek4"):
    attn_cycle = 2
  else:
    attn_cycle = 1

  inhomogeneous_cycle = max(1, int(getattr(src, "inhomogeneous_layer_cycle_interval", 1)))
  interleave_step = max(1, int(getattr(src, "interleave_moe_layer_step", 1)))
  nope_interval = max(1, int(getattr(src, "nope_layer_interval", 1)))

  arch_cycle_length = math.lcm(inhomogeneous_cycle, interleave_step, nope_interval, attn_cycle)
  full_cycles = remaining_layers // arch_cycle_length
  tail_layers = remaining_layers % arch_cycle_length

  # If a tiny test model has fewer layers than 1 full architectural cycle, treat remaining_layers as 1 unit
  if full_cycles == 0:
    arch_cycle_length = max(1, remaining_layers)
    full_cycles = 1
    tail_layers = 0

  pp_stages = _positive_degree(src, "ici_pipeline_parallelism", "dcn_pipeline_parallelism")
  layers_per_stage = max(1, int(getattr(src, "num_layers_per_pipeline_stage", 1) or 1))
  pp_divisor = pp_stages * layers_per_stage if pp_stages > 1 else 1

  has_kv_sharing = int(getattr(src, "num_kv_shared_layers", 0)) > 0
  min_cycles = min(full_cycles, 2) if (decoder_block == "gemma4_small" and has_kv_sharing) else 1

  # DeepStack visual injection: the ViT produces one feature per entry of
  # `deepstack_visual_indexes_for_vit`, and the decoder consumes feature `i` at decoder layer `i`
  # (`nnx_decoders.py`: `if deepstack_visual_embeds is not None and lyr < len(deepstack_visual_embeds)`).
  # The extraction indices live in the VISION layer domain and are correctly left untouched by
  # decoder reduction, but the number of them is a DECODER depth requirement: a slice with fewer
  # decoder layers than extracted features silently drops the surplus injection paths. Note this is
  # a consumer count (3 for Qwen3-VL), not the largest extraction index (17).
  deepstack_indices = getattr(src, "deepstack_visual_indexes_for_vit", None)
  min_layers_for_modality_coverage = 0
  if bool(getattr(src, "use_multimodal", False)) and isinstance(deepstack_indices, (list, tuple)) and deepstack_indices:
    min_layers_for_modality_coverage = len(deepstack_indices)
    needed_cycles = int(math.ceil(max(0, min_layers_for_modality_coverage - prefix_layers) / arch_cycle_length))
    min_cycles = max(min_cycles, min(full_cycles, max(1, needed_cycles)))

  return ArchitectureSpec(
      model_name=str(getattr(src, "model_name", "default")),
      decoder_block=decoder_block,
      prefix_layers=prefix_layers,
      arch_cycle_length=arch_cycle_length,
      full_cycles=full_cycles,
      tail_layers=tail_layers,
      pp_divisor=pp_divisor,
      min_cycles=min_cycles,
      layer_scale_multiplier=layer_scale_multiplier,
      min_layers_for_modality_coverage=min_layers_for_modality_coverage,
      has_vision_encoder=bool(getattr(src, "use_multimodal", False) or getattr(src, "vision_encoder_block", "")),
      has_per_layer_input_emb=int(getattr(src, "vocab_size_per_layer_input", 0)) > 0,
      has_kv_sharing=has_kv_sharing,
  )


def get_structural_layer_decomposition(config: Any) -> tuple[int, int, int]:
  """Returns `(prefix_layers, arch_cycle_length, full_cycles)` preserving true architectural periodicity."""
  spec = get_architecture_spec(config)
  return spec.prefix_layers, spec.arch_cycle_length, spec.full_cycles


def _get_dense_moe_layer_counts(config: Any, total_layers: int | None = None) -> tuple[int, int]:
  """Returns `(num_dense_layers, num_moe_layers)` for a given config and total layer count."""
  layers = int(total_layers if total_layers is not None else getattr(config, "num_decoder_layers", getattr(config, "base_num_decoder_layers", 0)))
  decoder_block = str(getattr(config, "decoder_block", "")).lower()
  if "." in decoder_block:
    decoder_block = decoder_block.split(".")[-1]

  num_experts = int(getattr(config, "num_experts", 1))
  first_dense = int(getattr(config, "first_num_dense_layers", 0))
  interleave_step = max(1, int(getattr(config, "interleave_moe_layer_step", 1)))

  if num_experts <= 1:
    return layers, 0
  if decoder_block == "deepseek":
    dense = min(layers, first_dense)
    return dense, max(0, layers - dense)
  if decoder_block == "llama4" or interleave_step > 1:
    moe = layers // interleave_step
    return layers - moe, moe
  return 0, layers


def estimate_model_parameters(config: Any) -> dict[str, int]:
  """Computes analytical total and active parameter counts using resolved dimensions (`global_parameter_scale` aware)."""
  src = getattr(config, "_pydantic_config", config)
  scale = int(getattr(src, "global_parameter_scale", 1))
  log_2_scale = math.floor(math.log2(scale)) if scale > 0 else 0
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale

  emb_dim = int((2**emb_scale) * int(getattr(src, "base_emb_dim", 2048)))
  mlp_dim = int((2**mlp_dim_scale) * int(getattr(src, "base_mlp_dim", 7168)))
  raw_moe_dim = int(getattr(src, "base_moe_mlp_dim", -1))
  moe_mlp_dim = int((2**mlp_dim_scale) * (raw_moe_dim if raw_moe_dim > 0 else int(getattr(src, "base_mlp_dim", 7168))))
  num_query_heads = int((2**num_head_scale) * int(getattr(src, "base_num_query_heads", 16)))
  num_kv_heads = int((2**num_head_scale) * int(getattr(src, "base_num_kv_heads", 16)))
  num_layers = int((2**layer_scale) * int(getattr(src, "base_num_decoder_layers", 16)))
  head_dim = int(getattr(src, "head_dim", 128))
  vocab_size = int(getattr(src, "vocab_size", 32768))

  decoder_block = str(getattr(src, "decoder_block", "")).lower()
  if "." in decoder_block:
    decoder_block = decoder_block.split(".")[-1]

  logits_via_emb = bool(getattr(src, "logits_via_embedding", False))
  emb_params = vocab_size * emb_dim * (1 if logits_via_emb else 2)

  # Gemma4-Small per-layer-input embedding: shape is (vocab_size_per_layer_input, num_layers * hidden_size_per_layer_input)
  # plus per-layer linear projections (2 * hidden_size_per_layer_input * emb_dim per layer).
  vocab_ple = int(getattr(src, "vocab_size_per_layer_input", 0))
  hidden_ple = int(getattr(src, "hidden_size_per_layer_input", 0))
  if vocab_ple > 0 and hidden_ple > 0:
    emb_params += (vocab_ple + 3 * emb_dim) * num_layers * hidden_ple

  # Attention parameters
  attention_type = str(getattr(src, "attention_type", "global")).lower()
  if "." in attention_type:
    attention_type = attention_type.split(".")[-1]

  if attention_type == "mla":
    q_lora_rank = int(getattr(src, "q_lora_rank", 0))
    kv_lora_rank = int(getattr(src, "kv_lora_rank", 512))
    qk_nope_dim = int(getattr(src, "qk_nope_head_dim", 128))
    qk_rope_dim = int(getattr(src, "qk_rope_head_dim", 64))
    v_head_dim = int(getattr(src, "v_head_dim", 128))
    if q_lora_rank > 0:
      q_params = emb_dim * q_lora_rank + q_lora_rank * num_query_heads * (qk_nope_dim + qk_rope_dim)
    else:
      q_params = emb_dim * num_query_heads * (qk_nope_dim + qk_rope_dim)
    kv_params = emb_dim * (kv_lora_rank + qk_rope_dim) + kv_lora_rank * num_query_heads * (qk_nope_dim + v_head_dim)
    out_params = num_query_heads * v_head_dim * emb_dim
    total_attn_params = num_layers * (q_params + kv_params + out_params)
  elif decoder_block in ("qwen3_5", "qwen3_next"):
    # Hybrid GatedDeltaNet (3/4 of layers) + Full Attention (1/4 of layers)
    cycle = max(1, int(getattr(src, "inhomogeneous_layer_cycle_interval", 4)))
    num_full_attn = num_layers // cycle
    num_gdn = num_layers - num_full_attn
    full_attn_per_layer = emb_dim * head_dim * (3 * num_query_heads + 2 * num_kv_heads)
    gdn_k_heads = int(getattr(src, "gdn_num_key_heads", 16))
    gdn_v_heads = int(getattr(src, "gdn_num_value_heads", 64))
    gdn_k_dim = int(getattr(src, "gdn_key_head_dim", 128))
    gdn_v_dim = int(getattr(src, "gdn_value_head_dim", 128))
    gdn_per_layer = emb_dim * (2 * gdn_k_heads * gdn_k_dim + 2 * gdn_v_heads * gdn_v_dim) + (gdn_v_heads * gdn_v_dim * emb_dim)
    total_attn_params = num_full_attn * full_attn_per_layer + num_gdn * gdn_per_layer
  elif decoder_block == "gemma4_small":
    model_name = str(getattr(src, "model_name", "")).lower()
    period = 5 if model_name == "gemma4-e2b" else 6
    g_head_dim = int(getattr(src, "global_head_dim", 0)) or head_dim
    kv_shared = min(num_layers, max(0, int(getattr(src, "num_kv_shared_layers", 0))))
    first_shared = num_layers - kv_shared
    total_attn_params = 0
    for l_idx in range(num_layers):
      d_h = g_head_dim if (l_idx % period == period - 1) else head_dim
      kv_m = 0 if l_idx >= first_shared else 2
      total_attn_params += emb_dim * d_h * (2 * num_query_heads + kv_m * num_kv_heads)
  else:
    share_kv = bool(getattr(src, "share_kv_projections", False))
    kv_mult = 1 if share_kv else 2
    attn_params_per_layer = emb_dim * head_dim * (2 * num_query_heads + kv_mult * num_kv_heads)
    total_attn_params = num_layers * attn_params_per_layer
    # Account for DeepSeek-V4 / Indexer auxiliary projections if enabled
    if bool(getattr(src, "use_indexer", False)):
      idx_heads = int(getattr(src, "indexer_n_heads", 64))
      idx_dim = int(getattr(src, "indexer_head_dim", 128))
      total_attn_params += num_layers * emb_dim * idx_heads * idx_dim

  # FFN parameters per layer
  mlp_activations = getattr(src, "mlp_activations", ["silu", "linear"])
  ffn_matrices = len(mlp_activations) + 1
  dense_ffn_per_layer = ffn_matrices * emb_dim * mlp_dim

  num_experts = int(getattr(src, "num_experts", 1))
  num_experts_per_tok = int(getattr(src, "num_experts_per_tok", 1))
  shared_experts = int(getattr(src, "shared_experts", 0))
  shared_mlp_dim = mlp_dim if decoder_block == "gemma4" else moe_mlp_dim

  if num_experts > 1:
    gate_per_layer = emb_dim * num_experts
    shared_per_layer = shared_experts * ffn_matrices * emb_dim * shared_mlp_dim
    routed_total_per_layer = num_experts * ffn_matrices * emb_dim * moe_mlp_dim
    routed_active_per_layer = num_experts_per_tok * ffn_matrices * emb_dim * moe_mlp_dim
    moe_total_per_layer = gate_per_layer + shared_per_layer + routed_total_per_layer
    moe_active_per_layer = gate_per_layer + shared_per_layer + routed_active_per_layer
  else:
    moe_total_per_layer = 0
    moe_active_per_layer = 0

  num_dense_layers, num_moe_layers = _get_dense_moe_layer_counts(src, num_layers)
  total_dense_ffn_params = num_dense_layers * dense_ffn_per_layer

  # Gemma4-Small widens the MLP by 2x on KV-shared layers (`use_double_wide_mlp=True`)
  if decoder_block == "gemma4_small" and bool(getattr(src, "use_double_wide_mlp", False)):
    kv_shared_layers = min(num_dense_layers, max(0, int(getattr(src, "num_kv_shared_layers", 0))))
    total_dense_ffn_params += kv_shared_layers * dense_ffn_per_layer

  total_moe_ffn_params = num_moe_layers * moe_total_per_layer
  active_moe_ffn_params = num_moe_layers * moe_active_per_layer

  # Vision encoder parameters when multimodal is enabled
  vision_params = 0
  if bool(getattr(src, "use_multimodal", False)):
    vit_hidden = int(getattr(src, "hidden_size_for_vit", 0))
    vit_inter = int(getattr(src, "intermediate_size_for_vit", 0))
    vit_layers = int(getattr(src, "num_hidden_layers_for_vit", 0))
    vit_out = int(getattr(src, "out_hidden_size_for_vit", emb_dim))
    if vit_hidden > 0 and vit_layers > 0:
      vision_params = vit_layers * (4 * vit_hidden * vit_hidden + 2 * vit_hidden * vit_inter) + vit_hidden * vit_out

  # Normalization scales. Previously omitted entirely, which biased every estimate low relative to
  # the constructed model. RMSNorm contributes one `emb_dim` vector per normalization site: pre-
  # attention and pre-FFN on every layer, plus the final decoder norm. Models with QK-norm add two
  # head-sized vectors per layer.
  norm_sites_per_layer = 2
  if bool(getattr(src, "use_post_attn_norm", False)):
    norm_sites_per_layer += 1
  if bool(getattr(src, "use_post_ffw_norm", False)):
    norm_sites_per_layer += 1
  norm_params = num_layers * norm_sites_per_layer * emb_dim + emb_dim
  if bool(getattr(src, "use_qk_norm", False)):
    head_dim_for_norm = int(getattr(src, "head_dim", 0)) or 0
    norm_params += num_layers * 2 * head_dim_for_norm

  total_params = (
      emb_params + total_attn_params + total_dense_ffn_params + total_moe_ffn_params + vision_params + norm_params
  )
  active_params = (
      emb_params + total_attn_params + total_dense_ffn_params + active_moe_ffn_params + vision_params + norm_params
  )

  return {
      "total_params": int(total_params),
      "active_params": int(active_params),
      "embedding_params": int(emb_params),
      "attention_params": int(total_attn_params),
      "dense_ffn_params": int(total_dense_ffn_params),
      "moe_ffn_params": int(total_moe_ffn_params),
      "vision_params": int(vision_params),
      "norm_params": int(norm_params),
      "num_dense_layers": int(num_dense_layers),
      "num_moe_layers": int(num_moe_layers),
  }


def count_abstract_model_parameters(config: Any) -> int:
  """Counts exact unique parameters by constructing the abstract parameter tree via `jax.eval_shape`."""
  from flax import linen as nn  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.layers import quantizations  # pylint: disable=import-outside-toplevel
  from maxtext.models import models  # pylint: disable=import-outside-toplevel
  from maxtext.utils import max_utils, maxtext_utils  # pylint: disable=import-outside-toplevel

  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
  abstract_params = maxtext_utils.get_abstract_param(model, config)
  return int(max_utils.calculate_num_params_from_pytree(abstract_params["params"]))


def _base_domain_quantum(resolved_quantum: int, multiplier: int) -> int:
  """Translates a RESOLVED-domain divisibility requirement into the base-unit lattice.

  A resolved dimension is `multiplier * base`. Requiring `multiplier * base` to be divisible by
  `resolved_quantum` is equivalent to requiring `base` to be divisible by
  `resolved_quantum / gcd(resolved_quantum, multiplier)`.

  Applying a resolved requirement directly to a base value (the previous behavior) over-constrains
  the search: at `global_parameter_scale=2` a model with `base_num_kv_heads=4` resolves to 8 heads,
  which satisfies tensor parallelism of 8, yet demanding 8 in the base domain is infeasible.
  """
  q = max(1, int(resolved_quantum))
  m = max(1, int(multiplier))
  return q // math.gcd(q, m)


def _reduction_floor(orig_base: int, base_quantum: int, desired_min_base: int) -> int:
  """Returns the smallest legal BASE value >= `desired_min_base` on the `base_quantum` lattice.

  Never raises. If no aligned value fits at or below the original, the dimension genuinely cannot
  shrink and the original is returned unchanged - a dimension that cannot shrink must not make an
  otherwise-valid reduction (e.g. a depth-only slice) fail.

  Note that the ORIGINAL value does not itself have to sit on the lattice. A model with
  `base_emb_dim=384` and a 256-unit quantum can still be reduced to 256; requiring the original to
  be aligned would pin the dimension and report a spuriously large minimum model.
  """
  orig = int(orig_base)
  q = max(1, int(base_quantum))
  if orig <= 0:
    return orig
  aligned = int(math.ceil(max(q, int(desired_min_base)) / q)) * q
  return aligned if 0 < aligned <= orig else orig


def _get_width_and_expert_floors(config: Any, strategy: str = "auto") -> dict[str, int]:
  """Computes the minimum hardware-aligned width and expert floors under a given strategy contract.

  Two properties matter here:

  1. **Coordinate system.** Sharding constraints (tensor parallelism, head chunking) apply to
     RESOLVED dimensions, but the planner may only write `base_*` fields. Each constraint is
     therefore translated into the base-unit lattice via `_base_domain_quantum`.
  2. **Strategy contract.** A floor is only computed for a field the strategy is allowed to change.
     For every other field the floor IS the original value, so a depth-only strategy never fails
     because a width floor it would never use cannot be constructed.
  """
  src = getattr(config, "_pydantic_config", config)
  allowed_fields = STRATEGY_MUTABLE_FIELDS.get(strategy, STRATEGY_MUTABLE_FIELDS["auto"])

  orig_emb_dim = int(src.base_emb_dim)
  orig_mlp_dim = int(src.base_mlp_dim)
  raw_moe_dim = int(getattr(src, "base_moe_mlp_dim", -1))
  orig_moe_mlp_dim = raw_moe_dim if raw_moe_dim > 0 else orig_mlp_dim
  orig_q_heads = int(src.base_num_query_heads)
  orig_kv_heads = int(src.base_num_kv_heads)
  orig_experts = int(getattr(src, "num_experts", 1))
  orig_k = max(1, int(getattr(src, "num_experts_per_tok", 1)))

  # `global_parameter_scale` multipliers (see `get_individual_scales` in types.py).
  scale = int(getattr(src, "global_parameter_scale", 1))
  log_2_scale = math.floor(math.log2(scale)) if scale > 0 else 0
  base_scale, rem = divmod(log_2_scale, 3)
  head_multiplier = 2 ** (base_scale + int(rem > 0))
  mlp_multiplier = head_multiplier
  emb_multiplier = 2 ** (base_scale + int(rem > 1))

  tp_degree = _positive_degree(src, "ici_tensor_parallelism", "dcn_tensor_parallelism")
  ep_degree = _positive_degree(src, "ici_expert_parallelism", "dcn_expert_parallelism")
  routing_groups = max(1, int(getattr(src, "n_routing_groups", 1)))
  topk_group = max(1, int(getattr(src, "topk_routing_group", 1)))

  # Preserve multi-expert behavioral coverage: never collapse top-k > 1 down to 1.
  # Only `budget` is allowed to reduce `num_experts_per_tok`, and even then never below 2 when orig_k > 1.
  if strategy == "budget" and orig_k > 1:
    min_k = max(2, topk_group)
  else:
    min_k = orig_k

  # `num_experts` is NOT scaled by `global_parameter_scale`, so it needs no domain translation.
  if orig_experts > 1 and "num_experts" in allowed_fields:
    # Baseline quantum: keep experts divisible by the expert-parallel degree and the routing group
    # count, and keep a hardware-friendly granularity.
    base_expert_quantum = math.lcm(ep_degree, routing_groups, 8 if orig_experts >= 16 else 2)
    expert_quantum = min(orig_experts, base_expert_quantum)

    # Grouped routing (DeepSeek-style `n_routing_groups` / `topk_routing_group`) imposes capacity
    # constraints that `num_experts >= top_k` does NOT capture:
    #   (a) E must be divisible by G, otherwise the group reshape is invalid;
    #   (b) E/G >= 2, because a group's score is the sum of its top-2 expert scores - a group with a
    #       single expert cannot be scored and the router raises;
    #   (c) K <= G_selected * (E/G), because only the selected groups' experts are reachable.
    # For DeepSeek-V3 (E=256, K=8, G=8, G_selected=4) these force a floor of 16 experts, not 8.
    expert_floor = max(expert_quantum, min_k, topk_group)
    if routing_groups > 1:
      expert_quantum = min(orig_experts, math.lcm(expert_quantum, routing_groups))
      expert_floor = max(
          expert_floor,
          _MIN_EXPERTS_PER_ROUTING_GROUP * routing_groups,  # (b)
          int(math.ceil(min_k / topk_group)) * routing_groups,  # (c)
      )

    min_experts = _reduction_floor(orig_experts, expert_quantum, expert_floor)
    min_k = min(min_experts, min_k)
    if routing_groups > 1:
      # Re-clamp top-k to what the selected groups can actually reach at the floor.
      min_k = min(min_k, topk_group * max(1, min_experts // routing_groups))
  elif orig_experts > 1:
    expert_quantum = 1
    min_experts = orig_experts
    min_k = orig_k
  else:
    expert_quantum = 1
    min_experts = 1
    min_k = 1

  # --- Attention heads: tensor parallelism and head chunking constrain RESOLVED head counts. ---
  kv_base_quantum = _base_domain_quantum(tp_degree, head_multiplier)
  min_kv_heads = (
      _reduction_floor(orig_kv_heads, kv_base_quantum, kv_base_quantum)
      if "base_num_kv_heads" in allowed_fields
      else orig_kv_heads
  )

  gqa_ratio = max(1, orig_q_heads // max(1, orig_kv_heads))
  head_chunk = max(
      1,
      int(getattr(src, "mla_qk_head_chunk_size", 0) or 0),
      int(getattr(src, "csa_qk_head_chunk_size", 0) or 0),
  )
  # Query heads must satisfy, in the resolved domain: divisibility by tensor parallelism and by the
  # head chunk size; and, in the base domain, divisibility by the retained KV head count (GQA).
  q_base_quantum = math.lcm(
      _base_domain_quantum(tp_degree, head_multiplier),
      _base_domain_quantum(head_chunk, head_multiplier),
      max(1, min_kv_heads),
  )
  min_q_heads = (
      _reduction_floor(orig_q_heads, q_base_quantum, max(q_base_quantum, min_kv_heads * min(gqa_ratio, 2)))
      if "base_num_query_heads" in allowed_fields
      else orig_q_heads
  )

  # Grouped-query attention is a hard architectural invariant: the query head count must be an exact
  # multiple of the KV head count. The two floors are derived separately, so reconcile them here
  # rather than letting an inconsistent pair reach the candidate search and be rejected as a whole.
  if min_q_heads % max(1, min_kv_heads) != 0:
    repaired = int(math.ceil(min_q_heads / min_kv_heads)) * min_kv_heads
    if repaired <= orig_q_heads:
      min_q_heads = repaired
    else:
      # No query count at or below the original divides this KV floor; neither dimension may move.
      min_kv_heads, min_q_heads = orig_kv_heads, orig_q_heads

  # --- Widths: the `tp_degree * 128` guidance is a RESOLVED-dimension requirement. ---
  emb_style_quantum = 256 if orig_emb_dim >= 512 else max(1, orig_emb_dim // 4)
  mlp_style_quantum = 256 if orig_mlp_dim >= 512 else max(1, orig_mlp_dim // 4)
  moe_style_quantum = 128 if orig_moe_mlp_dim >= 256 else max(1, orig_moe_mlp_dim // 4)

  emb_quantum = math.lcm(emb_style_quantum, _base_domain_quantum(tp_degree, emb_multiplier))
  mlp_quantum = math.lcm(mlp_style_quantum, _base_domain_quantum(tp_degree, mlp_multiplier))
  moe_quantum = math.lcm(moe_style_quantum, _base_domain_quantum(tp_degree, mlp_multiplier))

  # A resolved width of `tp_degree * 128` corresponds to this many base units.
  emb_min_base = int(math.ceil(tp_degree * 128 / max(1, emb_multiplier)))
  mlp_min_base = int(math.ceil(tp_degree * 128 / max(1, mlp_multiplier)))

  min_emb_dim = (
      _reduction_floor(orig_emb_dim, emb_quantum, max(emb_quantum, emb_min_base))
      if "base_emb_dim" in allowed_fields
      else orig_emb_dim
  )
  min_mlp_dim = (
      _reduction_floor(orig_mlp_dim, mlp_quantum, max(mlp_quantum, mlp_min_base))
      if "base_mlp_dim" in allowed_fields
      else orig_mlp_dim
  )
  min_moe_mlp_dim = (
      _reduction_floor(orig_moe_mlp_dim, moe_quantum, max(moe_quantum, mlp_min_base))
      if "base_moe_mlp_dim" in allowed_fields
      else orig_moe_mlp_dim
  )

  return {
      "min_experts": min_experts,
      "min_experts_per_tok": min_k,
      "expert_quantum": expert_quantum,
      "min_kv_heads": min_kv_heads,
      "min_q_heads": min_q_heads,
      "kv_quantum": kv_base_quantum,
      "q_quantum": q_base_quantum,
      "min_emb_dim": min_emb_dim,
      "min_mlp_dim": min_mlp_dim,
      "min_moe_mlp_dim": min_moe_mlp_dim,
      "emb_quantum": emb_quantum,
      "mlp_quantum": mlp_quantum,
      "moe_quantum": moe_quantum,
  }


def _legal_cycle_counts(spec: ArchitectureSpec) -> list[tuple[int, int]]:
  """Enumerates legal `(num_cycles, tail_layers)` pairs for a model.

  A pair is legal only if it satisfies ALL of:
    1. `num_cycles >= spec.min_cycles` (architectural minimum, e.g. KV-sharing needs 2 cycles);
    2. the pipelined layer span is divisible by `spec.pp_divisor` (placement);
    3. the resolved depth is an exact multiple of `spec.layer_scale_multiplier`, so it is
       representable in the `base_num_decoder_layers` unit the configuration actually stores.

  Every depth decision in the planner - discrete search, heuristic selection, direct
  `target_num_cycles` selection, and the hard-budget step-down loop - draws from this one set, so
  no branch can bypass a constraint another branch enforces.
  """
  multiplier = max(1, spec.layer_scale_multiplier)
  candidates: list[tuple[int, int]] = []

  def _consider(num_cycles: int, tail: int) -> None:
    resolved = spec.prefix_layers + num_cycles * spec.arch_cycle_length + tail
    if resolved <= 0:
      return
    if resolved % multiplier != 0:
      return  # Not representable in base units at this `global_parameter_scale`.
    pipe_layers = resolved - spec.prefix_layers if spec.decoder_block == "deepseek" else resolved
    if pipe_layers <= 0 or pipe_layers % spec.pp_divisor != 0:
      return
    candidates.append((num_cycles, tail))

  for k in range(spec.min_cycles, spec.full_cycles + 1):
    _consider(k, 0)
    if spec.tail_layers > 0:
      _consider(k, spec.tail_layers)

  if not candidates:
    # Degenerate case: no reduced depth is legal. The only legal slice is the original depth, which
    # the planner will then reject against any budget that requires an actual reduction.
    candidates.append((spec.full_cycles, spec.tail_layers))
  return sorted(set(candidates), key=lambda item: item[0] * spec.arch_cycle_length + item[1])


def compute_minimal_representative_model(config: Any, strategy: str = "auto") -> dict[str, Any]:
  """Computes the minimum configuration under the exact `STRATEGY_MUTABLE_FIELDS` contract for `strategy`.

  The minimum is built through `materialize_candidate`, the same path used by search probes and by
  the final selected candidate, so the feasibility floor it reports is directly comparable to the
  numbers the search produces.
  """
  strat = strategy.strip().lower()
  if strat not in STRATEGY_MUTABLE_FIELDS:
    raise ValueError(f"Unsupported strategy '{strategy}'. Valid strategies: {sorted(STRATEGY_MUTABLE_FIELDS)}.")

  allowed_fields = STRATEGY_MUTABLE_FIELDS[strat]
  src = getattr(config, "_pydantic_config", config)
  orig_stats = estimate_model_parameters(src)
  orig_total = orig_stats["total_params"]

  spec = get_architecture_spec(src)
  floors = _get_width_and_expert_floors(src, strategy=strat)

  choices: dict[str, Any] = {}
  resolved_layers = None
  if "base_num_decoder_layers" in allowed_fields:
    min_k, min_tail = _legal_cycle_counts(spec)[0]
    resolved_layers = _resolved_layers_for(spec, min_k, min_tail)

  if int(getattr(src, "num_experts", 1)) > 1:
    if "num_experts" in allowed_fields:
      choices["num_experts"] = floors["min_experts"]
    if "num_experts_per_tok" in allowed_fields:
      choices["num_experts_per_tok"] = floors["min_experts_per_tok"]
    if "base_moe_mlp_dim" in allowed_fields:
      choices["base_moe_mlp_dim"] = floors["min_moe_mlp_dim"]
    if "base_mlp_dim" in allowed_fields:
      choices["base_mlp_dim"] = floors["min_mlp_dim"]
  else:
    if "base_num_kv_heads" in allowed_fields:
      choices["base_num_kv_heads"] = floors["min_kv_heads"]
    if "base_num_query_heads" in allowed_fields:
      choices["base_num_query_heads"] = floors["min_q_heads"]
    if "base_emb_dim" in allowed_fields:
      choices["base_emb_dim"] = floors["min_emb_dim"]
    if "base_mlp_dim" in allowed_fields:
      choices["base_mlp_dim"] = floors["min_mlp_dim"]

  min_cfg = materialize_candidate(src, spec=spec, resolved_layers=resolved_layers, **choices)
  min_stats = estimate_model_parameters(min_cfg)
  min_total = max(1, min_stats["total_params"])

  return {
      "strategy": strat,
      "min_total_params": min_total,
      "min_active_params": min_stats["active_params"],
      "max_reduce_factor": orig_total / min_total,
      "min_layers": int(min_cfg.num_decoder_layers),
      "min_base_layers": int(min_cfg.base_num_decoder_layers),
      "min_config": min_cfg,
      "min_stats": min_stats,
  }


def _choose_head_pair(orig_q: int, orig_kv: int, dim_scale: float, floors: dict[str, int]) -> tuple[int, int]:
  """Chooses `(query_heads, kv_heads)` in BASE units, preserving GQA and hardware alignment.

  Grouped-query attention requires `num_query_heads % num_kv_heads == 0` on the RESOLVED values. The
  two counts share a single `global_parameter_scale` head multiplier, so enforcing the invariant in
  the base domain enforces it in the resolved domain too.

  The invariant is treated as HARD. Reducing heads is an optimization, so if no smaller pair
  satisfies every constraint this returns the originals unchanged rather than emitting an invalid
  pair (which would be caught later by `_validate_candidate_constraints` and would fail the entire
  slice, even one that only needed to lose depth).
  """
  if orig_q <= 0 or orig_kv <= 0:
    return orig_q, orig_kv

  kv_quantum = max(1, int(floors.get("kv_quantum", 1)))
  min_kv = max(1, int(floors.get("min_kv_heads", 1)))
  q_hw_quantum = max(1, int(floors.get("q_quantum", 1)))
  scale = max(1e-9, float(dim_scale))
  gqa_ratio = max(1, orig_q // orig_kv)

  # Legal KV values are the multiples of `kv_quantum` inside [min_kv, orig_kv].
  kv_hi = (orig_kv // kv_quantum) * kv_quantum
  kv_lo = int(math.ceil(min_kv / kv_quantum)) * kv_quantum
  if kv_hi <= 0 or kv_hi < kv_lo:
    return orig_q, orig_kv

  preferred_kv = int(round((orig_kv / scale) / kv_quantum)) * kv_quantum
  preferred_kv = max(kv_lo, min(kv_hi, preferred_kv))

  # Walk the KV lattice downward from the preferred value; the original pair is the final fallback
  # because `orig_q % orig_kv == 0` holds for any valid source model.
  kv_candidates = list(range(preferred_kv, kv_lo - 1, -kv_quantum))
  if orig_kv not in kv_candidates:
    kv_candidates.append(orig_kv)

  for kv in kv_candidates:
    if kv <= 0:
      continue
    # Query heads must lie on the hardware lattice AND be a multiple of the retained KV count.
    q_lattice = math.lcm(q_hw_quantum, kv)
    q_hi = (orig_q // q_lattice) * q_lattice
    if q_hi < q_lattice:
      continue  # no GQA-valid query count fits at or below the original
    desired_min = max(q_lattice, kv * min(gqa_ratio, 2))
    q_lo = min(q_hi, int(math.ceil(desired_min / q_lattice)) * q_lattice)
    q = int(round((orig_q / scale) / q_lattice)) * q_lattice
    q = max(q_lo, min(q_hi, q))
    if q > 0 and q <= orig_q and q % kv == 0:
      return q, kv

  return orig_q, orig_kv


def _apply_candidate_width_or_experts(
    candidate: Any,
    layer_factor: float,
    allowed_fields: frozenset[str],
    floors: dict[str, int],
) -> None:
  """Modifies only fields present in `allowed_fields` on `candidate` to reduce per-layer params by `layer_factor`."""
  if layer_factor <= 1.01:
    return

  num_experts = int(getattr(candidate, "num_experts", 1))
  tp_degree = _positive_degree(candidate, "ici_tensor_parallelism", "dcn_tensor_parallelism")

  if num_experts > 1:
    actual_expert_ratio = 1.0
    if "num_experts" in allowed_fields:
      orig_k = max(1, int(getattr(candidate, "num_experts_per_tok", 1)))
      min_exp = floors["min_experts"] if "num_experts_per_tok" in allowed_fields else max(floors["min_experts"], orig_k)
      target_experts = _round_to_multiple(
          num_experts / layer_factor,
          floors["expert_quantum"],
          min_val=min(num_experts, min_exp),
          max_val=num_experts,
      )
      actual_expert_ratio = num_experts / max(1, target_experts)
      candidate.num_experts = target_experts

      if "num_experts_per_tok" in allowed_fields and actual_expert_ratio > 1.0:
        new_k = max(floors["min_experts_per_tok"], int(round(orig_k / actual_expert_ratio)))
        candidate.num_experts_per_tok = min(candidate.num_experts, new_k)

    residual_factor = layer_factor / actual_expert_ratio
    if residual_factor > 1.02 and "base_moe_mlp_dim" in allowed_fields:
      candidate.base_moe_mlp_dim = _round_to_multiple(
          candidate.base_moe_mlp_dim / residual_factor,
          floors["moe_quantum"],
          min_val=floors["min_moe_mlp_dim"],
          max_val=candidate.base_moe_mlp_dim,
      )
      if "base_mlp_dim" in allowed_fields:
        candidate.base_mlp_dim = _round_to_multiple(
            candidate.base_mlp_dim / residual_factor,
            floors["mlp_quantum"],
            min_val=floors["min_mlp_dim"],
            max_val=candidate.base_mlp_dim,
        )
  else:
    dim_scale = math.sqrt(layer_factor)
    if "base_num_kv_heads" in allowed_fields and "base_num_query_heads" in allowed_fields:
      new_q_heads, new_kv_heads = _choose_head_pair(
          int(candidate.base_num_query_heads), int(candidate.base_num_kv_heads), dim_scale, floors
      )
      candidate.base_num_kv_heads = new_kv_heads
      candidate.base_num_query_heads = new_q_heads

    if "base_emb_dim" in allowed_fields:
      candidate.base_emb_dim = _round_to_multiple(
          candidate.base_emb_dim / dim_scale,
          floors["emb_quantum"],
          min_val=floors["min_emb_dim"],
          max_val=candidate.base_emb_dim,
      )
    if "base_mlp_dim" in allowed_fields:
      candidate.base_mlp_dim = _round_to_multiple(
          candidate.base_mlp_dim / dim_scale,
          floors["mlp_quantum"],
          min_val=floors["min_mlp_dim"],
          max_val=candidate.base_mlp_dim,
      )


def _fine_tune_candidate_mlp(
    source_config: Any,
    candidate: Any,
    target_params: float,
    allowed_fields: frozenset[str],
    floors: dict[str, int],
    orig_moe_mlp_dim: int,
    orig_mlp_dim: int,
    hard_budget: bool = False,
) -> None:
  """Fine-tunes MLP dimension within `allowed_fields` to match `target_params` (and obey `hard_budget`).

  Every intermediate estimate goes through `_refresh_candidate`, so the metadata used to score the
  candidate always corresponds to the candidate's own architecture rather than the source model's.
  """
  _refresh_candidate(source_config, candidate)
  stats = estimate_model_parameters(candidate)
  current_total = stats["total_params"]
  if current_total <= 0:
    return
  rel_error = abs(current_total - target_params) / target_params
  if rel_error <= 0.012 and (not hard_budget or current_total <= target_params):
    return

  num_experts = int(getattr(candidate, "num_experts", 1))
  if num_experts > 1 and "base_moe_mlp_dim" in allowed_fields and stats["moe_ffn_params"] > 0:
    fixed_params = current_total - stats["moe_ffn_params"]
    desired_moe_params = target_params - fixed_params
    if desired_moe_params > 0:
      ratio = desired_moe_params / stats["moe_ffn_params"]
      quantum = 64 if orig_moe_mlp_dim >= 256 else max(1, orig_moe_mlp_dim // 8)
      new_moe_dim = _round_to_multiple(
          candidate.base_moe_mlp_dim * ratio,
          quantum,
          min_val=floors["min_moe_mlp_dim"],
          max_val=orig_moe_mlp_dim,
      )
      candidate.base_moe_mlp_dim = new_moe_dim
      _refresh_candidate(source_config, candidate)
      if hard_budget and estimate_model_parameters(candidate)["total_params"] > target_params:
        if new_moe_dim - quantum >= floors["min_moe_mlp_dim"]:
          candidate.base_moe_mlp_dim = new_moe_dim - quantum
          _refresh_candidate(source_config, candidate)
  elif "base_mlp_dim" in allowed_fields and stats["dense_ffn_params"] > 0:
    fixed_params = current_total - stats["dense_ffn_params"]
    desired_dense_params = target_params - fixed_params
    if desired_dense_params > 0:
      ratio = desired_dense_params / stats["dense_ffn_params"]
      quantum = 128 if orig_mlp_dim >= 512 else max(1, orig_mlp_dim // 8)
      new_mlp_dim = _round_to_multiple(
          candidate.base_mlp_dim * ratio,
          quantum,
          min_val=floors["min_mlp_dim"],
          max_val=orig_mlp_dim,
      )
      candidate.base_mlp_dim = new_mlp_dim
      _refresh_candidate(source_config, candidate)
      if hard_budget and estimate_model_parameters(candidate)["total_params"] > target_params:
        if new_mlp_dim - quantum >= floors["min_mlp_dim"]:
          candidate.base_mlp_dim = new_mlp_dim - quantum
          _refresh_candidate(source_config, candidate)


def _resolved_layers_for(spec: ArchitectureSpec, num_cycles: int, tail_layers: int) -> int:
  """Returns the RESOLVED (post-`global_parameter_scale`) depth for a `(num_cycles, tail)` choice."""
  return spec.prefix_layers + num_cycles * spec.arch_cycle_length + tail_layers


def resolved_depth_to_base(spec: ArchitectureSpec, resolved_layers: int) -> int:
  """Converts a RESOLVED layer count back into the `base_num_decoder_layers` unit the config stores.

  `global_parameter_scale` multiplies `base_num_decoder_layers` by `2**layer_scale` to produce the
  actual depth. The architectural cycle structure lives in the resolved domain, so depth choices are
  made there and converted back here. A choice that is not exactly representable in base units is
  rejected rather than silently rounded (which would make the plan's reported depth differ from the
  depth the config actually resolves to).
  """
  multiplier = max(1, spec.layer_scale_multiplier)
  if resolved_layers % multiplier != 0:
    raise ValueError(
        f"Resolved depth {resolved_layers} is not representable for '{spec.model_name}' at "
        f"global_parameter_scale with layer multiplier {multiplier}: `base_num_decoder_layers` would be "
        f"{resolved_layers / multiplier}. Choose a depth that is a multiple of {multiplier}."
    )
  return resolved_layers // multiplier


def _refresh_candidate(source_config: Any, candidate: Any) -> Any:
  """Re-derives resolved dimensions AND layer metadata for `candidate` from the immutable source.

  Every parameter estimate must be taken after this call. Estimating a probe without re-applying
  `compute_remapped_metadata` uses the SOURCE model's metadata (e.g. `num_kv_shared_layers`,
  `compress_ratios`) against a reduced depth, which over-counts attention and double-wide MLP
  parameters and can cause an exactly-feasible slice to be rejected.
  """
  derive_dimensions(candidate)
  for key, val in compute_remapped_metadata(source_config, candidate.base_num_decoder_layers, candidate.base_emb_dim).items():
    setattr(candidate, key, val)
  return candidate


def materialize_candidate(
    source_config: Any,
    spec: ArchitectureSpec | None = None,
    resolved_layers: int | None = None,
    **choices: Any,
) -> Any:
  """Builds a fully-consistent candidate config from `source_config` plus a set of field choices.

  This is the SINGLE construction path used by structural search probes, the minimum-size
  computation, direct cycle selection, and the final selected candidate, so every parameter number
  the planner compares is produced identically:

    apply choices -> `derive_dimensions` -> `compute_remapped_metadata` (from the original source)
    -> `_validate_candidate_constraints`.
  """
  src = getattr(source_config, "_pydantic_config", source_config)
  if spec is None:
    spec = get_architecture_spec(src)
  candidate = copy.deepcopy(src)

  if resolved_layers is not None:
    candidate.base_num_decoder_layers = resolved_depth_to_base(spec, int(resolved_layers))
  for key, val in choices.items():
    if val is not None:
      setattr(candidate, key, val)

  _refresh_candidate(src, candidate)
  _validate_candidate_constraints(candidate, spec)
  return candidate


def _validate_candidate_constraints(candidate: Any, spec: ArchitectureSpec) -> None:
  """Validates hard architectural constraints a candidate must satisfy to be constructible."""
  layers = int(getattr(candidate, "num_decoder_layers", 0))
  if layers <= 0:
    raise ValueError(f"Candidate for '{spec.model_name}' has non-positive depth ({layers}).")

  prefix = max(int(getattr(candidate, "first_num_dense_layers", 0)), int(getattr(candidate, "first_num_hash_layers", 0)))
  if prefix >= layers and prefix > 0:
    raise ValueError(
        f"Candidate for '{spec.model_name}' has {layers} layers but {prefix} reserved prefix layers "
        f"(`first_num_dense_layers`/`first_num_hash_layers`); no periodic layers would remain."
    )

  # Pipeline placement: the pipelined layer span must divide evenly into the stage grid.
  if spec.pp_divisor > 1:
    pipe_layers = layers - prefix if spec.decoder_block == "deepseek" else layers
    if pipe_layers <= 0 or pipe_layers % spec.pp_divisor != 0:
      raise ValueError(
          f"Candidate depth {layers} for '{spec.model_name}' violates pipeline placement: "
          f"{pipe_layers} pipelined layers is not divisible by pp_divisor={spec.pp_divisor}."
      )

  # Grouped MoE routing capacity.
  num_experts = int(getattr(candidate, "num_experts", 1))
  if num_experts > 1:
    top_k = max(1, int(getattr(candidate, "num_experts_per_tok", 1)))
    groups = max(1, int(getattr(candidate, "n_routing_groups", 1)))
    selected_groups = max(1, int(getattr(candidate, "topk_routing_group", 1)))
    if top_k > num_experts:
      raise ValueError(f"Candidate has num_experts_per_tok={top_k} > num_experts={num_experts}.")
    if groups > 1:
      if num_experts % groups != 0:
        raise ValueError(
            f"Candidate num_experts={num_experts} is not divisible by n_routing_groups={groups}."
        )
      per_group = num_experts // groups
      # Grouped routing scores a group by the sum of its top-2 expert scores, so a group with a
      # single expert is not a valid group.
      if per_group < _MIN_EXPERTS_PER_ROUTING_GROUP:
        raise ValueError(
            f"Candidate num_experts={num_experts} over n_routing_groups={groups} leaves {per_group} "
            f"expert(s) per group; grouped routing requires at least {_MIN_EXPERTS_PER_ROUTING_GROUP}."
        )
      if selected_groups > groups:
        raise ValueError(f"topk_routing_group={selected_groups} exceeds n_routing_groups={groups}.")
      if top_k > selected_groups * per_group:
        raise ValueError(
            f"Candidate cannot route top-{top_k} tokens: only {selected_groups} selected group(s) x "
            f"{per_group} expert(s) per group = {selected_groups * per_group} reachable experts."
        )

  # Attention head chunking.
  q_heads = int(getattr(candidate, "num_query_heads", 0))
  for chunk_field in ("mla_qk_head_chunk_size", "csa_qk_head_chunk_size"):
    chunk = int(getattr(candidate, chunk_field, 0) or 0)
    if chunk > 0 and q_heads % chunk != 0:
      raise ValueError(f"num_query_heads={q_heads} is not divisible by {chunk_field}={chunk}.")

  kv_heads = int(getattr(candidate, "num_kv_heads", 0))
  if kv_heads > 0 and q_heads % kv_heads != 0:
    raise ValueError(f"num_query_heads={q_heads} is not divisible by num_kv_heads={kv_heads}.")


def _config_field_names(cfg: Any) -> list[str]:
  """Returns the set of declared configuration field names for a pydantic config object."""
  model_fields = getattr(type(cfg), "model_fields", None)
  if isinstance(model_fields, dict):
    return sorted(model_fields.keys())
  return sorted(k for k in vars(cfg) if not k.startswith("_"))


def _validate_plan_contract(
    source_config: Any,
    spec: ArchitectureSpec,
    plan: ReductionPlan,
    candidate: Any,
    explicit_budget: bool,
) -> None:
  """Proves the plan produces exactly the model it reports, for EVERY planning branch.

  Asserting `planned_fields <= allowed_fields` is not sufficient: a branch can mutate a candidate,
  score it, and then drop that mutation from the plan, so the applied model silently differs from
  the evaluated one. This validator instead dry-applies the plan through the real
  `apply_reduction_plan` and compares the result against the candidate that was actually scored.
  """
  src = getattr(source_config, "_pydantic_config", source_config)
  applied = copy.deepcopy(src)
  apply_reduction_plan(applied, plan, log=False)

  # 1. Applied architecture == evaluated candidate architecture.
  mismatches = [
      f"{fname}: applied={getattr(applied, fname, None)!r} != evaluated={getattr(candidate, fname, None)!r}"
      for fname in _config_field_names(src)
      if fname not in _REDUCTION_CONTROL_FIELDS and getattr(applied, fname, None) != getattr(candidate, fname, None)
  ]
  if mismatches:
    raise ValueError(
        f"Internal contract violation for '{plan.source_model_name}' (strategy='{plan.strategy}'): the applied "
        f"configuration differs from the candidate whose parameters were reported. Mismatches: {mismatches}"
    )

  # 2. Applied statistics == reported statistics.
  applied_stats = estimate_model_parameters(applied)
  if applied_stats["total_params"] != plan.final_stats["total_params"]:
    raise ValueError(
        f"Internal contract violation: applied model has {_format_params(applied_stats['total_params'])} "
        f"but the plan reports {_format_params(plan.final_stats['total_params'])}."
    )

  # 3. Applied depth is a legal placement, and agrees with the manifest's cycle accounting.
  _validate_candidate_constraints(applied, spec)
  manifest_layers = _resolved_layers_for(spec, plan.retained_cycles, plan.retained_tail_layers)
  if int(applied.num_decoder_layers) != manifest_layers:
    raise ValueError(
        f"Internal contract violation: manifest reports {plan.retained_cycles} cycle(s) + "
        f"{plan.retained_tail_layers} tail layer(s) = {manifest_layers} resolved layers, but the applied "
        f"configuration resolves to {applied.num_decoder_layers} layers."
    )

  # 4. An explicitly supplied budget is never rewritten by depth selection.
  if explicit_budget and plan.hard_budget and plan.final_stats["total_params"] > plan.target_params:
    raise ValueError(
        f"Internal contract violation: hard budget {_format_params(plan.target_params)} exceeded by "
        f"{_format_params(plan.final_stats['total_params'])}."
    )

  # 5. Every field that actually changed is either strategy-mutable, remapped metadata, or derived.
  allowed = set(STRATEGY_MUTABLE_FIELDS[plan.strategy]) | set(plan.remapped_metadata) | _DERIVED_FIELDS
  illegal = [
      fname
      for fname in _config_field_names(src)
      if fname not in _REDUCTION_CONTROL_FIELDS
      and fname not in allowed
      and getattr(applied, fname, None) != getattr(src, fname, None)
  ]
  if illegal:
    raise ValueError(
        f"Internal contract violation: strategy='{plan.strategy}' modified fields outside its contract: "
        f"{sorted(illegal)} (mutable: {sorted(STRATEGY_MUTABLE_FIELDS[plan.strategy])})."
    )


def verify_plan_with_abstract_params(config: Any, plan: ReductionPlan) -> ReductionPlan:
  """Upgrades a plan from `estimated` to `abstract` verification by building the parameter tree.

  `estimate_model_parameters` is an analytical approximation; it can differ from the constructed
  model by a small margin (normalization terms, tied weights, model-specific heads). A budget
  guarantee derived from it is therefore a guarantee about the ESTIMATE, not the constructed model.
  This routine builds the abstract parameter tree via `jax.eval_shape` for the already-applied
  `config` and records the exact count on the plan.

  It is intentionally NOT called from inside `plan_model_reduction`: planning runs during
  `MaxTextConfig.model_post_init`, before the configuration is complete enough to construct a
  model. Callers (the CLI, and tests) invoke it once initialization has finished. When it cannot
  run, the plan honestly remains `verification_mode == "estimated"`.
  """
  try:
    exact = count_abstract_model_parameters(config)
  except Exception as exc:  # pylint: disable=broad-except
    plan.verification_mode = "estimated"
    plan.verification_error = f"{type(exc).__name__}: {exc}"
    max_logging.log(
        f"[ModelReducer] Abstract parameter verification unavailable for '{plan.source_model_name}' "
        f"({plan.verification_error}); reported sizes remain analytical estimates."
    )
    return plan

  plan.verified_params = int(exact)
  plan.verification_mode = "abstract"
  estimated = max(1, plan.final_stats["total_params"])
  plan.estimator_relative_error = (exact - estimated) / estimated

  if plan.hard_budget and plan.target_params > 0 and exact > plan.target_params:
    raise ValueError(
        f"Constructed model for '{plan.source_model_name}' has {_format_params(exact)} parameters, which exceeds "
        f"hard_budget={_format_params(plan.target_params)} even though the analytical estimate "
        f"({_format_params(estimated)}) fit. The budget guarantee is not satisfied by the real model."
    )
  return plan


def plan_model_reduction(config: Any) -> ReductionPlan | None:
  """Pure planner: computes a `ReductionPlan` without mutating `config`.

  Raises `ValueError` if the request is self-inconsistent, or if the requested reduction or budget
  is infeasible under the strategy's preservation contract, leaving `config` completely untouched.

  Budget semantics:
    * `hard_budget=False` (default): `target_params` is a TARGET. The result must land within
      +/-`_MAX_ALLOWED_RELATIVE_DEVIATION` of it, otherwise the request is rejected.
    * `hard_budget=True`: `target_params` is a strict UPPER BOUND. Undershooting is permitted (and
      recorded in `relaxed_invariants`); only overshoot is rejected. `hard_budget` therefore
      requires an explicit budget and never silently redefines one.
  """
  src = getattr(config, "_pydantic_config", config)
  target_size_raw = str(getattr(src, "target_model_size", "") or "").strip()
  reduce_factor = float(getattr(src, "model_reduce_factor", 1.0))
  target_num_cycles = int(getattr(src, "target_num_cycles", -1))
  hard_budget = bool(getattr(src, "hard_budget", False))
  strategy = str(getattr(src, "model_reduce_strategy", "auto")).strip().lower()

  if not target_size_raw and reduce_factor <= 1.0 and target_num_cycles <= 0:
    if reduce_factor < 1.0:
      raise ValueError(f"`model_reduce_factor` must be >= 1.0, got {reduce_factor}.")
    return None

  if strategy not in STRATEGY_MUTABLE_FIELDS:
    raise ValueError(
        f"`model_reduce_strategy` must be one of {sorted(STRATEGY_MUTABLE_FIELDS)}, got '{strategy}'."
    )

  allowed_fields = STRATEGY_MUTABLE_FIELDS[strategy]
  spec = get_architecture_spec(src)
  orig_stats = estimate_model_parameters(src)
  orig_total = orig_stats["total_params"]
  orig_resolved_layers = max(1, orig_stats["num_dense_layers"] + orig_stats["num_moe_layers"])
  model_label = spec.model_name

  # --------------------------------------------------------------------------------------------
  # Phase 0: validate the REQUEST COMBINATION before building or scoring any candidate.
  # --------------------------------------------------------------------------------------------
  if target_size_raw:
    explicit_budget_source = "target_model_size"
  elif reduce_factor > 1.0:
    explicit_budget_source = "model_reduce_factor"
  else:
    explicit_budget_source = ""

  if target_num_cycles > 0 and "base_num_decoder_layers" not in allowed_fields:
    raise ValueError(
        f"`target_num_cycles={target_num_cycles}` requests a depth change, but "
        f"model_reduce_strategy='{strategy}' is contractually forbidden from changing depth "
        f"(mutable fields: {sorted(allowed_fields)}). Either drop `target_num_cycles` or choose a "
        f"depth-capable strategy such as 'structural', 'balanced', 'compact', or 'auto'."
    )

  if hard_budget and not explicit_budget_source:
    raise ValueError(
        "`hard_budget=True` is an upper bound on an explicit budget, but no budget was supplied. "
        "Set `target_model_size` (e.g. '500B') or `model_reduce_factor` (> 1.0)."
    )

  legal_pairs = _legal_cycle_counts(spec)

  # --------------------------------------------------------------------------------------------
  # Phase 1: resolve the BUDGET. Depth selection must never overwrite an explicit budget.
  # --------------------------------------------------------------------------------------------
  if explicit_budget_source == "target_model_size":
    target_params = parse_size_string(target_size_raw)
    if target_params <= 0 or target_params >= orig_total:
      raise ValueError(
          f"`target_model_size` ({target_size_raw} -> {_format_params(target_params)}) must be positive and smaller "
          f"than the original model size ({_format_params(orig_total)})."
      )
    reduce_factor = orig_total / target_params
  elif explicit_budget_source == "model_reduce_factor":
    target_params = orig_total / reduce_factor
  else:
    # Cycles-only request: the budget is DEFINED by the requested slice, computed after selection.
    target_params = 0.0

  min_info = compute_minimal_representative_model(src, strategy)
  min_total = min_info["min_total_params"]
  max_factor = min_info["max_reduce_factor"]
  min_cfg = min_info["min_config"]

  if explicit_budget_source:
    tolerance = 0.0 if hard_budget else _MAX_ALLOWED_RELATIVE_DEVIATION
    if target_params < min_total * (1.0 - tolerance):
      auto_info = (
          compute_minimal_representative_model(src, "compact")
          if strategy not in ("compact", "auto", "budget")
          else min_info
      )
      hint = ""
      if auto_info["min_total_params"] < min_total:
        hint = (
            f" Tip: Using model_reduce_strategy='compact' allows reducing '{model_label}' down to "
            f"{_format_params(auto_info['min_total_params'])} (max model_reduce_factor={auto_info['max_reduce_factor']:.2f}x)."
        )
      raise ValueError(
          f"Cannot reduce model '{model_label}' ({_format_params(orig_total)}) to {_format_params(target_params)} "
          f"(requested model_reduce_factor={reduce_factor:.2f}x) under strategy='{strategy}' preservation contract. "
          f"The minimum supported model size under strategy='{strategy}' has {_format_params(min_total)} "
          f"(max model_reduce_factor={max_factor:.2f}x: layers={min_cfg.num_decoder_layers}, "
          f"emb_dim={min_cfg.base_emb_dim}, mlp_dim={min_cfg.base_mlp_dim}, "
          f"moe_mlp_dim={getattr(min_cfg, 'base_moe_mlp_dim', 0)}, "
          f"experts={getattr(min_cfg, 'num_experts', 1)} [top-{getattr(min_cfg, 'num_experts_per_tok', 1)}]).{hint}"
      )

  floors = _get_width_and_expert_floors(src, strategy=strategy)
  orig_mlp_dim = int(src.base_mlp_dim)
  raw_moe_dim = int(getattr(src, "base_moe_mlp_dim", -1))
  orig_moe_mlp_dim = raw_moe_dim if raw_moe_dim > 0 else orig_mlp_dim

  # --------------------------------------------------------------------------------------------
  # Phase 2: choose DEPTH. Every branch selects from `legal_pairs`, so direct selection is subject
  # to exactly the same placement and minimum-cycle constraints as the search.
  # --------------------------------------------------------------------------------------------
  depth_is_pinned = target_num_cycles > 0
  retained_cycles, retained_tail = spec.full_cycles, spec.tail_layers

  if depth_is_pinned:
    pinned = [pair for pair in legal_pairs if pair[0] == target_num_cycles]
    if not pinned:
      legal_desc = ", ".join(
          f"{k} cycle(s){'' if t == 0 else f' + {t} tail'} = {_resolved_layers_for(spec, k, t)} layers"
          for k, t in legal_pairs
      )
      raise ValueError(
          f"`target_num_cycles={target_num_cycles}` is not a legal slice of '{model_label}'. "
          f"Legal choices (cycle_length={spec.arch_cycle_length}, prefix_layers={spec.prefix_layers}, "
          f"min_cycles={spec.min_cycles}, pp_divisor={spec.pp_divisor}, "
          f"layer_multiplier={spec.layer_scale_multiplier}): {legal_desc}."
      )
    # Prefer the pure-cycle form; fall back to the tail-preserving form when only it is legal.
    retained_cycles, retained_tail = min(pinned, key=lambda kt: kt[1])

  elif "base_num_decoder_layers" in allowed_fields:
    if strategy in ("structural", "depth"):
      # Depth is the only lever, so the discrete search over legal depths IS the budget fit.
      best_pair, best_diff = None, float("inf")
      for pair in legal_pairs:
        probe = materialize_candidate(src, spec=spec, resolved_layers=_resolved_layers_for(spec, *pair))
        probe_total = estimate_model_parameters(probe)["total_params"]
        if hard_budget and probe_total > target_params:
          continue
        diff = abs(probe_total - target_params)
        if diff < best_diff:
          best_diff, best_pair = diff, pair
      if best_pair is None:
        raise ValueError(
            f"No structural slice satisfies hard_budget={_format_params(target_params)} for '{model_label}'. "
            f"The smallest legal structural slice has {_format_params(min_total)}."
        )
      retained_cycles, retained_tail = best_pair
    else:
      # Mixed strategies split the reduction between depth and width.
      fixed_params = orig_stats["embedding_params"]
      scannable_orig = max(1.0, orig_total - fixed_params)
      desired_scannable = max(1.0, target_params - fixed_params)
      eff_factor = scannable_orig / desired_scannable
      depth_goal = eff_factor if strategy == "auto" else math.sqrt(eff_factor)
      ideal_layers = spec.prefix_layers + spec.full_cycles * spec.arch_cycle_length / max(1e-9, depth_goal)
      pure_cycle_pairs = [pair for pair in legal_pairs if pair[1] == 0 or pair[0] == spec.full_cycles] or legal_pairs
      retained_cycles, retained_tail = min(
          pure_cycle_pairs, key=lambda kt: abs(_resolved_layers_for(spec, *kt) - ideal_layers)
      )

  candidate = materialize_candidate(
      src, spec=spec, resolved_layers=_resolved_layers_for(spec, retained_cycles, retained_tail)
  )

  if not explicit_budget_source:
    # Cycles-only: the selected slice defines the target it is measured against.
    target_params = float(estimate_model_parameters(candidate)["total_params"])
    reduce_factor = orig_total / max(1.0, target_params)

  # --------------------------------------------------------------------------------------------
  # Phase 3: fit the residual with the strategy's WIDTH / EXPERT levers.
  # --------------------------------------------------------------------------------------------
  width_fields = allowed_fields - {"base_num_decoder_layers"}
  if explicit_budget_source and width_fields:
    fixed_params = float(orig_stats["embedding_params"])
    if spec.prefix_layers > 0:
      attn_per_layer = orig_stats["attention_params"] / orig_resolved_layers
      dense_per_layer = (
          orig_stats["dense_ffn_params"] / orig_stats["num_dense_layers"] if orig_stats["num_dense_layers"] > 0 else 0.0
      )
      fixed_params += spec.prefix_layers * (attn_per_layer + dense_per_layer)

    post_depth_total = estimate_model_parameters(candidate)["total_params"]
    residual_factor = max(1.0, (post_depth_total - fixed_params) / max(1.0, target_params - fixed_params))
    if residual_factor > 1.02:
      _apply_candidate_width_or_experts(candidate, residual_factor, allowed_fields, floors)
      _refresh_candidate(src, candidate)
    _fine_tune_candidate_mlp(
        src,
        candidate,
        target_params,
        allowed_fields,
        floors,
        orig_moe_mlp_dim,
        orig_mlp_dim,
        hard_budget=hard_budget,
    )
    _refresh_candidate(src, candidate)

  # --------------------------------------------------------------------------------------------
  # Phase 4: if a hard budget is still exceeded, step down through LEGAL depths only.
  # --------------------------------------------------------------------------------------------
  if hard_budget and not depth_is_pinned and "base_num_decoder_layers" in allowed_fields:
    current_layers = _resolved_layers_for(spec, retained_cycles, retained_tail)
    smaller_pairs = sorted(
        (pair for pair in legal_pairs if _resolved_layers_for(spec, *pair) < current_layers),
        key=lambda kt: -_resolved_layers_for(spec, *kt),
    )
    while estimate_model_parameters(candidate)["total_params"] > target_params and smaller_pairs:
      retained_cycles, retained_tail = smaller_pairs.pop(0)
      candidate.base_num_decoder_layers = resolved_depth_to_base(
          spec, _resolved_layers_for(spec, retained_cycles, retained_tail)
      )
      _refresh_candidate(src, candidate)

  _validate_candidate_constraints(candidate, spec)
  remapped_metadata = compute_remapped_metadata(src, candidate.base_num_decoder_layers, candidate.base_emb_dim)
  final_stats = estimate_model_parameters(candidate)
  final_total = max(1, final_stats["total_params"])
  actual_factor = orig_total / final_total

  # --------------------------------------------------------------------------------------------
  # Phase 5: acceptance.
  # --------------------------------------------------------------------------------------------
  if hard_budget and final_total > target_params:
    raise ValueError(
        f"Candidate model ({_format_params(final_total)}) exceeds hard_budget ({_format_params(target_params)}) "
        f"for '{model_label}' under strategy='{strategy}'. The smallest candidate under this policy has "
        f"{_format_params(min_total)}."
    )

  undershoot_note = ""
  if explicit_budget_source and not hard_budget:
    if abs(final_total - target_params) / target_params > _MAX_ALLOWED_RELATIVE_DEVIATION:
      raise ValueError(
          f"No candidate under strategy='{strategy}' satisfies target {_format_params(target_params)} "
          f"(±{int(_MAX_ALLOWED_RELATIVE_DEVIATION * 100)}%) for '{model_label}' ({_format_params(orig_total)}). "
          f"Best candidate under this reduction policy has {_format_params(final_total)} (factor={actual_factor:.2f}x); "
          f"minimum under this policy has {_format_params(min_total)} (max factor={max_factor:.2f}x). "
          f"Set `hard_budget=True` to accept any candidate at or below the budget."
      )
  elif hard_budget and (target_params - final_total) / target_params > _MAX_ALLOWED_RELATIVE_DEVIATION:
    undershoot_note = (
        f"Budget undershoot accepted under hard_budget=True "
        f"({_format_params(final_total)} vs budget {_format_params(target_params)}; "
        f"{(target_params - final_total) / target_params * 100:.1f}% below)"
    )

  # --------------------------------------------------------------------------------------------
  # Phase 6: build the manifest.
  # --------------------------------------------------------------------------------------------
  planned_fields: dict[str, Any] = {}
  changed_fields: list[str] = []
  for attr in sorted(allowed_fields):
    old_val = getattr(src, attr, None)
    new_val = getattr(candidate, attr, None)
    if new_val is not None:
      planned_fields[attr] = new_val
      if old_val != new_val:
        changed_fields.append(f"{attr}: {old_val} -> {new_val}")

  for key, val in remapped_metadata.items():
    old_val = getattr(src, key, None)
    if old_val != val:
      changed_fields.append(f"{key}: {old_val} -> {val}")

  preserved: list[str] = [
      f"Architectural cycle periodicity (cycle_length={spec.arch_cycle_length}, retained_cycles={retained_cycles})",
      f"Attention head dimension (head_dim={getattr(src, 'head_dim', 128)})",
  ]
  if spec.prefix_layers > 0:
    preserved.append(f"Prefix layers (prefix_layers={spec.prefix_layers})")
  if int(getattr(src, "num_experts", 1)) > 1:
    if int(getattr(candidate, "num_experts_per_tok", 1)) == int(getattr(src, "num_experts_per_tok", 1)):
      preserved.append(f"MoE active top-k routing (num_experts_per_tok={candidate.num_experts_per_tok})")
    if int(getattr(src, "n_routing_groups", 1)) > 1:
      preserved.append(
          f"Grouped routing capacity (n_routing_groups={src.n_routing_groups}, "
          f"{int(candidate.num_experts) // int(src.n_routing_groups)} experts/group)"
      )
    if int(getattr(src, "shared_experts", 0)) > 0:
      preserved.append(f"Shared expert path (shared_experts={src.shared_experts})")
  if int(candidate.base_emb_dim) == int(src.base_emb_dim):
    preserved.append(f"Hidden embedding dimension (base_emb_dim={src.base_emb_dim})")

  relaxed: list[str] = []

  # DeepStack visual injection: `nnx_decoders` consumes visual feature `i` at decoder layer `i`, so
  # the number of CONSUMERS is bounded by decoder depth, not by the largest ViT extraction index. A
  # slice shallower than the feature count silently drops the trailing features, which is exactly
  # the kind of loss the preservation report exists to surface.
  retained_layers = _resolved_layers_for(spec, retained_cycles, retained_tail)
  required_consumers = int(spec.min_layers_for_modality_coverage)
  if required_consumers > 0:
    if retained_layers >= required_consumers:
      preserved.append(
          f"DeepStack visual injection consumers ({required_consumers} of {required_consumers} "
          f"features retained across {retained_layers} decoder layers)"
      )
    else:
      relaxed.append(
          f"DeepStack visual injection coverage: only {retained_layers} of {required_consumers} "
          f"visual features are consumed ({required_consumers - retained_layers} dropped); "
          "multimodal fusion behavior is NOT representative at this depth"
      )

  if spec.tail_layers > 0 and retained_tail == 0:
    relaxed.append(f"Omitted partial tail layers (tail_layers: {spec.tail_layers} -> 0)")
  if int(getattr(candidate, "num_experts_per_tok", 1)) != int(getattr(src, "num_experts_per_tok", 1)):
    relaxed.append(
        f"Reduced MoE top-k ({src.num_experts_per_tok} -> {candidate.num_experts_per_tok}, kept >= 2)"
    )
  if undershoot_note:
    relaxed.append(undershoot_note)

  # Sharding feasibility provenance: distinguish "checked against declared degrees" from "assumed".
  unresolved_axes = _unresolved_parallelism_axes(src)

  plan = ReductionPlan(
      source_model_name=model_label,
      strategy=strategy,
      requested_factor=reduce_factor,
      actual_factor=actual_factor,
      target_params=int(target_params),
      hard_budget=hard_budget,
      explicit_budget=bool(explicit_budget_source),
      budget_source=explicit_budget_source or "target_num_cycles",
      orig_stats=orig_stats,
      final_stats=final_stats,
      min_total_params=int(min_total),
      max_reduce_factor=max_factor,
      arch_spec=spec,
      retained_cycles=retained_cycles,
      retained_tail_layers=retained_tail,
      retained_resolved_layers=retained_layers,
      mesh_validation="deferred" if unresolved_axes else "passed",
      unresolved_parallelism_axes=unresolved_axes,
      planned_fields=planned_fields,
      remapped_metadata=remapped_metadata,
      preserved_invariants=preserved,
      changed_fields=changed_fields,
      relaxed_invariants=relaxed,
  )

  # Phase 7: single final contract validator, run for EVERY branch.
  _validate_plan_contract(src, spec, plan, candidate, bool(explicit_budget_source))
  return plan


def apply_reduction_plan(config: Any, plan: ReductionPlan, log: bool = True) -> dict[str, Any]:
  """Atomically applies a validated `ReductionPlan` to `config` once.

  This is the ONLY function that mutates a config. `plan_model_reduction` dry-applies the plan to a
  copy of the source through this same function, so the applied architecture is guaranteed to equal
  the architecture the plan reports (see `_validate_plan_contract`).
  """
  for attr, val in plan.planned_fields.items():
    setattr(config, attr, val)
  derive_dimensions(config)
  for attr, val in plan.remapped_metadata.items():
    setattr(config, attr, val)
  # Neutralize the one-shot triggers so the applied config describes the slice rather than a
  # pending request to slice again.
  config.model_reduce_factor = plan.requested_factor
  config.target_num_cycles = -1

  if log:
    max_logging.log(
        f"[ModelReducer] Applied plan for '{plan.source_model_name}' ({plan.strategy}): "
        f"{_format_params(plan.orig_stats['total_params'])} -> {_format_params(plan.final_stats['total_params'])} "
        f"(target: {_format_params(plan.target_params)}, factor: {plan.actual_factor:.3f}x, "
        f"verification: {plan.verification_mode}, "
        f"min under policy: {_format_params(plan.min_total_params)} [max {plan.max_reduce_factor:.1f}x], "
        f"layers: {plan.orig_stats['num_dense_layers'] + plan.orig_stats['num_moe_layers']} -> {config.num_decoder_layers})"
    )
  return {
      "strategy": plan.strategy,
      "requested_factor": plan.requested_factor,
      "actual_factor": plan.actual_factor,
      "target_params": plan.target_params,
      "min_total_params": plan.min_total_params,
      "max_reduce_factor": plan.max_reduce_factor,
      "orig_stats": plan.orig_stats,
      "final_stats": plan.final_stats,
      "orig_layers": plan.orig_stats["num_dense_layers"] + plan.orig_stats["num_moe_layers"],
      # Resolved (post-`global_parameter_scale`) depth, matching `plan.retained_*` accounting.
      "final_layers": int(config.num_decoder_layers),
      "final_base_layers": int(config.base_num_decoder_layers),
      "verification_mode": plan.verification_mode,
      "plan": plan,
  }


# Reduction runs inside `MaxTextConfig.model_post_init`, which returns a config, not a manifest, so
# the manifest has to be retrievable afterwards. A single process-global "last plan" is wrong for
# library and notebook use: a second reduction, or a request that produces no plan at all, leaves a
# stale manifest that a caller can unknowingly report or verify against the WRONG config. The plan
# is therefore keyed by the identity of the config it was applied to.
#
# `WeakValueDictionary` cannot be used (plans are strongly held), and pydantic models are not
# reliably weak-referenceable, so identity is tracked by `id()` with a weakref finalizer to evict
# the entry when possible. `id()` can be recycled after an object dies, so each entry also stores a
# weakref to the config and `get_last_applied_plan(config)` re-checks that the referent is the very
# same object before returning the plan.
_PLAN_REGISTRY: dict[int, tuple[Any, ReductionPlan]] = {}
_LAST_APPLIED_PLAN: ReductionPlan | None = None


def _register_plan(config: Any, plan: ReductionPlan | None) -> None:
  """Associates `plan` with `config`, or clears any association when `plan` is None."""
  global _LAST_APPLIED_PLAN  # pylint: disable=global-statement
  key = id(config)
  if plan is None:
    _PLAN_REGISTRY.pop(key, None)
    _LAST_APPLIED_PLAN = None
    return
  ref: Any
  try:
    ref = weakref.ref(config)
    weakref.finalize(config, _PLAN_REGISTRY.pop, key, None)
  except TypeError:  # object does not support weak references
    ref = None
  _PLAN_REGISTRY[key] = (ref, plan)
  _LAST_APPLIED_PLAN = plan


def get_last_applied_plan(config: Any = None) -> ReductionPlan | None:
  """Returns the plan applied to `config`, or `None` if that config was not reduced.

  Always pass `config`. Omitting it falls back to "the last plan applied anywhere in this process",
  which is only safe in a single-shot CLI run; in a reused process that fallback can return a plan
  belonging to a different configuration.
  """
  if config is None:
    return _LAST_APPLIED_PLAN
  entry = _PLAN_REGISTRY.get(id(config))
  if entry is None:
    # The plan may have been registered against the underlying pydantic model rather than the
    # `HyperParameters` wrapper (or vice versa); try the counterpart before giving up.
    inner = getattr(config, "_pydantic_config", None)
    if inner is not None:
      entry = _PLAN_REGISTRY.get(id(inner))
    if entry is None:
      return None
    config = inner
  ref, plan = entry
  if ref is not None and ref() is not config:
    # `id()` was recycled by a different object: the plan does not belong to this config.
    return None
  return plan


def apply_model_reduce_factor(config: Any) -> dict[str, Any] | None:
  """Plans and atomically applies `model_reduce_factor` / `target_model_size` / `target_num_cycles`."""
  plan = plan_model_reduction(config)
  if plan is None:
    # Clear rather than leave a previous plan associated with this config object.
    _register_plan(config, None)
    return None
  result = apply_reduction_plan(config, plan)
  _register_plan(config, plan)
  return result


def save_reduced_model_yaml(base_model_name: str, reduced_cfg: Any, output_path: str) -> None:
  """Exports a self-contained reduced YAML config satisfying `architecture(reload(export(C))) == architecture(C)`.

  Preserves `model_name` identity (so model-name-keyed helpers like `gemma4-e2b` and `gemma3` retain their
  exact variant behavior), neutralizes re-reduction flags (`model_reduce_factor: 1.0`, `target_model_size: ""`),
  and serializes all architectural and layout fields.
  """
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel

  src = getattr(reduced_cfg, "_pydantic_config", reduced_cfg)
  model_yml_path = os.path.join(pyconfig.MAXTEXT_CONFIGS_DIR, "models", f"{base_model_name}.yml")
  if os.path.exists(model_yml_path):
    doc = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(model_yml_path), resolve=True)
  else:
    doc = {}

  # Preserve model identity & enable overrides so reloading this YAML via `base.yml` reproduces `reduced_cfg`
  doc["base_config"] = "base.yml"
  doc["model_name"] = base_model_name
  doc["override_model_config"] = True

  # Reduction control flags are re-written last (see below) so they can never be resurrected from
  # `reduced_cfg`, which still carries the factor that produced this slice.
  reduction_flag_keys = tuple(_REDUCTION_CONTROL_FIELDS)
  identity_keys = _EXPORT_IDENTITY_KEYS

  def _serializable(value: Any) -> Any:
    """Normalizes a resolved config value into a YAML-serializable scalar/list, or `_UNSET`."""
    if hasattr(value, "value"):  # enum-like
      value = value.value
    if isinstance(value, tuple):
      value = list(value)
    if isinstance(value, list):
      value = [_serializable(v) for v in value]
      return _UNSET if any(v is _UNSET for v in value) else value
    if value is None or isinstance(value, (bool, int, float, str)):
      return value
    return _UNSET

  spec_fields = get_exported_model_spec_fields()
  declared_defaults = _exported_field_defaults()

  # The export is the union of two provenance rules. A key is written iff:
  #
  #   (1) the ORIGINAL model YAML declares it. Re-writing it with the RESOLVED value is required
  #       because otherwise the original-model value (e.g. the full `base_num_decoder_layers`)
  #       survives into the artifact and overrides the slice on reload; or
  #   (2) it belongs to the exported model-spec schema AND its resolved value differs from the
  #       field's declared pydantic default.
  #
  # Rule (2) is what makes the export complete: a schema key that is in neither the original model
  # YAML nor the export resolves, on reload, to exactly the declared default - so writing only the
  # non-default keys is sufficient AND minimal. Minimality matters beyond file size: inferred
  # parallelism (`-1`) equals the default, so mesh axes are never accidentally pinned to the
  # exporting host's topology.
  #
  # This is deliberately NOT "dump every runtime field": job state (credentials, dataset paths,
  # run names, checkpoint directories) is outside `_EXPORTED_CONFIG_GROUPS` and never serialized.
  export_keys = set(doc.keys()) | set(spec_fields)
  for k in sorted(export_keys):
    if k in identity_keys or k in reduction_flag_keys or k == SNAPSHOT_MARKER_FIELD:
      continue
    if not hasattr(src, k):
      continue
    val = _serializable(getattr(src, k))
    if val is _UNSET:
      continue
    if k in doc:
      # Declared by the original model YAML: always rewrite with the resolved value.
      doc[k] = val
      continue
    # Schema-only key: write it only when it deviates from the declared default.
    default = declared_defaults.get(k, _UNSET)
    if default is _UNSET:
      doc[k] = val
      continue
    default_serialized = _serializable(default)
    if default_serialized is _UNSET or default_serialized != val:
      if val is not None:
        doc[k] = val

  # Mark this file as an already-resolved architecture snapshot. `pyconfig._collect_user_config_keys`
  # keys off this marker so the artifact behaves identically wherever it is stored - in particular,
  # placing it under `configs/models/` must not cause it to be treated as a packaged default (which
  # would silently restore the ORIGINAL architecture).
  doc[SNAPSHOT_MARKER_FIELD] = True

  # Neutralize re-reduction: reloading this YAML must reproduce the slice, not re-slice it.
  doc["model_reduce_factor"] = 1.0
  doc["target_model_size"] = ""
  doc["target_num_cycles"] = -1
  doc["hard_budget"] = False
  doc["model_reduce_strategy"] = "auto"

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  omegaconf.OmegaConf.save(config=omegaconf.OmegaConf.create(doc), f=output_path)
  max_logging.log(f"[ModelReducer] Saved reproducible reduced model config to {output_path}")


def _requires_unscanned_layers(model_name: str) -> bool:
  """Returns True for models whose architecture requires `scan_layers=False` (per-layer KV sharing or deepstack ViT)."""
  return model_name in (
      "gemma4-e2b",
      "gemma4-e4b",
      "qwen3-vl-2b",
      "qwen3-vl-4b",
      "cosmos3-nano-reasoner",
      "cosmos3-super-reasoner",
  )


def main(
    args: list[str],
    output_config_path: str = "",
    compare_all_strategies: bool = False,
    verify_params: bool = False,
) -> None:
  """Main execution matching `to_maxtext.py` / `to_huggingface.py` CLI conventions."""
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel

  extra_overrides = ["override_model_config=True", "skip_jax_distributed_system=True"]
  if not any(a.startswith("attention=") for a in args):
    extra_overrides.append("attention=dot_product")

  # Explicitly neutralize reduction settings when loading the unreduced baseline (even if present in input YAML)
  neutralize_reduction = [
      "model_reduce_factor=1.0",
      "target_model_size=",
      "target_num_cycles=-1",
  ]
  base_args = [
      a
      for a in args
      if not a.startswith("model_reduce_factor=")
      and not a.startswith("target_model_size=")
      and not a.startswith("target_num_cycles=")
      and not a.startswith("model_reduce_strategy=")
  ] + extra_overrides + neutralize_reduction

  # Check if target model requires scan_layers=False
  model_arg = next((a.split("=", 1)[1] for a in base_args if a.startswith("model_name=")), "default")
  if _requires_unscanned_layers(model_arg) and not any(a.startswith("scan_layers=") for a in base_args):
    base_args.append("scan_layers=False")

  orig_cfg = pyconfig.initialize_pydantic(base_args)
  orig_stats = estimate_model_parameters(orig_cfg)
  spec = get_architecture_spec(orig_cfg)
  struct_min = compute_minimal_representative_model(orig_cfg, "structural")
  compact_min = compute_minimal_representative_model(orig_cfg, "compact")

  max_logging.log(f"\n=== Model Reduction Summary for '{orig_cfg.model_name}' ===")
  max_logging.log(
      f"Original: Total={_format_params(orig_stats['total_params'])}, Active/tok={_format_params(orig_stats['active_params'])} | "
      f"layers={orig_cfg.base_num_decoder_layers} (prefix={spec.prefix_layers}, cycle={spec.arch_cycle_length}x{spec.full_cycles}, tail={spec.tail_layers}), "
      f"emb_dim={orig_cfg.base_emb_dim}, mlp_dim={orig_cfg.base_mlp_dim}, moe_mlp_dim={orig_cfg.base_moe_mlp_dim}, "
      f"experts={orig_cfg.num_experts} (top-{orig_cfg.num_experts_per_tok})"
  )
  max_logging.log(
      f"Minimum Structural Slice: {_format_params(struct_min['min_total_params'])} (max factor={struct_min['max_reduce_factor']:.2f}x) | "
      f"Minimum Compact Model: {_format_params(compact_min['min_total_params'])} (max factor={compact_min['max_reduce_factor']:.2f}x)"
  )

  run_args = list(args) + extra_overrides
  if _requires_unscanned_layers(model_arg) and not any(a.startswith("scan_layers=") for a in run_args):
    run_args.append("scan_layers=False")
  reduced_cfg = pyconfig.initialize_pydantic(run_args)

  # Running as `python -m maxtext.configs.model_reducer` imports this file twice: once as
  # `__main__` (this code) and once as `maxtext.configs.model_reducer` (what `types.py` calls during
  # `model_post_init`). The plan is recorded on the latter, so read the registry from the canonical
  # module rather than from this one.
  from maxtext.configs import model_reducer as canonical_module  # pylint: disable=import-outside-toplevel

  # Ask for the plan bound to THIS config, never for "the last plan applied anywhere".
  plan = canonical_module.get_last_applied_plan(reduced_cfg) or get_last_applied_plan(reduced_cfg)
  if plan is not None:
    if verify_params:
      # Verification must happen here, not inside the planner: planning runs during
      # `model_post_init`, before the configuration can construct a model.
      verify_plan_with_abstract_params(reduced_cfg, plan)
    max_logging.log("\n" + plan.format_fidelity_report())

  if compare_all_strategies:
    for strat in ("structural", "compact", "experts", "width", "budget"):
      cfg_copy = copy.deepcopy(orig_cfg)
      cfg_copy.model_reduce_factor = reduced_cfg.model_reduce_factor
      cfg_copy.target_model_size = reduced_cfg.target_model_size
      cfg_copy.target_num_cycles = getattr(reduced_cfg, "target_num_cycles", -1)
      cfg_copy.model_reduce_strategy = strat
      min_info = compute_minimal_representative_model(orig_cfg, strat)
      try:
        plan = plan_model_reduction(cfg_copy)
        if plan:
          apply_reduction_plan(cfg_copy, plan)
          f_stats = plan.final_stats
          max_logging.log(
              f"  [{strat:10s}] Total={_format_params(f_stats['total_params'])} ({plan.actual_factor:.2f}x), "
              f"Active/tok={_format_params(f_stats['active_params'])}, "
              f"MinPolicy={_format_params(min_info['min_total_params'])} (max {min_info['max_reduce_factor']:.1f}x) | "
              f"layers={cfg_copy.base_num_decoder_layers}, emb_dim={cfg_copy.base_emb_dim}, "
              f"mlp_dim={cfg_copy.base_mlp_dim}, moe_mlp_dim={cfg_copy.base_moe_mlp_dim}, "
              f"experts={cfg_copy.num_experts} (top-{cfg_copy.num_experts_per_tok})"
          )
      except ValueError as err:
        max_logging.log(
            f"  [{strat:10s}] INFEASIBLE (MinPolicy={_format_params(min_info['min_total_params'])}, "
            f"max factor={min_info['max_reduce_factor']:.2f}x): {err}"
        )

  if output_config_path:
    save_reduced_model_yaml(orig_cfg.model_name, reduced_cfg, output_config_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MaxText representative model reduction tool.")
  parser.add_argument(
      "--output_config_path",
      type=str,
      required=False,
      default="",
      help="Optional path to save the reduced model YAML config.",
  )
  parser.add_argument(
      "--compare_all_strategies",
      type=str2bool,
      required=False,
      default=False,
      help="Whether to print a side-by-side comparison across all reduction strategies.",
  )
  parser.add_argument(
      "--verify_params",
      type=str2bool,
      required=False,
      default=False,
      help=(
          "Whether to verify the reduced model against a constructed abstract parameter tree "
          "(jax.eval_shape). Without this, reported sizes are analytical estimates only."
      ),
  )

  normalized_argv = [sys.argv[0]]
  for raw_cli_arg in sys.argv[1:]:
    for cli_prefix in ("output_config_path=", "compare_all_strategies=", "verify_params="):
      if raw_cli_arg.startswith(cli_prefix):
        raw_cli_arg = "--" + raw_cli_arg
        break
    normalized_argv.append(raw_cli_arg)
  sys.argv = normalized_argv

  local_args, remaining_args = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining_args
  sys.argv = model_args

  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

  main(
      args=model_args,
      output_config_path=local_args.output_config_path,
      compare_all_strategies=local_args.compare_all_strategies,
      verify_params=local_args.verify_params,
  )
