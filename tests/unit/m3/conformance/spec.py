# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declarative schema and registry for MaxText m3 model conformance specs."""

from collections.abc import Callable, Mapping
import dataclasses
import enum
import re

from flax import nnx
import jax

_SCAN_MODES = ((False, "none"), (False, "full"), (True, "none"), (True, "full"))
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 256.
_DEFAULT_MIN_SHARDED_PARAM_ELEMENTS = 256


class CheckId(enum.Enum):
  """Identifiers for the 8 mandatory m3 conformance checks."""

  SCAN_REMAT_EQUIVALENCE = "check_1_scan_remat_equivalence"
  PARAM_AND_GRAD_HEALTH = "check_2_param_and_grad_health"
  NON_PARAM_STATE_STABILITY = "check_3_non_param_state_stability"
  AUX_LOSS_GRAD_COUPLING = "check_4_aux_loss_grad_coupling"
  DRIVER_MESH_AND_AXIS_NAMES = "check_5_driver_mesh_and_axis_names"
  NO_POST_INIT_PARAM_CONSTRAINTS = "check_6_no_post_init_param_constraints"
  WEIGHT_SHARDING_COVERAGE = "check_7_weight_sharding_coverage"
  NO_EXPERT_WEIGHT_ALL_GATHER = "check_8_no_expert_weight_all_gather"


class OutputKind(enum.Enum):
  """Kind of primary tensor produced by a model forward pass."""

  LOGITS = "logits"
  DIFFUSION = "diffusion"
  EMBEDDINGS = "embeddings"


class SkipStatus(enum.Enum):
  """Explicit opt-out status for structurally inapplicable checks."""

  NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclasses.dataclass(frozen=True)
class SkipDeclaration:
  """Explicit declaration that a check is structurally inapplicable."""

  status: SkipStatus
  reason: str


@dataclasses.dataclass(frozen=True)
class ForwardOutput:
  """Normalized output of a model forward pass for conformance checks."""

  primary: jax.Array
  aux_losses: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ConformanceSpec:
  """Declarative conformance specification for an m3 model.

  Regex fields (`re.search`): `frozen_param_patterns`, `aux_losses` target
  patterns,
  `stacked_param_path_patterns`, and `expert_weight_param_patterns`.
  Glob fields (`fnmatch.fnmatchcase`): `mutable_non_param_patterns`.
  """

  model_name: str
  output_kind: OutputKind
  build_model: Callable[[jax.sharding.Mesh, nnx.Rngs, bool, str], nnx.Module]
  build_batch: Callable[[jax.Array, jax.sharding.Mesh], Mapping[str, jax.Array]]
  forward_fn: Callable[[nnx.Module, Mapping[str, jax.Array]], ForwardOutput]
  loss_fn: Callable[
      [ForwardOutput, Mapping[str, jax.Array], Mapping[str, float]],
      jax.Array,
  ]
  scan_remat_variants: tuple[tuple[bool, str], ...] = _SCAN_MODES
  frozen_param_patterns: tuple[str, ...] = ()
  mutable_non_param_patterns: tuple[str, ...] = ()
  aux_losses: Mapping[str, tuple[str, ...]] = dataclasses.field(
      default_factory=dict,
  )
  stacked_param_path_patterns: tuple[str, ...] = ()
  num_scanned_blocks: int = 1
  min_sharded_param_elements: int = _DEFAULT_MIN_SHARDED_PARAM_ELEMENTS
  expert_weight_param_patterns: tuple[str, ...] = ()
  skips: Mapping[CheckId, SkipDeclaration] = dataclasses.field(
      default_factory=dict,
  )

  def validate_spec_invariants(self) -> None:
    """Validates static ConformanceSpec field invariants (`m3/README.md` Invariants (1)-(7))."""
    if not isinstance(self.model_name, str) or not self.model_name.strip():
      raise ValueError(
          f"ConformanceSpec.model_name is blank: {self.model_name!r}.",
      )
    if not isinstance(self.output_kind, OutputKind):
      raise ValueError(
          f"{self.model_name}: invalid output_kind {self.output_kind!r}.",
      )
    pats = self.mutable_non_param_patterns
    ok_pats = isinstance(pats, tuple) and all(isinstance(p, str) for p in pats)
    if not ok_pats or any(p.split() != [p] for p in pats):
      raise ValueError(
          f"{self.model_name}: invalid mutable_non_param_patterns {pats!r}.",
      )
    regex_fields = (
        ("stacked_param_path_patterns", self.stacked_param_path_patterns),
        ("expert_weight_param_patterns", self.expert_weight_param_patterns),
    )
    for pat_field, regex_pats in regex_fields:
      if not isinstance(regex_pats, tuple):
        raise ValueError(f"{self.model_name}: invalid {pat_field} {regex_pats!r}.")
      for p in regex_pats:
        if not (isinstance(p, str) and bool(p.strip()) and p == p.strip()):
          raise ValueError(f"{self.model_name}: invalid {pat_field} {regex_pats!r}.")
        try:
          re.compile(p)
        except re.error as exc:
          raise ValueError(f"{self.model_name}: invalid {pat_field} regex {p!r}: {exc}.") from exc
    for int_field, int_val in (
        ("num_scanned_blocks", self.num_scanned_blocks),
        ("min_sharded_param_elements", self.min_sharded_param_elements),
    ):
      if isinstance(int_val, bool) or not isinstance(int_val, int) or int_val < 1:
        raise ValueError(f"{self.model_name}: {int_field} must be int >= 1, got {int_val!r}.")


_REGISTERED_SPECS: list[ConformanceSpec | Callable[[], ConformanceSpec]] = []


def register_conformance_spec(
    entry: ConformanceSpec | Callable[[], ConformanceSpec],
) -> None:
  """Idempotently registers a ConformanceSpec or zero-argument factory."""
  if all(existing is not entry for existing in _REGISTERED_SPECS):
    _REGISTERED_SPECS.append(entry)


def get_registered_conformance_specs() -> tuple[ConformanceSpec, ...]:
  """Resolves and validates all registered ConformanceSpec instances."""
  by_name: dict[str, ConformanceSpec] = {}
  for entry in _REGISTERED_SPECS:
    resolved = entry() if callable(entry) else entry
    resolved.validate_spec_invariants()
    name = resolved.model_name
    if name in by_name:
      raise ValueError(f"Duplicate ConformanceSpec model_name: {name!r}.")
    by_name[name] = resolved
  return tuple(by_name.values())
