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

"""DeepSeek-V4.1 parity policy wrapper enforcing approved fp32 (1e-5) and fp64 (1e-10) ceilings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

import numpy as np

from tests.utils.parity import layer_parity as _base

HAS_TORCH = _base.HAS_TORCH
Metrics = _base.Metrics
GradientClasses = _base.GradientClasses
ParityReport = _base.ParityReport
compare = _base.compare
compare_padded_rows = _base.compare_padded_rows
hashed_normal = _base.hashed_normal
validate_gradient_classes = _base.validate_gradient_classes
check_forward_and_grads = _base.check_forward_and_grads
reference_noise_floor = _base.reference_noise_floor

# Approved DeepSeek-V4.1 fp32 vs fp64-reference parity ceilings: fp32 accumulation stays within
# 1e-5; fp64 checks enforce 1e-10.
FP32_ATOL_CEILING = 1e-5
FP32_RTOL_CEILING = 1e-5
FP32_FROB_REL_CEILING = 1e-5
FP64_FROB_REL_CEILING = 1e-10


@dataclasses.dataclass(frozen=True)
class Tolerance(_base.Tolerance):
  """DeepSeek-V4.1 `Tolerance` bounded by the approved fp32 ceilings (<= 1e-5)."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.atol > FP32_ATOL_CEILING or self.rtol > FP32_RTOL_CEILING or self.frob_rel > FP32_FROB_REL_CEILING:
      raise ValueError(
          "Tolerance exceeds approved fp32 ceilings "
          f"({FP32_ATOL_CEILING}, {FP32_RTOL_CEILING}, {FP32_FROB_REL_CEILING}): {self}"
      )


FP32_TOLERANCE = Tolerance(
    atol=FP32_ATOL_CEILING,
    rtol=FP32_RTOL_CEILING,
    frob_rel=FP32_FROB_REL_CEILING,
)
FP64_TOLERANCE = Tolerance(
    atol=FP64_FROB_REL_CEILING,
    rtol=FP64_FROB_REL_CEILING,
    frob_rel=FP64_FROB_REL_CEILING,
)


def _enforce_fp64_ceiling(tol: _base.Tolerance, label: str) -> None:
  """Enforces the 1e-10 fp64 ceiling when comparing float64 tensors."""
  if tol.atol > FP64_FROB_REL_CEILING or tol.rtol > FP64_FROB_REL_CEILING or tol.frob_rel > FP64_FROB_REL_CEILING:
    raise ValueError(f"{label} on float64 tensors exceeds FP64_FROB_REL_CEILING ({FP64_FROB_REL_CEILING}): {tol}")


def check_gradients_by_class(
    *,
    ref_params: Mapping[str, Any],
    ref_grads: Mapping[str, Any],
    cand_grads: Mapping[str, np.ndarray],
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    grad_tol: _base.Tolerance,
) -> tuple[dict[str, Metrics], list[str]]:
  """Delegates to `tests.utils.parity.layer_parity.check_gradients_by_class` with DS4.1 fp64 ceilings."""
  return _base.check_gradients_by_class(
      ref_params=ref_params,
      ref_grads=ref_grads,
      cand_grads=cand_grads,
      weight_map=weight_map,
      grad_classes=grad_classes,
      grad_tol=grad_tol,
      fp64_tol_validator=_enforce_fp64_ceiling,
  )


def check_parity(
    *,
    ref_fn: Callable[[dict[str, Any], Any], Any],
    ref_params: dict[str, Any],
    ref_input: Any,
    cand_fn: Callable[[Any, Any], Any],
    cand_params: Any,
    cand_input: Any,
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    fwd_tol: _base.Tolerance,
    grad_tol: _base.Tolerance,
    cotangent_seed: int,
) -> ParityReport:
  """Delegates to `tests.utils.parity.layer_parity.check_parity` with DS4.1 fp64 ceilings."""
  return _base.check_parity(
      ref_fn=ref_fn,
      ref_params=ref_params,
      ref_input=ref_input,
      cand_fn=cand_fn,
      cand_params=cand_params,
      cand_input=cand_input,
      weight_map=weight_map,
      grad_classes=grad_classes,
      fwd_tol=fwd_tol,
      grad_tol=grad_tol,
      cotangent_seed=cotangent_seed,
      fp64_tol_validator=_enforce_fp64_ceiling,
  )


def negative_control_fires(
    *,
    cand_fn: Callable[[Any, Any], Any],
    cand_params: Any,
    fwd_tol: _base.Tolerance,
    target_path: str,
    rel_scale: float,
    seed: int,
    cand_inputs: Any = None,
    ref_outputs: Mapping[str, np.ndarray] | None = None,
    cand_input: Any = None,
    ref_out: np.ndarray | None = None,
) -> tuple[bool, Any]:
  """Delegates to `tests.utils.parity.layer_parity.negative_control_fires` with DS4.1 fp64 ceilings."""
  return _base.negative_control_fires(
      cand_fn=cand_fn,
      cand_params=cand_params,
      fwd_tol=fwd_tol,
      target_path=target_path,
      rel_scale=rel_scale,
      seed=seed,
      cand_inputs=cand_inputs,
      ref_outputs=ref_outputs,
      cand_input=cand_input,
      ref_out=ref_out,
      fp64_tol_validator=_enforce_fp64_ceiling,
  )
