# Copyright 2026 Google LLC
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

"""Composable hardware emulation plugin architecture.

Defines:
1. `EmulationFeature`: Base interface for any pre-silicon hardware numerical emulation
   feature (e.g., Multi-Pass Precision Emulation, M:N structured sparsity, microscaling,
   custom accumulation/rounding). Multiple features compose sequentially in a pipeline.
2. `register_emulation_feature`: Registry decorator/function so new hardware
   features can be plugged in without modifying trainers, hooks, or entry scripts.
"""

from __future__ import annotations

from typing import Any, Callable


class EmulationFeature:
  """Base class for composable hardware numerical emulation features."""

  feature_name: str = "base"
  enabled: bool = False

  def summary(self) -> str:
    return f"{self.__class__.__name__}(enabled={self.enabled})"

  def to_dict(self) -> dict[str, Any]:
    return {"enabled": self.enabled}

  def transform_input_activation(self, x: Any, module_path: str) -> Any:
    """Optional hook to transform module input activations."""
    del module_path
    return x

  def transform_output_activation(self, x: Any, module_path: str) -> Any:
    """Optional hook to transform module output activations."""
    del module_path
    return x

  def transform_dot_operands(
      self,
      lhs: Any,
      rhs: Any,
      dimension_numbers: Any,
      module_path: str,
  ) -> tuple[Any, Any]:
    """Optional hook to transform `(lhs, rhs)` before `dot_general`."""
    del dimension_numbers, module_path
    return lhs, rhs

  def execute_dot_general(
      self,
      lhs: Any,
      rhs: Any,
      dimension_numbers: Any,
      module_path: str,
      base_dot_general_fn: Callable[..., Any],
      precision: Any = None,
      preferred_element_type: Any = None,
  ) -> Any:
    """Executes or wraps `dot_general` for this feature."""
    lhs_t, rhs_t = self.transform_dot_operands(
        lhs, rhs, dimension_numbers, module_path
    )
    return base_dot_general_fn(
        lhs_t,
        rhs_t,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  def execute_einsum(
      self,
      subscripts: str,
      lhs: Any,
      rhs: Any,
      module_path: str,
      base_einsum_fn: Callable[..., Any],
      **kwargs: Any,
  ) -> Any:
    """Executes or wraps 2-operand `einsum` for this feature."""
    del module_path
    return base_einsum_fn(subscripts, lhs, rhs, **kwargs)

  def transform_moe_operands(
      self,
      inputs: Any,
      w0_kernel: Any,
      w1_kernel: Any,
      wo_kernel: Any,
      fused_kernel: Any,
      module_path: str,
  ) -> tuple[Any, Any, Any, Any, Any]:
    """Transforms MoE `(inputs, w0, w1, wo, fused)` operands."""
    del module_path
    return inputs, w0_kernel, w1_kernel, wo_kernel, fused_kernel

  def materialize_model_weights(self, model: Any) -> int:
    """Materializes static weight transformations onto an NNX model's `Param`s."""
    del model
    return 0

  def materialize_pytree_weights(self, params_pytree: Any) -> tuple[Any, int]:
    """Materializes static weight transformations onto a parameter PyTree."""
    return params_pytree, 0

  def inspect_model_or_pytree(self, model_or_pytree: Any) -> dict[str, Any]:
    """Returns diagnostic/compliance metrics for this feature on a model or PyTree."""
    del model_or_pytree
    return {}


_FEATURE_REGISTRY: dict[str, type[EmulationFeature]] = {}


def register_emulation_feature(
    name: str, feature_cls: type[EmulationFeature] | None = None
) -> Any:
  """Registers an `EmulationFeature` subclass by name (usable as decorator or function)."""

  def _register(cls: type[EmulationFeature]) -> type[EmulationFeature]:
    cls.feature_name = name.strip().lower()
    _FEATURE_REGISTRY[cls.feature_name] = cls
    return cls

  if feature_cls is not None:
    return _register(feature_cls)
  return _register


def get_registered_emulation_features() -> dict[str, type[EmulationFeature]]:
  """Returns a copy of the registered `EmulationFeature` classes keyed by name."""
  return dict(_FEATURE_REGISTRY)


import sys as _sys  # pylint: disable=wrong-import-position

_sys.modules.setdefault("emulation_base", _sys.modules[__name__])
_sys.modules.setdefault("maxtext.codesign.emulation_base", _sys.modules[__name__])

