# Copyright 2023–2025 Google LLC
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
"""Quantization configuration management for Megablox ops."""

from dataclasses import dataclass, asdict
from typing import Optional, Mapping, Literal, Any
import jax.numpy as jnp
import qwix

Calib = str
# Type alias for the different phases of automatic differentiation.
Phase = Literal["fwd", "dlhs", "drhs"]

@dataclass(frozen=True)
class QuantizationConfig:
    """
    Specifies the quantization types and calibration methods for matrix multiplication.

    Attributes:
        lhs_quantize_dtype: The target dtype for the left-hand side tensor (e.g., jnp.int8).
                            If None, quantization is skipped.
        rhs_quantize_dtype: The target dtype for the right-hand side tensor.
        lhs_calibration_method: The method used for calibrating the LHS tensor.
        rhs_calibration_method: The method used for calibrating the RHS tensor.
    """
    lhs_quantize_dtype: Optional[jnp.dtype]  # e.g., jnp.int8 / jnp.int4 / None
    rhs_quantize_dtype: Optional[jnp.dtype]
    lhs_calibration_method: Calib
    rhs_calibration_method: Calib

def _require(add: Mapping[str, Any], key: str, phase: Phase) -> Any:
    """A helper to ensure a required key exists in the config mapping."""
    if key not in add:
        raise KeyError(
            f"additional_qt_config is missing the required key '{key}' for"
            f" phase='{phase}'"
        )
    return add[key]

class QuantizationManager:
    """
    Resolves and manages quantization configurations for each phase of a VJP.

    This class inspects a `qwix.QtRule` and pre-computes the quantization
    configurations for the forward pass ('fwd'), the gradient pass with respect
    to the inputs ('dlhs'), and the gradient pass with respect to the weights ('drhs').
    This avoids conditional logic in hot code paths.

    If `use_qwix_quantization` is False or `quantization_rule` is None, it gracefully
    falls back to a user-provided `QuantizationConfig`.
    """

    def __init__(
        self,
        *,
        quantization_rule: qwix.QtRule,
        use_qwix_quantization: bool,
        # Fallback config if qwix is not used. This ensures non-qwix behavior
        # is determined by the caller's explicit kwargs.
        fallback: QuantizationConfig,
    ):
        self._use_qwix = use_qwix_quantization and (quantization_rule is not None)
        self._fwd: QuantizationConfig
        self._dlhs: QuantizationConfig
        self._drhs: QuantizationConfig

        if not self._use_qwix:
            # If not using qwix, apply the same fallback config to all phases.
            self._fwd = self._dlhs = self._drhs = fallback
            return

        rule = quantization_rule
        add = rule.additional_qt_config

        # Forward pass: (activation × weight)
        self._fwd = QuantizationConfig(
            lhs_quantize_dtype=rule.act_qtype,
            rhs_quantize_dtype=rule.weight_qtype,
            lhs_calibration_method=rule.act_calibration_method,
            rhs_calibration_method=rule.weight_calibration_method,
        )

        # Backward pass for inputs (dlhs): (grad × weight_T)
        # Prefers per-phase overrides; otherwise defaults to (act × bwd).
        if add:
            self._dlhs = QuantizationConfig(
                lhs_quantize_dtype=_require(add, "dlhs_lhs_qtype", "dlhs"),
                rhs_quantize_dtype=_require(add, "dlhs_rhs_qtype", "dlhs"),
                lhs_calibration_method=_require(
                    add, "dlhs_lhs_calibration_method", "dlhs"
                ),
                rhs_calibration_method=_require(
                    add, "dlhs_rhs_calibration_method", "dlhs"
                ),
            )
        else:
            self._dlhs = QuantizationConfig(
                lhs_quantize_dtype=rule.act_qtype,
                rhs_quantize_dtype=rule.bwd_qtype,
                lhs_calibration_method=rule.act_calibration_method,
                rhs_calibration_method=rule.weight_calibration_method,
            )

        # Backward pass for weights (drhs): (activation_T × grad)
        # Prefers per-phase overrides; otherwise defaults to (bwd × act).
        if add:
            self._drhs = QuantizationConfig(
                lhs_quantize_dtype=_require(add, "drhs_lhs_qtype", "drhs"),
                rhs_quantize_dtype=_require(add, "drhs_rhs_qtype", "drhs"),
                lhs_calibration_method=_require(
                    add, "drhs_lhs_calibration_method", "drhs"
                ),
                rhs_calibration_method=_require(
                    add, "drhs_rhs_calibration_method", "drhs"
                ),
            )
        else:
            self._drhs = QuantizationConfig(
                lhs_quantize_dtype=rule.bwd_qtype,
                rhs_quantize_dtype=rule.act_qtype,
                lhs_calibration_method=rule.weight_calibration_method,
                rhs_calibration_method=rule.act_calibration_method,
            )

    def for_phase(self, phase: Phase) -> QuantizationConfig:
        """Simple accessor to retrieve the config for a specific phase."""
        if phase == "fwd":
            return self._fwd
        if phase == "dlhs":
            return self._dlhs
        if phase == "drhs":
            return self._drhs
        raise ValueError(f"Unknown phase: {phase}")

    def as_kwargs(self, phase: Phase) -> dict[str, Any]:
        """
        Convenience method to return the configuration as a dictionary of kwargs
        for backend gmm/tgmm functions.
        """
        q_config = self.for_phase(phase)
        return asdict(q_config)
