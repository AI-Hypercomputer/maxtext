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

"""Conformance spec template matching `tests/unit/m3/conformance/spec.py`.

Copy to `src/maxtext/m3/models/<family>/conformance_spec.py`, replace every
`_fill_in(...)` inside `build_spec()`, and uncomment
`register_conformance_spec(build_spec)` at the bottom of the copied file so
`tests/unit/m3/conformance/conformance_test.py` discovers and runs Checks 1-12
(including Check 9b `PACKING_AND_PADDING_ISOLATION`).
"""

from __future__ import annotations

from typing import Any

from tests.unit.m3.conformance import spec as conformance_spec

CheckId = conformance_spec.CheckId
ConformanceSpec = conformance_spec.ConformanceSpec
ModelConformanceSpec = conformance_spec.ConformanceSpec
ForwardOutput = conformance_spec.ForwardOutput
OutputKind = conformance_spec.OutputKind
SkipDeclaration = conformance_spec.SkipDeclaration
SkipStatus = conformance_spec.SkipStatus
register_conformance_spec = conformance_spec.register_conformance_spec


def _fill_in(what: str) -> Any:
  """Raises when an unfilled template placeholder is evaluated."""
  raise NotImplementedError(f"Conformance spec template not filled in: {what}")


def build_spec() -> ConformanceSpec:
  """Returns the validated `ConformanceSpec` for one `m3` model."""
  return ConformanceSpec(
      model_name=_fill_in("MODEL_REGISTRY key, e.g. 'qwen3-0.6b'"),
      output_kind=OutputKind.LOGITS,
      build_model=_fill_in("(mesh, rngs, scan_layers, remat_policy) -> nnx.Module mini-twin"),
      build_batch=_fill_in("(prng_key, mesh) -> Mapping[str, jax.Array] with input_ids, positions, segment_ids"),
      forward_fn=_fill_in("(model, batch) -> ForwardOutput(primary=..., aux_losses={...})"),
      loss_fn=_fill_in("(forward_output, batch, aux_weights) -> scalar float32 jax.Array"),
      frozen_param_patterns=(),
      mutable_non_param_patterns=(),
      aux_losses={},
      stacked_param_path_patterns=_fill_in(
          "tuple of regexes matching scanned layer parameter paths, e.g. (r'layers\\.stacked\\.',)"
      ),
      num_scanned_blocks=2,
      expert_weight_param_patterns=(),
      build_bf16_model=_fill_in("(mesh, rngs, scan_layers, remat_policy) -> bfloat16 nnx.Module, or None"),
      build_full_scale_abstract_model=_fill_in(
          "(mesh_or_abstract_mesh, scan_layers, remat_policy) -> abstract full-scale nnx.Module, or None"
      ),
      skips={
          # Example when a check is genuinely non-applicable (e.g. dense model):
          # CheckId.NO_EXPERT_WEIGHT_ALL_GATHER: SkipDeclaration(
          #     status=SkipStatus.NOT_APPLICABLE,
          #     reason="Dense architecture has no routed MoE expert weights.",
          # ),
      },
  )


# Uncomment in `src/maxtext/m3/models/<family>/conformance_spec.py`:
# register_conformance_spec(build_spec)
