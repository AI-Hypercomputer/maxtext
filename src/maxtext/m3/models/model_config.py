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

"""Base `ModelConfig` for m3 Model Garden families."""

from typing import Any

from maxtext.configs.components import DType
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
  """Frozen, strictly validated base architecture configuration for m3 Model Garden families.

  Uses native HuggingFace field names (`model_type`, `vocab_size`, `hidden_size`, `num_hidden_layers`).
  Concrete model families subclass `ModelConfig` inside `src/maxtext/m3/models/<family>/`.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  model_type: str = Field(..., description="HuggingFace architecture family identifier.")
  vocab_size: int = Field(..., gt=0, description="Vocabulary size.")
  hidden_size: int = Field(..., gt=0, description="Model hidden / embedding dimension.")
  num_hidden_layers: int = Field(..., gt=0, description="Number of transformer decoder layers.")
  model_name: str = Field("", description="Canonical registered model identifier.")
  param_dtype: DType = Field(DType.FLOAT32, description="Master parameter storage dtype.")
  compute_dtype: DType = Field(DType.BFLOAT16, description="Forward/backward activation compute dtype.")
  param_scan_axis: int = Field(
      1,
      ge=0,
      description="Default scanned-layer stacking axis (1 matches `train.py` checkpoints; m3-native families use 0).",
  )

  def supported_mesh_axes(self) -> frozenset[str]:
    """Returns the mesh axes this family implements; `validate_training_config` rejects any other axis set != 1.

    Defaults to every axis `m3/core/sharding.py` shards over. Families that do not implement an axis yet (e.g.
    `context` or `tensor_sequence`) override this to a subset.
    """
    return frozenset({"data", "fsdp", "fsdp_transpose", "context", "tensor", "tensor_sequence", "expert"})

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    """Returns `{mesh_axis: {field_name: value}}` dims that must divide evenly by that mesh axis size.

    Families override this (e.g. `{"tensor": {"num_attention_heads": ...}, "expert": {"num_experts": ...}}`) so
    cross-config checks use each family's own field names instead of guessing attribute names.
    """
    return {}

  def to_pyconfig_dict(self) -> dict[str, Any] | None:
    """Returns a `MaxTextConfig`-compatible override dict for `pyconfig.py` YAML-free fallback.

    Transitional: remove together with the `pyconfig.py` registry fallback once pre-training no longer reads
    `base.yml`.
    """
    return None

  @classmethod
  def from_model_name(cls, model_name: str, **overrides: Any) -> "ModelConfig":
    """Returns a frozen `ModelConfig` for `model_name`, applying strictly validated `overrides`."""
    from maxtext.m3.models.registry import get_model_config  # pylint: disable=import-outside-toplevel

    return get_model_config(model_name, cls, **overrides)
