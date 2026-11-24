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

"""Unit tests for base `ModelConfig` and `MODEL_GARDEN_REGISTRY`."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from maxtext.m3.models.model_config import ModelConfig
from maxtext.m3.models.registry import (
    MODEL_GARDEN_REGISTRY,
    get_pyconfig_model_dict,
    normalize_model_name,
)
from pydantic import ValidationError, computed_field


class _StubDenseConfig(ModelConfig):
  """Minimal dense `ModelConfig` subclass for registry tests."""

  model_type: str = "stub_dense"
  model_name: str = "stub-dense"
  vocab_size: int = 256
  hidden_size: int = 64
  num_hidden_layers: int = 2

  def to_pyconfig_dict(self) -> dict[str, Any]:
    return {"base_emb_dim": self.hidden_size, "vocab_size": self.vocab_size}


class _StubMoeConfig(ModelConfig):
  """Minimal MoE `ModelConfig` subclass for subclass-type checks."""

  model_type: str = "stub_moe"
  model_name: str = "stub-moe"
  vocab_size: int = 256
  hidden_size: int = 64
  num_hidden_layers: int = 2


class ModelConfigTest(absltest.TestCase):
  """Tests `ModelConfig` invariants and static `MODEL_GARDEN_REGISTRY` lookups."""

  def test_empty_registry_error_and_default_pyconfig_dict(self):
    self.assertEqual(normalize_model_name("qwen3-0.6b-Instruct"), "qwen3-0.6b")
    with mock.patch.dict(MODEL_GARDEN_REGISTRY, {}, clear=True):
      with self.assertRaisesRegex(KeyError, r"\(none\)"):
        ModelConfig.from_model_name("missing-model")
      self.assertIsNone(get_pyconfig_model_dict("missing-model"))

    base = ModelConfig(model_type="base", vocab_size=32, hidden_size=16, num_hidden_layers=1)
    self.assertIsNone(base.to_pyconfig_dict())

  def test_registry_lookup_overrides_and_type_check(self):
    fake_registry = {
        "stub-dense": _StubDenseConfig,
        "stub-moe": _StubMoeConfig,
    }
    with mock.patch.dict(MODEL_GARDEN_REGISTRY, fake_registry, clear=True):
      cfg = ModelConfig.from_model_name("stub-dense")
      cfg_instruct = ModelConfig.from_model_name("stub-dense-Instruct")
      self.assertIsInstance(cfg, _StubDenseConfig)
      self.assertEqual(cfg, cfg_instruct)

      overridden = _StubDenseConfig.from_model_name("stub-dense", num_hidden_layers=4)
      self.assertEqual(overridden.num_hidden_layers, 4)

      self.assertEqual(get_pyconfig_model_dict("stub-dense"), {"base_emb_dim": 64, "vocab_size": 256})
      self.assertIsNone(get_pyconfig_model_dict("stub-moe"))

      with self.assertRaises(TypeError):
        _StubDenseConfig.from_model_name("stub-moe")
      with self.assertRaisesRegex(KeyError, "stub-dense, stub-moe"):
        ModelConfig.from_model_name("unknown-model")
      with self.assertRaises(ValidationError):
        ModelConfig.from_model_name("stub-dense", nonexistent_field=123)
      with self.assertRaises(ValidationError):
        ModelConfig.from_model_name("stub-dense", hidden_size="not-an-int")

  def test_zero_arg_classmethod_preset_supports_overrides(self):
    class _PresetConfig(ModelConfig):
      model_type: str = "preset"
      vocab_size: int = 8
      hidden_size: int = 8
      num_hidden_layers: int = 2

      @classmethod
      def tiny(cls) -> "_PresetConfig":
        return cls(model_name="tiny")

    with mock.patch.dict(MODEL_GARDEN_REGISTRY, {"tiny": _PresetConfig.tiny}, clear=True):
      cfg = ModelConfig.from_model_name("tiny", num_hidden_layers=4)
      self.assertIsInstance(cfg, _PresetConfig)
      self.assertEqual((cfg.model_name, cfg.num_hidden_layers), ("tiny", 4))
      self.assertEqual(cfg.mesh_divisibility_constraints(), {})

  def test_overrides_recompute_computed_fields_and_keep_preset_values(self):
    """Verifies that computed fields are recomputed on override, while preset-set values survive unrelated overrides."""

    class _DerivedConfig(ModelConfig):
      """Minimal `ModelConfig` subclass with a computed field for testing."""

      model_type: str = "derived"
      vocab_size: int = 8
      hidden_size: int = 64
      num_hidden_layers: int = 2
      num_attention_heads: int = 4

      @computed_field
      @property
      def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

      @classmethod
      def big(cls) -> "_DerivedConfig":
        return cls(model_name="derived-big", hidden_size=256)

    with mock.patch.dict(MODEL_GARDEN_REGISTRY, {"derived": _DerivedConfig.big}, clear=True):
      cfg = ModelConfig.from_model_name("derived", num_attention_heads=8)
      # Preset-set values survive an unrelated override; computed fields are recomputed, not rejected.
      self.assertEqual((cfg.model_name, cfg.hidden_size, cfg.head_dim), ("derived-big", 256, 32))
      self.assertEqual(ModelConfig.from_model_name("derived", hidden_size=128).head_dim, 32)

  def test_frozen_extra_forbid_and_positive_bounds(self):
    cfg = _StubDenseConfig()
    with self.assertRaises(ValidationError):
      cfg.hidden_size = 32  # type: ignore[misc]
    with self.assertRaises(ValidationError):
      ModelConfig(model_type="x", vocab_size=0, hidden_size=16, num_hidden_layers=1)
    with self.assertRaises(ValidationError):
      ModelConfig(model_type="x", vocab_size=16, hidden_size=16, num_hidden_layers=1, extra_key=1)


if __name__ == "__main__":
  absltest.main()
