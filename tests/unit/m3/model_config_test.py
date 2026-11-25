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

"""Unit tests for `ModelConfig`, concrete m3 model family configs, and `MODEL_GARDEN_REGISTRY`."""

import json
import pathlib
from typing import Any
from unittest import mock

from absl.testing import absltest
from maxtext.configs.types import MaxTextConfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.training_config import TrainingConfig, TransformConfig, validate_training_config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5ModelConfig
from maxtext.m3.models.granitemoe.granitemoe_config import GraniteMoeModelConfig
from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeConfig
from maxtext.m3.models.model_config import ModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.models.qwen3_8_flash_next.qwen3_8_config import Qwen3_8FlashNextConfig
from maxtext.m3.models.qwen3_vl.qwen3_vl_config import Qwen3VLConfig
from maxtext.m3.models.registry import (
    MODEL_GARDEN_REGISTRY,
    get_pyconfig_model_dict,
    normalize_model_name,
)
from pydantic import ValidationError, computed_field
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODEL_YAML_DIR = _REPO_ROOT / "src" / "maxtext" / "configs" / "models"


def _with(preset: ModelConfig, **overrides: Any) -> ModelConfig:
  """Re-validates a registered preset with `overrides` (e.g. one invalid value for a validator test)."""
  return type(preset).model_validate({**preset.model_dump(), **overrides})


class _StubDenseConfig(ModelConfig):
  """Minimal dense `ModelConfig` subclass for isolated registry tests."""

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
  """Tests `ModelConfig` invariants, registry parity, YAML parity, and live model wiring."""

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

  def test_registry_matches_m3_model_registry(self):
    self.assertEqual(
        set(MODEL_GARDEN_REGISTRY),
        # `qwen3.8-flash-next-mini` is a CPU-test-only fixture in MODEL_REGISTRY, not a real model, so it has no preset.
        set(m3_models.MODEL_REGISTRY) - {"qwen3.8-flash-next-mini"},
    )

    expected_types: dict[str, type[ModelConfig]] = {
        "glm5.3-flash": Glm5ModelConfig,
        "granite-3.0-1b-a400m": GraniteMoeModelConfig,
        "granitemoe": GraniteMoeModelConfig,
        "qwen3-0.6b": Qwen3Config,
        "qwen3-vl-2b": Qwen3VLConfig,
        "qwen3.8-flash-next": Qwen3_8FlashNextConfig,
    }
    for name, expected_cls in expected_types.items():
      self.assertIs(type(ModelConfig.from_model_name(name)), expected_cls)

    # Alias resolves to identical default config.
    self.assertEqual(
        ModelConfig.from_model_name("granitemoe"),
        ModelConfig.from_model_name("granite-3.0-1b-a400m"),
    )
    # Qwen3-VL subclasses Qwen3, so the parent class accepts it but not the reverse.
    self.assertIsInstance(Qwen3Config.from_model_name("qwen3-vl-2b"), Qwen3VLConfig)
    with self.assertRaises(TypeError):
      Qwen3VLConfig.from_model_name("qwen3-0.6b")

  def test_to_pyconfig_dict_matches_model_yamls(self):
    cases = [
        ("qwen3-0.6b", "qwen3-0.6b.yml"),
        ("qwen3-vl-2b", "qwen3-vl-2b.yml"),
    ]
    with open(_MODEL_YAML_DIR.parent / "base.yml", encoding="utf-8") as f:
      base_yml = yaml.safe_load(f)
    for model_name, yaml_name in cases:
      with self.subTest(model=model_name, yaml=yaml_name):
        with open(_MODEL_YAML_DIR / yaml_name, encoding="utf-8") as f:
          expected = yaml.safe_load(f)
        actual = get_pyconfig_model_dict(model_name)
        assert actual is not None
        # Every YAML key matches; extra keys (overridable through the bridge) must equal the `base.yml` default.
        self.assertEqual({k: actual.get(k) for k in expected}, expected)
        for key in actual.keys() - expected.keys():
          default = base_yml[key] if key in base_yml else MaxTextConfig.model_fields[key].default
          self.assertEqual(actual[key], default, f"{model_name}.{key} differs from the base.yml default")

    # m3-only families have no model YAML; `pyconfig` reads their presets through the registry.
    for model_name in ("glm5.3-flash", "granite-3.0-1b-a400m", "qwen3.8-flash-next"):
      with self.subTest(yaml_free_model=model_name):
        self.assertFalse((_MODEL_YAML_DIR / f"{model_name}.yml").exists())
        self.assertTrue(get_pyconfig_model_dict(model_name)["use_m3_model"])
    self.assertEqual(get_pyconfig_model_dict("granitemoe"), get_pyconfig_model_dict("granite-3.0-1b-a400m"))

  def test_glm5_dataclass_and_hf_json_parity(self):
    full_cfg = Glm5ModelConfig.glm5_3_flash()
    self.assertEqual(full_cfg.to_glm5_config(), Glm5Config())
    self.assertEqual(full_cfg.qk_head_dim, full_cfg.qk_nope_head_dim)

    hf_json_path = _REPO_ROOT / "tests" / "unit" / "m3" / "testdata" / "glm5_3_flash_config.json"
    hf_text_cfg = json.loads(hf_json_path.read_text(encoding="utf-8"))["text_config"]
    from_hf_layers = Glm5ModelConfig(**{**full_cfg.model_dump(), "layer_types": tuple(hf_text_cfg["layer_types"])})
    self.assertEqual(from_hf_layers.resolved_layer_types, full_cfg.resolved_layer_types)
    self.assertEqual(full_cfg.resolved_indexer_types, tuple(hf_text_cfg["indexer_types"]))

    # Raw schedule inputs stay unset (no post-validation mutation), so dump round trips can change the depth.
    self.assertIsNone(full_cfg.layer_types)
    resized = Glm5ModelConfig(**{**full_cfg.model_dump(), "num_hidden_layers": 8})
    self.assertEqual(len(resized.resolved_layer_types), 8)
    self.assertEqual(len(resized.resolved_indexer_types), 8)

  def test_granitemoe_dataclass_and_from_config_parity(self):
    full_cfg = GraniteMoeModelConfig.granite_3_0_1b_a400m()
    self.assertEqual(full_cfg.to_granitemoe_config(), GraniteMoeConfig())

    self.assertEqual(GraniteMoeConfig.from_config(full_cfg), full_cfg.to_granitemoe_config())

  def test_converters_take_runtime_settings_from_training_config(self):
    glm5 = Glm5ModelConfig.glm5_3_flash()
    granite = GraniteMoeModelConfig.granite_3_0_1b_a400m()
    train_cfg = TrainingConfig(
        num_vocab_tiling=4,
        matmul_precision="highest",
        transforms=TransformConfig(scan_layers=False, remat_policy="none"),
    )
    glm = glm5.to_glm5_config(train_cfg)
    self.assertEqual(
        (glm.scan_layers, glm.remat_policy, glm.num_vocab_tiling, glm.routed_bias_update_rate), (False, "none", 4, 0.0)
    )
    # The router-bias step size now flows from TrainingConfig; GLM-5 rejects non-zero values until FU-D4 lands.
    with self.assertRaisesRegex(ValueError, "FU-D4"):
      glm5.to_glm5_config(TrainingConfig(routed_bias_update_rate=0.01))
    granite_dc = granite.to_granitemoe_config(train_cfg)
    self.assertEqual(
        (granite_dc.scan_layers, granite_dc.remat_policy, granite_dc.matmul_precision), (False, "none", "highest")
    )

    # Without a TrainingConfig, each dataclass keeps its own runtime defaults.
    self.assertEqual(glm5.to_glm5_config().remat_policy, Glm5Config.remat_policy)
    self.assertEqual(granite.to_granitemoe_config().remat_policy, GraniteMoeConfig.remat_policy)

    # Remat policies a family does not implement fail fast at conversion time.
    minimal = TrainingConfig(transforms=TransformConfig(remat_policy="minimal"))
    with self.assertRaisesRegex(ValueError, "GLM-5.3-Flash supports remat_policy"):
      glm5.to_glm5_config(minimal)
    self.assertEqual(granite.to_granitemoe_config(minimal).remat_policy, "minimal")
    with self.assertRaisesRegex(ValueError, "GraniteMoE supports remat_policy"):
      granite.to_granitemoe_config(TrainingConfig(transforms=TransformConfig(remat_policy="save_qkv_proj")))

  def test_supported_mesh_axes_match_model_registry(self):
    # `ModelConfig.supported_mesh_axes()` must agree with the per-model mesh axes model creation enforces.
    for name in MODEL_GARDEN_REGISTRY:
      with self.subTest(model=name):
        self.assertEqual(
            ModelConfig.from_model_name(name).supported_mesh_axes(), frozenset(m3_models.MODEL_SUPPORTED_MESH_AXES[name])
        )

  def test_qwen3_and_qwen3_vl_validators(self):
    with self.assertRaisesRegex(ValidationError, "divisible"):
      _with(Qwen3Config.qwen3_0_6b(), num_attention_heads=5, num_key_value_heads=2)
    with self.assertRaisesRegex(ValidationError, "even for RoPE"):
      _with(Qwen3Config.qwen3_0_6b(), head_dim=15)
    with self.assertRaises(ValidationError):
      _with(Qwen3Config.qwen3_0_6b(), final_logits_soft_cap=0.0)

    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), mrope_section=(4, 4, 4))
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), mrope_section=(-1, 5, 4))
    with self.assertRaisesRegex(ValidationError, "vision_hidden_size"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), vision_hidden_size=30, vision_num_heads=4)
    with self.assertRaisesRegex(ValidationError, "perfect square"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), vision_num_position_embeddings=15)
    with self.assertRaisesRegex(ValidationError, "deepstack_visual_indexes"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), deepstack_visual_indexes=(0, 24))
    with self.assertRaisesRegex(ValidationError, "vision_out_hidden_size"):
      _with(Qwen3VLConfig.qwen3_vl_2b(), vision_out_hidden_size=32)

  def test_glm5_granitemoe_and_qwen3_8_validators(self):
    with self.assertRaisesRegex(ValidationError, "layer_types"):
      _with(Glm5ModelConfig.glm5_3_flash(), layer_types=("linear_attention",))
    with self.assertRaisesRegex(ValidationError, "layer_types"):
      _with(Glm5ModelConfig.glm5_3_flash(), num_hidden_layers=1, layer_types=("bad_type",))
    with self.assertRaisesRegex(ValidationError, "indexer_types"):
      _with(Glm5ModelConfig.glm5_3_flash(), indexer_types=("full",))
    with self.assertRaisesRegex(ValidationError, "indexer_types"):
      _with(Glm5ModelConfig.glm5_3_flash(), num_hidden_layers=1, indexer_types=("sparse",))
    with self.assertRaisesRegex(ValidationError, "qk_rope_head_dim"):
      _with(Glm5ModelConfig.glm5_3_flash(), qk_rope_head_dim=64)
    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _with(Glm5ModelConfig.glm5_3_flash(), num_attention_heads=8, num_key_value_heads=4)
    with self.assertRaisesRegex(ValidationError, "linear_num_key_heads"):
      _with(Glm5ModelConfig.glm5_3_flash(), linear_num_key_heads=4, linear_num_value_heads=8)
    with self.assertRaisesRegex(ValidationError, "linear_key_head_dim"):
      _with(Glm5ModelConfig.glm5_3_flash(), linear_key_head_dim=64, linear_value_head_dim=128)
    with self.assertRaisesRegex(ValidationError, "index_kpool"):
      _with(Glm5ModelConfig.glm5_3_flash(), index_topk=10, index_kpool=4)
    with self.assertRaisesRegex(ValidationError, "linear_lower_bound"):
      _with(Glm5ModelConfig.glm5_3_flash(), kda_implementation="mosaic", linear_lower_bound=None)
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _with(Glm5ModelConfig.glm5_3_flash(), n_routed_experts=4, num_experts_per_tok=8)

    with self.assertRaisesRegex(ValidationError, "hidden_size"):
      _with(GraniteMoeModelConfig.granite_3_0_1b_a400m(), hidden_size=62, num_attention_heads=4)
    with self.assertRaisesRegex(ValidationError, "head_dim"):
      _with(GraniteMoeModelConfig.granite_3_0_1b_a400m(), hidden_size=12, num_attention_heads=4)
    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _with(GraniteMoeModelConfig.granite_3_0_1b_a400m(), hidden_size=48, num_attention_heads=6, num_key_value_heads=4)
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _with(GraniteMoeModelConfig.granite_3_0_1b_a400m(), num_local_experts=4, num_experts_per_tok=5)
    with self.assertRaisesRegex(ValidationError, "logits_scaling"):
      _with(GraniteMoeModelConfig.granite_3_0_1b_a400m(), logits_scaling=0.0)

    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), num_attention_heads=5, num_key_value_heads=2)
    with self.assertRaisesRegex(ValidationError, "full_attention_interval"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), num_hidden_layers=5, full_attention_interval=4)
    with self.assertRaisesRegex(ValidationError, "even integer"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), head_dim=64, partial_rotary_factor=0.3)
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), mrope_section=(4, 4, 4))
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), mrope_section=(-1, 2, 3))
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), n_routed_experts=4, num_experts_per_tok=6)
    with self.assertRaisesRegex(ValidationError, "linear_num_value_heads"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), linear_num_key_heads=4, linear_num_value_heads=6)
    with self.assertRaisesRegex(ValidationError, "indexer_kv_heads"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), indexer_n_heads=3, indexer_kv_heads=2)
    with self.assertRaisesRegex(ValidationError, "ple_layer_id"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), ple_layer_id=48)
    with self.assertRaisesRegex(ValidationError, "eos_token_id"):
      _with(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), vocab_size=256, eos_token_id=256)

  def test_cross_config_validation_with_model_configs(self):
    train_cfg = TrainingConfig(per_device_batch_size=1.0, max_target_length=16, num_vocab_tiling=4)
    for model_cfg in (
        Qwen3Config.qwen3_0_6b(),
        Qwen3VLConfig.qwen3_vl_2b(),
        Glm5ModelConfig.glm5_3_flash(),
        GraniteMoeModelConfig.granite_3_0_1b_a400m(),
        Qwen3_8FlashNextConfig.qwen3_8_flash_next(),
    ):
      validate_training_config(model_cfg, train_cfg, mesh_shape={"data": 1, "fsdp": 1, "tensor": 2, "expert": 2})

    with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
      validate_training_config(Qwen3Config.qwen3_0_6b(), train_cfg, mesh_shape={"tensor": 16})  # 8 KV heads
    with self.assertRaisesRegex(ValueError, "num_local_experts"):
      validate_training_config(GraniteMoeModelConfig.granite_3_0_1b_a400m(), train_cfg, mesh_shape={"expert": 3})
    with self.assertRaisesRegex(ValueError, "n_routed_experts"):
      validate_training_config(Glm5ModelConfig.glm5_3_flash(), train_cfg, mesh_shape={"expert": 5})  # 288 experts
    with self.assertRaisesRegex(ValueError, "n_routed_experts"):
      validate_training_config(Qwen3_8FlashNextConfig.qwen3_8_flash_next(), train_cfg, mesh_shape={"expert": 3})


if __name__ == "__main__":
  absltest.main()
