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
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3 import models as m3_models
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.training_config import TrainingConfig, validate_training_config
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
import numpy as np
from pydantic import ValidationError
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODEL_YAML_DIR = _REPO_ROOT / "src" / "maxtext" / "configs" / "models"


def _mini_qwen3(**overrides: Any) -> Qwen3Config:
  """Returns a tiny Qwen3 config for fast CPU unit tests."""
  defaults: dict[str, Any] = {
      "model_name": "qwen3-0.6b",
      "vocab_size": 256,
      "hidden_size": 64,
      "intermediate_size": 128,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 16,
  }
  defaults.update(overrides)
  return Qwen3Config(**defaults)


def _mini_qwen3_vl(**overrides: Any) -> Qwen3VLConfig:
  """Returns a tiny Qwen3-VL config for fast CPU unit tests."""
  defaults: dict[str, Any] = {
      "model_name": "qwen3-vl-2b",
      "vocab_size": 256,
      "hidden_size": 64,
      "intermediate_size": 128,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 16,
      "mrope_section": (3, 3, 2),
      "vision_image_size": 32,
      "vision_hidden_size": 32,
      "vision_intermediate_size": 64,
      "vision_num_heads": 4,
      "vision_depth": 2,
      "vision_patch_size": 4,
      "vision_out_hidden_size": 64,
      "vision_num_position_embeddings": 16,
      "deepstack_visual_indexes": (0, 1),
  }
  defaults.update(overrides)
  return Qwen3VLConfig(**defaults)


def _mini_glm5(**overrides: Any) -> Glm5ModelConfig:
  """Returns a tiny GLM-5.3-Flash config for fast CPU unit tests."""
  defaults: dict[str, Any] = {
      "vocab_size": 256,
      "hidden_size": 64,
      "intermediate_size": 128,
      "moe_intermediate_size": 32,
      "num_hidden_layers": 4,
      "first_k_dense_replace": 1,
      "linear_num_key_heads": 2,
      "linear_num_value_heads": 2,
      "linear_key_head_dim": 16,
      "linear_value_head_dim": 16,
      "num_attention_heads": 4,
      "num_key_value_heads": 4,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "v_head_dim": 16,
      "index_n_heads": 2,
      "index_head_dim": 16,
      "index_topk": 8,
      "index_kpool": 2,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "num_nextn_predict_layers": 0,
      "kda_implementation": "xla",
  }
  defaults.update(overrides)
  return Glm5ModelConfig(**defaults)


def _mini_granitemoe(**overrides: Any) -> GraniteMoeModelConfig:
  """Returns a tiny GraniteMoE config for fast CPU unit tests."""
  defaults: dict[str, Any] = {
      "model_name": "granite-3.0-1b-a400m",
      "vocab_size": 256,
      "hidden_size": 64,
      "intermediate_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "num_local_experts": 4,
      "num_experts_per_tok": 2,
  }
  defaults.update(overrides)
  return GraniteMoeModelConfig(**defaults)


def _mini_qwen3_8(**overrides: Any) -> Qwen3_8FlashNextConfig:
  """Returns a tiny Qwen3.8-Flash-Next config for fast CPU unit tests."""
  defaults: dict[str, Any] = {
      "model_name": "qwen3.8-flash-next",
      "hidden_size": 64,
      "num_hidden_layers": 4,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "vocab_size": 256,
      "hc_count": 4,
      "hc_lowrank": 16,
      "shared_expert_intermediate_size": 32,
      "moe_intermediate_size": 32,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "linear_key_head_dim": 16,
      "linear_value_head_dim": 16,
      "linear_num_key_heads": 2,
      "linear_num_value_heads": 4,
      "gdn_chunk_size": 8,
      "mrope_section": (2, 1, 1),
      "indexer_n_heads": 2,
      "indexer_kv_heads": 1,
      "indexer_head_dim": 16,
      "indexer_compress_ratio": 4,
      "indexer_budget": 8,
      "ple_embed_dim": 64,
      "heads_per_ngram": 2,
      "ngram_vocab_size_base": 64,
      "make_ngram_vocab_size_divisible_by": 4,
      "split_ngram_parts": 4,
      "eos_token_id": 3,
  }
  defaults.update(overrides)
  return Qwen3_8FlashNextConfig(**defaults)


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
      cfg = ModelConfig.from_model_name(name)
      self.assertIs(type(cfg), expected_cls)
      self.assertEqual(cfg, ModelConfig.from_model_name(f"{name}-Instruct"))
      self.assertIsInstance(hash(cfg), int)
      with self.assertRaises(ValidationError):
        cfg.hidden_size = 16  # type: ignore[misc]
      with self.assertRaises(ValidationError):
        ModelConfig.from_model_name(name, unknown_override=True)

    # Alias resolves to identical default config.
    self.assertEqual(
        ModelConfig.from_model_name("granitemoe"),
        ModelConfig.from_model_name("granite-3.0-1b-a400m"),
    )

    # Inheritance and expected-class type enforcement.
    overridden = Qwen3Config.from_model_name("qwen3-0.6b", num_hidden_layers=4)
    self.assertEqual(overridden.num_hidden_layers, 4)
    self.assertIsInstance(Qwen3Config.from_model_name("qwen3-vl-2b"), Qwen3VLConfig)

    with self.assertRaises(TypeError):
      Qwen3Config.from_model_name("glm5.3-flash")
    with self.assertRaises(TypeError):
      Qwen3VLConfig.from_model_name("qwen3-0.6b")
    with self.assertRaises(TypeError):
      Glm5ModelConfig.from_model_name("granitemoe")
    with self.assertRaisesRegex(KeyError, "glm5.3-flash"):
      ModelConfig.from_model_name("nonexistent-model")

  def test_to_pyconfig_dict_matches_model_yamls(self):
    cases = [
        ("qwen3-0.6b", "qwen3-0.6b.yml"),
        ("qwen3-vl-2b", "qwen3-vl-2b.yml"),
        ("glm5.3-flash", "glm5.3-flash.yml"),
        ("granite-3.0-1b-a400m", "granite-3.0-1b-a400m.yml"),
        ("granitemoe", "granite-3.0-1b-a400m.yml"),
        ("qwen3.8-flash-next", "qwen3.8-flash-next.yml"),
    ]
    for model_name, yaml_name in cases:
      with self.subTest(model=model_name, yaml=yaml_name):
        with open(_MODEL_YAML_DIR / yaml_name, encoding="utf-8") as f:
          expected = yaml.safe_load(f)
        actual = get_pyconfig_model_dict(model_name)
        self.assertEqual(actual, expected)
        assert actual is not None
        for key in expected:
          self.assertIs(type(actual[key]), type(expected[key]), f"Type mismatch for {model_name}.{key}")

  def test_glm5_dataclass_and_hf_json_parity(self):
    full_cfg = Glm5ModelConfig.glm5_3_flash()
    self.assertEqual(full_cfg.to_glm5_config(), Glm5Config())
    self.assertEqual(full_cfg.qk_head_dim, full_cfg.qk_nope_head_dim)

    hf_json_path = _REPO_ROOT / "tests" / "unit" / "m3" / "testdata" / "glm5_3_flash_config.json"
    hf_text_cfg = json.loads(hf_json_path.read_text(encoding="utf-8"))["text_config"]
    from_hf_layers = Glm5ModelConfig(layer_types=tuple(hf_text_cfg["layer_types"]))
    self.assertEqual(from_hf_layers.layer_types, full_cfg.layer_types)
    self.assertEqual(full_cfg.indexer_types, tuple(hf_text_cfg["indexer_types"]))

  def test_granitemoe_dataclass_and_from_config_parity(self):
    full_cfg = GraniteMoeModelConfig.granite_3_0_1b_a400m()
    self.assertEqual(full_cfg.to_granitemoe_config(), GraniteMoeConfig())

    mini = _mini_granitemoe()
    self.assertEqual(GraniteMoeConfig.from_config(mini), mini.to_granitemoe_config())

  def test_qwen3_and_qwen3_vl_validators(self):
    with self.assertRaisesRegex(ValidationError, "divisible"):
      _mini_qwen3(num_attention_heads=5, num_key_value_heads=2)
    with self.assertRaisesRegex(ValidationError, "even for RoPE"):
      _mini_qwen3(head_dim=15)
    with self.assertRaises(ValidationError):
      _mini_qwen3(final_logits_soft_cap=0.0)

    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _mini_qwen3_vl(mrope_section=(4, 4, 4))
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _mini_qwen3_vl(mrope_section=(-1, 5, 4))
    with self.assertRaisesRegex(ValidationError, "vision_hidden_size"):
      _mini_qwen3_vl(vision_hidden_size=30, vision_num_heads=4)
    with self.assertRaisesRegex(ValidationError, "perfect square"):
      _mini_qwen3_vl(vision_num_position_embeddings=15)
    with self.assertRaisesRegex(ValidationError, "deepstack_visual_indexes"):
      _mini_qwen3_vl(deepstack_visual_indexes=(0, 5))
    with self.assertRaisesRegex(ValidationError, "vision_out_hidden_size"):
      _mini_qwen3_vl(vision_out_hidden_size=32)

  def test_glm5_granitemoe_and_qwen3_8_validators(self):
    with self.assertRaisesRegex(ValidationError, "layer_types"):
      _mini_glm5(layer_types=("linear_attention",))
    with self.assertRaisesRegex(ValidationError, "layer_types"):
      _mini_glm5(num_hidden_layers=1, layer_types=("bad_type",))
    with self.assertRaisesRegex(ValidationError, "indexer_types"):
      _mini_glm5(indexer_types=("full",))
    with self.assertRaisesRegex(ValidationError, "indexer_types"):
      _mini_glm5(num_hidden_layers=1, indexer_types=("sparse",))
    with self.assertRaisesRegex(ValidationError, "qk_rope_head_dim"):
      _mini_glm5(qk_rope_head_dim=64)
    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _mini_glm5(num_attention_heads=8, num_key_value_heads=4)
    with self.assertRaisesRegex(ValidationError, "linear_num_key_heads"):
      _mini_glm5(linear_num_key_heads=4, linear_num_value_heads=8)
    with self.assertRaisesRegex(ValidationError, "linear_key_head_dim"):
      _mini_glm5(linear_key_head_dim=64, linear_value_head_dim=128)
    with self.assertRaisesRegex(ValidationError, "index_kpool"):
      _mini_glm5(index_topk=10, index_kpool=4)
    with self.assertRaisesRegex(ValidationError, "linear_lower_bound"):
      _mini_glm5(kda_implementation="mosaic", linear_lower_bound=None)
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _mini_glm5(n_routed_experts=4, num_experts_per_tok=8)

    with self.assertRaisesRegex(ValidationError, "hidden_size"):
      _mini_granitemoe(hidden_size=62, num_attention_heads=4)
    with self.assertRaisesRegex(ValidationError, "head_dim"):
      _mini_granitemoe(hidden_size=12, num_attention_heads=4)
    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _mini_granitemoe(hidden_size=48, num_attention_heads=6, num_key_value_heads=4)
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _mini_granitemoe(num_local_experts=4, num_experts_per_tok=5)
    with self.assertRaisesRegex(ValidationError, "logits_scaling"):
      _mini_granitemoe(logits_scaling=0.0)

    with self.assertRaisesRegex(ValidationError, "num_key_value_heads"):
      _mini_qwen3_8(num_attention_heads=5, num_key_value_heads=2)
    with self.assertRaisesRegex(ValidationError, "full_attention_interval"):
      _mini_qwen3_8(num_hidden_layers=5, full_attention_interval=4)
    with self.assertRaisesRegex(ValidationError, "even integer"):
      _mini_qwen3_8(head_dim=64, partial_rotary_factor=0.3)
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _mini_qwen3_8(mrope_section=(4, 4, 4))
    with self.assertRaisesRegex(ValidationError, "mrope_section"):
      _mini_qwen3_8(mrope_section=(-1, 2, 3))
    with self.assertRaisesRegex(ValidationError, "num_experts_per_tok"):
      _mini_qwen3_8(n_routed_experts=4, num_experts_per_tok=6)
    with self.assertRaisesRegex(ValidationError, "linear_num_value_heads"):
      _mini_qwen3_8(linear_num_key_heads=4, linear_num_value_heads=6)
    with self.assertRaisesRegex(ValidationError, "indexer_kv_heads"):
      _mini_qwen3_8(indexer_n_heads=3, indexer_kv_heads=2)
    with self.assertRaisesRegex(ValidationError, "ple_layer_id"):
      _mini_qwen3_8(ple_layer_id=4)
    with self.assertRaisesRegex(ValidationError, "eos_token_id"):
      _mini_qwen3_8(vocab_size=256, eos_token_id=256)

  def test_cross_config_validation_with_model_configs(self):
    train_cfg = TrainingConfig(per_device_batch_size=1.0, max_target_length=16, num_vocab_tiling=4)
    for model_cfg in (
        _mini_qwen3(),
        _mini_qwen3_vl(),
        _mini_glm5(),
        _mini_granitemoe(),
        _mini_qwen3_8(),
    ):
      validate_training_config(model_cfg, train_cfg, mesh_shape={"data": 1, "fsdp": 1, "tensor": 2, "expert": 2})

    with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
      validate_training_config(_mini_qwen3(), train_cfg, mesh_shape={"tensor": 4})
    with self.assertRaisesRegex(ValueError, "num_local_experts"):
      validate_training_config(_mini_granitemoe(), train_cfg, mesh_shape={"expert": 3})
    with self.assertRaisesRegex(ValueError, "n_routed_experts"):
      validate_training_config(_mini_glm5(), train_cfg, mesh_shape={"expert": 3})
    with self.assertRaisesRegex(ValueError, "n_routed_experts"):
      validate_training_config(_mini_qwen3_8(), train_cfg, mesh_shape={"expert": 3})

  def test_live_mini_model_construction(self):
    devices = np.array(jax.devices())
    num_devices = devices.size
    mesh_4 = jax.sharding.Mesh(devices.reshape((1, num_devices, 1, 1)), ("data", "fsdp", "tensor", "expert"))
    batch = max(2, num_devices)
    tokens = jnp.ones((batch, 8), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(8)[None, :], (batch, 8))

    gm = m3_models.create_model(_mini_granitemoe(), mesh_4, rngs=nnx.Rngs(0))
    gm_logits = gm(tokens, positions)
    self.assertEqual(gm_logits.shape, (batch, 8, 256))

    glm_cfg = _mini_glm5().to_glm5_config(scan_layers=False, remat_policy="none")
    glm = m3_models.create_model(glm_cfg, mesh_4, rngs=nnx.Rngs(0))
    self.assertEqual(type(glm).__name__, "Glm5Model")


if __name__ == "__main__":
  absltest.main()
