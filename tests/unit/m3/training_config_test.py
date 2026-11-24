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

"""Unit tests for `TrainingConfig` schemas, validators, and `base.yml` parity."""

import json
import pathlib

from absl.testing import absltest
from maxtext.configs import types as config_types
from maxtext.m3.configs.training_config import (
    CheckpointConfig,
    DataPipelineConfig,
    MeshConfig,
    OptimizerConfig,
    TrainingConfig,
    TransformConfig,
    validate_training_config,
)
from maxtext.m3.models.model_config import ModelConfig
from pydantic import ValidationError
import yaml


class _StubDenseConfig(ModelConfig):
  """Minimal dense `ModelConfig` subclass for cross-config divisibility tests."""

  model_type: str = "stub_dense"
  vocab_size: int = 256
  hidden_size: int = 64
  num_hidden_layers: int = 2
  num_attention_heads: int = 4
  num_key_value_heads: int = 2


class _StubMoeConfig(ModelConfig):
  """Minimal MoE `ModelConfig` subclass for EP divisibility tests."""

  model_type: str = "stub_moe"
  vocab_size: int = 256
  hidden_size: int = 64
  num_hidden_layers: int = 2
  n_routed_experts: int = 8


class TrainingConfigTest(absltest.TestCase):
  """Tests `TrainingConfig` sub-configs, validation, serialization, and `base.yml` parity."""

  def test_mesh_config_rejects_context_and_invalid_sizes(self):
    cfg = MeshConfig(ici={"fsdp": -1, "tensor": 2}, dcn={"data": -1})
    self.assertEqual(cfg.ici, {"fsdp": -1, "tensor": 2})

    with self.assertRaisesRegex(ValidationError, "unsupported mesh axis.*context"):
      MeshConfig(ici={"context": 2, "fsdp": -1})
    with self.assertRaisesRegex(ValidationError, "at most one -1"):
      MeshConfig(ici={"data": -1, "fsdp": -1})
    with self.assertRaisesRegex(ValidationError, "positive integer"):
      MeshConfig(ici={"fsdp": 0})
    with self.assertRaisesRegex(ValidationError, "positive integer"):
      MeshConfig(ici={"fsdp": True})

  def test_checkpoint_and_optimizer_validators(self):
    with self.assertRaisesRegex(ValidationError, "At most one"):
      CheckpointConfig(load_full_state_path="gs://a", load_parameters_path="gs://b")
    with self.assertRaisesRegex(ValidationError, "safetensors_dynamic"):
      CheckpointConfig(load_full_state_path="gs://a", source_checkpoint_layout="safetensors_dynamic")
    opt = OptimizerConfig(optimizer_type=config_types.OptimizerType.ADAMW)
    self.assertEqual(opt.optimizer_type, "adamw")

  def test_scoped_subconfigs_and_extra_forbid(self):
    cfg = TrainingConfig(
        base_output_directory="gs://bucket/run",
        steps=10,
        max_target_length=128,
        num_vocab_tiling=4,
        transforms=TransformConfig(scan_layers=False),
        optimizer=OptimizerConfig(learning_rate=1e-3),
        checkpoint=CheckpointConfig(enable_checkpointing=False),
        profiling=config_types.Profiling(profiler="xplane"),
    )
    self.assertEqual(cfg.base_output_directory, "gs://bucket/run")
    self.assertEqual(cfg.steps, 10)
    self.assertEqual(cfg.max_target_length, 128)
    self.assertEqual(cfg.num_vocab_tiling, 4)
    self.assertFalse(cfg.transforms.scan_layers)
    self.assertEqual(cfg.optimizer.learning_rate, 1e-3)
    self.assertFalse(cfg.checkpoint.enable_checkpointing)
    self.assertEqual(cfg.profiling.profiler, config_types.ProfilerType.XPLANE)

    # Extra/unknown and un-scoped top-level keys are rejected via `extra='forbid'`.
    with self.assertRaises(ValidationError):
      TrainingConfig(nonexistent_knob=True)
    with self.assertRaises(ValidationError):
      TrainingConfig(scan_layers=False)

  def test_secret_str_redacted_in_json(self):
    cfg = TrainingConfig(data=DataPipelineConfig(hf_access_token="hf_super_secret_token"))
    serialized = cfg.to_json()
    self.assertNotIn("hf_super_secret_token", serialized)
    payload = json.loads(serialized)
    self.assertEqual(payload["data"]["hf_access_token"], "**********")

  def test_cross_config_validation(self):
    dense = _StubDenseConfig()
    valid_train = TrainingConfig(per_device_batch_size=1.0, max_target_length=16, num_vocab_tiling=4)
    validate_training_config(dense, valid_train, mesh_shape={"data": 1, "fsdp": 2, "tensor": 2, "expert": 1})

    frac_train = TrainingConfig(per_device_batch_size=0.5, max_target_length=16, num_vocab_tiling=4)
    validate_training_config(dense, frac_train)

    bad_tiling = TrainingConfig(per_device_batch_size=1.0, max_target_length=10, num_vocab_tiling=4)
    with self.assertRaisesRegex(ValueError, "num_vocab_tiling"):
      validate_training_config(dense, bad_tiling)

    with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
      validate_training_config(dense, valid_train, mesh_shape={"tensor": 4})

    moe = _StubMoeConfig()
    with self.assertRaisesRegex(ValueError, "n_routed_experts"):
      validate_training_config(moe, valid_train, mesh_shape={"expert": 3})

  def test_defaults_match_base_yml(self):
    base_yml_path = pathlib.Path(__file__).resolve().parents[3] / "src" / "maxtext" / "configs" / "base.yml"
    with open(base_yml_path, encoding="utf-8") as f:
      base = yaml.safe_load(f)

    renames = {
        "seed": "init_weights_seed",
        "router_bias_update_rate": "routed_bias_update_rate",
        "param_dtype": "weight_dtype",
        "compute_dtype": "dtype",
        "optimizer_type": "opt_type",
        "cosine_learning_rate_final_fraction": "learning_rate_final_fraction",
        "freeze_vision_encoder": "freeze_vision_encoder_params",
        "todelete_full_path": "checkpoint_todelete_full_path",
    }
    tc = TrainingConfig()
    sections = [
        tc,
        tc.mesh,
        tc.transforms,
        tc.optimizer,
        tc.checkpoint,
        tc.metrics,
        tc.data,
        tc.aot,
        tc.orbax_storage,
        tc.profiling,
    ]
    for obj in sections:
      for name in type(obj).model_fields:
        val = getattr(obj, name)
        if hasattr(type(val), "model_fields"):
          continue
        base_key = renames.get(name, name)
        if base_key not in base:
          continue
        actual = getattr(val, "value", val)
        expected = base[base_key]
        if isinstance(actual, tuple) and isinstance(expected, list):
          actual = list(actual)
        # Match resolved semantics for YAML `"None"` / empty `""` optional sentinels.
        if actual is None and expected in ("None", ""):
          continue
        if name == "mu_dtype" and expected == "":
          self.assertEqual(actual, base["weight_dtype"])
          continue
        self.assertEqual(actual, expected, f"Default mismatch for {name!r} (base.yml key {base_key!r})")

    self.assertEqual(tc.mesh.ici, {"fsdp": base["ici_fsdp_parallelism"]})
    self.assertEqual(tc.mesh.dcn, {"data": base["dcn_data_parallelism"]})


if __name__ == "__main__":
  absltest.main()
