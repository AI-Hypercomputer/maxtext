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
import os

from absl.testing import absltest
from maxtext.configs import types as config_types
from maxtext.m3.configs.training_config import (
    MESH_AXES,
    CheckpointConfig,
    DataPipelineConfig,
    MeshAxes,
    MeshConfig,
    OptimizerConfig,
    ProfilingConfig,
    TrainingConfig,
    TransformConfig,
    validate_training_config,
)
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.model_config import ModelConfig
from maxtext.utils import globals as maxtext_globals
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

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    return {"tensor": {"num_attention_heads": self.num_attention_heads, "num_key_value_heads": self.num_key_value_heads}}


class _StubMoeConfig(ModelConfig):
  """Minimal MoE `ModelConfig` subclass for EP divisibility tests."""

  model_type: str = "stub_moe"
  vocab_size: int = 256
  hidden_size: int = 64
  num_hidden_layers: int = 2
  num_experts: int = 8

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    return {"expert": {"num_experts": self.num_experts}}


class TrainingConfigTest(absltest.TestCase):
  """Tests `TrainingConfig` sub-configs, validation, serialization, and `base.yml` parity."""

  def test_mesh_config_rejects_context_and_invalid_sizes(self):
    cfg = MeshConfig(ici={"fsdp": -1, "tensor": 2}, dcn={"data": -1})
    self.assertEqual(
        cfg.ici.model_dump(),
        {"data": 1, "fsdp": -1, "fsdp_transpose": 1, "context": 1, "tensor": 2, "tensor_sequence": 1, "expert": 1},
    )

    # Axes m3 does not shard over are rejected; `stage` is held at 1 until a pipeline transform exists.
    for axis in ("stage", "sequence", "pipeline"):
      with self.assertRaisesRegex(ValidationError, rf"{axis}\s+Extra inputs are not permitted"):
        MeshConfig(ici={axis: 2})
    with self.assertRaisesRegex(ValidationError, "at most one -1"):
      MeshConfig(ici={"data": -1, "fsdp": -1})
    with self.assertRaisesRegex(ValidationError, "positive integer"):
      MeshConfig(ici={"fsdp": 0})
    with self.assertRaisesRegex(ValidationError, "valid integer"):
      MeshConfig(ici={"fsdp": True})

  def test_mesh_matches_m3_sharding_axes_and_layout_flags(self):
    # Must stay in sync with the axes `m3/core/sharding.py` builds and shards over.
    self.assertEqual(tuple(MeshAxes.model_fields), m3_sharding.DEFAULT_SHARDING_AXES)
    self.assertEqual(MESH_AXES, m3_sharding.DEFAULT_SHARDING_AXES)
    self.assertEqual(frozenset(MESH_AXES), ModelConfig.supported_mesh_axes(_StubDenseConfig()))

    cfg = MeshConfig(
        ici={"fsdp": 4, "fsdp_transpose": 2, "context": 2, "tensor_sequence": 2},
        custom_mesh="hybrid_ring_64x4",
        allow_split_physical_axes=True,
        optimize_mesh_for_tpu_v6e=True,
    )
    self.assertEqual((cfg.ici.fsdp_transpose, cfg.ici.context, cfg.ici.tensor_sequence), (2, 2, 2))
    self.assertEqual(cfg.custom_mesh, "hybrid_ring_64x4")
    with self.assertRaises(ValidationError):
      MeshConfig(custom_mesh="hybrid_ring_16x16")

  def test_families_reject_mesh_axes_they_do_not_support(self):
    class _FourAxisConfig(_StubDenseConfig):

      def supported_mesh_axes(self) -> frozenset[str]:
        return frozenset({"data", "fsdp", "tensor", "expert"})

    train = TrainingConfig(mesh=MeshConfig(ici={"fsdp": -1, "context": 2}))
    validate_training_config(_StubDenseConfig(), train)  # Base default supports every m3 axis.
    with self.assertRaisesRegex(ValueError, r"does not support mesh axes \{'ici.context': 2\}"):
      validate_training_config(_FourAxisConfig(), train)
    # An unsupported `-1` auto-fill axis is active too.
    auto = TrainingConfig(mesh=MeshConfig(ici={"fsdp": 1, "tensor_sequence": -1}))
    with self.assertRaisesRegex(ValueError, "ici.tensor_sequence"):
      validate_training_config(_FourAxisConfig(), auto)
    validate_training_config(_FourAxisConfig(), TrainingConfig())  # Defaults (ici.fsdp=-1, dcn.data=-1) are fine.

  def test_partial_mesh_dicts_keep_base_yml_auto_axes(self):
    cfg = MeshConfig(ici={"tensor": 4}, dcn={"fsdp": 2})
    self.assertEqual(cfg.ici.model_dump()["fsdp"], -1)
    self.assertEqual(cfg.ici.model_dump()["tensor"], 4)
    self.assertEqual(cfg.dcn.model_dump()["data"], -1)
    self.assertEqual(cfg.dcn.model_dump()["fsdp"], 2)
    # Moving the auto axis requires pinning the default one, matching `base.yml`'s single -1 rule.
    with self.assertRaisesRegex(ValidationError, "at most one -1"):
      MeshConfig(ici={"data": -1})
    self.assertEqual(MeshConfig(ici={"data": -1, "fsdp": 1}).ici.data, -1)

  def test_training_config_is_deeply_frozen_and_hashable(self):
    cfg = TrainingConfig()
    self.assertEqual(hash(cfg), hash(TrainingConfig()))
    with self.assertRaises(ValidationError):
      cfg.mesh.ici.fsdp = 8  # type: ignore[misc]
    with self.assertRaises(ValidationError):
      cfg.aot.compile_topology = "v5p-8"  # type: ignore[misc]
    # Typos inside reused `types.py` leaf configs fail fast instead of being silently ignored.
    with self.assertRaises(ValidationError):
      TrainingConfig(aot={"compile_topologyy": "v5p-8"})

  def test_accepts_base_yml_sentinel_values(self):
    # `None` replaces base.yml's -1 sentinel for step horizons.
    default = TrainingConfig()
    self.assertEqual(default.steps, 150001)
    self.assertIsNone(default.eval_interval)
    self.assertIsNone(default.eval_steps)
    self.assertIsNone(default.optimizer.learning_rate_schedule_steps)
    inherit = TrainingConfig(steps=None, optimizer=OptimizerConfig(learning_rate_schedule_steps=10))
    self.assertIsNone(inherit.steps)
    self.assertEqual(inherit.optimizer.learning_rate_schedule_steps, 10)
    with self.assertRaisesRegex(ValidationError, "cannot both be None"):
      TrainingConfig(steps=None)
    # Only `None` or `> 0` are valid: `0` and the old `-1` sentinel are rejected (the CLI bridge maps -1 -> None).
    for bad in (0, -1):
      for case in (
          lambda v: TrainingConfig(steps=v),
          lambda v: TrainingConfig(eval_interval=v),
          lambda v: TrainingConfig(eval_steps=v),
          lambda v: OptimizerConfig(learning_rate_schedule_steps=v),
      ):
        with self.assertRaisesRegex(ValidationError, "greater than 0"):
          case(bad)
    cfg = TrainingConfig(eval_interval=100, eval_steps=10, optimizer=OptimizerConfig(learning_rate_schedule_steps=5))
    self.assertEqual((cfg.eval_interval, cfg.eval_steps, cfg.optimizer.learning_rate_schedule_steps), (100, 10, 5))
    data = DataPipelineConfig(grain_worker_count=-1, grain_packing_type="concat_then_split")
    self.assertEqual(data.grain_worker_count, -1)
    self.assertEqual(OptimizerConfig().mu_dtype, "")

  def test_optimizer_reuses_adamw_and_supports_wsd(self):
    opt = OptimizerConfig(mu_dtype=config_types.DType.BFLOAT16, adamw_mask=["bias", ".*norm"])
    self.assertIsInstance(opt, config_types.AdamW)
    self.assertEqual(opt.mu_dtype, "bfloat16")
    self.assertEqual(opt.adamw_mask, ("bias", ".*norm"))
    hash(opt)  # tuple `adamw_mask` keeps the frozen config hashable.
    with self.assertRaisesRegex(ValidationError, "mu_dtype"):
      OptimizerConfig(mu_dtype="float64")
    with self.assertRaisesRegex(ValidationError, r"\[0, 1\)"):
      OptimizerConfig(adam_b1=1.0)
    with self.assertRaisesRegex(ValidationError, "adam_eps"):
      OptimizerConfig(adam_eps=0.0)

    wsd = OptimizerConfig(lr_schedule_type="wsd", wsd_decay_steps_fraction=0.2, wsd_decay_style="cosine")
    self.assertEqual(wsd.wsd_decay_style, config_types.WsdDecayStyle.COSINE)
    with self.assertRaisesRegex(ValidationError, "must not exceed 1.0"):
      OptimizerConfig(lr_schedule_type="wsd", warmup_steps_fraction=0.6, wsd_decay_steps_fraction=0.5)

  def test_checkpoint_and_optimizer_validators(self):
    with self.assertRaisesRegex(ValidationError, "At most one"):
      CheckpointConfig(load_full_state_path="gs://a", load_parameters_path="gs://b")
    with self.assertRaisesRegex(ValidationError, "safetensors_dynamic"):
      CheckpointConfig(load_full_state_path="gs://a", source_checkpoint_layout="safetensors_dynamic")
    opt = OptimizerConfig(opt_type=config_types.OptimizerType.ADAMW)
    self.assertEqual(opt.opt_type, "adamw")

  def test_scoped_subconfigs_and_extra_forbid(self):
    cfg = TrainingConfig(
        base_output_directory="gs://bucket/run",
        steps=10,
        max_target_length=128,
        num_vocab_tiling=4,
        transforms=TransformConfig(scan_layers=False),
        optimizer=OptimizerConfig(learning_rate=1e-3),
        checkpoint=CheckpointConfig(enable_checkpointing=False),
        profiling=ProfilingConfig(profiler="xplane"),
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
    serialized = cfg.model_dump_json(indent=2)
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
    with self.assertRaisesRegex(ValueError, "num_experts"):
      validate_training_config(moe, valid_train, mesh_shape={"expert": 3})

  def test_defaults_match_base_yml(self):
    with open(os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml"), encoding="utf-8") as f:
      base = yaml.safe_load(f)

    # Fields where TrainingConfig uses None for base.yml's -1 ("unset"); compared as -1 below.
    none_for_minus_one = {"steps", "eval_interval", "eval_steps", "learning_rate_schedule_steps"}
    # m3-only fields with no `base.yml` counterpart; any other unmatched field fails the test.
    expected_missing = {
        "aux_loss_weight",
        "enable_pathways_persistence",
        "named_scopes",
        "pad_id",
        "remat_save_names",
        "skip_step_on_nan",
    }
    missing = set()
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
        if name not in base:
          missing.add(name)
          continue
        actual = getattr(val, "value", val)
        expected = base[name]
        if isinstance(actual, tuple) and isinstance(expected, list):
          actual = list(actual)
        # Match resolved semantics for YAML `"None"` / empty `""` optional sentinels.
        if actual is None and expected in ("None", ""):
          continue
        if actual is None and name in none_for_minus_one:
          actual = -1
        self.assertEqual(actual, expected, f"Default mismatch for {name!r}")

    self.assertEqual(missing, expected_missing)
    # `param_scan_axis` describes a family's checkpoint layout, so its default lives on `ModelConfig`.
    self.assertEqual(_StubDenseConfig().param_scan_axis, base["param_scan_axis"])
    for axis, size in tc.mesh.ici.model_dump().items():
      self.assertEqual(size, base[f"ici_{axis}_parallelism"], axis)
    for axis, size in tc.mesh.dcn.model_dump().items():
      self.assertEqual(size, base[f"dcn_{axis}_parallelism"], axis)


if __name__ == "__main__":
  absltest.main()
