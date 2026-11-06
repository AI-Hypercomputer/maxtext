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

"""Tests for the m3 config system (`maxtext.m3.configs.config`)."""

import os
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import yaml

from maxtext.m3.configs import config as config_lib
from maxtext.m3.configs.diffusers_config import DiffusionConfig


def _write_yaml(directory, rel_path, data):
  path = os.path.join(directory, rel_path)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f)
  return path


class ParseCliTest(unittest.TestCase):
  """CLI parsing: `argv[1]` is the YAML path, the rest typed `key=value` overrides."""

  def test_config_path_is_kept_as_given(self):
    path = "src/maxtext/m3/models/wan/configs/wan2.1-t2v-14b.yml"
    self.assertEqual(config_lib.parse_cli_args(["prog", path]), (path, {}))
    # `model_name=` is an ordinary override (it used to be taken as the config name).
    self.assertEqual(config_lib.parse_cli_args(["prog", path, "model_name=wan2.1"]), (path, {"model_name": "wan2.1"}))

  def test_bad_argv(self):
    with self.assertRaisesRegex(ValueError, "Usage"):
      config_lib.parse_cli_args(["prog"])
    with self.assertRaisesRegex(ValueError, "expected key=value"):
      config_lib.parse_cli_args(["prog", "cfg.yml", "stray_token"])

  def test_override_types(self):
    _, overrides = config_lib.parse_cli_args(
        [
            "prog",
            "cfg.yml",
            "num_frames=17",
            "flow_shift=5.0",
            "enable_profiler=False",
            "attention=flash",
            "mesh_axes=[data,fsdp]",
            "run_name='my run'",
            "flash_block_sizes={block_q:3024,block_kv:2048}",
            "lora_config='{rank:4,alpha:8}'",
            "--some-flag",
        ]
    )
    self.assertEqual(
        overrides,
        {
            "num_frames": 17,
            "flow_shift": 5.0,
            "enable_profiler": False,
            "attention": "flash",
            "mesh_axes": ["data", "fsdp"],
            "run_name": "my run",
            "flash_block_sizes": {"block_q": 3024, "block_kv": 2048},
            "lora_config": {"rank": 4, "alpha": 8},
        },
    )
    self.assertEqual(config_lib.parse_cli_overrides(["prog", "cfg.yml", "seed=3"]), {"seed": 3})


class YamlLoadingTest(unittest.TestCase):
  """YAML loading: modular `defaults`, interpolation and file lookup."""

  def test_defaults_are_merged_and_overridden(self):
    with tempfile.TemporaryDirectory() as d:
      _write_yaml(d, "base.yml", {"a": 1, "b": 1, "shared": "base"})
      _write_yaml(d, "scheduler/unipc.yml", {"flow_shift": 3.0, "shared": "scheduler"})
      model = _write_yaml(
          d,
          "model.yml",
          {
              "defaults": ["base", {"scheduler": "unipc"}, "_self_"],
              "hydra": {"run": {"dir": "."}},
              "b": 2,
              "shared": "model",
          },
      )
      raw = config_lib.load_yaml_config(model)
    self.assertEqual(raw, {"a": 1, "b": 2, "flow_shift": 3.0, "shared": "model"})

  def test_missing_file_raises(self):
    with self.assertRaises(FileNotFoundError):
      config_lib.load_yaml_config("/nonexistent/model.yml")

  def test_resolve_interpolations(self):
    d = {
        "name": "Wan-AI/Wan2.1",
        "copy": "${name}",
        "nested": {"x": 4, "ref": "${name}"},
        "group_ref": "${nested.x}",
        "embedded": "repo=${name}/sub",
        "unknown": "${missing}",
    }
    config_lib._resolve_interpolations(d)  # pylint: disable=protected-access
    self.assertEqual(d["copy"], "Wan-AI/Wan2.1")
    self.assertEqual(d["nested"]["ref"], "Wan-AI/Wan2.1")
    self.assertEqual(d["group_ref"], 4)
    self.assertEqual(d["embedded"], "repo=Wan-AI/Wan2.1/sub")
    self.assertEqual(d["unknown"], "${missing}")

  def test_initialize_requires_an_existing_yaml_path(self):
    with self.assertRaises(FileNotFoundError):
      config_lib.initialize(["prog", "wan2.1-t2v-14b"], unittest=True)  # a bare name is not looked up any more


class ProcessConfigTest(unittest.TestCase):
  """`process_config` post-processing."""

  def test_derived_fields(self):
    cfg = config_lib.process_config(
        {
            "run_name": "r",
            "base_output_directory": "gs://bucket/out",
            "logical_axis_rules": [["batch", ["data", "fsdp"]], ["embed", "fsdp"]],
            "weights_dtype": "float32",
            "activations_dtype": "bfloat16",
            "per_device_batch_size": 2,
        },
        unittest=True,
    )
    self.assertIsInstance(cfg, config_lib.BaseConfig)
    self.assertNotIsInstance(cfg, DiffusionConfig)
    self.assertEqual(cfg.tensorboard_dir, "gs://bucket/out/r/tensorboard/")
    self.assertEqual(cfg.checkpoint_dir, "gs://bucket/out/r/checkpoints/")
    self.assertEqual(cfg.metrics_dir, "gs://bucket/out/r/metrics/")
    self.assertEqual(cfg.logical_axis_rules, (("batch", ("data", "fsdp")), ("embed", "fsdp")))
    self.assertEqual(cfg.weights_dtype, jnp.float32)
    self.assertEqual(cfg.activations_dtype, jnp.bfloat16)
    num_devices = len(jax.devices())
    self.assertEqual(cfg.global_batch_size_to_train_on, 2 * num_devices)
    self.assertEqual(cfg.global_batch_size_to_load, 2 * num_devices)

  def test_fractional_per_device_batch_loads_one_per_device(self):
    cfg = config_lib.process_config({"run_name": "r", "per_device_batch_size": 0.125}, unittest=True)
    self.assertEqual(cfg.global_batch_size_to_load, len(jax.devices()))
    self.assertEqual(cfg.total_train_batch_size, len(jax.devices()))

  def test_default_run_name_and_empty_dirs(self):
    with mock.patch.dict(os.environ, {"JOBSET_NAME": "jobset-1"}):
      cfg = config_lib.process_config({}, unittest=True)
    self.assertEqual(cfg.run_name, "jobset-1")
    self.assertEqual((cfg.tensorboard_dir, cfg.checkpoint_dir, cfg.metrics_dir), ("", "", ""))

  def test_cli_overrides_and_kwargs_win(self):
    cfg = config_lib.process_config({"run_name": "r", "seed": 0}, cli_overrides={"seed": 1}, unittest=True, num_frames=9)
    self.assertEqual(cfg.seed, 1)
    self.assertIsInstance(cfg, DiffusionConfig)  # `num_frames` marks a diffusion config
    self.assertEqual(cfg.num_frames, 9)

  def test_distributed_init_runs_before_first_device_query(self):
    """#5418 A3: `jax.distributed.initialize()` must come before anything initializes the backends."""
    calls = []
    real_devices = jax.devices
    with (
        mock.patch.object(
            config_lib, "maybe_initialize_jax_distributed_system", side_effect=lambda raw: calls.append("init")
        ),
        mock.patch.object(config_lib.jax, "devices", side_effect=lambda *a: calls.append("devices") or real_devices()),
    ):
      config_lib.process_config({"run_name": "r"})
    self.assertEqual(calls[0], "init")
    self.assertIn("devices", calls)


class DistributedInitTest(unittest.TestCase):
  """`maybe_initialize_jax_distributed_system` early exits and the call itself."""

  def _run(self, raw, is_initialized=False):
    with (
        mock.patch.object(config_lib.jax.distributed, "is_initialized", return_value=is_initialized),
        mock.patch.object(config_lib.jax.distributed, "initialize") as init,
    ):
      config_lib.maybe_initialize_jax_distributed_system(raw)
    return init

  def test_skips(self):
    for raw, is_initialized in (
        ({"skip_jax_distributed_system": True}, False),
        ({}, True),
        ({"enable_single_controller": True}, False),
        ({"compile_topology": "v6e-32"}, False),
        ({"inference_benchmark_test": True}, False),
    ):
      with self.subTest(raw=raw, is_initialized=is_initialized):
        self._run(raw, is_initialized).assert_not_called()

  def test_initializes_with_timeout(self):
    self._run({"jax_distributed_initialization_timeout": 42}).assert_called_once_with(initialization_timeout=42)

  def test_failure_is_logged_not_raised(self):
    with (
        mock.patch.object(config_lib.jax.distributed, "is_initialized", return_value=False),
        mock.patch.object(config_lib.jax.distributed, "initialize", side_effect=RuntimeError("no cluster")),
    ):
      config_lib.maybe_initialize_jax_distributed_system({})


class BaseConfigTest(unittest.TestCase):
  """Attribute / dict access routed through the sub-configs."""

  def setUp(self):
    super().setUp()
    self.cfg = config_lib.BaseConfig.from_dict({"learning_rate": 1e-4, "ici_fsdp_parallelism": 8, "my_extra": "x"})

  def test_lookup(self):
    self.assertEqual(self.cfg.learning_rate, 1e-4)  # TrainerConfig field
    self.assertEqual(self.cfg.ici_fsdp_parallelism, 8)  # DeviceConfig field
    self.assertEqual(self.cfg.my_extra, "x")  # extra field
    self.assertEqual(self.cfg["learning_rate"], 1e-4)
    self.assertEqual(self.cfg.get("missing", 7), 7)
    self.assertIn("ici_fsdp_parallelism", self.cfg)
    self.assertNotIn("missing", self.cfg)
    with self.assertRaises(AttributeError):
      _ = self.cfg.missing
    with self.assertRaises(KeyError):
      _ = self.cfg["missing"]

  def test_setattr_is_visible_to_reads(self):
    """Reads after writes see the new value (`sample.py` toggles `enable_profiler`, the trainer sets
    `tensorboard_dir`); `from_dict` keeps a copy of each raw key in every sub-config and at the top level."""
    cfg = config_lib.BaseConfig.from_dict(
        {"learning_rate": 1e-4, "enable_profiler": True, "tensorboard_dir": "gs://a/tb/"}
    )
    cfg.learning_rate = 3e-4  # TrainerConfig field
    cfg.enable_profiler = False  # extra key
    cfg.tensorboard_dir = "gs://b/tb/"  # CheckpointConfig field
    cfg["new_key"] = 5
    self.assertEqual(cfg.learning_rate, 3e-4)
    self.assertEqual(cfg.trainer.learning_rate, 3e-4)
    self.assertFalse(cfg.enable_profiler)
    self.assertEqual(cfg.tensorboard_dir, "gs://b/tb/")
    self.assertEqual(cfg.checkpoint.tensorboard_dir, "gs://b/tb/")
    self.assertEqual(cfg.new_key, 5)
    keys = cfg.get_keys()
    self.assertEqual((keys["learning_rate"], keys["enable_profiler"], keys["new_key"]), (3e-4, False, 5))

  def test_diffusion_config_setattr(self):
    cfg = config_lib.process_config({"run_name": "r", "num_frames": 81, "enable_profiler": True}, unittest=True)
    cfg.enable_profiler = False
    cfg.num_frames = 17
    self.assertFalse(cfg.enable_profiler)
    self.assertEqual(cfg.num_frames, 17)


class InitializeTest(unittest.TestCase):
  """End to end: a Wan-style YAML (written to a temp dir, so the test does not depend on a model PR) + overrides."""

  def test_yaml_and_overrides(self):
    with tempfile.TemporaryDirectory() as d:
      path = _write_yaml(
          d,
          "wan.yml",
          {
              "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
              "model_name": "wan2.1",
              "wan_transformer_pretrained_model_name_or_path": "${pretrained_model_name_or_path}",
              "num_frames": 81,
              "attention": "dot_product",
              "logical_axis_rules": [["batch", ["data", "fsdp"]], ["heads", "tensor"]],
              "text_encoder_logical_axis_rules": [["heads", ["data", "fsdp", "context", "tensor"]], ["embed", None]],
          },
      )
      cfg = config_lib.initialize(["prog", path, "num_frames=17", "attention=flash", "run_name=t"], unittest=True)
    self.assertIsInstance(cfg, DiffusionConfig)
    self.assertEqual(cfg.model_name, "wan2.1")
    self.assertEqual(cfg.num_frames, 17)
    self.assertEqual(cfg.attention, "flash")
    # `${pretrained_model_name_or_path}` interpolation in the YAML.
    self.assertEqual(cfg.wan_transformer_pretrained_model_name_or_path, "Wan-AI/Wan2.1-T2V-14B-Diffusers")
    self.assertEqual(cfg.logical_axis_rules, (("batch", ("data", "fsdp")), ("heads", "tensor")))
    self.assertEqual(
        cfg.text_encoder_logical_axis_rules, (("heads", ("data", "fsdp", "context", "tensor")), ("embed", None))
    )


if __name__ == "__main__":
  unittest.main()
