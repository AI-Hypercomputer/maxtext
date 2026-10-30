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

"""Integration tests showing `ArgumentParser` and `from_pyconfig` with `loop.py` and `train.py`."""

import glob
import math
import os
import pathlib
import types as pytypes

from absl.testing import absltest
import numpy as np

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.m3.configs import pyconfig_bridge
from maxtext.m3.configs.cli_parser import ArgumentParser
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.pyconfig_bridge import ConfigMigrationError, from_pyconfig
from maxtext.m3.configs.training_config import TrainingConfig
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5ModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.models.registry import MODEL_GARDEN_REGISTRY
from maxtext.m3.train import loop
from maxtext.utils import train_utils
from tests.utils.test_helpers import get_test_config_path


class ConfigIntegrationTest(absltest.TestCase):
  """Demonstrates zero-YAML CLI and `pyconfig` bridge integration with `loop.py` and `train.py`."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_native_cli_runs_m3_loop(self):
    """Native path: zero-YAML CLI (`loop.parse_argv` / `ArgumentParser`) -> `loop.train_loop`."""
    saved_json = os.path.join(self.output_dir, "provenance.json")
    cli_argv = [
        "loop.py",
        "--model_name=qwen3-0.6b",
        "--hidden_size=64",
        "--intermediate_size=128",
        "--num_hidden_layers=1",
        "--num_attention_heads=4",
        "--num_key_value_heads=2",
        "--head_dim=16",
        "--vocab_size=256",
        "--steps=2",
        "--max_seq_length=16",
        "--per_device_batch_size=1",
        "--scan_layers=false",
        "--learning_rate=1e-3",
        "--profiler=xplane",
        "--skip_first_n_steps_for_profiler=0",
        "--profiler_steps=1",
        f"--output_dir={self.output_dir}",
        "--run_name=cli_loop_demo",
        f"--save_config_json={saved_json}",
    ]
    model_cfg, train_cfg = loop.parse_argv(cli_argv)

    self.assertIsInstance(model_cfg, Qwen3Config)
    self.assertIsInstance(train_cfg, TrainingConfig)
    self.assertFalse(train_cfg.transforms.scan_layers)
    self.assertEqual(train_cfg.optimizer.learning_rate, 1e-3)

    # Provenance JSON round-trip reproduces identical frozen configs.
    reloaded_model, reloaded_train = ArgumentParser().parse_args([f"--config_json={saved_json}"])
    self.assertEqual(reloaded_model, model_cfg)
    self.assertEqual(reloaded_train, train_cfg)

    loss = loop.train_loop(model_cfg, train_cfg)
    self.assertTrue(math.isfinite(loss), f"Non-finite loss: {loss}")
    tb_dir = os.path.join(self.output_dir, "cli_loop_demo", "tensorboard")
    traces = glob.glob(os.path.join(tb_dir, "**", "*.xplane.pb"), recursive=True)
    self.assertNotEmpty(traces, f"Expected .xplane.pb under {tb_dir}")

  def test_pyconfig_bridge_matches_native_cli(self):
    """Transitional `base.yml` path (`from_pyconfig`) matches native zero-YAML CLI in `loop.train_loop`."""
    yaml_argv = [
        "",
        get_test_config_path(),
        "model_name=qwen3-0.6b",
        "use_m3_model=true",
        "override_model_config=true",
        "scan_layers=false",
        "base_num_decoder_layers=1",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
        "per_device_batch_size=1",
        "max_target_length=16",
        "steps=2",
        "learning_rate=1e-3",
        "enable_checkpointing=false",
        f"base_output_directory={self.output_dir}",
        "run_name=bridge_demo",
        "skip_jax_distributed_system=true",
    ]
    raw_cfg = pyconfig.initialize(yaml_argv)
    # Same pre-flight validation called inside `train.py:initialize`.
    train_utils.validate_train_config(raw_cfg)

    # 1. Existing `loop.train_loop(raw_cfg)` bridges automatically via `from_pyconfig`.
    pycfg_loss = loop.train_loop(raw_cfg)
    self.assertTrue(math.isfinite(pycfg_loss), f"pycfg_loss={pycfg_loss}")

    # 2. `loop.parse_argv` with `base.yml` dispatches to `from_pyconfig`.
    bridge_model, bridge_train = loop.parse_argv(yaml_argv)
    self.assertIsInstance(bridge_model, Qwen3Config)
    self.assertEqual((bridge_model.hidden_size, bridge_model.num_hidden_layers), (64, 1))
    self.assertEqual((bridge_train.steps, bridge_train.max_seq_length), (2, 16))
    self.assertEqual(bridge_train.output_dir, self.output_dir)

    # 3. Equivalent native CLI invocation produces identical model config and bit-exact loss.
    cli_model, cli_train = loop.parse_argv(
        [
            "loop.py",
            "--model_name=qwen3-0.6b",
            "--hidden_size=64",
            "--intermediate_size=128",
            "--num_hidden_layers=1",
            "--num_attention_heads=4",
            "--num_key_value_heads=2",
            "--head_dim=16",
            "--vocab_size=256",
            "--steps=2",
            "--max_seq_length=16",
            "--per_device_batch_size=1",
            "--scan_layers=false",
            "--learning_rate=1e-3",
            f"--output_dir={self.output_dir}",
            "--run_name=bridge_demo",
        ]
    )
    self.assertEqual(bridge_model, cli_model)
    cli_loss = loop.train_loop(cli_model, cli_train)
    bridge_loss = loop.train_loop(bridge_model, bridge_train)
    np.testing.assert_allclose(pycfg_loss, bridge_loss, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bridge_loss, cli_loss, rtol=0.0, atol=0.0)

  def test_yaml_free_glm5_across_registry_cli_and_pyconfig_bridge(self):
    """Verifies `glm5.3-flash` works end-to-end with `glm5.3-flash.yml` deleted."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    deleted_yml = repo_root / "src" / "maxtext" / "configs" / "models" / "glm5.3-flash.yml"
    self.assertFalse(deleted_yml.exists(), f"Expected YAML-free model; found {deleted_yml}")

    # Direct static registry lookup works without side-effect registration.
    reg_cfg = MODEL_GARDEN_REGISTRY["glm5.3-flash"]()
    self.assertIsInstance(reg_cfg, Glm5ModelConfig)

    # `pyconfig.initialize` falls back lazily to `registry.get_pyconfig_model_dict`.
    raw_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=glm5.3-flash",
            "steps=5",
            "enable_checkpointing=false",
            f"base_output_directory={self.output_dir}",
            "run_name=glm5_bridge_demo",
            "skip_jax_distributed_system=true",
        ]
    )
    self.assertTrue(raw_cfg.use_m3_model)
    train_utils.validate_train_config(raw_cfg)

    bridge_model, bridge_train = from_pyconfig(raw_cfg)
    cli_model, cli_train = ArgumentParser().parse_args(
        [
            "--model_name=glm5.3-flash",
            "--steps=5",
            f"--output_dir={self.output_dir}",
            "--run_name=glm5_bridge_demo",
        ]
    )
    self.assertEqual(bridge_model, cli_model)
    self.assertEqual(bridge_train.steps, cli_train.steps)
    self.assertEqual(bridge_model.to_glm5_config(), Glm5Config())

  def test_from_pyconfig_rejects_unsupported_flags_and_modes(self):
    """Fail-fast checks catch unsupported CLI flags, custom YAML values, AQT quant, and AOT files."""
    # CLI override of unsupported bool flag.
    bad_cli_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "enable_wandb=true",
            "skip_jax_distributed_system=true",
        ]
    )
    with self.assertRaisesRegex(ConfigMigrationError, "enable_wandb"):
      from_pyconfig(bad_cli_cfg)

    # Live-value audit catches unsupported flags even when `model_fields_set` is empty (custom YAML).
    custom_yaml_stub = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        ici_pipeline_parallelism=2,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "ici_pipeline_parallelism=2"):
      from_pyconfig(custom_yaml_stub)

    # `compiled_trainstep_file` is rejected in `mode='train'` and permitted in `mode='aot'`.
    aot_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "compiled_trainstep_file=/tmp/compiled.bin",
            "skip_jax_distributed_system=true",
        ]
    )
    with self.assertRaisesRegex(ConfigMigrationError, "compiled_trainstep_file"):
      from_pyconfig(aot_cfg, mode="train")
    _, aot_train = from_pyconfig(aot_cfg, mode="aot")
    self.assertEqual(aot_train.aot.compiled_trainstep_file, "/tmp/compiled.bin")

    # Non-qwix quantization (`use_qwix_quantization=false`) is rejected; qwix int8 passes and runs in `loop`.
    bad_quant = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        quantization="int8",
        use_qwix_quantization=False,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "use_qwix_quantization=True"):
      from_pyconfig(bad_quant)

    q_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "use_m3_model=true",
            "override_model_config=true",
            "scan_layers=false",
            "base_num_decoder_layers=1",
            "base_emb_dim=64",
            "base_mlp_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
            "per_device_batch_size=1",
            "max_target_length=16",
            "steps=1",
            "quantization=int8",
            "use_qwix_quantization=true",
            "enable_checkpointing=false",
            f"base_output_directory={self.output_dir}",
            "run_name=quant_demo",
            "skip_jax_distributed_system=true",
        ]
    )
    self.assertTrue(math.isfinite(loop.train_loop(q_cfg)))

    # Unified `_UNSUPPORTED_FLAGS` also catches string/scalar non-goal values.
    scalar_stub = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        custom_mesh="1x1",
        moe_dropless_fallback="step",
        opt_type="adamw",
    )
    with self.assertRaisesRegex(
        ConfigMigrationError, "custom_mesh.*moe_dropless_fallback|moe_dropless_fallback.*custom_mesh"
    ):
      from_pyconfig(scalar_stub)

    # Invalid `steps` fails fast instead of being silently clamped to 1.
    bad_steps = pytypes.SimpleNamespace(model_name="qwen3-0.6b", model_fields_set=set(), steps=-5, opt_type="adamw")
    with self.assertRaisesRegex(ConfigMigrationError, "Invalid training config"):
      from_pyconfig(bad_steps)

  def test_argument_parser_and_bridge_edge_cases(self):
    """Guards `-Instruct` round-trip, shared dtype flags, positional errors, and schema drift."""
    parser = ArgumentParser()
    saved_json = os.path.join(self.output_dir, "edge.json")
    m_cfg, t_cfg = parser.parse_args(
        [
            "--model_name=qwen3-0.6b",
            "--compute_dtype=float32",
            "--param_dtype=bfloat16",
            f"--save_config_json={saved_json}",
        ]
    )
    # Shared flat dtype flags propagate to both `ModelConfig` and `TrainingConfig`.
    self.assertEqual(m_cfg.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(t_cfg.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(m_cfg.param_dtype, config_types.DType.BFLOAT16)
    self.assertEqual(t_cfg.param_dtype, config_types.DType.BFLOAT16)

    # `-Instruct` override on top of a saved JSON does not collide with `model_name`.
    m_inst, _ = parser.parse_args([f"--config_json={saved_json}", "--model_name=qwen3-0.6b-Instruct"])
    self.assertEqual(m_inst, m_cfg)

    # Bogus non-flag positional args fail fast in `ArgumentParser`; `argv[1]` `*.yml` routes to `pyconfig`.
    with self.assertRaisesRegex(ValueError, "Expected `--flag`"):
      parser.parse_args(["bogus_typo_arg", "--model_name=qwen3-0.6b"])
    with self.assertRaisesRegex(ValueError, "Expected `--flag`"):
      loop.parse_argv([get_test_config_path(), "--model_name=qwen3-0.6b"])
    with self.assertRaisesRegex(ValueError, "'--model_name'"):
      loop.parse_argv(["loop.py", get_test_config_path(), "--model_name=qwen3-0.6b", "skip_jax_distributed_system=true"])
    # Flag values ending in `.yml` still route to `ArgumentParser` rather than `pyconfig`.
    _, eq_yml_train = loop.parse_argv(["loop.py", "--run_name=foo.yml", "--model_name=qwen3-0.6b"])
    self.assertEqual(eq_yml_train.run_name, "foo.yml")
    _, sp_yml_train = loop.parse_argv(["loop.py", "--model_name", "qwen3-0.6b", "--run_name", "foo.yml"])
    self.assertEqual(sp_yml_train.run_name, "foo.yml")

    # `learning_rate_final_fraction` translates cleanly (no stale field-name typo).
    lr_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "learning_rate_final_fraction=0.05",
            "skip_jax_distributed_system=true",
        ]
    )
    _, lr_train = from_pyconfig(lr_cfg)
    self.assertAlmostEqual(lr_train.optimizer.cosine_learning_rate_final_fraction, 0.05)

    # Anti-drift check: every audited key exists on `MaxTextConfig`.
    mt_fields = set(config_types.MaxTextConfig.model_fields)
    self.assertEmpty(pyconfig_bridge._TRANSLATED_PYCONFIG_KEYS - mt_fields)  # pylint: disable=protected-access
    self.assertEmpty(pyconfig_bridge._IGNORED_PYCONFIG_KEYS - mt_fields)  # pylint: disable=protected-access
    self.assertEmpty(set(pyconfig_bridge._UNSUPPORTED_FLAGS) - mt_fields)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
