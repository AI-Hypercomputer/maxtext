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

"""Tests for pyconfig."""

import os.path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import omegaconf

from maxtext.configs import pyconfig
from maxtext.configs import pyconfig_deprecated
from maxtext.configs.pyconfig_deprecated import resolve_config_path, resolve_model_config_path
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR, MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_test_config_path


class PyconfigTest(unittest.TestCase):
  """Tests for 'pyconfig.py'."""

  def test_basic_override(self):
    raw_keys = {"megablox": None, "foo": ["bar", "baz"]}
    model_keys = {"foo": ["x", "y"]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(raw_keys, {"megablox": None, "foo": ["x", "y"]})

  def test_empty_string_parse_as_empty_string(self):
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,  # We should check for this automatically instead - b/407047411
        quantization="",
    )

    self.assertTrue(config.quantization is None or config.quantization == "")

  def test_logical_axis_override(self):
    raw_keys = {
        "megablox": None,
        "foo": ["bar", "baz"],
        "logical_axis_rules": [["activation", ["data", "fsdp"]], ["norm", "tensor"]],
    }
    model_keys = {"logical_axis_rules": [["activation", ["data", "fsdp_transpose"]], ["norm", "fsdp"]]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(
        raw_keys,
        {
            "megablox": None,
            "foo": ["bar", "baz"],
            "logical_axis_rules": [("activation", ["data", "fsdp_transpose"]), ("norm", "fsdp")],
        },
    )

  def test_logical_axis_partial_override(self):
    raw_keys = {
        "megablox": None,
        "foo": ["bar", "baz"],
        "logical_axis_rules": [["activation", ["data", "fsdp"]], ["norm", "tensor"]],
    }
    model_keys = {"logical_axis_rules": [["norm", "fsdp"]]}

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name="config")

    self.assertEqual(
        raw_keys,
        {
            "megablox": None,
            "foo": ["bar", "baz"],
            "logical_axis_rules": [("activation", ("data", "fsdp")), ("norm", "fsdp")],
        },
    )

  def test_multiple_unmodifiable_configs(self):
    config_train = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        ici_tensor_parallelism=-1,
        ici_fsdp_parallelism=4,
    )
    config_inference = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "decode.py"), get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        ici_tensor_parallelism=4,
        ici_fsdp_parallelism=-1,
    )
    self.assertNotEqual(
        config_train.ici_tensor_parallelism,
        config_inference.ici_tensor_parallelism,
    )
    with self.assertRaises(ValueError):
      config_inference.ici_fsdp_parallelism = 4

  def test_overriding_model(self):
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        model_name="gemma-7b",
        override_model_config=True,
        base_emb_dim=1024,  # Defined as 3072 in gemma-7b
    )

    self.assertEqual(config.base_emb_dim, 1024)
    self.assertEqual(config.base_mlp_dim, 24576)

  def test_resolve_config_path(self):
    self.assertEqual(resolve_config_path("foo"), os.path.join("src", "foo"))
    self.assertEqual(resolve_config_path(__file__), __file__)

  def test_deprecated_model_config_fallback_and_active_precedence(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_base_cfg = os.path.join(tmpdir, "base.yml")
      shutil.copyfile(os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), tmp_base_cfg)

      active_models_dir = os.path.join(tmpdir, "models")
      deprecated_models_dir = os.path.join(active_models_dir, "deprecated")
      os.makedirs(deprecated_models_dir, exist_ok=True)

      shutil.copyfile(
          os.path.join(MAXTEXT_CONFIGS_DIR, "models", "gemma-2b.yml"),
          os.path.join(deprecated_models_dir, "gemma-2b.yml"),
      )
      shutil.copyfile(
          os.path.join(MAXTEXT_CONFIGS_DIR, "models", "qwen3-0.6b.yml"),
          os.path.join(active_models_dir, "qwen3-0.6b.yml"),
      )
      with open(os.path.join(deprecated_models_dir, "qwen3-0.6b.yml"), "w", encoding="utf-8") as f:
        f.write("base_emb_dim: 99999\n")

      resolved_dep, is_dep = resolve_model_config_path(tmp_base_cfg, "gemma-2b", configs_dir=tmpdir)
      self.assertEqual(resolved_dep, os.path.join(deprecated_models_dir, "gemma-2b.yml"))
      self.assertTrue(is_dep)

      resolved_act, is_act = resolve_model_config_path(tmp_base_cfg, "qwen3-0.6b", configs_dir=tmpdir)
      self.assertEqual(resolved_act, os.path.join(active_models_dir, "qwen3-0.6b.yml"))
      self.assertFalse(is_act)

      base_raw_keys = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(tmp_base_cfg), resolve=True)
      with mock.patch.object(pyconfig_deprecated, "_DEFAULT_CONFIGS_DIR", tmpdir):
        dep_raw_keys = dict(base_raw_keys)
        dep_raw_keys["model_name"] = "gemma-2b"
        dep_raw_keys["override_model_config"] = False
        with warnings.catch_warnings(record=True) as caught:
          warnings.simplefilter("always", DeprecationWarning)
          pyconfig_deprecated._HyperParameters.update_model_vars(  # pylint: disable=protected-access
              tmp_base_cfg, dep_raw_keys, tmp_base_cfg, ["model_name"]
          )
        self.assertEqual(dep_raw_keys["base_emb_dim"], 2048)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning) and "gemma-2b" in str(w.message)]
        self.assertEqual(len(dep_warnings), 1)
        self.assertIn("MaxText Model Lifecycle Policy", str(dep_warnings[0].message))
        self.assertEqual(os.path.realpath(dep_warnings[0].filename), os.path.realpath(__file__))

        act_raw_keys = dict(base_raw_keys)
        act_raw_keys["model_name"] = "qwen3-0.6b"
        act_raw_keys["override_model_config"] = False
        with warnings.catch_warnings(record=True) as caught_active:
          warnings.simplefilter("always", DeprecationWarning)
          pyconfig_deprecated._HyperParameters.update_model_vars(  # pylint: disable=protected-access
              tmp_base_cfg, act_raw_keys, tmp_base_cfg, ["model_name"]
          )
        self.assertEqual(act_raw_keys["base_emb_dim"], 1024)
        active_dep_warnings = [
            w for w in caught_active if issubclass(w.category, DeprecationWarning) and "qwen3-0.6b" in str(w.message)
        ]
        self.assertEqual(len(active_dep_warnings), 0)


if __name__ == "__main__":
  unittest.main()
