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
import tempfile
import unittest

from maxtext.configs import pyconfig
from maxtext.configs.pyconfig import resolve_config_path, _CONFIG_FILE_MAPPING, _module_from_path
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR, MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_test_config_path, get_post_train_test_config_path


class PyconfigTest(unittest.TestCase):
  """Tests for 'pyconfig.py'."""

  def test_empty_string_parse_as_empty_string(self):
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,  # We should check for this automatically instead - b/407047411
        quantization="",
    )

    self.assertTrue(config.quantization is None or config.quantization == "")

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

    self.assertEqual(config.base_emb_dim, 1024)  # override
    self.assertEqual(config.base_mlp_dim, 24576)  # unchanged

  def test_overriding_model_raises_error(self):
    """Test that overriding a model config with override_model_config=False raises an error."""
    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          model_name="gemma-7b",
          override_model_config=False,
          base_emb_dim=1024,  # Defined as 3072 in gemma-7b
      )

  def test_overriding_model_in_sft(self):
    # TODO: Update MAXTEXT_PKG_DIR after repo restructuring is complete.
    config = pyconfig.initialize(
        [os.path.join("maxtext.trainers.post_train.sft.train_sft"), get_post_train_test_config_path("sft")],
        skip_jax_distributed_system=True,
        model_name="llama3.1-8b",
        override_model_config=True,
        base_emb_dim=1024,  # Defined as 4096 in llama3.1-8b
    )

    self.assertEqual(config.base_emb_dim, 1024)  # override
    self.assertEqual(config.base_mlp_dim, 14336)  # unchanged

  def test_tokenizer_path_resolution_for_qwen3_base(self):
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        model_name="qwen3-30b-a3b-base",
    )
    self.assertEqual(config.tokenizer_path, "Qwen/Qwen3-30B-A3B-Base")

  def test_resolve_config_path(self):
    self.assertEqual(resolve_config_path("foo"), os.path.join("src", "foo"))
    self.assertEqual(resolve_config_path(__file__), __file__)

  def test_resolve_config_path_pip_install(self):
    """Simulates pip-installed env where cwd has no src/ folder."""
    orig = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
      try:
        os.chdir(tmpdir)
        result = resolve_config_path("src/maxtext/configs/base.yml")
        self.assertEqual(result, os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"))
        result = resolve_config_path("src/maxtext/configs/post_train/rl.yml")
        self.assertEqual(result, os.path.join(MAXTEXT_CONFIGS_DIR, "post_train/rl.yml"))
      finally:
        os.chdir(orig)

  def test_config_file_mapping(self):
    for module, relative_path in _CONFIG_FILE_MAPPING.items():
      full_path = os.path.join(MAXTEXT_CONFIGS_DIR, relative_path)
      self.assertTrue(os.path.isfile(full_path), f"Default config for '{module}' not found at {full_path}")

  def test_module_from_path(self):
    import maxtext.trainers.pre_train.train as train_module  # pylint: disable=import-outside-toplevel

    module_file = train_module.__file__
    result = _module_from_path(module_file)
    self.assertEqual(result, "maxtext.trainers.pre_train.train")

  def test_hlo_dump_module_names_none_coercion(self):
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        dump_hlo_local_module_name=None,
        dump_hlo_module_name=None,
    )
    self.assertEqual(config.dump_hlo_local_module_name, "")
    self.assertEqual(config.dump_hlo_module_name, "")

  def test_unknown_module_falls_back_to_base_yml(self):
    """An unknown module should fall back to base.yml with a warning (not raise)."""
    config = pyconfig.initialize_pydantic(["/custom_rl/module.py", "run_name=test", "skip_jax_distributed_system=True"])
    self.assertEqual(config.run_name, "test")


if __name__ == "__main__":
  unittest.main()
