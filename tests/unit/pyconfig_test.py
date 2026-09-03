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
import subprocess
import sys
import tempfile
import unittest

from maxtext.configs import pyconfig
from maxtext.configs.pyconfig import resolve_config_path, _CONFIG_FILE_MAPPING, _module_from_path
from maxtext.configs.types import _normalize_axes, _resolved_fsdp_size, infer_cp_axes, infer_ep_axes
from maxtext.input_pipeline import data_processing_utils
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

  def test_gmm_v2_heuristic_tiling_requires_gmm_v2(self):
    with self.assertRaisesRegex(ValueError, "`use_gmm_v2_heuristic_tiling=True` requires `use_gmm_v2=True`."):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          use_gmm_v2_heuristic_tiling=True,
          use_gmm_v2=False,
      )

  def test_managed_mldiagnostics_storage_path(self):
    # Test completely omitting the parameter (defaults to "" from base.yml)
    config_omitted = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        run_name="test_run_1",
        base_output_directory="gs://base_dir1",
    )
    self.assertEqual(
        config_omitted.managed_mldiagnostics_dir,
        "gs://base_dir1/test_run_1/managed-mldiagnostics",
    )

    config_none = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        run_name="test_run_2",
        base_output_directory="gs://base_dir2",
        managed_mldiagnostics_storage_path="",
    )
    self.assertEqual(
        config_none.managed_mldiagnostics_dir,
        "gs://base_dir2/test_run_2/managed-mldiagnostics",
    )

    config_custom = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        run_name="test_run_3",
        base_output_directory="gs://base_dir3",
        managed_mldiagnostics_storage_path="gs://custom_base",
    )
    self.assertEqual(
        config_custom.managed_mldiagnostics_dir,
        "gs://custom_base/test_run_3/managed-mldiagnostics",
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

  def _zero1_config(self, **kwargs):
    return pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        shard_optimizer_over_data=True,
        **kwargs,
    )

  def test_zero1_without_fsdp_is_allowed(self):
    """Data-parallel-only Zero-1 is the supported configuration."""
    config = self._zero1_config(ici_data_parallelism=-1, ici_fsdp_parallelism=1)
    self.assertTrue(config.shard_optimizer_over_data)

  def test_zero1_with_mesh_axes_omitting_fsdp_allowed(self):
    """Custom mesh_axes that omit 'fsdp' or 'fsdp_transpose' do not raise KeyError."""
    self.assertEqual(_resolved_fsdp_size(["data", "tensor"], [2, 4], 8), 1)

  def test_zero1_with_fsdp_raises_error(self):
    """Zero-1 shards the optimizer moments over "data" on top of the parameter layout.

    FSDP shards the parameters over "fsdp", so the two together leave the gradients
    sharded P('fsdp', ...) while the moments they are added to are sharded
    P(('data', 'fsdp'), ...) — an outright type error under explicit sharding, and an
    extra collective under auto. The combination is refused when the config is built.
    """
    with self.assertRaisesRegex(ValueError, "cannot be combined with FSDP"):
      self._zero1_config(ici_data_parallelism=1, ici_fsdp_parallelism=2)

  def test_zero1_with_fsdp_transpose_raises_error(self):
    with self.assertRaisesRegex(ValueError, "cannot be combined with FSDP"):
      self._zero1_config(ici_data_parallelism=1, ici_fsdp_parallelism=1, ici_fsdp_transpose_parallelism=2)

  def test_zero1_with_autofilled_fsdp_raises_error(self):
    """`ici_fsdp_parallelism=-1` absorbs whatever data parallelism leaves behind.

    The check has to resolve the -1 the way mesh creation later will, otherwise this
    config — four-way FSDP on a v5p-8 — would slip through and fail at trace time.
    """
    with self.assertRaisesRegex(ValueError, "cannot be combined with FSDP"):
      self._zero1_config(
          compile_topology="v5p-8",
          compile_topology_num_slices=1,
          ici_data_parallelism=2,
          ici_fsdp_parallelism=-1,
      )

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

  def test_explicit_sharding_qwen3_decoder_support(self):
    """The Qwen3 decoders that have been onboarded to explicit sharding are accepted."""
    for decoder_block in ("qwen3", "qwen3_moe", "qwen3_custom_moe"):
      with self.subTest(decoder_block=decoder_block):
        config = pyconfig.initialize(
            [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
            skip_jax_distributed_system=True,
            shard_mode="explicit",
            decoder_block=decoder_block,
        )
        self.assertEqual(config.decoder_block.value, decoder_block)

    # Qwen3-Next and Qwen3.5 use gated-delta-net linear attention, and the
    # Qwen3-VL/Omni encoders are multimodal; neither is onboarded yet.
    for decoder_block in ("qwen3_next", "qwen3_5"):
      with self.subTest(decoder_block=decoder_block):
        with self.assertRaisesRegex(Exception, "not supported with 'explicit' sharding"):
          pyconfig.initialize(
              [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
              skip_jax_distributed_system=True,
              shard_mode="explicit",
              decoder_block=decoder_block,
          )

    with self.assertRaisesRegex(Exception, "not supported with `use_multimodal`"):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          shard_mode="explicit",
          model_name="qwen3-vl-4b",
          override_model_config=True,
          use_multimodal=True,
          scan_layers=False,  # Required by the Qwen3-VL deepstack path; unrelated to sharding.
      )

  def test_explicit_sharding_mistral_decoder_support(self):
    """The Mistral-family decoders that have been onboarded to explicit sharding are accepted."""
    for decoder_block in ("mistral", "mixtral"):
      with self.subTest(decoder_block=decoder_block):
        config = pyconfig.initialize(
            [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
            skip_jax_distributed_system=True,
            shard_mode="explicit",
            decoder_block=decoder_block,
        )
        self.assertEqual(config.decoder_block.value, decoder_block)

  def test_lm_head_weight_grad_in_kernel_order_default(self):
    """Left unset, the flag is on exactly where it can do something."""

    def resolve(**kwargs):
      return pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          **kwargs,
      ).lm_head_weight_grad_in_kernel_order

    # The transpose it removes only exists under explicit sharding, on an untied head.
    self.assertTrue(resolve(shard_mode="explicit"))
    self.assertFalse(resolve(shard_mode="auto"))
    self.assertFalse(resolve(shard_mode="explicit", logits_via_embedding=True))
    # Writing it out still wins over the default, in both directions.
    self.assertFalse(resolve(shard_mode="explicit", lm_head_weight_grad_in_kernel_order=False))
    self.assertTrue(resolve(shard_mode="auto", lm_head_weight_grad_in_kernel_order=True))
    # But asking for it on a tied head is still a hard error, not a silent no-op.
    with self.assertRaisesRegex(Exception, "only applies to the untied LM head"):
      resolve(shard_mode="explicit", logits_via_embedding=True, lm_head_weight_grad_in_kernel_order=True)

  def test_dense_weight_grad_in_kernel_order_default(self):
    """The in-loop counterpart follows the same rule, minus the tied-head carve-out."""

    def resolve(**kwargs):
      return pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          **kwargs,
      ).dense_weight_grad_in_kernel_order

    self.assertTrue(resolve(shard_mode="explicit"))
    self.assertFalse(resolve(shard_mode="auto"))
    # Unlike the LM head, every model has these projections, tied or not.
    self.assertTrue(resolve(shard_mode="explicit", logits_via_embedding=True))
    # Writing it out still wins over the default, in both directions.
    self.assertFalse(resolve(shard_mode="explicit", dense_weight_grad_in_kernel_order=False))
    self.assertTrue(resolve(shard_mode="auto", dense_weight_grad_in_kernel_order=True))

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

  def test_train_import_without_tensorflow(self):
    """Verifies that importing the pre-training entrypoint does not require TensorFlow.

    This runs in a subprocess because TensorFlow may already be cached in the main process.
    The subprocess temporarily replaces Python's built-in import function with
    a wrapper that raises ``ModuleNotFoundError`` only for TensorFlow imports.
    A successful subprocess proves that ``train`` remains importable and sets
    ``_TF_AVAILABLE`` to False when TensorFlow is absent.
    """
    script = """
import builtins

# Save Python's real import function so non-TensorFlow imports continue to work.
original_import = builtins.__import__


def import_without_tensorflow(name, *args, **kwargs):
  if name == "tensorflow" or name.startswith("tensorflow."):
    raise ModuleNotFoundError("TensorFlow blocked by test")
  return original_import(name, *args, **kwargs)


# Simulate TensorFlow not being installed for the remainder of this subprocess.
builtins.__import__ = import_without_tensorflow

from maxtext.trainers.pre_train import train

assert train._TF_AVAILABLE is False
"""

    subprocess.run([sys.executable, "-c", script], check=True)

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

  def test_identical_override_allowed(self):
    """Test that overriding a model config key with an identical value is allowed."""
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        model_name="qwen3-8b",
        override_model_config=False,
        tokenizer_type="huggingface",  # Defined as huggingface in qwen3-8b
    )
    self.assertEqual(config.tokenizer_type, "huggingface")

  def test_list_config_coercion(self):
    """Verifies that string/tuple inputs for list[str] config fields are coerced to lists."""
    # Case 1: Plain string (coerced to single-item list)
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="messages",
    )
    self.assertEqual(config.train_data_columns, ["messages"])

    # Case 2: Stringified list literal
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="['col1', 'col2']",
    )
    self.assertEqual(config.train_data_columns, ["col1", "col2"])

    # Case 3: Stringified list literal with whitespace
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="[ 'col1' ,    'col2' ]",
    )
    self.assertEqual(config.train_data_columns, ["col1", "col2"])

    # Case 4: Stringified tuple literal
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="('col1', 'col2')",
    )
    self.assertEqual(config.train_data_columns, ["col1", "col2"])

    # Case 5: Real tuple value (passed via kwargs)
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns=("col1", "col2"),
    )
    self.assertEqual(config.train_data_columns, ["col1", "col2"])

    # Case 6: Malformed stringified list (falls back to wrapping as single-item list)
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="[malformed, list",
    )
    self.assertEqual(config.train_data_columns, ["[malformed, list"])

  def test_coerced_list_is_validated_successfully(self):
    """Verifies that a coerced list from pyconfig is successfully validated by the dataset pipeline."""
    # Simulate a user passing `train_data_columns=messages` on the CLI
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        train_data_columns="messages",
    )
    # Verify coercion to list was successful
    self.assertEqual(config.train_data_columns, ["messages"])

    # Verify that passing this coerced list to the SFT column validator passes without error (Scenario A)
    data_processing_utils.validate_and_configure_sft_columns(config.train_data_columns, None)

  def test_local_sa_flags_inherit_from_global_when_unset(self):
    """local_sa_* flags default to None and should inherit the corresponding sa_* value."""
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        sa_block_q=64,
        sa_block_kv=128,
        sa_block_kv_compute=192,
        sa_block_q_dkv=64,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=192,
        sa_block_q_dq=64,
        sa_block_kv_dq=128,
        sa_use_fused_bwd_kernel=True,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        use_splash_scheduler=True,
    )
    self.assertEqual(config.local_sa_block_q, 64)
    self.assertEqual(config.local_sa_block_kv, 128)
    self.assertEqual(config.local_sa_block_kv_compute, 192)
    self.assertEqual(config.local_sa_block_q_dkv, 64)
    self.assertEqual(config.local_sa_block_kv_dkv, 128)
    self.assertEqual(config.local_sa_block_kv_dkv_compute, 192)
    self.assertEqual(config.local_sa_block_q_dq, 64)
    self.assertEqual(config.local_sa_block_kv_dq, 128)
    self.assertTrue(config.local_sa_use_fused_bwd_kernel)
    self.assertEqual(config.local_sa_q_layout, "HEAD_DIM_MINOR")
    self.assertEqual(config.local_sa_k_layout, "HEAD_DIM_MINOR")
    self.assertEqual(config.local_sa_v_layout, "HEAD_DIM_MINOR")
    self.assertTrue(config.local_use_splash_scheduler)

  def test_local_sa_flags_explicit_override(self):
    """Explicitly set local_sa_* flags should not be overridden by the global sa_* value."""
    config = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        sa_block_q=512,
        local_sa_block_q=64,
        sa_block_kv=512,
        local_sa_block_kv=128,
        sa_block_kv_compute=512,
        local_sa_block_kv_compute=192,
        sa_block_q_dkv=512,
        local_sa_block_q_dkv=64,
        sa_block_kv_dkv=512,
        local_sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=512,
        local_sa_block_kv_dkv_compute=192,
        sa_block_q_dq=512,
        local_sa_block_q_dq=64,
        sa_block_kv_dq=512,
        local_sa_block_kv_dq=128,
        sa_use_fused_bwd_kernel=False,
        local_sa_use_fused_bwd_kernel=True,
        sa_q_layout="HEAD_DIM_MINOR",
        local_sa_q_layout="SEQ_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        local_sa_k_layout="SEQ_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        local_sa_v_layout="SEQ_MINOR",
        use_splash_scheduler=True,
        local_use_splash_scheduler=False,
    )
    self.assertEqual(config.local_sa_block_q, 64)
    self.assertEqual(config.local_sa_block_kv, 128)
    self.assertEqual(config.local_sa_block_kv_compute, 192)
    self.assertEqual(config.local_sa_block_q_dkv, 64)
    self.assertEqual(config.local_sa_block_kv_dkv, 128)
    self.assertEqual(config.local_sa_block_kv_dkv_compute, 192)
    self.assertEqual(config.local_sa_block_q_dq, 64)
    self.assertEqual(config.local_sa_block_kv_dq, 128)
    self.assertTrue(config.local_sa_use_fused_bwd_kernel)
    self.assertEqual(config.local_sa_q_layout, "SEQ_MINOR")
    self.assertEqual(config.local_sa_k_layout, "SEQ_MINOR")
    self.assertEqual(config.local_sa_v_layout, "SEQ_MINOR")
    self.assertFalse(config.local_use_splash_scheduler)

  def test_eval_start_step_config(self):
    """Verifies that eval_start_step defaults to 0 and can be overridden via pyconfig."""
    config_default = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
    )
    self.assertEqual(config_default.eval_start_step, 0)

    config_override = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        eval_start_step=50,
    )
    self.assertEqual(config_override.eval_start_step, 50)

  def test_eval_start_step_negative_raises_error(self):
    """Verifies that eval_start_step < 0 raises a validation error."""
    with self.assertRaises((ValueError, Exception)):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          eval_start_step=-1,
      )

  # ------------------------------------------------------------------
  # Tests for infer_cp_axes / infer_ep_axes and EP rank flag disabling
  # ------------------------------------------------------------------

  def test_normalize_axes_basics(self):
    """_normalize_axes handles None, str, list, and empty list."""
    self.assertEqual(_normalize_axes(None), ())
    self.assertEqual(_normalize_axes("expert"), ("expert",))
    self.assertEqual(_normalize_axes(["a", "b"]), ("a", "b"))
    self.assertEqual(_normalize_axes([]), ())

  def test_moe_sharding_strategy_mutual_exclusivity(self):
    """Ensure that shard_exp_on_fsdp, use_2d_fsdp_sharding, and shard_embed_moe_on_fsdp are mutually exclusive."""

    def init_config(**kwargs):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          **kwargs,
      )

    with self.assertRaisesRegex(ValueError, "Only one of shard_exp_on_fsdp"):
      init_config(shard_exp_on_fsdp=True, use_2d_fsdp_sharding=True)

    with self.assertRaisesRegex(ValueError, "Only one of shard_exp_on_fsdp"):
      init_config(shard_exp_on_fsdp=True, shard_embed_moe_on_fsdp=True)

    with self.assertRaisesRegex(ValueError, "Only one of shard_exp_on_fsdp"):
      init_config(use_2d_fsdp_sharding=True, shard_embed_moe_on_fsdp=True)

  def test_ep_rank_1_raises_on_ep_flags(self):
    """When EP rank is 1 (no EP rules), setting EP-only flags must raise ValueError."""
    # No 'exp' rule -> infer_ep_axes returns () -> EP rank is 1.
    rules_no_ep = [["activation_length", ["context"]]]
    self.assertEqual(infer_ep_axes(rules_no_ep), ())

    # Each flag that must be disabled when EP rank == 1.
    ep_disabled_flags = {
        "use_random_routing": (False, True),
        "use_ragged_sort": (False, True),
        "ragged_buffer_factor": (-1.0, 2.0),
        "use_ring_of_experts": (False, True),
        "num_moe_emb_chunks": (0, 2),
    }
    for flag_name, (_, bad_value) in ep_disabled_flags.items():
      with self.subTest(flag=flag_name):
        with self.assertRaises(ValueError, msg=f"{flag_name}={bad_value} should raise when EP rank is 1"):
          pyconfig.initialize(
              [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
              skip_jax_distributed_system=True,
              **{flag_name: bad_value},
          )

  def test_cp_as_ep_infer_axes(self):
    """cp-as-ep: exp -> ['context', 'expert'], so ici_context_parallelism contributes to EP rank."""
    cp_as_ep_rules = [
        ["exp", ["context", "expert"]],
        ["activation_length", ["context"]],
    ]
    self.assertEqual(infer_ep_axes(cp_as_ep_rules), ("context", "expert"))
    # CP still inferred from activation_length
    self.assertEqual(infer_cp_axes(cp_as_ep_rules), ("context",))

  def test_ep_as_cp_infer_axes(self):
    """ep-as-cp: activation_length -> ['expert'], exp -> 'expert'. Expert axis serves both CP and EP."""
    ep_as_cp_rules = [
        ["activation_length", ["expert"]],
        ["exp", "expert"],
    ]
    self.assertEqual(infer_cp_axes(ep_as_cp_rules), ("expert",))
    self.assertEqual(infer_ep_axes(ep_as_cp_rules), ("expert",))

  def test_shard_embed_moe_on_fsdp_requires_quantization(self):
    """Verifies that a ValueError is raised when shard_embed_moe_on_fsdp
    is used without fixed weight quantization calibration."""
    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          shard_embed_moe_on_fsdp=True,
          quantization="",
      )

    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
          skip_jax_distributed_system=True,
          shard_embed_moe_on_fsdp=True,
          quantization="int8",
          weight_quantization_calibration_method="absmax",
      )

    # This should not raise
    pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        shard_embed_moe_on_fsdp=True,
        quantization="int8",
        weight_quantization_calibration_method="fixed,-1,1",
    )


if __name__ == "__main__":
  unittest.main()
