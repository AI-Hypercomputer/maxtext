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

"""Unit tests for `from_pyconfig` bridge and `pyconfig.py` YAML-free fallback."""

import os
import pathlib
import types as pytypes
from unittest import mock

from absl.testing import absltest
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.m3.configs import pyconfig_bridge
from maxtext.m3.configs.pyconfig_bridge import ConfigMigrationError, from_pyconfig
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5ModelConfig
from maxtext.m3.models.granitemoe.granitemoe_config import GraniteMoeModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.models.qwen3_8_flash_next.qwen3_8_config import Qwen3_8FlashNextConfig
from maxtext.m3.models.qwen3_vl.qwen3_vl_config import Qwen3VLConfig

_BASE_YML = str(pathlib.Path(__file__).resolve().parents[3] / "src" / "maxtext" / "configs" / "base.yml")


class PyconfigBridgeTest(absltest.TestCase):
  """Tests `from_pyconfig` translation, fail-fast checks, and `pyconfig` registry fallback."""

  def setUp(self):
    super().setUp()
    self._tmpdir = self.create_tempdir().full_path

  def _init_cfg(self, *overrides: str):
    return pyconfig.initialize(
        [
            "",
            _BASE_YML,
            f"base_output_directory={self._tmpdir}",
            "run_name=bridge_test",
            "skip_jax_distributed_system=true",
            *overrides,
        ]
    )

  def test_all_registered_models_translate_and_fallback_matches_yaml(self):
    orig_exists = os.path.exists

    def _hide_model_yamls(path: str) -> bool:
      if str(path).endswith(".yml") and f"{os.sep}models{os.sep}" in str(path):
        return False
      return orig_exists(path)

    models_dir = pathlib.Path(_BASE_YML).parent / "models"
    for model_name in ("glm5.3-flash", "granite-3.0-1b-a400m", "qwen3.8-flash-next", "qwen3.8-flash-next-mini"):
      self.assertFalse((models_dir / f"{model_name}.yml").exists())

    cases: list[tuple[str, type, tuple[str, ...]]] = [
        ("qwen3-0.6b", Qwen3Config, ()),
        ("qwen3-vl-2b", Qwen3VLConfig, ("scan_layers=false",)),
        ("glm5.3-flash", Glm5ModelConfig, ()),
        ("granite-3.0-1b-a400m", GraniteMoeModelConfig, ()),
        ("qwen3.8-flash-next", Qwen3_8FlashNextConfig, ()),
        ("qwen3.8-flash-next-mini", Qwen3_8FlashNextConfig, ()),
    ]
    for model_name, expected_cls, extra in cases:
      with self.subTest(model=model_name):
        yml_cfg = self._init_cfg(f"model_name={model_name}", *extra)
        with mock.patch("maxtext.configs.pyconfig.os.path.exists", side_effect=_hide_model_yamls):
          fallback_cfg = self._init_cfg(f"model_name={model_name}", *extra)

        yml_dump = yml_cfg._pydantic_config.model_dump()  # pylint: disable=protected-access
        fb_dump = fallback_cfg._pydantic_config.model_dump()  # pylint: disable=protected-access
        # `dump_hlo_xla_flags` reads process env lazily inside MaxTextConfig validation.
        yml_dump.pop("dump_hlo_xla_flags", None)
        fb_dump.pop("dump_hlo_xla_flags", None)
        self.assertEqual(fb_dump, yml_dump)

        model_cfg, train_cfg = from_pyconfig(fallback_cfg)
        self.assertIs(type(model_cfg), expected_cls)
        self.assertEqual(from_pyconfig(yml_cfg), (model_cfg, train_cfg))
        # Direct `MaxTextConfig` input matches `HyperParameters` wrapper input.
        self.assertEqual(
            from_pyconfig(yml_cfg._pydantic_config),  # pylint: disable=protected-access
            (model_cfg, train_cfg),
        )
        self.assertEqual(train_cfg.eval_interval, -1)
        self.assertEqual(train_cfg.eval_steps, -1)
        self.assertEqual(train_cfg.mesh.shard_mode, "auto")

    # Verify unknown model warns when neither YAML nor registry contains it.
    with mock.patch("maxtext.configs.pyconfig.os.path.exists", side_effect=_hide_model_yamls):
      with self.assertLogs("maxtext.configs.pyconfig", level="WARNING") as logs:
        self._init_cfg("model_name=llama3-8b")
    self.assertTrue(any("llama3-8b" in msg for msg in logs.output))

    # Double-override guard still fires under the YAML-free fallback when `override_model_config=false`.
    with mock.patch("maxtext.configs.pyconfig.os.path.exists", side_effect=_hide_model_yamls):
      with self.assertRaisesRegex(ValueError, "vocab_size"):
        self._init_cfg("model_name=qwen3-0.6b", "vocab_size=256")

  def test_architecture_overrides_across_model_families(self):
    q3_cfg = self._init_cfg(
        "model_name=qwen3-0.6b",
        "override_model_config=true",
        "base_emb_dim=64",
        "base_mlp_dim=128",
        "base_num_decoder_layers=2",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "vocab_size=256",
        "final_logits_soft_cap=30.0",
        "dtype=float32",
        "weight_dtype=bfloat16",
    )
    q3_model, q3_train = from_pyconfig(q3_cfg)
    self.assertIsInstance(q3_model, Qwen3Config)
    self.assertEqual(q3_model.hidden_size, 64)
    self.assertEqual(q3_model.intermediate_size, 128)
    self.assertEqual(q3_model.num_hidden_layers, 2)
    self.assertEqual(q3_model.num_attention_heads, 4)
    self.assertEqual(q3_model.num_key_value_heads, 2)
    self.assertEqual(q3_model.head_dim, 16)
    self.assertEqual(q3_model.vocab_size, 256)
    self.assertEqual(q3_model.final_logits_soft_cap, 30.0)
    self.assertEqual(q3_model.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(q3_model.param_dtype, config_types.DType.BFLOAT16)
    self.assertEqual(q3_train.optimizer.mu_dtype, config_types.DType.BFLOAT16)

    # `final_logits_soft_cap=0.0` translates to `None` (disabled sentinel).
    q3_nocap = self._init_cfg("model_name=qwen3-0.6b", "final_logits_soft_cap=0.0")
    self.assertIsNone(from_pyconfig(q3_nocap)[0].final_logits_soft_cap)

    # GraniteMoE routes `num_experts` -> `num_local_experts` and validates derived `head_dim`.
    gm_cfg = self._init_cfg(
        "model_name=granite-3.0-1b-a400m",
        "override_model_config=true",
        "base_emb_dim=64",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=16",
        "base_mlp_dim=32",
        "base_moe_mlp_dim=32",
        "num_experts=4",
        "num_experts_per_tok=2",
    )
    gm_model, gm_train = from_pyconfig(gm_cfg)
    self.assertIsInstance(gm_model, GraniteMoeModelConfig)
    self.assertEqual(gm_model.num_local_experts, 4)
    self.assertEqual(gm_model.intermediate_size, 32)
    self.assertEqual(gm_model.head_dim, 16)
    self.assertEqual(gm_train.transforms.param_scan_axis, 0)

    bad_gm_head = self._init_cfg(
        "model_name=granite-3.0-1b-a400m",
        "override_model_config=true",
        "base_emb_dim=64",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=32",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "head_dim"):
      from_pyconfig(bad_gm_head)

    bad_gm_mlp = pytypes.SimpleNamespace(
        model_name="granite-3.0-1b-a400m",
        model_fields_set={"base_mlp_dim", "base_moe_mlp_dim"},
        base_mlp_dim=32,
        base_moe_mlp_dim=64,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "Conflicting overrides"):
      from_pyconfig(bad_gm_mlp)

    # Qwen3.8-Flash-Next routes `base_mlp_dim` -> `shared_expert_intermediate_size`,
    # `base_moe_mlp_dim` -> `moe_intermediate_size`, and `num_experts` -> `n_routed_experts`.
    q38_stub = pytypes.SimpleNamespace(
        model_name="qwen3.8-flash-next",
        model_fields_set={"base_mlp_dim", "base_moe_mlp_dim", "num_experts", "gdn_chunk_inverse"},
        base_mlp_dim=64,
        base_moe_mlp_dim=128,
        num_experts=16,
        gdn_chunk_inverse="solve",
        opt_type="adamw",
    )
    q38_model, _ = from_pyconfig(q38_stub)
    self.assertIsInstance(q38_model, Qwen3_8FlashNextConfig)
    self.assertEqual(q38_model.shared_expert_intermediate_size, 64)
    self.assertEqual(q38_model.moe_intermediate_size, 128)
    self.assertEqual(q38_model.n_routed_experts, 16)
    self.assertEqual(q38_model.gdn_chunk_inverse, "solve")

    # Dense model rejects MoE override.
    bad_dense = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set={"num_experts"},
        num_experts=8,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "num_experts"):
      from_pyconfig(bad_dense)

    # `-Instruct` suffix normalizes when passed via stub config.
    inst_stub = pytypes.SimpleNamespace(model_name="qwen3-0.6b-Instruct", model_fields_set=set(), opt_type="adamw")
    self.assertIsInstance(from_pyconfig(inst_stub)[0], Qwen3Config)

  def test_training_fields_renames_mesh_and_mode(self):
    cfg = self._init_cfg(
        "model_name=qwen3-0.6b",
        "steps=12",
        "max_target_length=64",
        "per_device_batch_size=1",
        "num_vocab_tiling=4",
        "init_weights_seed=99",
        "learning_rate=1e-3",
        "learning_rate_final_fraction=0.05",
        "freeze_vision_encoder_params=false",
        "checkpoint_todelete_full_path=/tmp/del",
        "shard_mode=explicit",
        "ici_fsdp_parallelism=2",
        "ici_tensor_parallelism=2",
        "scan_layers=false",
        "remat_policy=none",
        "quantization=int8",
        "use_qwix_quantization=true",
        "profiler=xplane",
    )
    _, train_cfg = from_pyconfig(cfg)
    self.assertEqual(train_cfg.steps, 12)
    self.assertEqual(train_cfg.max_target_length, 64)
    self.assertEqual(train_cfg.num_vocab_tiling, 4)
    self.assertEqual(train_cfg.seed, 99)
    self.assertAlmostEqual(train_cfg.optimizer.learning_rate, 1e-3)
    self.assertAlmostEqual(train_cfg.optimizer.cosine_learning_rate_final_fraction, 0.05)
    self.assertFalse(train_cfg.optimizer.freeze_vision_encoder)
    self.assertEqual(train_cfg.checkpoint.todelete_full_path, "/tmp/del")
    self.assertEqual(train_cfg.mesh.shard_mode, "explicit")
    self.assertEqual(train_cfg.mesh.ici, {"fsdp": 2, "tensor": 2})
    self.assertFalse(train_cfg.transforms.scan_layers)
    self.assertEqual(train_cfg.transforms.remat_policy, "none")
    self.assertEqual(train_cfg.transforms.quantization, config_types.QuantizationType.INT8)
    self.assertEqual(train_cfg.profiling.profiler, config_types.ProfilerType.XPLANE)

    # `compiled_trainstep_file` rejected in `mode='train'`, allowed in `mode='aot'`.
    aot_cfg = self._init_cfg("model_name=qwen3-0.6b", "compiled_trainstep_file=/tmp/step.bin")
    with self.assertRaisesRegex(ConfigMigrationError, "compiled_trainstep_file"):
      from_pyconfig(aot_cfg, mode="train")
    _, aot_train = from_pyconfig(aot_cfg, mode="aot")
    self.assertEqual(aot_train.aot.compiled_trainstep_file, "/tmp/step.bin")

    with self.assertRaisesRegex(ValueError, "mode"):
      from_pyconfig(cfg, mode="invalid")  # type: ignore[arg-type]

  def test_goodput_default_vs_explicit_override(self):
    # Outside test runners, `base.yml` enables goodput defaults passively; `from_pyconfig` must not fail.
    with mock.patch("maxtext.configs.pyconfig._is_running_in_test", return_value=False):
      prod_default_cfg = self._init_cfg("model_name=qwen3-0.6b")
    self.assertTrue(prod_default_cfg.enable_goodput_recording)
    self.assertTrue(prod_default_cfg.monitor_goodput)
    from_pyconfig(prod_default_cfg)

    # Explicitly disabling goodput succeeds cleanly.
    off_cfg = self._init_cfg("model_name=qwen3-0.6b", "enable_goodput_recording=false", "monitor_goodput=false")
    from_pyconfig(off_cfg)

    # Explicitly enabling goodput fails fast with `ConfigMigrationError`.
    on_cfg = self._init_cfg("model_name=qwen3-0.6b", "enable_goodput_recording=true")
    with self.assertRaisesRegex(ConfigMigrationError, "enable_goodput_recording=True"):
      from_pyconfig(on_cfg)

  def test_unsupported_flags_and_migration_errors(self):
    # Context / pipeline / ulysses / fsdp_transpose axes fail fast.
    for axis_flag in (
        "ici_context_parallelism",
        "dcn_context_parallelism",
        "ici_pipeline_parallelism",
        "ici_fsdp_transpose_parallelism",
    ):
      stub = pytypes.SimpleNamespace(
          model_name="qwen3-0.6b",
          model_fields_set=set(),
          opt_type="adamw",
          **{axis_flag: 2},
      )
      with self.assertRaisesRegex(ConfigMigrationError, axis_flag):
        from_pyconfig(stub)

    # Live non-goal scalar/bool/enum checks catch values even when absent from `model_fields_set`.
    scalar_stub = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        custom_mesh="1x1",
        moe_dropless_fallback="step",
        enable_diloco=True,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "custom_mesh"):
      from_pyconfig(scalar_stub)

    # Quantization without qwix rejected.
    bad_quant = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        quantization="int8",
        use_qwix_quantization=False,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "use_qwix_quantization=True"):
      from_pyconfig(bad_quant)

    # Unsupported optimizer, explicit unmapped flag, and unknown model name.
    bad_opt = pytypes.SimpleNamespace(model_name="qwen3-0.6b", model_fields_set=set(), opt_type="muon")
    with self.assertRaisesRegex(ConfigMigrationError, "opt_type"):
      from_pyconfig(bad_opt)

    bad_explicit = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set={"global_parameter_scale"},
        global_parameter_scale=2,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "global_parameter_scale"):
      from_pyconfig(bad_explicit)

    bad_model = pytypes.SimpleNamespace(model_name="llama3-8b", model_fields_set=set(), opt_type="adamw")
    with self.assertRaisesRegex(ConfigMigrationError, "Unknown m3 model"):
      from_pyconfig(bad_model)

    # Invalid `steps` and `num_vocab_tiling` indivisibility fail fast via `ConfigMigrationError`.
    bad_steps = pytypes.SimpleNamespace(model_name="qwen3-0.6b", model_fields_set=set(), steps=-5, opt_type="adamw")
    with self.assertRaisesRegex(ConfigMigrationError, "Invalid training config"):
      from_pyconfig(bad_steps)

    bad_tiling = pytypes.SimpleNamespace(
        model_name="qwen3-0.6b",
        model_fields_set=set(),
        per_device_batch_size=1.0,
        max_target_length=10,
        num_vocab_tiling=4,
        opt_type="adamw",
    )
    with self.assertRaisesRegex(ConfigMigrationError, "num_vocab_tiling"):
      from_pyconfig(bad_tiling)

  def test_audited_key_sets_match_maxtext_config_schema(self):
    mt_fields = set(config_types.MaxTextConfig.model_fields)
    # pylint: disable=protected-access
    self.assertEmpty(pyconfig_bridge._TRANSLATED_PYCONFIG_KEYS - mt_fields)
    self.assertEmpty(pyconfig_bridge._IGNORED_PYCONFIG_KEYS - mt_fields)
    self.assertEmpty(pyconfig_bridge._EXPLICIT_GOODPUT_FLAGS - mt_fields)
    self.assertEmpty(set(pyconfig_bridge._UNSUPPORTED_FLAGS) - mt_fields)
    self.assertEmpty(pyconfig_bridge._TRANSLATED_PYCONFIG_KEYS & pyconfig_bridge._IGNORED_PYCONFIG_KEYS)
    self.assertEmpty(pyconfig_bridge._TRANSLATED_PYCONFIG_KEYS & set(pyconfig_bridge._UNSUPPORTED_FLAGS))
    # pylint: enable=protected-access


if __name__ == "__main__":
  absltest.main()
