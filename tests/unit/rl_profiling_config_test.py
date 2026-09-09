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

"""Config-level tests for RL profiling and managed ML Diagnostics on `RLConfig`.

These build configs through the real `pyconfig` path (so the pydantic validators
and the unknown-key check run) and need no accelerator or network.
"""

import sys

from absl.testing import absltest
from absl.testing import parameterized
from maxtext.configs import pyconfig
from maxtext.configs import types
from tests.utils.test_helpers import get_test_config_path
import pydantic

# Tiny-model overrides that let RLConfig validate quickly on CPU.
_MODEL_OVERRIDES = {
    "enable_checkpointing": False,
    "base_num_decoder_layers": 1,
    "attention": "dot_product",
    "max_target_length": 8,
    "base_emb_dim": 128,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 256,
    "max_prefill_predict_length": 4,
    "model_name": "llama2-7b",
    "override_model_config": True,
    "weight_dtype": "bfloat16",
    "tokenizer_path": "meta-llama/Llama-2-7b",
}

# Step-window profiling flags that exist on the pretrain `Profiling` mixin and
# must be rejected by `RLConfig`.
_LEGACY_PROFILING_FLAGS = (
    "skip_first_n_steps_for_profiler=1",
    "profiler_steps=2",
    "profile_periodically_period=10",
    "profile_cleanly=false",
    "upload_all_profiler_results=true",
    "enable_continuous_profiling=true",
)


def _rl_config(*cli_args: str, **overrides):
  return pyconfig.initialize_pydantic(
      [sys.argv[0], get_test_config_path("post_train/rl.yml"), *cli_args],
      config_class=types.RLConfig,
      **{**_MODEL_OVERRIDES, **overrides},
  )


class RLProfilingConfigTest(parameterized.TestCase):

  def test_rl_yml_declares_invocation_window_defaults(self):
    cfg = _rl_config(run_name="rl_profiling_test")
    self.assertEqual(cfg.rl.profiler_start_invocation, 2)
    self.assertEqual(cfg.rl.profiler_num_invocations, 1)
    self.assertEqual(cfg.rl.profiler_timeout_secs, 60.0)
    self.assertEqual(cfg.profiler, types.ProfilerType.NONE)

  def test_cli_overrides_reach_nested_rl_block(self):
    cfg = _rl_config(
        "rl.profiler_start_invocation=5",
        "rl.profiler_num_invocations=3",
        "rl.profiler_timeout_secs=120",
        "profiler=xplane",
        run_name="rl_profiling_test",
    )
    self.assertEqual(cfg.rl.profiler_start_invocation, 5)
    self.assertEqual(cfg.rl.profiler_num_invocations, 3)
    self.assertEqual(cfg.rl.profiler_timeout_secs, 120.0)
    self.assertEqual(cfg.profiler, types.ProfilerType.XPLANE)

  @parameterized.parameters(
      {"rl": {"profiler_start_invocation": -1}},
      {"rl": {"profiler_num_invocations": 0}},
      {"rl": {"profiler_timeout_secs": 0}},
  )
  def test_invocation_window_bounds(self, **bad):
    with self.assertRaises(pydantic.ValidationError):
      types.RLConfig(model_name="llama2-7b", tokenizer_path="meta-llama/Llama-2-7b", **bad)

  @parameterized.parameters(*_LEGACY_PROFILING_FLAGS)
  def test_legacy_step_profiling_flags_are_rejected_by_rl_config(self, flag):
    key = flag.split("=", 1)[0]
    with self.assertRaisesRegex(ValueError, key):
      _rl_config(flag, run_name="rl_profiling_test")

  def test_legacy_step_profiling_flags_still_exist_on_pretrain_config(self):
    for flag in _LEGACY_PROFILING_FLAGS:
      self.assertIn(flag.split("=", 1)[0], types.MaxTextConfig.model_fields)
    self.assertIn("profiler", types.RLConfig.model_fields)
    self.assertNotIn("rl", types.MaxTextConfig.model_fields)

  def test_managed_mldiagnostics_dir_is_derived_without_trailing_slash(self):
    cfg = _rl_config(run_name="run1", base_output_directory="gs://bucket/out")
    self.assertEqual(
        cfg.managed_mldiagnostics_dir,
        "gs://bucket/out/run1/managed-mldiagnostics",
    )
    cfg = _rl_config(
        run_name="run1",
        base_output_directory="gs://bucket/out",
        managed_mldiagnostics_storage_path="gs://telemetry",
    )
    self.assertEqual(
        cfg.managed_mldiagnostics_dir,
        "gs://telemetry/run1/managed-mldiagnostics",
    )
    self.assertEqual(
        cfg.managed_mldiagnostics_dir,
        types.derive_managed_mldiagnostics_dir("gs://bucket/out", "run1", "gs://telemetry"),
    )

  def test_managed_mldiagnostics_dir_matches_pretrain_derivation(self):
    pretrain = pyconfig.initialize_pydantic(
        [sys.argv[0], get_test_config_path("base.yml")],
        run_name="run1",
        base_output_directory="gs://bucket/out",
        **_MODEL_OVERRIDES,
    )
    rl = _rl_config(run_name="run1", base_output_directory="gs://bucket/out")
    self.assertEqual(pretrain.managed_mldiagnostics_dir, rl.managed_mldiagnostics_dir)

  def test_managed_requires_gcs_run_directory(self):
    with self.assertRaisesRegex(ValueError, "gs://"):
      _rl_config(run_name="run1", managed_mldiagnostics=True)  # no base_output_directory
    with self.assertRaisesRegex(ValueError, "gs://"):
      _rl_config(
          run_name="run1",
          base_output_directory="/tmp/local",
          managed_mldiagnostics=True,
      )
    with self.assertRaisesRegex(ValueError, "gs://"):
      _rl_config(
          run_name="",
          base_output_directory="gs://bucket/out",
          managed_mldiagnostics=True,
      )

  def test_managed_xplane_rejects_on_demand_profiling(self):
    with self.assertRaisesRegex(ValueError, "on_demand_profiling"):
      _rl_config(
          "profiler=xplane",
          run_name="run1",
          base_output_directory="gs://bucket/out",
          managed_mldiagnostics=True,
          managed_mldiagnostics_on_demand_profiling=True,
      )
    cfg = _rl_config(
        "profiler=xplane",
        run_name="run1",
        base_output_directory="gs://bucket/out",
        managed_mldiagnostics=True,
        managed_mldiagnostics_on_demand_profiling=False,
    )
    self.assertTrue(cfg.managed_mldiagnostics)
    # Without xplane, on-demand profiling remains allowed.
    cfg = _rl_config(
        run_name="run1",
        base_output_directory="gs://bucket/out",
        managed_mldiagnostics=True,
        managed_mldiagnostics_on_demand_profiling=True,
    )
    self.assertTrue(cfg.managed_mldiagnostics_on_demand_profiling)

  def test_unmanaged_config_is_not_validated_for_gcs(self):
    cfg = _rl_config(
        run_name="run1",
        base_output_directory="/tmp/local",
        managed_mldiagnostics=False,
    )
    self.assertEqual(cfg.managed_mldiagnostics_dir, "/tmp/local/run1/managed-mldiagnostics")


if __name__ == "__main__":
  absltest.main()
