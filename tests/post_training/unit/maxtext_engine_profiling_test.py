# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that the engine drives its profiler and labels its trace regions."""

import dataclasses

from absl.testing import absltest
from flax import struct
from typing import Any
from unittest.mock import patch
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import micro_step_profiler
from maxtext.utils import maxtext_utils
import pytest
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.post_training]


def _tiny_config(**overrides) -> pyconfig.HyperParameters:
  """MaxText config with a tiny model for testing."""
  argv = [
      "maxtext_engine_profiling_test.py",
      get_test_config_path("base.yml"),
      "model_name=default",
      "run_name=engine_profiling_test",
      "enable_checkpointing=False",
      "convert_checkpoint_if_possible=False",
      "enable_tensorboard=False",
      "skip_jax_distributed_system=True",
      "gradient_accumulation_steps=1",
      "vocab_size=8",
      "base_emb_dim=8",
      "base_mlp_dim=16",
      "base_num_decoder_layers=2",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "head_dim=4",
      "per_device_batch_size=1",
      "max_target_length=4",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


def _mesh(cfg: pyconfig.HyperParameters) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)


@struct.dataclass(frozen=True, kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  token_ids: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))
  token_mask: Any = dataclasses.field(default_factory=lambda: jnp.ones((2, 2)))


def _engine(cfg: pyconfig.HyperParameters, mesh: jax.sharding.Mesh) -> maxtext_engine.MaxTextTrainingEngine:
  """Creates a `MaxTextTrainingEngine` with a dummy loss function for testing."""
  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
  return engine.with_loss_fn(
      lambda *args, **kwargs: (
          abstract_engine.WeightedMetric(unreduced_sum=jnp.array(0.25), denominator=jnp.array(1.0)),
          {},
      )
  )


class MaxTextTrainingEngineProfilingTest(absltest.TestCase):
  """Tests that the engine drives its profiler and labels its trace regions."""

  def setUp(self):
    super().setUp()
    self.start_trace = patch("jax.profiler.start_trace").start()
    self.stop_trace = patch("jax.profiler.stop_trace").start()
    self.addCleanup(patch.stopall)

  def test_a_step_labels_every_region_with_its_step_number(self):
    """Per-call regions are only grouped into a step by the `step_num` they share."""
    cfg = _tiny_config()
    mesh = _mesh(cfg)
    with patch("jax.profiler.StepTraceAnnotation") as annotation, jax.set_mesh(mesh):
      engine = _engine(cfg, mesh)
      engine.fwd_bwd(payload=DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2))))
      engine.fwd_bwd(payload=DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2))))
      engine.update()

    regions = [(call.args[0], call.kwargs["step_num"]) for call in annotation.call_args_list]
    self.assertEqual(regions, [("fwd_bwd", 0), ("fwd_bwd", 0), ("update", 0)])

  def test_the_engine_opens_and_closes_the_configured_window(self):
    """`fwd_bwd` starts the profile and `update` stops it, without an outer loop."""
    cfg = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        profiler="xplane",
        skip_first_n_steps_for_profiler=1,
        profiler_steps=1,
    )
    mesh = _mesh(cfg)
    with jax.set_mesh(mesh):
      engine = _engine(cfg, mesh)
      for step in range(3):
        engine.fwd_bwd(payload=DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2))))
        self.assertEqual(
            self.start_trace.call_count,
            0 if step == 0 else 1,
            "the profile should be open only for step 1.",
        )
        engine.update()
      engine.close()

    # The window is step 1 alone: one step skipped, then one step profiled.
    self.start_trace.assert_called_once()
    self.stop_trace.assert_called_once()

  def test_close_rescues_a_profile_left_open_by_an_interrupted_step(self):
    cfg = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        profiler="xplane",
        skip_first_n_steps_for_profiler=0,
        profiler_steps=5,
    )
    mesh = _mesh(cfg)
    with jax.set_mesh(mesh):
      engine = _engine(cfg, mesh)
      engine.fwd_bwd(payload=DummyPayload(token_ids=jnp.ones((2, 2)), token_mask=jnp.ones((2, 2))))
      self.start_trace.assert_called_once()
      self.stop_trace.assert_not_called()

      engine.close()

    self.stop_trace.assert_called_once()

  def test_profiler_disabled_when_profiler_mode_is_not_xplane(self):
    cfg = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        skip_first_n_steps_for_profiler=0,
        profiler_steps=5,
    )
    self.assertEqual(cfg.profiler, types.ProfilerType.NONE)
    prof = micro_step_profiler.MicroStepProfiler(cfg)
    self.assertTrue(prof.do_not_profile)
    prof.maybe_activate(total_micro_steps=0, current_micro_step=0, train_step=0)
    self.start_trace.assert_not_called()

  def test_upload_all_profiler_results_enables_non_zero_process_profiling(self):
    cfg_host0_only = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        profiler="xplane",
        skip_first_n_steps_for_profiler=0,
        profiler_steps=1,
        upload_all_profiler_results=False,
    )
    cfg_all_hosts = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        profiler="xplane",
        skip_first_n_steps_for_profiler=0,
        profiler_steps=1,
        upload_all_profiler_results=True,
    )
    with patch("jax.process_index", return_value=1):
      prof_host0_only = micro_step_profiler.MicroStepProfiler(cfg_host0_only)
      self.assertTrue(prof_host0_only.do_not_profile)
      prof_host0_only.maybe_activate(total_micro_steps=0, current_micro_step=0, train_step=0)
      self.start_trace.assert_not_called()

      prof_all_hosts = micro_step_profiler.MicroStepProfiler(cfg_all_hosts)
      self.assertFalse(prof_all_hosts.do_not_profile)
      prof_all_hosts.maybe_activate(total_micro_steps=0, current_micro_step=0, train_step=0)
      self.start_trace.assert_called_once()
      prof_all_hosts.maybe_deactivate(total_micro_steps=0, current_micro_step=0, train_step=0)
      self.stop_trace.assert_called_once()

  def test_tpu_and_power_profiling_options_populated_in_advanced_configuration(self):
    cfg = _tiny_config(
        base_output_directory=self.create_tempdir().full_path,
        profiler="xplane",
        skip_first_n_steps_for_profiler=0,
        profiler_steps=1,
        enable_tpu_profiling_options=True,
        tpu_num_chips_to_profile_per_task=2,
        tpu_num_sparse_cores_to_trace=4,
        tpu_num_sparse_core_tiles_to_trace=8,
        profile_power_events=True,
        xprof_tpu_power_trace_level=1,
        xprof_e2e_enable_fw_throttle_event=True,
        xprof_e2e_enable_fw_power_level_event=True,
        xprof_e2e_enable_fw_thermal_event=True,
        enable_continuous_profiling=True,
    )
    with patch("jax.profiler.ProfileOptions") as mock_options_cls:
      prof = micro_step_profiler.MicroStepProfiler(cfg)
      mock_options_cls.assert_called_once()
      self.assertEqual(
          prof.profiling_options.advanced_configuration,
          {
              "tpu_power_trace_level": 1,
              "e2e_enable_fw_throttle_event": True,
              "e2e_enable_fw_power_level_event": True,
              "e2e_enable_fw_thermal_event": True,
              "tpu_num_chips_to_profile_per_task": 2,
              "tpu_num_sparse_core_tiles_to_trace": 8,
              "tpu_num_sparse_cores_to_trace": 4,
              "enable_continuous_profiling": True,
          },
      )
      self.assertIs(type(prof.profiling_options.advanced_configuration["tpu_power_trace_level"]), int)
      prof.maybe_activate(total_micro_steps=0, current_micro_step=0, train_step=0)
      self.start_trace.assert_called_once_with(
          cfg.tensorboard_dir,
          profiler_options=prof.profiling_options,
      )


if __name__ == "__main__":
  absltest.main()
