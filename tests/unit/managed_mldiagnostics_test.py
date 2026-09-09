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

"""Unit tests for ManagedMLDiagnostics."""

import json
from unittest import mock

from absl.testing import absltest
from maxtext.common import managed_mldiagnostics
from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics, mldiag
from maxtext.configs import types


class ManagedMLDiagnosticsTest(absltest.TestCase):
  # pylint: disable=protected-access

  def setUp(self):
    super().setUp()
    # Reset singleton instance between tests
    ManagedMLDiagnostics._instance = None

  def test_not_enabled_noop(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = False

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_not_called()

  def test_enabled_empty_region_passes_none(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {"key1": "val1"}

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1"},
          gcs_path="gs://test_dir",
          on_demand_xprof=False,
          region=None,
      )

  def test_enabled_populated_region_passes_region(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = "us-east1"
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {"key1": "val1"}

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1"},
          gcs_path="gs://test_dir",
          on_demand_xprof=False,
          region="us-east1",
      )

  def test_sampler_config_merging_differing_keys(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {"key1": "val1", "num_slices": 1}

    mock_sampler_config = mock.MagicMock()
    mock_sampler_config.get_keys.return_value = {
        "key1": "val1",
        "num_slices": 2,
    }

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config, sampler_config=mock_sampler_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1", "num_slices": 1, "sampler.num_slices": 2},
          gcs_path="gs://test_dir",
          on_demand_xprof=False,
          region=None,
      )

  def test_sampler_config_same_object_no_prefix(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {"key1": "val1"}

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config, sampler_config=mock_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1"},
          gcs_path="gs://test_dir",
          on_demand_xprof=False,
          region=None,
      )

  def test_stub_sdk_warns_and_still_no_ops(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.get_keys.return_value = {}

    with (
        mock.patch.object(managed_mldiagnostics, "_mldiag_is_stub", True),
        mock.patch.object(managed_mldiagnostics.max_logging, "warning") as mock_warning,
        mock.patch.object(mldiag, "machinelearning_run") as mock_run,
    ):
      self.assertFalse(managed_mldiagnostics.mldiagnostics_available())
      ManagedMLDiagnostics(mock_config)
      mock_warning.assert_called_once()
      self.assertIn("not available", mock_warning.call_args.args[0])
      # The stub's machinelearning_run is a no-op; pretrain behavior is unchanged.
      mock_run.assert_called_once()

  def test_mldiagnostics_available_when_real_sdk_loaded(self):
    with mock.patch.object(managed_mldiagnostics, "_mldiag_is_stub", False):
      self.assertTrue(managed_mldiagnostics.mldiagnostics_available())

  def test_gcs_path_is_passed_through_unchanged(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = ""
    mock_config.managed_mldiagnostics_dir = "gs://bucket/run/managed-mldiagnostics"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {}

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      self.assertEqual(
          mock_run.call_args.kwargs["gcs_path"],
          "gs://bucket/run/managed-mldiagnostics",
      )

  def test_sampler_keys_skipped_when_not_json_serializable(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = ""
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.managed_mldiagnostics_on_demand_profiling = False
    mock_config.get_keys.return_value = {"a": 1}

    mock_sampler_config = mock.MagicMock()
    mock_sampler_config.get_keys.return_value = {
        "a": 1,
        "b": object(),
        "c": float("nan"),
        "d": 2,
    }

    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config, sampler_config=mock_sampler_config)
      self.assertEqual(mock_run.call_args.kwargs["configs"], {"a": 1, "sampler.d": 2})

  def test_config_with_get_keys(self):
    class PydanticConfigMock:
      managed_mldiagnostics = True
      managed_mldiagnostics_region = ""
      run_name = "pydantic_run"
      managed_mldiagnostics_run_group = "pydantic_group"
      managed_mldiagnostics_dir = "gs://pydantic_dir"
      managed_mldiagnostics_on_demand_profiling = False

      def get_keys(self):
        return {"pydantic_key": "pydantic_val", "epochs": 10}

    mock_config = PydanticConfigMock()
    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_called_once_with(
          name="pydantic_run",
          run_group="pydantic_group",
          configs={"pydantic_key": "pydantic_val", "epochs": 10},
          gcs_path="gs://pydantic_dir",
          on_demand_xprof=False,
          region=None,
      )

  def test_pydantic_rl_config_without_get_keys(self):
    """The RL trainer passes pydantic `RLConfig` objects (initialize_pydantic), which have no get_keys()."""
    trainer = types.RLConfig(
        model_name="llama2-7b",
        tokenizer_path="meta-llama/Llama-2-7b",
        run_name="rl_run",
        base_output_directory="gs://bucket/out",
        managed_mldiagnostics=True,
        managed_mldiagnostics_on_demand_profiling=False,
        managed_mldiagnostics_region="us-central1",
    )
    sampler = trainer.model_copy(update={"debug": not trainer.debug})
    with mock.patch.object(mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(trainer, sampler_config=sampler)
    kwargs = mock_run.call_args.kwargs
    self.assertEqual(kwargs["name"], "rl_run")
    self.assertEqual(kwargs["gcs_path"], "gs://bucket/out/rl_run/managed-mldiagnostics")
    self.assertEqual(kwargs["region"], "us-central1")
    self.assertFalse(kwargs["on_demand_xprof"])
    self.assertEqual(kwargs["configs"]["model_name"], "llama2-7b")
    self.assertEqual(kwargs["configs"]["sampler.debug"], sampler.debug)
    self.assertNotIn("sampler.model_name", kwargs["configs"])
    json.dumps(kwargs["configs"], allow_nan=False)  # everything uploaded is JSON-serializable


if __name__ == "__main__":
  absltest.main()
