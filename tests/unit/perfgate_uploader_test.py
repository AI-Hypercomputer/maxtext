# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for perfgate_uploader."""

from unittest import mock

from absl.testing import absltest
from maxtext import perfgate_uploader

from google3.testing.performance.perfgate.helpers.py.quickstore import quickstore


class PerfgateUploaderTest(absltest.TestCase):

  @mock.patch.object(quickstore, "Quickstore")
  @mock.patch("google3.testing.performance.perfgate.helpers.py.build_cl.build_cl.ApplyBuildCLOnQuickstoreInput")
  def test_upload_metrics(self, mock_apply_build_cl, mock_quickstore):
    perfgate_benchmark_key = "test_benchmark_key"
    perfgate_tags = ["tag1", "tag2"]
    perfgate_metric_key = "test_metric_key"
    step_times = [2.5] * 40

    mock_quickstore_instance = mock_quickstore.return_value

    perfgate_uploader.upload_metrics(
        step_times,
        perfgate_benchmark_key,
        perfgate_tags,
        perfgate_metric_key,
    )

    # Verify Quickstore was initialized with correct args
    mock_quickstore.assert_called_once()
    _, kwargs = mock_quickstore.call_args
    self.assertEqual(kwargs["benchmark_key"], perfgate_benchmark_key)
    self.assertEqual(kwargs["quickstore_input"].tags, perfgate_tags)

    # Verify AddSamplePoint was called for each step
    self.assertEqual(mock_quickstore_instance.AddSamplePoint.call_count, 40)
    expected_calls = [mock.call(i, {perfgate_metric_key: 2.5}) for i in range(40)]
    mock_quickstore_instance.AddSamplePoint.assert_has_calls(expected_calls)

    # Verify Store was called
    mock_quickstore_instance.Store.assert_called_once()

    # Verify ApplyBuildCLOnQuickstoreInput was called
    mock_apply_build_cl.assert_called_once()

  def test_upload_metrics_no_data(self):
    perfgate_uploader.upload_metrics([], "key", [], "metric")
    # Should just return without error
    pass


if __name__ == "__main__":
  absltest.main()
