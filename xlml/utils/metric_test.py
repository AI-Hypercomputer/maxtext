# Copyright 2023 Google LLC
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

"""Tests for benchmark metric.py."""

import hashlib
import os
import sys
from typing import Iterable, Optional
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from xlml.apis import metric_config, gcp_config, test_config
from dags import composer_env
from xlml.utils import bigquery, composer, metric
import jsonlines
import tensorflow as tf
from dags.vm_resource import TpuVersion, RuntimeVersion


"""Tests for Benchmark metric.py."""


class BenchmarkMetricTest(parameterized.TestCase, absltest.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def generate_tb_file(self):
    temp_dir = self.get_tempdir()
    summary_writer = tf.summary.create_file_writer(temp_dir)

    with summary_writer.as_default():
      tf.summary.scalar("loss", 0.345, step=1)
      tf.summary.scalar("loss", 0.234, step=2)
      tf.summary.scalar("loss", 0.123, step=3)
      tf.summary.scalar("accuracy", 0.901, step=1)
      tf.summary.scalar("accuracy", 0.935, step=2)
      tf.summary.scalar("accuracy", 0.987, step=3)
      tf.summary.text("key1", "value1", step=1)
      tf.summary.text("key2", "value2", step=1)
      tf.summary.text("key3", "value3", step=1)
      tf.summary.image(
          "image",
          [
              tf.random.uniform(shape=[8, 8, 1]),
              tf.random.uniform(shape=[8, 8, 1]),
          ],
          step=2,
      )
      tf.summary.histogram("histogram", tf.random.uniform([100, 50]), step=2)
      summary_writer.flush()

    return os.path.join(temp_dir, os.listdir(temp_dir)[0])

  def assert_metric_and_dimension_equal(
      self, actual_metrics, expected_metrics, actual_metadata, expected_metadata
  ):
    for index in range(len(expected_metrics)):
      self.assertEqual(len(actual_metrics[index]), len(expected_metrics[index]))
      for ac, ex in zip(actual_metrics[index], expected_metrics[index]):
        self.assertEqual(ac.job_uuid, ex.job_uuid)
        self.assertEqual(ac.metric_key, ex.metric_key)
        self.assertAlmostEqual(ac.metric_value, ex.metric_value)

    for index in range(len(expected_metadata)):
      self.assertListEqual(actual_metadata[index], expected_metadata[index])

  @parameterized.named_parameters(
      ("default", "train_accuracy", None, None, True),
      ("inclusive_pattern", "train_accuracy", ["eval*"], None, False),
      ("exclusive_pattern", "train_accuracy", None, ["eval*"], True),
      ("both_patterns", "train_accuracy", ["train*"], ["train*"], False),
  )
  def test_is_valid_tag(
      self,
      tag: str,
      include_tag_patterns: Optional[Iterable[str]],
      exclude_tag_patterns: Optional[Iterable[str]],
      expected_value: bool,
  ):
    actual_value = metric.is_valid_tag(tag, include_tag_patterns, exclude_tag_patterns)
    self.assertEqual(actual_value, expected_value)

  def test_read_from_tb(self):
    path = self.generate_tb_file()
    actual_metric, actual_dimension = metric.read_from_tb(path, None, None)
    expected_metric = {
        "loss": [
            metric.TensorBoardScalar(0.345, 1),
            metric.TensorBoardScalar(0.234, 2),
            metric.TensorBoardScalar(0.123, 3),
        ],
        "accuracy": [
            metric.TensorBoardScalar(0.901, 1),
            metric.TensorBoardScalar(0.935, 2),
            metric.TensorBoardScalar(0.987, 3),
        ],
    }
    expected_dimension = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }

    self.assertEqual(actual_metric.keys(), expected_metric.keys())
    for key in actual_metric.keys():
      for index in range(3):
        self.assertAlmostEqual(
            actual_metric[key][index].metric_value,
            expected_metric[key][index].metric_value,
        )
        self.assertEqual(
            actual_metric[key][index].step, expected_metric[key][index].step
        )

    self.assertDictEqual(actual_dimension, expected_dimension)

  @parameterized.named_parameters(
      ("LAST", metric_config.AggregationStrategy.LAST, 5),
      ("AVERAGE", metric_config.AggregationStrategy.AVERAGE, 2.75),
      ("MEDIAN", metric_config.AggregationStrategy.MEDIAN, 2.5),
  )
  def test_aggregate_metrics(
      self, strategy: metric_config.AggregationStrategy, expected_value: float
  ):
    metrics = [
        metric.TensorBoardScalar(1.0, 1),
        metric.TensorBoardScalar(2.0, 2),
        metric.TensorBoardScalar(3.0, 3),
        metric.TensorBoardScalar(5.0, 4),
    ]

    actual_value = metric.aggregate_metrics(metrics, strategy)
    self.assertAlmostEqual(actual_value, expected_value)

  @mock.patch("xlml.utils.metric.download_object_from_gcs")
  def test_process_json_lines(self, download_object_from_gcs):
    path = "/tmp/ml-auto-solutions-metrics.jsonl"
    test_run1 = {
        "metrics": {"accuracy": 0.95, "MFU": 0.50},
        "dimensions": {"framework": "jax"},
    }
    test_run2 = {
        "metrics": {"accuracy": 0.97, "MFU": 0.45},
        "dimensions": {"framework": "tf"},
    }

    with jsonlines.open(path, mode="w") as writer:
      writer.write_all([test_run1, test_run2])

    base_id = "test_json_lines"
    actual_metrics, actual_metadata = metric.process_json_lines("test_json_lines", path)
    uuid_1 = hashlib.sha256(str(base_id + "0").encode("utf-8")).hexdigest()

    accuracy_metric_1 = bigquery.MetricHistoryRow(
        job_uuid=uuid_1, metric_key="accuracy", metric_value=0.95
    )

    mfu_metric_1 = bigquery.MetricHistoryRow(
        job_uuid=uuid_1, metric_key="MFU", metric_value=0.50
    )
    dimension_1 = bigquery.MetadataHistoryRow(
        job_uuid=uuid_1, metadata_key="framework", metadata_value="jax"
    )

    uuid_2 = hashlib.sha256(str(base_id + "1").encode("utf-8")).hexdigest()
    accuracy_metric_2 = bigquery.MetricHistoryRow(
        job_uuid=uuid_2, metric_key="accuracy", metric_value=0.97
    )
    mfu_metric_2 = bigquery.MetricHistoryRow(
        job_uuid=uuid_2, metric_key="MFU", metric_value=0.45
    )
    dimension_2 = bigquery.MetadataHistoryRow(
        job_uuid=uuid_2, metadata_key="framework", metadata_value="tf"
    )

    expected_metrics = [
        [accuracy_metric_1, mfu_metric_1],
        [accuracy_metric_2, mfu_metric_2],
    ]
    expected_metadata = [[dimension_1], [dimension_2]]

    self.assert_metric_and_dimension_equal(
        actual_metrics, expected_metrics, actual_metadata, expected_metadata
    )

  def test_process_tensorboard_summary(self):
    base_id = "test"
    summary_config = metric_config.SummaryConfig(
        file_location=self.generate_tb_file(),
        aggregation_strategy=metric_config.AggregationStrategy.LAST,
        include_tag_patterns=None,
        exclude_tag_patterns=None,
    )
    actual_metrics, actual_metadata = metric.process_tensorboard_summary(
        base_id, summary_config
    )

    uuid = hashlib.sha256(str(base_id + "0").encode("utf-8")).hexdigest()
    loss_metric = bigquery.MetricHistoryRow(
        job_uuid=uuid, metric_key="loss", metric_value=0.123
    )
    accuracy_metric = bigquery.MetricHistoryRow(
        job_uuid=uuid, metric_key="accuracy", metric_value=0.987
    )
    expected_metrics = [[loss_metric, accuracy_metric]]
    dimension_1 = bigquery.MetadataHistoryRow(
        job_uuid=uuid, metadata_key="key1", metadata_value="value1"
    )
    dimension_2 = bigquery.MetadataHistoryRow(
        job_uuid=uuid, metadata_key="key2", metadata_value="value2"
    )
    dimension_3 = bigquery.MetadataHistoryRow(
        job_uuid=uuid, metadata_key="key3", metadata_value="value3"
    )
    expected_metadata = [[dimension_1, dimension_2, dimension_3]]
    self.assert_metric_and_dimension_equal(
        actual_metrics, expected_metrics, actual_metadata, expected_metadata
    )

  @parameterized.named_parameters(
      ("empty", "", ""),
      (
          "raw_url",
          "manual__2023-08-07T21:03:49.181263+00:00",
          "manual__2023-08-07T21%3A03%3A49.181263%2B00%3A00",
      ),
  )
  def test_encode_url(self, raw_url, expected_value):
    actual_value = metric.encode_url(raw_url)
    self.assertEqual(actual_value, expected_value)

  def test_add_airflow_metadata(self):
    base_id = "test_run"
    uuid = hashlib.sha256(str(base_id + str(0)).encode("utf-8")).hexdigest()

    with mock.patch("xlml.utils.metric.get_current_context") as mock_context:
      mock_dag_id = mock.MagicMock()
      mock_dag_id.dag_id.return_value = "benchmark_test"
      mock_task_id = mock.MagicMock()
      mock_task_id.task_id.return_value = "post_process"

      mock_context.return_value = {
          "run_id": "manual__2023-08-07T21:03:49.181263+00:00",
          "prev_start_date_success": "2023-08-08",
          "dag_run": mock_dag_id,
          "task": mock_task_id,
      }

      with mock.patch.dict(
          os.environ,
          {
              "COMPOSER_LOCATION": "test_location",
              "COMPOSER_ENVIRONMENT": "test_env",
          },
      ) as mock_variable:
        with mock.patch.object(
            composer, "get_airflow_url", return_value="http://airflow"
        ) as mock_object:
          raw_meta = [
              [
                  bigquery.MetadataHistoryRow(
                      job_uuid=uuid,
                      metadata_key="framework",
                      metadata_value="jax",
                  )
              ]
          ]
          actual_value = metric.add_airflow_metadata(
              base_id,
              "test_project",
              raw_meta,
          )
          print("actual_value", actual_value)

          expected_value = raw_meta
          print("expected_value", expected_value)
          expected_value[0].append(
              bigquery.MetadataHistoryRow(
                  job_uuid=uuid,
                  metadata_key="run_id",
                  metadata_value="manual__2023-08-07T21:03:49.181263+00:00",
              )
          )
          expected_value[0].append(
              bigquery.MetadataHistoryRow(
                  job_uuid=uuid,
                  metadata_key="prev_start_date_success",
                  metadata_value="2023-08-08",
              )
          )
          expected_value[0].append(
              bigquery.MetadataHistoryRow(
                  job_uuid=uuid,
                  metadata_key="airflow_dag_run_link",
                  metadata_value="http://airflow/dags/benchmark_test/grid?dag_run_id=manual__2023-08-07T21%3A03%3A49.181263%2B00%3A00&task_id=post_process",
              )
          )

          self.assert_metric_and_dimension_equal([], [], actual_value, expected_value)

  def test_add_test_config_metadata(self):
    base_id = "test_run"
    uuid = hashlib.sha256(str(base_id + str(0)).encode("utf-8")).hexdigest()

    raw_meta = [
        [
            bigquery.MetadataHistoryRow(
                job_uuid=uuid,
                metadata_key="framework",
                metadata_value="jax",
            )
        ]
    ]
    task_test_config = test_config.TpuVmTest(
        test_config.Tpu(
            version=TpuVersion.V4,
            cores=8,
            runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
            network="default",
            subnetwork="default",
        ),
        test_name="test_name",
        set_up_cmds="set_up_cmds",
        run_model_cmds="run_model_cmds",
        time_out_in_min=60,
        task_owner="test_owner",
        num_slices=1,
    )

    task_gcp_config = gcp_config.GCPConfig(
        project_name="test_project",
        zone="tpu_zone",
        dataset_name="dataset_name",
        dataset_project="test_project",
        composer_project="test_project",
    )

    task_metric_config = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="test_file_location",
            aggregation_strategy=metric_config.AggregationStrategy.MEDIAN,
            use_regex_file_location=True,
        ),
    )

    actual_value = metric.add_test_config_metadata(
        base_id,
        task_test_config,
        task_gcp_config,
        task_metric_config,
        raw_meta,
    )
    print("actual_value", actual_value)

    expected_value = raw_meta
    print("expected_value", expected_value)
    expected_value[0].append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="accelerator",
            metadata_value="v4-8",
        )
    )
    expected_value[0].append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="project",
            metadata_value="test-project",
        )
    )
    expected_value[0].append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="num_slices",
            metadata_value="1",
        )
    )
    expected_value[0].append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="topology",
            metadata_value="1xv4-8",
        )
    )
    expected_value[0].append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="metric_aggregation_strategy",
            metadata_value="MEDIAN",
        )
    )
    self.assert_metric_and_dimension_equal([], [], actual_value, expected_value)

  def test_get_gcs_file_location_with_regex(self):
    with mock.patch("xlml.utils.metric.storage") as mock_storage:
      mock_gcs_client = mock_storage.Client.return_value

      expected_path = "path/to/events.out.tfevents.123"
      mock_blob_1 = mock.MagicMock()
      mock_blob_1.name = expected_path

      mock_blob_2 = mock.MagicMock()
      mock_blob_2.name = "path/to/events.out.tfevents.234"

      mock_gcs_client.list_blobs.return_value = [mock_blob_1, mock_blob_2]

      actual_value = metric.get_gcs_file_location_with_regex(
          "gs://my-bucket/path/to/events.out.tfevents.1*"
      )
      mock_storage.Client.assert_called_once()
      mock_gcs_client.list_blobs.assert_called_once()
      self.assertEqual(actual_value, f"gs://my-bucket/{expected_path}")


if __name__ == "__main__":
  absltest.main()
