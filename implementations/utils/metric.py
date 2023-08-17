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


"""Utilities to process Benchmark metrics."""

import dataclasses
import datetime
import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional
import uuid
from absl import logging
from airflow.decorators import task
from airflow.operators.python import get_current_context
from apis import gcp_config, test_config
from apis import metric_config
from implementations.utils import bigquery
from implementations.utils import composer
import jsonlines
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2


@dataclasses.dataclass
class TensorBoardScalar:
  metric_value: float
  step: int


def is_valid_tag(
    tag: str,
    include_tag_patterns: Optional[Iterable[str]],
    exclude_tag_patterns: Optional[Iterable[str]],
) -> bool:
  """Check if it is a valid tag.

  Args:
    tag: The tag to check.
    include_tag_patterns: A list of patterns should be included.
    exclude_tag_patterns: A list of patterns should be excluded. This pattern
      has higher priority to include_tag_pattern, if any conflict.

  Returns:
    A bool to indicate if this tag should be included.
  """
  if exclude_tag_patterns and any(
      re.match(x, tag) for x in exclude_tag_patterns
  ):
    # check if tag in exclude_tag_patterns
    return False
  if include_tag_patterns:
    # check if tag in include_tag_patterns
    return any(re.match(x, tag) for x in include_tag_patterns)
  return True


def read_from_tb(
    file_location: str,
    include_tag_patterns: Optional[Iterable[str]],
    exclude_tag_patterns: Optional[Iterable[str]],
) -> (Dict[str, List[TensorBoardScalar]], Dict[str, str]):
  """Read metrics and dimensions from TensorBoard file.

  Args:
    file_location: The full path of a file in GCS.
    include_tag_patterns: The matching pattern of tags that wil be included.
    exclude_tag_patterns: The matching pattern of tags that will be excluded.
      This pattern has higher priority to include_tag_pattern, if any conflict.

  Returns:
    A dict that maps metric name to a list of TensorBoardScalar, and
    a dict that maps dimension name to dimenstion value.
  """
  metrics = {}
  metadata = {}

  serialized_examples = tf.data.TFRecordDataset(file_location)
  logging.info(f"TensorBoard metric_location is: {file_location}")
  for ex in serialized_examples:
    event = event_pb2.Event.FromString(ex.numpy())
    for value in event.summary.value:
      if not is_valid_tag(
          value.tag, include_tag_patterns, exclude_tag_patterns
      ):
        continue
      value_type = value.metadata.plugin_data.plugin_name
      if value_type == "scalars":
        if value.tag not in metrics:
          metrics[value.tag] = []
        t = tf.make_ndarray(value.tensor)
        metrics[value.tag].append(TensorBoardScalar(float(t), event.step))
      elif value_type == "text":
        metadata[value.tag] = bytes(value.tensor.string_val[0]).decode("utf-8")
      else:
        logging.info(
            f"Discarding data point {value.tag} with type {value_type}."
        )

  return metrics, metadata


def aggregate_metrics(
    metrics: Iterable[TensorBoardScalar],
    strategy: metric_config.AggregationStrategy,
) -> float:
  """Get the aggregated value based on stragety.

  Args:
    metrics: The TensorBoardScalar from TensorBoard file.
    strategy: The strategy for aggregate values.

  Returns:
    A value after aggregation.
  """
  if strategy == metric_config.AggregationStrategy.LAST:
    last_value = max(metrics, key=lambda p: p.step)
    return last_value.metric_value
  elif strategy == metric_config.AggregationStrategy.AVERAGE:
    return np.mean([m.metric_value for m in metrics])
  elif strategy == metric_config.AggregationStrategy.MEDIAN:
    return np.median([m.metric_value for m in metrics])
  else:
    raise NotImplementedError(f"Unknown aggregation strategy: {strategy}")


def process_json_lines(
    base_id: str,
    file_location: str,
) -> (
    List[List[bigquery.MetricHistoryRow]],
    List[List[bigquery.MetadataHistoryRow]],
):
  """Process metrics and dimensions from JSON Lines file.

  Args:
    base_id: The unique ID for this test job.
    file_location: The full path of a file in GCS.

  Returns:
    A list of MetricHistoryRow for all test runs, and
    a list of MetadataHistoryRow ofr all test runs in a test job.
  """
  metric_list = []
  metadata_list = []
  with jsonlines.open(file_location) as reader:
    index = 0
    for object in reader:
      uuid = generate_row_uuid(base_id, index)
      index += 1
      raw_metrics = object["metrics"]
      metadata = object["dimensions"]
      metric_history_rows = []
      metadata_history_rows = []

      for key, value in raw_metrics.items():
        metric_history_rows.append(
            bigquery.MetricHistoryRow(
                job_uuid=uuid, metric_key=key, metric_value=value
            )
        )

      for key, value in metadata.items():
        metadata_history_rows.append(
            bigquery.MetadataHistoryRow(
                job_uuid=uuid, metadata_key=key, metadata_value=value
            )
        )

      metric_list.append(metric_history_rows)
      metadata_list.append(metadata_history_rows)

    return metric_list, metadata_list


def process_tensorboard_summary(
    base_id: str,
    summary_config: metric_config.SummaryConfig,
) -> (
    List[List[bigquery.MetricHistoryRow]],
    List[List[bigquery.MetadataHistoryRow]],
):
  """Process metrics and dimensions from TensorBoard file.

  Args:
    base_id: The unique ID for this test job.
    summary_config: The configs for TensorBoard summary.

  Returns:
    A list of MetricHistoryRow for a test run, and
    a list of MetadataHistoryRow ofr a test run in a test job.
  """
  uuid = generate_row_uuid(base_id, 0)
  file_location = summary_config.file_location
  aggregation_strategy = summary_config.aggregation_strategy
  include_tag_patterns = summary_config.include_tag_patterns
  exclude_tag_patterns = summary_config.exclude_tag_patterns

  metrics, metadata = read_from_tb(
      file_location, include_tag_patterns, exclude_tag_patterns
  )
  aggregated_metrics = {}
  for key, value in metrics.items():
    aggregated_metrics[key] = aggregate_metrics(value, aggregation_strategy)
  print("aggregated_metrics", aggregated_metrics)

  metric_history_rows = []
  metadata_history_rows = []

  for key, value in aggregated_metrics.items():
    metric_history_rows.append(
        bigquery.MetricHistoryRow(
            job_uuid=uuid, metric_key=key, metric_value=value
        )
    )

  for key, value in metadata.items():
    metadata_history_rows.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key=key, metadata_value=value
        )
    )

  return [metric_history_rows], [metadata_history_rows]


# TODO(qinwen): implement profile metrics & upload to Vertex AI TensorBoard
def process_profile(
    uuid: str, file_location: str
) -> List[List[bigquery.MetricHistoryRow]]:
  raise NotImplementedError


def encode_url(url: str) -> str:
  """Replace characters with % followed by two hexadecimal digits.

  Args:
    url: The url to be encoded.

  Returns:
    An encoded url.
  """
  return str(url).replace(":", "%3A").replace("+", "%2B")


def add_airflow_metadata(
    base_id: str,
    project_name: str,
    metadata: List[List[bigquery.MetricHistoryRow]],
) -> List[List[bigquery.MetricHistoryRow]]:
  """Add airflow metadata: run_id, prev_start_date_success, and airflow_dag_run_link.

  Args:
    base_id: The base id to generate uuid.
    metadata: The data to append airflow metadata.
    configs: The GCP configs to get composer metadata.

  Returns:
    The data with airflow metadata.
  """
  context = get_current_context()
  run_id = context["run_id"]
  prev_start_date_success = str(context["prev_start_date_success"])
  dag_run = context["dag_run"]
  dag_id = dag_run.dag_id
  task_id = context["task"].task_id
  dag_run_id = encode_url(run_id)
  airflow_link = composer.get_airflow_url(
      project_name,
      os.environ.get("COMPOSER_LOCATION"),
      os.environ.get("COMPOSER_ENVIRONMENT"),
  )
  airflow_dag_run_link = f"{airflow_link}/dags/{dag_id}/grid?dag_run_id={dag_run_id}&task_id={task_id}"
  logging.info(f"airflow_dag_run_link is {airflow_dag_run_link}")

  # append airflow metadata for each test run.
  for index in range(len(metadata)):
    uuid = generate_row_uuid(base_id, index)
    airflow_meta = []

    airflow_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key="run_id", metadata_value=run_id
        )
    )
    if context["prev_start_date_success"]:
      airflow_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="prev_start_date_success",
              metadata_value=prev_start_date_success,
          )
      )
    airflow_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="airflow_dag_run_link",
            metadata_value=airflow_dag_run_link,
        )
    )

    metadata[index].extend(airflow_meta)
  return metadata


def generate_row_uuid(base_id: str, index: int) -> str:
  """Generate uuid for entry.

  Args:
    base_id: The process id generated once per post process task group.
    index: The index of test runs.

  Returns:
    A uuid for table entry.
  """
  return hashlib.sha256(str(base_id + str(index)).encode("utf-8")).hexdigest()


@task
def generate_process_id() -> str:
  """Generate a process id that will be a base id for uuid of test runs.

  Returns:
    A random uuid.
  """
  return str(uuid.uuid4())


# TODO(ranran):
# 1) handle job status
# 2) handle Airflow retry to avoid duplicate records in tables
@task
def process_metrics(
    base_id: str,
    task_test_config: test_config.TestConfig[test_config.Tpu],
    task_metric_config: metric_config.MetricConfig,
    task_gcp_config: gcp_config.GCPConfig,
) -> None:
  benchmark_id = task_test_config.benchmark_id
  current_time = datetime.datetime.now()
  has_profile = False
  metric_history_rows_list = [[]]
  metadata_history_rows_list = [[]]
  profile_history_rows_list = []

  # process metrics, metadata, and profile
  if task_metric_config is not None:
    if task_metric_config.json_lines:
      metric_history_rows_list, metadata_history_rows_list = process_json_lines(
          base_id, task_metric_config.json_lines.file_location
      )
    if task_metric_config.tensorboard_summary:
      metric_history_rows_list, metadata_history_rows_list = (
          process_tensorboard_summary(
              base_id, task_metric_config.tensorboard_summary
          )
      )
    if task_metric_config.profile:
      has_profile = True
      num_profiles = len(task_metric_config.profile.file_locations)
      for index in range(num_profiles):
        profile_history_rows = process_profile(
            base_id, task_metric_config.profile.file_locations[index]
        )
        profile_history_rows_list.append(profile_history_rows)

  # add default airflow metadata
  metadata_history_rows_list = add_airflow_metadata(
      base_id, task_gcp_config.project_name, metadata_history_rows_list
  )

  # append profile metrics to metric_history_rows_list if any
  if has_profile:
    if len(metric_history_rows_list) != len(profile_history_rows_list):
      logging.error(
          f"The num of profile is {len(profile_history_rows_list)}, but it is"
          " different to the number of test runs"
          f" {len(metric_history_rows_list)}. Ignoring profiles."
      )
    else:
      for index in range(len(metric_history_rows_list)):
        metric_history_rows_list[index].extend(profile_history_rows_list[index])

  test_run_rows = []
  bigquery_metric = bigquery.BigQueryMetricClient(
      task_gcp_config.project_name, task_gcp_config.dataset_name.value
  )

  for index in range(len(metadata_history_rows_list)):
    job_history_row = bigquery.JobHistoryRow(
        uuid=generate_row_uuid(base_id, index),
        timestamp=current_time,
        owner=task_test_config.task_owner,
        job_name=benchmark_id,
        job_status=bigquery.JobStatus.SUCCESS.value,
    )
    test_run_row = bigquery.TestRun(
        job_history_row,
        metric_history_rows_list[index],
        metadata_history_rows_list[index],
    )
    test_run_rows.append(test_run_row)

  print("Test run rows:", test_run_rows)

  # if it's a manual run, no entries are inserted into tables
  context = get_current_context()
  run_id = context["run_id"]
  if run_id.startswith("manual"):
    logging.info(
        "This is a manual run, and no entries are inserted into tables."
    )
    return

  bigquery_metric.insert(test_run_rows)
