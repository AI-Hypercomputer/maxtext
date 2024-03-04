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
import enum
import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional
import uuid
from absl import logging
import airflow
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import TaskInstance
from airflow.operators.python import get_current_context
from xlml.apis import gcp_config, test_config
from xlml.apis import metric_config
from dags import composer_env
from google.cloud import storage
from xlml.utils import bigquery, composer
import jsonlines
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2
from urllib.parse import urlparse


@dataclasses.dataclass
class TensorBoardScalar:
  metric_value: float
  step: int


class TaskState(enum.Enum):
  FAILED = "failed"
  SKIPPED = "upstream_failed"
  SUCCESS = "success"


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
  if exclude_tag_patterns and any(re.match(x, tag) for x in exclude_tag_patterns):
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
      if not is_valid_tag(value.tag, include_tag_patterns, exclude_tag_patterns):
        continue
      value_type = value.metadata.plugin_data.plugin_name
      if value_type == "scalars":
        if value.tag not in metrics:
          metrics[value.tag] = []
        t = tf.make_ndarray(value.tensor)
        metrics[value.tag].append(TensorBoardScalar(float(t), event.step))
      elif value_type == "text":
        metadata[value.tag] = bytes(value.tensor.string_val[0]).decode("utf-8")
      elif value.HasField("simple_value"):
        # simple_value indicates the value is a float:
        # https://github.com/tensorflow/tensorflow/blob/4dacf3f/tensorflow/core/framework/summary.proto#L122
        scalar = TensorBoardScalar(value.simple_value, event.step)
        metrics.setdefault(value.tag, []).append(scalar)
      else:
        logging.info(f"Discarding data point {value.tag} with type {value_type}.")

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


def download_object_from_gcs(source_location: str, destination_location: str) -> None:
  """Download object from GCS bucket.

  Args:
    source_location: The full path of a file in GCS.
    destination_location: The local path of the file.
  """

  storage_client = storage.Client()
  bucket_name = source_location.split("/")[2]
  object_name = "/".join(source_location.split("/")[3:])

  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(object_name)
  blob.download_to_filename(destination_location)
  logging.info(
      f"Download JSON Lines file from {source_location} to {destination_location}"
  )


def process_json_lines(
    base_id: str,
    file_location: str,
) -> (List[List[bigquery.MetricHistoryRow]], List[List[bigquery.MetadataHistoryRow]],):
  """Process metrics and dimensions from JSON Lines file.

  Args:
    base_id: The unique ID for this test job.
    file_location: The full path of a file in GCS.

  Returns:
    A list of MetricHistoryRow for all test runs, and
    a list of MetadataHistoryRow ofr all test runs in a test job.
  """

  tmp_location = "/tmp/ml-auto-solutions-metrics.jsonl"
  download_object_from_gcs(file_location, tmp_location)
  metric_list = []
  metadata_list = []

  with jsonlines.open(tmp_location) as reader:
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
            bigquery.MetricHistoryRow(job_uuid=uuid, metric_key=key, metric_value=value)
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
) -> (List[List[bigquery.MetricHistoryRow]], List[List[bigquery.MetadataHistoryRow]],):
  """Process metrics and dimensions from TensorBoard file.

  Args:
    base_id: The unique ID for this test job.
    summary_config: The configs for TensorBoard summary.

  Returns:
    A list of MetricHistoryRow for a test run, and
    a list of MetadataHistoryRow ofr a test run in a test job.
  """
  uuid = generate_row_uuid(base_id, 0)

  if isinstance(summary_config.file_location, airflow.XComArg):
    file_location = summary_config.file_location.resolve(get_current_context())
  else:
    file_location = summary_config.file_location

  if summary_config.use_regex_file_location:
    file_location = get_gcs_file_location_with_regex(file_location)

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
        bigquery.MetricHistoryRow(job_uuid=uuid, metric_key=key, metric_value=value)
    )

  for key, value in metadata.items():
    metadata_history_rows.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key=key, metadata_value=value
        )
    )

  return [metric_history_rows], [metadata_history_rows]


def get_gcs_file_location_with_regex(file_location: str) -> str:
  """
  Get a file from GCS given a regex in the form of `gs://<your_bucket>/<your_file_path_regex>`.
  Does not support bucket name or path regex. Only supports file name regex.

  Args:
    file_location: File location regex in the form of `gs://<your_bucket>/<path>/<your_file_name_regex>`.

  Returns:
    The file location of the first file that fits the given regex.
  """
  storage_client = storage.Client()

  url = urlparse(file_location)
  bucket_name = url.netloc
  file_path = url.path.strip("/")
  file_path_regex = re.compile(file_path)
  prefix = "/".join(file_path.split("/")[:-1])

  all_blobs_names = [
      b.name for b in storage_client.list_blobs(bucket_name, prefix=prefix)
  ]

  try:
    return f"gs://{bucket_name}/{next(filter(file_path_regex.match, all_blobs_names))}"
  except StopIteration:
    raise AirflowFailException(f"No objects matched supplied regex: {file_location}")


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
      os.environ.get(composer_env.COMPOSER_LOCATION),
      os.environ.get(composer_env.COMPOSER_ENVIRONMENT),
  )
  airflow_dag_run_link = (
      f"{airflow_link}/dags/{dag_id}/grid?dag_run_id={dag_run_id}&task_id={task_id}"
  )
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


def add_test_config_metadata(
    base_id: str,
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    task_gcp_config: gcp_config.GCPConfig,
    task_metric_config: metric_config.MetricConfig,
    metadata: List[List[bigquery.MetricHistoryRow]],
) -> List[List[bigquery.MetricHistoryRow]]:
  for index in range(len(metadata)):
    uuid = generate_row_uuid(base_id, index)
    test_config_meta = []

    test_config_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="accelerator",
            metadata_value=task_test_config.accelerator.name,
        )
    )
    test_config_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="project",
            metadata_value=task_gcp_config.project_name,
        )
    )
    if hasattr(task_test_config, "num_slices"):
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="num_slices",
              metadata_value=task_test_config.num_slices,
          )
      )
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="multislice_topology",
              metadata_value=f"{task_test_config.num_slices}x{task_test_config.accelerator.name}",
          )
      )
    if task_metric_config is not None and task_metric_config.tensorboard_summary:
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="metric_aggregation_strategy",
              metadata_value=task_metric_config.tensorboard_summary.aggregation_strategy.name,
          )
      )
    metadata[index].extend(test_config_meta)

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


@task(trigger_rule="all_done")
def generate_process_id() -> str:
  """Generate a process id that will be a base id for uuid of test runs.

  Returns:
    A random uuid.
  """
  return str(uuid.uuid4())


def update_dataset_name_if_needed(
    prod_dataset_name: metric_config.DatasetOption,
) -> str:
  """Update the dataset name based on stage (if needed).

  All data from prod env will be sent to benchmark_dataset or xlml_dataset;
  the rest will be sent to dev_benchmark_dataset or dev_xlml_dataset.
  """

  if not composer_env.is_prod_env():
    logging.info("This is a non-prod run, and send all data to dev dataset.")
    return f"dev_{prod_dataset_name.value}"
  return prod_dataset_name.value


def get_gke_job_status(benchmark_id: str) -> bigquery.JobStatus:
  """Get job status for the GKE run.

  FAILED - if any failure occurs in run_model
  SUCCESS - end-to-end model tests are successful in run_model
  """
  context = get_current_context()
  execution_date = context["dag_run"].logical_date
  current_dag = context["dag"]

  workload_completion = current_dag.get_task(
      task_id=f"{benchmark_id}.run_model.wait_for_workload_completion"
  )
  workload_completion_ti = TaskInstance(workload_completion, execution_date)
  workload_completion_state = workload_completion_ti.current_state()

  if workload_completion_state == TaskState.SUCCESS.value:
    logging.info(
        "The wait_for_workload_completion state is success, and the job status"
        " is success."
    )
    return bigquery.JobStatus.SUCCESS

  logging.info(
      "The wait_for_workload_completion state is not success, and the job"
      " status is failed."
  )
  return bigquery.JobStatus.FAILED


def get_gce_job_status(
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    use_startup_script: bool,
) -> bigquery.JobStatus:
  """Get job status for the GCE run.

  MISSED - if any failure occurs in initialize & create_queued_resource
  FAILED - if any failure occurs in setup & run_model (including timeout of
  run_model) for SSH method.
  FAILED - if any failure occurs in check_if_startup_script_end (including timeout of
  check_if_startup_script_end) for startup script method.
  SUCCESS - end-to-end model tests are successful from provision to run_model
  """
  context = get_current_context()
  execution_date = context["dag_run"].logical_date
  current_dag = context["dag"]
  benchmark_id = task_test_config.benchmark_id

  # GCE SSH method
  if not use_startup_script:
    if isinstance(task_test_config.accelerator, test_config.Tpu):
      # check wait status to see if wait_for_ready_queued_resource step is successful
      wait_task = current_dag.get_task(
          task_id=f"{benchmark_id}.provision.create_queued_resource.wait_for_ready_queued_resource"
      )
    elif isinstance(task_test_config.accelerator, test_config.Gpu):
      wait_task = current_dag.get_task(
          task_id=f"{benchmark_id}.provision.create_resource.get_ip_address"
      )
    else:
      raise NotImplementedError(
          f"Unable to get task for {type(task_test_config.accelerator)}."
      )
    wait_ti = TaskInstance(wait_task, execution_date)
    wait_state = wait_ti.current_state()

    if wait_state == TaskState.SKIPPED.value:
      logging.info(
          "The wait_for_ready_queued_resource state is skipped, and the job status is missed."
      )
      return bigquery.JobStatus.MISSED

    # check setup status to see if setup step is successful
    setup_task = current_dag.get_task(task_id=f"{benchmark_id}.provision.setup")
    setup_ti = TaskInstance(setup_task, execution_date)
    setup_state = setup_ti.current_state()
    if setup_state == TaskState.FAILED.value:
      logging.info("The setup state is failed, and the job status is failed.")
      return bigquery.JobStatus.FAILED

    # check run_model status to see if run_model step is successful
    run_model_task = current_dag.get_task(task_id=f"{benchmark_id}.run_model")
    run_model_ti = TaskInstance(run_model_task, execution_date)
    run_model_state = run_model_ti.current_state()

    if run_model_state == TaskState.SUCCESS.value:
      logging.info("The run_model state is success, and the job status is success.")
      return bigquery.JobStatus.SUCCESS

    logging.info("The run_model state is failed, and the job status is failed.")
    return bigquery.JobStatus.FAILED
  # GCE startup script method
  else:
    # check wait status to see if provision step is successful
    wait_task = current_dag.get_task(
        task_id=f"{benchmark_id}.provision_with_startup_script.create_queued_resource.wait_for_ready_queued_resource"
    )
    wait_ti = TaskInstance(wait_task, execution_date)
    wait_state = wait_ti.current_state()

    if wait_state == TaskState.SKIPPED.value:
      logging.info(
          "The wait_for_ready_queued_resource state is skipped, and the job status is missed."
      )
      return bigquery.JobStatus.MISSED

    # check startup_script status to see if startup_script step is successful
    startup_script_task = current_dag.get_task(
        task_id=f"{benchmark_id}.provision_with_startup_script.create_queued_resource.check_if_startup_script_end"
    )
    startup_script_ti = TaskInstance(startup_script_task, execution_date)
    startup_script_state = startup_script_ti.current_state()
    if startup_script_state == TaskState.FAILED.value:
      logging.info("The startup_script state is failed, and the job status is failed.")
      return bigquery.JobStatus.FAILED
    else:
      logging.info(
          "The startup_script state is success, and the job status is success."
      )
      return bigquery.JobStatus.SUCCESS


# TODO(ranran): handle Airflow retry to avoid duplicate records in tables
@task
def process_metrics(
    base_id: str,
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    task_metric_config: Optional[metric_config.MetricConfig],
    task_gcp_config: gcp_config.GCPConfig,
    use_startup_script: bool = False,
    file_location: Optional[str] = None,
) -> None:
  benchmark_id = task_test_config.benchmark_id
  current_time = datetime.datetime.now()
  has_profile = False
  metric_history_rows_list = [[]]
  metadata_history_rows_list = [[]]
  profile_history_rows_list = []

  # process metrics, metadata, and profile
  if task_metric_config:
    if task_metric_config.json_lines:
      metric_history_rows_list, metadata_history_rows_list = process_json_lines(
          base_id, task_metric_config.json_lines.file_location
      )
    elif task_metric_config.use_runtime_generated_filename:
      metric_history_rows_list, metadata_history_rows_list = process_json_lines(
          base_id, file_location
      )
    if task_metric_config.tensorboard_summary:
      (
          metric_history_rows_list,
          metadata_history_rows_list,
      ) = process_tensorboard_summary(base_id, task_metric_config.tensorboard_summary)
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
      base_id, task_gcp_config.composer_project, metadata_history_rows_list
  )

  metadata_history_rows_list = add_test_config_metadata(
      base_id,
      task_test_config,
      task_gcp_config,
      task_metric_config,
      metadata_history_rows_list,
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

  dataset_name = update_dataset_name_if_needed(task_gcp_config.dataset_name)
  bigquery_metric = bigquery.BigQueryMetricClient(
      task_gcp_config.dataset_project, dataset_name
  )

  if hasattr(task_test_config, "cluster_name"):
    test_job_status = get_gke_job_status(task_test_config.benchmark_id)
  else:
    test_job_status = get_gce_job_status(task_test_config, use_startup_script)

  for index in range(len(metadata_history_rows_list)):
    job_history_row = bigquery.JobHistoryRow(
        uuid=generate_row_uuid(base_id, index),
        timestamp=current_time,
        owner=task_test_config.task_owner,
        job_name=benchmark_id,
        job_status=test_job_status.value,
    )
    test_run_row = bigquery.TestRun(
        job_history_row,
        metric_history_rows_list[index],
        metadata_history_rows_list[index],
    )
    test_run_rows.append(test_run_row)

  print("Test run rows:", test_run_rows)
  bigquery_metric.insert(test_run_rows)
