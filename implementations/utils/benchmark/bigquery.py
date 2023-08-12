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

"""Utilities for Benchmark tests to integrate with BigQuery."""


import dataclasses
import datetime
import enum
import math
from typing import Iterable, Optional
from absl import logging
import google.auth
from google.cloud import bigquery


BENCHMARK_DATASET_NAME = "benchmark_dataset"
BENCHMARK_BQ_JOB_TABLE_NAME = "job_history"
BENCHMARK_BQ_METRIC_TABLE_NAME = "metric_history"
BENCHMARK_BQ_METADATA_TABLE_NAME = "metadata_history"


@dataclasses.dataclass
class JobHistoryRow:
  uuid: str
  timestamp: datetime.datetime
  owner: str
  job_name: str
  job_status: str


@dataclasses.dataclass
class MetricHistoryRow:
  job_uuid: str
  metric_key: str
  metric_value: float


@dataclasses.dataclass
class MetadataHistoryRow:
  job_uuid: str
  metadata_key: str
  metadata_value: str


@dataclasses.dataclass
class BenchmarkTestRun:
  job_history: JobHistoryRow
  metric_history: Iterable[MetricHistoryRow]
  metadata_history: Iterable[MetadataHistoryRow]


class JobStatus(enum.Enum):
  SUCCESS = 0
  FAILURE = 1
  TIMEOUT = 2
  MISSED = 3


class BigQueryMetricClient:
  """BigQuery metric client for benchmark tests.

  Attributes:
    project: The project name for database.
    database: The database name for BigQuery.
  """

  def __init__(
      self,
      project: Optional[str] = None,
      database: Optional[str] = None,
  ):
    self.project = google.auth.default()[1] if project is None else project
    self.database = BENCHMARK_DATASET_NAME if database is None else database
    self.client = bigquery.Client(
        project=project,
        default_query_job_config=bigquery.job.QueryJobConfig(
            default_dataset=".".join((self.project, self.database)),
        ),
    )

  @property
  def job_history_table_id(self):
    return ".".join((self.project, self.database, BENCHMARK_BQ_JOB_TABLE_NAME))

  @property
  def metric_history_table_id(self):
    return ".".join(
        (self.project, self.database, BENCHMARK_BQ_METRIC_TABLE_NAME)
    )

  @property
  def metadata_history_table_id(self):
    return ".".join(
        (self.project, self.database, BENCHMARK_BQ_METADATA_TABLE_NAME)
    )

  def is_valid_metric(self, value: float):
    """Check if float metric is valid for BigQuery table."""
    invalid_values = [math.inf, -math.inf, math.nan]
    return not (value in invalid_values or math.isnan(value))

  def insert(self, test_runs: Iterable[BenchmarkTestRun]) -> None:
    """Insert Benchmark test runs into the table.

    Args:
      test_runs: Test runs in a benchmark test job.
    """
    for run in test_runs:
      # job hisotry rows
      job_history_rows = [dataclasses.astuple(run.job_history)]

      # metric hisotry rows
      metric_history_rows = []
      for each in run.metric_history:
        if self.is_valid_metric(each.metric_value):
          metric_history_rows.append(dataclasses.astuple(each))
        else:
          logging.error(f"Discarding metric as {each.metric_value} is invalid.")

      # metadata hisotry rows
      metadata_history_rows = []
      for each in run.metadata_history:
        metadata_history_rows.append(dataclasses.astuple(each))

      for table_id, rows in [
          (self.job_history_table_id, job_history_rows),
          (self.metric_history_table_id, metric_history_rows),
          (self.metadata_history_table_id, metadata_history_rows),
      ]:
        if not rows:
          continue
        logging.info(
            f"Inserting {len(rows)} rows into BigQuery table {table_id}."
        )
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows(table, rows)

        if errors:
          raise RuntimeError(f"Failed to add rows to Bigquery: {errors}.")
        else:
          logging.info("Successfully added rows to Bigquery.")
