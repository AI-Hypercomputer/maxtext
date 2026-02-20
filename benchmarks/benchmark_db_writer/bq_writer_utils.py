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

"""
Utilities and factory functions for creating BigQuery writer clients.

This module provides helper functions to simplify the instantiation of the
`DataclassBigQueryWriter`. It centralizes the configuration, such as
project and dataset IDs, making it easier to create database clients
for specific tables.
Copied & Modified from https://github.com/AI-Hypercomputer/aotc/blob/main/
src/aotc/benchmark_db_writer/src/benchmark_db_writer/bigquery_types.py
"""
from typing import Type
from benchmarks.benchmark_db_writer import dataclass_bigquery_writer


def create_bq_writer_object(project, dataset, table, dataclass_type):
  """Creates a BQ writer config and uses it to create BQ writer object."""

  config = dataclass_bigquery_writer.BigqueryWriterConfig(project, dataset, table)

  writer = dataclass_bigquery_writer.DataclassBigQueryWriter(dataclass_type, config)

  return writer


def get_db_client(table: str, dataclass_type: Type) -> create_bq_writer_object:
  """Creates a BigQuery client object.

  Args:
    table: The name of the BigQuery table.
    dataclass_type: The dataclass type corresponding to the table schema.

  Returns:
    A BigQuery client object.
  """

  project = "ml-workload-benchmarks"
  dataset = "benchmark_dataset_v2"
  return create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )
