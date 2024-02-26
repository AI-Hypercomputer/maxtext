# Copyright 2024 Google LLC
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

import datetime
import os
from airflow.decorators import task
from dags import gcs_bucket


@task
def generate_run_name(benchmark_id: str) -> str:
  """Generates a unique run name by appending the current datetime to benchmark_id.

  Args:
    benchmark_id: Benchmark id of the test
  """
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  return f"{benchmark_id}-{current_datetime}"


@task
def generate_tb_file_location(run_name: str, base_output_directory: str) -> str:
  """Generates a path to the tensorboard file to be used as a regex. Assumes the file is located in <base_output_directory>/<run_name>/tensorboard/events.out.tfevents.*

  Args:
    run_name: run name for the tensorboard file location
    base_output_directory: GCS bucket path
  """
  return os.path.join(
      base_output_directory, run_name, "tensorboard", "events.out.tfevents.*"
  )


@task
def generate_gcs_file_location(benchmark_id: str) -> str:
  """Generates result file location in GCS.

  Args:
    benchmark_id: Benchmark id of the test

  Returns:
    gsc file name with location
  """
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  return f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/{benchmark_id}-{current_datetime}/metric_report.jsonl"
