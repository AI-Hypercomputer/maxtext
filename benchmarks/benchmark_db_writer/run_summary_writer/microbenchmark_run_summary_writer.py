"""Sample script for uploading manually to run summary table."""

import logging
import os
from typing import Optional
import uuid

from benchmarks.benchmark_db_writer import bigquery_types
from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    hardware_info_schema,
)
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    microbenchmark_run_summary_schema,
)
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    microbenchmark_workload_info_schema,
)
import gspread
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_workload_id(workload_id: str, is_test: bool = False) -> bool:
  """Validates a workload ID against the microbenchmark_workload_info table."""

  return bq_writer_utils.validate_id(
      logger,
      workload_id,
      "microbenchmark_workload_info",
      "workload_id",
      microbenchmark_workload_info_schema.MicrobenchmarkWorkloadInfo,
      is_test,
  )


def validate_hardware_id(hardware_id: str, is_test: bool = False) -> bool:
  """Validates a hardware ID against the hardware_info table."""
  return bq_writer_utils.validate_id(
      logger,
      hardware_id,
      "hardware_info",
      "hardware_id",
      hardware_info_schema.HardwareInfo,
      is_test,
  )


def write_run_manually(
    run_id: str,
    workload_id: str,
    workload_parameters: str,
    run_date: bigquery_types.TimeStamp,
    hardware_id: str,
    hardware_num_chips: int,
    hardware_num_nodes: Optional[int],
    hardware_num_slices: Optional[str],
    result_success: bool,
    configs_num_iterations: int,
    configs_other: Optional[str],
    logs_artifact_directory: Optional[str],
    benchmarker_ldap: str,
    run_source: str,
    run_type: str,
    metrics_type: str,
    metrics_unit: str,
    metrics_p50: float,
    metrics_p90: float,
    metrics_p99: float,
    metrics_avg: Optional[float],
    metrics_stdev: Optional[float],
    metrics_other: Optional[str],
    result_error: Optional[str],
    logs_profile: Optional[str],
    logs_cloud_logs: Optional[str],
    logs_comments: Optional[str],
    logs_other: Optional[str],
    update_person_ldap: str = os.getenv("USER"),
    is_test: bool = True,
) -> None:
  """Writes a microbenchmark run manually to the database.

  This function validates the provided workload ID and, if valid, constructs a
  MicrobenchmarkRunSummarySchema object with the given data and writes it to the
  "microbenchmark_run_summary" table in BigQuery.

    Args:
      is_test: Whether to use the testing project or the production project.

  Raises:
    ValueError: If any of the IDs are invalid.
  """

  if validate_workload_id(workload_id, is_test) and validate_hardware_id(hardware_id, is_test):

    summary = microbenchmark_run_summary_schema.MicrobenchmarkRunSummarySchema(
        run_id=run_id if run_id else f"run-{uuid.uuid4()}",
        workload_id=workload_id,
        workload_parameters=workload_parameters,
        run_date=run_date,
        hardware_id=hardware_id,
        hardware_num_chips=hardware_num_chips,
        hardware_num_nodes=hardware_num_nodes,
        hardware_num_slices=hardware_num_slices,
        result_success=result_success,
        configs_num_iterations=configs_num_iterations,
        configs_other=configs_other,
        logs_artifact_directory=logs_artifact_directory,
        benchmarker_ldap=benchmarker_ldap,
        run_source=run_source,
        run_type=run_type,
        metrics_type=metrics_type,
        metrics_unit=metrics_unit,
        metrics_p50=metrics_p50,
        metrics_p90=metrics_p90,
        metrics_p99=metrics_p99,
        metrics_avg=metrics_avg,
        metrics_stdev=metrics_stdev,
        metrics_other=metrics_other,
        result_error=result_error,
        logs_profile=logs_profile,
        logs_cloud_logs=logs_cloud_logs,
        logs_comments=logs_comments,
        logs_other=logs_other,
        update_person_ldap=update_person_ldap,
    )

    client = bq_writer_utils.get_db_client(
        "microbenchmark_run_summary",
        microbenchmark_run_summary_schema.MicrobenchmarkRunSummarySchema,
        is_test,
    )
    client.write([summary])

  else:
    raise ValueError("Could not upload data in run summary table")


def write_row(row, is_test):
  kwargs = {k: None if v == "" else v for k, v in row.to_dict().items()}
  kwargs["is_test"] = is_test
  write_run_manually(**kwargs)


def upload_from_spreadsheet(spreadsheet_name, is_test):
  # Uses supercomputer-testing service account
  # benchmarkdb-gsheets-sa@supercomputer-testing.iam.gserviceaccount.com
  # to authenticate. The GSheet must be shared with that service account.
  gc = gspread.service_account()
  sheet = gc.open(spreadsheet_name)
  df = pd.DataFrame(sheet.sheet1.get_all_records())
  print(df.dtypes)
  rows_to_upload = df[~(df["Uploaded to BQ"].str.lower() == "true")]
  rows_to_upload = rows_to_upload.drop("Uploaded to BQ", axis=1)
  print(rows_to_upload)
  rows_to_upload.apply(lambda row: write_row(row, is_test), axis=1)


if __name__ == "__main__":
  # spreadsheet = "Microbenchmark results Shared with service account for upload"
  spreadsheet = None
  upload_from_spreadsheet(
      spreadsheet,
      is_test=False,
  )
