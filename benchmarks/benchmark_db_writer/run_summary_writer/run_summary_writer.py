"""Sample script for uploading manually to run summary table."""

import logging
import os
import uuid

from benchmarks.benchmark_db_writer import bigquery_types
from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    hardware_info_schema,
)
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import model_info_schema
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    software_info_schema,
)
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import storage_info_schema
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)
import gspread
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_model_id(model_id: str, is_test: bool = False) -> bool:
  """Validates a model ID against the model_info table."""

  return bq_writer_utils.validate_id(
      logger,
      model_id,
      "model_info",
      "model_id",
      model_info_schema.ModelInfo,
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


def validate_software_id(software_id: str, is_test: bool = False) -> bool:
  """Validates a software ID against the software_info table."""
  return bq_writer_utils.validate_id(
      logger,
      software_id,
      "software_info",
      "software_id",
      software_info_schema.SoftwareInfo,
      is_test,
  )


def validate_storage_id(storage_id: str, is_test: bool = False) -> bool:
  """Validates a storage ID against the storage_info table."""
  if storage_id:
    return bq_writer_utils.validate_id(
        logger,
        storage_id,
        "storage_info",
        "storage_id",
        storage_info_schema.StorageInfo,
        is_test,
    )
  return True


def write_run_manually(
    is_test: bool = False,
    **kwargs,
) -> None:
  """Writes a workload benchmark run manually to the database.

  This function validates the provided IDs and, if valid, constructs a
  WorkloadBenchmarkV2Schema object with the given data and writes it to the
  "run_summary" table in BigQuery.


  Raises:
    ValueError: If any of the IDs are invalid.
  """

  assert "model_id" in kwargs and "hardware_id" in kwargs and "software_id" in kwargs

  model_id = kwargs["model_id"]
  hardware_id = kwargs["hardware_id"]
  software_id = kwargs["software_id"]
  storage_id = kwargs["storage_id"] if "storage_id" in kwargs else None
  if "run_id" not in kwargs:
    kwargs["run_id"] = f"run-{uuid.uuid4()}"
  if "experiment_id" not in kwargs:
    kwargs["experiment_id"] = f"experiment-manual-{uuid.uuid4()}"

  kwargs["run_source"] = "manual"

  if (
      validate_model_id(model_id, is_test)
      and validate_hardware_id(hardware_id, is_test)
      and validate_software_id(software_id, is_test)
      and validate_storage_id(storage_id, is_test)
  ):

    summary = workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema(**kwargs)

    client = bq_writer_utils.get_db_client(
        "run_summary",
        workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
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
  # spreadsheet = "BenchmarkDB Training Dataset"
  spreadsheet = None
  upload_from_spreadsheet(
      spreadsheet,
      is_test=False,
  )
