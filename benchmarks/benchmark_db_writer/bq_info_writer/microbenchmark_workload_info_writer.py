"""TODO: Update hardware info in the main function & run the script."""

import logging
import os

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    microbenchmark_workload_info_schema,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_microbenchmark_workload_config(
    project,
    dataset,
    table,
    dataclass_type,
    workload_id,
    update_person_ldap=os.getenv("USER", "imo-eng"),
    description="",
):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )

  microbenchmark_workload_info = writer.query(where={"workload_id": workload_id})
  if microbenchmark_workload_info:
    raise ValueError("Workload id %s is already present in the %s table" % (microbenchmark_workload_info, table))

  workload_data = microbenchmark_workload_info_schema.MicrobenchmarkWorkloadInfo(
      workload_id=workload_id,
      update_person_ldap=update_person_ldap,
      description=description,
  )

  logging.info("Writing Data %s to %s table.", workload_data, table)
  writer.write([workload_data])


def insert(workload_id, description=""):
  table_configs = [
      {
          "project": "ml-workload-benchmarks",
          "dataset": "benchmark_dataset_v2",
          "table": "microbenchmark_workload_info",
      },
      {
          "project": "supercomputer-testing",
          "dataset": "mantaray_v2",
          "table": "microbenchmark_workload_info",
      },
  ]

  assert workload_id is not None

  for table_config in table_configs:
    write_microbenchmark_workload_config(
        project=table_config["project"],
        dataset=table_config["dataset"],
        table=table_config["table"],
        dataclass_type=microbenchmark_workload_info_schema.MicrobenchmarkWorkloadInfo,
        workload_id=workload_id,
        description=description,
    )


if __name__ == "__main__":

  # workloads = ["all_gather", "ppermute", "psum", "psum_scatter"]
  workloads = []
  for workload in workloads:
    insert(workload, "")
