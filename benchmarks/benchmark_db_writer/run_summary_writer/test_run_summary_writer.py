"""This script validates the schema of big query table & dataclass present in code.

Modify arguments of check_schema() method in main function
to check prod & test env.
"""

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)


def check_schema(is_test: bool = True) -> None:

  try:
    bq_writer_utils.get_db_client(
        "run_summary",
        workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
        is_test,
    )
    print("No schema mismatch found")
  except Exception as e:
    print("Schema mismatch found.", e)


if __name__ == "__main__":
  # Change is_test flat to True for test env's table
  check_schema(is_test=True)
