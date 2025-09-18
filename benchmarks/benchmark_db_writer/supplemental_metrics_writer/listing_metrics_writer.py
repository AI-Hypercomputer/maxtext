"""
Update the listing_metrics table of the benchmark dataset.
"""

import logging
from typing import Sequence

from absl import app
from absl import flags
from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.supplemental_metrics import listing_metrics_schema
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)
from benchmarks.benchmark_db_writer.supplemental_metrics_writer import common_flags

_FIRST_ITERATION_METRICS_GCS_LIST_TIME_AVG = flags.DEFINE_float(
    "first_iteration_metrics_gcs_list_time_avg",
    None,
    "The average time it takes to perform the first GCS listing, in seconds.",
)
_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_AVG = flags.DEFINE_float(
    "first_iteration_metrics_fuse_list_time_avg",
    None,
    "The average time it takes to perform the first GCS Fuse listing, in" " seconds.",
)
_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_AVG = flags.DEFINE_float(
    "first_iteration_metrics_gcs_subdir_list_time_avg",
    None,
    "The average time it takes to perform the first GCS subdirectory listing," " in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P50 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the first GCS listing, in" " seconds.",
)
_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P50 = flags.DEFINE_float(
    "first_iteration_metrics_fuse_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the first GCS Fuse listing," " in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P50 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_subdir_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the first GCS subdirectory" " listing, in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P90 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the first GCS listing, in" " seconds.",
)
_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P90 = flags.DEFINE_float(
    "first_iteration_metrics_fuse_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the first GCS Fuse listing," " in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P90 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_subdir_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the first GCS subdirectory" " listing, in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P99 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the first GCS listing, in" " seconds.",
)
_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P99 = flags.DEFINE_float(
    "first_iteration_metrics_fuse_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the first GCS Fuse listing," " in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P99 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_subdir_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the first GCS subdirectory" " listing, in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P100 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the first GCS listing, in" " seconds.",
)
_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P100 = flags.DEFINE_float(
    "first_iteration_metrics_fuse_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the first GCS Fuse listing," " in seconds.",
)
_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P100 = flags.DEFINE_float(
    "first_iteration_metrics_gcs_subdir_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the first GCS subdirectory" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_AVG = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_list_time_avg",
    None,
    "The average time it takes to perform the subsequent GCS listing, in" " seconds.",
)
_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_AVG = flags.DEFINE_float(
    "subsq_iteration_metrics_fuse_list_time_avg",
    None,
    "The average time it takes to perform the subsequent GCS Fuse listing, in" " seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_AVG = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_subdir_list_time_avg",
    None,
    "The average time it takes to perform the subsequent GCS subdirectory" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P50 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the subsequent GCS listing," " in seconds.",
)
_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P50 = flags.DEFINE_float(
    "subsq_iteration_metrics_fuse_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the subsequent GCS Fuse" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P50 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_subdir_list_time_p50",
    None,
    "50 percentile of the time it takes to perform the subsequent GCS" " subdirectory listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P90 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the subsequent GCS listing," " in seconds.",
)
_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P90 = flags.DEFINE_float(
    "subsq_iteration_metrics_fuse_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the subsequent GCS Fuse" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P90 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_subdir_list_time_p90",
    None,
    "90 percentile of the time it takes to perform the subsequent GCS" " subdirectory listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P99 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the subsequent GCS listing," " in seconds.",
)
_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P99 = flags.DEFINE_float(
    "subsq_iteration_metrics_fuse_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the subsequent GCS Fuse" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P99 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_subdir_list_time_p99",
    None,
    "99 percentile of the time it takes to perform the subsequent GCS" " subdirectory listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P100 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the subsequent GCS listing," " in seconds.",
)
_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P100 = flags.DEFINE_float(
    "subsq_iteration_metrics_fuse_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the subsequent GCS Fuse" " listing, in seconds.",
)
_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P100 = flags.DEFINE_float(
    "subsq_iteration_metrics_gcs_subdir_list_time_p100",
    None,
    "100 percentile of the time it takes to perform the subsequent GCS" " subdirectory listing, in seconds.",
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_listing_metrics(
    project,
    dataset,
    table,
    dataclass_type,
    run_id,
    first_iteration_metrics_gcs_list_time_avg,
    first_iteration_metrics_fuse_list_time_avg,
    first_iteration_metrics_gcs_subdir_list_time_avg,
    first_iteration_metrics_gcs_list_time_p50,
    first_iteration_metrics_fuse_list_time_p50,
    first_iteration_metrics_gcs_subdir_list_time_p50,
    first_iteration_metrics_gcs_list_time_p90,
    first_iteration_metrics_fuse_list_time_p90,
    first_iteration_metrics_gcs_subdir_list_time_p90,
    first_iteration_metrics_gcs_list_time_p99,
    first_iteration_metrics_fuse_list_time_p99,
    first_iteration_metrics_gcs_subdir_list_time_p99,
    first_iteration_metrics_gcs_list_time_p100,
    first_iteration_metrics_fuse_list_time_p100,
    first_iteration_metrics_gcs_subdir_list_time_p100,
    subsq_iteration_metrics_gcs_list_time_avg,
    subsq_iteration_metrics_fuse_list_time_avg,
    subsq_iteration_metrics_gcs_subdir_list_time_avg,
    subsq_iteration_metrics_gcs_list_time_p50,
    subsq_iteration_metrics_fuse_list_time_p50,
    subsq_iteration_metrics_gcs_subdir_list_time_p50,
    subsq_iteration_metrics_gcs_list_time_p90,
    subsq_iteration_metrics_fuse_list_time_p90,
    subsq_iteration_metrics_gcs_subdir_list_time_p90,
    subsq_iteration_metrics_gcs_list_time_p99,
    subsq_iteration_metrics_fuse_list_time_p99,
    subsq_iteration_metrics_gcs_subdir_list_time_p99,
    subsq_iteration_metrics_gcs_list_time_p100,
    subsq_iteration_metrics_fuse_list_time_p100,
    subsq_iteration_metrics_gcs_subdir_list_time_p100,
    is_test=False,
):

  if bq_writer_utils.validate_id(
      logger,
      run_id,
      "run_summary",
      "run_id",
      workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
      is_test,
  ):

    writer = bq_writer_utils.create_bq_writer_object(
        project=project,
        dataset=dataset,
        table=table,
        dataclass_type=dataclass_type,
    )

    listing_metrics_data = listing_metrics_schema.ListingMetricsInfo(
        run_id=run_id,
        first_iteration_metrics_gcs_list_time_avg=first_iteration_metrics_gcs_list_time_avg,
        first_iteration_metrics_fuse_list_time_avg=first_iteration_metrics_fuse_list_time_avg,
        first_iteration_metrics_gcs_subdir_list_time_avg=first_iteration_metrics_gcs_subdir_list_time_avg,
        first_iteration_metrics_gcs_list_time_p50=first_iteration_metrics_gcs_list_time_p50,
        first_iteration_metrics_fuse_list_time_p50=first_iteration_metrics_fuse_list_time_p50,
        first_iteration_metrics_gcs_subdir_list_time_p50=first_iteration_metrics_gcs_subdir_list_time_p50,
        first_iteration_metrics_gcs_list_time_p90=first_iteration_metrics_gcs_list_time_p90,
        first_iteration_metrics_fuse_list_time_p90=first_iteration_metrics_fuse_list_time_p90,
        first_iteration_metrics_gcs_subdir_list_time_p90=first_iteration_metrics_gcs_subdir_list_time_p90,
        first_iteration_metrics_gcs_list_time_p99=first_iteration_metrics_gcs_list_time_p99,
        first_iteration_metrics_fuse_list_time_p99=first_iteration_metrics_fuse_list_time_p99,
        first_iteration_metrics_gcs_subdir_list_time_p99=first_iteration_metrics_gcs_subdir_list_time_p99,
        first_iteration_metrics_gcs_list_time_p100=first_iteration_metrics_gcs_list_time_p100,
        first_iteration_metrics_fuse_list_time_p100=first_iteration_metrics_fuse_list_time_p100,
        first_iteration_metrics_gcs_subdir_list_time_p100=first_iteration_metrics_gcs_subdir_list_time_p100,
        subsq_iteration_metrics_gcs_list_time_avg=subsq_iteration_metrics_gcs_list_time_avg,
        subsq_iteration_metrics_fuse_list_time_avg=subsq_iteration_metrics_fuse_list_time_avg,
        subsq_iteration_metrics_gcs_subdir_list_time_avg=subsq_iteration_metrics_gcs_subdir_list_time_avg,
        subsq_iteration_metrics_gcs_list_time_p50=subsq_iteration_metrics_gcs_list_time_p50,
        subsq_iteration_metrics_fuse_list_time_p50=subsq_iteration_metrics_fuse_list_time_p50,
        subsq_iteration_metrics_gcs_subdir_list_time_p50=subsq_iteration_metrics_gcs_subdir_list_time_p50,
        subsq_iteration_metrics_gcs_list_time_p90=subsq_iteration_metrics_gcs_list_time_p90,
        subsq_iteration_metrics_fuse_list_time_p90=subsq_iteration_metrics_fuse_list_time_p90,
        subsq_iteration_metrics_gcs_subdir_list_time_p90=subsq_iteration_metrics_gcs_subdir_list_time_p90,
        subsq_iteration_metrics_gcs_list_time_p99=subsq_iteration_metrics_gcs_list_time_p99,
        subsq_iteration_metrics_fuse_list_time_p99=subsq_iteration_metrics_fuse_list_time_p99,
        subsq_iteration_metrics_gcs_subdir_list_time_p99=subsq_iteration_metrics_gcs_subdir_list_time_p99,
        subsq_iteration_metrics_gcs_list_time_p100=subsq_iteration_metrics_gcs_list_time_p100,
        subsq_iteration_metrics_fuse_list_time_p100=subsq_iteration_metrics_fuse_list_time_p100,
        subsq_iteration_metrics_gcs_subdir_list_time_p100=subsq_iteration_metrics_gcs_subdir_list_time_p100,
    )

    logging.info("Writing Data %s to %s table.", listing_metrics_data, table)
    writer.write([listing_metrics_data])

  else:
    raise ValueError("Could not upload data in run summary table")


def main(_: Sequence[str]):
  write_listing_metrics(
      project=("supercomputer-testing" if common_flags.IS_TEST.value else "ml-workload-benchmarks"),
      dataset=("mantaray_v2" if common_flags.IS_TEST.value else "benchmark_dataset_v2"),
      table="listing_metrics",
      dataclass_type=listing_metrics_schema.ListingMetricsInfo,
      run_id=common_flags.RUN_ID.value,
      first_iteration_metrics_gcs_list_time_avg=_FIRST_ITERATION_METRICS_GCS_LIST_TIME_AVG.value,
      first_iteration_metrics_fuse_list_time_avg=_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_AVG.value,
      first_iteration_metrics_gcs_subdir_list_time_avg=_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_AVG.value,
      first_iteration_metrics_gcs_list_time_p50=_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P50.value,
      first_iteration_metrics_fuse_list_time_p50=_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P50.value,
      first_iteration_metrics_gcs_subdir_list_time_p50=_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P50.value,
      first_iteration_metrics_gcs_list_time_p90=_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P90.value,
      first_iteration_metrics_fuse_list_time_p90=_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P90.value,
      first_iteration_metrics_gcs_subdir_list_time_p90=_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P90.value,
      first_iteration_metrics_gcs_list_time_p99=_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P99.value,
      first_iteration_metrics_fuse_list_time_p99=_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P99.value,
      first_iteration_metrics_gcs_subdir_list_time_p99=_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P99.value,
      first_iteration_metrics_gcs_list_time_p100=_FIRST_ITERATION_METRICS_GCS_LIST_TIME_P100.value,
      first_iteration_metrics_fuse_list_time_p100=_FIRST_ITERATION_METRICS_FUSE_LIST_TIME_P100.value,
      first_iteration_metrics_gcs_subdir_list_time_p100=_FIRST_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P100.value,
      subsq_iteration_metrics_gcs_list_time_avg=_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_AVG.value,
      subsq_iteration_metrics_fuse_list_time_avg=_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_AVG.value,
      subsq_iteration_metrics_gcs_subdir_list_time_avg=_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_AVG.value,
      subsq_iteration_metrics_gcs_list_time_p50=_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P50.value,
      subsq_iteration_metrics_fuse_list_time_p50=_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P50.value,
      subsq_iteration_metrics_gcs_subdir_list_time_p50=_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P50.value,
      subsq_iteration_metrics_gcs_list_time_p90=_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P90.value,
      subsq_iteration_metrics_fuse_list_time_p90=_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P90.value,
      subsq_iteration_metrics_gcs_subdir_list_time_p90=_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P90.value,
      subsq_iteration_metrics_gcs_list_time_p99=_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P99.value,
      subsq_iteration_metrics_fuse_list_time_p99=_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P99.value,
      subsq_iteration_metrics_gcs_subdir_list_time_p99=_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P99.value,
      subsq_iteration_metrics_gcs_list_time_p100=_SUBSQ_ITERATION_METRICS_GCS_LIST_TIME_P100.value,
      subsq_iteration_metrics_fuse_list_time_p100=_SUBSQ_ITERATION_METRICS_FUSE_LIST_TIME_P100.value,
      subsq_iteration_metrics_gcs_subdir_list_time_p100=_SUBSQ_ITERATION_METRICS_GCS_SUBDIR_LIST_TIME_P100.value,
      is_test=common_flags.IS_TEST.value,
  )


if __name__ == "__main__":
  app.run(main)
