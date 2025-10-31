"""Update the data_loading_metrics table of the benchmark dataset."""

import logging
from typing import Sequence
from absl import app
from absl import flags
import json

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.supplemental_metrics_writer import common_flags
from benchmarks.benchmark_db_writer.schema.supplemental_metrics import (
    data_loading_metrics_schema,
)
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)


_DATA_LOADING_TOKENS_PER_SEC_P50 = flags.DEFINE_float(
    "data_loading_tokens_per_sec_p50",
    None,
    "50 percentile of the egress throughput delivered by the storage system" " per step, in tokens per second.",
)
_DATA_LOADING_TOKENS_PER_SEC_P90 = flags.DEFINE_float(
    "data_loading_tokens_per_sec_p90",
    None,
    "90 percentile of the egress throughput delivered by the storage system" " per step, in tokens per second.",
)
_DATA_LOADING_TOKENS_PER_SEC_P99 = flags.DEFINE_float(
    "data_loading_tokens_per_sec_p99",
    None,
    "99 percentile of the egress throughput delivered by the storage system" " per step, in tokens per second.",
)
_DATA_LOADING_TOKENS_PER_SEC_P100 = flags.DEFINE_float(
    "data_loading_tokens_per_sec_p100",
    None,
    "100 percentile of the egress throughput delivered by the storage system" " per step, in tokens per second.",
)
_ACCELERATOR_BLOCKED_TIME = flags.DEFINE_float(
    "accelerator_blocked_time",
    None,
    "The duration an accelerator is unavailable for processing tasks due to " "data loading.",
)
_ACCELERATOR_BLOCKED_PERCENT = flags.DEFINE_float(
    "accelerator_blocked_percent",
    None,
    "The percent of time an accelerator is unavailable for processing tasks " "due to data loading.",
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_data_loading_metrics(
    project,
    dataset,
    table,
    dataclass_type,
    run_id,
    data_loading_tokens_per_sec_p50,
    data_loading_tokens_per_sec_p90,
    data_loading_tokens_per_sec_p99,
    data_loading_tokens_per_sec_p100,
    accelerator_blocked_time,
    accelerator_blocked_percent,
    additional_metrics,
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

    data_loading_metrics_data = data_loading_metrics_schema.DataLoadingMetricsInfo(
        run_id=run_id,
        data_loading_tokens_per_sec_p50=data_loading_tokens_per_sec_p50,
        data_loading_tokens_per_sec_p90=data_loading_tokens_per_sec_p90,
        data_loading_tokens_per_sec_p99=data_loading_tokens_per_sec_p99,
        data_loading_tokens_per_sec_p100=data_loading_tokens_per_sec_p100,
        accelerator_blocked_time=accelerator_blocked_time,
        accelerator_blocked_percent=accelerator_blocked_percent,
        additional_metrics=json.loads(additional_metrics),
    )

    logging.info("Writing Data %s to %s table.", data_loading_metrics_data, table)
    writer.write([data_loading_metrics_data])

  else:
    raise ValueError("Could not upload data in data_loading_metrics table.")


def main(_: Sequence[str]):
  write_data_loading_metrics(
      project=("supercomputer-testing" if common_flags.IS_TEST.value else "ml-workload-benchmarks"),
      dataset=("mantaray_v2" if common_flags.IS_TEST.value else "benchmark_dataset_v2"),
      table="data_loading_metrics",
      dataclass_type=data_loading_metrics_schema.DataLoadingMetricsInfo,
      run_id=common_flags.RUN_ID.value,
      data_loading_tokens_per_sec_p50=_DATA_LOADING_TOKENS_PER_SEC_P50.value,
      data_loading_tokens_per_sec_p90=_DATA_LOADING_TOKENS_PER_SEC_P90.value,
      data_loading_tokens_per_sec_p99=_DATA_LOADING_TOKENS_PER_SEC_P99.value,
      data_loading_tokens_per_sec_p100=_DATA_LOADING_TOKENS_PER_SEC_P100.value,
      accelerator_blocked_time=_ACCELERATOR_BLOCKED_TIME.value,
      accelerator_blocked_percent=_ACCELERATOR_BLOCKED_PERCENT.value,
      additional_metrics=common_flags.ADDITIONAL_METRICS.value,
      is_test=common_flags.IS_TEST.value,
  )


if __name__ == "__main__":
  app.run(main)
