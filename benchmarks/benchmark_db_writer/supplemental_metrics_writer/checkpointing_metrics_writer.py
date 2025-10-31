"""Update the checkpointing_metrics table of the benchmark dataset."""

import json
import logging
from typing import Sequence

from absl import app
from absl import flags
from benchmark_db_writer import bq_writer_utils
from benchmark_db_writer.schema.supplemental_metrics import (
    checkpointing_metrics_schema,
)
from benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)
from benchmark_db_writer.supplemental_metrics_writer import common_flags

_RESTORE_TIME_P50 = flags.DEFINE_float(
    "restore_time_p50",
    None,
    "50 percentile of the time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_P90 = flags.DEFINE_float(
    "restore_time_p90",
    None,
    "90 percentile of the time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_P99 = flags.DEFINE_float(
    "restore_time_p99",
    None,
    "99 percentile of the time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_P100 = flags.DEFINE_float(
    "restore_time_p100",
    None,
    "100 percentile of the time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_MAX = flags.DEFINE_float(
    "restore_time_max",
    None,
    "The max time it takes to restore a checkpoint from storage, in seconds.",
)
_RESTORE_TIME_MIN = flags.DEFINE_float(
    "restore_time_min",
    None,
    "The min time it takes to restore a checkpoint from storage, in seconds.",
)
_RESTORE_TIME_AVG = flags.DEFINE_float(
    "restore_time_avg",
    None,
    "The average time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_STDDEV = flags.DEFINE_float(
    "restore_time_stddev",
    None,
    "The average time it takes to restore a checkpoint from storage," " in seconds.",
)
_RESTORE_TIME_INITIAL = flags.DEFINE_float(
    "restore_time_initial",
    None,
    "The time it takes to restore a checkpoint from storage for the first time"
    "(the initial restore may be cached), in seconds.",
)
_WRITE_TIME_P50 = flags.DEFINE_float(
    "write_time_p50",
    None,
    "50 percentile of the time it takes to write a checkpoint to storage, in" " seconds.",
)
_WRITE_TIME_P90 = flags.DEFINE_float(
    "write_time_p90",
    None,
    "90 percentile of the time it takes to write a checkpoint to storage, in" " seconds.",
)
_WRITE_TIME_P99 = flags.DEFINE_float(
    "write_time_p99",
    None,
    "99 percentile of the time it takes to write a checkpoint to storage, in" " seconds.",
)
_WRITE_TIME_P100 = flags.DEFINE_float(
    "write_time_p100",
    None,
    "100 percentile of the time it takes to write a checkpoint to storage, in" " seconds.",
)
_WRITE_TIME_MAX = flags.DEFINE_float(
    "write_time_max",
    None,
    "The max time it takes to write a checkpoint, in seconds. We measure"
    " the time elapsed from when a checkpoint begins blocking training until"
    " all checkpoint data is successfully saved to storage.",
)
_WRITE_TIME_MIN = flags.DEFINE_float(
    "write_time_min",
    None,
    "The min time it takes to write a checkpoint, in seconds. We measure"
    " the time elapsed from when a checkpoint begins blocking training until"
    " all checkpoint data is successfully saved to storage.",
)
_WRITE_TIME_AVG = flags.DEFINE_float(
    "write_time_avg",
    None,
    "The average time it takes to write a checkpoint, in seconds. We measure"
    " the time elapsed from when a checkpoint begins blocking training until"
    " all checkpoint data is successfully saved to storage.",
)
_WRITE_TIME_STDDEV = flags.DEFINE_float(
    "write_time_stddev",
    None,
    "The average time it takes to write a checkpoint, in seconds. We measure"
    " the time elapsed from when a checkpoint begins blocking training until"
    " all checkpoint data is successfully saved to storage.",
)
_ACCELERATOR_TO_CPU_TIME_MAX = flags.DEFINE_float(
    "accelerator_to_cpu_time_max",
    None,
    "The max time it takes to transfer a checkpoint from GPU to CPU memory," " in seconds.",
)
_ACCELERATOR_TO_CPU_TIME_MIN = flags.DEFINE_float(
    "accelerator_to_cpu_time_min",
    None,
    "The min time it takes to transfer a checkpoint from GPU to CPU memory," " in seconds.",
)
_ACCELERATOR_TO_CPU_TIME_AVG = flags.DEFINE_float(
    "accelerator_to_cpu_time_avg",
    None,
    "The average time it takes to transfer a checkpoint from GPU to CPU memory," " in seconds.",
)
_STORAGE_SAVE_TIME_MAX = flags.DEFINE_float(
    "storage_save_time_max",
    None,
    "The max time it takes to write a checkpoint from CPU memory to storage," " in seconds.",
)
_STORAGE_SAVE_TIME_MIN = flags.DEFINE_float(
    "storage_save_time_min",
    None,
    "The min time it takes to write a checkpoint from CPU memory to storage," " in seconds.",
)
_STORAGE_SAVE_TIME_AVG = flags.DEFINE_float(
    "storage_save_time_avg",
    None,
    "The average time it takes to write a checkpoint from CPU memory to" " storage, in seconds.",
)
_NUM_RESTORE_DATAPOINTS = flags.DEFINE_integer("num_restore_datapoints", None, "The number of the restore datapoints.")
_NUM_WRITE_DATAPOINTS = flags.DEFINE_integer("num_write_datapoints", None, "The number of the write datapoints.")


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_checkpointing_metrics(
    project,
    dataset,
    table,
    dataclass_type,
    run_id,
    restore_time_p50=None,
    restore_time_p90=None,
    restore_time_p99=None,
    restore_time_p100=None,
    restore_time_max=None,
    restore_time_min=None,
    restore_time_avg=None,
    restore_time_stddev=None,
    restore_time_initial=None,
    write_time_p50=None,
    write_time_p90=None,
    write_time_p99=None,
    write_time_p100=None,
    write_time_max=None,
    write_time_min=None,
    write_time_avg=None,
    write_time_stddev=None,
    accelerator_to_cpu_time_max=None,
    accelerator_to_cpu_time_min=None,
    accelerator_to_cpu_time_avg=None,
    storage_save_time_max=None,
    storage_save_time_min=None,
    storage_save_time_avg=None,
    num_restore_datapoints=None,
    num_write_datapoints=None,
    additional_metrics=None,
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

    checkpointing_metrics_data = checkpointing_metrics_schema.CheckpointingMetricsInfo(
        run_id=run_id,
        restore_time_p50=restore_time_p50,
        restore_time_p90=restore_time_p90,
        restore_time_p99=restore_time_p99,
        restore_time_p100=restore_time_p100,
        restore_time_max=restore_time_max,
        restore_time_min=restore_time_min,
        restore_time_avg=restore_time_avg,
        restore_time_stddev=restore_time_stddev,
        restore_time_initial=restore_time_initial,
        write_time_p50=write_time_p50,
        write_time_p90=write_time_p90,
        write_time_p99=write_time_p99,
        write_time_p100=write_time_p100,
        write_time_max=write_time_max,
        write_time_min=write_time_min,
        write_time_avg=write_time_avg,
        write_time_stddev=write_time_stddev,
        accelerator_to_cpu_time_max=accelerator_to_cpu_time_max,
        accelerator_to_cpu_time_min=accelerator_to_cpu_time_min,
        accelerator_to_cpu_time_avg=accelerator_to_cpu_time_avg,
        storage_save_time_max=storage_save_time_max,
        storage_save_time_min=storage_save_time_min,
        storage_save_time_avg=storage_save_time_avg,
        num_restore_datapoints=num_restore_datapoints,
        num_write_datapoints=num_write_datapoints,
        additional_metrics=json.loads(additional_metrics) if additional_metrics is not None else None,
    )

    logging.info("Writing Data %s to %s table.", checkpointing_metrics_data, table)
    writer.write([checkpointing_metrics_data])

  else:
    raise ValueError("Could not upload data in run summary table")


def main(_: Sequence[str]):
  write_checkpointing_metrics(
      project=("supercomputer-testing" if common_flags.IS_TEST.value else "ml-workload-benchmarks"),
      dataset=("mantaray_v2" if common_flags.IS_TEST.value else "benchmark_dataset_v2"),
      table="checkpointing_metrics",
      dataclass_type=checkpointing_metrics_schema.CheckpointingMetricsInfo,
      run_id=common_flags.RUN_ID.value,
      restore_time_p50=_RESTORE_TIME_P50.value,
      restore_time_p90=_RESTORE_TIME_P90.value,
      restore_time_p99=_RESTORE_TIME_P99.value,
      restore_time_p100=_RESTORE_TIME_P100.value,
      restore_time_max=_RESTORE_TIME_MAX.value,
      restore_time_min=_RESTORE_TIME_MIN.value,
      restore_time_avg=_RESTORE_TIME_AVG.value,
      restore_time_initial=_RESTORE_TIME_INITIAL.value,
      restore_time_stddev=_RESTORE_TIME_STDDEV.value,
      write_time_p50=_WRITE_TIME_P50.value,
      write_time_p90=_WRITE_TIME_P90.value,
      write_time_p99=_WRITE_TIME_P99.value,
      write_time_p100=_WRITE_TIME_P100.value,
      write_time_max=_WRITE_TIME_MAX.value,
      write_time_min=_WRITE_TIME_MIN.value,
      write_time_avg=_WRITE_TIME_AVG.value,
      write_time_stddev=_WRITE_TIME_STDDEV.value,
      accelerator_to_cpu_time_max=_ACCELERATOR_TO_CPU_TIME_MAX.value,
      accelerator_to_cpu_time_min=_ACCELERATOR_TO_CPU_TIME_MIN.value,
      accelerator_to_cpu_time_avg=_ACCELERATOR_TO_CPU_TIME_AVG.value,
      storage_save_time_max=_STORAGE_SAVE_TIME_MAX.value,
      storage_save_time_min=_STORAGE_SAVE_TIME_MIN.value,
      storage_save_time_avg=_STORAGE_SAVE_TIME_AVG.value,
      num_restore_datapoints=_NUM_RESTORE_DATAPOINTS.value,
      num_write_datapoints=_NUM_WRITE_DATAPOINTS.value,
      additional_metrics=common_flags.ADDITIONAL_METRICS.value,
      is_test=common_flags.IS_TEST.value,
  )


if __name__ == "__main__":
  app.run(main)
