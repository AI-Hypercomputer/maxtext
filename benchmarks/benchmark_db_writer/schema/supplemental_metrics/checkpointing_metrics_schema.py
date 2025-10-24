import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class CheckpointingMetricsInfo:
  run_id: str
  restore_time_p50: Optional[float]
  restore_time_p90: Optional[float]
  restore_time_p99: Optional[float]
  restore_time_p100: Optional[float]
  restore_time_max: Optional[float]
  restore_time_min: Optional[float]
  restore_time_avg: Optional[float]
  restore_time_stddev: Optional[float]
  restore_time_initial: Optional[float]
  write_time_p50: Optional[float]
  write_time_p90: Optional[float]
  write_time_p99: Optional[float]
  write_time_p100: Optional[float]
  write_time_max: Optional[float]
  write_time_min: Optional[float]
  write_time_avg: Optional[float]
  write_time_stddev: Optional[float]
  accelerator_to_cpu_time_max: Optional[float]
  accelerator_to_cpu_time_min: Optional[float]
  accelerator_to_cpu_time_avg: Optional[float]
  storage_save_time_max: Optional[float]
  storage_save_time_min: Optional[float]
  storage_save_time_avg: Optional[float]
  num_restore_datapoints: Optional[int]
  num_write_datapoints: Optional[int]
  additional_metrics: Optional[dict]
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
