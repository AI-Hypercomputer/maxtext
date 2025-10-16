import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class DataLoadingMetricsInfo:
  run_id: str
  data_loading_tokens_per_sec_p50: Optional[float]
  data_loading_tokens_per_sec_p90: Optional[float]
  data_loading_tokens_per_sec_p99: Optional[float]
  data_loading_tokens_per_sec_p100: Optional[float]
  accelerator_blocked_time: Optional[float]
  accelerator_blocked_percent: Optional[float]
  additional_metrics: Optional[dict]
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
