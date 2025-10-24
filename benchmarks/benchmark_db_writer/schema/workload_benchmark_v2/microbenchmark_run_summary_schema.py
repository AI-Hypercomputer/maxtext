import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types

# Table: microbenchmark_run_summary


@dataclasses.dataclass
class MicrobenchmarkRunSummarySchema:
  run_id: str

  # Unique workload_id to map to microbenchmark_workload_info table
  workload_id: str
  workload_parameters: str
  run_date: bigquery_types.TimeStamp

  # Foreign  key to join with hardware info
  hardware_id: str
  hardware_num_chips: int

  result_success: bool

  configs_num_iterations: int

  benchmarker_ldap: str

  metrics_type: str
  metrics_unit: str
  metrics_p50: float
  metrics_p90: float
  metrics_p99: float
  update_person_ldap: str

  hardware_num_nodes: Optional[int] = None
  hardware_num_slices: Optional[str] = None
  configs_other: Optional[str] = None

  logs_artifact_directory: Optional[str] = None
  run_source: str = "manual"
  run_type: str = "perf_optimization"

  metrics_avg: Optional[float] = None
  metrics_stdev: Optional[float] = None
  metrics_other: Optional[str] = None

  result_error: Optional[str] = None

  logs_profile: Optional[str] = None
  logs_cloud_logs: Optional[str] = None
  logs_comments: Optional[str] = None
  logs_other: Optional[str] = None

  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
