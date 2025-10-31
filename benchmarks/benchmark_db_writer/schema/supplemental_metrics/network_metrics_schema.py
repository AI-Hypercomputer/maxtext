import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class NetworkMetricsInfo:
  run_id: str
  server_max_egress: Optional[float] = None
  server_avg_egress: Optional[float] = None
  server_max_ingress: Optional[float] = None
  server_avg_ingress: Optional[float] = None
  server_max_qps: Optional[float] = None
  server_avg_qps: Optional[float] = None
  client_max_egress: Optional[float] = None
  client_avg_egress: Optional[float] = None
  client_max_ingress: Optional[float] = None
  client_avg_ingress: Optional[float] = None
  client_max_qps: Optional[float] = None
  client_avg_qps: Optional[float] = None
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
