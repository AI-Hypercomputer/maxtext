import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


# Table microbenchmark_workload_info


@dataclasses.dataclass
class MicrobenchmarkWorkloadInfo:
  workload_id: str
  update_person_ldap: str
  description: Optional[str] = ""
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
  display_name: Optional[str] = None
