import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class SoftwareInfo:
  software_id: str
  ml_framework: str
  os: str
  training_framework: str
  update_person_ldap: str
  compiler: Optional[str] = None
  description: Optional[str] = ""
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
