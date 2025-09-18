import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class ModelInfo:
  model_id: str
  name: str
  variant: str
  parameter_size_in_billions: float
  update_person_ldap: str
  description: Optional[str] = ""
  details: Optional[str] = ""
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
