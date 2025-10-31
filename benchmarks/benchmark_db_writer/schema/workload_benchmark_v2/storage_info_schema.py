import dataclasses
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class StorageInfo:
  storage_id: str
  storage_product: Optional[str]
  description: Optional[str]
  config: Optional[dict]
  update_person_ldap: Optional[str]
  update_timestamp: Optional[bigquery_types.TimeStamp]
