import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class HardwareInfo:
  hardware_id: str
  gcp_accelerator_name: str
  chip_name: str
  bf_16_tflops: int
  memory: float
  hardware_type: str
  provider_name: str
  update_person_ldap: str
  chips_per_node: Optional[int] = None
  description: Optional[str] = ""
  other: Optional[str] = ""
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
  host_vcpus: Optional[int] = None
  host_memory: Optional[int] = None  # host_memory in GB
