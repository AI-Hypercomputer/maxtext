import dataclasses
import datetime
from typing import Optional
from benchmarks.benchmark_db_writer import bigquery_types


@dataclasses.dataclass
class WorkloadBenchmarkV2Schema:
  run_id: str

  # Unique model id to map model info table
  model_id: str

  # Foreign  key to join with  software info
  software_id: str
  # Foreign  key to join with hardware info
  hardware_id: str
  hardware_num_chips: int

  result_success: bool

  update_person_ldap: str
  configs_framework: Optional[str] = None
  configs_container_version: Optional[str] = None
  logs_artifact_directory: Optional[str] = None
  configs_env: Optional[str] = None
  hardware_num_nodes: Optional[int] = None

  # Foreign key to join with storage info
  storage_id: Optional[str] = None

  run_source: str = "manual"
  is_run_externally_visible: bool = False
  run_type: str = "perf_optimization"

  run_release_status: Optional[str] = "local"
  k8_jobset_yaml_file_path: Optional[str] = None

  benchmark_type: Optional[str] = None

  experiment_id: Optional[str] = None

  workload_gbs: Optional[int] = None
  workload_mbs: Optional[int] = None
  workload_precision: Optional[str] = None
  workload_optimizer: Optional[str] = None
  workload_others: Optional[str] = None
  workload_manager: Optional[str] = None
  workload_type: str = "training"
  workload_sequence_length: Optional[int] = None

  metrics_step_time: Optional[float] = None
  metrics_mfu: Optional[float] = None
  metrics_tokens_per_second: Optional[float] = None
  metrics_e2e_time: Optional[float] = None
  metrics_num_steps: Optional[int] = None
  metrics_other: Optional[str] = None
  metrics_tflops_per_second: Optional[float] = None

  hardware_num_superblocks: Optional[str] = None
  hardware_num_slices: Optional[int] = None
  hardware_topology: Optional[str] = None
  hardware_num_cores: Optional[int] = None
  result_error: Optional[str] = None
  hardware_nccl_driver_nickname: Optional[str] = None

  configs_xla_flags: Optional[str] = None
  configs_dataset: Optional[str] = None
  configs_reviewer: Optional[str] = None
  configs_other: Optional[str] = None

  logs_profile: Optional[str] = None
  logs_cloud_logs: Optional[str] = None
  logs_comments: Optional[str] = None
  logs_other: Optional[str] = None

  checkpointing_async: Optional[bool] = None
  checkpointing_interval_every_n_steps: Optional[int] = None
  checkpointing_size_in_gibs: Optional[float] = None
  checkpointing_individual_file_size: Optional[int] = None
  checkpointing_file_format: Optional[str] = None

  max_epochs: Optional[int] = None
  max_steps: Optional[int] = None
  training_dataset_samples: Optional[int] = None
  data_loader_num_workers: Optional[int] = None
  data_loader_prefetch_factor: Optional[int] = None
  training_dataset_file_format: Optional[str] = None

  start_time: Optional[bigquery_types.TimeStamp] = None
  end_time: Optional[bigquery_types.TimeStamp] = None

  gcs_metrics_bucket: Optional[str] = None
  gcsfuse_csi_driver: Optional[str] = None
  cloud_region: Optional[str] = None
  source_bucket: Optional[str] = None

  cluster_name: Optional[str] = None

  reviewer_ldap: str = ""
  update_timestamp: Optional[bigquery_types.TimeStamp] = datetime.datetime.now()
