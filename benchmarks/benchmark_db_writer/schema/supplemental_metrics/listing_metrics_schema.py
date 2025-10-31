import dataclasses
from typing import Optional


@dataclasses.dataclass
class ListingMetricsInfo:
  run_id: str
  first_iteration_metrics_gcs_list_time_avg: Optional[float]
  first_iteration_metrics_fuse_list_time_avg: Optional[float]
  first_iteration_metrics_gcs_subdir_list_time_avg: Optional[float]
  first_iteration_metrics_gcs_list_time_p50: Optional[float]
  first_iteration_metrics_fuse_list_time_p50: Optional[float]
  first_iteration_metrics_gcs_subdir_list_time_p50: Optional[float]
  first_iteration_metrics_gcs_list_time_p90: Optional[float]
  first_iteration_metrics_fuse_list_time_p90: Optional[float]
  first_iteration_metrics_gcs_subdir_list_time_p90: Optional[float]
  first_iteration_metrics_gcs_list_time_p99: Optional[float]
  first_iteration_metrics_fuse_list_time_p99: Optional[float]
  first_iteration_metrics_gcs_subdir_list_time_p99: Optional[float]
  first_iteration_metrics_gcs_list_time_p100: Optional[float]
  first_iteration_metrics_fuse_list_time_p100: Optional[float]
  first_iteration_metrics_gcs_subdir_list_time_p100: Optional[float]
  subsq_iteration_metrics_gcs_list_time_avg: Optional[float]
  subsq_iteration_metrics_fuse_list_time_avg: Optional[float]
  subsq_iteration_metrics_gcs_subdir_list_time_avg: Optional[float]
  subsq_iteration_metrics_gcs_list_time_p50: Optional[float]
  subsq_iteration_metrics_fuse_list_time_p50: Optional[float]
  subsq_iteration_metrics_gcs_subdir_list_time_p50: Optional[float]
  subsq_iteration_metrics_gcs_list_time_p90: Optional[float]
  subsq_iteration_metrics_fuse_list_time_p90: Optional[float]
  subsq_iteration_metrics_gcs_subdir_list_time_p90: Optional[float]
  subsq_iteration_metrics_gcs_list_time_p99: Optional[float]
  subsq_iteration_metrics_fuse_list_time_p99: Optional[float]
  subsq_iteration_metrics_gcs_subdir_list_time_p99: Optional[float]
  subsq_iteration_metrics_gcs_list_time_p100: Optional[float]
  subsq_iteration_metrics_fuse_list_time_p100: Optional[float]
  subsq_iteration_metrics_gcs_subdir_list_time_p100: Optional[float]
