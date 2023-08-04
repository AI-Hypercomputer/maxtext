bigquery_benchmark_dataset = {
  id       = "benchmark_dataset"
  location = "US"
}

bigquery_benchmark_table = [
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "job_history"
    schema_id      = "schema/benchmark_job_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/benchmark_metric_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/benchmark_metadata_history.json"
    partition_type = "MONTH"
  }
]