bigquery_datasets = [
  {
    id       = "benchmark_dataset"
    location = "US"
  },
  {
    id       = "xlml_dataset"
    location = "US"
  }
]

bigquery_tables = [
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
  }
]