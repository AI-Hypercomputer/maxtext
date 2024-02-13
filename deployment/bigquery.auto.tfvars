bigquery_datasets = [
  {
    id        = "benchmark_dataset"
    location  = "US"
    env_stage = "prod"
  },
  {
    id        = "xlml_dataset"
    location  = "US"
    env_stage = "prod"
  },
  {
    id        = "dev_benchmark_dataset"
    location  = "US"
    env_stage = "dev"
  },
  {
    id        = "dev_xlml_dataset"
    location  = "US"
    env_stage = "dev"
  }
]

bigquery_tables = [
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "benchmark_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "xlml_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
    env_stage      = "prod"
  },
  {
    dataset_id     = "dev_benchmark_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  },
  {
    dataset_id     = "dev_benchmark_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  },
  {
    dataset_id     = "dev_benchmark_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  },
  {
    dataset_id     = "dev_xlml_dataset"
    table_id       = "job_history"
    schema_id      = "schema/job_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  },
  {
    dataset_id     = "dev_xlml_dataset"
    table_id       = "metric_history"
    schema_id      = "schema/metric_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  },
  {
    dataset_id     = "dev_xlml_dataset"
    table_id       = "metadata_history"
    schema_id      = "schema/metadata_history.json"
    partition_type = "MONTH"
    env_stage      = "dev"
  }
]