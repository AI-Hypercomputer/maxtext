variable "bigquery_benchmark_dataset" {
  type = object({
    id       = string
    location = string
  })
}

variable "bigquery_benchmark_table" {
  type = list(object({
    dataset_id     = string
    table_id       = string
    schema_id      = string
    partition_type = string
  }))
}

locals {
  tables = { for table in var.bigquery_benchmark_table : table.table_id => table }
}

resource "google_bigquery_dataset" "dataset_setup" {
  project    = var.project_config.project_name
  dataset_id = var.bigquery_benchmark_dataset.id
  location   = var.bigquery_benchmark_dataset.location

  labels = {
    env = "prod"
  }
}

resource "google_bigquery_table" "table_setup" {
  for_each   = local.tables
  project    = var.project_config.project_name
  dataset_id = each.value.dataset_id
  table_id   = each.value.table_id
  schema     = file(each.value.schema_id)

  time_partitioning {
    type = each.value.partition_type
  }

  labels = {
    env = "prod"
  }

  depends_on = [
    google_bigquery_dataset.dataset_setup,
  ]
}