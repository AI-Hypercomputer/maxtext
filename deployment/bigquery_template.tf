variable "bigquery_datasets" {
  type = list(object({
    id        = string
    location  = string
    env_stage = string
  }))
}

variable "bigquery_tables" {
  type = list(object({
    dataset_id     = string
    table_id       = string
    schema_id      = string
    partition_type = string
    env_stage      = string
  }))
}

locals {
  datasets = { for dataset in var.bigquery_datasets : dataset.id => dataset }
  tables   = { for table in var.bigquery_tables : "${table.dataset_id}-${table.table_id}" => table }
}

resource "google_bigquery_dataset" "dataset_setup" {
  project    = var.project_config.project_name
  for_each   = local.datasets
  dataset_id = each.value.id
  location   = each.value.location

  labels = {
    env = each.value.env_stage
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
    env = each.value.env_stage
  }
  depends_on = [
    google_bigquery_dataset.dataset_setup,
  ]
}