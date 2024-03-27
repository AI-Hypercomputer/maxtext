variable "project_config" {
  type = object({
    project_name   = string
    project_number = string
    project_region = string
  })
}

variable "environment_config" {
  type = list(
    object({
      environment_name   = string
      service_account_id = string
    })
  )
}

locals {
  environment_config_dict = {
    for o in var.environment_config : o.environment_name => o
  }
}

resource "google_project_service" "composer_api" {
  provider           = google-beta
  project            = var.project_config.project_name
  service            = "composer.googleapis.com"
  disable_on_destroy = false
}

resource "google_service_account" "custom_service_account" {
  provider     = google-beta
  for_each     = local.environment_config_dict
  account_id   = each.value.service_account_id
  display_name = "ML Automation Solutions Service Account"
}

resource "google_project_iam_member" "composer_worker_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/composer.worker"
}

resource "google_project_iam_member" "tpu_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/tpu.admin"
}

resource "google_project_iam_member" "service_account_user_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/iam.serviceAccountUser"
}

resource "google_project_iam_member" "big_query_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/bigquery.admin"
}

resource "google_project_iam_member" "storage_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/storage.admin"
}

resource "google_project_iam_member" "vertex_ai_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/aiplatform.admin"
}

resource "google_project_iam_member" "artifact_registry_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/artifactregistry.admin"
}

resource "google_project_iam_member" "monitoring_viewer_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/monitoring.viewer"
}

resource "google_project_iam_member" "compute_instance_admin_role" {
  provider = google-beta
  for_each = local.environment_config_dict
  project  = var.project_config.project_name
  member   = format("serviceAccount:%s", google_service_account.custom_service_account[each.key].email)
  role     = "roles/compute.instanceAdmin.v1"
}

resource "google_service_account_iam_member" "custom_service_account" {
  provider           = google-beta
  for_each           = local.environment_config_dict
  service_account_id = google_service_account.custom_service_account[each.key].name
  role               = "roles/composer.ServiceAgentV2Ext"
  member             = "serviceAccount:service-${var.project_config.project_number}@cloudcomposer-accounts.iam.gserviceaccount.com"
}

resource "google_composer_environment" "example_environment" {
  provider = google-beta
  for_each = local.environment_config_dict
  name     = each.value.environment_name

  config {
    environment_size = "ENVIRONMENT_SIZE_MEDIUM"
    software_config {
      image_version = "composer-2.6.1-airflow-2.6.3"
      airflow_config_overrides = {
        # TODO: Update this to allowed_deserialization_classes_regexp with Airflow 2.8.1
        # https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#allowed-deserialization-classes-regexp
        core-allowed_deserialization_classes = ".*"
        scheduler-min_file_process_interval  = "120"
      }
      # Note: keep this in sync with .github/requirements.txt
      pypi_packages = {
        apache-airflow-providers-sendgrid = ""
        fabric                            = ""
        google-cloud-tpu                  = ">=1.16.0"
        jsonlines                         = ""
        # These packages are already in the default composer environment.
        # See https://cloud.google.com/composer/docs/concepts/versioning/composer-versions
        # google-cloud-bigquery             = ""
        # google-cloud-storage              = ""
        # google-cloud-container            = ""
        # tensorflow-cpu                    = ""
        # kubernetes                        = ""
      }
    }

    workloads_config {
      scheduler {
        cpu        = 28
        memory_gb  = 80
        storage_gb = 10
        count      = 2
      }
      web_server {
        cpu        = 2
        memory_gb  = 8
        storage_gb = 10
      }
      worker {
        cpu        = 8
        memory_gb  = 48
        storage_gb = 10
        min_count  = 1
        max_count  = 100
      }
    }

    node_config {
      service_account = google_service_account.custom_service_account[each.key].email
    }
  }
}
