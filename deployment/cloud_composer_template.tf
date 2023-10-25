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
    software_config {
      image_version = "composer-2.3.3-airflow-2.5.1"
      airflow_config_overrides = {
        core-allowed_deserialization_classes = ".*"
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
        # tensorflow-cpu                    = ""
      }
    }

    node_config {
      service_account = google_service_account.custom_service_account[each.key].email
    }
  }
}
