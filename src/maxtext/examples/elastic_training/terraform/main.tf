# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.pathways.endpoint}"
  cluster_ca_certificate = base64decode(google_container_cluster.pathways.master_auth[0].cluster_ca_certificate)
  token                  = data.google_client_config.default.access_token
}

data "google_client_config" "default" {}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "tpu.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# GCS bucket for checkpoints and Pathways scratch
resource "google_storage_bucket" "training" {
  name          = var.gcs_bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.apis]
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "maxtext" {
  location      = var.region
  repository_id = "maxtext-elastic"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# Service account for GKE workloads (Pathways + MaxText)
resource "google_service_account" "gke_workload" {
  account_id   = "maxtext-elastic-workload"
  display_name = "GKE workload SA for ${var.cluster_name}"

  depends_on = [google_project_service.apis]
}

# Workload SA needs full GCS access for checkpoints and Pathways scratch
resource "google_storage_bucket_iam_member" "workload_gcs" {
  bucket = google_storage_bucket.training.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gke_workload.email}"
}

# Workload SA needs AR read access to pull training images.
# Granted at the project level so it works consistently across organization IAM setups.
resource "google_project_iam_member" "workload_ar" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_workload.email}"
}

# Workload SA needs logging access
resource "google_project_iam_member" "workload_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_workload.email}"
}

# Workload SA needs monitoring access
resource "google_project_iam_member" "workload_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_workload.email}"
}

# Cloud Build service account. Dedicated SA (new projects disable the legacy
# default compute SA), passed to `gcloud builds submit --service-account`.
resource "google_service_account" "cloudbuild" {
  account_id   = "maxtext-elastic-build"
  display_name = "Cloud Build SA for ${var.cluster_name}"

  depends_on = [google_project_service.apis]
}

locals {
  cloudbuild_sa = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Build logs (required when submitting with a user-specified SA + CLOUD_LOGGING_ONLY)
resource "google_project_iam_member" "cloudbuild_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = local.cloudbuild_sa
}

# Cloud Build SA needs GKE access to get credentials, install APIs, and deploy manifests
resource "google_project_iam_member" "cloudbuild_gke" {
  project = var.project_id
  role    = "roles/container.admin"
  member  = local.cloudbuild_sa
}

# Cloud Build SA needs storage access to read source and check GCS during data prep
resource "google_project_iam_member" "cloudbuild_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = local.cloudbuild_sa
}

# Cloud Build SA needs AR access to push images.
# Granted at the project level (same rationale as workload_ar above).
resource "google_project_iam_member" "cloudbuild_ar" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = local.cloudbuild_sa
}

# Cloud Build SA needs Secret Manager access to read HF token
resource "google_project_iam_member" "cloudbuild_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = local.cloudbuild_sa
}

# GKE cluster
resource "google_container_cluster" "pathways" {
  name     = var.cluster_name
  location = var.zone

  # Empty cluster_version => omit the pin and let the RAPID channel choose.
  min_master_version = var.cluster_version != "" ? var.cluster_version : null

  network    = var.network
  subnetwork = var.subnetwork

  initial_node_count       = 1
  remove_default_node_pool = true
  deletion_protection      = false

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  release_channel {
    channel = "RAPID"
  }

  depends_on = [google_project_service.apis]
}

# CPU node pool for Pathways controller (resource manager + proxy server)
resource "google_container_node_pool" "cpu_pathways" {
  name     = "cpu-pathways-np"
  location = var.zone
  cluster  = google_container_cluster.pathways.name

  node_count = var.num_cpu_nodes

  node_config {
    machine_type    = var.cpu_machine_type
    service_account = google_service_account.gke_workload.email

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    workload_metadata_config {
      mode = "GCE_METADATA"
    }
  }
}

# TPU node pools, one per slice
resource "google_container_node_pool" "tpu_slices" {
  count = var.num_tpu_slices

  name     = "tpu-np-${count.index + 1}"
  location = var.zone
  cluster  = google_container_cluster.pathways.name

  node_count = var.nodes_per_slice

  node_config {
    machine_type    = var.tpu_machine_type
    service_account = google_service_account.gke_workload.email

    # Spot TPUs: cheaper, reclaimable. Off by default; opt in with tpu_spot=true.
    spot = var.tpu_spot

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    workload_metadata_config {
      mode = "GCE_METADATA"
    }

    dynamic "reservation_affinity" {
      for_each = var.tpu_reservation != "" ? [1] : []
      content {
        consume_reservation_type = "SPECIFIC_RESERVATION"
        key                      = "compute.googleapis.com/reservation-name"
        values                   = [var.tpu_reservation]
      }
    }
  }

  placement_policy {
    type         = "COMPACT"
    tpu_topology = var.tpu_topology
  }

  # Spot and a specific reservation are mutually exclusive on GCP; catch the
  # conflict at plan time with a clear message instead of an opaque apply error.
  lifecycle {
    precondition {
      condition     = !(var.tpu_spot && var.tpu_reservation != "")
      error_message = "tpu_spot and tpu_reservation are mutually exclusive: Spot TPUs cannot consume a specific reservation."
    }
  }
}

# HuggingFace token, stored in Secret Manager (Cloud Build) and K8s (pods)
resource "google_secret_manager_secret" "hf_token" {
  secret_id = "maxtext-elastic-hf-token"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "hf_token" {
  secret      = google_secret_manager_secret.hf_token.id
  secret_data = var.hf_token
}

resource "kubernetes_secret_v1" "hf_token" {
  metadata {
    name      = "hf-token"
    namespace = "default"
  }

  data = {
    token = var.hf_token
  }

  depends_on = [
    google_container_cluster.pathways,
    google_container_node_pool.cpu_pathways,
  ]
}
