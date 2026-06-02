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

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.pathways.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.pathways.endpoint
  sensitive   = true
}

output "cluster_zone" {
  description = "GKE cluster zone"
  value       = google_container_cluster.pathways.location
}

output "gcs_bucket" {
  description = "GCS bucket for checkpoints and Pathways scratch"
  value       = google_storage_bucket.training.name
}

output "tpu_node_pools" {
  description = "TPU node pool names"
  value       = [for np in google_container_node_pool.tpu_slices : np.name]
}

output "workload_service_account" {
  description = "GKE workload service account email"
  value       = google_service_account.gke_workload.email
}

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "cloudbuild_service_account" {
  description = "Dedicated Cloud Build service account email"
  value       = google_service_account.cloudbuild.email
}

output "cloudbuild_command" {
  description = "Command to run the Cloud Build pipeline"
  value       = "gcloud builds submit --config=deploy/deploy.yaml --service-account=projects/${var.project_id}/serviceAccounts/${google_service_account.cloudbuild.email} --substitutions=_CLUSTER_NAME=${google_container_cluster.pathways.name},_CLUSTER_ZONE=${google_container_cluster.pathways.location},_BUCKET_NAME=${google_storage_bucket.training.name} --project=${var.project_id}"
}
