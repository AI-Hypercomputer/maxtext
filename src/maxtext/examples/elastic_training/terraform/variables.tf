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

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the GKE cluster"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the GKE cluster and TPU node pools"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "maxtext-elastic-training"
}

variable "cluster_version" {
  description = "Pin a GKE master version (e.g. 1.35.3-gke.1993000). Leave empty to let the RAPID release channel pick the current version, recommended, since a pinned patch eventually ages out of the channel and breaks 'terraform apply'."
  type        = string
  default     = ""
}

variable "network" {
  description = "VPC network name"
  type        = string
  default     = "default"
}

variable "subnetwork" {
  description = "VPC subnetwork name"
  type        = string
  default     = "default"
}

# TPU configuration
variable "tpu_machine_type" {
  description = "TPU machine type for worker node pools"
  type        = string
  default     = "ct5lp-hightpu-4t"
}

variable "tpu_topology" {
  description = "TPU topology per slice"
  type        = string
  default     = "4x4"
}

variable "num_tpu_slices" {
  description = "Number of TPU slices (node pools)"
  type        = number
  default     = 3
}

variable "nodes_per_slice" {
  description = "Number of nodes per TPU node pool"
  type        = number
  default     = 4
}

# CPU node pool for Pathways controller
variable "cpu_machine_type" {
  description = "Machine type for the Pathways CPU node pool"
  type        = string
  default     = "n2-standard-64"
}

variable "num_cpu_nodes" {
  description = "Number of CPU nodes for Pathways controller"
  type        = number
  default     = 1
}

# HuggingFace token (Qwen3 is ungated, but wired through for swapping to gated models)
variable "hf_token" {
  description = "HuggingFace API token for accessing gated models"
  type        = string
  sensitive   = true
}

# TPU reservation (optional, use if you have a reserved capacity)
variable "tpu_reservation" {
  description = "TPU reservation name. Leave empty to use on-demand."
  type        = string
  default     = ""
}

# Spot (preemptible) TPUs, cheaper, but can be reclaimed at any time. Handy for
# this demo since elastic recovery already tolerates a slice disappearing. Leave
# false for on-demand or when using a reservation (the two are mutually exclusive).
variable "tpu_spot" {
  description = "Use Spot (preemptible) TPU VMs instead of on-demand."
  type        = bool
  default     = false
}

# GCS bucket for checkpoints and Pathways scratch
variable "gcs_bucket_name" {
  description = "GCS bucket name for checkpoints and Pathways artifacts"
  type        = string
}
