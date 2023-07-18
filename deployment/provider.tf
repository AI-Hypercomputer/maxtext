provider "google-beta" {
  project = var.project_config.project_name
  region  = var.project_config.project_region
}

terraform {
  backend "gcs" {
    bucket = "composer-ml-auto-solutions-tfstate"
    prefix = "terraform/state"
  }
}