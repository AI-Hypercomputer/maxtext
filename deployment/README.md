# Cloud Composer Deployment

This page explains how to create a Cloud Composer 2 environment via Terraform. More detailed instruction can be found [here](https://cloud.google.com/composer/docs/composer-2/terraform-create-environments).

## Prerequisites
* You have a [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) to use.
* You have installed [gCloud CLI](https://cloud.google.com/sdk/docs/install).
* You have installed Terraform for [Google Cloud project](https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/install-cli?in=terraform%2Fgcp-get-started).
* You have [required permissions](https://cloud.google.com/storage/docs/creating-buckets#permissions-console) to create a GCS bucket.

## Step 1 - Define variables

Define all needed variables in `cloud_composer.auto.tfvars` file. You can add multiple entries into `environment_config` with unique `environment_name`.

```
project_config = {
  project_name   = <your_project_name>
  project_number = <your_project_number>
  project_region = <your_project_region>
}

environment_config = [
  {
    environment_name   = <your_environment_name>
    service_account_id = <your_custom_service_acount_id>
  },
  ...
]
```

Project configs:
* `project_name` is the name of the Google Cloud project.
* `project_number` is the number of the Google Cloud project.
* `project_region` is the location of the GKE cluster in the project, and it does not need to be same as your Virtual Machines.

Environment configs:
* `envrionment_name` is the name of the environment.
* `service_account_id` is the custom service account ID that will be created.

## Step 2 - Obtain access credentials

To authenticate with Google Cloud, run `gcloud auth application-default login`.

## Step 3 - Create a GCS bucket

To create a Google Cloud Storage (GCS) bucket to save Terraform state, run `gcloud storage buckets create gs://<your_tfstate_bucket_name>`. Please note this bucket name should be globally unique.

Then, update `bucket` name to yours in the `backend "gcs"` block of `cloud_composer_template.tf` file.

## Step 4 - Create Cloud Composer environment

Under this `deployment` directory, run commands below to create a new Cloud Composer environment with your custom service account.

```
terraform init
terraform plan
terraform apply -auto-approve
```

* `terraform init`: performs Backend Initialization, Child Module Installation, and Plugin Installation.
* `terraform plan`: shows what actions will be taken without actually performing the planned actions.
* `terraform apply -auto-approve`: applies changes without having to interactively type ‘yes’ to the plan.