# ML Automation Solutions

A simplified and automated orchestration workflow to perform ML end-to-end (E2E) model tests and benchmarking on Cloud VMs across different frameworks.

## Getting Started

1. Follow the [instruction](deployment/README.md) to create a Cloud Composer environment using Terraform. This step may take about 30min to complete.
2. Identify your `dags` folder. See the instructions to [access the bucket of your environment](https://cloud.google.com/composer/docs/composer-2/manage-dags#console).
3. In the root directory of the repository, run the following command to upload tests and utilities to the `dags` folder you identified in the previous step ([gsutil command-line tool](https://cloud.google.com/storage/docs/gsutil_install) is required).
```
bash scripts/upload-tests.sh gs://<your_bucket_name>/dags
```
4. After the automatically scheduled tests start running, integrate [Looker Studio](https://cloud.google.com/bigquery/docs/bi-engine-looker-studio) or any other dashboard with BigQuery to monitor metrics.

If you have a use case that ML Automation Solutions does not cover, please email ml-testing-accelerators-users@googlegroups.com. We're here to help!

## Contributing

Thank you for your interest in contributing to this project!

Please review the [contribution guidelines](docs/contributing.md), and note that all contributions must adhere to the [code of conduct](docs/code-of-conduct.md).

## License

[Apache License 2.0](LICENSE)