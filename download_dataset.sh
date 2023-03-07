#!/bin/bash
# This script downloads c4/en/3.0.1 to your gcs bucket directory
# Usage bash download_dataset.sh <<gcp project>> <<gcs bucket name>>
# Usage example: bash download_dataset.sh cloud-tpu-multipod-dev gs://maxtext-dataset
gsutil -u $1 -m cp 'gs://allennlp-tensorflow-datasets/c4/en/3.0.1/*' $2/c4/en/3.0.1
