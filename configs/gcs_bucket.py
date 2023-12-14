# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCS bucket for data."""

# GCS bucket for training data
# TODO(ranran): migrate data to `cloud-ml-auto-solutions` project
IMAGENET_DIR = "gs://xl-ml-test-us-central2/data/imagenet"
TFDS_DATA_DIR = "gs://xl-ml-test-us-central2/tfds-data"
PAX_DIR = "gs://cloud-tpu-checkpoints/pax"

# GCS bucket for output
BENCHMARK_OUTPUT_DIR = "gs://ml-auto-solutions/output/benchmark"
XLML_OUTPUT_DIR = "gs://ml-auto-solutions/output/xlml"
