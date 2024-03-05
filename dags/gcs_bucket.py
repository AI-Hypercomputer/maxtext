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
CRITEO_DIR = "gs://ml-auto-solutions/data/criteo/terabyte_processed_shuffled"
IMAGENET_DIR = "gs://ml-auto-solutions/data/imagenet"
TFDS_DATA_DIR = "gs://ml-auto-solutions/data/tfds-data"
PAX_DIR = "gs://cloud-tpu-checkpoints/pax"
MAXTEXT_DIR = "gs://max-datasets-rogue"

# GCS bucket for output
BENCHMARK_OUTPUT_DIR = "gs://ml-auto-solutions/output/benchmark"
XLML_OUTPUT_DIR = "gs://ml-auto-solutions/output/xlml"
