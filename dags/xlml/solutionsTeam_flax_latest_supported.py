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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from airflow import models
from configs.xlml.jax import solutionsTeam_flax_latest_supported_config as flax_config

US_CENTRAL2_B = "us-central2-b"

with models.DAG(
    dag_id="flax_latest_supported",
    schedule="0 2 * * *",  # Run once a day at 2 am
    tags=["solutions_team", "flax", "latest", "supported"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # ResNet
  jax_resnet_v4_8 = flax_config.get_flax_resnet_config(
      tpu_version=4,
      tpu_cores=8,
      tpu_zone=US_CENTRAL2_B,
      data_dir="gs://xl-ml-test-us-central2/tfds-data",
      time_out_in_min=60,
  ).run()

  # GPT2
  jax_gpt2_v4_extra_flags = [
      "--per_device_train_batch_size=64",
      "--per_device_eval_batch_size=64",
  ]
  jax_gpt2_v4_8 = flax_config.get_flax_gpt2_config(
      tpu_version=4,
      tpu_cores=8,
      tpu_zone=US_CENTRAL2_B,
      time_out_in_min=120,
      extraFlags=" ".join(jax_gpt2_v4_extra_flags),
  ).run()

  # Stable Diffusion
  jax_sd_v4_8 = flax_config.get_flax_sd_config(
      tpu_version=4,
      tpu_cores=8,
      tpu_zone=US_CENTRAL2_B,
      time_out_in_min=60,
      num_train_epochs=1,
  ).run()

  # Run ResNet model first to ensure E2E functionality
  jax_resnet_v4_8 >> [jax_gpt2_v4_8, jax_sd_v4_8]
