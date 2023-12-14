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

"""A DAG to run all supported ML models with the nightly TensorFlow version."""

import datetime
from airflow import models
from configs import composer_env, vm_resource
from configs.xlml.tensorflow import solutionsteam_tf_nightly_supported_config as tf_config


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="tf_latest_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # ResNet
  tf_resnet_v2_8 = tf_config.get_tf_resnet_config(
      tpu_version=2,
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
  ).run()

  tf_resnet_v3_8 = tf_config.get_tf_resnet_config(
      tpu_version=3,
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  tf_resnet_v4_8 = tf_config.get_tf_resnet_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  tf_resnet_v4_32 = tf_config.get_tf_resnet_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_pod=True,
  ).run()

  # Test dependencies
  tf_resnet_v2_8
  tf_resnet_v3_8
  tf_resnet_v4_8 >> tf_resnet_v4_32
