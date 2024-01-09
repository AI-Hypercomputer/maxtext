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
from configs import composer_env
from configs.vm_resource import TpuVersion, Project, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from configs.xlml.tensorflow import solutionsteam_tf_nightly_supported_config as tf_config
from configs.xlml.tensorflow import common
from airflow.operators.dummy import DummyOperator


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="tf_nightly_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # Keras - tests run in sequence order
  tf_keras_v2_8 = [DummyOperator(task_id="tf_nightly_keras_v2-8")]
  for feature, name in common.FEATURE_NAME.items():
    test = tf_config.get_tf_keras_config(
        tpu_version=TpuVersion.V2,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=common.FEATURE_TIMEOUT.get(feature),
        test_feature=feature,
        test_name=name,
    ).run()
    tf_keras_v2_8[-1] >> test
    tf_keras_v2_8.append(test)

  tf_keras_v5e_4 = [DummyOperator(task_id="tf_nightly_keras_v5litepod-4")]
  for feature, name in common.FEATURE_NAME.items():
    test = tf_config.get_tf_keras_config(
        project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
        tpu_version=TpuVersion.V5E,
        tpu_cores=4,
        tpu_zone=Zone.US_EAST1_C.value,
        time_out_in_min=common.FEATURE_TIMEOUT.get(feature),
        test_feature=feature,
        test_name=name,
        network=V5_NETWORKS,
        subnetwork=V5E_SUBNETWORKS,
    ).run()
    tf_keras_v5e_4[-1] >> test
    tf_keras_v5e_4.append(test)

  tf_keras_v5p_8 = [DummyOperator(task_id="tf_nightly_keras_v5p-8")]
  for feature, name in common.FEATURE_NAME.items():
    test = tf_config.get_tf_keras_config(
        project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
        tpu_version=TpuVersion.V5P,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_A.value,
        time_out_in_min=common.FEATURE_TIMEOUT.get(feature),
        test_feature=feature,
        test_name=name,
        network=V5_NETWORKS,
        subnetwork=V5P_SUBNETWORKS,
    ).run()
    tf_keras_v5p_8[-1] >> test
    tf_keras_v5p_8.append(test)

  # ResNet
  tf_resnet_v2_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
      global_batch_size=1024,
  ).run()

  tf_resnet_v3_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  tf_resnet_v4_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  tf_resnet_v4_32 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_pod=True,
      runtime_version=RuntimeVersion.TPU_VM_TF_2150_POD_PJRT.value,
  ).run()

  tf_resnet_v5e_4 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=4,
      tpu_zone=Zone.US_EAST1_C.value,
      time_out_in_min=60,
      global_batch_size=2048,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
  ).run()

  tf_resnet_v5e_16 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=16,
      tpu_zone=Zone.US_EAST1_C.value,
      time_out_in_min=60,
      global_batch_size=2048,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      is_pod=True,
      runtime_version=RuntimeVersion.TPU_VM_TF_2150_POD_PJRT.value,
  ).run()

  tf_resnet_v5p_8 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
  ).run()

  tf_resnet_v5p_32 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      is_pod=True,
      runtime_version=RuntimeVersion.TPU_VM_TF_2150_POD_PJRT.value,
  ).run()

  # Test dependencies
  tf_keras_v2_8
  tf_keras_v5e_4
  tf_keras_v5p_8
  tf_resnet_v2_8
  tf_resnet_v3_8
  tf_resnet_v4_8 >> tf_resnet_v4_32
  tf_resnet_v5e_4 >> tf_resnet_v5e_16
  tf_resnet_v5p_8 >> tf_resnet_v5p_32
