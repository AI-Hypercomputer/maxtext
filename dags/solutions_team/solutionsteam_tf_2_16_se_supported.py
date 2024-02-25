# Copyright 2024 Google LLC
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
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from dags.solutions_team.configs.tensorflow import solutionsteam_tf_2_16_supported_config as tf_config
from dags.solutions_team.configs.tensorflow import common


# Run once a day at 8 pm UTC (12 pm PST)
SCHEDULED_TIME = "0 20 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="tf_2_16_se_nightly_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "se", "2.16", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 4),
    catchup=False,
) as dag:
  # Keras - tests run in sequence order
  tf_keras_v2_8 = []
  for feature, name in common.FEATURE_NAME.items():
    test = tf_config.get_tf_keras_config(
        tpu_version=TpuVersion.V2,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=common.FEATURE_TIMEOUT.get(feature),
        test_feature=feature,
        test_name=name,
        is_pjrt=False,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
    ).run()
    if tf_keras_v2_8:
      tf_keras_v2_8[-1] >> test
    tf_keras_v2_8.append(test)

  # ResNet
  tf_resnet_v2_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
      global_batch_size=1024,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_resnet_v2_32 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL1_A.value,
      time_out_in_min=60,
      global_batch_size=1024,
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_resnet_v3_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_resnet_v3_32 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_resnet_v4_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()
  tf_resnet_v4_32 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()
  # DLRM
  tf_dlrm_v2_8 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, 16],
      embedding_dim=16,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_dlrm_v2_32 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL1_A.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, 64],
      embedding_dim=64,
      train_steps=256054,
      extraFlags="--mode=train_and_eval",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_dlrm_v4_8 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, 64],
      embedding_dim=64,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  tf_dlrm_v4_32 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, 128],
      embedding_dim=128,
      train_steps=256054,
      extraFlags="--mode=train_and_eval",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  # Test dependencies
  tf_keras_v2_8
  tf_resnet_v2_8 >> tf_resnet_v2_32
  tf_resnet_v3_8 >> tf_resnet_v3_32
  tf_resnet_v4_8 >> tf_resnet_v4_32
  tf_dlrm_v2_8 >> tf_dlrm_v2_32
  tf_dlrm_v4_8 >> tf_dlrm_v4_32
