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
from dags.vm_resource import TpuVersion, Project, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from dags.solutions_team.configs.tensorflow import solutionsteam_tf_release_supported_config as tf_config
from dags.solutions_team.configs.tensorflow import common


# Release tests only need to run once, they can be run manually as needed
SCHEDULED_TIME = None
VERSION = f"{tf_config.MAJOR_VERSION}.{tf_config.MINOR_VERSION}"


with models.DAG(
    dag_id=f"tf_{tf_config.MAJOR_VERSION}_{tf_config.MINOR_VERSION}_se_nightly_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "se", VERSION, "supported", "xlml"],
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
  embedding_dim = 16
  tf_dlrm_v2_8 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  embedding_dim = 64
  tf_dlrm_v2_32 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL1_A.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=256054,
      extraFlags="--mode=train_and_eval",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  embedding_dim = 64
  tf_dlrm_v4_8 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  embedding_dim = 128
  tf_dlrm_v4_32 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=256054,
      extraFlags="--mode=train_and_eval",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  ).run()

  embedding_dim = 32
  tf_dlrm_v5p_8 = tf_config.get_tf_dlrm_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pod=False,
      is_pjrt=False,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      runtime_version=RuntimeVersion.TPU_VM_TF_V5P_ALPHA.value,
  ).run()

  # Test dependencies
  tf_keras_v2_8
  tf_resnet_v2_8 >> tf_resnet_v2_32
  tf_resnet_v3_8 >> tf_resnet_v3_32
  tf_resnet_v4_8 >> tf_resnet_v4_32
  tf_dlrm_v2_8 >> tf_dlrm_v2_32
  tf_dlrm_v4_8 >> tf_dlrm_v4_32
  tf_dlrm_v5p_8
