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

"""
A DAG to run MXLA MaxText tests.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName, Project
from dags.multipod.configs import gke_config

# Run once a day at 9 am UTC (1 am PST)
SCHEDULED_TIME = "0 9 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="mxla_gpt_6b_nightly_gke",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "gke", "nightly", "gpt_6b"],
    start_date=datetime.datetime(2024, 3, 18),
    catchup=False,
) as dag:
  jax_nightly_image = DockerImage.MAXTEXT_TPU_JAX_NIGHTLY
  default_gpt3_6b_test_name = "mxla-gpt3-6b-nightly-gke"

  gpt3_6b_nightly_1slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run()

  gpt3_6b_nightly_2slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      num_slices=2,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run()

  gpt3_6b_nightly_4slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      num_slices=4,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run()

  gpt3_6b_nightly_8slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      num_slices=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run()

  gpt3_6b_nightly_1slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      cluster_name=ClusterName.V5P_8_MULTISLICE_CLUSTER.value,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
      project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
  ).run()

  gpt3_6b_nightly_2slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      num_slices=2,
      cluster_name=ClusterName.V5P_8_MULTISLICE_CLUSTER.value,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
      project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
  ).run()

  gpt3_6b_nightly_4slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      num_slices=4,
      cluster_name=ClusterName.V5P_8_MULTISLICE_CLUSTER.value,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
      project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
  ).run()

  gpt3_6b_nightly_8slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      num_slices=8,
      cluster_name=ClusterName.V5P_8_MULTISLICE_CLUSTER.value,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
      project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
  ).run()

  (
      gpt3_6b_nightly_1slice_v4_8
      >> gpt3_6b_nightly_2slice_v4_8
      >> gpt3_6b_nightly_4slice_v4_8
      >> gpt3_6b_nightly_8slice_v4_8
  )

  (
      gpt3_6b_nightly_1slice_v5p_8
      >> gpt3_6b_nightly_2slice_v5p_8
      >> gpt3_6b_nightly_4slice_v5p_8
      >> gpt3_6b_nightly_8slice_v5p_8
  )
