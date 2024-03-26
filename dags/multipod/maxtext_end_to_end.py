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

"""A DAG to run end-to-end MaxText tests."""


import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, DockerImage, GpuVersion, ClusterName
from dags.multipod.configs import gke_config


# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_end_to_end",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"
  test_models = {
      "llama2": ["test_llama2_7b"],
      "mistral": ["test_mistral"],
      "gemma": ["test_gemma"],
      "gpt3": ["test_gpt3"],
  }

  for model in test_models.keys():
    for test_script in test_models[model]:
      stable_tpu = gke_config.get_gke_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"{test_name_prefix}-stable-{test_script}",
          run_model_cmds=(f"bash end_to_end/{test_script}.sh",),
          docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
          test_owner=test_owner.JON_B,
      ).run()
      nightly_tpu = gke_config.get_gke_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"{test_name_prefix}-nightly-{test_script}",
          run_model_cmds=(f"bash end_to_end/{test_script}.sh",),
          docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
          test_owner=test_owner.JON_B,
      ).run()
      stable_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
          accelerator_type=GpuVersion.XPK_H100,
          gpu_zone=Zone.US_CENTRAL1_C.value,
          time_out_in_min=300,
          test_name=f"{test_name_prefix}-stable-{test_script}",
          test_script=test_script,
          num_slices=2,
          cluster_name=ClusterName.A3_CLUSTER.value,
          docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
          test_owner=test_owner.NINA_C,
      ).run()
      nightly_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
          accelerator_type=GpuVersion.XPK_H100,
          gpu_zone=Zone.US_CENTRAL1_C.value,
          time_out_in_min=300,
          test_name=f"{test_name_prefix}-nightly-{test_script}",
          test_script=test_script,
          num_slices=2,
          cluster_name=ClusterName.A3_CLUSTER.value,
          docker_image=DockerImage.MAXTEXT_GPU_JAX_NIGHTLY.value,
          test_owner=test_owner.NINA_C,
      ).run()
      stable_tpu >> nightly_tpu >> stable_gpu >> nightly_gpu
