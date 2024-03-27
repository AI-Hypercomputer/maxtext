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
A DAG to run MaxText checkpointing tests.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_checkpointing",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_checkpointing"
  dataset_path = gcs_bucket.MAXTEXT_DIR
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]
  test_configs = {
      # accelerator: list of slices to test
      "v4-8": [1],
      "v4-16": [1, 2],
  }
  cluster_names = {
      # accelerator: cluster name
      "v4-8": ClusterName.V4_8_MULTISLICE_CLUSTER,
      "v4-16": ClusterName.V4_16_MULTISLICE_CLUSTER,
  }

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      cores = accelerator.rsplit("-", maxsplit=1)[-1]
      for slice_num in slices:
        command = (
            "bash end_to_end/test_checkpointing.sh"
            f" checkpointing-{mode.value}-{slice_num}x-{accelerator}"
            f" {base_output_directory} {dataset_path} true",
        )
        maxtext_v4_configs_test = gke_config.get_gke_config(
            tpu_version=TpuVersion.V4,
            tpu_cores=cores,
            num_slices=slice_num,
            cluster_name=cluster_names[accelerator].value,
            tpu_zone=Zone.US_CENTRAL2_B.value,
            time_out_in_min=60,
            test_name=f"maxtext-checkpointing-{mode.value}",
            run_model_cmds=command,
            docker_image=image.value,
            test_owner=test_owner.SURBHI_J,
        ).run()
