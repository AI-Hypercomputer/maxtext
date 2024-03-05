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
An example DAG to launch a sweep of MaxText GKE XPK workloads
sweeping over a range of per_device_batch_size values for a 16B model
on 1xv4-128.
"""

import datetime
from airflow import models
from dags import test_owner
from dags.vm_resource import TpuVersion, Zone, Project, ClusterName, DockerImage
from dags.multipod.configs import maxtext_sweep_gke_config

# Set concurrency to number of workers otherwise tasks may time out
# if there are more concurrent tasks running at a time than number of workers
with models.DAG(
    dag_id="maxtext_sweep_gke_example_dag",
    schedule=None,
    tags=["multipod_team", "maxtext"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
    concurrency=2,
) as dag:
  # MaxText set up and run commands
  base_output_directory = "gs://maxtext-experiments-multipod"
  base_run_model_cmds = [
      f"python3 MaxText/train.py MaxText/configs/base.yml base_output_directory={base_output_directory} dataset_path=gs://max-datasets-rogue enable_checkpointing=false global_parameter_scale=16 steps=10",
  ]

  # Get list of MaxText GKE XPK jobs
  maxtext_sweep_gke_test = (
      maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
          test_owner=test_owner.RAYMOND_Z,
          project_name=Project.TPU_PROD_ENV_MULTIPOD.value,
          cluster_name=ClusterName.V4_128_MULTISLICE_CLUSTER.value,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          base_output_directory=base_output_directory,
          tpu_version=TpuVersion.V4,
          tpu_cores=128,
          num_slices=[1],
          docker_image=DockerImage.XPK_MAXTEXT_TEST.value,
          run_name_prefix="maxtext-16b",
          base_run_model_cmds=base_run_model_cmds,
          sweep_params={"M_PER_DEVICE_BATCH_SIZE": [2, 4, 8]},
      )
  )

  # Run jobs
  for test in maxtext_sweep_gke_test:
    test.run_with_run_name_generation()
