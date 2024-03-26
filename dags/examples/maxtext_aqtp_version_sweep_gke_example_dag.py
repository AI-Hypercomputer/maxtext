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
sweeping over aqtp version and quantization option on different tpu topologies.
Used upon library upgrades.
"""

import datetime
from airflow import models
from dags import test_owner
from dags.vm_resource import TpuVersion, Zone, Project, ClusterName, DockerImage
from dags.multipod.configs import maxtext_sweep_gke_config

# Set concurrency to number of workers otherwise tasks may time out
# if there are more concurrent tasks running at a time than number of workers
with models.DAG(
    dag_id="maxtext_aqtp_version_sweep_gke_example_dag",
    schedule=None,
    tags=["multipod_team", "maxtext"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = "gs://maxtext-experiments-multipod"

  sweep_model_configs = {
      "v5e": [("16b", 256), ("32b", 256), ("64b", 256), ("128b", 256)],
  }

  shared_task_config = {
      "test_owner": test_owner.ZHIYU_L,
      "project_name": Project.TPU_PROD_ENV_MULTIPOD.value,
      "cluster_name": ClusterName.V5E_256_US_WEST_4_MULTISLICE_CLUSTER.value,
      "tpu_zone": Zone.US_WEST4_B.value,
      "time_out_in_min": 60,
      "base_output_directory": base_output_directory,
      "tpu_version": TpuVersion.V5E,
      "num_slices": [1, 2],
      "docker_image": DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
  }

  tests = []
  for tpu, models in sweep_model_configs.items():
    for model_size, num_cores in models:
      run_cmds = [
          "pip show aqtp",
          f"bash MaxText/configs/{tpu}/{model_size}.sh EXECUTABLE=train.py OUTPUT_PATH={base_output_directory} PLATFORM=gke",
      ]

      tests.extend(
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              **shared_task_config,
              tpu_cores=num_cores,
              run_name_prefix=f"bf16-{model_size}",
              base_run_model_cmds=run_cmds,
              sweep_params={
                  "M_QUANTIZATION": [
                      "",
                  ]
              },
          )
      )

      tests.extend(
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              **shared_task_config,
              tpu_cores=num_cores,
              run_name_prefix=f"int8-aqtp061-{model_size}",
              base_run_model_cmds=["pip install aqtp==0.6.1"] + run_cmds,
              sweep_params={
                  "M_QUANTIZATION": [
                      "int8",
                  ]
              },
          )
      )

      tests.extend(
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              **shared_task_config,
              tpu_cores=num_cores,
              run_name_prefix=f"int8-aqpt062-{model_size}",
              base_run_model_cmds=["pip install aqtp==0.6.2"] + run_cmds,
              sweep_params={
                  "M_QUANTIZATION": [
                      "int8",
                  ]
              },
          )
      )

  # Run jobs
  for test in tests:
    test.run_with_run_name_generation()
