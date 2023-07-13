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

"""A DAG to run all supported ML models with the latest JAX version."""

import datetime
from airflow import models
from apis import xlml_controller
from configs.xlml import solutionsTeam_jax_latest_supported_config as jax_config


with models.DAG(
    dag_id="jax_latest_supported",
    schedule=None,
    tags=["jax", "latest", "supported"],
    owner_links={"solutions_team": "https://go/cloudtpu-solutions-team"},
    start_date=datetime.datetime(2023, 7, 12),
) as dag:
  jax_resnet_task_v4_8, jax_gcp_v4_8, jax_resnet_test_v4_8 = (
      jax_config.get_jax_resnet_config(8, 60)
  )

  jax_resnet_task_v4_32, jax_gcp_v4_32, jax_resnet_test_v4_32 = (
      jax_config.get_jax_resnet_config(32, 600)
  )

  jax_resnet_v4_8 = xlml_controller.run(
      task_id_suffix="jax_resnet_v4_8",
      job_task=jax_resnet_task_v4_32,
      job_gcp_config=jax_gcp_v4_8,
      job_test_config=jax_resnet_test_v4_8,
  )

  jax_resnet_v4_32 = xlml_controller.run(
      task_id_suffix="jax_resnet_v4_32",
      job_task=jax_resnet_task_v4_32,
      job_gcp_config=jax_gcp_v4_32,
      job_test_config=jax_resnet_test_v4_32,
  )

  jax_resnet_v4_8 >> jax_resnet_v4_32
