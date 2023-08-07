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

"""A DAG to run all NPI benchmark with the latest JAX version."""

import datetime
from airflow import models
from configs.benchmark import solutionsTeam_jax_npi_config as jax_npi_config

with models.DAG(
    dag_id="jax_latest_npi",
    schedule=None,
    tags=["jax", "latest", "npi"],
    start_date=datetime.datetime(2023, 8, 6),
    catchup=False,
) as dag:
  jax_vit_v4_8 = jax_npi_config.get_jax_vit_config(8, 60).run()
