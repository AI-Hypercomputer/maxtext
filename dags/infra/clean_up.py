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

"""A DAG to clean up idle accelerator resources."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import Project, Zone
from xlml.utils import tpu


# Run every 10min
SCHEDULED_TIME = "*/10 * * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="clean_up",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "clean_up"],
    start_date=datetime.datetime(2024, 2, 22),
    catchup=False,
) as dag:
  # List tpu zones for project cloud_ml_auto_solutions to avoid permission issue
  tpu_zones = [
      Zone.US_CENTRAL1_A,
      Zone.US_CENTRAL1_B,
      Zone.US_CENTRAL2_B,
      Zone.US_CENTRAL1_C,
      Zone.US_EAST1_D,
  ]
  tpu.clean_up_idle_queued_resources(Project.CLOUD_ML_AUTO_SOLUTIONS.value, tpu_zones)
