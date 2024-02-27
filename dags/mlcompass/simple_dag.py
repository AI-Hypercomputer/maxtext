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

"""A Simple DAG.

Usage:
gcloud composer environments run ml-automation-solutions \
  --project=cloud-ml-auto-solutions \
  --location=us-central1 dags trigger \
  -- \
  mlcompass_simple_dag \
  --conf={\\\"commit_sha\\\":\\\"your-commit-sha\\\"}

"""

import datetime
from airflow import models
from airflow.operators.bash import BashOperator
from dags.mlcompass.configs.simple_config import get_simple_config

with models.DAG(
    dag_id="mlcompass_simple_dag",
    schedule=None,
    tags=["simple", "mlcompass"],
    start_date=datetime.datetime(2024, 2, 5),
    catchup=False,
    params={
        "commit_sha": "my-commit-sha",
    },
) as dag:
  t1 = BashOperator(
      task_id="print_env",
      bash_command="echo {{params.commit_sha}}",
  )

  simple = get_simple_config().run()
