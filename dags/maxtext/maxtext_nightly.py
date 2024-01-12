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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from airflow import models
from configs import composer_env
from configs.vm_resource import Project, TpuVersion, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from configs.maxtext import maxtext_gce_config


# Run once a day at 2 am UTC (6 pm PST)
SCHEDULED_TIME = "0 2 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_test",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
) as dag:
  # Maxtext
  maxtext_nightly_v4_8 = maxtext_gce_config.get_maxtext_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  # Test dependencies
  maxtext_nightly_v4_8
