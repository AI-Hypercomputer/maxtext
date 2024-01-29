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

"""A DAG to run MaxText tests with nightly version."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone
from dags.multipod.configs import maxtext_gce_config


# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_nightly",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "nightly"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
) as dag:
  default_test_name = "maxtext-nightly"

  # Maxtext
  maxtext_nightly_1slice_v4_8 = maxtext_gce_config.get_maxtext_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_tpu_reserved=False,
      test_name=default_test_name,
  ).run()

  maxtext_nightly_2slice_v4_8 = maxtext_gce_config.get_maxtext_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_tpu_reserved=False,
      num_slices=2,
      test_name=default_test_name,
  ).run()

  maxtext_nightly_4slice_v4_8 = maxtext_gce_config.get_maxtext_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_tpu_reserved=False,
      num_slices=4,
      test_name=default_test_name,
  ).run()

  maxtext_nightly_8slice_v4_8 = maxtext_gce_config.get_maxtext_nightly_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_tpu_reserved=False,
      num_slices=8,
      test_name=default_test_name,
  ).run()

  # Test dependencie
  (
      maxtext_nightly_1slice_v4_8
      >> maxtext_nightly_2slice_v4_8
      >> maxtext_nightly_4slice_v4_8
      >> maxtext_nightly_8slice_v4_8
  )
