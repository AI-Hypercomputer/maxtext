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
An example DAG to run maxtext on 1 and 8 slices of v4-8 with startup script on GCE.
"""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone
from dags.multipod.configs import maxtext_gce_config
from dags.multipod.configs.common import SetupMode, Platform

with models.DAG(
    dag_id="maxtext_nightly_startup_script_example_dag",
    schedule=None,
    tags=["multipod_team", "maxtext"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
) as dag:
  default_test_name = "maxtext-nightly-startup-script"

  test_mode = SetupMode.NIGHTLY

  # Maxtext
  maxtext_nightly_1slice_v4_8_startup_script = (
      maxtext_gce_config.get_maxtext_nightly_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          automated_test=False,
          test_name=default_test_name,
          test_mode=test_mode,
      ).run_with_startup_script()
  )

  maxtext_nightly_8slice_v4_8_startup_script = (
      maxtext_gce_config.get_maxtext_nightly_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          automated_test=False,
          num_slices=8,
          test_name=default_test_name,
          test_mode=test_mode,
      ).run_with_startup_script()
  )

  # Test dependencie
  (
      maxtext_nightly_1slice_v4_8_startup_script
      >> maxtext_nightly_8slice_v4_8_startup_script
  )
