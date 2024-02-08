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

"""A DAG to run MaxText tests with nightly version."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone
from dags.multipod.configs import mxla_collective_config
from dags.multipod.configs.common import SetupMode, Platform


# Run once a day at 8 am UTC (12 pm PST)
SCHEDULED_TIME = "0 8 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="mxla_collective_nightly",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "mxla_collective", "nightly"],
    start_date=datetime.datetime(2024, 2, 7),
    catchup=False,
) as dag:
  mxla_1mb_test_name = "mxla-collective-nightly-1mb"
  mxla_256mb_test_name = "mxla-collective-nightly-256mb"

  mxla_collective_1mb_nightly_4slice_v4_8 = (
      mxla_collective_config.get_mxla_collective_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          bytes_to_transfer=1000000,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=4,
          test_name=mxla_1mb_test_name,
      ).run()
  )

  mxla_collective_1mb_nightly_8slice_v4_8 = (
      mxla_collective_config.get_mxla_collective_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          bytes_to_transfer=1000000,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=8,
          test_name=mxla_1mb_test_name,
      ).run()
  )

  mxla_collective_256mb_nightly_4slice_v4_8 = (
      mxla_collective_config.get_mxla_collective_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          bytes_to_transfer=256000000,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=4,
          test_name=mxla_256mb_test_name,
      ).run()
  )

  mxla_collective_256mb_nightly_8slice_v4_8 = (
      mxla_collective_config.get_mxla_collective_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          bytes_to_transfer=256000000,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=8,
          test_name=mxla_256mb_test_name,
      ).run()
  )
  # Test dependencie
  (
      mxla_collective_1mb_nightly_4slice_v4_8
      >> mxla_collective_256mb_nightly_4slice_v4_8
      >> mxla_collective_1mb_nightly_8slice_v4_8
      >> mxla_collective_256mb_nightly_8slice_v4_8
  )
