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

"""A DAG to run all supported ML models with the latest PAX version."""

import datetime
from airflow import models
from configs import composer_env, gcs_bucket, vm_resource
from configs.xlml.jax import solutionsTeam_pax_latest_supported_config as pax_config


# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="pax_latest_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "pax", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 11, 8),
    catchup=False,
) as dag:
  # Language model with SPMD
  pax_lmspmd2b_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{gcs_bucket.XLML_OUTPUT_DIR}/pax/lmspmd2b/v4-8",
      exp_path="tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps",
      model_name="lmspmd2b",
  ).run()

  pax_lmspmd2b_ckpt_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{gcs_bucket.XLML_OUTPUT_DIR}/pax/lmspmd2b_ckpt/v4-8",
      exp_path="tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps",
      model_name="lmspmd2b_ckpt",
      ckp_path=f"{gcs_bucket.PAX_DIR}/lmcloudspmd2B/pax-nightly-lmspmd2b-func-v4-8-1vm-run1/*",
  ).run()

  # Language model transformer with adam
  pax_transformer_adam_extra_flags = [
      "--jax_fully_async_checkpoint=False",
      "--pmap_use_tensorstore=True",
  ]
  pax_lmtransformeradam_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{gcs_bucket.XLML_OUTPUT_DIR}/pax/lmtransformeradam/v4-8",
      exp_path="tasks.lm.params.lm_cloud.LmCloudTransformerAdamLimitSteps",
      model_name="lmtransformeradam",
      extraFlags=" ".join(pax_transformer_adam_extra_flags),
  ).run()

  # Test dependencies
  pax_lmspmd2b_v4_8
  pax_lmspmd2b_ckpt_v4_8
  pax_lmtransformeradam_v4_8
