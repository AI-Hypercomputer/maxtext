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

"""A DAG to run all supported ML models with the nightly PAX version."""

import datetime
from airflow import models
from configs import composer_env, gcs_bucket, vm_resource
from configs.xlml.pax import solutionsteam_pax_supported_config as pax_config


# Run once a day at 12 am UTC (4 am PST)
SCHEDULED_TIME = "0 12 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="pax_nightly_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "pax", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2023, 12, 5),
    catchup=False,
) as dag:
  log_dir_prefix = f"{gcs_bucket.XLML_OUTPUT_DIR}/pax/nightly"

  # Language model with SPMD
  pax_lmspmd2b_extra_flags = [
      "--jax_fully_async_checkpoint=False",
  ]
  lmspmd2b_exp_path = "tasks.lm.params.lm_cloud.LmCloudSpmd2BLimitSteps"
  pax_nightly_lmspmd2b_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmspmd2b/v4-8",
      pax_version=pax_config.PaxVersion.NIGHTLY,
      exp_path=lmspmd2b_exp_path,
      model_name="lmspmd2b",
      extraFlags=" ".join(pax_lmspmd2b_extra_flags),
  ).run()

  pax_nightly_lmspmd2b_ckpt_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmspmd2b_ckpt/v4-8",
      pax_version=pax_config.PaxVersion.NIGHTLY,
      exp_path=lmspmd2b_exp_path,
      model_name="lmspmd2b_ckpt",
      ckp_path=f"{gcs_bucket.PAX_DIR}/lmcloudspmd2B/pax-nightly-lmspmd2b-func-v4-8-1vm-run1/*",
  ).run()

  # Language model transformer with adam
  pax_transformer_adam_extra_flags = [
      "--jax_fully_async_checkpoint=False",
      "--pmap_use_tensorstore=True",
  ]
  lmtransformeradam_exp_path = (
      "tasks.lm.params.lm_cloud.LmCloudTransformerAdamLimitSteps"
  )
  pax_nightly_lmtransformeradam_v4_8 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmtransformeradam/v4-8",
      exp_path=lmtransformeradam_exp_path,
      pax_version=pax_config.PaxVersion.NIGHTLY,
      model_name="lmtransformeradam",
      extraFlags=" ".join(pax_transformer_adam_extra_flags),
  ).run()

  pax_nightly_lmtransformeradam_v4_16 = pax_config.get_pax_lm_config(
      tpu_version="4",
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmtransformeradam/v4-16",
      exp_path=lmtransformeradam_exp_path,
      pax_version=pax_config.PaxVersion.NIGHTLY,
      model_name="lmtransformeradam",
      extraFlags=" ".join(pax_transformer_adam_extra_flags),
  ).run()

  pax_nightly_lmtransformeradam_v5e_4 = pax_config.get_pax_lm_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5litepod",
      tpu_cores=4,
      tpu_zone=vm_resource.Zone.US_EAST1_C.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmtransformeradam/v5litepod-4",
      exp_path=lmtransformeradam_exp_path,
      pax_version=pax_config.PaxVersion.NIGHTLY,
      model_name="lmtransformer",
      extraFlags=" ".join(pax_transformer_adam_extra_flags),
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
  ).run()

  pax_nightly_lmtransformeradam_v5e_16 = pax_config.get_pax_lm_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5litepod",
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_EAST1_C.value,
      time_out_in_min=60,
      log_dir=f"{log_dir_prefix}/lmtransformeradam/v5litepod-16",
      exp_path=lmtransformeradam_exp_path,
      pax_version=pax_config.PaxVersion.NIGHTLY,
      model_name="lmtransformer",
      extraFlags=" ".join(pax_transformer_adam_extra_flags),
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
  ).run()

  # Test dependencies
  pax_nightly_lmspmd2b_v4_8 >> pax_nightly_lmspmd2b_ckpt_v4_8
  pax_nightly_lmtransformeradam_v4_8 >> pax_nightly_lmtransformeradam_v4_16
  pax_nightly_lmtransformeradam_v5e_4 >> pax_nightly_lmtransformeradam_v5e_16
