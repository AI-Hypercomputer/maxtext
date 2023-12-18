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
from configs import composer_env, vm_resource
from configs.xlml.jax import solutionsteam_flax_latest_supported_config as flax_config


# Run once a day at 2 am UTC (6 pm PST)
SCHEDULED_TIME = "0 2 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="flax_latest_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "flax", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # ResNet
  jax_resnet_v2_8 = flax_config.get_flax_resnet_config(
      tpu_version="2",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v2_32 = flax_config.get_flax_resnet_config(
      tpu_version="2",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL1_A.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v3_8 = flax_config.get_flax_resnet_config(
      tpu_version="3",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v3_32 = flax_config.get_flax_resnet_config(
      tpu_version="3",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v4_8 = flax_config.get_flax_resnet_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v4_32 = flax_config.get_flax_resnet_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5e_4 = flax_config.get_flax_resnet_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5litepod",
      tpu_cores=4,
      tpu_zone=vm_resource.Zone.US_EAST1_C.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5e_16 = flax_config.get_flax_resnet_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5litepod",
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_EAST1_C.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5p_8 = flax_config.get_flax_resnet_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5p",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_EAST5_A.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5P_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5p_32 = flax_config.get_flax_resnet_config(
      project_name=vm_resource.Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version="5p",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_EAST5_A.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5P_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  # ViT
  jax_vit_v4_extra_flags = [
      "--per_device_train_batch_size=64",
      "--per_device_eval_batch_size=64",
  ]
  jax_vit_v4_8 = flax_config.get_flax_vit_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      extraFlags=" ".join(jax_vit_v4_extra_flags),
  ).run()

  jax_vit_v4_32_extra_flags = jax_vit_v4_extra_flags + [
      "--model_name_or_path google/vit-base-patch16-224-in21k",
  ]
  jax_vit_v4_32 = flax_config.get_flax_vit_conv_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      extraFlags=" ".join(jax_vit_v4_32_extra_flags),
  ).run()

  # GPT2
  jax_gpt2_v4_extra_flags = [
      "--per_device_train_batch_size=64",
      "--per_device_eval_batch_size=64",
  ]
  jax_gpt2_v4_8 = flax_config.get_flax_gpt2_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=120,
      extraFlags=" ".join(jax_gpt2_v4_extra_flags),
  ).run()

  jax_gpt2_v4_32 = flax_config.get_flax_gpt2_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=120,
      extraFlags=" ".join(jax_gpt2_v4_extra_flags),
  ).run()

  # Stable Diffusion
  jax_sd_v4_8 = flax_config.get_flax_sd_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      num_train_epochs=1,
  ).run()

  jax_sd_v4_32 = flax_config.get_flax_sd_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      num_train_epochs=1,
  ).run()

  # BART
  jax_bart_v4_8_extra_flags = [
      "--per_device_train_batch_size=64",
      "--per_device_eval_batch_size=64",
  ]
  jax_bart_v4_8 = flax_config.get_flax_bart_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      extraFlags=" ".join(jax_bart_v4_8_extra_flags),
  ).run()

  jax_bart_v4_32_extra_flags = [
      "--per_device_train_batch_size=32",
      "--per_device_eval_batch_size=32",
  ]
  jax_bart_v4_32 = flax_config.get_flax_bart_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      extraFlags=" ".join(jax_bart_v4_32_extra_flags),
  ).run()

  # BERT
  jax_bert_mnli_extra_flags = [
      "--max_seq_length 512",
      "--eval_steps 1000",
  ]
  jax_bert_v4_mnli_extra_flags = jax_bert_mnli_extra_flags + [
      "--per_device_train_batch_size=8",
      "--per_device_eval_batch_size=8",
  ]
  jax_bert_mnli_v4_8 = flax_config.get_flax_bert_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      task_name="mnli",
      extraFlags=" ".join(jax_bert_v4_mnli_extra_flags),
  ).run()

  jax_bert_mnli_v4_32 = flax_config.get_flax_bert_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      task_name="mnli",
      extraFlags=" ".join(jax_bert_v4_mnli_extra_flags),
  ).run()

  jax_bert_mrpc_extra_flags = [
      "--max_seq_length 128",
      "--eval_steps 100",
  ]
  jax_bert_v4_mrpc_extra_flags = jax_bert_mrpc_extra_flags + [
      "--per_device_train_batch_size=8",
      "--per_device_eval_batch_size=8",
  ]
  jax_bert_mrpc_v4_8 = flax_config.get_flax_bert_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      task_name="mrpc",
      extraFlags=" ".join(jax_bert_v4_mrpc_extra_flags),
  ).run()

  jax_bert_mrpc_v4_32 = flax_config.get_flax_bert_config(
      tpu_version="4",
      tpu_cores=32,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      task_name="mrpc",
      extraFlags=" ".join(jax_bert_v4_mrpc_extra_flags),
  ).run()

  # WMT
  jax_wmt_v4_8 = flax_config.get_flax_wmt_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      num_train_steps=10,
  ).run()

  # Test dependencies
  jax_resnet_v2_8 >> jax_resnet_v2_32
  jax_resnet_v3_8 >> jax_resnet_v3_32
  jax_resnet_v4_8 >> jax_resnet_v4_32
  jax_resnet_v5e_4 >> jax_resnet_v5e_16
  jax_resnet_v5p_8 >> jax_resnet_v5p_32
  jax_vit_v4_8 >> jax_vit_v4_32
  jax_gpt2_v4_8 >> jax_gpt2_v4_32
  jax_sd_v4_8 >> jax_sd_v4_32
  jax_bart_v4_8 >> jax_bart_v4_32
  jax_bert_mnli_v4_8 >> jax_bert_mnli_v4_32
  jax_bert_mrpc_v4_8 >> jax_bert_mrpc_v4_32
  jax_wmt_v4_8
