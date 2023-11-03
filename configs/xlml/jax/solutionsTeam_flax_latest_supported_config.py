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

"""Utilities to construct configs for solutionsTeam_flax_latest_supported DAG."""

from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource
from configs.xlml.jax import common


def get_flax_resnet_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    extraFlags: str = "",
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_flax()

  run_model_cmds = (
      (
          f"export TFDS_DATA_DIR={data_dir} &&"
          " JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/imagenet/main.py"
          " --config=/tmp/flax/examples/imagenet/configs/tpu.py"
          f" --workdir=/tmp/imagenet --config.num_epochs=1 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=True,
      ),
      test_name="flax_resnet_imagenet",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_gpt2_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    extraFlags: str = "",
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_hugging_face_transformers() + (
      (
          "pip install -r"
          " /tmp/transformers/examples/flax/language-modeling/requirements.txt"
      ),
      (
          "gsutil cp -r"
          " gs://cloud-tpu-tpuvm-artifacts/config/xl-ml-test/jax/gpt2"
          " /tmp/transformers/examples/flax/language-modeling"
      ),
      "pip install ml_dtypes==0.2.0",
  )

  run_model_cmds = (
      (
          "cd /tmp/transformers/examples/flax/language-modeling &&"
          " JAX_PLATFORM_NAME=TPU python3 run_clm_flax.py --output_dir=./gpt2"
          " --model_type=gpt2 --config_name=./gpt2 --tokenizer_name=./gpt2"
          " --dataset_name=oscar"
          " --dataset_config_name=unshuffled_deduplicated_no --do_train"
          " --do_eval --block_size=512 --learning_rate=5e-3"
          " --warmup_steps=1000 --adam_beta1=0.9 --adam_beta2=0.98"
          " --weight_decay=0.01 --overwrite_output_dir --num_train_epochs=1"
          f" --logging_steps=500 --eval_steps=2500 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=True,
      ),
      test_name="flax_gpt2_oscar",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_sd_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_epochs: int,
    extraFlags: str = "",
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_hugging_face_diffusers() + (
      (
          "pip install -U -r"
          " /tmp/diffusers/examples/text_to_image/requirements_flax.txt"
      ),
  )

  run_model_cmds = (
      (
          "cd /tmp/diffusers/examples/text_to_image && JAX_PLATFORM_NAME=TPU"
          " python3 train_text_to_image_flax.py"
          " --pretrained_model_name_or_path='duongna/stable-diffusion-v1-4-flax'"
          " --dataset_name='lambdalabs/pokemon-blip-captions' --resolution=512"
          " --center_crop --random_flip --train_batch_size=8"
          f" --num_train_epochs={num_train_epochs} --learning_rate=1e-05"
          " --max_grad_norm=1 --output_dir='./sd-pokemon-model' --cache_dir"
          f" /tmp {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=True,
      ),
      test_name="flax_sd_pokemon",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
