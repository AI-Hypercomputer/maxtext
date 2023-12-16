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

"""Utilities to construct configs for solutionsteam_flax_latest_supported DAG."""

from typing import Tuple
import uuid
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource
from configs.xlml.jax import common


PROJECT_NAME = vm_resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value
IS_TPU_RESERVED = True


def get_flax_resnet_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_flax()

  work_dir = generate_unique_dir("/tmp/imagenet")
  run_model_cmds = (
      (
          f"export TFDS_DATA_DIR={data_dir} &&"
          " JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/imagenet/main.py"
          " --config=/tmp/flax/examples/imagenet/configs/tpu.py"
          f" --workdir={work_dir} --config.num_epochs=1 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=IS_TPU_RESERVED,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name="flax_resnet_imagenet",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_vit_setup_cmds() -> Tuple[str]:
  return common.set_up_hugging_face_transformers() + (
      "pip install -r /tmp/transformers/examples/flax/vision/requirements.txt",
      "pip install ml_dtypes==0.2.0",
      (
          "cd /tmp/transformers && wget"
          " https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
      ),
      "cd /tmp/transformers && tar -xvzf imagenette2.tgz",
  )


def get_flax_vit_run_model_cmds(
    num_train_epochs: int,
    extraFlags: str = "",
    extra_run_cmds: Tuple[str] = ("echo",),
) -> Tuple[str]:
  return (
      (
          "JAX_PLATFORM_NAME=TPU python3"
          " /tmp/transformers/examples/flax/vision/run_image_classification.py"
          " --model_name_or_path google/vit-base-patch16-224-in21k"
          f" --num_train_epochs {num_train_epochs} --output_dir"
          " '/tmp/transformers/vit-imagenette' --train_dir"
          " '/tmp/transformers/imagenette2/train' --validation_dir"
          " '/tmp/transformers/imagenette2/val' --learning_rate 1e-3"
          f" --preprocessing_num_workers 32 --overwrite_output_dir {extraFlags}"
      ),
  ) + extra_run_cmds


def get_flax_vit_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_epochs: int = 3,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = get_flax_vit_setup_cmds()
  run_model_cmds = get_flax_vit_run_model_cmds(num_train_epochs, extraFlags)

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=True,
      ),
      test_name="flax_vit_imagenette",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_vit_conv_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_epochs: int = 30,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = get_flax_vit_setup_cmds()
  tf_summary_location = (
      "/tmp/transformers/vit-imagenette/events.out.tfevents.jax-vit.v2"
  )
  gcs_location = (
      f"{gcs_bucket.XLML_OUTPUT_DIR}/flax/vit/metric/events.out.tfevents.jax-vit.v2"
  )
  extra_run_cmds = (
      (
          "cp /tmp/transformers/vit-imagenette/events.out.tfevents.*"
          f" {tf_summary_location} || exit 0"
      ),
      f"gsutil cp {tf_summary_location} {gcs_location} || exit 0",
  )
  run_model_cmds = get_flax_vit_run_model_cmds(
      num_train_epochs, extraFlags, extra_run_cmds
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=True,
      ),
      test_name="flax_vit_imagenette_conv",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  job_metric_config = metric_config.MetricConfig(
      tensorboard_summary=metric_config.SummaryConfig(
          file_location=gcs_location,
          aggregation_strategy=metric_config.AggregationStrategy.LAST,
      )
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


def get_flax_gpt2_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
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
          runtime_version=RUNTIME_IMAGE,
          reserved=IS_TPU_RESERVED,
      ),
      test_name="flax_gpt2_oscar",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
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
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_hugging_face_diffusers() + (
      (
          "pip install -U -r"
          " /tmp/diffusers/examples/text_to_image/requirements_flax.txt"
      ),
  )

  work_dir = generate_unique_dir("./sd-pokemon-model")
  run_model_cmds = (
      (
          "cd /tmp/diffusers/examples/text_to_image && JAX_PLATFORM_NAME=TPU"
          " python3 train_text_to_image_flax.py"
          " --pretrained_model_name_or_path='duongna/stable-diffusion-v1-4-flax'"
          " --dataset_name='lambdalabs/pokemon-blip-captions' --resolution=512"
          " --center_crop --random_flip --train_batch_size=8"
          f" --num_train_epochs={num_train_epochs} --learning_rate=1e-05"
          f" --max_grad_norm=1 --output_dir={work_dir} --cache_dir /tmp"
          f" {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=IS_TPU_RESERVED,
      ),
      test_name="flax_sd_pokemon",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_bart_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_hugging_face_transformers() + (
      "pip install -r examples/flax/summarization/requirements.txt",
      "pip install ml_dtypes==0.2.0",
  )

  run_model_cmds = (
      (
          "cd /tmp/transformers/examples/flax/summarization &&"
          " JAX_PLATFORM_NAME=TPU python3 run_summarization_flax.py"
          " --model_name_or_path facebook/bart-base --tokenizer_name"
          " facebook/bart-base --dataset_name wiki_summary --do_train"
          " --do_eval --do_predict --predict_with_generate --learning_rate"
          " 5e-5 --warmup_steps 0 --output_dir=./bart-base-wiki"
          " --overwrite_output_dir --num_train_epochs 3 --max_source_length"
          f" 512 --max_target_length 64 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=IS_TPU_RESERVED,
      ),
      test_name="flax_bart_wiki",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_bert_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    task_name: str,
    num_train_epochs: int = 1,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_hugging_face_transformers() + (
      "pip install -r examples/flax/text-classification/requirements.txt",
      "pip install ml_dtypes==0.2.0",
  )

  run_model_cmds = (
      (
          "cd /tmp/transformers/examples/flax/text-classification &&"
          " JAX_PLATFORM_NAME=TPU python3 run_flax_glue.py --output_dir"
          " ./bert-glue --model_name_or_path bert-base-cased"
          f" --overwrite_output_dir --task_name {task_name} --num_train_epochs"
          f" {num_train_epochs} --logging_dir ./bert-glue {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=IS_TPU_RESERVED,
      ),
      test_name=f"flax_bert_{task_name}",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_wmt_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_steps: int,
    data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_flax() + (
      "pip install tf-nightly-cpu",
      "pip install tensorflow-datasets",
      "pip install tensorflow-text-nightly",
      "pip install sentencepiece",
  )

  work_dir = generate_unique_dir("/tmp/wmt")
  run_model_cmds = (
      (
          f"export TFDS_DATA_DIR={data_dir} &&"
          " JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/wmt/main.py"
          " --config=/tmp/flax/examples/wmt/configs/default.py"
          f" --workdir={work_dir} --config.num_train_steps={num_train_steps}"
          f" --config.per_device_batch_size=16 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=IS_TPU_RESERVED,
      ),
      test_name="flax_wmt_translate",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.SHIVA_S,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def generate_unique_dir(prefix: str) -> str:
  """Generate a unique dir based on prefix to avoid skipping runs during retry."""
  short_id = str(uuid.uuid4())[:8]
  return f"{prefix}_{short_id}"
