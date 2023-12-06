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

"""Utilities to construct configs for solutionsteam_jax_npi DAG."""

from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource


# TODO(ranran): This is an example to test E2E benchmarking, and
# update/remove after benchmark models are ready.
def get_jax_vit_config(
    tpu_version: int,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = (
      "pip install -U pip",
      "pip install --upgrade clu tensorflow tensorflow-datasets",
      (
          "pip install jax[tpu] -f"
          " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
      ),
      (
          "git clone https://github.com/huggingface/transformers.git"
          " /tmp/transformers"
      ),
      "cd /tmp/transformers && pip install .",
      "pip install -r /tmp/transformers/examples/flax/_tests_requirements.txt",
      "pip install --upgrade huggingface-hub urllib3 zipp",
      "pip install -r /tmp/transformers/examples/flax/vision/requirements.txt",
      (
          "cd /tmp/transformers && wget"
          " https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
      ),
      "cd /tmp/transformers && tar -xvzf imagenette2.tgz",
  )

  tf_summary_location = (
      "/tmp/transformers/vit-imagenette/events.out.tfevents.jax-vit.v2"
  )
  gcs_location = f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/jax_vit_config/events.out.tfevents.jax-vit.v2"

  run_script_cmds = (
      (
          "JAX_PLATFORM_NAME=TPU python3"
          " /tmp/transformers/examples/flax/vision/run_image_classification.py"
          " --model_name_or_path google/vit-base-patch16-224-in21k"
          " --num_train_epochs 3 --output_dir"
          " '/tmp/transformers/vit-imagenette' --train_dir"
          " '/tmp/transformers/imagenette2/train' --validation_dir"
          " '/tmp/transformers/imagenette2/val' --learning_rate 1e-3"
          " --preprocessing_num_workers 32 --per_device_train_batch_size 64"
          " --per_device_eval_batch_size 64"
      ),
      (
          "cp /tmp/transformers/vit-imagenette/events.out.tfevents.*"
          f" {tf_summary_location}"
      ),
      f"gsutil cp {tf_summary_location} {gcs_location}",
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=True,
      ),
      "jax_vit",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.RAN_R,
  )

  job_metric_config = metric_config.MetricConfig(
      tensorboard_summary=metric_config.SummaryConfig(
          file_location=gcs_location,
          aggregation_strategy=metric_config.AggregationStrategy.LAST,
      )
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
