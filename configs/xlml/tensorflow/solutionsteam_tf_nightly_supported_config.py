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

"""Utilities to construct configs for solutionsteam_tf_nightly_supported DAG."""

import uuid
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner
from configs.xlml.tensorflow import common
from configs.vm_resource import TpuVersion, Project, RuntimeVersion


def get_tf_keras_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_feature: str,
    test_name: str,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_2150_PJRT.value,
    is_pod: bool = False,
    network: str = "default",
    subnetwork: str = "default",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_pjrt_nightly() + common.set_up_tensorflow_keras()
  # TODO(ranran): enable tests for pod and is blocked by
  # https://github.com/GoogleCloudPlatform/ml-auto-solutions/pull/15
  skipped_tag = "--tags=-failspod" if is_pod else ""
  run_model_cmds = (
      (
          "export PATH=$PATH:/home/ml-auto-solutions/.local/bin &&"
          " export TPU_NAME=local &&"
          " cd /tmp/tf2-api-tests && NEXT_PLUGGABLE_DEVICE_USE_C_API=true"
          " TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so"
          " TF_USE_LEGACY_KERAS=1 behave -e ipynb_checkpoints"
          f" --tags=-fails {skipped_tag} -i {test_feature}"
      ),
  )
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=f"tf_keras_api_{test_name}",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.ERIC_L,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_tf_resnet_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_2150_PJRT.value,
    network: str = "default",
    subnetwork: str = "default",
    is_pod: bool = False,
    imagenet_dir: str = gcs_bucket.IMAGENET_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    global_batch_size: int = 4096,
    train_steps: int = 320,
    validation_interval: int = 320,
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_pjrt_nightly() + common.set_up_google_tensorflow_models()
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "train_data": {
              "input_path": imagenet_dir + "/train*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
          "validation_data": {
              "input_path": imagenet_dir + "/valid*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval,
      },
  }

  test_name = "tf_resnet_imagenet"
  tpu_name = create_tpu_name(test_name, tpu_version, tpu_cores)
  tpu_name_param = tpu_name if is_pod else "local"
  env_variable = export_env_variable(is_pod)
  run_model_cmds = (
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " PYTHONPATH='.' NEXT_PLUGGABLE_DEVICE_USE_C_API=true"
          " TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so"
          " TF_USE_LEGACY_KERAS=1"
          " python3 official/vision/train.py"
          f" --tpu={tpu_name_param} --experiment=resnet_imagenet"
          " --mode=train_and_eval --model_dir=/tmp/output"
          " --params_override='%s'" % str(params_override)
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.CHANDRA_D,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      custom_tpu_name=tpu_name,
      suffix_tpu_name=False,
      all_workers=not is_pod,
  )


def export_env_variable(is_pod: bool) -> str:
  """Export environment variables for training if any."""
  return "export TPU_LOAD_LIBRARY=0" if is_pod else "echo"


def create_tpu_name(test_name: str, tpu_version: TpuVersion, tpu_cores: int) -> str:
  """Create a custom TPU name."""
  return f"{test_name}-v{tpu_version.value}-{tpu_cores}-{str(uuid.uuid4())}"
