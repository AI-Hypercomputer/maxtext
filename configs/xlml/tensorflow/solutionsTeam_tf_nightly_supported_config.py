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

"""Utilities to construct configs for solutionsTeam_tf_nightly_supported DAG."""

import uuid
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource
from configs.xlml.tensorflow import common


def get_tf_resnet_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    is_pod: bool = False,
    imagenet_dir: str = gcs_bucket.IMAGENET_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    train_steps: int = 320,
    validation_interval: int = 320,
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_tensorflow_models()
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "train_data": {
              "input_path": imagenet_dir + "/train*",
              "tfds_data_dir": tfds_data_dir,
          },
          "validation_data": {
              "input_path": imagenet_dir + "/valid*",
              "tfds_data_dir": tfds_data_dir,
          },
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval,
      },
  }

  test_name = "tf_resnet_imagenet"
  tpu_name = create_tpu_name(test_name, tpu_version, tpu_cores)
  env_variable = export_env_variable(is_pod)
  run_model_cmds = (
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " PYTHONPATH='.' python3 official/vision/train.py"
          f" --tpu={tpu_name} --experiment=resnet_imagenet"
          " --mode=train_and_eval --model_dir=/tmp/output"
          " --params_override='%s'"
          % str(params_override)
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=get_tpu_runtime(is_pod),
          reserved=True,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.CHANDRA_D,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      custom_tpu_name=tpu_name,
      suffix_tpu_name=False,
      all_workers=not is_pod,
  )


def get_tf_bert_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    is_pod: bool = False,
    tf_nlp_bert_dir: str = gcs_bucket.TF_NLP_BERT_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    train_steps: int = 2000,
    validation_interval: int = 1000,
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_tensorflow_models()
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "init_checkpoint": (
              tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/bert_model.ckpt"
          ),
          "train_data": {
              "tfds_data_dir": tfds_data_dir,
              "vocab_file": (
                  tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/vocab.txt"
              ),
          },
          "validation_data": {
              "tfds_data_dir": tfds_data_dir,
              "vocab_file": (
                  tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/vocab.txt"
              ),
          },
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval,
      },
  }

  test_name = "tf_bert_glue_mnli"
  tpu_name = create_tpu_name(test_name, tpu_version, tpu_cores)
  env_variable = export_env_variable(is_pod)
  run_model_cmds = (
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " PYTHONPATH='.' python3 official/nlp/train.py"
          f" --tpu={tpu_name} --experiment=bert/sentence_prediction_text"
          " --config_file=official/nlp/configs/experiments/glue_mnli_text.yaml"
          " --mode=train_and_eval --model_dir=/tmp/output"
          " --params_override='%s'"
          % str(params_override)
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=get_tpu_runtime(is_pod),
          reserved=True,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.CHANDRA_D,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      custom_tpu_name=tpu_name,
      suffix_tpu_name=False,
      all_workers=not is_pod,
  )


def export_env_variable(is_pod: bool) -> str:
  """Export environment variables for training if any."""
  return "export TPU_LOAD_LIBRARY=0" if is_pod else "echo"


def get_tpu_runtime(is_pod: bool) -> str:
  """Get TPU runtime image."""
  if is_pod:
    return vm_resource.RuntimeVersion.VM_NIGHTLY_POD.value
  return vm_resource.RuntimeVersion.VM_NIGHTLY.value


def create_tpu_name(test_name: str, tpu_version: str, tpu_cores: int) -> str:
  """Create a custom TPU name."""
  return f"{test_name}-v{tpu_version}-{tpu_cores}-{str(uuid.uuid4())}"
