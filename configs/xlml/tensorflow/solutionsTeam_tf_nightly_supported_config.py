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

from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner
from configs.xlml.tensorflow import common

# TODO(ranran): update project_name & zone once reserved capacity is ready in project cloud-ml-auto-solutions
PROJECT_NAME = "tpu-prod-env-one-vm"
RUNTIME_VERSION = "1vm-nightly"


def get_tf_resnet_config(
    tpu_version: int,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    imagenet_dir: str = gcs_bucket.IMAGENET_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    train_steps: int = 320,
    validation_interval: int = 320,
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_tensorflow_models()
  params_override = {
      "runtime": {
          "distribution_strategy": "tpu"
      },
      "task": {
          "train_data": {
              "input_path": imagenet_dir + "/train*",
              "tfds_data_dir": tfds_data_dir,
          },
          "validation_data": {
              "input_path": imagenet_dir + "/valid*",
              "tfds_data_dir": tfds_data_dir,
          }
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval
      }
  }

  # TODO(ranran): handle Pod case with tpu name with TPU_LOAD_LIBRARY=0
  run_model_cmds = (("cd /usr/share/tpu/models && PYTHONPATH='.'"
                     " python3 official/vision/train.py"
                     " --tpu=local --experiment=resnet_imagenet"
                     " --mode=train_and_eval --model_dir=/tmp/output"
                     " --params_override='%s'" % str(params_override)),)

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_VERSION,
      ),
      test_name="tf_resnet_imagenet",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.CHANDRA_D,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_tf_bert_config(
    tpu_version: int,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    tf_nlp_bert_dir: str = gcs_bucket.TF_NLP_BERT_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    train_steps: int = 2000,
    validation_interval: int = 1000,
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_tensorflow_models()
  params_override = {
      "runtime": {
          "distribution_strategy": "tpu"
      },
      "task": {
          "init_checkpoint":
              tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/bert_model.ckpt",
          "train_data": {
              "tfds_data_dir": tfds_data_dir,
              "vocab_file": tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/vocab.txt"
          },
          "validation_data": {
              "tfds_data_dir": tfds_data_dir,
              "vocab_file": tf_nlp_bert_dir + "/uncased_L-12_H-768_A-12/vocab.txt"
          }
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval
      }
  }

  # TODO(ranran): handle Pod case with tpu name with TPU_LOAD_LIBRARY=0
  run_model_cmds = ((
      "cd /usr/share/tpu/models && PYTHONPATH='.'"
      " python3 official/nlp/train.py"
      " --tpu=local --experiment=bert/sentence_prediction_text"
      " --config_file=official/nlp/configs/experiments/glue_mnli_text.yaml"
      " --mode=train_and_eval --model_dir=/tmp/output"
      " --params_override='%s'" % str(params_override)),)

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_VERSION,
      ),
      test_name="tf_bert_glue_mnli",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.CHANDRA_D,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
