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

"""Utilities to construct configs for solutionsTeam_jax_latest_supported DAG."""

from apis import gcp_config, task, test_config


# TODO(ranran): This is an example to test QR creation & deletion funcitonality, and
# remove after python-API is well-organized
def get_jax_resnet_config(tpu_size: int, test_time_out: int):
  job_tpu_task = task.TPUTask(
      version="4",
      size=tpu_size,
      runtime_version="tpu-vm-v4-base",
      task_owner="ranran",
  )

  job_gcp_config = gcp_config.GCPConfig(
      project_name="tpu-prod-env-one-vm",
      project_number="630405687483",
      zone="us-central2-b",
  )

  set_up_cmds = (
      "pip install -U pip",
      "pip install --upgrade clu tensorflow tensorflow-datasets",
      (
          "pip install jax[tpu] -f"
          " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
      ),
      "git clone https://github.com/google/flax.git ~/flax",
      "pip install --user flax",
  )

  run_model_cmds = (
      "cd ~/flax/examples/mnist",
      (
          "JAX_PLATFORM_NAME=TPU python3 main.py --config=configs/default.py"
          " --workdir=/tmp/mnist --config.learning_rate=0.05"
          " --config.num_epochs=3"
      ),
  )

  job_test_config = test_config.TestConfig(
      time_out_in_min=test_time_out,
      set_up_cmd=set_up_cmds,
      run_model_cmd=run_model_cmds,
  )

  return job_tpu_task, job_gcp_config, job_test_config
