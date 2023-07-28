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

from apis import gcp_config, test_config
from apis.xlml import task


# TODO(ranran): This is an example to test QR creation & deletion funcitonality, and
# remove after python-API is well-organized
def get_jax_resnet_config(tpu_size: int, test_time_out: int) -> task.TPUTask:
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
      "git clone https://github.com/google/flax.git /tmp/flax",
      "pip install --user flax",
  )

  run_model_cmds = (
      (
          "JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/mnist/main.py"
          " --config=/tmp/flax/examples/mnist/configs/default.py"
          " --workdir=/tmp/mnist --config.learning_rate=0.05"
          " --config.num_epochs=3"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=4,
          cores=tpu_size,
          runtime_version="tpu-ubuntu2204-base",
      ),
      "jax_resnet",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=test_time_out,
      task_owner="ranran",
  )
  return task.TPUTask(job_test_config, job_gcp_config)
