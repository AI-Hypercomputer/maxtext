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

"""Utilities to construct configs for solutionsTeam_pax_latest_supported DAG."""

import uuid
from apis import gcp_config, metric_config, task, test_config
from configs import test_owner, vm_resource
from configs.xlml.jax import common


def get_pax_lm_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    exp_path: str,
    model_name: str,
    log_dir: str,
    ckp_path: str = "",
    extraFlags: str = "",
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  short_id = str(uuid.uuid4())[:8]
  job_log_dir = f"{log_dir}/{model_name}-{short_id}"
  ckp_cmds = f"gsutil -m cp -r {ckp_path} {job_log_dir}" if ckp_path else "echo"
  set_up_cmds = common.set_up_google_pax() + (ckp_cmds,)

  run_model_cmds = (
      (
          "python3 .local/lib/python3.8/site-packages/paxml/main.py"
          f" --exp={exp_path} --job_log_dir={job_log_dir} {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_VM_V4_BASE.value,
          reserved=True,
      ),
      test_name=f"pax_{model_name}_c4",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.GERSON_K,
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
