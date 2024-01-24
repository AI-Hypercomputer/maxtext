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

"""Utilities to construct configs for pax DAGs."""

from datetime import datetime
import enum
from typing import Tuple
import uuid
from absl import logging
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.solutions_team.configs.pax import common
from dags.vm_resource import TpuVersion, RuntimeVersion, Project


class PaxVersion(enum.Enum):
  NIGHTLY = "nightly"
  STABLE = "stable"


def get_setup_cmds(
    pax_version: PaxVersion,
    ckp_path: str,
    job_log_dir: str,
) -> Tuple[str]:
  if pax_version is PaxVersion.STABLE:
    logging.info("Running the latest stable Pax version.")
    ckp_cmds = f"gsutil -m cp -r {ckp_path} {job_log_dir}" if ckp_path else "echo"
    return common.set_up_google_pax() + (ckp_cmds,)
  elif pax_version is PaxVersion.NIGHTLY:
    logging.info("Running nightly Pax version.")
    build_date = datetime.today().strftime("%Y%m%d")
    ckp_cmds = f"gsutil -m cp -r {ckp_path} {job_log_dir}" if ckp_path else "echo"
    return (
        ckp_cmds,
        (
            "set -x; set -e; gsutil cp"
            f" gs://pax-on-cloud-tpu-project/wheels/{build_date}/paxml*.whl ."
        ),
        (
            "set -x; set -e; gsutil cp"
            f" gs://pax-on-cloud-tpu-project/wheels/{build_date}/praxis*.whl ."
        ),
        "pip install --upgrade pip",
        "pip install praxis*.whl",
        "pip install paxml*.whl",
        "sudo pip uninstall --yes jax jaxlib libtpu-nightly",
        "pip install git+https://github.com/google/jax.git",
        (
            "pip install --pre -U jaxlib -f"
            " https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        ),
        (
            "pip install --no-index -U libtpu-nightly -f"
            " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        ),
    )
  else:
    raise RuntimeError(f"Please specify set up cmds for: {pax_version.value}.")


def get_runtime_version(pax_version: PaxVersion, tpu_version: TpuVersion) -> str:
  if tpu_version is TpuVersion.V5E:
    return RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
  elif tpu_version is TpuVersion.V5P:
    return RuntimeVersion.V2_ALPHA_TPUV5.value
  else:
    if pax_version is PaxVersion.STABLE:
      return RuntimeVersion.TPU_VM_V4_BASE.value
    elif pax_version is PaxVersion.NIGHTLY:
      return RuntimeVersion.TPU_UBUNTU2204_BASE.value
    else:
      raise RuntimeError(f"Please specify runtime version for: {pax_version.value}.")


def get_pax_lm_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    exp_path: str,
    model_name: str,
    log_dir: str,
    pax_version: PaxVersion = PaxVersion.STABLE,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    ckp_path: str = "",
    extraFlags: str = "",
    network: str = "default",
    subnetwork: str = "default",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  short_id = str(uuid.uuid4())[:8]
  job_log_dir = f"{log_dir}/{model_name}-{short_id}"
  set_up_cmds = get_setup_cmds(pax_version, ckp_path, job_log_dir)

  runtime_version = get_runtime_version(pax_version, tpu_version)

  if runtime_version == RuntimeVersion.TPU_VM_V4_BASE.value:
    package_version = "python3.8"
  else:
    package_version = "python3.10"

  run_model_cmds = (
      (
          f"python3 .local/lib/{package_version}/site-packages/paxml/main.py"
          f" --exp={exp_path} --job_log_dir={job_log_dir} {extraFlags}"
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
      test_name=f"pax_{pax_version.value}_{model_name}",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.GERSON_K,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
