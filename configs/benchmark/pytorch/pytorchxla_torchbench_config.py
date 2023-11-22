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

"""Utilities to construct configs for pytorchxla_torchbench DAG."""

from typing import Tuple
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource


# TODO(ranran or PyTroch/XLA team): this is an example for benchmark test with hardcode compatible versions,
# we need to dynamically generate date on daily basis.
def set_up_torchbench(model_name: str = "") -> Tuple[str]:
  """Common set up for TorchBench."""
  return (
      "pip install -U setuptools",
      "sudo systemctl stop unattended-upgrades",
      "sudo apt-get -y update",
      "sudo apt install -y libopenblas-base",
      "sudo apt install -y libsndfile-dev",
      "sudo apt-get install libgl1 -y",
      "pip install numpy pandas",
      (
          "pip install --user --pre torch torchvision --index-url"
          " https://download.pytorch.org/whl/nightly/cpu"
      ),
      (
          "pip install --user 'torch_xla[tpuvm] @"
          " https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl'"
      ),
      (
          "pip install --pre torchtext==0.17.0.dev20231025+cpu --index-url "
          " https://download.pytorch.org/whl/nightly/cpu --no-dependencies"
      ),
      (
          "pip install --pre torchaudio==2.2.0.dev20231118+cpu --index-url"
          " https://download.pytorch.org/whl/nightly/cpu --no-dependencies"
      ),
      "pip install Pillow --no-dependencies",
      "sudo chmod 777 /usr/local/lib/python3.10/dist-packages/",
      "sudo chmod 777 /usr/local/bin/",
      "pip install torchdata --no-dependencies",
      "pip install tqdm --no-dependencies",
      "pip install psutil",
      "cd; git clone https://github.com/pytorch/benchmark.git",
      "cd; git clone https://github.com/zpcore/xla.git",
      "cd ~/xla && git checkout benchmark",
  )


# TODO(ranran or PyTroch/XLA team) & notes:
# 1) If you want to run all models, do not pass in model_name
# 2) All filters of benchmark can be passed via extraFlags
# 3) Update test owner to PyTroch/XLA team
def get_torchbench_config(
    tpu_version: int,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    model_name: str = "",
    extraFlags: str = "",
) -> task.TpuTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench(model_name)
  local_output_location = "~/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = (
      f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/metric_report.jsonl"
  )

  run_script_cmds = (
      (
          "cd ~/xla/benchmarks && python experiment_runner.py"
          " --suite-name=torchbench --xla=PJRT --accelerator=tpu --progress-bar"
          f" {extraFlags}"
      ),
      "python ~/xla/benchmarks/result_analyzer.py",
      f"gsutil cp {local_output_location} {gcs_location}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
          reserved=True,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.PEI_Z,
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig(
          file_location=gcs_location,
      )
  )

  return task.TpuTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
