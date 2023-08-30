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
      (
          "pip3 install torch --index-url"
          " https://download.pytorch.org/whl/test/cpu"
      ),
      (
          "pip3 install"
          " https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly%2B20230829-cp310-cp310-linux_x86_64.whl"
          " --no-dependencies"
      ),
      (
          "pip3 install --pre"
          " torchvision==0.16.0.dev20230828+cpu --index-url"
          " https://download.pytorch.org/whl/nightly/cpu --no-dependencies"
      ),
      (
          "pip3 install --pre"
          " torchtext==0.16.0.dev20230828+cpu --index-url"
          " https://download.pytorch.org/whl/nightly/cpu --no-dependencies"
      ),
      (
          "pip3 install --pre"
          " torchaudio==2.1.0.dev20230828+cpu --index-url"
          " https://download.pytorch.org/whl/nightly/cpu --no-dependencies"
      ),
      "pip3 install Pillow --no-dependencies",
      "pip3 install torchdata --no-dependencies",
      "pip3 install tqdm --no-dependencies",
      "pip3 install numpy --no-dependencies",
      "git clone https://github.com/RissyRan/benchmark.git /tmp/benchmark",
      "cd /tmp/benchmark && git checkout xla_benchmark",
      f"cd /tmp/benchmark && python3 install.py {model_name}",
      "pip3 install numpy",
      "pip3 install pandas",
      "sudo apt-get update -y && sudo apt-get install libgl1 -y",
      "git clone https://github.com/RissyRan/xla.git /tmp/xla",
      "cd /tmp/xla && git checkout benchmark",
      (
          "pip3 install --upgrade --force-reinstall torch --index-url"
          " https://download.pytorch.org/whl/test/cpu --no-dependencies"
      ),
      (
          "pip3 install --upgrade --force-reinstall"
          " https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly%2B20230829-cp310-cp310-linux_x86_64.whl"
          " --no-dependencies"
      ),
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
  local_output_location = "/tmp/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = (
      f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/metric_report.jsonl"
  )

  run_script_cmds = (
      (
          "cd /tmp/xla/benchmarks && python3 experiment_runner.py"
          " --suite-name=torchbench --accelerator=tpu --progress-bar"
          f" {extraFlags}"
      ),
      "python3 /tmp/xla/benchmarks/result_analyzer.py",
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
      task_owner=test_owner.RAN_R,
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
