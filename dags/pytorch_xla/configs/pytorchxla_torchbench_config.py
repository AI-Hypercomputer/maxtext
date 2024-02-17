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

import datetime
from typing import Tuple
from xlml.apis import gcp_config, metric_config, task, test_config
import dags.vm_resource as resource
from dags import gcs_bucket, test_owner


def set_up_torchbench_tpu(model_name: str = "") -> Tuple[str]:
  """Common set up for TorchBench."""

  def model_install_cmds(output_file=None) -> str:
    """Installs torchbench models.

    Args:
      output_file: If not None, model installation message will be piped to a file.

    Returns:
      Command to install the model.
    """
    pipe_file_cmd = f" > {output_file}" if output_file else ""
    if not model_name or model_name.lower() == "all":
      return f"python install.py --continue_on_fail {pipe_file_cmd}"
    return f"python install.py models {model_name} {pipe_file_cmd}"

  return (
      "pip3 install -U setuptools",
      "sudo systemctl stop unattended-upgrades",
      "sudo apt-get -y update",
      "sudo apt install -y libopenblas-base",
      "sudo apt install -y libsndfile-dev",
      "sudo apt-get install libgl1 -y",
      "pip3 install --user numpy pandas",
      (
          "pip3 install --user --pre torch torchvision torchaudio --index-url"
          " https://download.pytorch.org/whl/nightly/cpu"
      ),
      (
          "pip3 install --user 'torch_xla[tpuvm] @"
          " https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl'"
      ),
      "pip3 install --user psutil",
      "cd; git clone https://github.com/pytorch/benchmark.git",
      f"cd benchmark && {model_install_cmds()}",
      "cd; git clone https://github.com/pytorch/pytorch.git",
      "cd; git clone https://github.com/pytorch/xla.git",
  )


def get_torchbench_tpu_config(
    tpu_version: resource.TpuVersion,
    tpu_cores: int,
    project: resource.Project,
    tpu_zone: resource.Zone,
    runtime_version: resource.RuntimeVersion,
    time_out_in_min: int,
    network: str = "default",
    subnetwork: str = "default",
    model_name: str = "",
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=tpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_tpu(model_name)
  local_output_location = "~/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/{tpu_version}/metric_report_tpu.jsonl"

  if not model_name or model_name.lower() == "all":
    run_filter = " "
  else:
    run_filter = f" --filter={model_name} "
  run_script_cmds = (
      (
          "export PJRT_DEVICE=TPU && cd ~/xla/benchmarks && python experiment_runner.py"
          " --suite-name=torchbench --xla=PJRT --accelerator=tpu --progress-bar"
          f" {run_filter}"
      ),
      "rm -rf ~/xla/benchmarks/output/metric_report.jsonl",
      "python ~/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {gcs_location}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version.value,
          network=network,
          subnetwork=subnetwork,
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
      ),
      clean_up_gcs=True,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


# Below is the setup for torchbench GPU run.
def set_up_torchbench_gpu(model_name: str, nvidia_driver_version: str) -> Tuple[str]:
  """Common set up for TorchBench."""

  def model_install_cmds(output_file=None) -> str:
    """Installs torchbench models.

    Args:
      output_file: If not None, model installation message will be piped to a file.

    Returns:
      Command to install the model.
    """
    pipe_file_cmd = f" > {output_file}" if output_file else ""
    if not model_name or model_name.lower() == "all":
      return f"python install.py --continue_on_fail {pipe_file_cmd}"
    return f"python install.py models {model_name} {pipe_file_cmd}"

  def get_nvidia_driver_install_cmd(driver_version: str) -> str:
    nvidia_driver_install = (
        "curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py",
        # Command `apt update/upgrade` receives 403 bad gateway error when connecting to the google apt repo.
        # This can be a transient error. We use the following command to fix the issue for now.
        # TODO(piz): remove the following statement for temporary fix once the `apt update/upgrade` is removed or updated.
        "sed -i '/^\\s*run(\"apt update\")/,/^\\s*return True/ s/^/# /'  install_gpu_driver.py",
        f"sed -i 's/^\\(DRIVER_VERSION = \\).*/\\1\"{driver_version}\"/' install_gpu_driver.py",
        "sudo python3 install_gpu_driver.py --force",
        "sudo nvidia-smi",
    )
    return nvidia_driver_install

  docker_cmds_ls = (
      "apt-get update && apt-get install -y libgl1",
      "apt install -y liblapack-dev libopenblas-dev",
      # Below are the required dependencies for building detectron2_* models.
      "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-12-1_12.1.0.106-1_amd64.deb",
      "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-12-1_11.4.5.107-1_amd64.deb",
      "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-12-1_12.1.3.1-1_amd64.deb",
      "dpkg -i libcusparse* libcusolver* libcublas*",
      # Below are the dependencies for benchmark data processing:
      "pip3 install --user numpy pandas",
      # torch related dependencies
      "pip3 install --user --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121",
      "cd /tmp/ && git clone https://github.com/pytorch/benchmark.git",
      f" cd benchmark && {model_install_cmds()}",
      "cd /tmp/ && git clone https://github.com/pytorch/pytorch.git",
      "cd /tmp/ && git clone https://github.com/pytorch/xla.git",
  )
  docker_cmds = "\n".join(docker_cmds_ls)

  return (
      *get_nvidia_driver_install_cmd(nvidia_driver_version),
      "sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common",
      "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
      "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -",
      "curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list",
      "sudo apt-get install -y nvidia-container-toolkit",
      # Stabilize clock freqs
      "sudo nvidia-smi --lock-gpu-clocks=1200,1200",
      "sudo systemctl restart docker",
      "sudo nvidia-smi -pm 1",
      (
          "sudo docker pull"
          " us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1"
      ),
      (
          "sudo docker run --shm-size 16g --gpus all -it -d --network host --name ml-automation-torchbench"
          " us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1"
      ),
      f"sudo docker exec -i ml-automation-torchbench /bin/bash -c '{docker_cmds}'",
  )


def get_torchbench_gpu_config(
    machine_type: resource.MachineVersion,
    image_project: resource.ImageProject,
    image_family: resource.ImageFamily,
    accelerator_type: resource.GpuVersion,
    count: int,
    gpu_zone: resource.Zone,
    time_out_in_min: int,
    nvidia_driver_version: str = "525.125.06",
    model_name: str = "",
    extraFlags: str = "",
) -> task.GpuCreateResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_gpu(model_name, nvidia_driver_version)
  local_output_location = "/tmp/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/{accelerator_type}/metric_report_gpu.jsonl"

  if not model_name or model_name.lower() == "all":
    run_filter = " "
  else:
    run_filter = f" --filter={model_name} "
  cmd_list = (
      "export PJRT_DEVICE=CUDA",
      f"export GPU_NUM_DEVICES={count}",
      "cd /tmp/xla/benchmarks",
      f"python experiment_runner.py  --suite-name=torchbench --accelerator=cuda --progress-bar --xla=PJRT --xla=None {run_filter}",
      "rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl",
      "python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl",
  )
  cmds = "\n".join(cmd_list)
  run_script_cmds = (
      (
          "sudo docker exec -i $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f" /bin/bash -c '{cmds}'"
      ),
      (
          "sudo docker cp $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f":{local_output_location} ./"
      ),
      f"gsutil cp metric_report.jsonl {gcs_location}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
          runtime_version=resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
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
      ),
      clean_up_gcs=True,
  )

  return task.GpuCreateResourceTask(
      image_project.value,
      image_family.value,
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
