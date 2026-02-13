# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""  This file contains data classes and runner logic to execute the XPK runs triggered by benchmarks.benchmark_runner"

"""
# Improvements:
# Toggle Vertex AI Experiment on/off.
# Group libtpu / jax / jaxlib dependencies instead of just libtpu.
# Split out maxtext command generation and xpk runner from this file.

import dataclasses
import enum
import json
import logging
import os
import queue
import random
import string
import subprocess
import threading
import time

import omegaconf

import benchmarks.maxtext_trillium_model_configs as model_configs
from benchmarks.globals import MAXTEXT_CONFIGS_DIR
from benchmarks.command_utils import run_command_with_updates
import benchmarks.xla_flags_library as xla_flags
from benchmarks.disruption_management.disruption_handler import DisruptionConfig
from benchmarks.disruption_management.disruption_manager import DisruptionManager
from benchmarks.xpk_configs import XpkClusterConfig


log = logging.getLogger(__name__)

# Assumes you built maxtext dep image.
# Assumes you have xpk installed in a git clone repo of ~/{wl_config.xpk_path}/xpk.py
_DEFAULT_MAXTEXT_BASE_DOCKER_IMAGE_NAME = "maxtext_base_image"

COMPLETION_TIMEOUT_SECONDS = 10

hardware_id_to_num_chips_per_node = {
    "v4": 4,
    "v5e": 4,
    "v5p": 4,
    "v6e": 4,
    "v6e-8": 8,
    "v6e-1": 1,
    "6p": 4,
}


class LibTpuType(enum.Enum):
  """Enum for different libtpu types."""

  NIGHTLY = "nightly-libtpu"
  # In order to use a custom libtpu, put a libtpu.so file in your local
  # working directory.
  CUSTOM = "custom"
  MAXTEXT = "maxtext-docker"


@dataclasses.dataclass
class PathwaysConfig:
  """Configuration for Pathways-specific settings."""

  server_image: str = None
  proxy_server_image: str = None
  runner_image: str = None
  colocated_python_sidecar_image: str = None
  server_flags: str = ""
  proxy_flags: str = ""
  worker_flags: str = ""
  headless: bool = False


# TODO(@vbarr): Split out parameters related to XPK workload and a General workload
@dataclasses.dataclass
class WorkloadConfig:
  """Class representing for passing general workload parameters"""

  model: model_configs.MaxTextModel
  num_slices: int
  device_type: str
  base_output_directory: str
  base_docker_image: str
  libtpu_type: LibTpuType
  libtpu_nightly_version: str = None  # A date in %Y%M%D format, 20241201
  num_steps: int = 20
  max_restarts: int = 0
  priority: str = "medium"
  xpk_path: str = os.path.join("~", "xpk")
  pathways_config: PathwaysConfig = None
  run_name: str = None
  generate_metrics_and_upload_to_big_query: bool = True
  hardware_id: str = "v6e"
  metrics_gcs_file: str = ""
  base_config: str = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  topology: str = dataclasses.field(init=False)
  num_devices_per_slice: int = dataclasses.field(init=False)
  db_project: str = ""
  db_dataset: str = ""
  disruption_configs: DisruptionConfig = None
  xpk_storage: None | list[str] = None
  hlo_dump: None | bool = None

  def __post_init__(self):
    """Initializes num_devices_per_slice and topology for recording the run into BigQuery"""
    if not self.generate_metrics_and_upload_to_big_query:
      return
    if self.device_type is None:
      raise ValueError(
          "device_type is None and generate_metrics_and_upload_to_big_query is enabled. "
          "Device_type is required for uploading run results to BigQuery"
      )
    size = int(self.device_type.split("-")[-1])
    if (
        self.device_type.startswith("v6e")
        or self.device_type.startswith("v5e")
        or self.device_type.startswith("v5litepod")
    ):
      if size == 256:
        self.num_devices_per_slice = 256
        self.topology = "16x16"
      elif size == 128:
        self.num_devices_per_slice = 128
        self.topology = "8x16"
      elif size == 64:
        self.num_devices_per_slice = 64
        self.topology = "8x8"
      elif size == 32:
        self.num_devices_per_slice = 32
        self.topology = "4x8"
      elif size == 16:
        self.num_devices_per_slice = 16
        self.topology = "4x4"
      elif size == 8:
        self.num_devices_per_slice = 8
        self.topology = "2x4"
      elif size == 4:
        self.num_devices_per_slice = 4
        self.topology = "2x2"
      else:
        raise ValueError(f"Unsupported v5e or v6e size: {size}")
    else:
      self.num_devices_per_slice = size / 2
      self.topology = ""
    self.hardware_id = self.device_type.split("-")[0]
    if self.hardware_id == "v5litepod":
      self.hardware_id = "v5e"


def wait_for_xpk_workload_completion(cluster_config: XpkClusterConfig, workload_name, xpk_path) -> int:
  """Waits for the given XPK workload to complete.
  Args:
    cluster_config: XPK cluster configuration.
    workload_name: Name of the workload to wait for.
    xpk_path: Path to the xpk.py script.
  Returns:
    return_code: 0 if successful and non-zero otherwise.
  """
  wait_command = [
      f"python3 {xpk_path}/xpk.py workload list",
      f"--cluster={cluster_config.cluster_name}",
      f"--project={cluster_config.project}",
      f"--zone={cluster_config.zone}",
      f"--wait-for-job-completion={workload_name}",
  ]
  wait_command_str = " ".join(wait_command)
  print(f'Waiting for workload "{workload_name}" to complete...')
  return_code = run_command_with_updates(wait_command_str, f"Wait for {workload_name} completion")
  if return_code != 0:
    print(f"Error waiting for workload {workload_name} to complete. Return code:" f" {return_code}")
  else:
    print(f'Workload "{workload_name}" completed successfully.')
  return return_code


def wait_for_xpk_workloads_completion_async(cluster_config: XpkClusterConfig, workload_names, xpk_path):
  """Waits for a list of XPK workloads to complete in parallel and yields names and exit codes as they complete.
  Args:
    cluster_config: XPK cluster configuration.
    workload_names: list of workload names to wait for.
    xpk_path: Path to the xpk.py script.
  Yields:
    tuple[workload_name, return_code]: The name of the workload that has just
      completed and its return code.
  """
  threads = []
  result_queue = queue.Queue()

  def _wait_for_completion_threaded(name):
    return_code = wait_for_xpk_workload_completion(cluster_config, name, xpk_path)
    result_queue.put((name, return_code))

  for name in workload_names:
    thread = threading.Thread(target=_wait_for_completion_threaded, args=(name,))
    threads.append(thread)
    thread.start()

  completed_count = 0
  while completed_count < len(workload_names):
    try:
      # Wait for a result with a timeout
      workload_name, return_code = result_queue.get(timeout=COMPLETION_TIMEOUT_SECONDS)
      completed_count += 1

      # Yield the result as soon as it's available
      yield workload_name, return_code
    except queue.Empty:
      # Queue is empty, no thread has finished yet, continue waiting
      print("Waiting for workloads to complete...")
      time.sleep(10)


def _get_config_tuning_params(wl_config: WorkloadConfig):
  """Get config tuning parameters for the workload.

  Args:
    wl_config: Workload configuration.

  Returns:
    A string of config tuning parameters.
  """
  is_pw_enabled = wl_config.pathways_config is not None

  config_tuning_params = ""
  unified_tuning_params = wl_config.model.tuning_params.copy()  # Create a copy

  # Overwrite the tuning params with pathways specific tuning params if present.
  # otherwise add them to the dictionary. If pathays tuning params are not
  # present, add the default pathways tuning params.
  if is_pw_enabled:
    if wl_config.model.pathways_tuning_params is None:
      print(
          "WARNING: Pathways tuning params are not present for model:"
          f" {wl_config.model.model_name}, Adding the following base params to"
          f" support pathways: {model_configs.BASE_PATHWAYS_TUNING_PARAMS}"
      )
      wl_config.model.pathways_tuning_params = model_configs.BASE_PATHWAYS_TUNING_PARAMS

    # Automatically inject Base Pathways tuning params if not present. The user
    # can override these values if they want, but if not present, we will add
    # them to the dictionary.
    for key, value in model_configs.BASE_PATHWAYS_TUNING_PARAMS.items():
      if key not in wl_config.model.pathways_tuning_params:
        wl_config.model.pathways_tuning_params[key] = value

      print(
          f"WARNING: {key} is not present in pathways tuning"
          f" params for model: {wl_config.model.model_name}, Adding the"
          f" param {key}={value} to support pathways."
      )

    print(
        f"Pathways tuning params for model: {wl_config.model.model_name} are:"
        f" {wl_config.model.pathways_tuning_params}"
    )
    for key, value in wl_config.model.pathways_tuning_params.items():
      unified_tuning_params[key] = value

  print(f"Unified tuning params for model are:" f" {unified_tuning_params}")

  for key, value in unified_tuning_params.items():
    config_tuning_params += f"{key}={value} "

  return config_tuning_params


def _build_args_from_config(wl_config: WorkloadConfig) -> dict:
  """Builds a dictionary of arguments for metrics upload from the workload config.

  This function extracts various configuration details and formats them into a
  dictionary that will be used to generate command-line arguments for the
  `upload_metrics_to_bq.py` script.

  Args:
    wl_config: The WorkloadConfig object containing all run details.

  Returns:
    A dictionary of arguments for the metrics upload script.
  """
  base_config = omegaconf.OmegaConf.load(wl_config.base_config)

  # Extract per_device_batch_size arg
  if "per_device_batch_size" not in wl_config.model.tuning_params:
    per_device_batch_size = base_config.per_device_batch_size
  else:
    per_device_batch_size = wl_config.model.tuning_params["per_device_batch_size"]

  # Extract precision arg
  if "matmul_precision" not in wl_config.model.tuning_params:
    precision = base_config.matmul_precision
  else:
    precision = wl_config.model.tuning_params["matmul_precision"]

  # Extract optimizer arg
  if "opt_type" not in wl_config.model.tuning_params:
    optimizer = base_config.opt_type
  else:
    optimizer = wl_config.model.tuning_params["opt_type"]

  # Extract sequence_length arg
  if "max_target_length" not in wl_config.model.tuning_params:
    sequence_length = base_config.opt_type
  else:
    sequence_length = wl_config.model.tuning_params["max_target_length"]

  # Extract dataset arg
  if "dataset_type" not in wl_config.model.tuning_params:
    dataset = base_config.opt_type
  else:
    dataset = wl_config.model.tuning_params["dataset_type"]

  # Extract xla_flags arg
  xla_flags_str = wl_config.model.xla_flags.strip().replace(" ", ",")

  # Extract tuning_params arg
  tuning_params_str = json.dumps(wl_config.model.tuning_params)

  num_steps = wl_config.num_steps  # default case

  if (
      "steps" in wl_config.model.tuning_params and wl_config.num_steps == -1
  ):  # replace num_step for not provided num_steps and configuration exist in tuning_params.
    num_steps = wl_config.model.tuning_params["steps"]
    log.info("using steps=(%d) in model convergence test setup", num_steps)

  return {
      "metrics_gcs_file": wl_config.metrics_gcs_file,
      "model_id": wl_config.model.model_type,
      "hardware_id": wl_config.hardware_id,
      "software_id": "jax_maxtext",
      "hardware_num_slices": wl_config.num_slices,
      "number_of_chips": wl_config.num_devices_per_slice * wl_config.num_slices,
      "container_image_name": wl_config.base_docker_image,
      "global_batch_size": per_device_batch_size * wl_config.num_devices_per_slice * wl_config.num_slices,
      "precision": precision,
      "optimizer": optimizer,
      "seq_length": sequence_length,
      "number_of_steps": wl_config.num_steps,
      "xla_flags": f"'{xla_flags_str}'",
      "dataset": dataset,
      "run_type": "maxtext-xpk",
      "config_file": os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
      "topology": wl_config.topology,
      "tuning_params": f"'{tuning_params_str}'",
      "db_project": wl_config.db_project,
      "db_dataset": wl_config.db_dataset,
  }


def build_user_command(
    name: str,
    wl_config: WorkloadConfig,
):
  """Builds the user command string to be executed within the XPK workload.

  Args:
    name: The name of the workload.
    wl_config: The WorkloadConfig object containing configuration details.

  Returns:
    A string representing the full command to run MaxText.
  """
  is_pw_enabled = wl_config.pathways_config is not None

  config_tuning_params = _get_config_tuning_params(wl_config)

  install_libtpu_cmd = ""
  jax_platforms = None
  vertex_tensorboard = ""
  # TODO() support modifying nightly / stable dependencies in pathway flow
  if is_pw_enabled:
    jax_platforms = "proxy"
  else:
    if wl_config.libtpu_type == LibTpuType.NIGHTLY:
      if wl_config.libtpu_nightly_version:
        install_libtpu_cmd += (
            f" python3 -m pip install libtpu=={wl_config.libtpu_nightly_version} -f"
            " https://storage.googleapis.com/libtpu-wheels/index.html &&"
        )
      else:
        # If no version is specified, install the latest stable libtpu.
        install_libtpu_cmd += (
            " python3 -m pip install libtpu --pre -f" " https://storage.googleapis.com/libtpu-wheels/index.html &&"
        )
    elif wl_config.libtpu_type == LibTpuType.CUSTOM:
      # In order to use a custom libtpu, put a libtpu.so file in your local
      # working directory.
      install_libtpu_cmd += " mv libtpu.so /lib/ &&"
    elif wl_config.libtpu_type == LibTpuType.MAXTEXT:
      # Use the libtpu dependent built in the docker image provided.
      install_libtpu_cmd += ""

    jax_platforms = "tpu,cpu"
    vertex_tensorboard = 'use_vertex_tensorboard=false vertex_tensorboard_project="" vertex_tensorboard_region=""'

  assert jax_platforms is not None, "Error in setting jax_platforms"

  libtpu_flags = f"LIBTPU_INIT_ARGS='{wl_config.model.xla_flags}'"

  if name is None:
    run_name_command = ""
  else:
    run_name_command = f"run_name={name}"

  enable_metrics_cmd = ""
  if wl_config.generate_metrics_and_upload_to_big_query:
    # 'metrics_file=metrics.txt
    # Save metrics to gcs bucket so that we can upload them to bq in post processing.
    enable_metrics_cmd = "gcs_metrics=true"

  upload_hlo_dump = ""
  hlo_dump = ""
  if wl_config.hlo_dump:
    hlo_dump = "XLA_FLAGS='--xla_dump_large_constants --xla_dump_to=/tmp/xla_dump'"
    upload_hlo_dump = (
        f" && gsutil -m cp -r /tmp/xla_dump  {wl_config.base_output_directory}/{wl_config.run_name}/hlo_dump"
    )
  # Construct the command string with proper formatting and line continuations
  command = " ".join(
      [
          f"{install_libtpu_cmd}",
          f"echo {libtpu_flags} &&" if not is_pw_enabled else "",
          f"export {libtpu_flags} &&" if not is_pw_enabled else "",
          "export ENABLE_PATHWAYS_PERSISTENCE=1 &&",
          f"export JAX_PLATFORMS={jax_platforms} &&",
          "export ENABLE_PJRT_COMPATIBILITY=true &&",
          "export MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets MAXTEXT_PKG_DIR=/deps/src/MaxText MAXTEXT_REPO_ROOT=/deps &&"
          f'{hlo_dump} python3 -m MaxText.train {os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")}',
          f"{config_tuning_params}",
          f"steps={wl_config.num_steps}",
          f"model_name={wl_config.model.model_type}",
          f"base_output_directory={wl_config.base_output_directory}",
          f"{vertex_tensorboard}",
          f"{run_name_command}",
          f"{enable_metrics_cmd}",
          f"{upload_hlo_dump}",
      ]
  )
  return command


def _get_pathways_proxy_flags(wl_config: WorkloadConfig):
  """Get the pathways proxy flags for the workload and removes any extras."""
  # Add in the xla flags alongside the proxy flags from the pathways config.
  pw_config = wl_config.pathways_config

  # Get proxy and xla flag string from model config
  proxy_flags_string = pw_config.proxy_flags
  xla_flags_string = wl_config.model.xla_flags if not pw_config.headless else ""

  # Split both proxy_flags_string and xla_flags_string into lists of flags
  proxy_flags_list = proxy_flags_string.strip().split()
  xla_flags_list = xla_flags_string.strip().split()

  # Combine the two lists of flags into a single list
  proxy_flags = proxy_flags_list + xla_flags_list

  # Remove the flags that are specified to be removed.
  if not pw_config.headless and (
      wl_config.model.pathways_xla_flag_options and xla_flags.REMOVE in wl_config.model.pathways_xla_flag_options
  ):
    flags_to_remove = wl_config.model.pathways_xla_flag_options[xla_flags.REMOVE]
    updated_proxy_flags = []
    for flag in proxy_flags:
      if flag not in flags_to_remove:
        updated_proxy_flags.append(flag)
    proxy_flags = updated_proxy_flags

  # Add the flags that are specified to be added.
  if not pw_config.headless and (
      wl_config.model.pathways_xla_flag_options and xla_flags.ADD_PROXY in wl_config.model.pathways_xla_flag_options
  ):
    flags_to_add = wl_config.model.pathways_xla_flag_options[xla_flags.ADD_PROXY]
    flags_to_add_list = flags_to_add.strip().split()
    proxy_flags += flags_to_add_list

  # Join the list of flags back into a single string, space-separated
  return " ".join(proxy_flags)


def _combine_flag_strings(base_flags: str, flags_to_add: str) -> str:
  """Combines two flag strings and removes extraneous whitespace."""
  all_flags = base_flags.split() + flags_to_add.split()
  return " ".join(all_flags)


def _get_pathways_worker_flags(wl_config: WorkloadConfig):
  """Get the pathways worker flags for the workload and removes any extras."""
  # Add in the xla flags alongside the worker flags from the pathways config.
  pw_config = wl_config.pathways_config

  # Get worker and xla flag string from model config
  worker_flags = pw_config.worker_flags

  # Add the flags that are specified to be added.
  if not pw_config.headless and (
      wl_config.model.pathways_xla_flag_options and xla_flags.ADD_WORKER in wl_config.model.pathways_xla_flag_options
  ):
    flags_to_add = wl_config.model.pathways_xla_flag_options[xla_flags.ADD_WORKER]

    worker_flags = _combine_flag_strings(worker_flags, flags_to_add)

  # Join the list of flags back into a single string, space-separated
  return worker_flags


def _get_pathways_server_flags(wl_config: WorkloadConfig):
  """Get the pathways server flags for the workload and removes any extras."""
  # Add in the xla flags alongside the server flags from the pathways config.
  pw_config = wl_config.pathways_config

  # Get server and xla flag string from model config
  server_flags = pw_config.server_flags

  # Add the flags that are specified to be added.
  if not pw_config.headless and (
      wl_config.model.pathways_xla_flag_options and xla_flags.ADD_SERVER in wl_config.model.pathways_xla_flag_options
  ):
    flags_to_add = wl_config.model.pathways_xla_flag_options[xla_flags.ADD_SERVER]
    server_flags = _combine_flag_strings(server_flags, flags_to_add)

  # Join the list of flags back into a single string, space-separated
  return server_flags


def _get_pathways_specific_flags(wl_config: WorkloadConfig):
  """Generates Pathways-specific flags for the XPK workload creation command.

  These flags include image paths, GCS locations, and custom arguments for
  Pathways server, proxy, and worker components.

  Args:
    wl_config: The WorkloadConfig object.

  Returns:
    A string of space-separated Pathways-specific flags.
  """
  pw_config = wl_config.pathways_config
  if pw_config is None:
    return ""

  colocated_python_sidecar_image_flag = (
      f" --colocated-python-sidecar-image={pw_config.colocated_python_sidecar_image}"
      if pw_config.colocated_python_sidecar_image is not None
      else ""
  )
  server_image_flag = f" --server-image={pw_config.server_image}" if pw_config.server_image is not None else ""
  proxy_server_image_flag = (
      f" --proxy-server-image={pw_config.proxy_server_image}" if pw_config.proxy_server_image is not None else ""
  )

  proxy_flags = _get_pathways_proxy_flags(wl_config)
  worker_flags = _get_pathways_worker_flags(wl_config)
  server_flags = _get_pathways_server_flags(wl_config)

  pathways_specific_flags = (
      f" {server_image_flag} "
      f" {proxy_server_image_flag} "
      f" {colocated_python_sidecar_image_flag} "
      f" --termination-grace-period-seconds=300 "
      f" --pathways-gcs-location={wl_config.base_output_directory} "
      f' --custom-pathways-server-args="{server_flags}" '
      f' --custom-pathways-proxy-server-args="{proxy_flags}" '
      f' --custom-pathways-worker-args="{worker_flags}" '
      f' {"--headless" if pw_config.headless else ""}'
  )
  return pathways_specific_flags


def generate_xpk_workload_cmd(
    cluster_config: XpkClusterConfig,
    wl_config: WorkloadConfig,
    workload_name=None,
    user=os.environ["USER"],
    temp_key=None,
    exp_name=None,
):
  """Generates a command to run a maxtext model on XPK."""

  is_pathways_enabled = wl_config.pathways_config is not None
  is_pathways_headless_enabled = wl_config.pathways_config and wl_config.pathways_config.headless

  time.localtime()
  length_of_random_str = 3
  # Allow DAG to resolve workload name for cleanup, preventing reliance on random IDs
  if temp_key is not None:
    temp_post_fix = temp_key
  else:
    temp_post_fix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length_of_random_str))

  truncate_model_name = 10
  truncate_prefix = 3
  post_fix = f"-{wl_config.num_slices}-{time.strftime('%m%d%H', time.localtime())}-{temp_post_fix}"
  common_prefix = user
  pw_prefix = "pw-"

  if workload_name is None:
    if is_pathways_enabled:
      post_fix = f"-{wl_config.num_slices}-{temp_post_fix}"
      name = f"{pw_prefix}{wl_config.model.model_name.replace('_', '-')[:truncate_model_name - len(pw_prefix)]}"
    else:
      name = f"{wl_config.model.model_name.replace('_', '-')[:truncate_model_name]}"
    name = f"{common_prefix[:truncate_prefix]}-{name}{post_fix}"
  else:
    name = workload_name  # Use provided name

  wl_config.run_name = name
  wl_config.metrics_gcs_file = os.path.join(wl_config.base_output_directory, wl_config.run_name, "metrics")

  user_command = ""
  if not is_pathways_headless_enabled:
    user_command = build_user_command(name=name, wl_config=wl_config)

  additional_flags = ""
  if not is_pathways_enabled and wl_config.libtpu_type == LibTpuType.CUSTOM:
    additional_flags = '--env="TPU_LIBRARY_PATH=/lib/libtpu.so"'

  # docker_image_flag = ""
  # pathways-related flags
  # pathways_specific_flags = ""
  workload_create_command = f"python3 {wl_config.xpk_path}/xpk.py workload create"
  device_type = f" --device-type={cluster_config.device_type}"
  if is_pathways_enabled:
    pw_config = wl_config.pathways_config
    device_type = f" --tpu-type={wl_config.device_type}"
    workload_create_command = f"python3 {wl_config.xpk_path}/xpk.py workload create-pathways"
    docker_image_flag = f"--docker-image={pw_config.runner_image}"
  else:
    docker_image_flag = f'--base-docker-image="{wl_config.base_docker_image}"'

  upload_metrics_to_bq_cmd = ""
  if wl_config.generate_metrics_and_upload_to_big_query and not is_pathways_headless_enabled:
    # TODO (optionally) make it so that this upload step is done on local device instead of within the workload.
    args = _build_args_from_config(wl_config)
    args_str = ""
    for k, v in args.items():
      args_str += f"--{k}={v} "
    upload_metrics_to_bq_cmd = f"&& python3 -m benchmarks.upload_metrics_to_bq {args_str}"

  print(f"User command: {user_command}")
  all_xpk_storage = ""
  if wl_config.xpk_storage:
    all_xpk_storage = " ".join(f"--storage={storage_test}" for storage_test in wl_config.xpk_storage)

  hlo_dump = ""
  if wl_config.hlo_dump:
    # HLO dump gets saved in a subdirectory called "hlo_dump" of the base output directory.
    hlo_dump = f"--debug-dump-gcs={wl_config.base_output_directory}/{wl_config.run_name}/hlo_dump"

  return (
      (
          f"{workload_create_command}"
          f" {_get_pathways_specific_flags(wl_config)}"
          f" --cluster={cluster_config.cluster_name}"
          f" --project={cluster_config.project}"
          f" --zone={cluster_config.zone}"
          f" {device_type}"
          f" {all_xpk_storage}"
          f" --num-slices={wl_config.num_slices}"
          f' --command="{user_command} {upload_metrics_to_bq_cmd}"'
          f" {docker_image_flag}"
          " --enable-debug-logs"
          f" --workload={name}"
          f" --priority={wl_config.priority}"
          f" --max-restarts={wl_config.max_restarts}"
          f" {hlo_dump}"
          # ' --use-vertex-tensorboard'
          # f' --experiment-name={exp_name}'
          f" {additional_flags}"
      ),
      name,
  )


def run_xpk_workload(
    cluster_config: XpkClusterConfig,
    wl_config: WorkloadConfig,
    wait_for_completion: bool = False,
) -> int:
  """Runs a maxtext model on XPK and waits for completion.

  Args:
    cluster_config: XPK cluster configuration.
    wl_config: Workload configuration.
    wait_for_completion: Whether to wait for workload completion. Defaults to
      False.

  Returns:
    return_code: Return code of the workload creation command, or workload
    completion wait command if wait_for_completion is True.
  """
  assert (
      cluster_config.device_type == wl_config.device_type
  ), f"The workload device size {wl_config.device_type}, and cluster device size {cluster_config.device_type} don't match."
  command, workload_name = generate_xpk_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)
  return_code = run_command_with_updates(command, "Run XPK workload")
  if return_code == 0 and wait_for_completion:
    return_code = wait_for_xpk_workload_completion(
        cluster_config, workload_name, wl_config.xpk_path
    )  # Wait for completion after successful run
  return return_code


def xpk_benchmark_runner(
    cluster_config: XpkClusterConfig,
    workload_configs: list[WorkloadConfig],
    user=os.environ["USER"],
    disruption_manager: DisruptionManager = DisruptionManager(),
    exp_name: str = None,
):
  """Runs a list of MaxText workloads on XPK.

  This function generates and executes XPK workload creation commands for each
  WorkloadConfig provided. It also integrates with a DisruptionManager if
  disruption configurations are present.

  Args:
    cluster_config: The XpkClusterConfig object for the target cluster.
    workload_configs: A list of WorkloadConfig objects to run.
    disruption_manager: An optional DisruptionManager instance.
    exp_name: Optional. An experiment name for Vertex AI TensorBoard.

  Returns:
    The DisruptionManager instance, potentially updated with new workloads.
  """
  xpk_workload_names = []
  xpk_workload_cmds = []
  for wl_config in workload_configs:
    command, name = generate_xpk_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        user=user,
        exp_name=exp_name,
    )

    print(f"Name of the workload is: {name} \n")
    xpk_workload_names.append(name)

    print(f"XPK command to be used is: {command} \n")
    xpk_workload_cmds.append(command)

    if wl_config.disruption_configs:
      disruption_manager.add_workload(
          workload_name=name,
          cluster_config=cluster_config,
          disruption_configs=wl_config.disruption_configs,
      )

  # TODO(@vbarr) Support batch workloads.
  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    # Starts the workloads, but does not wait for the workloads to complete.
    return_code = run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      # If the workload fails to start, remove it from the disruption manager.
      # No-op if disruption manager does not contain the workload name.
      disruption_manager.remove_workload(xpk_workload_name)
      print(f"Unable to run xpk workload: {xpk_workload_name}")

  return disruption_manager


def on_device_benchmark_runner(
    workload_configs: list[WorkloadConfig],
):
  """Runs a list of MaxText workloads directly on the device.

  This function iterates through the provided workload configurations and
  executes the generated user command directly using `subprocess.run`.

  Args:
    workload_configs: A list of WorkloadConfig objects to run.
  """
  for wl_config in workload_configs:
    user_command = build_user_command(name=wl_config.run_name, wl_config=wl_config)
    print(f"User command: {user_command}")
    subprocess.run(user_command, shell=True, text=True, check=True)


# Run maxtext_xpk_runner.py as a script for executing multiple workloads pythonically!
def main() -> int:
  """Main function to configure and run MaxText XPK benchmarks.

  This function sets up cluster configurations, defines a list of models,
  and then iterates through various configurations (clusters, slices, libtpu
  types) to generate and execute XPK workload commands.

  Returns:
    An integer representing the exit status (os.EX_OK for success).
  """
  # Variables to configure:
  output_bucket = "gs://maxtext-experiments-temp/"
  base_docker_image = _DEFAULT_MAXTEXT_BASE_DOCKER_IMAGE_NAME
  # Configure these for writing to benchmark DB
  db_project = "supercomputer-testing"
  db_dataset = "mantaray_v2"

  # Set up the clusters to run workloads on!
  # v5e_cluster_config = XpkClusterConfig(
  #     cluster_name="v5e-256",
  #     project="my-cool-project",
  #     zone="us-central2-b",
  #     device_type="v5litepod-256",
  # )

  v6e_cluster_config = XpkClusterConfig(
      cluster_name="v6e-256",
      project="my-cool-project",
      zone="us-central2-b",
      device_type="v6e-256",
  )

  xpk_workload_cmds = []
  xpk_workload_names = []

  list_of_models = [
      # model_configs.llama2_70b_4096_sc,
      # model_configs.default_128
      model_configs.llama3_1_70b_131072,
  ]

  # Loop possibilities:
  # 1. Test different libtpu nightly versions.
  #  for libtpu_type in [
  #           LibTpuType.NIGHTLY
  #       ]:
  #     todays_date = time.strftime('%Y%m%d')
  #    for date in ['20241201', '20241202', todays_date]:

  # 2. Test different model configurations.
  # for remat_policy in ['qkv_proj_offloaded', 'minimal']:
  #   model.tuning_params['remat_policy'] = remat_policy

  # 3. See other examples below

  user = os.environ["USER"]
  base_output_dir = os.path.join(output_bucket, user)

  for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
        # v5e_cluster_config,
        v6e_cluster_config,
        # another_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in [
          1,
      ]:
        # Use the libtpu dependencies from:
        for libtpu_type in [
            # LibTpuType.CUSTOM
            LibTpuType.MAXTEXT
            # LibTpuType.NIGHTLY
        ]:
          wl_config = WorkloadConfig(
              db_project=db_project,
              db_dataset=db_dataset,
              model=model,
              num_slices=num_slices,
              device_type=cluster_config.device_type,
              base_output_directory=base_output_dir,
              priority="medium",
              max_restarts=0,
              libtpu_type=libtpu_type,
              libtpu_nightly_version="",
              base_docker_image=base_docker_image,
              pathways_config=None,
          )
          command, name = generate_xpk_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)

          print(f"Name of the workload is: {name} \n")
          xpk_workload_names.append(name)

          print(f"XPK command to be used is: {command} \n")
          xpk_workload_cmds.append(command)

  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    return_code = run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")

  # Support Batch workloads one day. Note that this doesn't show the xpk logs per workload.
  # They are saved to file instead.
  # return_codes = run_commands(
  #     xpk_workload_cmds,
  #     'Run XPK workloads',
  #     xpk_workload_names,
  #     batch=1,  # Parallel execution of workloads is not supported in XPK yet.
  #     dry_run=False,
  # )
  # print(f'Return_codes: {return_codes}')
  return os.EX_OK


if __name__ == "__main__":
  main()
