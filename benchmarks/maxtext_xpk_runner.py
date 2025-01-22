"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""  This file contains data classes and runner logic to execute the XPK runs triggered by benchmarks/benchmark_runner.py"

"""
# Improvements:
# Toggle Vertex AI Experiment on/off.
# Group libtpu / jax / jaxlib dependencies instead of just libtpu.
# Split out maxtext command generation and xpk runner from this file.
# Enable hlo dumps.

import dataclasses
import datetime
import enum
import os
import random
import string
import subprocess
import sys
import tempfile
import time

import maxtext_trillium_model_configs as model_configs

# Assumes you built maxtext dep image.
# Assumes you have xpk installed in a git clone repo of ~/{wl_config.xpk_path}/xpk.py
_DEFAULT_MAXTEXT_BASE_DOCKER_IMAGE_NAME = 'maxtext_base_image'

class LibTpuType(enum.Enum):
  NIGHTLY = 'nightly-libtpu'
  # In order to use a custom libtpu, put a libtpu.so file in your local
  # working directory.
  CUSTOM = 'custom'
  MAXTEXT = 'maxtext-docker'


@dataclasses.dataclass
class PathwaysConfig:
  server_image: str
  proxy_image: str
  runner_image: str


# TODO(@vbarr): Split out parameters related to XPK workload and a General workload
@dataclasses.dataclass
class WorkloadConfig:
  """Class representing for passing general workload parameters"""

  model: model_configs.MaxTextModel
  num_slices: str
  device_type: str
  base_output_directory: str
  base_docker_image: str
  libtpu_type: LibTpuType
  libtpu_nightly_version: str = None # A date in %Y%M%D format, 20241201
  num_steps: int = 20
  max_restarts: int = 0
  priority: str = "medium"
  xpk_path: str = '~/xpk'
  pathways_config: PathwaysConfig = None


@dataclasses.dataclass
class XpkClusterConfig:
  """Holds details related to a XPK cluster to run workloads on."""

  cluster_name: str
  project: str
  zone: str
  device_type: str


def chunks(lst: list, n: int):
  """Return a list of n-sized chunks from lst.

  Args:
    lst: input list to get chunks from.
    n: size of each chunk.

  Returns:
    List of n-sized chunks for lst.
  """
  return [lst[i : i + n] for i in range(0, len(lst), n)]


def make_tmp_files(per_command_name):
  """Make temporary files for each command.

  Args:
    per_command_name: list of command names.

  Returns:
    A list of temporary files for each command.
  """
  # Supports removal of spaces from command names before converting to file name.
  return [
      tempfile.NamedTemporaryFile(
          delete=False, prefix=command.replace(' ', '-') + '-'
      )
      for command in per_command_name
  ]


def run_commands(commands, jobname, per_command_name, batch=10, dry_run=False):
  """Run commands in groups of `batch`.

  Args:
    commands: list of command.
    jobname: the name of the job.
    per_command_name: list of command names.
    batch: number of commands to run in parallel.
    dry_run: enables dry_run if set to true.

  Returns:
    0 if successful and 1 otherwise.
  """
  temporary_files_batches = chunks(make_tmp_files(per_command_name), batch)
  commands_batched = chunks(commands, batch)
  per_command_name_batches = chunks(per_command_name, batch)

  print(
      f'Breaking up a total of {len(commands)} commands into'
      f' {len(commands_batched)} batches'
  )
  if dry_run:
    print('Pretending all the jobs succeeded')
    return 0

  max_return_code = 0
  for i, _ in enumerate(commands_batched):
    print(f'Dispatching batch {i}/{len(commands_batched)}')
    batch_max_return_code, _ = run_command_batch(
        commands_batched[i],
        jobname,
        per_command_name_batches[i],
        temporary_files_batches[i],
    )
    max_return_code = max(max_return_code, batch_max_return_code)
    if max_return_code > 0:
      return max_return_code
  return max_return_code


def run_command_batch(commands, jobname, per_command_name, output_logs):
  """Runs commands in parallel.

  Args:
    commands: list of n commands, each command is a a list of strings
    jobname: Useful debugging name for the group of commands
    per_command_name: specific name per task
    output_logs: list of n log paths, each command will output to each log.

  Returns:
    The max return code and a list of all the return codes.
  """

  children = []
  start_time = datetime.datetime.now()
  for i, command in enumerate(commands):
    children.append(
        # subprocess managed by list pylint: disable=consider-using-with
        subprocess.Popen(
            command, stdout=output_logs[i], stderr=output_logs[i], shell=True
        )
    )

  while True:
    returncodes = [child.poll() for child in children]
    max_returncode = max([0] + [r for r in returncodes if r is not None])
    completed = len([r for r in returncodes if r is not None])
    total = len(returncodes)
    seconds_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if completed < total:
      slow_worker_index = returncodes.index(None)
      slow_worker_text = per_command_name[slow_worker_index]
      slow_str = (
          f', task {slow_worker_text} still working, logfile'
          f' {output_logs[slow_worker_index].name}'
      )
    else:
      slow_str = ''
    print(
        f'[t={seconds_elapsed:.2f}, {jobname}] Completed'
        f' {completed}/{total}{slow_str}'
    )
    if max_returncode > 0:
      failing_index = [
          i for i, x in enumerate(returncodes) if x is not None and x > 0
      ][0]
      print(f'Terminating all {jobname} processes since at least one failed.')
      print(
          f'Failure is {per_command_name[failing_index]}'
          f' and logfile {output_logs[failing_index].name}'
      )
      for child in children:
        child.terminate()
      break

    if completed == total:
      break

    time.sleep(1)
  return max_returncode, returncodes


def run_command_with_updates(command, task, verbose=True) -> int:
  """Generic run commands function with updates.

  Args:
    command: command to execute
    task: user-facing name of the task
    global_args: user provided arguments for running the command.
    verbose: shows stdout and stderr if set to true. Set to True by default.

  Returns:
    0 if successful and 1 otherwise.
  """

  if verbose:
    print(
        f'Task: `{task}` is implemented by `{command}`, streaming output live.'
    )
    with subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
    ) as child:
      i = 0
      while True:
        return_code = child.poll()
        if return_code is None:
          print(f'Waiting for `{task}`, for {i} seconds')
          time.sleep(1)
          i += 1
        else:
          print(f'Task: `{task}` terminated with code `{return_code}`')
          return return_code
  else:
    print(
        f'Task: `{task}` is implemented by `{command}`, hiding output unless'
        ' there is an error.'
    )
    try:
      subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print(
          f'Task: `{task}` terminated with ERROR `{e.returncode}`, printing'
          ' logs'
      )
      print('*' * 80)
      print(e.output)
      print('*' * 80)
      return e.returncode
    print(f'Task: `{task}` succeeded.')
    return 0


def build_user_command(
    name: str,
    wl_config: WorkloadConfig,
):
  is_pw_enabled = wl_config.pathways_config is not None

  config_tuning_params = ''
  for key, value in wl_config.model.tuning_params.items():
    config_tuning_params += f'{key}={value} '

  install_libtpu_cmd = ''
  jax_platforms = None
  vertex_tensorboard = None
  # TODO() support modifying nightly / stable dependencies in pathway flow
  if is_pw_enabled:
    jax_platforms = 'proxy'
  else:
    if wl_config.libtpu_type == LibTpuType.NIGHTLY:
      install_libtpu_cmd += (
          f' pip install libtpu-nightly==0.1.dev{wl_config.libtpu_nightly_version} -f'
          ' https://storage.googleapis.com/libtpu-releases/index.html &&'
      )
    elif wl_config.libtpu_type == LibTpuType.CUSTOM:
      # In order to use a custom libtpu, put a libtpu.so file in your local
      # working directory.
      install_libtpu_cmd += ' mv libtpu.so /lib/ &&'
    elif wl_config.libtpu_type == LibTpuType.MAXTEXT:
      # Use the libtpu dependent built in the docker image provided.
      install_libtpu_cmd += ''

    jax_platforms = 'tpu,cpu'
    vertex_tensorboard = 'use_vertex_tensorboard=false vertex_tensorboard_project="" vertex_tensorboard_region=""'

  assert jax_platforms is not None, 'Error in setting jax_platforms'

  libtpu_flags = f'LIBTPU_INIT_ARGS=\'{wl_config.model.xla_flags}\''

  # Construct the command string with proper formatting and line continuations
  command = ' '.join([
      f'{install_libtpu_cmd}',
      f'echo {libtpu_flags} &&' if not is_pw_enabled else '',
      f'export {libtpu_flags} &&' if not is_pw_enabled else '',
      'export ENABLE_PATHWAYS_PERSISTENCE=1 &&',
      f'export JAX_PLATFORMS={jax_platforms} &&',
      'export ENABLE_PJRT_COMPATIBILITY=true &&',
      'python3 MaxText/train.py MaxText/configs/base.yml',
      f'{config_tuning_params}',
      f'steps={wl_config.num_steps}',
      f'model_name={wl_config.model.model_type}',
      f'base_output_directory={wl_config.base_output_directory}',
      f'{vertex_tensorboard}',
      f'run_name={name}'
  ])
  return command


def generate_xpk_workload_cmd(
    cluster_config: XpkClusterConfig,
    wl_config: WorkloadConfig,
):
  """Generates a command to run a maxtext model on XPK."""

  is_pathways_enabled = wl_config.pathways_config is not None

  time.localtime()
  length_of_random_str = 3
  temp_post_fix = ''.join(
      random.choice(string.ascii_lowercase + string.digits) for _ in range(length_of_random_str)
  )

  truncate_model_name = 12
  truncate_prefix = 5
  common_post_fix = f"-{wl_config.num_slices}-{time.strftime('%m%d%H', time.localtime())}-{temp_post_fix}"
  common_prefix = os.environ['USER']
  pw_prefix = "pw-"

  if is_pathways_enabled:
    name = (
        f"{pw_prefix}{wl_config.model.model_name.replace('_', '-')[:truncate_model_name - len(pw_prefix)]}"
    )
  else:
    name = (
      f"{wl_config.model.model_name.replace('_', '-')[:truncate_model_name]}"
    )
  name = f"{common_prefix[:truncate_prefix]}-{name}{common_post_fix}"

  user_command = build_user_command(
      name=name,
      wl_config=wl_config
  )

  additional_flags = ''
  if not is_pathways_enabled and wl_config.libtpu_type == LibTpuType.CUSTOM:
    additional_flags = '--env="TPU_LIBRARY_PATH=/lib/libtpu.so"'

  docker_image_flag = ''
  # pathways-related flags
  pathways_specific_flags = ''
  if is_pathways_enabled:
    pw_config = wl_config.pathways_config
    pathways_specific_flags = (
        '--use-pathways'
        f' --server-image={pw_config.server_image}'
        f' --proxy-server-image={pw_config.proxy_image}'
        ' --termination-grace-period-seconds=300'
        f' --pathways-gcs-location={wl_config.base_output_directory}'
        f' --restart-on-user-code-failure'
        f' --debug-dump-gcs={wl_config.base_output_directory}'
    )
    docker_image_flag = (
        f'--docker-image={pw_config.runner_image}'
    )
  else:
    docker_image_flag = f'--base-docker-image="{wl_config.base_docker_image}"'


  print(f'User command: {user_command}')
  return (
      (
          f'python3 {wl_config.xpk_path}/xpk.py workload create'
          f' {pathways_specific_flags}'
          f' --cluster={cluster_config.cluster_name}'
          f' --project={cluster_config.project}'
          f' --zone={cluster_config.zone}'
          f' --device-type={cluster_config.device_type}'
          f' --num-slices={wl_config.num_slices}'
          f' --command="{user_command}"'
          f' {docker_image_flag}'
          ' --enable-debug-logs'
          f' --workload={name}'
          f' --priority={wl_config.priority}'
          f' --max-restarts={wl_config.max_restarts}'
          # ' --use-vertex-tensorboard'
          # f' --experiment-name={test_purpose_name}'
          f' {additional_flags}'
      ),
      name,
  )


def run_xpk_workload(
    cluster_config: XpkClusterConfig,
    wl_config: WorkloadConfig,
):
  """Runs a maxtext model on XPK.

  Args:
    model:
    cluster_config:

  Returns:
  """
  assert cluster_config.device_type == wl_config.device_type, f"The workload device size {wl_config.device_type}, and cluster device size {cluster_config.device_type} don't match."
  command, _ = generate_xpk_workload_cmd(
      cluster_config=cluster_config,
      wl_config=wl_config
  )
  return run_command_with_updates(command, 'Run XPK workload')


def xpk_benchmark_runner(
    cluster_config: XpkClusterConfig,
    workload_configs: list[WorkloadConfig],
):
  xpk_workload_names = []
  xpk_workload_cmds = []
  for wl_config in workload_configs:
    command, name = generate_xpk_workload_cmd(
      cluster_config=cluster_config,
      wl_config=wl_config
    )

    print(f"Name of the workload is: {name} \n")
    xpk_workload_names.append(name)

    print(f"XPK command to be used is: {command} \n")
    xpk_workload_cmds.append(command)

  # TODO(@vbarr) Support batch workloads.
  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    return_code = run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      print('Unable to run xpk workload: {xpk_workload_name}')


# Run maxtext_xpk_runner.py as a script for executing multiple workloads pythonically!
def main() -> int:
  # Variables to configure:
  output_bucket = 'gs://DIR'
  base_docker_image = _DEFAULT_MAXTEXT_BASE_DOCKER_IMAGE_NAME

  # Set up the clusters to run workloads on!
  v5e_cluster_config = XpkClusterConfig(
      cluster_name='v5e-256',
      project='my-cool-project',
      zone='us-central2-b',
      device_type='v5litepod-256',
  )

  v6e_cluster_config = XpkClusterConfig(
      cluster_name='v6e-256',
      project='my-cool-project',
      zone='us-central2-b',
      device_type='v6e-256',
  )

  xpk_workload_cmds = []
  xpk_workload_names = []

  list_of_models = [
    model_configs.llama2_70b_4096_sc,
    # model_configs.default_128
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

  user = os.environ['USER']
  base_output_dir = os.path.join(output_bucket,user)

  for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
      # v5e_cluster_config,
      # v6e_cluster_config,
      v6e_cluster_config_yucmhab,
      # another_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in [1,]:
        # Use the libtpu dependencies from:
        for libtpu_type in [
            # LibTpuType.CUSTOM
            LibTpuType.MAXTEXT
            # LibTpuType.NIGHTLY
        ]:
          wl_config = WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_dir,
            priority="medium",
            max_restarts=0,
            libtpu_type=libtpu_type,
            libtpu_nightly_version="",
            base_docker_image=base_docker_image,
            pathways_config=None
          )
          command, name = generate_xpk_workload_cmd(
            cluster_config=cluster_config,
            wl_config=wl_config
          )

          print(f"Name of the workload is: {name} \n")
          xpk_workload_names.append(name)

          print(f"XPK command to be used is: {command} \n")
          xpk_workload_cmds.append(command)

  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    return_code = run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      print('Unable to run xpk workload: {xpk_workload_name}')

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



if __name__ == '__main__':
  main()
