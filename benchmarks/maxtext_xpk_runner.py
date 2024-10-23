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
import dataclasses
import datetime
import enum
import random
import string
import subprocess
import sys
import tempfile
import time

import maxtext_trillium_model_configs as model_configs

# Assumes you built maxtext dep image.
# Assumes you locally downloaded the libtpu wheel.
BASE_DOCKER_IMAGE = 'maxtext_base_image'


class LibTpuType(enum.Enum):
  V6E_NIGHTLY_WHEEL = 'v6e-nightly-wheel'
  NIGHTLY = 'nightly-libtpu'
  CUSTOM = 'custom'
  MAXTEXT = 'maxtext-docker'


@dataclasses.dataclass
class XpkConfig:
  cluster_name: str
  project: str
  zone: str
  num_slices: str
  device_type: str
  base_output_directory: str


@dataclasses.dataclass
class HWConfig:
  num_slices: str
  device_type: str


@dataclasses.dataclass
class SWconfig:
  libtpu_version: str
  base_docker_image: str


@dataclasses.dataclass
class BenchmarkRunner:
  model_name: str
  hardware_config: HWConfig
  software_config: SWconfig


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
    model: model_configs.MaxTextModel,
    num_slices: int,
    num_steps: int,
    libtpu_type: LibTpuType,
    libtpu_date: str,
    cluster_config: XpkConfig,
    base_output_directory: str, 
    buffer_size: int,
):
  config_tuning_params = ''
  for key, value in model.tuning_params.items():
    config_tuning_params += f'{key}={value} '

  install_libtpu_cmd = ''
  if libtpu_type == LibTpuType.NIGHTLY:
    install_libtpu_cmd += (
        f' pip install libtpu-nightly==0.1.dev{libtpu_date} -f'
        ' https://storage.googleapis.com/libtpu-releases/index.html &&'
    )
  elif libtpu_type == LibTpuType.CUSTOM:
    install_libtpu_cmd += f' mv libtpu.so /lib/ &&'
  elif libtpu_type == LibTpuType.MAXTEXT:
    install_libtpu_cmd += ''
  # model.xla_flags += ' --megascale_verify_checksums=true'
  # Enable chaotic good.
  # model.xla_flags += ' --megascale_grpc_use_chaotic_good=true'
  # model.xla_flags += ' --megascale_grpc_use_event_engine_allocator=true'
  # model.xla_flags += ' --grpc_enable_tcp_recv_zerocopy=false'
  # model.xla_flags += ' --grpc_enable_rpc_receive_coalescing=true'
  # model.xla_flags += ' --grpc_experiments=tcp_rcv_lowat'

  libtpu_flags = f"LIBTPU_INIT_ARGS='{model.xla_flags}'"

  return (
      # f'python3 -m pip install google-cloud-aiplatform==v1.61.0 &&'
      # f'pip install -U "jax[tpu]==0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&'
      # f' pip install https://storage.googleapis.com/jax-releases/nightly/nocuda/jaxlib-0.4.27.dev20240501-cp310-cp310-manylinux2014_x86_64.whl &&'
      # f' pip install git+https://github.com/google/jax.git@57bfe81260545556ec22509347f7ced112496200 &&'
      f' {install_libtpu_cmd}'
      # f' mv libtpu.so /lib/ &&'
      # f' export TPU_LIBRARY_PATH=$PWD/libtpu.so &&'
      f' echo {libtpu_flags} &&'
      # f' echo {model.tuning_params["sa_block_q"]}-q-dq-{model.tuning_params["sa_block_q_dq"]}-q-dkv-{model.tuning_params["sa_block_q_dkv"]} &&'
      # f' echo {model.tuning_params["ici_fsdp_parallelism"]} {model.tuning_params["ici_tensor_parallelism"]} &&'
      f' export JAX_PLATFORMS=tpu,cpu &&'
      # f' export JAX_DEBUG_NANS=True &&'
      # f' export TPU_MEGACORE=megachip_tccontrol &&'
      # f' echo TPU MEGACORE: $TPU_MEGACORE &&'
      f' export TPU_PREMAPPED_BUFFER_SIZE={buffer_size} &&'
      f' echo {buffer_size} &&'
      f' export ENABLE_PJRT_COMPATIBILITY=true &&'
      f' export {libtpu_flags} && '
      ' python3 MaxText/train.py MaxText/configs/base.yml'
      f' {config_tuning_params} steps={num_steps} enable_checkpointing=false'
      f' model_name={model.model_type}'
      f' base_output_directory={base_output_directory}'
      f' use_vertex_tensorboard=false'
      ' vertex_tensorboard_project="" vertex_tensorboard_region=""'
      f' run_name="{model.model_name}-{num_slices}-{libtpu_date}"'
  )


def generate_xpk_workload_cmd(
    model: model_configs.MaxTextModel,
    cluster_config: XpkConfig,
    num_slices: int,
    libtpu_type: LibTpuType,
    libtpu_version: str,
    base_output_directory: str,
    buffer_size: int,
):
  """Generates a command to run a maxstar model on XPK."""
  num_steps = 20
  time.localtime()
  test_purpose_name = f'maxstar-benchmarks-{model.model_name}-{libtpu_version}'
  N = 3
  temp_post_fix = ''.join(
      random.choice(string.ascii_lowercase + string.digits) for _ in range(N)
  )

  name = (
      f"{model.model_name.replace('_', '-')}-{cluster_config.num_slices}-{time.strftime('%m%d%H', time.localtime())}-{temp_post_fix}"
  )
  user_command = build_user_command(
      model,
      num_slices,
      num_steps,
      libtpu_type,
      libtpu_version,
      cluster_config,
      base_output_directory,
      buffer_size,
  )

  additional_flags = ''
  if libtpu_type == LibTpuType.CUSTOM:
    additional_flags = '--env="TPU_LIBRARY_PATH=/lib/libtpu.so"'
  # additional_flags = '--env="TPU_MEGACORE=megachip_tccontrol"'

  perf_optimzation_dcn = (
      'kubectl apply -f'
      ' https://raw.githubusercontent.com/GoogleCloudPlatform/ai-on-gke/9ff340f07f70be0130454f9e7238551587242b75/scripts/network-setup/v6e-network-optimization.yaml'
  )

  print(f'User command: {user_command}')
  return (
      (
          # f'{perf_optimzation_dcn} &&'
          'python3 ~/xpk/xpk.py workload create'
          f' --cluster={cluster_config.cluster_name}'
          f' --project={cluster_config.project}'
          f' --zone={cluster_config.zone}'
          f' --device-type={cluster_config.device_type}'
          f' --num-slices={cluster_config.num_slices}'
          f' --command="{user_command}"'
          f' --base-docker-image="{BASE_DOCKER_IMAGE}"'
          ' --enable-debug-logs'
          f' --workload={name}'
          ' --priority=medium'
          # ' --use-vertex-tensorboard'
          # f' --experiment-name={test_purpose_name}'
          f' {additional_flags}'
      ),
      name,
  )


def run_xpk_workload(
    model: model_configs.MaxTextModel,
    cluster_config: XpkConfig,
    num_slices: int,
    libtpu_type: LibTpuType,
    libtpu_version: str,
    buffer_size: int,
):
  """Runs a maxstar model on XPK.

  Args:
    model:
    cluster_config:

  Returns:
  """
  command, _ = generate_xpk_workload_cmd(
      model, cluster_config, num_slices, libtpu_type, libtpu_version, base_output_directory=cluster_config.base_output_directory, buffer_size=buffer_size
  )
  return run_command_with_updates(command, 'Run XPK workload', cluster_config)


def xpk_benchmark_runner(cluster_config: XpkConfig, benchmarks: list[BenchmarkRunner]):
  xpk_workload_names = []
  xpk_workload_cmds = []
  for benchmark in benchmarks:
    command, name = generate_xpk_workload_cmd(
        model=benchmark.model_name,
        cluster_config=cluster_config,
        num_slices=benchmark.hardware_config.num_slices,
        libtpu_type=LibTpuType.NIGHTLY,
        libtpu_version=benchmark.software_config.libtpu_version,
        base_output_directory=cluster_config.base_output_directory,
        buffer_size=4294967296,
    )
    xpk_workload_names.append(name)
    xpk_workload_cmds.append(command)

  returncodes = run_commands(
      xpk_workload_cmds,
      'Run XPK workloads',
      xpk_workload_names,
      batch=1,
      dry_run=False,
  )
  print(f'Returncodes: {returncodes}')
