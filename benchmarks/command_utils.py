# Copyright 2023–2026 Google LLC
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

"""This file contains utility functions for running shell commands.

It provides helpers for:
  - Running a single command with real-time output streaming.
  - Running multiple commands in parallel batches.
  - Creating temporary log files for command outputs.
"""

import datetime
import subprocess
import sys
import tempfile
import time


def chunks(lst: list, n: int):
  """Return a list of n-sized chunks from lst.

  Args:
    lst: input list to get chunks from.
    n: size of each chunk.

  Returns:
    list of n-sized chunks for lst.
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
      tempfile.NamedTemporaryFile(delete=False, prefix=command.replace(" ", "-") + "-") for command in per_command_name
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

  print(f"Breaking up a total of {len(commands)} commands into" f" {len(commands_batched)} batches")
  if dry_run:
    print("Pretending all the jobs succeeded")
    return 0

  max_return_code = 0
  for i, _ in enumerate(commands_batched):
    print(f"Dispatching batch {i}/{len(commands_batched)}")
    batch_max_return_code, _ = run_command_batch(
        commands_batched[i],
        jobname,
        per_command_name_batches[i],
        temporary_files_batches[i],
    )
    max_return_code = max(max_return_code, batch_max_return_code)
    if max_return_code > 0:
      break
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
        subprocess.Popen(command, stdout=output_logs[i], stderr=output_logs[i], shell=True)
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
      slow_str = f", task {slow_worker_text} still working, logfile" f" {output_logs[slow_worker_index].name}"
    else:
      slow_str = ""
    print(f"[t={seconds_elapsed:.2f}, {jobname}] Completed" f" {completed}/{total}{slow_str}")
    if max_returncode > 0:
      failing_index = [i for i, x in enumerate(returncodes) if x is not None and x > 0][0]
      print(f"Terminating all {jobname} processes since at least one failed.")
      print(f"Failure is {per_command_name[failing_index]}" f" and logfile {output_logs[failing_index].name}")
      for child in children:
        child.terminate()
      break

    if completed == total:
      break

    # Sleep for 1 second before polling processes again
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
    print(f"Task: `{task}` is implemented by `{command}`, streaming output live.")
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
          print(f"Waiting for `{task}`, for {i} seconds")
          time.sleep(1)
          i += 1
        else:
          print(f"Task: `{task}` terminated with code `{return_code}`")
          return return_code
  else:
    print(f"Task: `{task}` is implemented by `{command}`, hiding output unless" " there is an error.")
    try:
      subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print(f"Task: `{task}` terminated with ERROR `{e.returncode}`, printing" " logs")
      print("*" * 80)
      print(e.output)
      print("*" * 80)
      return e.returncode
    print(f"Task: `{task}` succeeded.")
    return 0
