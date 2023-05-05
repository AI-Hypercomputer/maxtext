"""
 Copyright 2023 Google LLC

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

# pylint: disable=consider-using-with
""" Script to run a command in a multislice/multihost environment

The "runner" host (the one which runs this script) and the "worker" hosts (the TPUS found by --TPU_PREFIX and
where the --COMMAND is run) should be different. You can use either a TPUVM or a non-TPUVM runner host,
but in the former case this host needs to be in the same project as the worker hosts, but not be one of the workers.

Example usages:
  Assuming runner.sh lives in directory path/to/dir:
  python3 multihost_runner.py --TPU_PREFIX=mytpu-name --COMMAND="bash runner.sh" --SCRIPT_DIR=path/to/dir
    This will recursively copy all of the files in path/to/dir to each tpu and run runner.sh

Common issues:
  Make sure your gcloud zone in set with e.g.
    gcloud config set compute/zone us-central2-b
    gcloud config set project <project_name>
    before running this script.

  You may have to create/authorize ssh-keys when first sshing into the TPUs.
  For this purpose you may need to first run:
    ssh-keygen -f ~/.ssh/google_compute_engine
"""

import argparse
import sys
from collections import namedtuple
import subprocess
import time
from datetime import datetime
import os
import re

##### Define flags #####
parser = argparse.ArgumentParser(description='TPU configuration options')
parser.add_argument('--TPU_PREFIX', type=str, default=None, required=True,
                    help="Prefix of worker TPU's. E.g. if TPU's are named user-0 and user-1, \
                          TPU_PREFIX should be set as user")
parser.add_argument('--SCRIPT_DIR', type=str, default=os.getcwd(),
                    help="The local location of the directory to copy to the TPUs and run the main command from. \
                          Defaults to current working directory.")
parser.add_argument('--COMMAND', type=str, default=None, required=True,
                    help="Main command to run on each TPU. \
                          This command is run from a copied version of SCRIPT_DIR on each TPU worker.")
parser.add_argument('--RUN_NAME', type=str, default=None,
                    help="Name for the code directory on the TPU")
parser.add_argument('--USE_EXISTING_FOLDER', type=str, default="False",
                    help='If true, use the existing code directory on the TPU')
parser.add_argument('--INTERNAL_IP', type=str, default="False",
                    help="Set true if running script locally from a TPU or GCE instance, false otherwise.")
args = parser.parse_args()
args.USE_EXISTING_FOLDER = args.USE_EXISTING_FOLDER.lower() == "true"
args.INTERNAL_IP = args.INTERNAL_IP.lower() == "true"

if not args.TPU_PREFIX:
  raise ValueError("--TPU_PREFIX must be a non-empty string specifying your TPU slice names.")

if args.USE_EXISTING_FOLDER is True and not args.RUN_NAME:
  raise ValueError("When USE_EXISTING_FOLDER is true, RUN_NAME must be specified.")

Slice = namedtuple('Slice', ['name', 'slice_num', 'num_workers', 'version'])

def get_slices(tpu_prefix):
  """ Returns a list of slices matching tpu_prefix """
  command = [
      "gcloud", "alpha", "compute", "tpus", "tpu-vm", "list",
      f"--filter=name~{tpu_prefix}", "--format=csv(name,TYPE)"
  ]
  completed_command = subprocess.run(command, capture_output=True, check=True)
  instances = completed_command.stdout.decode()
  instance_list = instances.strip().split('\n')
  instance_list = filter_instances(instance_list[1:], tpu_prefix) # First row is headers
  num_slices = len(instance_list)
  slices = [None for _ in range(num_slices)]

  if num_slices > 0:
    print(f"{num_slices} slices found.", flush=True)
  else:
    print(f"No TPUs found with name {tpu_prefix} or matching regex {tpu_prefix}-[0-9]+")
    return []

  slice_names = [instance.split(',')[0] for instance in instance_list]
  slice_versions = [instance.split(',')[1] for instance in instance_list]
  # Get number of workers in any slice (assume same worker count for all slices.)
  command = [
      "gcloud", "compute", "tpus", "describe", slice_names[0],
      "--flatten=networkEndpoints[]", "--format=csv[no-heading](networkEndpoints.ipAddress)"
  ]
  completed_command = subprocess.run(command, capture_output=True, check=True)
  num_workers = len(completed_command.stdout.decode().strip().split('\n'))

  for slice_name, version in zip(slice_names, slice_versions):
    if num_slices > 1:
      slice_num = int(slice_name.split('-')[-1])
    else:
      slice_num = 0
    slices[slice_num] = Slice(slice_name, slice_num, num_workers, version)
  return slices

def filter_instances(instance_list, tpu_prefix):
  # First look for exact match with tpu_prefix
  for instance in instance_list:
    if instance.split(',')[0] == tpu_prefix:
      return [instance]

  # If no exact match, reg-exp full match "<tpu_prefx>-[0-9]+"
  re_pattern = tpu_prefix + "-[0-9]+"
  return [instance for instance in instance_list if re.fullmatch(re_pattern, instance.split(',')[0])]

def get_run_name():
  now = datetime.now()
  return now.strftime("%Y-%m-%d-%H-%M-%S")

def write_kill_script(script_dir, kill_processes_script_name):
  kill_processes_script = os.path.join(script_dir, kill_processes_script_name)
  with open(kill_processes_script, "w", encoding="utf-8") as f:
    f.write(kill_existing_processes_str())

def kill_existing_processes_str():
  return """#!/bin/bash
_TPU_VERSION_NAME=${1}
device_name="accel0"
if [[ "${_TPU_VERSION_NAME}" =~ ^v5.* ]]; then
  device_name="vfio/0"
fi
echo -e "Searching for existing processes on device ${device_name}..."
pid=$(sudo lsof -w /dev/${device_name} | awk 'END{print $2}')
if [[ ! -z "${pid}" ]]
then
 echo -e "Existing process found with pid ${pid}"
 echo -e "Killing process ${pid}..."
 kill -9 "${pid}"
 tail --pid=$pid -f /dev/null
 echo -e "Orphaned process ${pid} on your TPU was killed successfully."
else
 echo -e "No existing processes found."
fi
sudo rm -f /tmp/libtpu_lockfile"""

def scps(script_dir, slices, run_name_dir, zip_name, internal_ip):
  """ Zip the script directory, scp it to the TPUs, and unzip it there. """
  original_working_directory = os.getcwd()
  os.chdir(script_dir) # To tar script_dir, it is most convenient to cd there.

  # Zip script directory
  # Save the zip both to the logging directory, and the script directory.
  # It will be removed from the script directory after the transfer to the TPUs
  os.makedirs(run_name_dir, exist_ok=True)
  zip_path = os.path.join(run_name_dir, zip_name)
  command = ["tar","--exclude=tmp", "-czf", zip_path, "./"]
  subprocess.run(command, check=True)

  # Move zip file to each tpuvm worker
  commands = []
  worker_list = []
  for cur_slice in slices:
    for worker_num in range(cur_slice.num_workers):
      command = [
          "gcloud", "compute", "tpus", "tpu-vm", "scp", f"--worker={worker_num}", zip_path,
          f"{cur_slice.name}:~/", "--strict-host-key-checking=no"
      ]
      if internal_ip:
        command.append("--internal-ip")
      commands.append(command)
      worker_list.append([cur_slice.slice_num, worker_num])
  return_code, _ = run_commands(commands, 0, "SCP", worker_list)
  if return_code != 0:
    print("Failed to scp zipped code directory with error code ", return_code)
    return return_code

  # Cleanup
  os.chdir(original_working_directory)

  return return_code

def execute_main_command(main_command,slices, local_log_dir, run_name, zip_name, internal_ip, use_existing_folder):
  """ Run the main command on each worker, logging each separately. """
  kill_script_name = "kill_existing_processes.sh" # File written on worker machines
  commands = []
  output_logs = []
  worker_list = []
  os.makedirs(local_log_dir, exist_ok=True)

  for slice_num, cur_slice  in enumerate(slices):
    for worker_num in range(cur_slice.num_workers):
      output_filename = os.path.join(local_log_dir, f"output_slice_{cur_slice.slice_num:04d}_worker_{worker_num:04d}.txt")
      output_logs.append(output_filename)
      mkdir_command = f"mkdir -p {run_name}"
      mv_zip_command = f"mv {zip_name} {run_name}"
      cd_command = f"cd {run_name}"
      unzip_command = f"tar xzf {zip_name}"
      write_kill_script_command = f"echo {kill_existing_processes_str()} > {kill_script_name}"
      kill_existing_command = f"bash {kill_script_name} {cur_slice.version}"

      if use_existing_folder is False:
        remote_command_list = [mkdir_command , mv_zip_command , cd_command , unzip_command ,
                        write_kill_script_command , kill_existing_command , main_command]
      else:
        remote_command_list = [cd_command, write_kill_script_command , kill_existing_command , main_command]
      remote_command_list_str = " && ".join(remote_command_list)
      gcloud_command=[
          "gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", cur_slice.name, f"--worker={worker_num}",
          "--command", remote_command_list_str, "--strict-host-key-checking=no"]
      if internal_ip:
        gcloud_command.append("--internal-ip")
      commands.append(gcloud_command)
      worker_list.append([slice_num, worker_num])

  return_code, return_codes = run_commands(commands, 0, "MAIN COMMAND", worker_list, output_logs=output_logs)
  if return_code > 0:
    failure_index = next((i for i, x in enumerate(return_codes) if x), None)
    print(f"Main command failed on slice {worker_list[failure_index][0]} worker"\
        f" {worker_list[failure_index][1]} with error code {return_codes[failure_index]}, see logs for details", flush=True)
  return return_code

def run_commands(commands, id_to_print, jobname, worker_list, is_shell=False, output_logs=None, fail_fast=True):
  ''' Runs commands in parallel.
  Inputs:
     commands: list of n commands, each command is a a list of strings
     id_to_print: which command is printed to the terminal, typically 0 or None
     jobname: Useful debugging name for the group of commands, such as SCP
     worker_list: list of n pairs of (slice_id, worker_id)
     is_shell: Boolean directly passed as shell argument to subprocess.Popen
     output_logs: list of n log paths, each command will output to each log.
     fail_fast: If true, when one command fails immediately terminate others
  '''

  children = []
  start_time = datetime.now()
  for i, command in enumerate(commands):
    if output_logs and i == id_to_print:
      persistent_log = open(output_logs[i], "w", encoding="utf-8")
      output_log = Tee(sys.stdout, persistent_log)
    elif output_logs:
      output_log = open(output_logs[i], "w", encoding="utf-8")
    elif i == id_to_print:
      output_log = None
    else:
      output_log = subprocess.DEVNULL

    children.append(subprocess.Popen(command, stdout=output_log, stderr=output_log, shell=is_shell))

  while True:
    returncodes = [child.poll() for child in children]
    max_returncode = max([0]+[r for r in returncodes if r is not None])
    completed = len([r for r in returncodes if r is not None])
    total = len(returncodes)
    seconds_elapsed = (datetime.now() - start_time).total_seconds()
    if completed < total:
      slow_worker_index = returncodes.index(None)
      slow_worker = worker_list[slow_worker_index]
      slow_str = f", slice {slow_worker[0]} worker {slow_worker[1]} still working"
    else:
      slow_str = ""
    print(f"[t={seconds_elapsed:.2f}, {jobname}] Completed {completed}/{total}{slow_str}...")

    if seconds_elapsed >= 60 and not 0 in returncodes and jobname == "SCP":
      print("SCP operation timed out - terminating all processes."\
        " Please check that --INTERNAL_IP flag is set correctly.")
      for child in children:
        child.terminate()
      max_returncode = 255
      break

    if fail_fast and max_returncode > 0:
      print(f"Terminating all {jobname} processes since at least one failed.")
      for child in children:
        child.terminate()
      break

    if completed == total:
      break

    time.sleep(1)
  return max_returncode, returncodes

def assert_project_and_zone_set():
  """ Asserts that project and zone are set in gcloud config, erroring out if not. """
  # Project
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(project_outputs) < 1:
    sys.exit("You must set the project by running 'gcloud compute set project <project>'")

  # zone
  completed_command = subprocess.run(["gcloud", "config", "get", "compute/zone"], check=True, capture_output=True)
  zone_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(zone_outputs) < 1:
    sys.exit("You must set the zone by running 'gcloud compute set compute/zone <zone>'")

def assert_script_dir_exists(script_dir):
  if not os.path.isdir(script_dir):
    sys.exit(f"No directory named {script_dir} found.")

class Tee:
  """ Helper class to print subprocess to both stdout and a log file. """
  def __init__(self, *files, bufsize=1):
    files = [x.fileno() if hasattr(x, 'fileno') else x for x in files]
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid:
      os.close(read_fd)
      self._fileno = write_fd
      self.child_pid = pid
      return
    os.close(write_fd)
    while buf := os.read(read_fd, bufsize):
      for f in files:
        os.write(f, buf)
    os._exit(0)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def fileno(self):
    return self._fileno

  def close(self):
    os.close(self._fileno)
    os.waitpid(self.child_pid, 0)

################### Main ###################
def main() -> None:
  print("Starting multihost runner...", flush=True)

  #### Parse flags ####
  tpu_prefix = args.TPU_PREFIX
  script_dir = args.SCRIPT_DIR
  main_command = args.COMMAND
  run_name = args.RUN_NAME
  use_existing_folder = args.USE_EXISTING_FOLDER
  internal_ip = args.INTERNAL_IP

  assert_project_and_zone_set()
  assert_script_dir_exists(script_dir)

  ##### Step 1: Get the workers #####
  slices = get_slices(tpu_prefix)
  if not slices:
    print(f"Failed to retrieve slices with name prefix {tpu_prefix}", flush=True)
    return 1

  run_name = run_name or get_run_name() # Used for local logging files and remote directory.
  local_log_dir = os.path.join("/tmp", run_name, "")
  zip_name = "script_dir_zip_" + run_name + ".tar.gz"

  if use_existing_folder is False:
    ##### Step 2 when using a new folder: Zip code and move it to the TPUs #####
    return_code = scps(script_dir, slices, local_log_dir, zip_name, internal_ip)
    if return_code > 0:
      print(f"Moving the directory {script_dir} to the VMs failed with error code {return_code}")
      return return_code

  ##### Step 3: Unzip if using a new folder, kill existing processes, and run #####
  print(f"Running main command, logs located in: {local_log_dir}", flush=True)
  return_code = execute_main_command(main_command, slices, local_log_dir, run_name, zip_name,
                                     internal_ip, use_existing_folder)
  if return_code == 0:
    print(f"Main command completed successfully, logs located in: {local_log_dir}", flush=True)
    print("Multihost runner finished successfully!", flush=True)
    return 0
  else:
    print(f"Main command finished with errors, check the logs located in: {local_log_dir}", flush=True)
    return return_code

if __name__ == '__main__':
  main()
