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

""" Script to run a productionized job in a multislice/multihost environment
                          ***** IMPORTANT *****
This script provisions new TPUs! You cannot run jobs on existing TPUs with this script,
we recommend trying multihost_runner instead for this purpose. In addition you must include all
of your installation commands inside of the --COMMAND arg, e.g. --COMMAND="bash setup.sh && python3 train.py"

This script:
  1) Creates specified TPU(s)
  2) Loads your scripts onto each TPU
  3) Runs your job on each TPU
  4) Logs the output of the job in real time to cloud logging
  5) Stores a copy of the logs and zipped code in GCS at the job's end
  6) Deletes the TPUs and QR
Example usages:
  Minimal customization, assuming runner.sh lives in present working directory:
  python3 multihost_job.py  --COMMAND="bash runner.sh" --BUCKET_NAME=<my-bucket>

  Maximal customization, assuming runner.sh lives in path/to/dir
  python3 multihost_job.py  --COMMAND="bash runner.sh" --BUCKET_NAME=<my-bucket> --BUCKET_DIR=job-log-directory \
    --SCRIPT_DIR=path/to/dir --PROJECT=<my-project> --ZONE=<zone> \
    --VERSION=tpu-ubuntu2204-base --TPU_TYPE=v4-8 --NUM_SLICES=2 --RUN_NAME=$USER-run-job
Common issues:
  You must have local write permissions to BUCKET_NAME. The allocated TPUVMs must also have read permissions to this
  bucket, which is granted through service account roles, such as Storage Object Admin.
"""
import argparse
import sys
import subprocess
from datetime import datetime
import os
import shutil



def get_project():
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(project_outputs) < 1 or project_outputs[-1]=='':
    sys.exit("You must specify the project in the PROJECT flag or set it with 'gcloud config set project <project>'")
  return project_outputs[-1] # The project name lives on the last line of the output

def get_zone():
  completed_command = subprocess.run(["gcloud", "config", "get", "compute/zone"], check=True, capture_output=True)
  zone_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(zone_outputs) < 1 or zone_outputs[-1]=='':
    sys.exit("You must specify the zone in the ZONE flag or set it with 'gcloud config set compute/zone <zone>'")
  return zone_outputs[-1] # The zone name lives on the last line of the output

def get_run_name():
  now = datetime.now()
  return os.getlogin() + "-" + now.strftime("%Y-%m-%d-%H-%M-%S")

def normalize_gcs_bucket_name(args):
  """ Remove the gs:// from bucket_name if passed."""
  if len(args.BUCKET_NAME) > 5 and args.BUCKET_NAME[0:5]=="gs://":
    args.BUCKET_NAME=args.BUCKET_NAME[5:]
  return args

def print_flags(args):
  """ Print configuration values after defaults have been filled in. """
  print("Running multihost_job with the following configuration:")
  print(f"Project               (--PROJECT)               = {args.PROJECT}")
  print(f"Zone                  (--ZONE)                  = {args.ZONE}")
  print(f"TPU type              (--TPU_TYPE)              = {args.TPU_TYPE}")
  print(f"TPU runtime version   (--VERSION)               = {args.VERSION}")
  print(f"Number of slices      (--NUM_SLICES)            = {args.NUM_SLICES}")
  print(f"Script dir            (--SCRIPT_DIR)            = {args.SCRIPT_DIR}")
  print(f"Bucket name           (--BUCKET_NAME)           = {args.BUCKET_NAME}")
  print(f"Bucket dir            (--BUCKET_DIR)            = {args.BUCKET_DIR}")
  print(f"Run name              (--RUN_NAME)              = {args.RUN_NAME}")
  print(f"Extra CQR args        (--CQR_EXTRA_ARGS)        = {args.CQR_EXTRA_ARGS}")
  print(f"Command to run        (--COMMAND)               = {args.COMMAND}")
  print(f"Enable Autocheckpoint (--ENABLE_AUTOCHECKPOINT) = {args.ENABLE_AUTOCHECKPOINT}\n")

def move_script_dir_to_gcs(script_dir, tmp_dir, zip_name, bucket_path):
  """ Zip the script directory, cp it to GCS """
  original_working_directory = os.getcwd()
  os.chdir(script_dir) # To tar script_dir, it is most convenient to cd there.

  # Zip script directory, storing it in the logging directory.
  os.makedirs(tmp_dir, exist_ok=True)
  zip_path = os.path.join(tmp_dir, zip_name)
  command = ["tar", "--exclude=tmp", "-czf", zip_path, "./"]
  subprocess.run(command, check=True)

  # Move zip file to GCS
  zip_in_gcs_path = os.path.join(bucket_path, zip_name)
  command = ["gsutil", "mv", zip_path, zip_in_gcs_path]
  captured_output = subprocess.run(command, check=True, capture_output=True)

  # Cleanup
  os.chdir(original_working_directory)

  return captured_output

def run_create_resources(startup_script_file, args):
  """ Run the Create Queued Resources (CQR) request """
  # pylint: disable=line-too-long
  command = fr'gcloud alpha compute tpus queued-resources create {args.RUN_NAME} --accelerator-type={args.TPU_TYPE} --runtime-version={args.VERSION} --project={args.PROJECT} --zone={args.ZONE}'
  if args.NUM_SLICES > 1:
    command = command + f' --node-prefix={args.RUN_NAME} --node-count={args.NUM_SLICES}'
  else:
    command = command + f' --node-id={args.RUN_NAME}'

  if args.CQR_EXTRA_ARGS:
    command = command + ' ' + args.CQR_EXTRA_ARGS

  if args.ENABLE_AUTOCHECKPOINT:
    command = command + ' --autocheckpoint-enabled'

  command = command + f' --metadata-from-file=startup-script={startup_script_file}'

  captured_output = subprocess.run(command, check=False, shell=True, capture_output=True)
  return captured_output

def write_startup_script(zip_gcs_path, zip_name, log_name, bucket_path, startup_script_file, args):
  """ Write the startup script locally into a file to be passed to the CQR command. """
  startup_script = f"""#!/bin/bash
mkdir -p {args.RUN_NAME}
cd {args.RUN_NAME}
{get_env_command_str(args.NUM_SLICES)}
{setup_ops_str(args.RUN_NAME, log_name)}
sudo python3 -m virtualenv venv
source venv/bin/activate
ulimit -n 100000
(({download_from_gcs(zip_gcs_path)}
tar xzf {zip_name}
{args.COMMAND}) 2>&1) >> {log_name}
(echo "{finish_status_str()}") >> {log_name}
gsutil cp {log_name} "{bucket_path}/"
(({create_kill_command_str(args)}) 2>&1 ) >> {log_name}"""

  with open(startup_script_file, "w", encoding="utf-8") as f:
    f.write(startup_script)
  return startup_script

def get_env_command_str(num_slices):
  """ Define environment variables on the TPUS """
  # pylint: disable=line-too-long
  env_str = """curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env # store the metadata
NODE_ID=$(grep '^NODE_ID' /tmp/tpu-env | cut -d "'" -f 2)
WORKER_ID=$(grep '^WORKER_ID' /tmp/tpu-env | cut -d "'" -f 2)
ZONE=$(grep '^ZONE' /tmp/tpu-env | cut -d "'" -f 2)
PROJECT=$(grep '^CONSUMER_PROJECT_ID' /tmp/tpu-env | cut -d "'" -f 2)"""
  if num_slices == 1:
    slice_assignment = "SLICE_ID=0"
  else:
    slice_assignment = """SLICE_ID=$(grep '^MEGASCALE_SLICE_ID' /tmp/tpu-env | cut -d "'" -f 2)"""
  return env_str + "\n" + slice_assignment

def finish_status_str():
  # pylint: disable=line-too-long
  return """multihost_job finished main command on slice $SLICE_ID worker $WORKER_ID at $(date "+%Y-%m-%d %H:%M:%S") UTC with exit status $?.
This worker will immediately send its logs to GCS."""

def create_kill_command_str(args):
  # pylint: disable=line-too-long
  return f"""if [[ $SLICE_ID -eq 0 && $WORKER_ID -eq 0 ]]; then
  echo "This worker (slice 0 worker 0) will wait 10 minutes before tearing down the job to allow other workers to gracefully exit."
  sleep 600
  gcloud alpha compute tpus queued-resources delete {args.RUN_NAME} --force --quiet --project={args.PROJECT} --zone={args.ZONE}
  fi"""

def download_from_gcs(zip_gcs_path):
  return f"""
    echo "{write_download_from_gcs_sh(zip_gcs_path)}" > download_from_gcs.sh
    bash download_from_gcs.sh
  """

def write_download_from_gcs_sh(zip_gcs_path):
  # pylint: disable=anomalous-backslash-in-string
  return f"""GCS_READ_SUCCESS=0
while [ \$GCS_READ_SUCCESS -eq 0 ]
do
  {{ # try
      gsutil cp {zip_gcs_path} . &&
      echo 'Code download from GCS successful!' && GCS_READ_SUCCESS=1
  }} || {{ # catch
      echo 'Failed to read GCS via gsutil, trying again'
      sleep 10
  }}
done"""

def setup_ops_str(run_name, log_name):
  return f"""
    echo "{install_ops_script_str(run_name, log_name)}" > install_ops_wait_dpkg.sh
    bash install_ops_wait_dpkg.sh &
  """

def install_ops_script_str(run_name, log_name):
  # pylint: disable=anomalous-backslash-in-string
  return f"""OPS_FILE=/etc/google-cloud-ops-agent/config.yaml
  if ! test -f \$OPS_FILE;
  then
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    downloaded=0
    while [ \$downloaded -eq 0 ]
    do
      pid=\$(sudo lsof /var/lib/dpkg/lock-frontend | awk \\"END{{print \$2}}\\")
      if [[ ! -z \"\${{pid}}\" ]]
      then
        sleep 10
      else
        sudo bash add-google-cloud-ops-agent-repo.sh --also-install
        downloaded=1
      fi
    done
  fi
  sudo chmod 777 /etc/google-cloud-ops-agent/config.yaml
  sudo echo \\"{create_ops_config_str(run_name, log_name)}\\" >> /etc/google-cloud-ops-agent/config.yaml
  sudo service google-cloud-ops-agent restart
"""

def create_ops_config_str(run_name, log_name):
  return f"""logging:
  receivers:
    {run_name}_log:
      type: files
      include_paths:
      - /{run_name}/{log_name}
      record_log_file_path: true
  service:
    pipelines:
      default_pipeline:
        receivers: [{run_name}_log]"""

def google_cloud_logging_url(run_name, project):
  # pylint: disable=line-too-long
  return f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22gce_instance%22%20AND%0Alog_id%2528%22{run_name}_log%22%2529;?project={project}"


def google_cloud_logging_single_host_url(run_name, project):
  # pylint: disable=line-too-long
  return f"https://console.cloud.google.com/logs/query;query=resource.type%3D%22gce_instance%22%20AND%0Alog_id%2528%22{run_name}_log%22%2529%20AND%0Alabels.%22agent.googleapis.com%2Flog_file_path%22%3D%20%22%2F{run_name}%2Fmain_command_log_slice_0_worker_0%22;?project={project}"

def gcs_bucket_url(bucket_name, bucket_dir, project):
  bucket_path = os.path.join(bucket_name, bucket_dir)
  return f"https://console.cloud.google.com/storage/browser/{bucket_path}?project={project}"

################### Main ###################
def main(raw_args=None) -> None:
    ##### Define flags #####
  parser = argparse.ArgumentParser(description='TPU configuration options')
  parser.add_argument('--TPU_TYPE', type=str, default='v4-8',
                      help='The type of the TPU')
  parser.add_argument('--VERSION', type=str, default='tpu-ubuntu2204-base',
                      help='The runtime version of the TPU')
  parser.add_argument('--NUM_SLICES', type=int, default=2,
                      help='The number of slices to run the job on')
  parser.add_argument('--SCRIPT_DIR', type=str, default=os.getcwd(),
                      help='The local location of the directory to copy to the TPUs and run the main command from. \
                        Defaults to current working directory.')
  parser.add_argument('--COMMAND', type=str, default=None, required=True,
                      help='Main command to run on each TPU. \
                        This command is run from a copied version of SCRIPT_DIR on each TPU worker. \
                        You must include your dependency installations here, \
                        e.g. --COMMAND=\'bash setup.sh && python3 train.py\'')
  parser.add_argument('--BUCKET_NAME', type=str, default=None, required=True,
                      help='Name of GCS bucket, e.g. my-bucket')
  parser.add_argument('--BUCKET_DIR', type=str, default="",
                      help='Directory within the GCS bucket, can be None, e.g. my-dir')
  parser.add_argument('--PROJECT', type=str, default=None,
                      help='GCE project name, defaults to gcloud config project')
  parser.add_argument('--ZONE', type=str, default=None,
                      help='GCE zone, e.g. us-central2-b, defaults to gcloud config compute/zone')
  parser.add_argument('--RUN_NAME', type=str, default=None,
                      help='Run name used for temporary files, defaults to timestamp.')
  parser.add_argument('--CQR_EXTRA_ARGS', type=str, default=None,
                      help='Additional arguments to be passed verbatim to the CQR request, e.g. \
                      --CQR_EXTRA_ARGS="--reserved --service-account=my-service-account-email-address')
  parser.add_argument('--ENABLE_AUTOCHECKPOINT', type=bool, default=False,
                      help='Whether to enable the Autocheckpoint feature')
  args = parser.parse_args(raw_args)


  print("\nStarting multihost_job...\n", flush=True)

  #### Parse flags ####
  if not args.PROJECT:
    args.PROJECT = get_project()
  if not args.ZONE:
    args.ZONE = get_zone()
  if not args.RUN_NAME:
    args.RUN_NAME = get_run_name() # Used for QR name, TPU_PREFIX, logging file, and tmp json file.
  args = normalize_gcs_bucket_name(args)

  print_flags(args)

  ##### Step 1: Zip code and move it to GCS #####
  tmp_dir_relative_to_script = os.path.join("tmp", args.RUN_NAME, "")
  tmp_dir = os.path.join(args.SCRIPT_DIR, tmp_dir_relative_to_script)
  zip_name = "script_dir_zip_" + args.RUN_NAME + ".tar.gz"
  bucket_dir = os.path.join(args.BUCKET_DIR, args.RUN_NAME)
  bucket_path = os.path.join(f"gs://{args.BUCKET_NAME}", bucket_dir)
  startup_script_file = os.path.join(tmp_dir, "startup_script.txt")

  print(f"Moving {args.SCRIPT_DIR} to {bucket_path}...")
  captured_output = move_script_dir_to_gcs(args.SCRIPT_DIR, tmp_dir_relative_to_script, zip_name, bucket_path)
  if captured_output.returncode != 0:
    print("\n\n Moving code to GCS failed")
    print(f"Running 'gsutil mv zip {bucket_path}' failed with error: ")
    print(captured_output.stderr.decode())
    print("\nYou may need to run 'gcloud auth login'")
    return -1
  print("Move successful!\n")

  #### Step 2: Run the CQR command ####
  log_name = "main_command_log_slice_${SLICE_ID}_worker_${WORKER_ID}"
  zip_gcs_path = os.path.join(bucket_path, zip_name)
  write_startup_script(zip_gcs_path, zip_name, log_name, bucket_path, startup_script_file, args)

  print("Running CQR command...")
  captured_output = run_create_resources(startup_script_file, args)
  if captured_output.returncode != 0:
    print(f"\n\nCreate resource request returned with ERROR returncode {captured_output.returncode}.\n")
    print("Create resource error:\n" + captured_output.stderr.decode())
    return 1
  print("CQR command received! TPUs are being created.\n")

  #### Step 3: Cleanup ####
  # Cleanup locally created directory
  shutil.rmtree(tmp_dir)
  # We leave the zipped script dir and log in GCS

  print("------------------------------------ \n")
  print("multihost_job finished running, TPUs are firing up now to run your job remotely.\n")

  print(f"Your job is being logged, follow it here:\n{google_cloud_logging_url(args.RUN_NAME, args.PROJECT)}\n")

  print(f"To see the output of a single host, you may edit the slice and worker number in the log_file_path property here:"\
      f"\n{google_cloud_logging_single_host_url(args.RUN_NAME, args.PROJECT)}\n")

  print(f"When your job is finished, the main command log is in the GCS bucket here:"\
      f"\n{gcs_bucket_url(args.BUCKET_NAME, bucket_dir, args.PROJECT)}\n")

  print("View the status of the created TPUs via: ")
  print(f"gcloud alpha compute tpus queued-resources list "\
    f"--filter={args.RUN_NAME} --zone={args.ZONE} --project={args.PROJECT}\n")
  return 0

if __name__ == '__main__':
  print("Name is __main__")
  main()
