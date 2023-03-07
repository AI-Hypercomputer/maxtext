""" Script to run a productionized job in a multislice/multihost environment
This script:
  1) Creates specified TPU(s)
  2) Loads your scripts onto each TPU
  3) Runs your job on each TPU
  4) Logs the output of the job to GCS
  5) Deletes the TPUs
  However the job leaves behind an orphaned QR which has to be cleaned up (by you).
  Use "gcloud alpha compute tpus queued-resources list" to view the QR and
  "gcloud alpha compute tpus queued-resources delete <qr-name>" when your job
  is done. These instructions are also printed at the end of running this script.
Example usages:
  Minimal customization, assuming runner.sh lives in present working directory:
  python3 multihost_job.py  --COMMAND="bash runner.sh" --BUCKET_NAME=<my-bucket>

  Maximal customization, assuming runner.sh lives in path/to/dir
  python3 multihost_job.py  --COMMAND="bash runner.sh" --BUCKET_NAME=<my-bucket> --BUCKET_DIR=job-log-directory \
    --SCRIPT_DIR=path/to/dir --PROJECT=<my-project> --ZONE=<zone> \
    --VERSION=tpu-vm-v4-base --TPU_TYPE=v4-8 --NUM_SLICES=2 -- RUN_NAME=$USER-run-job
Common issues:
  You must be able to communicate with the BUCKET_NAME e.g. via gsutil. You may have to run
  gcloud auth login 
  and
  gcloud auth application-default login
"""

from absl import app, flags
import sys
import subprocess
from datetime import datetime
import os
import json
import shutil

##### Define flags #####
FLAGS = flags.FLAGS
tpu_type_flag = flags.DEFINE_string("TPU_TYPE", "v4-8", "The type of the TPU")
tpu_runtime_version_flag = flags.DEFINE_string("VERSION", "tpu-vm-v4-base", "The runtime version of the TPU")
num_slices_flag = flags.DEFINE_integer("NUM_SLICES", 2, "The number of slices to run the job on")
script_dir_flag = flags.DEFINE_string("SCRIPT_DIR", os.getcwd(), "The local location of the directory to copy to"\
    " the TPUs and run the main command from. Defaults to current working directory.")
command_flag = flags.DEFINE_string("COMMAND", None, "Main command to run on each TPU. This command is run from"\
    " a copied version of SCRIPT_DIR on each TPU worker.")
bucket_name_flag= flags.DEFINE_string("BUCKET_NAME", None, "Name of GCS bucket, e.g. my-bucket")
bucket_dir_flag = flags.DEFINE_string("BUCKET_DIR", "", "Directory within the GCS bucket, can be None, e.g. my-dir")
project_flag = flags.DEFINE_string("PROJECT", None, "GCE project name, defaults to gcloud config project")
zone_flag = flags.DEFINE_string("ZONE", None, "GCE zone, e.g. us-central2-b, defaults to gcloud config compute/zone")
endpoint_flag = flags.DEFINE_string("ENDPOINT", "tpu.googleapis.com", "The endpoint for google API requests.")
run_name_flag = flags.DEFINE_string("RUN_NAME", None, "Run name used for temporary files, defaults to timestamp.")


flags.mark_flag_as_required('COMMAND')
flags.mark_flag_as_required('BUCKET_NAME')

def get_run_name():
  now = datetime.now()
  return os.getlogin() + "-" + now.strftime("%Y-%m-%d-%H-%M-%S")

def move_script_dir_to_gcs(script_dir, tmp_dir, zip_name, bucket_path):
  """ Zip the script directory, cp it to GCS """
  original_working_directory = os.getcwd()
  os.chdir(script_dir) # To tar script_dir, it is most convenient to cd there.

  # Zip script directory, storing it in the logging directory.
  os.makedirs(tmp_dir, exist_ok=True)
  zip_path = os.path.join(tmp_dir, zip_name)
  command = ["tar","--exclude=tmp", "-czf", zip_path, "./"]
  subprocess.run(command, check=True)

  # Move zip file to GCS
  zip_in_gcs_path = os.path.join(bucket_path,zip_name)
  command = ["gsutil","mv",zip_path,zip_in_gcs_path]
  captured_output = subprocess.run(command, check=True, capture_output=True)

  # Cleanup
  os.chdir(original_working_directory)

  return captured_output

def get_project():
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split('\n')
  print(project_outputs)
  if len(project_outputs) < 1:
    sys.exit("You must either specify the project in the PROJECT flag, or set it with 'gcloud compute set project <project>'")
  return project_outputs[-1] # The project name lives on the last line of the output

def get_zone():
  completed_command = subprocess.run(["gcloud", "config", "get", "compute/zone"], check=True, capture_output=True)
  zone_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(zone_outputs) < 1:
    sys.exit("You must either specify the zone in the ZONE flag, or set it with 'gcloud compute set compute/zone <zone>'")
  return zone_outputs[-1] # The zone name lives on the last line of the output

def run_create_resources(run_name, json_path, endpoint, project, zone):
  # pylint: disable=line-too-long
  command = fr'curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" -d @{json_path} https://{endpoint}/v2alpha1/projects/{project}/locations/{zone}/queuedResources\?queued_resource_id\={run_name}'
  captured_output = subprocess.run(command, check=True, shell=True, capture_output=True)
  return captured_output

def create_startup_script_str(run_name, zip_gcs_path, zip_name, main_command, log_name, bucket_path, endpoint):
  return f"""#!/bin/bash
mkdir -p {run_name}
cd {run_name}
gsutil cp {zip_gcs_path} .
tar xzf {zip_name}
python3 -m virtualenv venv
source venv/bin/activate
{get_env_command_str()}
{main_command} > {log_name}
gsutil cp {log_name} "{bucket_path}/{log_name}_slice_"$SLICE_ID"_worker_"$WORKER_ID
{create_kill_command_str(endpoint)}"""

def get_env_command_str():
  # pylint: disable=line-too-long
  return """curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env # store the metadata
NODE_ID=$(grep '^NODE_ID' /tmp/tpu-env | cut -d "'" -f 2)
SLICE_ID=$(echo $NODE_ID | awk -F- '{print $NF}')
WORKER_ID=$(grep '^WORKER_ID' /tmp/tpu-env | cut -d "'" -f 2)
ZONE=$(grep '^ZONE' /tmp/tpu-env | cut -d "'" -f 2)
PROJECT=$(grep '^CONSUMER_PROJECT_ID' /tmp/tpu-env | cut -d "'" -f 2)"""

def create_kill_command_str(endpoint):
  # TODO(b/271321971): Delete the QR in one swoop once this is possible.
  return f"""if [ $WORKER_ID==0 ]; then
  curl -X DELETE -H "Authorization: Bearer $(gcloud auth print-access-token)" https://{endpoint}/v2alpha1/projects/$PROJECT/locations/$ZONE/nodes/$NODE_ID
fi"""

def write_cqr_json_file(json_filename, project, zone, tpu_type, runtime_version, num_slices, run_name, startup_script_str):
  """ Write json file for CQR """
  json_dict = {
      "guaranteed": {"reserved": True},
      "tpu": {
        "node_spec": [
          {
            "parent": f"projects/{project}/locations/{zone}",
            "node": {
              "accelerator_type": tpu_type,
              "runtime_version": runtime_version,
              "network_config": {
                "network": "default",
                "subnetwork": "default",
                "enable_external_ips": True
              },
              "metadata": {
                "startup-script": startup_script_str
              }
            },
            "multi_node_params": {
              "node_count": num_slices,
              "node_id_prefix": run_name
            }
        }
      ]
    }
  }

  with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(json_dict, f, indent=3) # Indent doesn't matter, but is nice.

def print_flags(tpu_type, runtime_version, num_slices, script_dir, main_command, bucket_name, bucket_dir, endpoint, project,
 zone, run_name):
  """ Print configuration values after defaults have been filled in. """
  print("Running multihost_job with the following configuration:")
  print(f"Project             (--PROJECT)     = {project}")
  print(f"Zone                (--ZONE)        = {zone}")
  print(f"TPU type            (--TPU_TYPE)    = {tpu_type}")
  print(f"TPU runtime version (--VERSION)     = {runtime_version}")
  print(f"Number of slices    (--NUM_SLICES)  = {num_slices}")
  print(f"Script dir          (--SCRIPT_DIR)  = {script_dir}")
  print(f"Bucket name         (--BUCKET_NAME) = {bucket_name}")
  print(f"Bucket dir          (--BUCKET_DIR)  = {bucket_dir}")
  print(f"Command to run      (--COMMAND)     = {main_command}")
  print(f"Endpoint            (--ENDPOINT)    = {endpoint}")
  print(f"Run name            (--RUN_NAME)    = {run_name}\n")

################### Main ###################
def main(argv) -> None:
  print("\nStarting multihost_job...\n", flush=True)

  #### Parse flags ####
  FLAGS(argv)  # parses the python command inputs into FLAG objects
  tpu_type = tpu_type_flag.value
  tpu_runtime_version = tpu_runtime_version_flag.value
  num_slices = num_slices_flag.value
  script_dir = script_dir_flag.value
  main_command = command_flag.value
  bucket_name = bucket_name_flag.value
  bucket_dir = bucket_dir_flag.value
  endpoint = endpoint_flag.value
  project = project_flag.value
  zone = zone_flag.value
  run_name = run_name_flag.value

  if not project:
    project = get_project()
  if not zone:
    zone = get_zone()
  if not run_name:
    run_name = get_run_name() # Used for QR name, TPU_PREFIX, logging file, and tmp json file.

  print_flags(tpu_type, tpu_runtime_version, num_slices, script_dir, main_command, bucket_name, bucket_dir,
    endpoint, project, zone, run_name)

  ##### Step 1: Zip code and move it to GCS #####
  tmp_dir_relative_to_script = "tmp/" + run_name + "/"
  tmp_dir = os.path.join(script_dir, tmp_dir_relative_to_script)
  zip_name = "script_dir_zip_" + run_name + ".tar.gz"
  bucket_dir = os.path.join(bucket_dir, run_name)
  bucket_path = os.path.join(f"gs://{bucket_name}", bucket_dir)

  print(f"Moving {script_dir} to {bucket_path}...")
  captured_output = move_script_dir_to_gcs(script_dir, tmp_dir_relative_to_script, zip_name, bucket_path)
  if captured_output.returncode !=0:
    print("\n\n Moving code to GCS failed")
    print(f"Running 'gsutil mv zip {bucket_path}' failed with error: ")
    print(captured_output.stderr.decode())
    print("\nYou may need to run 'gcloud auth login'")
    return -1
  print("Move successful!\n")

  #### Step 2: Run the CQR command ####
  log_name = "main_command_log.txt" # TODO: This will be replaced by a cloud logging solution.
  zip_path = os.path.join(bucket_path, zip_name)
  startup_script_str = create_startup_script_str(run_name, zip_path, zip_name, main_command, log_name, bucket_path, endpoint)
  json_filename = 'cqr_request_' + run_name + '.json'
  json_path = os.path.join(tmp_dir, json_filename)

  write_cqr_json_file(json_path, project, zone, tpu_type, tpu_runtime_version, num_slices, run_name, startup_script_str)
  print("Running CQR command...")
  captured_output = run_create_resources(run_name, json_path, endpoint, project, zone)
  if captured_output.returncode !=0 or "Warning" in captured_output.stderr.decode():
    print("\n\nCreate resource request returned with ERROR status.\n")
    print("Create resource error:\n" + captured_output.stderr.decode())
    return 1
  print("CQR command received! TPUs are being created.\n")

  #### Step 3: Cleanup ####
  # Cleanup locally created directory
  shutil.rmtree(tmp_dir)
  # We leave the zipped script dir and log in GCS

  print("------------------------------------ \n")
  print("multihost_job finished running, TPUs are firing up now to run your job remotely.\n")

  print(f"When your job is finished, the main command log is in the GCS bucket: {bucket_path}\n")

  print("View the status of the created TPUs via: ")
  print(f"gcloud compute tpus tpu-vm list --filter={run_name}\n")

  print("Once your job is finished you should delete your QR with: ")
  print(f"gcloud alpha compute tpus queued-resources delete {run_name}")
  return 0

if __name__ == '__main__':
  app.run(main)
