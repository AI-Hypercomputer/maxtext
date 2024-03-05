import argparse
import subprocess
import time
import csv
import os
import sys

from google.cloud import storage
from google.cloud import bigquery

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="",
                    help='The name of the current run')
parser.add_argument('--xpk_dir', type=str, default="",
                    help='The path to the xpk.py file')
parser.add_argument('--hardware', type=str, default="",
                    help='The hardware to run the benchmark')
parser.add_argument('--cluster', type=str, default="",
                    help='The cluster name to run the benchmark')
parser.add_argument('--zone', type=str, default="",
                    help='The zone of the cluster')
parser.add_argument('--project', type=str, default="",
                    help='The project of the cluster')
parser.add_argument('--num_slices', type=str, default="",
                    help='The number of slices')
parser.add_argument('--device_type', type=str, default="",
                    help='The type of the device')
parser.add_argument('--tpu_type', type=str, default="",
                    help='The type of TPU devices')
parser.add_argument('--mode', type=str, default="",
                    help='Mode of the checkpointing run, either read or write')
parser.add_argument('--model_size', type=str, default="",
                    help='Size of the model, e.g. 16, 32, 64, 128...')
parser.add_argument('--steps', type=str, default="",
                    help='Number of steps to run')
parser.add_argument('--output_path', type=str, default="",
                    help='The path to store checkpoints')
parser.add_argument('--dataset_path', default="",
                    help="The dataset path to load dataset from")
parser.add_argument('--gcs_metrics_dir', default="",
                    help='The GCS dir to store the metrics files, if not set, no metrics files will be uploaded')
parser.add_argument('--bq_dataset', type=str, default="",
                    help='The Big Query dataset to store the metrics table, if not set, no bq tables will be created')
parser.add_argument('--previous_state', type=str, default="",
                    help='The path to load checkpoints from')
args = parser.parse_args()

NUM_PROCESSES=64
GCS_METRICS_FILE='combined.csv'
TESS_PROJECT='gcs-tess'

def construct_xpk_command():
  """Construct xpk command"""
  xpk_command = ['python3', args.xpk_dir, 'workload', 'create', f'--cluster={args.cluster}',
                 '--base-docker-image=maxtext_base_image', f'--workload={args.run_name}',
                 f'--num-slices={args.num_slices}', f'--zone={args.zone}', f'--project={args.project}',
                 '--priority=high']
  if args.hardware == 'tpu':
    xpk_command.append(f'--tpu-type={args.tpu_type}')
  elif args.hardware == 'cpu':
    xpk_command.append(f'--device-type={args.device_type}')
  xpk_command.append('--command')
  return xpk_command

def construct_command():
  """Construct xpk and maxtext commands to run optimal configs for checkpointing"""
  xpk_command = construct_xpk_command()
  executable = f"standalone_checkpointer_{args.mode}.py"
  # executable = 'train.py'
  maxtext_command = (
    f'bash MaxText/configs/v5e/{args.model_size}b.sh '
    f'RUN_NAME={args.run_name} '
    f'STEPS={args.steps} '
    f'MODE={args.mode} '
    f'EXECUTABLE={executable} '
    f'GCS_METRICS_DIR={args.gcs_metrics_dir} '
    f'BQ_DATASET={args.bq_dataset} '
    f'OUTPUT_PATH={args.output_path} '
    f'DATASET_PATH={args.dataset_path} '
    f'HARDWARE={args.hardware} '
    f'PLATFORM=gke '
  )
  if args.hardware == 'cpu':
    maxtext_command = 'JAX_PLATFORMS=cpu ' + maxtext_command
  if args.previous_state != '':
    maxtext_command += f'PREVIOUS_STATE={args.previous_state} '

  # Combine xpk command with maxtext command.
  xpk_command.append(maxtext_command)
  return xpk_command

def wait_for_pods_to_finish():
  """Return the status of the pods"""
  while True:
    output = subprocess.check_output(["kubectl", "get", "pods"])
    output = output.decode("utf-8")
    lines = output.splitlines()
    for line in lines:
      if args.run_name in line:
        if 'Completed' in line:
          return True
        elif 'Error' in line:
          return False
        else:
          break
    print("Checking pods status every 60s")
    time.sleep(60)

def download_blob(source_gcs_name, destination_file_name):
  """Downloads a file from a GCS location and save to a local file"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(source_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  # Download the file to a destination
  blob.download_to_filename(destination_file_name)

def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key

def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)

def create_big_query_table(table_id):
  """Create a big query table from a CSV file stored in GCS"""
  client = bigquery.Client()
  job_config = bigquery.LoadJobConfig(
    autodetect=True,
    # The source format defaults to CSV, so the line below is optional.
    source_format=bigquery.SourceFormat.CSV,
    # WRITE_TRUNCATE will erase existing data before writing new data.
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
  )
  gcs_source = os.path.join(args.gcs_metrics_dir, GCS_METRICS_FILE)
  load_job = client.load_table_from_uri(
    gcs_source, table_id, job_config=job_config
  )  # Make an API request.
  load_job.result()  # Waits for the job to complete.
  destination_table = client.get_table(table_id)  # Make an API request.
  print(f"Big Query: successfully created table from GCS: {destination_table}")

def aggregate_metrics():
  """Combines the csv files from each process and aggregate into one to be consumed by Big Query"""
  combine_data = []
  for n in range(NUM_PROCESSES):
    path = f"{args.gcs_metrics_dir}/{args.run_name}_{n}.csv"
    download_blob(path, "tmp.csv")
    with open("tmp.csv", mode='r', encoding="utf-8") as file:
      csv_file = csv.reader(file)
      for line in csv_file:
        combine_data.append(line)
  with open(GCS_METRICS_FILE, 'w', encoding="utf-8", newline='') as file:
    writer = csv.writer(file)
    header = ["process", "step", f"ckpt_{args.mode}_time"]
    writer.writerow(header)
    for _, t in enumerate(combine_data):
      writer.writerow(t)

def main() -> None:
  command = construct_command()
  subprocess.run(command, capture_output=True, check=True)
  print(f"Running xpk command and waiting for pods to finish: {command}")
  if not wait_for_pods_to_finish():
    sys.exit("An error occurred, please check pod logs")
  if args.gcs_metrics_dir != '':
    # Aggregate csv files from each process and combine them into one.
    aggregate_metrics()
    # Upload the combined metrics file to GCS.
    upload_blob(f"{args.gcs_metrics_dir}/{GCS_METRICS_FILE}", GCS_METRICS_FILE)
    print(f"Uploaded combined csv file to {args.gcs_metrics_dir}")
    if args.bq_dataset != '':
      # Create a table for the uploaded combined csv file.
      table_id = f"{TESS_PROJECT}.{args.bq_dataset}.{args.run_name}"
      create_big_query_table(table_id)
    # Clean up local files.
    os.remove(GCS_METRICS_FILE)
    os.remove("tmp.csv")
  print(f"{args.run_name} finished successfully!")
    
if __name__ == "__main__":
  main()