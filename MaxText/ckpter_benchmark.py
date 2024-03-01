import argparse
import subprocess
import time
import csv
import os

from google.cloud import bigquery
from google.cloud import storage

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default="",
                    help='The name of the current run')
parser.add_argument('--mode', type=str, default="",
                    help='Mode of the checkpointing run, either read or write')
parser.add_argument('--model_size', type=str, default="",
                    help='Size of the model, e.g. 16, 32, 64, 128...')
parser.add_argument('--steps', type=str, default="",
                    help='Number of steps to run')
parser.add_argument('--output_path', type=str, default="",
                    help='The path to store checkpoints')
parser.add_argument('--gcs_metrics_dir', default="",
                    help='The GCS dir to store the metrics files, if not set, no metrics files will be uploaded')
parser.add_argument('--bq_dataset', type=str, default="",
                    help='The Big Query dataset to store the metrics table, if not set, no bq tables will be created')
parser.add_argument('--previous_state', type=str, default="",
                    help='The path to load checkpoints from')
parser.add_argument('--dataset_path', type=str, default="",
                    help='The path to read the dataset')
args = parser.parse_args()

NUM_PROCESSES=64
GCS_METRICS_FILE='combined.csv'
TESS_PROJECT='gcs-tess'

def construct_command():
  """Construct xpk command to run optimal configs for checkpointing"""
  xpk_command = ['xpk', 'workload', 'create', '--cluster=v5e-256-bodaborg-us-west4', 
                 '--base-docker-image=maxtext_base_image', f'--workload={args.run_name}',
                 '--tpu-type=v5litepod-256', '--num-slices=1', '--zone=us-west4-a',
                 '--project=tpu-prod-env-multipod', '--priority=high', '--command']
  if args.mode == 'read':
    script = 'standalone_checkpointer_read.py'
  elif args.mode == 'write':
    script = 'standalone_checkpointer.py'
  script = 'train.py'
  maxtext_command = (
    f'bash MaxText/configs/v5e/{args.model_size}b.sh '
    f'RUN_NAME={args.run_name} '
    f'STEPS={args.steps} '
    f'MODE={args.mode} '
    f'SCRIPT={script} '
    f'GCS_METRICS_DIR={args.gcs_metrics_dir} '
    f'BQ_DATASET={args.bq_dataset} '
    f'PREVIOUS_STATE={args.previous_state} '
    f'OUTPUT_PATH={args.output_path} '
    f'DATASET_PATH={args.dataset_path} '
    f'PLATFORM=gke '
  )
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

def download_blob(source_gcs_name, destination_file_name):
  """Downloads a file from a GCS location and save to a local file"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(source_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  # Download the file to a destination
  blob.download_to_filename(destination_file_name)

def main() -> None:
  command = construct_command()
  subprocess.run(command, capture_output=True, check=True)
  print(f"Running xpk command and waiting for pods to finish: {command}")
  if wait_for_pods_to_finish():
    # Aggregate csv files from each process and combine them into one.
    aggregate_metrics()
    # Upload the combined metrics file to GCS.
    upload_blob(f"{args.gcs_metrics_dir}/{GCS_METRICS_FILE}", GCS_METRICS_FILE)
    print(f"Uploaded combined csv file to {args.gcs_metrics_dir}")
    # Create a table for the uploaded combined csv file.
    table_id = f"{TESS_PROJECT}.{args.bq_dataset}.{args.run_name}"
    create_big_query_table(table_id)
    # Clean up local files.
    os.remove(GCS_METRICS_FILE)
    os.remove("tmp.csv")
  else:
    print("An error occurred, please check pod logs")
    
if __name__ == "__main__":
  main()