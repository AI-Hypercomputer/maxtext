"""
Goodput Prototype (TODO: Move to packaged library)
"""

import os
import max_logging
import time
import jax
import datetime
import json
from google.cloud import storage
import logging
logger = logging.getLogger(__name__)
step_time_start_dict = {}
step_time_filename = f"goodput_step_times.txt"
job_runtime_filename = f"goodput_job_runtime.txt"

def should_log_goodput(config):
  """Determine whether or not to enable Goodput logging"""
  if jax.process_index == 0 and config.enable_goodput_metrics:
      return True
  return False

def get_goodput_dir(config):
  """Derive the goodput directory where logs will be stored"""
  goodput_dir = os.path.join(config.base_output_directory, config.run_name, "goodput", "")
  return goodput_dir

def get_gcs_bucket_and_blob_name(destination_gcs_name):
  """Derive the GCS blob and bucket name"""
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  blob = "/".join(path_parts)
  return bucket, blob

def upload_file_to_gcs(destination_gcs_name, source_file_name):
  """Upload a file to a GCS location"""
  try:
    bucket_name, blob_name = get_gcs_bucket_and_blob_name(destination_gcs_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error("Error in uploading file to GCS location.", e)

def download_file_from_gcs(source_gcs_name, destination_file_name):
   """Download a file from a GCS location"""
   try:
      bucket_name, blob_name = get_gcs_bucket_and_blob_name(source_gcs_name)
      storage_client = storage.Client()
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(blob_name)
      blob.download_to_filename(destination_file_name)
      return True
   except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error("Error in downloading file from GCS location.", e)
      return False

def record_data(gcs_location, filename, json_serialized_dict):
  """Record data from serialized object to a file"""
  if download_file_from_gcs(gcs_location, filename):
    with open(filename, 'a', encoding="utf8") as record_file:
      record_file.write(str(json.dumps(json_serialized_dict))+'\n')
      max_logging.log("Writing to exisiting file in GCS")
  else:
    with open(filename, 'w', encoding="utf8") as record_file:
      record_file.write(str(json.dumps(json_serialized_dict))+'\n')
      max_logging.log("Creating a new file to store in GCS")
    
  record_file.close()
  max_logging.log(f"Uploading file {filename} to GCS...")
  upload_file_to_gcs(gcs_location, filename)
  max_logging.log(f"File {filename} uploaded successfully!")
   
def goodput_log_step_time(config, step, step_time):
  """Log step time"""
  # TODO: Record start and end time and aggregate into a single dictionary before recording
  step_time_seconds = step_time.total_seconds()
  log_unix_time = time.mktime(datetime.datetime.now().timetuple())
  step_time_start_dict =  {'job_name': str(config.run_name) ,'step_count' : int(step), 'step_time' : float(step_time_seconds), 'timestamp' : float(log_unix_time)}
  gcs_location = os.path.join(get_goodput_dir(config), step_time_filename)
  record_data(gcs_location, step_time_filename, step_time_start_dict)
      
def goodput_log_job_runtime(config, runtime):
  """Log job runtime"""
  # TODO: Record start and end time and aggregate into a single dictionary before recording
  runtime_seconds = runtime.total_seconds()
  log_unix_time = time.mktime(datetime.datetime.now().timetuple())
  runtime_dict =  {'job_name': str(config.run_name),'runtime' : float(runtime_seconds), 'timestamp' : float(log_unix_time)}
  gcs_location = os.path.join(get_goodput_dir(config), job_runtime_filename)
  record_data(gcs_location, job_runtime_filename, runtime_dict)
  
def get_productive_training_time(config):
  """Get productive training time"""
  # TODO: Compare with checkpoint progress start time and whether checkpoint was saved
  training_time = 0
  step_time_gcs_location = os.path.join(get_goodput_dir(config), step_time_filename)
  if download_file_from_gcs(step_time_gcs_location, step_time_filename):
    with open(step_time_filename, 'r') as step_file:
      step_time_dict = {}
      for line in step_file:
        data = json.loads(line)
        step_count = data['step_count']
        step_time = data['step_time']
        step_time_dict[step_count] = step_time
      training_time = sum(step_time_dict.values())
  return training_time
  
def get_total_job_time(config):
  """Get total job run time"""
  job_time = 1
  job_time_gcs_location = os.path.join(get_goodput_dir(config), job_runtime_filename)
  if download_file_from_gcs(job_time_gcs_location, job_runtime_filename):
    with open(job_runtime_filename, 'r') as runtime_file:
      for line in runtime_file:
        pass
      final_runtime_info = line
      data = json.loads(final_runtime_info)
      job_time = data['runtime']
  return job_time

def get_job_goodput(config):
  """Compute Goodput"""
  productive_training_time = get_productive_training_time(config)
  total_job_runtime = get_total_job_time(config)
  return float(productive_training_time / total_job_runtime) * 100
