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

""" Common GCS Utils needed by multiple modules"""
import shutil
import json
import os
import socket
from pathlib import Path

import yaml

import jax

from maxtext.utils import max_logging
from maxtext.common.gcloud_stub import is_decoupled, gcs_storage

storage = gcs_storage()


def _gcs_guard(operation_name: str) -> bool:
  """Check GCS availability for an operation."""
  if getattr(storage, "_IS_STUB", False):
    if is_decoupled():
      max_logging.log(f"[GCS NO-OP] {operation_name}")
      return False
    raise RuntimeError(f"google-cloud-storage missing for {operation_name}. Install or set DECOUPLE_GCLOUD=TRUE.")
  return True


def write_config_raw_keys_for_gcs(raw_keys):
  """Writes config raw keys to GCS (no-op if disabled or decoupled)."""
  if not raw_keys.get("save_config_to_gcs") or jax.process_index() != 0:
    return
  if not _gcs_guard("write_config_raw_keys_for_gcs"):
    return
  max_logging.log("Writing config to GCS...")

  raw_keys_dict = dict(raw_keys)
  filename = "config.yml"
  with open(filename, "wt", encoding="utf8") as config_for_gcs:
    yaml.dump(raw_keys_dict, config_for_gcs)
  config_for_gcs.close()

  gcs_filename = os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], filename)
  max_logging.log(f"Moving file {filename} to GCS...")
  upload_blob(gcs_filename, filename)
  max_logging.log(f"File {filename} moved successfully!")


def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key


def add_trailing_slash(path):
  if not path.endswith("/"):
    return path + "/"
  return path


def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location (no-op if not found and decoupled)."""
  if not _gcs_guard("upload_blob"):
    return
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)


def upload_dump(local_dir, target_dir, module_name=None, delete_local_after=True, all_host_upload=False):
  """Uploads a directory to a GCS location, with an optional filter (no-op if not found and decoupled)."""
  if not all_host_upload and jax.process_index() != 0:
    return
  if not _gcs_guard("upload_dump"):
    return
  storage_client = storage.Client()
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(target_dir)
  bucket = storage_client.get_bucket(bucket_name)
  if all_host_upload:
    hostname = socket.gethostname()  # Alternatively can use jax.process_id()
    prefix_name = os.path.join(prefix_name, hostname)
    target_dir = os.path.join(target_dir, hostname)
  max_logging.log(f"Uploading Dump to {target_dir}...")
  for root, _, files in os.walk(local_dir):
    for file in files:
      if module_name and module_name not in file:
        continue
      else:
        max_logging.log(f"Uploading {file}")
      local_path = os.path.join(root, file)
      relative_path = os.path.relpath(local_path, local_dir)
      blob_name = os.path.join(prefix_name, relative_path)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(local_path)
  max_logging.log(f"Dump Uploaded to {target_dir}!")
  if delete_local_after:
    shutil.rmtree(local_dir)


def gcs_path_exists(file_path):
  """Checks if a GCS file_path exists (no-op if not found and decoupled)."""
  if not _gcs_guard("gcs_path_exists"):
    return False
  try:
    storage_client = storage.Client()
    bucket_name, file_name = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    return blob.exists()
  except ValueError as e:
    print(f"Error while accessing {file_path} from GCE: {str(e)}")
    return False


def gcs_list_directories(directory_path):
  """
  Lists "directories" (prefixes one level down) within a GCS "directory".

  Args:
      directory_path: The prefix representing the parent "directory".

  Returns:
      A list of "directory" names (prefixes).
  """
  if not _gcs_guard("gcs_list_directories"):
    return []
  storage_client = storage.Client()
  bucket_name, directory_prefix = parse_gcs_bucket_and_prefix(directory_path)
  bucket = storage_client.bucket(bucket_name)

  # Ensures the prefix has a trailing slash to simulate a directory
  if not directory_prefix.endswith("/"):
    directory_prefix += "/"

  # Use list_blobs with a delimiter to get "directories"
  delimiter = "/"
  blobs = bucket.list_blobs(prefix=directory_prefix, delimiter=delimiter)

  directories = []
  # Iterate through the blobs and extract the "directories"
  for page in blobs.pages:
    for prefix in page.prefixes:
      path_obj = Path(prefix)

      directory = path_obj.name

      directories.append(directory)

  return directories


def gcs_glob_pattern(pattern):
  """
  Globs GCS files and returns a list of full GCS paths.
  """
  storage_client = storage.Client()
  bucket_name, glob_pattern = parse_gcs_bucket_and_prefix(pattern)
  blobs = storage_client.list_blobs(bucket_name, match_glob=glob_pattern)
  data_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
  return data_files


def read_bytes_from_gcs(file_path):
  """Read raw bytes from a GCS file.

  Args:
    file_path: The gcs path of the file (e.g. gs://bucket/path/to/file).

  Returns:
    The file contents as bytes, or None if unavailable.
  """
  if not _gcs_guard("read_bytes_from_gcs"):
    return None
  try:
    storage_client = storage.Client()
    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_prefix)
    return blob.download_as_bytes()
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Error reading bytes from GCS path {file_path}: {e}")
    return None


def read_json_from_gcs(file_path):
  """
  Read a json file from gcs bucket.

  Args:
    file_path: The gcs path of the json file.

  Returns:
    A dictionary with content from json file.
  """
  try:
    raw = read_bytes_from_gcs(file_path)
    if raw is None:
      return None
    return json.loads(raw)
  except (ValueError, TypeError, json.JSONDecodeError) as e:
    max_logging.log(f"Error reading JSON file from GCS: {str(e)}")
    return None


def write_dict_to_gcs_json(data_dict, file_path):
  """
  Writes a Python dictionary to a JSON file in GCS.

  Args:
    data_dict: The Python dictionary to write
    file_path: GCS path (Bucket + blob) to create the json file
  """
  if not _gcs_guard("write_dict_to_gcs_json"):
    return
  try:
    storage_client = storage.Client()
    bucket_name, file_prefix = parse_gcs_bucket_and_prefix(file_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_prefix)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(data_dict, indent=4)

    # Upload the JSON string to GCS
    blob.upload_from_string(json_string, content_type="application/json")
  except (ValueError, TypeError, RecursionError) as e:
    print(f"Failed to write json file at {file_path} with error: {str(e)}")
