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

"""Common GCS Utils needed by multiple modules"""
import shutil
import json
import os
import re
import socket
from pathlib import Path
from etils import epath
import uuid
from concurrent.futures import ThreadPoolExecutor

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
  """Uploads a directory to a GCS location cleanly, without raising on missing paths or IO errors."""
  try:
    process_index = jax.process_index()
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.warning(f"Failed to query jax.process_index() during upload_dump: {e}; assuming process 0.")
    process_index = 0
  if not all_host_upload and process_index != 0:
    return
  if not _gcs_guard("upload_dump"):
    return
  if not local_dir or not os.path.isdir(local_dir):
    max_logging.warning(f"Local dump directory '{local_dir}' does not exist; skipping upload to '{target_dir}'.")
    return
  if not target_dir or not str(target_dir).startswith("gs://"):
    max_logging.warning(
        f"Target directory '{target_dir}' is not a GCS path (gs://...); "
        f"skipping GCS upload and keeping local dump at '{local_dir}'."
    )
    return
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(str(target_dir))
  if not bucket_name:
    max_logging.warning(
        f"Target GCS directory '{target_dir}' is missing a bucket name; "
        f"skipping GCS upload and keeping local dump at '{local_dir}'."
    )
    return

  files_to_upload = []
  try:
    for root, _, files in os.walk(local_dir):
      for file in files:
        if module_name and module_name not in file:
          continue
        local_path = os.path.join(root, file)
        if os.path.isfile(local_path):
          files_to_upload.append((local_path, os.path.relpath(local_path, local_dir)))
  except OSError as e:
    max_logging.error(f"Failed to scan local dump directory '{local_dir}': {e}")
    return

  if not files_to_upload:
    max_logging.warning(
        f"No matching dump files found in '{local_dir}' (module_name={module_name!r}); "
        f"skipping upload to '{target_dir}'."
    )
    return

  try:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.error(
        f"Failed to access GCS destination '{target_dir}' for dump upload: {e}. Keeping local dump at '{local_dir}'."
    )
    return

  if all_host_upload:
    hostname = socket.gethostname()  # Alternatively can use jax.process_id()
    prefix_name = os.path.join(prefix_name, hostname)
    target_dir = os.path.join(str(target_dir), hostname)

  max_logging.log(f"Uploading Dump to {target_dir}...")
  upload_failed = False
  for local_path, relative_path in files_to_upload:
    file_name = os.path.basename(local_path)
    blob_name = os.path.join(prefix_name, relative_path)
    try:
      max_logging.log(f"Uploading {file_name}")
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(local_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      upload_failed = True
      max_logging.error(f"Failed to upload '{local_path}' to 'gs://{bucket_name}/{blob_name}': {e}")

  if upload_failed:
    max_logging.warning(f"One or more dump files failed to upload to {target_dir}; keeping local dump at '{local_dir}'.")
    return

  max_logging.log(f"Dump Uploaded to {target_dir}!")
  if delete_local_after:
    try:
      shutil.rmtree(local_dir)
    except FileNotFoundError:
      pass
    except OSError as e:
      max_logging.warning(f"Failed to remove local dump directory '{local_dir}': {e}")


def upload_hlo_dump(config):
  """Best-effort HLO dump upload from config without raising into caller cleanup."""
  if not getattr(config, "dump_hlo", False):
    return
  try:
    upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.error(f"Unexpected error during HLO dump upload: {e}")


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


def gcs_delete_directory(directory_path: str):
  """Deletes a "directory" (all blobs with the prefix) from GCS.

  Args:
      directory_path: The GCS path (gs://...) representing the "directory" to delete.
  """
  if not _gcs_guard("gcs_delete_directory"):
    return
  storage_client = storage.Client()
  bucket_name, directory_prefix = parse_gcs_bucket_and_prefix(directory_path)
  bucket = storage_client.bucket(bucket_name)

  # Ensures the prefix has a trailing slash to avoid deleting more than intended.
  if not directory_prefix.endswith("/"):
    directory_prefix += "/"

  blobs = list(bucket.list_blobs(prefix=directory_prefix))
  if blobs:
    # Uses a ThreadPoolExecutor to delete blobs in parallel to match gsutil -m performance.
    def _delete_blob(blob):
      try:
        blob.delete()
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"Error deleting blob {blob.name}: {e}")

    with ThreadPoolExecutor(max_workers=32) as executor:
      executor.map(_delete_blob, blobs)


def gcs_glob_pattern(pattern):
  """
  Globs GCS files and returns a list of full GCS paths.
  """
  storage_client = storage.Client()
  bucket_name, glob_pattern = parse_gcs_bucket_and_prefix(pattern)
  # Without a prefix, GCS scans the whole bucket to evaluate `match_glob`, which can take hours on a large bucket.
  literal_prefix = re.split(r"[*?\[{\\]", glob_pattern, maxsplit=1)[0]
  blobs = storage_client.list_blobs(bucket_name, prefix=literal_prefix or None, match_glob=glob_pattern)
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


def mkdir_and_check_permissions(path: str | epath.Path) -> epath.Path:
  """Creates a directory if it doesn't exist and verifies write permissions.

  This function prevents the program from hanging when an output directory is inaccessible. The standard
  `epath.Path.mkdir` can hang or fail silently when pointed at a path in a non-existent or inaccessible GCS bucket.

  For example, the following code can hang indefinitely:

    from etils import epath
    path = epath.Path("gs://no_such_bucket/path/to/output")
    path.mkdir(exist_ok=True, parents=True)
  """
  if isinstance(path, str):
    path = epath.Path(path)

  is_gcs = path.as_posix().startswith("gs://")
  if is_gcs:
    if len(path.parts) < 3:
      raise ValueError(f"Invalid GCS path (missing bucket name): '{path}'")
    bucket_name = path.parts[2]
    try:
      storage_client = storage.Client()
      storage_client.get_bucket(bucket_name)
    except Exception as e:
      raise FileNotFoundError(f"GCS bucket 'gs://{bucket_name}' not found or accessible.") from e
  path.mkdir(exist_ok=True, parents=True)

  # GCS directories are virtual: gcsfs makes mkdir a no-op, so exists() can return False.
  # Only check exists() for local paths; rely on the write test below for GCS.
  if not is_gcs and not path.exists():
    raise PermissionError(f"Failed to create the directory '{path}'. Please check that you have write access.")

  # Verify write permissions by creating and deleting a temporary file.
  # This handles the case where the directory exists but is not writable.
  temp_file_path = path / f".write_test_{uuid.uuid4()}"
  try:
    temp_file_path.write_text("test")
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise PermissionError(f"Directory '{path}' exists, but is not writable. Please check permissions.") from e
  finally:
    try:
      temp_file_path.unlink()  # Delete the temp file.
    except Exception:  # pylint: disable=broad-exception-caught
      pass  # Suppress errors during cleanup to not hide the original error.

  return path
