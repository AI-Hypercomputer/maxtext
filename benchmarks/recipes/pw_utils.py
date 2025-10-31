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

"""
This module provides utility functions for Pathways-related benchmark recipes.
"""

import typing

import maxtext_xpk_runner as mxr
from google.cloud import storage
from google.cloud.exceptions import NotFound


def build_user_models(
    selected_model_framework: typing.List[str],
    selected_model_names: typing.List[str],
    device_base_type: str,
    available_model_frameworks: typing.List[str],
    available_models: typing.Dict,
) -> typing.Dict:
  """
  Validates user-selected model frameworks and names, then builds the final models dictionary.

  Args:
    selected_model_framework: A list of user-selected frameworks (e.g., 'pathways').
    selected_model_names: A list of user-selected model names.
    device_base_type: The base device type (e'g', 'v5e').
    available_model_frameworks: A list of all available frameworks.
    available_models: A dictionary mapping device types to available models.

  Returns:
    A dictionary containing the final model configurations.

  Raises:
    ValueError: If a selected framework or model name is not available.
  """
  # Iterate through the list of user-selected model frameworks, validating each one
  for model_framework in selected_model_framework:
    if model_framework not in available_model_frameworks:
      raise ValueError(
          f"Model framework '{model_framework}' not available. "
          f"Available model frameworks are: {list(available_model_frameworks)}"
      )

  # Initialize the model_set list to store the user's selected model configurations
  if device_base_type not in available_models:
    raise ValueError(f"Unknown device base type: {device_base_type}. " f"Original device type was: {device_base_type}")

  # Iterate through the list of user-selected model names, validating each one
  for model_name in selected_model_names:
    if model_name not in available_models[device_base_type]:
      raise ValueError(
          f"Model name '{model_name}' not available for device type '{device_base_type}'. "
          f"Available model names are: {list(available_models[device_base_type].keys())}"
      )

  # Build the model configuration
  models = {}
  for model_framework in selected_model_framework:
    models[model_framework] = []
    for model_name in selected_model_names:
      models[model_framework].append(available_models[device_base_type][model_name])

  return models


def get_cluster_config(cluster_name, project, zone, device_type):
  """
  Generates Cluster configuration objects from a UserConfig.
  """
  cluster_config = mxr.XpkClusterConfig(
      cluster_name=cluster_name,
      project=project,
      zone=zone,
      device_type=device_type,
  )

  return cluster_config


def get_pathways_config(
    server_image, proxy_image, runner, colocated_python_image, headless, server_flags="", proxy_flags="", worker_flags=""
):
  """
  Generates Pathways configuration objects from a UserConfig.
  """
  pathways_config = mxr.PathwaysConfig(
      server_image=server_image,
      proxy_server_image=proxy_image,
      runner_image=runner,
      colocated_python_sidecar_image=colocated_python_image,
      headless=headless,
      server_flags=server_flags,
      proxy_flags=proxy_flags,
      worker_flags=worker_flags,
  )
  return pathways_config

def check_and_create_bucket(storage_client, bucket_name, region):
  """
  Checks if the GCS bucket exists.
  Prints a message if it exists.
  Attempts to create the bucket if it does not exist.
  """
  print(f"Checking GCS bucket: {bucket_name}...")
  
  try:
    # Attempt to retrieve the bucket's metadata
    bucket = storage_client.get_bucket(bucket_name)
    
    # If successful, the bucket exists
    print(f"GCS bucket '{bucket_name}' already exists. No creation needed.")
    return bucket

  except NotFound:
    # If NotFound error is raised, the bucket does not exist
    print(f"GCS bucket '{bucket_name}' not found. Attempting to create...")
    try:
      # Create a new bucket
      # Note: GCS bucket names must be globally unique!
      # The location parameter is used to specify the geographical region (e.g., 'us-central1', 'asia-east1')
      # If not specified, it uses the project's default location.
      new_bucket = storage_client.create_bucket(
          bucket_or_name=bucket_name,
          location=region  # Use the provided region as location
      )
      print(f"Successfully created GCS bucket: '{new_bucket.name}' in region: {new_bucket.location}.")
      return new_bucket
    
    except Exception as e:
      # Catch other potential errors during creation (e.g., permission denied or name already taken)
      print(f"Failed to create GCS bucket! Error message: {e}")
  
  except Exception as e:
    # Catch other potential errors (e.g., authentication failure)
    print(f"An error occurred while checking GCS bucket! Error message: {e}")