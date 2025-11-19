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
This module provides utility functions for custom argument parsing 
and defines a comprehensive set of command-line arguments for configuring a machine learning workload.
"""

import argparse


def parse_int_list(arg):
  """Parses a string with comma-separated values into a list of integers."""
  return [int(x) for x in arg.split(",")]


def parse_str_list(arg):
  """Parses a string with space-separated values into a list of strings."""
  return [s.strip() for s in arg.split(",")]


def str2bool(v):
  """Parses a string representation of a boolean value into a Python boolean."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("true"):
    return True
  elif v.lower() in ("false"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected (e.g., True or False).")


def add_arguments(parser: argparse.ArgumentParser):
  """Add arguments to arg parsers that need it.

  Args:
    parser:  parser to add shared arguments to.
  """
  # Add the arguments for each parser.
  # GCP Configuration
  parser.add_argument("--user", type=str, default="user_name", help="GCP user name.")
  parser.add_argument(
      "--cluster_name",
      type=str,
      default="test-v5e-32-cluster",
      help="Name of the TPU cluster.",
  )
  parser.add_argument("--project", type=str, default="cloud-tpu-cluster", help="GCP project ID.")
  parser.add_argument("--zone", type=str, default="us-south1-a", help="GCP zone for the cluster.")
  parser.add_argument(
      "--device_type",
      type=str,
      default="v5litepod-32",
      help="Type of TPU device (e.g., v5litepod-32).",
  )
  parser.add_argument(
      "--priority",
      type=str,
      choices=["low", "medium", "high", "very high"],
      default="medium",
      help="Priority of the job.",
  )

  # Image Configuration
  parser.add_argument(
      "--server_image",
      type=str,
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server",
      help="Docker image for the proxy server.",
  )
  parser.add_argument(
      "--proxy_image",
      type=str,
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_serve",
      help="Docker image for the server.",
  )
  parser.add_argument(
      "--runner",
      type=str,
      default="us-docker.pkg.dev/path/to/maxtext_runner",
      help="Docker image for the runner.",
  )
  parser.add_argument(
      "--colocated_python_image",
      type=str,
      default=None,
      help="Colocated Python image.",
  )
  parser.add_argument("--worker_flags", type=str, default="", help="Worker flags.")
  parser.add_argument("--proxy_flags", type=str, default="", help="Proxy flags.")
  parser.add_argument("--server_flags", type=str, default="", help="Server flags.")

  # Model Configuration
  parser.add_argument("--benchmark_steps", type=int, default=20, help="Number of benchmark steps.")
  parser.add_argument(
      "--headless",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Run in headless mode.",
  )
  parser.add_argument(
      "--selected_model_framework",
      type=parse_str_list,
      default=["pathways"],
      help="List of model frameworks (e.g., pathways, mcjax)",
  )
  parser.add_argument(
      "--selected_model_names",
      type=parse_str_list,
      default=["llama3_1_8b_8192_v5e_256"],
      help="List of model names (e.g., llama3_1_8b_8192_v5e_256, llama2-7b-v5e-256)",
  )
  parser.add_argument(
      "--num_slices_list",
      type=parse_int_list,
      default=[2],
      help="List of number of slices.",
  )

  # BigQuery configuration
  parser.add_argument(
      "--bq_enable",
      type=str2bool,
      default=False,
      help="Enable BigQuery logging. Must be True or False. Defaults to False.",
  )

  parser.add_argument(
      "--bq_db_project",
      type=str,
      default="",
      help="BigQuery project ID where the logging dataset resides.",
  )

  parser.add_argument(
      "--bq_db_dataset",
      type=str,
      default="",
      help="BigQuery dataset name where metrics will be written.",
  )

  # Other configurations
  parser.add_argument("--xpk_path", type=str, default="~/xpk", help="Path to xpk.")
  parser.add_argument("--delete", action="store_true", help="Delete the cluster workload")
  parser.add_argument("--max_restarts", type=int, default=0, help="Maximum number of restarts")
  parser.add_argument("--temp_key", type=str, default=None, help="Temporary placeholder code")
