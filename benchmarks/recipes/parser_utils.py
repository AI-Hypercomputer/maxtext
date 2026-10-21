# Copyright 2023–2026 Google LLC
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
from collections.abc import Callable
import re


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
  if v.lower() in ("true", "1", "yes"):
    return True
  elif v.lower() in ("false", "0", "no"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected (e.g., True or False).")


# (MaxText key, argparse type, help). Each entry becomes a `--<key>` CLI flag and a
# UserConfig field. None (the default) means "don't emit", so the model config /
# base.yml value applies.
MAXTEXT_FLAG_SPECS: tuple[tuple[str, Callable, str], ...] = (
    # Elastic training
    ("elastic_enabled", str2bool, "Enable elastic training (Pathways)."),
    ("enable_single_controller", str2bool, "Enable single-controller mode."),
    ("elastic_min_slice_count", int, "Min slices to keep training; -1 = pause-resume, 1 = replica-resize."),
    # Goodput
    ("enable_pathways_goodput", str2bool, "Enable Pathways goodput."),
    ("enable_goodput_recording", str2bool, "Record goodput events."),
    ("goodput_upload_interval_seconds", int, "Goodput upload interval in seconds."),
    ("monitor_goodput", str2bool, "Run the goodput monitor."),
    # Checkpointing
    ("enable_checkpointing", str2bool, "Enable checkpointing."),
    ("async_checkpointing", str2bool, "Use async checkpointing."),
    ("enable_checkpoint_cloud_logger", str2bool, "Enable checkpoint cloud logger."),
    ("checkpoint_period", int, "Checkpoint every N steps."),
    # Colocated Python data input / dataset
    ("colocated_python_data_input", str2bool, "Use colocated Python for data input (Pathways)."),
    ("tokenizer_path", str, "Tokenizer path."),
    ("dataset_type", str, "Dataset type, e.g. synthetic, grain, tfds, hf."),
    ("grain_train_files", str, "Grain train file pattern, e.g. gs://bucket/*.array_record."),
)
MAXTEXT_FLAG_NAMES = tuple(spec[0] for spec in MAXTEXT_FLAG_SPECS)

_OVERRIDE_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class _KeyValueAction(argparse.Action):
  """Repeatable KEY=VALUE flag merged into a dict. An empty VALUE maps to None (unset)."""

  def __call__(self, parser, namespace, values, option_string=None):
    key, sep, value = values.partition("=")
    if not sep or not _OVERRIDE_KEY_RE.fullmatch(key):
      parser.error(f"{option_string} expects KEY=VALUE, got {values!r}")
    merged = dict(getattr(namespace, self.dest, None) or {})  # copy; never mutate the default
    merged[key] = value if value != "" else None
    setattr(namespace, self.dest, merged)


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
      default="v6e-256-cluster",
      help="Name of the TPU cluster.",
  )
  parser.add_argument("--project", type=str, default="tpu-prod-env-cluster", help="GCP project ID.")
  parser.add_argument(
      "--zone",
      type=str,
      default="us-east5-b",
      help="GCP zone for the cluster (e.g., us-east5-b). For regional GKE clusters, pass the region via --location.",
  )
  parser.add_argument(
      "--location",
      type=str,
      default=None,
      help=(
          "GCP location (zone or region) for the cluster (e.g., us-central2 for regional clusters or us-south1-a for"
          " zonal clusters)."
      ),
  )
  parser.add_argument(
      "--device_type",
      type=str,
      default="v6e-256",
      help="Type of TPU device (e.g., v6e-256).",
  )
  parser.add_argument(
      "--priority",
      type=str,
      choices=["very-low", "low", "medium", "high", "very-high"],
      default="medium",
      help="Priority of the job.",
  )
  parser.add_argument(
      "--base_output_directory",
      type=str,
      default=None,
      help="GCS bucket path for benchmark outputs.",
  )

  # Image Configuration
  parser.add_argument(
      "--server_image",
      type=str,
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server",
      help="Docker image for the server.",
  )
  parser.add_argument(
      "--proxy_image",
      type=str,
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server",
      help="Docker image for the proxy server.",
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
      default=["llama3_1_8b_8192"],
      help="List of model names (e.g., llama3_1_8b_8192, llama2-7b-v5e-256)",
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

  parser.add_argument(
      "--skip-validation",
      action="store_true",
      default=False,
      help="Skip gcluster prerequisite and dependency validation during workload execution",
  )

  # MaxText training overrides (elastic training, goodput, checkpointing, data input)
  overrides_group = parser.add_argument_group("MaxText training overrides")
  for name, arg_type, help_text in MAXTEXT_FLAG_SPECS:
    overrides_group.add_argument(f"--{name}", type=arg_type, default=None, help=help_text)
  overrides_group.add_argument(
      "--maxtext_override",
      dest="maxtext_overrides",
      action=_KeyValueAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra MaxText key=value override (repeatable). An empty VALUE removes the key, e.g. 'profiler='.",
  )

  # Other configurations
  parser.add_argument(
      "--xpk_path",
      type=str,
      default=None,
      help="Deprecated (no-op): retained for backward compatibility.",
  )
  parser.add_argument("--delete", action="store_true", help="Delete the cluster workload")
  parser.add_argument("--max_restarts", type=int, default=0, help="Maximum number of restarts")
  parser.add_argument("--temp_key", type=str, default=None, help="Temporary placeholder code")
  parser.add_argument("--workload_id", type=str, default=None, help="Optional custom name for the workload.")
