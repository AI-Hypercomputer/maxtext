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

"""Launches the DeepSeek-V3 671B FP8 TPU7x recipes with Cluster Toolkit.

Requires `gcluster` and `kubectl` (>= 1.31) on PATH, and a GKE TPU7x cluster
with a slice of the recipe's topology (see `config.RECIPES`).

Example (8k):

  python3 -m benchmarks.deepseek3_671b_tpu7x.benchmark \\
    --recipe=8k \\
    --project=<project> --location=<zone or region> --cluster_name=<cluster> \\
    --docker_image=<maxtext image> \\
    --base_output_directory=gs://<bucket>/<dir> \\
    --dataset_path=gs://<bucket>/tfds-reshard-16384/ \\
    --load_parameters_path=gs://<bucket>/<checkpoint>/0/items

Add `--dry_run` to print the `gcluster job submit` command without running it.
"""

import argparse
import dataclasses
import getpass
import re
import sys
import time

from benchmarks.command_utils import run_command_with_updates
from benchmarks.ctk_configs import ClusterConfig
from benchmarks.deepseek3_671b_tpu7x import config
from benchmarks.maxtext_ctk_runner import generate_workload_cmd
from benchmarks.maxtext_ctk_runner import LibTpuType
from benchmarks.maxtext_ctk_runner import WorkloadConfig


def _parse_args(argv):
  """Parses command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--recipe", choices=sorted(config.RECIPES), default="8k", help="Recipe (scale) to run.")
  parser.add_argument("--project", required=True, help="GCP project of the GKE cluster.")
  parser.add_argument(
      "--zone",
      "--location",
      dest="zone",
      required=True,
      help="Zone (or region) of the GKE cluster, passed to gcluster as --location.",
  )
  parser.add_argument("--cluster_name", required=True, help="GKE cluster name.")
  parser.add_argument("--docker_image", required=True, help="MaxText Docker image to run.")
  parser.add_argument("--base_output_directory", required=True, help="GCS directory for logs and metrics.")
  parser.add_argument("--dataset_path", required=True, help="C4 TFDS dataset path (dataset_path).")
  parser.add_argument(
      "--load_parameters_path",
      required=True,
      help="Pre-sharded DeepSeek-V3 671B MaxText checkpoint (load_parameters_path).",
  )
  parser.add_argument("--run_name", default=None, help="Workload name. Defaults to <user>-ds3-<recipe>-<MMDDHHMM>.")
  parser.add_argument("--num_steps", type=int, default=100, help="Maximum training steps.")
  parser.add_argument("--priority", default="high", help="Workload priority.")
  parser.add_argument("--max_restarts", type=int, default=1, help="JobSet max restarts.")
  parser.add_argument("--dry_run", action="store_true", help="Print the gcluster command instead of running it.")
  return parser.parse_args(argv)


def _default_run_name(recipe_name: str) -> str:
  """Returns `<user>-ds3-<recipe>-<MMDDHHMM>`, a DNS label of at most 28 characters."""
  user = re.sub(r"[^a-z0-9]+", "-", getpass.getuser().lower()).strip("-")[:10] or "user"
  return f"{user}-ds3-{recipe_name}-{time.strftime('%m%d%H%M')}"


def main(argv=None) -> int:
  """Launches (or, with --dry_run, prints) the gcluster command; returns its exit code."""
  args = _parse_args(sys.argv[1:] if argv is None else argv)
  recipe = config.RECIPES[args.recipe]
  run_name = args.run_name or _default_run_name(args.recipe)

  model = dataclasses.replace(
      recipe.model,
      tuning_params={
          **recipe.model.tuning_params,
          "dataset_path": args.dataset_path,
          "load_parameters_path": args.load_parameters_path,
      },
  )

  cluster_config = ClusterConfig(
      cluster_name=args.cluster_name,
      project=args.project,
      zone=args.zone,
      device_type=recipe.device_type,
      compute_type=recipe.compute_type,
      topology=recipe.topology,
  )

  workload_config = WorkloadConfig(
      model=model,
      num_slices=recipe.num_slices,
      device_type=recipe.device_type,
      topology=recipe.topology,
      compute_type=recipe.compute_type,
      base_output_directory=args.base_output_directory,
      base_docker_image=args.docker_image,
      libtpu_type=LibTpuType.MAXTEXT,
      num_steps=args.num_steps,
      priority=args.priority,
      max_restarts=args.max_restarts,
      run_name=run_name,
      generate_metrics_and_upload_to_big_query=False,
  )

  command, name = generate_workload_cmd(cluster_config, workload_config, workload_name=run_name)
  if args.dry_run:
    print(command)
    return 0

  return run_command_with_updates(command, name)


if __name__ == "__main__":
  sys.exit(main())
