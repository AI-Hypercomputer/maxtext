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

""" Script to run a benchmark/benchmarks on existing xpk or QR nodes (to be implemented)
                          ***** IMPORTANT *****
This script will run specific tuned workload on specified hardware and software environments
Example usages:
  python3 benchmark_runner.py xpk --project=<my-project> --zone=<zone> \
    --cluster_name=<xpk_cluster_name> --base_output_directory=<output_gcloud_bucket> --device_type=v6e-256 
    --num_slices=1 --model_name="llama2_70b_4096" --libtpu_version=20241009 --base_docker_image=maxtext_base_image
"""
import argparse
import os
import time

from benchmarks.benchmark_utils import str2bool
from benchmarks.maxtext_trillium_model_configs import trillium_model_dict
from benchmarks.maxtext_v5p_model_configs import v5p_model_dict
from benchmarks.maxtext_v5e_model_configs import v5e_model_dict
from benchmarks.convergence.c4_exp import c4_pretrain_model_dict
from benchmarks.maxtext_xpk_runner import PathwaysConfig
from benchmarks.maxtext_xpk_runner import WorkloadConfig
from benchmarks.maxtext_xpk_runner import xpk_benchmark_runner
from benchmarks.maxtext_xpk_runner import on_device_benchmark_runner
from benchmarks.xpk_configs import XpkClusterConfig
from benchmarks.maxtext_xpk_runner import LibTpuType


def add_pathways_arguments(parser: argparse.ArgumentParser):
  """Add pathways arguments to arg parsers that need it.

  Args:
    parser:  parser to add shared arguments to.
  """
  # Add the arguments for each parser.
  parser.add_argument(
      "--pathways_server_image",
      type=str,
      default=("us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"),
      help="version of pathways server image to be benchmarked command.",
  )
  parser.add_argument(
      "--pathways_proxy_server_image",
      type=str,
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server",
      help="version of pathways proxy image to be benchmarked command.",
  )
  parser.add_argument(
      "--pathways_runner_image",
      type=str,
      help="version of pathways runner image to be benchmarked command.",
  )
  parser.add_argument(
      "--colocated_python_sidecar_image",
      type=str,
      help="version of colocated python sidecar image to be benchmarked command.",
  )
  parser.add_argument(
      "--use_pathways",
      type=str2bool,
      default=False,
      help="whether to use pathways or not.",
  )


def add_xpk_runner_arguments(custom_parser: argparse.ArgumentParser):
  """Add arguments to the xpk runner parser.

  Args:
    custom_parser: parser to add shared arguments to.
  """
  custom_parser.add_argument(
      "--project",
      type=str,
      default=None,
      required=True,
      help='GCE project name, defaults to "gcloud config project."',
  )
  custom_parser.add_argument(
      "--zone",
      type=str,
      default=None,
      required=True,
      help=(
          'GCE zone, e.g. us-central2-b, defaults to "gcloud config '
          'compute/zone." Only one of --zone or --region is allowed in a '
          "command."
      ),
  )
  custom_parser.add_argument(
      "--cluster_name",
      type=str,
      default=None,
      required=True,
      help="cluster name The name of the cluster to run the job on. command.",
  )
  custom_parser.add_argument(
      "--base_output_directory",
      type=str,
      default=None,
      required=True,
      help="gcloud bucket to store artifacts.",
  )
  custom_parser.add_argument(
      "--device_type",
      type=str,
      default=None,
      required=True,
      help="tpu device type command.",
  )
  custom_parser.add_argument(
      "--num_slices",
      type=int,
      default="1",
      help="Number of slices for tpu devices command.",
  )
  custom_parser.add_argument(
      "--model_name",
      type=str,
      choices=list(trillium_model_dict.keys())
      + list(v5p_model_dict.keys())
      + list(v5e_model_dict.keys())
      + list(c4_pretrain_model_dict.keys()),
      default=list(trillium_model_dict.keys())[0],
      help="model to be benchmarked, supported models are the command choices.",
  )
  custom_parser.add_argument(
      "--libtpu_version",
      type=str,
      default="",
      help="version of libtpu-nightly to be benchmarked command.",
  )
  custom_parser.add_argument(
      "--libtpu_type",
      type=str,
      choices=[t.value for t in LibTpuType],
      default="maxtext-docker",
      help="type of libtpu to be benchmarked command.",
  )
  custom_parser.add_argument(
      "--base_docker_image",
      type=str,
      default="maxtext_base_image",
      help="version of base docker image to be benchmarked command.",
  )
  custom_parser.add_argument(
      "--xpk_path",
      type=str,
      default=os.path.join("~", "xpk"),
      help="path to xpk dir.",
  )
  custom_parser.add_argument(
      "--priority",
      type=str,
      default="medium",
      help="Priority the XPK workload should run with.",
  )
  custom_parser.add_argument(
      "--num_steps",
      type=int,
      default=20,
      help="Number of steps to run the workload for.",
  )
  custom_parser.add_argument(
      "--max_restarts",
      type=int,
      default=0,
      help="Number of restarts to attempt.",
  )
  # To create storage follow https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#storage.
  custom_parser.add_argument(
      "--xpk_storage",
      default=None,
      action="append",
      help="Names of XPK storages the workload uses. Example, --xpk_storage=storage_test1 --xpk_storage=storage_test2",
  )


def add_on_device_runner_arguments(custom_parser: argparse.ArgumentParser):
  """Add arguments to the on-device runner parser.

  Args:
    custom_parser: parser to add shared arguments to.
  """
  custom_parser.add_argument(
      "--base_output_directory",
      type=str,
      default=None,
      required=True,
      help="gcloud bucket to store artifacts.",
  )
  custom_parser.add_argument(
      "--run_name",
      type=str,
      default=None,
      help="run_name for model run",
  )
  custom_parser.add_argument(
      "--model_name",
      type=str,
      choices=list(trillium_model_dict.keys())
      + list(v5p_model_dict.keys())
      + list(v5e_model_dict.keys())
      + list(c4_pretrain_model_dict.keys()),
      default=list(trillium_model_dict.keys())[0],
      help=("model to be benchmarked, supported models are the command choices."),
  )
  custom_parser.add_argument(
      "--libtpu_version",
      type=str,
      default="",
      help="version of libtpu-nightly to be benchmarked command.",
  )
  custom_parser.add_argument(
      "--libtpu_type",
      type=str,
      choices=[t.value for t in LibTpuType],
      default="maxtext-docker",
      help="type of libtpu to be benchmarked command.",
  )
  custom_parser.add_argument(
      "--num_steps",
      type=int,
      default=20,
      help="Number of steps to run the workload for.",
  )


def add_healthscan_runner_arguments(custom_parser: argparse.ArgumentParser):
  """Add arguments to the healthscan runner parser.

  Args:
    custom_parser: parser to add shared arguments to.
  """
  custom_parser.add_argument(
      "--base_output_directory",
      type=str,
      default=None,
      required=True,
      help="gcloud bucket to store artifacts.",
  )
  custom_parser.add_argument(
      "--device_type",
      type=str,
      default=None,
      required=True,
      help="tpu device type command.",
  )
  custom_parser.add_argument(
      "--run_name",
      type=str,
      default=None,
      help="run_name for model run",
  )
  custom_parser.add_argument(
      "--num_steps",
      type=int,
      default=20,
      help="Number of steps to run the workload for.",
  )


def main() -> None:
  parser = argparse.ArgumentParser(prog="benchmark runner", usage="%(prog)s [options]")

  subparsers = parser.add_subparsers(help="", dest="runner")
  xpk_runner_parser = subparsers.add_parser("xpk")
  on_device_runner_parser = subparsers.add_parser("on-device")
  healthscan_runner_parser = subparsers.add_parser("healthscan")
  add_xpk_runner_arguments(xpk_runner_parser)
  add_on_device_runner_arguments(on_device_runner_parser)
  add_pathways_arguments(parser)
  add_healthscan_runner_arguments(healthscan_runner_parser)
  options = parser.parse_args()

  # Check that there are no duplicate model configs
  duplicates = trillium_model_dict.keys() & v5p_model_dict.keys() & v5e_model_dict.keys() & c4_pretrain_model_dict.keys()
  assert len(duplicates) == 0, f"Found duplicate model config {duplicates}"
  libtpu_type = None
  model = None

  if options.runner != "healthscan":
    model = (
        trillium_model_dict.get(options.model_name)
        or v5e_model_dict.get(options.model_name)
        or v5p_model_dict.get(options.model_name)
        or c4_pretrain_model_dict.get(options.model_name)
    )
    match options.libtpu_type:
      case LibTpuType.NIGHTLY.value:
        libtpu_type = LibTpuType.NIGHTLY
      case LibTpuType.CUSTOM.value:
        libtpu_type = LibTpuType.CUSTOM
      case LibTpuType.MAXTEXT.value:
        libtpu_type = LibTpuType.MAXTEXT

  # Set up pathways configs
  pw_config = None
  if options.use_pathways:
    pw_config = PathwaysConfig(
        server_image=options.pathways_server_image,
        proxy_server_image=options.pathways_proxy_server_image,
        runner_image=options.pathways_runner_image,
        colocated_python_sidecar_image=options.colocated_python_sidecar_image,
    )

  if options.runner == "xpk":
    cluster_config = XpkClusterConfig(
        cluster_name=options.cluster_name, project=options.project, zone=options.zone, device_type=options.device_type
    )

    workload_config = WorkloadConfig(
        model=model,
        num_slices=options.num_slices,
        num_steps=options.num_steps,
        device_type=options.device_type,
        base_output_directory=options.base_output_directory,
        priority=options.priority,
        max_restarts=options.max_restarts,
        libtpu_type=libtpu_type,
        libtpu_nightly_version=options.libtpu_version,
        base_docker_image=options.base_docker_image,
        xpk_path=options.xpk_path,
        pathways_config=pw_config,
        # Internal only support, not for customers
        generate_metrics_and_upload_to_big_query=False,
        xpk_storage=options.xpk_storage,
    )

    xpk_benchmark_runner(cluster_config, [workload_config])
  elif options.runner == "on-device":
    # Generate a run_name if it is not passed from CLI or M_RUN_NAME env variable is empty
    curr_date = time.strftime("%Y%m%d")
    if options.run_name is None:
      try:
        run_name = os.environ["M_RUN_NAME"]
        if run_name == "":
          options.run_name = f"{options.model_name}-{curr_date}"
      except KeyError:
        options.run_name = f"{options.model_name}-{curr_date}"
    workload_config = WorkloadConfig(
        model=model,
        num_slices=None,
        device_type=None,
        base_docker_image=None,
        num_steps=options.num_steps,
        base_output_directory=options.base_output_directory,
        libtpu_type=libtpu_type,
        libtpu_nightly_version=options.libtpu_version,
        run_name=options.run_name,
        pathways_config=pw_config,
        # Internal only support, not for customers
        generate_metrics_and_upload_to_big_query=False,
    )
    on_device_benchmark_runner(workload_configs=[workload_config])
  elif options.runner == "healthscan":

    # Pick a model to run based on device_type to stress test
    models = {
        "v5p-128": v5p_model_dict.llama2_70b_v5p_128,
        "v5p-256": v5p_model_dict.llama4_scout_dropless_v5p_256,
        "v5p-512": v5p_model_dict.deepseek_v3_ep_256_v5p_512,
        "v5p-1024": v5p_model_dict.deepseek_v3_ep_256_v5p_512,
        "v5p-2048": v5p_model_dict.deepseek_v3_ep_256_v5p_512,
        "v6e-8": trillium_model_dict.llama2_7b_4096,
        "v6e-16": trillium_model_dict.llama2_7b_4096,
        "v6e-32": trillium_model_dict.llama2_7b_4096,
        "v6e-64": trillium_model_dict.llama2_7b_4096,
        "v6e-128": trillium_model_dict.llama2_70b_4096,
        "v6e-256": trillium_model_dict.llama2_70b_4096,
    }

    curr_date = time.strftime("%Y%m%d")
    workload_config = WorkloadConfig(
        model=models[options.device_type],
        num_slices=None,
        device_type=options.device_type,
        libtpu_type=LibTpuType.MAXTEXT,
        base_docker_image=None,
        num_steps=options.num_steps,
        base_output_directory=options.base_output_directory,
        run_name=f"{curr_date}-health-test",
        # Internal only support, not for customers
        generate_metrics_and_upload_to_big_query=False,
    )

    workload_config.model.tuning_params["gcs_metrics"] = False
    on_device_benchmark_runner(workload_configs=[workload_config])


if __name__ == "__main__":
  main()
