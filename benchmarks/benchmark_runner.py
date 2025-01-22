"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Script to run a benchmark/benchmarks on existing xpk or QR nodes (to be implemented)
                          ***** IMPORTANT *****
This script will run specific tuned workload on specified hardware and software environments
Example usages:
  python3 benchmark_runner.py  --project=<my-project> --zone=<zone> \
    --cluster_name=<xpk_cluster_name> --base_output_directory=<output_gcloud_bucket> --device_type=v6e-256 --num_slices=1 --model_name="llama2_70b_4096" --libtpu_version=20241009 --base_docker_image=maxtext_base_image
"""
import argparse

from maxtext_trillium_model_configs import trillium_model_dict
from maxtext_xpk_runner import PathwaysConfig
from maxtext_xpk_runner import WorkloadConfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkClusterConfig
from maxtext_xpk_runner import LibTpuType

def add_shared_arguments(custom_parser: argparse.ArgumentParser):
  """Add shared arguments to the parser.

  Args:
    custom_parser: parser to add shared arguments to.
  """
  custom_parser.add_argument(
      '--project',
      type=str,
      default=None, required=True,
      help='GCE project name, defaults to "gcloud config project."',
  )
  custom_parser.add_argument(
      '--zone',
      type=str,
      default=None, required=True,
      help=(
          'GCE zone, e.g. us-central2-b, defaults to "gcloud config '
          'compute/zone." Only one of --zone or --region is allowed in a '
          'command.'
      ),
  )
  custom_parser.add_argument(
      '--cluster_name',
      type=str,
      default=None, required=True,
      help='cluster name The name of the cluster to run the job on. command.',
  )
  custom_parser.add_argument(
      '--base_output_directory',
      type=str,
      default=None, required=True,
      help='gcloud bucket to store artifacts.',
  )
  custom_parser.add_argument(
      '--device_type',
      type=str,
      default=None, required=True,
      help='tpu device type command.',
  )
  custom_parser.add_argument(
      '--num_slices',
      type=int,
      default='1',
      help='Number of slices for tpu devices command.',
  )
  custom_parser.add_argument(
      '--model_name',
      type=str,
      choices=list(trillium_model_dict.keys()),
      default=list(trillium_model_dict.keys())[0],
      help=(
        f'model to be benchmarked, supported models are the command choices.'
      ),
  )
  custom_parser.add_argument(
      '--libtpu_version',
      type=str,
      default='20241009',
      help='version of libtpu-nightly to be benchmarked command.',
  )
  custom_parser.add_argument(
      '--base_docker_image',
      type=str,
      default='maxtext_base_image',
      help='version of base docker image to be benchmarked command.',
  )
  custom_parser.add_argument(
      '--pathways_server_image',
      type=str,
      default=(
          'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/server:latest'
      ),
      help='version of pathways server image to be benchmarked command.',
  )
  custom_parser.add_argument(
      '--pathways_proxy_image',
      type=str,
      default='us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/proxy_server:latest',
      help='version of pathways proxy image to be benchmarked command.',
  )
  custom_parser.add_argument(
      '--pathways_runner_image',
      type=str,
      default='us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest',
      help='version of pathways runner image to be benchmarked command.',
  )
  custom_parser.add_argument(
      '--use_pathways',
      type=bool,
      default=False,
      help='whether to use pathways or not.',
  )
  custom_parser.add_argument(
      '--xpk_path',
      type=str,
      default='~/xpk',
      help='path to xpk dir.',
  )
  custom_parser.add_argument(
      '--priority',
      type=str,
      default='medium',
      help='Priority the XPK workload should run with.',
  )
  custom_parser.add_argument(
      '--num_steps',
      type=int,
      default=20,
      help='Number of steps to run the workload for.',
  )
  custom_parser.add_argument(
      '--max_restarts',
      type=int,
      default=0,
      help='Number of restarts to attempt.',
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      prog='benchmark runner', usage='%(prog)s [options]'
  )
  add_shared_arguments(parser)
  options = parser.parse_args()

  cluster_config = XpkClusterConfig(
        cluster_name=options.cluster_name,
        project=options.project,
        zone=options.zone,
        device_type=options.device_type
      )

  pw_config = None
  if options.use_pathways:
    pw_config = PathwaysConfig(
      server_image=options.pathways_server_image,
      proxy_image=options.pathways_proxy_image,
      runner_image=options.pathways_runner_image,
    )

  assert trillium_model_dict.get(options.model_name) is not None, f'Invalid model name: {options.model_name}'
  workload_config = WorkloadConfig(
    model=trillium_model_dict.get(options.model_name),
    num_slices=options.num_slices,
    num_steps=options.num_steps,
    device_type=options.device_type,
    base_output_directory=options.base_output_directory,
    priority=options.priority,
    max_restarts=options.max_restarts,
    libtpu_type=LibTpuType.NIGHTLY,
    libtpu_nightly_version=options.libtpu_version,
    base_docker_image=options.base_docker_image,
    xpk_path=options.xpk_path,
    pathways_config=pw_config
  )

  xpk_benchmark_runner(cluster_config, [workload_config])


if __name__ == '__main__':
  main()
