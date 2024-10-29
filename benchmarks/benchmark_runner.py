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

""" Script to run a benchmark/benchmarks on exsiting xpk or QR nodes (to be implmented)
                          ***** IMPORTANT *****
This script will run specific tuned workload on specified hardwear and software enviroments
Example usages:
  python3 benchmark_runner.py  --project=<my-project> --zone=<zone> \
    --cluster_name=<xpk_cluster_name> --base_output_directory=<output_gcloud_bucket> --device_type=v6e-256 --num_slices=1 --model_name="llama2_70b_4096" --libtpu_version=20241009 --base_docker_image=maxtext_base_image
"""
import argparse
import importlib

import maxtext_trillium_model_configs
from maxtext_xpk_runner import BenchmarkRunner
from maxtext_xpk_runner import HWConfig
from maxtext_xpk_runner import SWconfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkConfig


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
      help='gcloud bucket to store arfifacts.',
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
      choices=[
          'gpt_3_175b',
          'llama2_7b_4096',
          'llama2_70b_4096',
          'llama2_70b_4096_real_data',
          'llama3_1_405b_8192_fsdp_dcn',
          'mixtral_8x7b_dropped',
          'mixtral_8x7b_dropped_int8',
          'mixtral_8x7b_dropless',
          'gemma2_9b_8192',
          'gemma2_27b_8192',
      ],
      default='llama2_70b_4096',
      help=(
          'model to be benchmarked, supported models are gpt_3_175b '
          'llama2_7b_4096 '
          'llama2_70b_4096 '
          'llama2_70b_4096_real_data '
          'llama3_1_405b_8192_fsdp_dcn '
          'mixtral_8x7b_dropped '
          'mixtral_8x7b_dropped_int8 '
          'mixtral_8x7b_dropless '
          'gemma2_9b_8192 '
          'gemma2_27b_8192 '
          'command.'
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

def main() -> None:
  parser = argparse.ArgumentParser(
      prog='benchmark runner', usage='%(prog)s [options]'
  )
  add_shared_arguments(parser)
  options = parser.parse_args()

  cluster_config = XpkConfig(
      cluster_name=options.cluster_name,
      project=options.project,
      zone=options.zone,
      num_slices=options.num_slices,
      device_type=options.device_type,
      base_output_directory=options.base_output_directory,
  )

  v6e_env_configs = SWconfig(
      base_docker_image=options.base_docker_image,
      libtpu_version=options.libtpu_version,
  )

  v6e_256_configs = HWConfig(
      num_slices=options.num_slices, device_type=options.device_type
  )

  model_sets = importlib.import_module('maxtext_trillium_model_configs')
  benchmark_model = getattr(model_sets, options.model_name)

  model_runner = BenchmarkRunner(
      model_name=benchmark_model,
      software_config=v6e_env_configs,
      hardware_config=v6e_256_configs,
  )

  xpk_benchmark_runner(cluster_config, [model_runner])


if __name__ == '__main__':
  main()
