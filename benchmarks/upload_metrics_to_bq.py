"""
 Copyright 2025 Google LLC

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
import argparse
import dataclasses
import getpass
import json
import os
import sys
from typing import Any, Dict, Sequence

import fnmatch

from statistics import median

import omegaconf

from benchmarks.command_utils import run_command_with_updates
from benchmarks.benchmark_db_utils import write_run
from benchmarks.benchmark_db_utils import DEFAULT_LOCAL_DIR
from benchmarks.benchmark_db_utils import recover_tuning_params
from benchmarks.benchmark_db_utils import Metrics

from MaxText.inference_utils import str2bool


hardware_id_to_bf16_tflops = {"v4": 275,
                              "v5e": 197,
                              "v5p": 459,
                              "v6e": 918,
                              "v6e-8": 918,
                              "v6e-1": 918,
                              "a3mega": 989,
                              "a3ultra": 989,
                              }


def add_parser_arguments(parser: argparse.ArgumentParser):
  """Add arguments to the parser.

  Args:
    parser: parser to add shared arguments to.
  """
  parser.add_argument(
      '--metrics_gcs_file',
      type=str,
      required=True,
      help='Path to the metrics file in GCS',
  )
  parser.add_argument(
      '--model_id',
      type=str,
      required=True,
      help='ID of the model',
  )
  parser.add_argument(
      '--hardware_id',
      type=str,
      required=True,
      help='ID of the hardware',
  )
  parser.add_argument(
      '--software_id',
      type=str,
      required=True,
      help='ID of the software',
  )
  parser.add_argument(
      '--number_of_chips',
      type=int,
      required=True,
      help='Number of chips used',
  )
  parser.add_argument(
      '--container_image_name',
      type=str,
      required=True,
      help='Name of the container image used',
  )
  parser.add_argument(
      '--global_batch_size',
      type=int,
      required=True,
      help='Global batch size',
  )
  parser.add_argument(
      '--precision',
      type=str,
      required=True,
      help='Precision used (e.g. fp32, bf16)',
  )
  parser.add_argument(
      '--optimizer',
      type=str,
      required=True,
      help='Optimizer used (e.g. adam, sgd)',
  )
  parser.add_argument(
      '--seq_length',
      type=int,
      required=True,
      help='Sequence length',
  )
  parser.add_argument(
      '--number_of_steps',
      type=int,
      required=True,
      help='Number of steps',
  )
  parser.add_argument(
      '--xla_flags',
      type=str,
      required=True,
      help='XLA flags',
  )
  parser.add_argument(
      '--dataset',
      type=str,
      required=True,
      help='Dataset used',
  )
  parser.add_argument(
      '--run_type',
      type=str,
      required=True,
      help='Type of run (e.g. perf_optimization)',
  )
  parser.add_argument(
      '--config_file',
      type=str,
      required=True,
      help='Configuration file path',
  )
  parser.add_argument(
      '--topology',
      type=str,
      required=True,
      help='The topology of the hardware used in the run (valid for TPUs)',
  )
  parser.add_argument(
      '--tuning_params',
      type=str,
      required=True,
      help='Tuning parameters',
  )
  parser.add_argument(
      '--db_project',
      type=str,
      required=True,
      help='Project of the database',
  )
  parser.add_argument(
      '--db_dataset',
      type=str,
      required=True,
      help='Dataset of the database',
  )
  parser.add_argument(
      '--is_test',
      type=str2bool,
      required=False,
      default=True,
      help='Whether to use the testing project or production project',
  )


def download_metrics_file_locally(metrics_gcs_file: str, local_file: str) -> int:
  command = f"gsutil cp -r {metrics_gcs_file} {local_file}"
  return run_command_with_updates(command, f"Download {metrics_gcs_file} in {local_file}")


# Get the last n datapoints for a target metric
def get_last_n_data(metrics_file, target, n=10):
  last_n_data = []
  with open(metrics_file, 'rt', encoding='utf8') as file:
    lines = file.readlines()
    for line in lines[::-1]:
      metrics = json.loads(line)
      if target in metrics:
        last_n_data.append(metrics[target])
        if len(last_n_data) >= n:
          break
  return last_n_data

# Get the average of the last n datapoints for a specific target metric
def get_metric_average(metrics_file, target, n=10):
  last_n_data = get_last_n_data(metrics_file, target, n=n)
  return sum(last_n_data) / len(last_n_data)


def get_metric_median(metrics_file, target, n=10):
  last_n_data = get_last_n_data(metrics_file, target, n=n)
  return median(last_n_data)

def get_metrics_sum(metrics_file, target, n=10):
  last_n_data = get_last_n_data(metrics_file, target, n=n)
  return sum(last_n_data)

# metric file example:
# {"learning/grad_norm": 1.0000004768371582, "learning/loss": 10.8693265914917, "learning/moe_lb_loss": 0.0, "learning/param_norm": 8122.166015625, "learning/raw_grad_norm": 5.087409973144531, "learning/total_weights": 79694.0, "perf/step_time_seconds": 4.986376, "perf/per_device_tflops": 172.941153140736, "perf/per_device_tflops_per_sec": 34.6827341421377, "perf/per_device_tokens": 24576.0, "perf/per_device_tokens_per_sec": 4928.629529742643, "learning/current_learning_rate": 2.9999999242136255e-05, "step": 0.0, "run_name": "mattdavidow-train-base"}

# It is important to parse the last n steps avoiding steps that have tracing turned on, by default last_n_steps should be total_steps - 10 to avoid initial tracing settings.
# TODO() Support avoiding tracing when the user explicitly enables tracing for other steps.
def parse_metrics(local_metrics_file, total_steps, last_n_steps=10) -> Metrics:
  avg_tflops = get_metric_average(local_metrics_file, "perf/per_device_tflops_per_sec", n=last_n_steps)
  avg_tokens_per_second = get_metric_average(local_metrics_file, "perf/per_device_tokens_per_sec", n=last_n_steps)
  median_step_time = get_metric_median(local_metrics_file, "perf/step_time_seconds", n=last_n_steps)
  e2e_step_time = get_metrics_sum(local_metrics_file, "perf/step_time_seconds", n=total_steps)

  metrics = Metrics(
    avg_tflops_per_sec=avg_tflops,
    avg_tokens_per_sec=avg_tokens_per_second,
    median_step_time=median_step_time,
    e2e_step_time=e2e_step_time)

  return metrics


def update_config_with_tuning_params(base_config: omegaconf.DictConfig,
                                     tuning_params: Dict[str, Any]):
  """Updates base_config with key-value pairs from tuning_params."""
  if tuning_params:
    for key, value in tuning_params.items():
      omegaconf.OmegaConf.update(base_config, key, value, merge=True)
  return base_config


def main(argv: Sequence[str]) -> None:
  is_pathways = os.environ.get('JAX_PLATFORMS', '') == 'proxy'
  is_mcjax_0th_worker = int(os.environ.get('TPU_WORKER_ID', -1)) == 0

  # Only write once for McJAX. Pathways is single controller,
  # so only can write once.
  if not (is_pathways or is_mcjax_0th_worker):
    return

  parser = argparse.ArgumentParser(
      prog='BigQuery metrics uploader', usage='%(prog)s [options]'
  )
  add_parser_arguments(parser)
  options = parser.parse_args()

  print(f"BigQuery metrics uploader got: {options}")
  
  local_dir = DEFAULT_LOCAL_DIR

  # Download metrics from the GCS Bucket
  local_metrics_file = local_dir
  print(f"Attempting metrics download from {options.metrics_gcs_file} to {local_metrics_file}.",flush=True)
  rc = download_metrics_file_locally(metrics_gcs_file=options.metrics_gcs_file, local_file=local_metrics_file)
  if rc != os.EX_OK:
    print("metrics download FAIL")
    sys.exit(rc)
  print("metrics download SUCCESS")

  # Parse Metrics
  # If there are more than 10 steps, have a buffer to avoid profiling bad perf:
  number_of_steps = int(options.number_of_steps)
  if number_of_steps - 10 > 0:
    compute_metrics_of_n_steps = number_of_steps - 10
  else:
    compute_metrics_of_n_steps = number_of_steps
  for file in os.listdir(os.path.join(local_metrics_file,'metrics')):
    if fnmatch.fnmatch(file, 'metrics_step*.txt'):
      file_to_parse = os.path.join(local_metrics_file,'metrics',file)
      print(f"Found metrics file to parse: {file_to_parse}")
      break
  metrics_from_file = parse_metrics(file_to_parse, number_of_steps, compute_metrics_of_n_steps)
  print(f'Metrics: {metrics_from_file}')

  # Convert number of chips to number of nodes (number of vms)
  number_of_chips = int(options.number_of_chips)
  if options.hardware_id.startswith("v"):
    number_of_nodes = number_of_chips // 4
  elif options.hardware_id in ("a3mega", "a3ultra"):
    number_of_nodes = number_of_chips // 8
  else:
    number_of_nodes = number_of_chips

  # Convert tflops to MFU based on hardware_id
  avg_mfu = metrics_from_file.avg_tflops_per_sec / hardware_id_to_bf16_tflops[options.hardware_id]

  run_release_status = "local"

  # Write env variables
  env_dict = {"env_vars": dict(os.environ)}
  env_vars = json.dumps(env_dict)

  # Framework config in json
  base_config = omegaconf.OmegaConf.load(options.config_file)
  print(f"tuning_params: {options.tuning_params}")
  tuning_params_dict = recover_tuning_params(options.tuning_params)
  config = update_config_with_tuning_params(base_config, tuning_params_dict)
  config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
  framework_config = json.dumps({"config": config_dict})

  # Load metrics to bq
  write_run(
      options=options,
      metrics=metrics_from_file,
      mfu=avg_mfu,
      number_of_steps=number_of_steps,
      number_of_nodes=number_of_nodes,
      number_of_chips=number_of_chips,
      run_success=True,
      framework_config_in_json=framework_config,
      env_variables=env_vars,
      run_release_status=run_release_status,
      other_metrics_in_json="",
      comment="",
      nccl_driver_nickname=None,
  )
  print(f"DB write complete in project: {options.db_project}, dataset: {options.db_dataset}.")


if __name__ == "__main__":
  main(sys.argv[1:])
