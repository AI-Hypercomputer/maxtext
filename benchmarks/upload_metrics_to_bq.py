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

from typing import Sequence
from statistics import median
from command_utils import run_command_with_updates
from benchmark_db_utils import install_mantaray_locally
from benchmark_db_utils import write_run
from benchmark_db_utils import DEFAULT_LOCAL_DIR
import dataclasses
import getpass
import json
import os


@dataclasses.dataclass
class Metrics:
  avg_tflops_per_sec: float
  avg_tokens_per_sec: float
  median_step_time: float
  e2e_step_time: float

hardware_id_to_bf16_tflops = {"v4": 275,
                              "v5e": 197,
                              "v5p": 459,
                              "v6e": 918,
                              "v6e-8": 918,
                              "v6e-1": 918,
                              "v6p": 1992,
                              "a3mega": 989,
                              "a3ultra": 989,
                              }

def download_metrics_file_locally(metrics_gcs_file: str, local_file: str) -> int:
  command = f"gsutil cp {metrics_gcs_file} {local_file}"
  return run_command_with_updates(command, f"Download {metrics_gcs_file} in {local_file}")

# Get the last n datapoints for a target metric
def get_last_n_data(metrics_file, target, n=10):
  last_n_data = []
  with open(metrics_file, 'r', encoding='utf8') as file:
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

# Args
# metric gcs file location
#
def main(argv: Sequence[str]) -> None:
  metrics_gcs_file = argv[0]
  model_id = argv[1]
  hardware_id = argv[2]
  software_id = argv[3]
  number_of_chips = argv[4]
  container_image_name = argv[5]
  global_batch_size = argv[6]
  precision = argv[7]
  optimizer = argv[8]
  seq_length = argv[9]
  number_of_steps = argv[10]
  xla_flags = argv[11]
  dataset = argv[12]
  run_type = argv[13]
  config_file = argv[14]
  topology = argv[15]

  local_dir = DEFAULT_LOCAL_DIR

  # 1.Download metrics from the GCS Bucket
  local_metrics_file = local_dir
  rc = download_metrics_file_locally(metrics_gcs_file=metrics_gcs_file, local_file=local_metrics_file)
  if rc != 0:
    exit(rc)

  # 2.Install MantaRay locally
  rc = install_mantaray_locally(local_dir)
  if rc != 0:
    exit(rc)

  # 3.Parse Metrics
  # If there are more than 10 steps, have a buffer to avoid profiling bad perf:
  number_of_steps = int(number_of_steps)
  if number_of_steps - 10 > 0:
    compute_metrics_of_n_steps = number_of_steps - 10
  else:
    compute_metrics_of_n_steps = number_of_steps

  metrics_from_file = parse_metrics(local_metrics_file, number_of_steps, compute_metrics_of_n_steps)

  # TODO() convert hardware_id to bq acceptable term
  hardware_id = hardware_id

  # Convert number of chips to number of nodes (number of vms)
  # Trillium 4 chips per vm
  number_of_chips = int(number_of_chips)
  if hardware_id.startswith("v"):
    number_of_nodes = number_of_chips // 4
  elif hardware_id == "a3mega" or hardware_id == "'a3ultra":
    number_of_nodes = number_of_chips // 8
  else:
    number_of_nodes = number_of_chips

  # Convert tflops to MFU based on hardware_id
  # Trillium 918
  avg_mfu = metrics_from_file.avg_tflops_per_sec / hardware_id_to_bf16_tflops[hardware_id]

  run_release_status = "local"

  # Write env variables
  env_dict = {"env_vars": dict(os.environ)}
  env_vars = json.dumps(env_dict)

  # TODO() support framework config in json
  framework_config = json.dumps({"config": config_file})

  # TODO() convert hardware_id to topology
  # topology = "16x16"

  # Load metrics to bq
  write_run(
    model_id=model_id,
    hardware_id=hardware_id,
    software_id=software_id,
    number_of_nodes=number_of_nodes,
    number_of_chips=number_of_chips,
    container_image_name=container_image_name,
    global_batch_size=global_batch_size,
    precision=precision,
    optimizer=optimizer,
    seq_length=seq_length,
    median_step_time=metrics_from_file.median_step_time,
    e2e_time=metrics_from_file.e2e_step_time,
    number_of_steps=number_of_steps,
    mfu=avg_mfu,
    tokens_per_second=metrics_from_file.avg_tokens_per_sec,
    writer_path=local_dir,
    run_success=True,
    run_type=run_type,
    run_release_status=run_release_status,
    other_metrics_in_json="",
    nccl_driver_nickname=None,
    env_variables=env_vars,
    framework_config_in_json=framework_config,
    xla_flags=xla_flags,
    topology=topology,
    dataset=dataset,
    num_of_superblock=None,
    update_person_ldap=getpass.getuser(),
    comment="",
    is_test=False,
  )


