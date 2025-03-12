# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Bash helper commands for AOTC artifacts"
import sys
import os
import getpass
from typing import Dict, Any

from command_utils import run_command_with_updates

DEFAULT_LOCAL_DIR = "/tmp/"
# bq_writer_repo_root = get_bq_writer_path(DEFAULT_LOCAL_DIR)

DEFAULT_TUNING_PARAMS_FILE = "/tmp/tuning_params.json"


def get_bq_writer_path() -> str:
  return "/benchmark-automation/benchmark_db_writer/src"


def recover_tuning_params(tuning_params: str) -> Dict[str, Any]:
  items = tuning_params[1:-1].split(",")
  tuning_params_dict = {}
  for item in items:
    key, value = item.split(":", 1)
    key = key.strip()
    # Convert values to appropriate types
    try:
      value = int(value.strip()) 
    except ValueError:
      try:
        value = float(value.strip())
      except ValueError:
        value = value.strip()
        if value.lower() == "true":
          value = True
        elif value.lower() == "false":
          value = False
    tuning_params_dict[key] = value
  return tuning_params_dict


def write_run(
    db_project: str,
    db_dataset: str,
    model_id: str,
    hardware_id: str,
    software_id: str,
    number_of_nodes: int,
    number_of_chips: int,
    container_image_name: str,
    global_batch_size: int,
    precision: str,
    optimizer: str,
    seq_length: int,
    median_step_time: float,
    e2e_time: float,
    number_of_steps: int,
    mfu: float,
    tokens_per_second: float,
    writer_path: str,
    run_success: bool = True,  # True because if mfu is none, writing to db will fail anyway.
    run_type: str = "perf_optimization",
    run_release_status: str = "local",
    other_metrics_in_json: str = "",
    nccl_driver_nickname: str = None,
    env_variables: str = "",
    framework_config_in_json: str = "",
    xla_flags: str = "",
    topology: str = "",
    dataset: str = "",
    num_of_superblock: int = None,
    update_person_ldap: str = getpass.getuser(),
    comment: str = "",
    is_test: bool = False,
):
  """Writes a workload benchmark run manually to the database.

  This function validates the provided IDs and, if valid, constructs a
  WorkloadBenchmarkV2Schema object with the given data and writes it to the
  "run_summary" table in BigQuery.

  Args:
    model_id: The ID of the model used in the run.
    hardware_id: The ID of the hardware used in the run.
    software_id: The ID of the software stack used in the run.
    number_of_nodes: The number of nodes used in the run.
    number_of_chips: The number of chips used in the run.
    container_image_name: The name of the container image used in the run.
    global_batch_size: The global batch size used in the run.
    precision: The precision used in the run (e.g., fp32, bf16).
    optimizer: The optimizer used in the run (e.g., adam, sgd).
    seq_length: The sequence length used in the run.
    median_step_time: The median step time of the run.
    e2e_time: The end-to-end time of the run.
    number_of_steps: The number of steps taken in the run.
    mfu: The MFU (model flops utilization) achieved in the run.
    tokens_per_second: The tokens per second achieved in the run.
    run_type: The type of run (default: "perf_optimization").
    run_release_status: possible values "local" ( code changes are done locally), "prep_release" ( all code code changes are present in the image)
    other_metrics_in_json: A JSON string containing other metrics.
    nccl_driver_nickname: The nickname of the NCCL driver used.
    env_variables: A string containing environment variables.
    framework_config_in_json: A JSON string containing framework configurations.
    xla_flags: A json string containing all the XLA flags.
    topology: The topology of the hardware used in the run. ( valid for TPUs)
    dataset: The dataset used in the run.
    num_of_superblock: The number of superblocks in the hardware. ( valid for GPUs)
    update_person_ldap: The LDAP ID of the person updating the record (default: current user).
    comment: A comment about the run.
    is_test: Whether to use the testing project or the production project.

  Raises:
    ValueError: If any of the IDs are invalid.
  """
  bq_writer_repo_root = get_bq_writer_path()
  sys.path.append(bq_writer_repo_root)

  # pylint: disable=import-outside-toplevel
  import logging
  import uuid
  from typing import Type

  from benchmark_db_writer import bq_writer_utils
  from benchmark_db_writer import dataclass_bigquery_writer
  from benchmark_db_writer.run_summary_writer import sample_run_summary_writer
  from benchmark_db_writer.schema.workload_benchmark_v2 import workload_benchmark_v2_schema
 
  # pylint: enable=import-outside-toplevel
  logging.basicConfig(
      format="%(asctime)s %(levelname)-8s %(message)s",
      level=logging.INFO,
      datefmt="%Y-%m-%d %H:%M:%S",
  )
  logger = logging.getLogger(__name__)

  def get_db_client(
      project: str, dataset: str,
      table: str, dataclass_type: Type, is_test: bool = False
  ) -> dataclass_bigquery_writer.DataclassBigQueryWriter:
    """Creates a BigQuery client object.

    Args:
      table: The name of the BigQuery table.
      dataclass_type: The dataclass type corresponding to the table schema.
      is_test: Whether to use the testing project or the production project.

    Returns:
      A BigQuery client object.
    """

    return bq_writer_utils.create_bq_writer_object(
        project=project,
        dataset=dataset,
        table=table,
        dataclass_type=dataclass_type,
    )

  print(model_id)

  if (
      sample_run_summary_writer.validate_model_id(model_id, is_test)
      and sample_run_summary_writer.validate_hardware_id(hardware_id, is_test)
      and sample_run_summary_writer.validate_software_id(software_id, is_test)
  ):
    summary = workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema(
        run_id=f"run-{uuid.uuid4()}",
        model_id=model_id,
        software_id=software_id,
        hardware_id=hardware_id,
        hardware_num_chips=number_of_chips,
        hardware_num_nodes=number_of_nodes,
        result_success=run_success,
        configs_framework=framework_config_in_json,
        configs_env=env_variables,
        configs_container_version=container_image_name,
        configs_xla_flags=xla_flags,
        configs_dataset=dataset,
        logs_artifact_directory="",
        update_person_ldap=update_person_ldap,
        run_source="automation",
        run_type=run_type,
        run_release_status=run_release_status,
        workload_precision=precision,
        workload_gbs=global_batch_size,
        workload_optimizer=optimizer,
        workload_sequence_length=seq_length,
        metrics_e2e_time=e2e_time,
        metrics_mfu=mfu,
        metrics_step_time=median_step_time,
        metrics_tokens_per_second=tokens_per_second,
        metrics_num_steps=number_of_steps,
        metrics_other=other_metrics_in_json,
        hardware_nccl_driver_nickname=nccl_driver_nickname,
        hardware_topology=topology,
        hardware_num_superblocks=num_of_superblock,
        logs_comments=comment,
    )

    client = get_db_client(
        db_project,
        db_dataset,
        "run_summary",
        workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
        is_test,
    )
    client.write([summary])

  else:
    raise ValueError("Could not upload data in run summary table")
