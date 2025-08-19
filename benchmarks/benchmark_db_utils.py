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

"Bash helper commands for AOTC artifacts"
from tempfile import gettempdir
from typing import Any, Type
import dataclasses
import getpass
import logging
import os
import sys
import uuid

from argparse import Namespace

BQ_WRITER_PATH = "/benchmark-automation/benchmark_db_writer/src"
temp_dir = gettempdir()
DEFAULT_LOCAL_DIR = os.path.join(temp_dir, "")
# bq_writer_repo_root = get_bq_writer_path(DEFAULT_LOCAL_DIR)

DEFAULT_TUNING_PARAMS_FILE = os.path.join(temp_dir, "tuning_params.json")


@dataclasses.dataclass
class Metrics:
  avg_tflops_per_sec: float
  avg_tokens_per_sec: float
  median_step_time: float
  e2e_step_time: float


def recover_tuning_params(tuning_params: str) -> dict[str, Any]:
  """
  Parse tuning params from json str format
  e.g. {"per_device_batch_size": 2, "ici_fsdp_parallelism": 1, ...}
  
  Args:
    tuning_params: Tuning parameters in json str format
  Return type:
    dict[str, Any]: Dictionary mapping tuning param name to its value
  """
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
    options: Namespace,
    metrics: Metrics,
    mfu: float,
    number_of_steps: int,
    number_of_nodes: int,
    number_of_chips: int,
    run_success: bool = True,  # True because if mfu is none, writing to db will fail anyway.
    framework_config_in_json: str = "",
    env_variables: str = "",
    run_release_status: str = "local",
    other_metrics_in_json: str = "",
    comment: str = "",
    nccl_driver_nickname: str = "",
):
  """Writes a workload benchmark run manually to the database.

  This function validates the provided IDs and, if valid, constructs a
  WorkloadBenchmarkV2Schema object with the given data and writes it to the
  "run_summary" table in BigQuery.

  Args:
    options: Namespace of options from argparse. Should contain these attributes:
        model_id: The ID of the model used in the run.
        hardware_id: The ID of the hardware used in the run.
        software_id: The ID of the software stack used in the run.
        container_image_name: The name of the container image used in the run.
        global_batch_size: The global batch size used in the run.
        precision: The precision used in the run (e.g., fp32, bf16).
        optimizer: The optimizer used in the run (e.g., adam, sgd).
        seq_length: The sequence length used in the run.
        run_type: The type of run (default: "perf_optimization").
        xla_flags: A json string containing all the XLA flags.
        topology: The topology of the hardware used in the run. ( valid for TPUs)
        dataset: The dataset used in the run.
        num_of_superblock: The number of superblocks in the hardware. ( valid for GPUs)
        update_person_ldap: The LDAP ID of the person updating the record (default: current user).    
        is_test: Whether to use the testing project or the production project.
    metrics: Metrics object containing:
        median_step_time: The median step time of the run.
        e2e_step_time: The end-to-end time of the run.
        avg_tokens_per_sec: The tokens per second achieved in the run.
    mfu: The MFU (model flops utilization) achieved in the run.
    number_of_steps: The number of steps taken in the run.
    number_of_nodes: The number of nodes used in the run.
    number_of_chips: The number of chips used in the run.
    run_success: Whether the run succeeded or not.
    framework_config_in_json: A JSON string containing framework configurations.
    env_variables: A string containing environment variables.
    run_release_status: "local": code changes are done locally; "prep_release": code changes present in the image
    other_metrics_in_json: A JSON string containing other metrics.
    comment: A comment about the run.
    nccl_driver_nickname: The nickname of the NCCL driver used.

  Raises:
    ValueError: If any of the IDs are invalid.
  """
  bq_writer_repo_root = BQ_WRITER_PATH
  sys.path.append(bq_writer_repo_root)

  # pylint: disable=import-outside-toplevel

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

  print(options.model_id)

  if (
      sample_run_summary_writer.validate_model_id(options.model_id, options.is_test)
      and sample_run_summary_writer.validate_hardware_id(options.hardware_id, options.is_test)
      and sample_run_summary_writer.validate_software_id(options.software_id, options.is_test)
  ):
    summary = workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema(
        run_id=f"run-{uuid.uuid4()}",
        model_id=options.model_id,
        software_id=options.software_id,
        hardware_id=options.hardware_id,
        hardware_num_chips=number_of_chips,
        hardware_num_nodes=number_of_nodes,
        result_success=run_success,
        configs_framework=framework_config_in_json,
        configs_env=env_variables,
        configs_container_version=options.container_image_name,
        configs_xla_flags=options.xla_flags.replace(",", " "),
        configs_dataset=options.dataset,
        logs_artifact_directory="",
        update_person_ldap=getpass.getuser(),
        run_source="automation",
        run_type=options.run_type,
        run_release_status=run_release_status,
        workload_precision=options.precision,
        workload_gbs=int(options.global_batch_size),
        workload_optimizer=options.optimizer,
        workload_sequence_length=int(options.seq_length),
        metrics_e2e_time=metrics.e2e_step_time,
        metrics_mfu=mfu,
        metrics_step_time=metrics.median_step_time,
        metrics_tokens_per_second=metrics.avg_tokens_per_sec,
        metrics_num_steps=number_of_steps,
        metrics_other=other_metrics_in_json,
        hardware_nccl_driver_nickname=nccl_driver_nickname,
        hardware_topology=options.topology,
        hardware_num_superblocks=0,
        logs_comments=comment,
    )

    client = get_db_client(
        options.db_project,
        options.db_dataset,
        "run_summary",
        workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
        options.is_test,
    )
    client.write([summary])

  else:
    raise ValueError("Could not upload data in run summary table")
