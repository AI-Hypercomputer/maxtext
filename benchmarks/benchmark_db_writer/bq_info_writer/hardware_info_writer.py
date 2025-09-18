"""TODO: Update hardware info in the main function & run the script."""

import logging
import os

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    hardware_info_schema,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_hardware_config(
    project,
    dataset,
    table,
    dataclass_type,
    hardware_id,
    gcp_accelerator_name,
    chip_name,
    bf_16_tflops,
    memory,
    hardware_type,
    provider_name,
    chips_per_node=None,
    update_person_ldap=os.getenv("USER", "mrv2"),
    description="",
    other="",
    host_memory=None,
    host_vcpus=None,
):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )

  hardware_info = writer.query(where={"hardware_id": hardware_id})
  if hardware_info:
    raise ValueError("Hardware id %s is already present in the %s table" % (hardware_id, table))

  hardware_data = hardware_info_schema.HardwareInfo(
      hardware_id=hardware_id,
      gcp_accelerator_name=gcp_accelerator_name,
      chip_name=chip_name,
      bf_16_tflops=bf_16_tflops,
      memory=memory,
      chips_per_node=chips_per_node,
      hardware_type=hardware_type,
      provider_name=provider_name,
      update_person_ldap=update_person_ldap,
      description=description,
      other=other,
      host_memory=host_memory,
      host_vcpus=host_vcpus,
  )

  logging.info("Writing Data %s to %s table.", hardware_data, table)
  writer.write([hardware_data])


if __name__ == "__main__":

  table_configs = [
      {
          "project": "ml-workload-benchmarks",
          "dataset": "benchmark_dataset_v2",
          "table": "hardware_info",
      },
      {
          "project": "supercomputer-testing",
          "dataset": "mantaray_v2",
          "table": "hardware_info",
      },
  ]

  # Update it on every run
  hardware_id = "a4"
  gcp_accelerator_name = "A4"
  chip_name = "B200"
  bf_16_tflops = 2237
  memory = 180
  chips_per_node = 8
  hardware_type = "GPU"
  provider_name = "Nvidia"
  description = ""

  for table_config in table_configs:
    write_hardware_config(
        project=table_config["project"],
        dataset=table_config["dataset"],
        table=table_config["table"],
        dataclass_type=hardware_info_schema.HardwareInfo,
        hardware_id=hardware_id,
        gcp_accelerator_name=gcp_accelerator_name,
        chip_name=chip_name,
        bf_16_tflops=bf_16_tflops,
        memory=memory,
        chips_per_node=chips_per_node,
        description=description,
        hardware_type=hardware_type,
        provider_name=provider_name,
    )
