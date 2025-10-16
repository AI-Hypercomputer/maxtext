"""TODO: Update software info in the main function & run the script."""

import logging
import os

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    software_info_schema,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_software_config(
    project,
    dataset,
    table,
    dataclass_type,
    software_id,
    ml_framework,
    os,
    compiler,
    training_framework,
    update_person_ldap=os.getenv("USER", "mrv2"),
    description="",
):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )

  software_info = writer.query(where={"software_id": software_id})
  if software_info:
    raise ValueError("Software id %s is already present in the %s table" % (software_id, table))

  software_data = software_info_schema.SoftwareInfo(
      software_id=software_id,
      ml_framework=ml_framework,
      os=os,
      compiler=compiler,
      training_framework=training_framework,
      update_person_ldap=update_person_ldap,
      description=description,
  )

  logging.info("Writing Data %s to %s table.", software_data, table)
  writer.write([software_data])


if __name__ == "__main__":

  project = "ml-workload-benchmarks"
  dataset = "benchmark_dataset_v2"
  table = "software_info"

  # Update it on every run
  software_id = "jax_maxtext"
  ml_framework = "JAX"
  os = "cos"
  compiler = "XLA"
  training_framework = "MaxText"
  description = "https://github.com/AI-Hypercomputer/maxtext"

  write_software_config(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=software_info_schema.SoftwareInfo,
      software_id=software_id,
      ml_framework=ml_framework,
      os=os,
      compiler=compiler,
      training_framework=training_framework,
      description=description,
  )
