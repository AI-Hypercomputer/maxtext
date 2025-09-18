"""TODO: Update model info in the main function & run the script."""

import logging
import os

from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import model_info_schema

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_model_config(
    project,
    dataset,
    table,
    dataclass_type,
    model_id,
    name,
    variant,
    parameter_size_in_billions,
    update_person_ldap=os.getenv("USER", "mrv2"),
    description="",
    details="",
):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )

  model_info = writer.query(where={"model_id": model_id})
  if model_info:
    raise ValueError("Model id %s is already present in the %s table" % (model_id, table))

  # Check if there is already a model info based on name,
  # variant and parameter size
  model_info = writer.query(
      where={
          "name": name,
          "variant": variant,
          "parameter_size_in_billions": parameter_size_in_billions,
      }
  )
  if model_info:
    raise ValueError(
        "Model with name %s, variant %s and "
        "parameter size %s is already present in the %s "
        "table" % (name, variant, parameter_size_in_billions, table)
    )

  model_data = model_info_schema.ModelInfo(
      model_id=model_id,
      name=name,
      variant=variant,
      parameter_size_in_billions=parameter_size_in_billions,
      update_person_ldap=update_person_ldap,
      description=description,
      details=details,
  )

  logging.info("Writing Data %s to %s table.", model_data, table)
  writer.write([model_data])


if __name__ == "__main__":

  table_configs = [
      {
          "project": "ml-workload-benchmarks",
          "dataset": "benchmark_dataset_v2",
          "table": "model_info",
      },
      {
          "project": "supercomputer-testing",
          "dataset": "mantaray_v2",
          "table": "model_info",
      },
  ]

  # Update it on every run
  model_id = "mistral-7b"
  name = "Mistral"
  variant = "7B"
  parameter_size_in_billions = 7
  description = "https://huggingface.co/mistralai/Mistral-7B-v0.3"

  for table_config in table_configs:
    write_model_config(
        project=table_config["project"],
        dataset=table_config["dataset"],
        table=table_config["table"],
        model_id=model_id,
        dataclass_type=model_info_schema.ModelInfo,
        name=name,
        variant=variant,
        parameter_size_in_billions=parameter_size_in_billions,
        description=description,
    )
