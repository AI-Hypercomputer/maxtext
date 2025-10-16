import logging
from typing import Type
from benchmarks.benchmark_db_writer import dataclass_bigquery_writer


def create_bq_writer_object(project, dataset, table, dataclass_type):
  """Creates a BQ writer config and uses it to create BQ writer object."""

  config = dataclass_bigquery_writer.BigqueryWriterConfig(project, dataset, table)

  writer = dataclass_bigquery_writer.DataclassBigQueryWriter(dataclass_type, config)

  return writer


def get_db_client(table: str, dataclass_type: Type, is_test: bool = False) -> create_bq_writer_object:
  """Creates a BigQuery client object.

  Args:
    table: The name of the BigQuery table.
    dataclass_type: The dataclass type corresponding to the table schema.
    is_test: Whether to use the testing project or the production project.

  Returns:
    A BigQuery client object.
  """

  project = "cloud-tpu-multipod-dev" if is_test else "ml-workload-benchmarks"
  dataset = "benchmark_dataset" if is_test else "benchmark_dataset_v2"
  return create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )


def validate_id(
    logger: logging.Logger,
    id_value: str,
    table_name: str,
    id_field: str,
    dataclass_type: Type,
    is_test: bool = False,
) -> bool:
  """Generic function to validate an ID against a BigQuery table.

  Args:
    logger: The logging instance which represents a single logging channel.
    id_value: The ID value to validate.
    table_name: The name of the BigQuery table.
    id_field: The name of the ID field in the table.
    dataclass_type: The dataclass type corresponding to the table schema.
    is_test: Whether to use the testing project or the production project.

  Returns:
    True if the ID is valid, False otherwise.
  """

  client = get_db_client(table_name, dataclass_type, is_test)
  result = client.query(where={id_field: id_value})

  if not result:
    logger.info(
        "%s: %s is not present in the %s table ",
        id_field.capitalize(),
        id_value,
        table_name,
    )
    logger.info(
        "Please add %s specific row in %s table before adding the new rows to" " the target table",
        id_value,
        table_name,
    )
    return False
  return True
