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

"""
Core BigQuery writer class that maps Python dataclasses to BQ tables.

This module provides the primary `DataclassBigQueryWriter`, a generic class
that uses a Python dataclass to define the schema of a BigQuery table. It
handles table creation, schema validation, schema updates, and provides
methods for writing dataclass instances to the table and reading BQ rows
back as dataclass instances.
Copied & Modified from https://github.com/AI-Hypercomputer/aotc/blob/main/
src/aotc/benchmark_db_writer/src/benchmark_db_writer/dataclass_bigquery_writer.py
"""
import copy
import dataclasses
import logging
import pprint
import time
from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar

from benchmarks.benchmark_db_writer import bigquery_types
from benchmarks.benchmark_db_writer import dataclass_converter_utils
from benchmarks.benchmark_db_writer import row_transformer_utils
import google.api_core.exceptions
from google.cloud import bigquery

# The type of the generic dataclass
T = TypeVar("T")

logger = logging.getLogger(__name__)


def _field_type_str(field_type: str):
  """Normalizes the field type to a string.

  Args:
    field_type: the field type to convert to a string.

  Returns:
    The string representation of the field type.
  """
  if isinstance(field_type, bigquery_types.BigQueryTypes):
    return field_type.value
  else:
    return bigquery_types.BigQueryTypes[field_type].value


def _field_to_dict(field: bigquery.schema.SchemaField):
  """A concise dict representation of a SchemaField.

  This is only to compare schemas to check if the schema fields have changed.

  Args:
    field: the schema field to convert to a dict

  Returns:
    A dict representation of the schema field.
  """
  return {
      "field_type": _field_type_str(field.field_type),
      "mode": field.mode,
      "fields": schema_to_dict(field.fields),
  }


def schema_to_dict(schema: Sequence[bigquery.schema.SchemaField]):
  """A concise dict representation of a bigquery schema.

  This is used to compare the current schema against the dataclass generated
  schema.

  Args:
    schema: the schema to convert to a dict.

  Returns:
    A dict representation of the schema field.
  """
  return {field.name: _field_to_dict(field) for field in schema}


def _recursive_struct_param(
    name: str, schema: dict[str, Any], values: Optional[dict[str, Any]] = None
) -> bigquery.StructQueryParameter:
  """Recursively builds a StructQueryParameter from schema and values.

  Args:
      name: The name of the struct parameter.
      schema: The concise schema dict for the struct (from `schema_to_dict`).
      values: An optional dict of values for the struct.

  Returns:
      A `bigquery.StructQueryParameter` object.
  """
  params = []
  # match up schema to values
  for field_name, field_schema in schema.items():
    value = values[field_name] if values else None
    param = _query_param(field_name, field_schema, value)
    assert param
    params.append(param)
  return bigquery.StructQueryParameter(name, *params)


def _query_param(name: str, schema_field: dict[str, Any], value: Any):  # -> bigquery._AbstractQueryParameter:
  """Builds a BigQuery query parameter from schema and a value.

  Handles nested STRUCTs by calling `_recursive_struct_param`.

  Args:
      name: The name of the parameter (used as @name in the query).
      schema_field: The concise schema dict for the field.
      value: The value for the parameter.

  Returns:
      A `bigquery.ScalarQueryParameter` or `bigquery.StructQueryParameter`.
  """
  if schema_field["field_type"] == "STRUCT":
    assert value is None or isinstance(value, dict)
    # recurse the schema even for None/NULL values which we have to propagate
    # all the way through the struct
    return _recursive_struct_param(name, schema=schema_field["fields"], values=value)
  else:
    return bigquery.ScalarQueryParameter(name, schema_field["field_type"], value)


@dataclasses.dataclass
class BigqueryWriterConfig:
  project: str
  dataset: str
  table: str


class DataclassBigQueryWriter(Generic[T]):
  """Uses the `bq-schema` package to write a dataclass to a BigQuery table."""

  def __init__(self, dataclass_type: Type[T], config: BigqueryWriterConfig):
    """Initializes the writer.

    Args:
      dataclass_type: the dataclass type to use as the schema
      project: the GCP project to write to
      dataset: the dataset to write to
      table: the table to write to
    """
    self.client = bigquery.Client(project=config.project)
    self.row_transformer = None
    self.table_id = f"{config.project}.{config.dataset}.{config.table}"
    self.dataclass_type = dataclass_type

    self.input_data_schema = dataclass_converter_utils.dataclass_to_schema(self.dataclass_type)
    # Get or create table
    try:
      self.table = self.client.get_table(self.table_id)
    except google.api_core.exceptions.NotFound:
      logger.warning("Table %s not found, creating it", self.table_id)
      self.client.create_table(self.table_id)
      self.table = self.client.get_table(self.table_id)
      # When creating the table for the first time, always update schema.
      self.update_schema()

    # Check schema of table and input dataclass
    self.check_schema()

  def check_schema(self):
    """Validates the dataclass schema against the live table schema.

    Raises:
        ValueError: If the dataclass schema is incompatible with the
            table schema.
            - If a column exists in the dataclass but not the table.
            - If a REQUIRED column exists in the table but not the dataclass.
    """
    table_schema = schema_to_dict(self.table.schema)
    data_schema = schema_to_dict(self.input_data_schema)

    # Check whether dataclass has any additional column
    for dataclass_column in data_schema.keys():
      if dataclass_column not in table_schema:
        raise ValueError(
            f"Schema of table {self.table_id} is different than input data."
            " Please check both schema and re-run.\n"
            f"Column: {dataclass_column} is absent in table whereas it's "
            "present in dataclass."
        )

    # Check whether big query table has any additional column which are not "nullable"
    for table_column, column_attributes in table_schema.items():
      if table_column not in data_schema and column_attributes["mode"] != bigquery_types.BigQueryFieldModes.NULLABLE:

        raise ValueError(
            f"Schema of table {self.table_id} is different than input data."
            " Please check both schema and re-run.\n"
            f"Column: {table_column} is absent in dataclass whereas it's "
            "present in table & is of Required type."
        )

  def update_schema(self):
    """When new table is created, this function gets called to update the schema."""
    logger.info(
        "DataclassBigQueryWriter: updating schema to %s",
        pprint.pformat(self.input_data_schema),
    )
    old_schema = copy.deepcopy(self.table.schema)
    try:
      self.table.schema = self.input_data_schema
      self.table = self.client.update_table(self.table, ["schema"])
      logger.info("BigQueryResultWriter: waiting for some time for the schema to" " propagate")
      time.sleep(60)
    except google.api_core.exceptions.GoogleAPICallError as e:
      logger.exception("Failed to update bigquery schema with error %s", e)
      self.table.schema = old_schema

  def transform(self, dataclass: T) -> dict:
    return row_transformer_utils.RowTransformer.dataclass_instance_to_bq_row(dataclass)

  def read(self, where: Optional[str] = None) -> tuple[list[T], list[T]]:
    """Reads the bigquery table using `where` as the WHERE clause.

    Args:
      where: used as the `WHERE` expression when querying the database.

    Returns:
      The list of bigquery entries as the dataclass T.
    """
    row_transformer = row_transformer_utils.RowTransformer[T](self.dataclass_type)
    query = "SELECT * FROM " + self.table_id
    if where:
      query += " WHERE " + where
    raw_rows = []
    rows = []
    for bq_row in self.client.query(query=query):
      raw_rows.append(bq_row)
      dataclass = row_transformer.bq_row_to_dataclass_instance(bq_row)
      assert isinstance(dataclass, self.dataclass_type)
      rows.append(dataclass)
    return rows, raw_rows

  def _get_field_schema_dict(self, field_name):
    schema_dict = {"fields": schema_to_dict(self.input_data_schema)}

    field_dir = field_name.split(".")
    for key in field_dir:
      schema_dict = schema_dict["fields"][key]
    return schema_dict

  def _get_query_for_value(self, field_name, value):  # -> Tuple[str, bigquery._AbstractQueryParameter]:
    if dataclasses.is_dataclass(value):
      value = row_transformer_utils.RowTransformer.dataclass_instance_to_bq_row(value)
    # # find schema for `field_name`:
    field_schema = self._get_field_schema_dict(field_name)
    at_name = "_".join(field_name.split("."))
    return f"{field_name} = @{at_name}", _query_param(at_name, field_schema, value)

  def query_column(self, column_name) -> List[Any]:
    """Returns all values of the given column name."""

    query_str = f"SELECT {column_name} FROM {self.table_id}"
    query_result = self.client.query(query=query_str)

    return [row[0] for row in query_result]

  def query(self, where: Optional[dict[str, Any]] = None) -> list[T]:
    """Reads the bigquery table using `where` dict as the WHERE clause.

    Args:
      where: A dict with key value pair using which WHERE clause is constructed.

    Returns:
      The list of bigquery entries as the dataclass T.
    """
    if where is None:
      where = {}
    where_exprs = []
    params = []
    for field_name, value in where.items():
      where_expr, param = self._get_query_for_value(field_name, value)
      params.append(param)
      where_exprs.append(where_expr)
    query_str = f"SELECT * FROM {self.table_id}"
    if where_exprs:
      where_stmt = " AND ".join(where_exprs)
      query_str += f" WHERE {where_stmt}"
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    row_transformer = row_transformer_utils.RowTransformer[T](self.dataclass_type)
    rows = []
    for bq_row in self.client.query(query=query_str, job_config=job_config):
      dataclass = row_transformer.bq_row_to_dataclass_instance(bq_row)
      assert isinstance(dataclass, self.dataclass_type)
      rows.append(dataclass)
    return rows

  def write(self, rows: List[T]):
    """Bulk write to big query.

    Args:
      rows: list of rows (dataclasses) to write to bigquery
    """
    serialized_rows = [self.transform(row) for row in rows]
    try:
      logger.info("Writing to BigQuery: %d rows", len(serialized_rows))
      insert_errors = self.client.insert_rows(table=self.table, rows=serialized_rows)
      if insert_errors:
        logger.error(
            "There were errors while writing to Bigquery:\n%s",
            pprint.pformat(insert_errors),
        )
      else:
        logger.info("Successfully wrote to BigQuery")
    except google.api_core.exceptions.GoogleAPICallError as e:
      logger.exception("Failed to write to BigQuery with error %s", e)
