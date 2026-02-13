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
Convert a python dataclass into a BigQuery schema definition.
Copied & Modified from https://github.com/AI-Hypercomputer/aotc/blob/main/
src/aotc/benchmark_db_writer/src/benchmark_db_writer/dataclass_converter_utils.py
"""

import dataclasses
import logging
from typing import Any, List, Optional, Type, Union, get_type_hints

from benchmarks.benchmark_db_writer import bigquery_types
from google.cloud.bigquery import SchemaField
import typing_extensions

logger = logging.getLogger(__name__)

_BASIC_TYPES_TO_NAME = {primitive_type: bq_type for bq_type, primitive_type in bigquery_types.TypeMapping.items()}
_NoneType = type(None)


def parse_inner_type_of_list(list_type: Any) -> Type:
  return typing_extensions.get_args(list_type)[0]


def parse_inner_type_of_optional(optional_type: Any) -> Type:
  args = typing_extensions.get_args(optional_type)
  if not (len(args) == 2 and any(arg is _NoneType for arg in args)):
    raise TypeError(f"Unsupported type: {optional_type}.")

  return next(arg for arg in args if arg is not _NoneType)


def _parse_field_description(field: dataclasses.Field) -> Optional[str]:
  if "description" in field.metadata:
    return field.metadata["description"]
  return None


def _parse_fields(field_type: Type) -> List[SchemaField]:
  """Recursive call for nested dataclasses."""

  if dataclasses.is_dataclass(field_type):
    return dataclass_to_schema(field_type)
  return []


def _parse_list(field: dataclasses.Field) -> SchemaField:
  field_type = parse_inner_type_of_list(field.type)
  return SchemaField(
      name=field.name,
      field_type=_python_type_to_big_query_type(field_type),
      mode=bigquery_types.BigQueryFieldModes.REPEATED,
      description=_parse_field_description(field),
      fields=_parse_fields(field_type),
  )


def _python_type_to_big_query_type(
    field_type: Any,
) -> bigquery_types.BigQueryTypes:
  """
  Args:
      field_type: The Python type (e.g., `str`, `int`, a dataclass).

  Returns:
      The corresponding `bigquery_types.BigQueryTypes` enum value.

  Raises:
      TypeError: If the Python type is not supported or mapped.
  """
  if dataclasses.is_dataclass(field_type):
    return bigquery_types.BigQueryTypes.STRUCT

  bq_type = _BASIC_TYPES_TO_NAME.get(field_type)
  if bq_type:
    return bq_type

  raise TypeError(f"Unsupported type: {field_type}")


def _parse_optional(field: dataclasses.Field) -> SchemaField:
  field_type = parse_inner_type_of_optional(field.type)
  return SchemaField(
      name=field.name,
      field_type=_python_type_to_big_query_type(field_type),
      mode=bigquery_types.BigQueryFieldModes.NULLABLE,
      description=_parse_field_description(field),
      fields=_parse_fields(field_type),
  )


def _field_to_schema(field: dataclasses.Field) -> SchemaField:
  """
  Args:
      field: The `dataclasses.Field` to convert.

  Returns:
      A corresponding `SchemaField` object.

  Raises:
      TypeError: If the field's type is complex and unsupported.
  """
  field_type = _BASIC_TYPES_TO_NAME.get(field.type)
  if field_type:
    return SchemaField(
        name=field.name,
        field_type=field_type,
        description=_parse_field_description(field),
        mode=bigquery_types.BigQueryFieldModes.REQUIRED,
    )

  if dataclasses.is_dataclass(field.type):
    return SchemaField(
        name=field.name,
        field_type=bigquery_types.BigQueryTypes.STRUCT,
        mode=bigquery_types.BigQueryFieldModes.REQUIRED,
        description=_parse_field_description(field),
        fields=_parse_fields(field.type),
    )

  # typing.Optional is the same as typing.Union[SomeType, NoneType]
  if typing_extensions.get_origin(field.type) is Union:
    return _parse_optional(field)

  if typing_extensions.get_origin(field.type) is list:
    return _parse_list(field)

  raise TypeError(f"Unsupported type: {field.type}.")


def dataclass_to_schema(dataclass: Type, localns: Optional[dict] = None) -> List[SchemaField]:
  """Transform a dataclass into a list of SchemaField.

  If you want to transform a dataclass that is not defined in the
  global scope you need to pass your locals.

  def my_func():
    @dataclass
    class Example1:
      a: int

    @dataclass
    class Example2:
      b: Example1

    dataclass_to_schema(Example2, localns=locals())
  """
  if not dataclasses.is_dataclass(dataclass):
    raise TypeError("Not a dataclass.")

  type_hints = get_type_hints(dataclass, localns=localns)
  dataclass_fields = dataclasses.fields(dataclass)

  for field in dataclass_fields:
    field.type = type_hints[field.name]
  return [_field_to_schema(field) for field in dataclass_fields]
