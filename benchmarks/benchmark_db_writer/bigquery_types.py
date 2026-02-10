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
This module defines enumerations for BigQuery data types (e.g., `STRING`,
`INT64`) and field modes (e.g., `NULLABLE`, `REQUIRED`).

It also defines a primary mapping, `TypeMapping`, which translates these
BigQuery types into their corresponding standard Python types (like `str`, `int`,
`datetime.datetime`). Custom types (`TimeStamp`, `Geography`) are included
for specific BQ types not perfectly represented by Python built-ins.
Copied & Modified from https://github.com/AI-Hypercomputer/aotc/blob/main/src/aotc/
benchmark_db_writer/src/benchmark_db_writer/bigquery_types.py
"""
import datetime
import decimal
import enum
from typing import Dict, NewType, Type


class BigQueryFieldModes(str, enum.Enum):
  """
  Enums for BigQueryFieldModes
  """

  NULLABLE = "NULLABLE"
  REQUIRED = "REQUIRED"
  REPEATED = "REPEATED"


class BigQueryTypes(str, enum.Enum):
  """
  Enums for BigQueryTypes
  """

  STRING = "STRING"
  BYTES = "BYTES"
  INTEGER = "INT64"
  INT64 = "INT64"
  FLOAT64 = "FLOAT64"
  FLOAT = "FLOAT64"
  NUMERIC = "NUMERIC"
  BOOL = "BOOL"
  BOOLEAN = "BOOL"
  STRUCT = "STRUCT"
  RECORD = "STRUCT"
  TIMESTAMP = "TIMESTAMP"
  DATE = "DATE"
  TIME = "TIME"
  DATETIME = "DATETIME"
  GEOGRAPHY = "GEOGRAPHY"
  JSON = "JSON"


Geography = NewType("Geography", str)


class TimeStamp(datetime.datetime):
  pass


TypeMapping: Dict[BigQueryTypes, Type] = {
    BigQueryTypes.STRING: str,
    BigQueryTypes.BYTES: bytes,
    BigQueryTypes.INT64: int,
    BigQueryTypes.FLOAT64: float,
    BigQueryTypes.NUMERIC: decimal.Decimal,
    BigQueryTypes.BOOL: bool,
    BigQueryTypes.TIMESTAMP: TimeStamp,
    BigQueryTypes.DATE: datetime.date,
    BigQueryTypes.TIME: datetime.time,
    BigQueryTypes.DATETIME: datetime.datetime,
    BigQueryTypes.GEOGRAPHY: Geography,
    BigQueryTypes.JSON: dict,
}
