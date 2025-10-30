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
Provides a utility class for transforming data.

This module contains the `RowTransformer`, a generic class that uses `dacite`
to convert `google.cloud.bigquery.table.Row` objects into specific
Python dataclass instances and vice-versa.
Copied & Modified from https://github.com/AI-Hypercomputer/aotc/blob/main/src/
aotc/benchmark_db_writer/src/benchmark_db_writer/row_transformer_utils.py
"""
import dataclasses
from typing import Generic, Type, TypeVar

import dacite
from google.cloud.bigquery.table import Row

T = TypeVar("T")  # pylint: disable=invalid-name


class RowTransformer(Generic[T]):
  """Serialized / deserialize rows."""

  def __init__(self, schema: Type[T]):
    self._schema: Type[T] = schema

  def bq_row_to_dataclass_instance(self, bq_row: Row) -> T:
    """Create a dataclass instance from a row returned by the bq library."""

    row_dict = dict(bq_row.items())

    return dacite.from_dict(self._schema, row_dict, config=dacite.Config(check_types=False))

  @staticmethod
  def dataclass_instance_to_bq_row(instance: T) -> dict:
    """Convert a dataclass instance into a dictionary, which can be inserted into bq."""
    return dataclasses.asdict(instance)
