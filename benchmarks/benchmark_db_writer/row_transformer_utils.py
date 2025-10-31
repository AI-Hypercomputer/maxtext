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
