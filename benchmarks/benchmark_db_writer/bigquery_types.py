import datetime
import decimal
import enum
from typing import Dict, NewType, Type


class BigQueryFieldModes(str, enum.Enum):
  NULLABLE = "NULLABLE"
  REQUIRED = "REQUIRED"
  REPEATED = "REPEATED"


class BigQueryTypes(str, enum.Enum):
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
