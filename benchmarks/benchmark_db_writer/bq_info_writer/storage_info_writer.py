"""Update the storage_info table of the benchmark dataset."""

import logging
from typing import Sequence

from absl import app
from absl import flags
from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import storage_info_schema

_STORAGE_PRODUCT = flags.DEFINE_string("storage_product", "", "Type of the storage product.")
_CONFIG = flags.DEFINE_string("config", "", "The configs of the storage system.")
_DESCRIPTION = flags.DEFINE_string("description", "", "The description of the storage system.")
_IS_TEST = flags.DEFINE_bool("is_test", False, "True to write the storage info to the test project.")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_storage_config(
    project,
    dataset,
    table,
    dataclass_type,
    storage_product,
    config,
    description,
):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type,
  )

  storage_data = storage_info_schema.StorageInfo(
      storage_id=None,
      storage_product=storage_product,
      config=config,
      description=description,
      update_person_ldap=None,
      update_timestamp=None,
  )

  logging.info("Writing Data %s to %s table.", storage_data, table)
  writer.write([storage_data])


def main(_: Sequence[str]):
  write_storage_config(
      project=("supercomputer-testing" if _IS_TEST.value else "ml-workload-benchmarks"),
      dataset="mantaray_v2" if _IS_TEST.value else "benchmark_dataset_v2",
      table="storage_info",
      dataclass_type=storage_info_schema.StorageInfo,
      storage_product=_STORAGE_PRODUCT.value,
      config=_CONFIG.value,
      description=_DESCRIPTION.value,
  )


if __name__ == "__main__":
  app.run(main)
