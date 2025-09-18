"""
Update the network_metrics table of the benchmark dataset.
"""

import logging
from typing import Sequence

from absl import app
from absl import flags
from benchmarks.benchmark_db_writer import bq_writer_utils
from benchmarks.benchmark_db_writer.schema.supplemental_metrics import network_metrics_schema
from benchmarks.benchmark_db_writer.schema.workload_benchmark_v2 import (
    workload_benchmark_v2_schema,
)
from benchmarks.benchmark_db_writer.supplemental_metrics_writer import common_flags

_SERVER_MAX_EGRESS = flags.DEFINE_float(
    "server_max_egress",
    None,
    "The peak value of the storage egress throughput from the server side.",
)
_SERVER_AVG_EGRESS = flags.DEFINE_float(
    "server_avg_egress",
    None,
    "The average value of the storage egress throughput from the server side.",
)
_SERVER_MAX_INGRESS = flags.DEFINE_float(
    "server_max_ingress",
    None,
    "The peak value of the storage ingress throughput from the server side.",
)
_SERVER_AVG_INGRESS = flags.DEFINE_float(
    "server_avg_ingress",
    None,
    "The average value of the storage ingress throughput from the server side.",
)
_SERVER_MAX_QPS = flags.DEFINE_float(
    "server_max_qps",
    None,
    "The peak value of the storage QPS from the server side.",
)
_SERVER_AVG_QPS = flags.DEFINE_float(
    "server_avg_qps",
    None,
    "The average value of the storage QPS throughput from the server side.",
)
_CLIENT_MAX_EGRESS = flags.DEFINE_float(
    "client_max_egress",
    None,
    "The peak value of the storage egress throughput from the client side.",
)
_CLIENT_AVG_EGRESS = flags.DEFINE_float(
    "client_avg_egress",
    None,
    "The average value of the storage egress throughput from the client side.",
)
_CLIENT_MAX_INGRESS = flags.DEFINE_float(
    "client_max_ingress",
    None,
    "The peak value of the storage ingress throughput from the client side.",
)
_CLIENT_AVG_INGRESS = flags.DEFINE_float(
    "client_avg_ingress",
    None,
    "The average value of the storage ingress throughput from the client side.",
)
_CLIENT_MAX_QPS = flags.DEFINE_float(
    "client_max_qps",
    None,
    "The peak value of the storage QPS from the client side.",
)
_CLIENT_AVG_QPS = flags.DEFINE_float(
    "client_avg_qps",
    None,
    "The average value of the storage QPS from the client side.",
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_network_metrics(
    project,
    dataset,
    table,
    dataclass_type,
    run_id,
    server_max_egress,
    server_avg_egress,
    server_max_ingress,
    server_avg_ingress,
    server_max_qps,
    server_avg_qps,
    client_max_egress,
    client_avg_egress,
    client_max_ingress,
    client_avg_ingress,
    client_max_qps,
    client_avg_qps,
    is_test=False,
):

  if bq_writer_utils.validate_id(
      logger,
      run_id,
      "run_summary",
      "run_id",
      workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
      is_test,
  ):

    writer = bq_writer_utils.create_bq_writer_object(
        project=project,
        dataset=dataset,
        table=table,
        dataclass_type=dataclass_type,
    )

    network_metrics_data = network_metrics_schema.NetworkMetricsInfo(
        run_id=run_id,
        server_max_egress=server_max_egress,
        server_avg_egress=server_avg_egress,
        server_max_ingress=server_max_ingress,
        server_avg_ingress=server_avg_ingress,
        server_max_qps=server_max_qps,
        server_avg_qps=server_avg_qps,
        client_max_egress=client_max_egress,
        client_avg_egress=client_avg_egress,
        client_max_ingress=client_max_ingress,
        client_avg_ingress=client_avg_ingress,
        client_max_qps=client_max_qps,
        client_avg_qps=client_avg_qps,
    )

    logging.info("Writing Data %s to %s table.", network_metrics_data, table)
    writer.write([network_metrics_data])

  else:
    raise ValueError("Could not upload data in run summary table")


def main(_: Sequence[str]):
  write_network_metrics(
      project=("supercomputer-testing" if common_flags.IS_TEST.value else "ml-workload-benchmarks"),
      dataset=("mantaray_v2" if common_flags.IS_TEST.value else "benchmark_dataset_v2"),
      table="network_metrics",
      dataclass_type=network_metrics_schema.NetworkMetricsInfo,
      run_id=common_flags.RUN_ID.value,
      server_max_egress=_SERVER_MAX_EGRESS.value,
      server_avg_egress=_SERVER_AVG_EGRESS.value,
      server_max_ingress=_SERVER_MAX_INGRESS.value,
      server_avg_ingress=_SERVER_AVG_INGRESS.value,
      server_max_qps=_SERVER_MAX_QPS.value,
      server_avg_qps=_SERVER_AVG_QPS.value,
      client_max_egress=_CLIENT_MAX_EGRESS.value,
      client_avg_egress=_CLIENT_AVG_EGRESS.value,
      client_max_ingress=_CLIENT_MAX_INGRESS.value,
      client_avg_ingress=_CLIENT_AVG_INGRESS.value,
      client_max_qps=_CLIENT_MAX_QPS.value,
      client_avg_qps=_CLIENT_AVG_QPS.value,
      is_test=common_flags.IS_TEST.value,
  )


if __name__ == "__main__":
  app.run(main)
