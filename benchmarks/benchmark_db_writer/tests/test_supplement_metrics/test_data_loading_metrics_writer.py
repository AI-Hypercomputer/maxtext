"""Tests for data_loading_metrics_writer.py."""

from unittest import mock
from benchmarks.benchmark_db_writer.schema.supplemental_metrics import data_loading_metrics_schema
from benchmarks.benchmark_db_writer.supplemental_metrics_writer import data_loading_metrics_writer
import pytest


@mock.patch("benchmark_db_writer.bq_writer_utils.validate_id")
def test_write_data_loading_metrics_success(mock_validate_id):
  """Test write_data_loading_metrics valid case."""

  mock_validate_id.return_value = True
  try:
    data_loading_metrics_writer.write_data_loading_metrics(
        project="supercomputer-testing",
        dataset="mantaray_v2",
        table="data_loading_metrics",
        dataclass_type=data_loading_metrics_schema.DataLoadingMetricsInfo,
        run_id="run_id",
        data_loading_tokens_per_sec_p50=50.0,
        data_loading_tokens_per_sec_p90=90.0,
        data_loading_tokens_per_sec_p99=99.0,
        data_loading_tokens_per_sec_p100=100.0,
        accelerator_blocked_time=55,
        accelerator_blocked_percent=12.2,
        additional_metrics='{"other_metrics":1}',
        is_test=True,
    )
  except Exception as e:
    pytest.fail(f"data_loading_metrics_writer() raised unexpected error: {str(e)}")


@mock.patch("benchmark_db_writer.bq_writer_utils.validate_id")
def test_write_data_loading_metrics_invalid_run_id(mock_validate_id):
  """Test write_data_loading_metrics invalid run_id case."""

  mock_validate_id.return_value = False
  with pytest.raises(Exception) as err_info:
    data_loading_metrics_writer.write_data_loading_metrics(
        project="supercomputer-testing",
        dataset="mantaray_v2",
        table="data_loading_metrics",
        dataclass_type=data_loading_metrics_schema.DataLoadingMetricsInfo,
        run_id="run_id",
        data_loading_tokens_per_sec_p50=50.0,
        data_loading_tokens_per_sec_p90=90.0,
        data_loading_tokens_per_sec_p99=99.0,
        data_loading_tokens_per_sec_p100=100.0,
        accelerator_blocked_time=55,
        accelerator_blocked_percent=12.2,
        additional_metrics='{"other_metrics":1}',
        is_test=True,
    )
  assert str(err_info.value) == "Could not upload data in data_loading_metrics table."
