import unittest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta
import queue # Use the actual queue for testing queue interactions

# Assume the new classes are in a file named sft_core.py
# If they are in the main script, you might need to adjust imports
# or ensure the script can be imported as a module.
from sft_trainer import (
  StopTraining,
  DataLoader,
  EvalMetrics,
  MetricsRecorder,
  Profiler,
  TrainingContext, # For type hinting and mock creation
  # LoopContext, # If needed for more complex Profiler state mocking
)

# Define a small epsilon for float comparisons if not directly importable
# This should match the EPS from MaxText.train
EPS = 1e-8

class TestDataLoader(unittest.TestCase):
  def setUp(self):
    self.mock_config = MagicMock()
    self.mock_recorder = MagicMock()
    self.mock_data_iterator = MagicMock()

  def test_initialization(self):
    self.mock_config.reuse_example_batch = False
    loader = DataLoader(self.mock_config, self.mock_data_iterator, self.mock_recorder)
    self.assertFalse(loader.reuse_example_batch)
    self.assertEqual(loader.data_iterator, self.mock_data_iterator)
    self.assertEqual(loader.recorder, self.mock_recorder)
    self.assertIsNone(loader.last_batch)

  def test_try_load_next_batch_success(self):
    self.mock_config.reuse_example_batch = False
    expected_batch = {"data": "sample1"}
    self.mock_data_iterator.__next__.return_value = expected_batch
    
    loader = DataLoader(self.mock_config, self.mock_data_iterator, self.mock_recorder)
    batch = loader.try_load_next_batch()
    
    self.assertEqual(batch, expected_batch)
    self.assertEqual(loader.last_batch, expected_batch)
    self.mock_data_iterator.__next__.assert_called_once()

  def test_try_load_next_batch_reuse_true(self):
    self.mock_config.reuse_example_batch = True
    expected_batch = {"data": "sample_reused"}
    self.mock_data_iterator.__next__.return_value = expected_batch
    
    loader = DataLoader(self.mock_config, self.mock_data_iterator, self.mock_recorder)
    
    batch1 = loader.try_load_next_batch()
    self.assertEqual(batch1, expected_batch)
    self.assertEqual(loader.last_batch, expected_batch)
    self.mock_data_iterator.__next__.assert_called_once() # Called first time

    batch2 = loader.try_load_next_batch() # Should reuse
    self.assertEqual(batch2, expected_batch)
    self.mock_data_iterator.__next__.assert_called_once() # Still called only once

  def test_try_load_next_batch_reuse_false(self):
    self.mock_config.reuse_example_batch = False
    batch_1 = {"data": "sample1"}
    batch_2 = {"data": "sample2"}
    self.mock_data_iterator.__next__.side_effect = [batch_1, batch_2]
    
    loader = DataLoader(self.mock_config, self.mock_data_iterator, self.mock_recorder)
    
    res_batch_1 = loader.try_load_next_batch()
    self.assertEqual(res_batch_1, batch_1)
    self.assertEqual(loader.last_batch, batch_1)
    
    res_batch_2 = loader.try_load_next_batch()
    self.assertEqual(res_batch_2, batch_2)
    self.assertEqual(loader.last_batch, batch_2)
    
    self.assertEqual(self.mock_data_iterator.__next__.call_count, 2)

  @patch('sft_core.max_logging.log') # Assuming max_logging is in sft_core or globally accessible
  def test_try_load_next_batch_exhausted(self, mock_max_logging_log):
    self.mock_config.reuse_example_batch = False
    self.mock_data_iterator.__next__.side_effect = StopIteration("Iterator exhausted")
    
    loader = DataLoader(self.mock_config, self.mock_data_iterator, self.mock_recorder)
    
    with self.assertRaises(StopTraining):
      loader.try_load_next_batch()
    
    self.assertIsNone(loader.last_batch)
    mock_max_logging_log.assert_called_once()
    self.assertIn("load_next_batch failed", mock_max_logging_log.call_args[0][0])


class TestEvalMetrics(unittest.TestCase):
  def setUp(self):
    self.eval_metrics = EvalMetrics()

  def test_initialization(self):
    self.assertEqual(self.eval_metrics.all_eval_metrics, [])

  def test_bool_empty(self):
    self.assertFalse(bool(self.eval_metrics))

  def test_bool_not_empty(self):
    self.eval_metrics.append({"metric": 1})
    self.assertTrue(bool(self.eval_metrics))

  def test_append(self):
    metric1 = {"scalar": {"evaluation/total_loss": 10.0, "evaluation/total_weights": 2.0, "evaluation/moe_lb_loss": 0.1}}
    metric2 = {"scalar": {"evaluation/total_loss": 12.0, "evaluation/total_weights": 3.0, "evaluation/moe_lb_loss": 0.2}}
    self.eval_metrics.append(metric1)
    self.eval_metrics.append(metric2)
    self.assertEqual(self.eval_metrics.all_eval_metrics, [metric1, metric2])

  def test_aggregate_empty(self):
    # If all_eval_metrics is empty, aggregate will divide by EPS.
    # This is fine as per the code, len(self.all_eval_metrics) will be 0.
    aggregated, eval_loss, total_weights = self.eval_metrics.aggregate()
    
    self.assertAlmostEqual(aggregated["scalar"]["eval/total_loss"], 0.0)
    self.assertAlmostEqual(aggregated["scalar"]["eval/total_weights"], 0.0)
    self.assertAlmostEqual(aggregated["scalar"]["eval/moe_lb_loss"], 0.0)
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_loss"], 0.0 / EPS) # 0.0 / EPS
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_moe_lb_loss"], 0.0 / EPS) # 0.0 / EPS
    self.assertAlmostEqual(eval_loss, 0.0 / EPS)
    self.assertAlmostEqual(total_weights, 0.0)


  def test_aggregate_single_metric(self):
    metric = {"scalar": {"evaluation/total_loss": 10.0, "evaluation/total_weights": 2.0, "evaluation/moe_lb_loss": 0.1}}
    self.eval_metrics.append(metric)
    
    aggregated, eval_loss, total_weights = self.eval_metrics.aggregate()
    
    self.assertAlmostEqual(aggregated["scalar"]["eval/total_loss"], 10.0)
    self.assertAlmostEqual(aggregated["scalar"]["eval/total_weights"], 2.0)
    self.assertAlmostEqual(aggregated["scalar"]["eval/moe_lb_loss"], 0.1)
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_loss"], 10.0 / (2.0 + EPS))
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_moe_lb_loss"], 0.1 / (1.0 + EPS))
    self.assertAlmostEqual(eval_loss, 10.0 / (2.0 + EPS))
    self.assertAlmostEqual(total_weights, 2.0)

  def test_aggregate_multiple_metrics(self):
    metric1 = {"scalar": {"evaluation/total_loss": 10.0, "evaluation/total_weights": 2.0, "evaluation/moe_lb_loss": 0.1}}
    metric2 = {"scalar": {"evaluation/total_loss": 12.0, "evaluation/total_weights": 3.0, "evaluation/moe_lb_loss": 0.2}}
    metric3 = {"scalar": {"evaluation/total_loss": 8.0, "evaluation/total_weights": 1.0, "evaluation/moe_lb_loss": 0.05}}
    self.eval_metrics.append(metric1)
    self.eval_metrics.append(metric2)
    self.eval_metrics.append(metric3)

    expected_total_loss = 10.0 + 12.0 + 8.0 # 30.0
    expected_total_weights = 2.0 + 3.0 + 1.0 # 6.0
    expected_moe_lb_loss = 0.1 + 0.2 + 0.05 # 0.35
    eval_step_count = 3

    aggregated, eval_loss, total_weights_val = self.eval_metrics.aggregate()

    self.assertAlmostEqual(aggregated["scalar"]["eval/total_loss"], expected_total_loss)
    self.assertAlmostEqual(aggregated["scalar"]["eval/total_weights"], expected_total_weights)
    self.assertAlmostEqual(aggregated["scalar"]["eval/moe_lb_loss"], expected_moe_lb_loss)
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_loss"], expected_total_loss / (expected_total_weights + EPS))
    self.assertAlmostEqual(aggregated["scalar"]["eval/avg_moe_lb_loss"], expected_moe_lb_loss / (eval_step_count + EPS))
    self.assertAlmostEqual(eval_loss, expected_total_loss / (expected_total_weights + EPS))
    self.assertAlmostEqual(total_weights_val, expected_total_weights)

  def test_aggregate_zero_weights(self):
    metric = {"scalar": {"evaluation/total_loss": 10.0, "evaluation/total_weights": 0.0, "evaluation/moe_lb_loss": 0.1}}
    self.eval_metrics.append(metric)
    aggregated, eval_loss, total_weights = self.eval_metrics.aggregate()
    self.assertAlmostEqual(eval_loss, 10.0 / EPS) # Division by EPS


@patch('sft_core.maxtext_utils.calculate_tflops_training_per_device', return_value=(100.0, 0, 0))
@patch('sft_core.maxtext_utils.calculate_tokens_training_per_device', return_value=1024)
@patch('sft_core.GCPWorkloadMonitor')
@patch('sft_core.MetricLogger')
@patch('sft_core.record_scalar_metrics') # Mock the global function
class TestMetricsRecorder(unittest.TestCase):
  def setUp(self):
    self.mock_config = MagicMock()
    self.mock_config.gcs_metrics = False
    self.mock_config.report_heartbeat_metric_for_gcp_monitoring = False
    self.mock_config.report_performance_metric_for_gcp_monitoring = False
    self.mock_config.run_name = "test_run"
    self.mock_config.heartbeat_reporting_interval_in_seconds = 60

    self.mock_train_ctx = MagicMock(spec=TrainingContext)
    self.mock_train_ctx.writer = MagicMock()
    self.mock_train_ctx.learning_rate_schedule = MagicMock(return_value=0.001) # LR schedule callable

  def test_initialization(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    
    mock_calc_tflops.assert_called_once_with(self.mock_config)
    mock_calc_tokens.assert_called_once_with(self.mock_config)
    MockMetricLogger.assert_called_once_with(self.mock_train_ctx.writer, self.mock_config)
    
    self.assertEqual(recorder.per_device_tflops, 100.0)
    self.assertEqual(recorder.per_device_tokens, 1024)
    self.assertIsNone(recorder.running_gcs_metrics) # gcs_metrics is False
    self.assertIsNone(recorder.performance_metric_queue)
    self.assertIsNotNone(recorder.last_step_completion)

  def test_get_performance_metric_queue_none(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    self.mock_config.report_heartbeat_metric_for_gcp_monitoring = False
    self.mock_config.report_performance_metric_for_gcp_monitoring = False
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    self.assertIsNone(recorder.performance_metric_queue)
    MockGCPMonitor.assert_not_called()


  def test_get_performance_metric_queue_heartbeat_only(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    self.mock_config.report_heartbeat_metric_for_gcp_monitoring = True
    self.mock_config.report_performance_metric_for_gcp_monitoring = False
    
    mock_gcp_instance = MockGCPMonitor.return_value
    
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    
    MockGCPMonitor.assert_called_once_with(self.mock_config.run_name)
    mock_gcp_instance.start_heartbeat_reporting_thread.assert_called_once_with(self.mock_config.heartbeat_reporting_interval_in_seconds)
    mock_gcp_instance.start_performance_reporting_thread.assert_not_called()
    self.assertIsNone(recorder.performance_metric_queue)

  def test_get_performance_metric_queue_performance_only(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    self.mock_config.report_heartbeat_metric_for_gcp_monitoring = False
    self.mock_config.report_performance_metric_for_gcp_monitoring = True
    
    mock_gcp_instance = MockGCPMonitor.return_value
    
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    
    MockGCPMonitor.assert_called_once_with(self.mock_config.run_name)
    mock_gcp_instance.start_heartbeat_reporting_thread.assert_not_called()
    mock_gcp_instance.start_performance_reporting_thread.assert_called_once()
    self.assertIsInstance(recorder.performance_metric_queue, queue.Queue)

  def test_get_performance_metric_queue_both(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    self.mock_config.report_heartbeat_metric_for_gcp_monitoring = True
    self.mock_config.report_performance_metric_for_gcp_monitoring = True
    
    mock_gcp_instance = MockGCPMonitor.return_value
    
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    
    MockGCPMonitor.assert_called_once_with(self.mock_config.run_name)
    mock_gcp_instance.start_heartbeat_reporting_thread.assert_called_once()
    mock_gcp_instance.start_performance_reporting_thread.assert_called_once()
    self.assertIsInstance(recorder.performance_metric_queue, queue.Queue)

  @patch('sft_core.datetime') # Mock datetime inside MetricsRecorder
  def test_record_train_metrics(self, mock_datetime, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    # Setup
    self.mock_config.report_performance_metric_for_gcp_monitoring = True # To test queue
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx) # Re-initialize to get queue

    # Mock datetime.now()
    now_time = datetime(2023, 1, 1, 12, 0, 0)
    recorder.last_step_completion = now_time - timedelta(seconds=5) # Previous step was 5s ago
    mock_datetime.now.return_value = now_time
    
    metrics_data = {"loss": 0.5}
    step = 100
    expected_lr = self.mock_train_ctx.learning_rate_schedule.return_value

    # Action
    recorder.record_train_metrics(metrics_data, step)

    # Assertions
    self.mock_train_ctx.learning_rate_schedule.assert_called_once_with(step)
    expected_step_time_delta = timedelta(seconds=5)
    mock_record_scalar.assert_called_once_with(
      metrics_data, 
      expected_step_time_delta, 
      recorder.per_device_tflops, 
      expected_lr, 
      recorder.per_device_tokens
    )
    
    self.assertEqual(recorder.performance_metric_queue.get_nowait(), 5.0) # 5 seconds
    recorder.metric_logger.write_metrics.assert_called_once_with(
      recorder.running_gcs_metrics, metrics_data, step
    )
    self.assertEqual(recorder.last_step_completion, now_time)


  def test_record_eval_metrics(self, mock_record_scalar, MockMetricLogger, MockGCPMonitor, mock_calc_tokens, mock_calc_tflops):
    recorder = MetricsRecorder(self.mock_config, self.mock_train_ctx)
    aggregated_metrics_data = {"eval_loss": 0.8}
    step = 100

    recorder.record_eval_metrics(aggregated_metrics_data, step)

    recorder.metric_logger.write_metrics.assert_called_once_with(
      recorder.running_gcs_metrics, aggregated_metrics_data, step, is_training=False
    )


@patch('sft_core.profiler.Profiler') # Mock the profiler.Profiler from MaxText
class TestProfilerSFT(unittest.TestCase): # Renamed to avoid conflict if sft_core.Profiler is also named Profiler
  def setUp(self):
    self.mock_config = MagicMock()
    self.mock_config.profiler = "tensorboard" # Non-empty to enable profiler
    self.mock_config.steps = 1000
    self.mock_config.profile_periodically_period = 0 # Default, no periodic

    self.mock_state = MagicMock()
    self.start_step = 0
    
    # Mock the external Profiler instance
    self.mock_maxtext_profiler_instance = MagicMock()
    self.mock_maxtext_profiler_instance.start_initial_profile_step = 10
    self.mock_maxtext_profiler_instance.finished_initial_profile_step = 20
    self.mock_maxtext_profiler_instance.should_activate_periodic_profile.return_value = False
    self.mock_maxtext_profiler_instance.should_deactivate_periodic_profile.return_value = False
    

  def test_initialization(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    MockMaxTextProfiler.assert_called_once_with(self.mock_config, offset_step=self.start_step)
    self.assertEqual(profiler_sft.prof, self.mock_maxtext_profiler_instance)

  def test_initialization_profiling_past_steps_error(self, MockMaxTextProfiler):
    self.mock_maxtext_profiler_instance.start_initial_profile_step = 1001 # Past config.steps
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    
    with self.assertRaises(ValueError) as context:
      Profiler(self.mock_config, self.mock_state, self.start_step)
    self.assertIn("Profiling requested but initial profiling step set past training final step", str(context.exception))

  def test_start_train_step_initial_profile_activate(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    step_to_profile = self.mock_maxtext_profiler_instance.start_initial_profile_step # 10
    profiler_sft._start_train_step(step_to_profile)
    self.mock_maxtext_profiler_instance.activate.assert_called_once_with(
      blocking_object=self.mock_state, optional_postfix=""
    )

  def test_start_train_step_periodic_profile_activate(self, MockMaxTextProfiler):
    self.mock_config.profile_periodically_period = 50 # Enable periodic
    self.mock_maxtext_profiler_instance.should_activate_periodic_profile.return_value = True
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    step_to_profile = 150 # Assume this step triggers periodic
    profiler_sft._start_train_step(step_to_profile)
    self.mock_maxtext_profiler_instance.activate.assert_called_once_with(
      blocking_object=self.mock_state, optional_postfix=f"step_{step_to_profile}"
    )
    self.mock_maxtext_profiler_instance.should_activate_periodic_profile.assert_called_once_with(step_to_profile)

  def test_start_train_step_no_profile(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    profiler_sft._start_train_step(5) # Not start_initial_profile_step, not periodic
    self.mock_maxtext_profiler_instance.activate.assert_not_called()

  def test_end_train_step_initial_profile_deactivate(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    step_to_finish_profile = self.mock_maxtext_profiler_instance.finished_initial_profile_step # 20
    profiler_sft._end_train_step(step_to_finish_profile)
    self.mock_maxtext_profiler_instance.deactivate.assert_called_once_with(blocking_object=self.mock_state)

  def test_end_train_step_periodic_profile_deactivate(self, MockMaxTextProfiler):
    self.mock_maxtext_profiler_instance.should_deactivate_periodic_profile.return_value = True
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    step_to_deactivate = 200 # Assume this step triggers periodic deactivate
    profiler_sft._end_train_step(step_to_deactivate)
    self.mock_maxtext_profiler_instance.deactivate.assert_called_once_with(blocking_object=self.mock_state)
    self.mock_maxtext_profiler_instance.should_deactivate_periodic_profile.assert_called_once_with(step_to_deactivate)

  def test_end_train_step_no_profile(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    profiler_sft._end_train_step(25) # Not finished_initial_profile_step, not periodic
    self.mock_maxtext_profiler_instance.deactivate.assert_not_called()

  def test_train_step_context_manager(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    profiler_sft._start_train_step = MagicMock()
    profiler_sft._end_train_step = MagicMock()
    
    current_step = 50
    with profiler_sft.train_step(current_step):
      # Simulate work inside context
      pass 
      
    profiler_sft._start_train_step.assert_called_once_with(current_step)
    profiler_sft._end_train_step.assert_called_once_with(current_step)

  def test_train_loop_context_manager(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    profiler_sft._end_train_loop = MagicMock()
    
    with profiler_sft.train_loop():
      # Simulate work inside context
      pass
      
    profiler_sft._end_train_loop.assert_called_once()

  def test_end_train_loop_deactivates_profiler(self, MockMaxTextProfiler):
    MockMaxTextProfiler.return_value = self.mock_maxtext_profiler_instance
    profiler_sft = Profiler(self.mock_config, self.mock_state, self.start_step)
    
    profiler_sft._end_train_loop()
    self.mock_maxtext_profiler_instance.deactivate.assert_called_once_with() # No blocking object here by design


if __name__ == '__main__':
  # This allows running the tests from the command line
  # You would need to ensure sft_core.py is in your PYTHONPATH
  # or in the same directory.
  # Example: python -m unittest test_sft_refactor.py
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
