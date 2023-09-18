"""Profiler tests for TPUs."""
import glob
import json
import os
import unittest

from tensorboard_plugin_profile.convert import raw_to_tool_data


class TpuJAXTest(unittest.TestCase):

  """Test for profile collected with JAX."""

  def _get_session_snapshot(self):
    """Gets a session snapshot of current session. assume only one session."""
    profile_plugin_root ="/tmp/tensorboard/plugins/profile"
    # The session exists under a director whose name is time-dependent.
    profile_session_glob = os.path.join(profile_plugin_root, '*', '*.xplane.pb')
    return glob.glob(profile_session_glob)

  def test_xplane_is_present(self):
    files = self._get_session_snapshot()
    self.assertLen(files, 1)

  def test_tools_are_in_list(self):
    xspace_filenames = self._get_session_snapshot()
    result = raw_to_tool_data.xspace_to_tool_names(xspace_filenames)
    result.sort()
    expected = [
        'trace_viewer@^',
        'overview_page^',
        'input_pipeline_analyzer^',
        'tensorflow_stats^',
        'memory_profile^',
        'pod_viewer^',
        'tf_data_bottleneck_analysis^',
        'op_profile^',
        'memory_viewer^',
        'graph_viewer^',
    ]
    expected.sort()
    self.assertListEqual(expected, result)

  def test_overview_page(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(xspace_filenames,
                                                     'overview_page^', {})
    result = json.loads(result)
    run_environment = result[2]
    self.assertEqual(run_environment['p']['host_count'], '1')
    self.assertRegex(run_environment['p']['device_type'], 'TPU.*')

  def test_op_profile(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(
        xspace_filenames, 'op_profile^', {}
    )
    result = json.loads(result)
    self.assertIn('byCategory', result)
    self.assertIn('metrics', result['byCategory'])
    overall_metrics = result['byCategory']['metrics']
    self.assertIn('flops', overall_metrics)
    self.assertIn('bandwidthUtils', overall_metrics)
    self.assertGreater(overall_metrics['flops'], 0)
    contains_value = False
    for m in overall_metrics['bandwidthUtils']:
      if m > 0:
        contains_value = True
    self.assertTrue(contains_value)

  def test_device_trace_contains_threads(self):
    xspace_filenames = self._get_session_snapshot()
    result, _ = raw_to_tool_data.xspace_to_tool_data(
        xspace_filenames, 'trace_viewer^', {}
    )
    result = json.loads(result)
    thread_names = []
    for event in result['traceEvents']:
      if 'name' in event and event['name'] == 'thread_name':
        thread_names.append((event['args']['name']))
    self.assertContainsSubset(
        [
            'TensorFlow Name Scope',
            'TensorFlow Ops',
            'XLA Modules',
            'XLA Ops',
            'XLA TraceMe',
            'Steps',
        ],
        thread_names,
    )


if __name__ == '__main__':
  unittest.main()
