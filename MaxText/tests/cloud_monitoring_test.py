"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Tests for Cloud Monitoring API """
import sys
import jax
import unittest

import monitoring_api
import pyconfig

jax.config.update('jax_platform_name', 'cpu')

class CloudMonitoringTests(unittest.TestCase):
  """Test for writing time series step using monitoring_api.py"""
  def test_write_time_series_step(self):
    pyconfig.initialize(sys.argv + ['configs/base.yml'], per_device_batch_size=1, run_name='test', cloud_zone='us-central2-b')
    monitoring_api.create_custom_metric('test_metric', "This is an example metric")
    create_time_series_result = monitoring_api.write_time_series_step('test_metric', True, pyconfig, 1)
    query_time_series_result = monitoring_api.get_time_series_step_data('test_metric')
    self.assertEqual(create_time_series_result, query_time_series_result)


if __name__ == '__main__':
  unittest.main()

