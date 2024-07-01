"""
Copyright 2024 Google LLC
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


import unittest
import pyconfig

class PyconfigTest(unittest.TestCase):
  """Tests for pyconfig.py"""

  def test_basic_override(self):
    raw_keys = {
      'megablox': None,
      'foo': ['bar', 'baz']
    }
    model_keys = {
      'foo': ['x', 'y']
    }

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name='config')

    self.assertEqual(raw_keys,  {
      'megablox': None,
      'foo': ['x', 'y']
    })

  def test_logical_axis_override(self):
    raw_keys = {
      'megablox': None,
      'foo': ['bar', 'baz'],
      'logical_axis_rules': [
        ['activation', ['data', 'fsdp']],
        ['norm', 'tensor']
      ]
    }
    model_keys = {
      'logical_axis_rules': [
        ['activation', ['data', 'fsdp_transpose']],
        ['norm', 'fsdp']
      ]
    }

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name='config')

    self.assertEqual(raw_keys, {
      'megablox': None,
      'foo': ['bar', 'baz'],
      'logical_axis_rules': [
        ('activation', ['data', 'fsdp_transpose']),
        ('norm', 'fsdp')
      ]
    })

  def test_logical_axis_partial_override(self):
    raw_keys = {
      'megablox': None,
      'foo': ['bar', 'baz'],
      'logical_axis_rules': [
        ['activation', ['data', 'fsdp']],
        ['norm', 'tensor']
      ]
    }
    model_keys = {
      'logical_axis_rules': [
        ['norm', 'fsdp']
      ]
    }

    pyconfig.validate_and_update_keys(raw_keys, model_keys, config_name='config')

    self.assertEqual(raw_keys, {
      'megablox': None,
      'foo': ['bar', 'baz'],
      'logical_axis_rules': [
        ('activation', ['data', 'fsdp']),
        ('norm', 'fsdp')
      ]
    })
