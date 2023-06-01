# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for aqt.jax.train_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax_legacy.jax import train_utils


class UpdateUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      # No updates before the start step of 5
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 0,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 3,
          'should_update': False
      },

      # Updates expected every 10 steps after step 5 and no time else
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 15,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 25,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 13,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 18,
          'should_update': False
      },

      # Test an update frequency of -1, which indicates no bounds update
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 6,
          'should_update': False
      },
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 4,
          'should_update': False
      },

      # Test a start step of -1, which indicates no bounds update at any step.
      {
          'frequency': -1,
          'start_step': -1,
          'current_step': 3,
          'should_update': False
      },
      {
          'frequency': 5,
          'start_step': -1,
          'current_step': 12,
          'should_update': False
      })
  def test_should_update_bounds(self, frequency, start_step, current_step,
                                should_update):
    self.assertEqual(
        train_utils.should_update_bounds(
            activation_bound_update_freq=frequency,
            activation_bound_start_step=start_step,
            step=current_step), should_update)

  @parameterized.parameters(
      # No updates before the start step of 5
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 0,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 3,
          'should_update': False
      },

      # Updates expected every 10 steps after step 5 and no time else
      {
          'frequency': 8,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 15,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 25,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 13,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 18,
          'should_update': False
      },

      # Test frequency of 0, which indicates no update except at start step
      {
          'frequency': 0,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': 0,
          'start_step': 5,
          'current_step': 10,
          'should_update': False
      },
      {
          'frequency': 0,
          'start_step': 5,
          'current_step': 4,
          'should_update': False
      },

      # Test a start step of -1, which indicates no mask update at any step.
      {
          'frequency': 0,
          'start_step': -1,
          'current_step': 3,
          'should_update': False
      },
      {
          'frequency': 5,
          'start_step': -1,
          'current_step': 5,
          'should_update': False
      })
  def test_should_update_sparsity_mask(self, frequency, start_step,
                                       current_step, should_update):
    self.assertEqual(
        train_utils.update_sparsity_mask(
            sparsity_start_step=start_step,
            sparsity_update_freq=frequency,
            step=current_step), should_update)

  @parameterized.parameters(
      {
          'step': 100,
          'sparsity_start_step': 10,
          'sparsity_update_freq': 10,
          'num_update_sparsity': 9
      },
      {
          'step': 10,
          'sparsity_start_step': 10,
          'sparsity_update_freq': 10,
          'num_update_sparsity': 0
      },
      {
          'step': 100,
          'sparsity_start_step': 10,
          'sparsity_update_freq': 0,
          'num_update_sparsity': 0
      },)
  def test_num_update_sparsity(self, step, sparsity_start_step,
                               sparsity_update_freq, num_update_sparsity):
    dc = train_utils.get_dynamic_context_for_step(
        activation_bound_update_freq=1,
        activation_bound_start_step=1,
        step=step,
        collect_acts_stats=False,
        prefer_int8_to_int32_dot=False,
        sparsity_start_step=sparsity_start_step,
        sparsity_update_freq=sparsity_update_freq)
    self.assertEqual(dc.num_update_sparsity, num_update_sparsity)


if __name__ == '__main__':
  absltest.main()
