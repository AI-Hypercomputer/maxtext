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

""" Tests for the common MaxText utilities """
import unittest
import jax.numpy as jnp

import maxtext_utils

class TestGradientClipping(unittest.TestCase):
    def test_grad_clipping_with_no_fp8_stats(self):
        raw_grads = {"params": jnp.array([3.0, -4.0]), "wi_0": jnp.array([5.0, -6.0])}
        clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)
        for param_name in raw_grads.keys():
            # The grads should all be clipped and not equal to what they were before
            self.assertFalse(jnp.array_equal(raw_grads[param_name], clipped_grads[param_name]))

    def test_fp8_stats_not_clipped_but_others_are(self):
        raw_grads = {"params": {"wi_0":jnp.array([5.0, -6.0]), "wi_1":jnp.array([7.0, -8.0])}}
        # Create the reference for how the params would be clipped if no fp8 stats were present
        expected_clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

        raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT] = {"amax_history_wi_0": jnp.array([3.0, -4.0]), "scale_wi_0": jnp.array([13.2, -4.4])}
        clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

        # Check all non-fp8 parameters have been clipped in a manner as if the fp8 stats were not present at all
        for param_name in raw_grads['params'].keys():
            self.assertTrue(jnp.array_equal(expected_clipped_grads['params'][param_name], clipped_grads['params'][param_name]))

        # Then check all fp8 parameters were not clipped at all
        for param_name, raw_value in raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT].items():
            self.assertTrue(jnp.array_equal(raw_value, clipped_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT][param_name]))

if __name__ == '__main__':
    unittest.main()
