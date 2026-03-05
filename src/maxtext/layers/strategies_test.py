# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for strategies.py."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from unittest import mock
import jax
import jax.numpy as jnp
from maxtext.layers import strategies
from flax import linen as nn


class StrategiesTest(absltest.TestCase):

  def test_default_strategy_scanned(self):
    mock_decoder = mock.Mock()
    mock_decoder.config.scan_layers = True
    mock_decoder.config.num_decoder_layers = 2
    mock_decoder.config.inhomogeneous_layer_cycle_interval = 1
    mock_decoder.config.using_pipeline_parallelism = False
    mock_decoder.mesh = mock.Mock()
    
    # Mock scan_decoder_layers to return a function that returns (y, None)
    mock_scan_fn = mock.Mock(return_value=(jnp.zeros((1, 1)), None))
    mock_decoder.scan_decoder_layers.return_value = mock_scan_fn

    strategy = strategies.DefaultStrategy()
    
    block_layers = [mock.Mock()]
    broadcast_args = (mock.Mock(), mock.Mock(), True, "train")
    
    y = jnp.zeros((1, 1))
    
    result = strategy.apply_layers(
        mock_decoder,
        y,
        block_layers,
        broadcast_args,
        model_mode="train"
    )

    mock_decoder.scan_decoder_layers.assert_called_once()
    self.assertEqual(result, y) # Since mock returns input y

  def test_default_strategy_unscanned(self):
    mock_decoder = mock.Mock()
    mock_decoder.config.scan_layers = False
    mock_decoder.config.num_decoder_layers = 1
    mock_decoder.config.using_pipeline_parallelism = False
    mock_decoder.mesh = mock.Mock()
    mock_decoder.quant = mock.Mock()

    strategy = strategies.DefaultStrategy()
    
    # Mock layer instance
    mock_layer_instance = mock.Mock()
    mock_layer_instance.return_value = (jnp.ones((1, 1)), None) # (output, cache)
    
    # Mock block layer class
    mock_block_layer = mock.Mock(return_value=mock_layer_instance)
    block_layers = [mock_block_layer]
    
    broadcast_args = (mock.Mock(), mock.Mock(), True, "train")
    kv_caches = [mock.Mock()]
    
    y = jnp.zeros((1, 1))
    
    result = strategy.apply_layers(
        mock_decoder,
        y,
        block_layers,
        broadcast_args,
        model_mode="train",
        kv_caches=kv_caches
    )
    
    mock_block_layer.assert_called_once()
    mock_layer_instance.assert_called_once()
    self.assertTrue((result == jnp.ones((1, 1))).all())

if __name__ == '__main__':
  absltest.main()
