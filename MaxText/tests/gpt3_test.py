
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

""" Tests for GPT3 """
import sys
import jax
import unittest
import max_utils
from jax.sharding import Mesh
from layers import models
from layers import embeddings
from layers import quantizations

import jax.numpy as jnp

import pyconfig
import pytest


Mesh = jax.sharding.Mesh
Embed = embeddings.Embed


def init_random_model_vars(model, rng, example_batch):
  """initialze random model vars."""
  model_vars = model.init(
      {'params': rng, 'aqt': rng},
      example_batch['inputs'],
      example_batch['inputs_position'],
      enable_dropout=False,
  )
  def _replace_initialization(key, value):
    keystr = jax.tree_util.keystr(key)
    # replace zero initializer to ensure strong test cases
    #   including Gpt3LayerNorm scale, Gpt3LayerNorm bias, and DenseGeneral bias
    if "scale" in keystr or "bias" in keystr:
      value = jax.nn.initializers.normal(1.0)(rng, value.shape, dtype=value.dtype)
    return value

  model_vars = jax.tree_util.tree_map_with_path(_replace_initialization, model_vars)
  return model_vars


class GPT3(unittest.TestCase):
  """numerical tests for GPT3."""
  def setUp(self):
    super().setUp()
    pyconfig.initialize(
      [sys.argv[0], 'configs/base.yml'],
      run_name='test',
      enable_checkpointing=False,
      model_name='gpt3-52k',
      dtype='float32',
    )

    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(1234)

    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    self.model = models.Transformer(config = self.cfg, mesh = mesh, quant = quant)
    self.example_batch = {
        'inputs': jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        'inputs_position': jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        'inputs_segmentation': jnp.array([[1, 1, 1, 1, 1]], dtype=jnp.int32),
        'targets': jnp.array([[12, 13, 14, 15,  1]], dtype=jnp.int32),
        'targets_position': jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        'targets_segmentation': jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }
    self.model_vars = init_random_model_vars(self.model, self.rng, self.example_batch)

  @pytest.mark.tpu
  def test_logits_numerically(self):
    # ground truth values are calculated from paxml after loading above model_vars
    # note we expect all xents are the same except the padding one since:
    #    paxml applies padding in mlp layer
    #    while maxtext implementaiton applies padding in attention mask instead
    # the two implementation are equivalent in valid non-padding tokens
    per_example_xent_truth = jnp.array([[31.976467, 25.806253, 17.311134, 45.362663, 0.]], dtype=jnp.float32)
    logits, _ = self.model.apply(self.model_vars,
                         self.example_batch['inputs'],
                         self.example_batch['inputs_position'],
                         decoder_segment_ids=self.example_batch['inputs_segmentation'],
                         enable_dropout=self.cfg.enable_dropout,
                         rngs={'dropout': self.rng, 'aqt': self.rng}, mutable='intermediates')

    one_hot_targets = jax.nn.one_hot(self.example_batch['targets'], self.cfg.vocab_size)
    per_example_xent = -jnp.sum(jax.nn.log_softmax(logits) * one_hot_targets, axis=-1, dtype=jnp.float32)
    # Mask out paddings at the end of each example.
    per_example_xent = per_example_xent * (self.example_batch['targets_segmentation'] != 0)

    self.assertTrue(
        jax.numpy.allclose(
            per_example_xent, per_example_xent_truth, rtol=1e-03, atol=1e-03
        )
    )
