
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
from layers import gpt3
import common_types

import jax.numpy as jnp

import pyconfig
import pytest


Mesh = jax.sharding.Mesh


class GPT3(unittest.TestCase):
  '''numerical tests for GPT3'''
  def setUp(self):
    super().setUp()
    pyconfig.initialize([sys.argv[0], 'configs/base.yml'], per_device_batch_size = 1.0, run_name='test',
                         enable_checkpointing=False, model_name='gpt3-dummy')

    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(1234)

  @pytest.mark.tpu
  def test_numerical_outputs(self):

    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    model = models.Transformer(config = self.cfg, mesh = mesh)

    example_batch = {
        'inputs': jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        'inputs_position': jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        'inputs_segmentation': jnp.array([[1, 1, 1, 1, 1]], dtype=jnp.int32),
        'targets': jnp.array([[20, 21, 22, 23,  1]], dtype=jnp.int32),
        'targets_position': jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        'targets_segmentation': jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }

    model_vars = model.init(
        {'params': self.rng, 'aqt': self.rng},
        example_batch['inputs'],
        example_batch['inputs_position'],
        enable_dropout=False
    )

    def _replace_initialization(key, value):
      keystr = jax.tree_util.keystr(key)

      # replace zero initializer with normal one to ensure strong test cases
      # vars include LayerNorm scale, LayerNorm bias, and DenseGeneral bias
      if "scale" in keystr or "bias" in keystr:
        value = jax.random.normal(self.rng, value.shape, dtype=value.dtype)
      return value

    model_vars = jax.tree_util.tree_map_with_path(_replace_initialization, model_vars)

    # ground truth values are calculated from paxml after loading above model_vars
    per_example_xent_truth = jnp.array([[29.789932, 27.322535, 59.460423, 57.470047, 33.668594]], dtype=jnp.float32)
    padding_mask = example_batch['targets_segmentation'] != 0
    logits, _ = model.apply(model_vars,
                         example_batch['inputs'],
                         example_batch['inputs_position'],
                         decoder_segment_ids=example_batch['inputs_segmentation'],
                         padding_mask=padding_mask,
                         enable_dropout=self.cfg.enable_dropout,
                         rngs={'dropout': self.rng, 'aqt': self.rng}, mutable='intermediates')

    one_hot_targets = jax.nn.one_hot(['targets'], self.cfg.vocab_size)
    per_example_xent = -jnp.sum(jax.nn.log_softmax(logits) * one_hot_targets, axis=-1, dtype=jnp.float32)

    self.assertTrue(
        jax.numpy.allclose(
            per_example_xent, per_example_xent_truth, rtol=1e-06, atol=1e-06
        )
    )

    layer_output_truth = jnp.array(
        [[[  8.036469  ,   2.2698436 ,   1.9434949 ,  -0.27334738,
           -4.430069  ,   2.2529886 ,   3.2230558 ,  -2.4278908 ,
           -3.0491018 ,   8.154987  ,   2.574966  , -11.159233  ,
            0.43259466,   1.5124562 ,  -2.0205994 ,  -2.8799374 ],
         [  7.4823494 ,   1.8260989 ,   1.2726617 ,   1.3042107 ,
           -3.698185  ,  -1.9114767 ,   4.9564285 ,  -4.11675   ,
            0.0938518 ,   6.1295643 ,   4.4330087 , -11.308109  ,
            0.15514785,   4.2591534 ,  -1.7135942 ,  -0.34174103],
         [  8.640661  ,   3.1477747 ,   2.9887805 ,  -1.2705741 ,
           -2.0668025 ,   1.4862754 ,   4.6173515 ,  -0.65943384,
           -1.0033573 ,  11.178499  ,   7.15163   ,  -7.998401  ,
            0.11884853,   4.3759336 ,  -2.7728863 ,  -1.3837149 ],
         [  7.329855  ,   2.7009866 ,   2.966628  ,   0.03845453,
           -1.8673432 ,  -0.49191236,   3.1982498 ,  -2.855019  ,
           -0.05889666,   8.059224  ,   4.7625556 ,  -9.31919   ,
            1.640122  ,   2.3713841 ,  -1.0217882 ,  -3.5475233 ],
         [  3.8139625 ,   1.4492183 ,   0.04837036,  -3.9460216 ,
           -3.88204   ,  -0.7129477 ,   2.6025453 ,  -3.2159119 ,
            0.01289868,   3.099125  ,   5.8068113 ,  -6.834159  ,
            2.1959217 ,   1.549537  ,  -1.3732374 ,  -3.5248797 ]]],
        dtype=jnp.float32,
    )

    model_bind = model.bind(model_vars, rngs={'dropout': self.rng, 'aqt': self.rng})
    embed_output = model_bind.shared_embedding(example_batch['inputs']) + model_bind.position_embedding(example_batch['inputs_position'])
    decoder_layer = gpt3.GPT3DecoderLayer(self.cfg, mesh)
    decoder_layer_vars = jax.tree_map(lambda x: jnp.squeeze(x, self.cfg.param_scan_axis), model_vars['params']['decoder']['decoder'])
    layer_output, _ = decoder_layer.apply({'params': decoder_layer_vars},
                                    embed_output, 
                                    example_batch['inputs_segmentation'],
                                    example_batch['inputs_position'],
                                    padding_mask,
                                    deterministic= not self.cfg.enable_dropout,
                                    model_mode=common_types.MODEL_MODE_TRAIN,
                                    rngs={'dropout': self.rng, 'aqt': self.rng},
                                    )
    self.assertTrue(
        jax.numpy.allclose(
            layer_output, layer_output_truth, rtol=1e-06, atol=1e-06
        )
    )












