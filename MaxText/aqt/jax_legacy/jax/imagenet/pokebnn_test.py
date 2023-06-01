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
"""Tests for pokebnn.py."""

import hashlib
from absl.testing import absltest
from aqt.jax_legacy.jax.imagenet import hparams_config
from aqt.jax_legacy.jax.imagenet import models
from aqt.jax_legacy.jax.imagenet import train_utils as imagenet_train_utils
from aqt.jax_legacy.jax.imagenet.configs.pokebnn.pokebnn_config import get_config
from aqt.jax_legacy.utils import hparams_utils as os_hparams_utils
import jax
import jax.numpy as jnp


class PokeBNNTest(absltest.TestCase):

  def test_size_stats(self):
    config_dict = get_config().configs[0]
    hparams = os_hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams, config_dict)

    model, _ = imagenet_train_utils.create_model(  #
        jax.random.PRNGKey(0),
        1,
        224,
        jnp.bfloat16,
        hparams=hparams.model_hparams,
        train=False,
        is_teacher=False)  # specify is_teacher=False for student

    # Regression test.
    stats = model.size_stats()
    full_str = str(stats).replace('}', '}\n')
    hashed = hashlib.sha224(full_str.encode()).hexdigest()

    self.assertEqual(
        hashed,
        'fd3bc7e2978b7260a5ed41dfcf3f8f783aa5d614c5e71d12ffb1186f',
        msg='Model size_stats changed. If expected, please update this hash.' +
        full_str)

    # Delete large fields and explicitely check a small core part.
    del stats['conv_stats']
    del stats['residual_params']
    self.assertEqual(
        stats, {
            'conv_ace': 4091469824.0,
            'dense_ace': 131072000,
            'batch_norm_sops': 17912832,
            'bprelu_sops': 9106944,
            'reshape_add_ch_shrink_sops': 5419008,
            'reshape_add_ch_shrink_mul_sops': 1705984,
            'reshape_add_avg_pool_sops': 7902720,
            'reshape_add_block_residual_sops': 5519360,
            'reshape_add_local_residual_sops': 8805888,
            'se_global_pool_add_sops': 8906240,
            'se_global_pool_mul_sops': 20672,
            'se_relu_sops': 2584,
            'se_relu6_sops': 22656,
            'se_mul_sops': 8805888,
            'pokebnn_global_pool_sops': 100352,
            'dense_stats': {
                'dense': {
                    'input_shape': 2048,
                    'input_prec': 8,
                    'weight_shape': (2048, 1000),
                    'weight_prec': 8,
                    'output_shape': 1000,
                    'dense_ace': 131072000
                }
            }
        })


if __name__ == '__main__':
  absltest.main()
