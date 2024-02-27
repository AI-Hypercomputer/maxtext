#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
import unittest

import common_types

from flax.core import freeze
import jax
import jax.numpy as jnp
import max_utils
import numpy as np
import pytest

import pyconfig

from layers import models
from layers import quantizations

Mesh = jax.sharding.Mesh
MAX_PREFILL_PREDICT_LENGTH = 4 

class TestModel(unittest.TestCase):
  """Test the Whole Model """
  def setUp(self):
    super().setUp()
    pyconfig.initialize([sys.argv[0], 'configs/base.yml'], per_device_batch_size = 1.0, run_name='test',
                         enable_checkpointing=False, base_num_decoder_layers=2, attention="dot_product",
                         max_target_length=16, base_emb_dim=256, base_num_query_heads=2, base_num_kv_heads=2, max_prefill_predict_length=4)
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

  def get_data(self):
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(
        self.rng,
        s,
        0,
        self.cfg.vocab_size
    )

    decoder_segment_ids = jax.numpy.zeros(s) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    decoder_positions = jnp.stack([
          jnp.arange(self.cfg.max_target_length, dtype=jnp.int32)
          for _ in range(self.cfg.global_batch_size_to_train_on)
    ])

    return ids, decoder_segment_ids, decoder_positions
  
  @pytest.mark.tpu
  def test_train_vs_prefill_and_autoregress(self):
    PREFILL_RANGE = MAX_PREFILL_PREDICT_LENGTH

    devices_array = max_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    model = models.Transformer(config = self.cfg, mesh = mesh, quant=quant)

    ids, decoder_segment_ids, decoder_positions = self.get_data()

    transformer_vars = model.init(
        {'params': self.rng, 'aqt': self.rng},
        ids,
        decoder_positions,
        decoder_segment_ids,
        enable_dropout=False
    )

    full_train_logits = model.apply(
      transformer_vars,
      ids,
      decoder_positions,
      decoder_segment_ids,
      enable_dropout=False,
      model_mode = common_types.MODEL_MODE_TRAIN,
      rngs={'aqt': self.rng}
    )

    partial_prefill_logits, partial_cache = model.apply(
      transformer_vars,
      ids[:, :PREFILL_RANGE],
      decoder_positions[:, :PREFILL_RANGE],
      decoder_segment_ids=decoder_segment_ids[:, :PREFILL_RANGE],
      enable_dropout=False,
      model_mode = common_types.MODEL_MODE_PREFILL,
      rngs={'aqt': self.rng},
      mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(
            full_train_logits[:,:PREFILL_RANGE,:], partial_prefill_logits, rtol=1e-01, atol=1e-01, equal_nan=False
        )
    )

    for idx in range(PREFILL_RANGE, self.cfg.max_target_length):
      ids_idx = ids[:, idx:idx+1]
      decoder_positions_idx = decoder_positions[:, idx:idx+1]
      transformer_vars.update(partial_cache)
      ar_logits, partial_cache = model.apply(
        transformer_vars,
        ids_idx,
        decoder_positions_idx,
        enable_dropout=False,
        model_mode = common_types.MODEL_MODE_AUTOREGRESSIVE,
        rngs={'aqt': self.rng},
        mutable=["cache"],
      )

      full_train_logits_idx  = full_train_logits[:,idx:idx+1,:]
      self.assertTrue(
        full_train_logits_idx.shape == ar_logits.shape
      )
      self.assertTrue(
        jax.numpy.allclose(
            full_train_logits_idx, ar_logits, rtol=1e-01, atol=1e-01, equal_nan=False
        )
      )

if __name__ == '__main__':
  unittest.main()