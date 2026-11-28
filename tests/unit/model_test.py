# Copyright 2023–2025 Google LLC
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
"""Model test."""

import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np

MAX_PREFILL_PREDICT_LENGTH = 4


class TestModel(unittest.TestCase):
  """Test the Whole Model."""

  def setUp(self):
    """Init the test model, call the super call, setup random seed, and init pyconfig."""
    super().setUp()
    self.cfg = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    """Init pyconfig."""
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        **kwargs,
    )
    return config

  def get_data(self):
    """Get data."""
    s = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length)
    ids = jax.random.randint(self.rng, s, 0, self.cfg.vocab_size)

    decoder_segment_ids = jax.numpy.zeros(s) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    decoder_positions = jnp.stack(
        [jnp.arange(self.cfg.max_target_length, dtype=jnp.int32) for _ in range(self.cfg.global_batch_size_to_train_on)]
    )

    return ids, decoder_segment_ids, decoder_positions

  def _test_logits_cast_driver(self, cast_logits_to_fp32, expected_dtype):
    """Helper method to test the dtype of the logits returned by the full model at the end.

    Does not perform any actual flops.
    """
    new_config = self.init_pyconfig(cast_logits_to_fp32=cast_logits_to_fp32, logits_dot_in_fp32=False)
    devices_array = maxtext_utils.create_device_mesh(new_config)
    mesh = Mesh(devices_array, new_config.mesh_axes)
    model = nnx.eval_shape(
        lambda: models.Transformer(
            config=new_config,
            mesh=mesh,
            quant=None,
            model_mode=MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(params=self.rng, aqt=self.rng, dropout=self.rng),
        )
    )

    ids, decoder_segment_ids, decoder_positions = self.get_data()

    logits = nnx.eval_shape(
        lambda m: m(
            ids,
            decoder_positions,
            decoder_segment_ids,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        ),
        model,
    )

    self.assertEqual(logits.dtype, expected_dtype)

  def test_logits_dtype_with_cast_to_fp32(self):
    """Test logits datatype with cast to 32-bit floating point."""
    self._test_logits_cast_driver(cast_logits_to_fp32=True, expected_dtype=jnp.float32)

  def test_logits_dtype_without_cast(self):
    """Test logits datatype without casting."""
    self._test_logits_cast_driver(cast_logits_to_fp32=False, expected_dtype=jnp.bfloat16)

  def test_train_vs_prefill_and_autoregress(self):
    """Test train versus prefill and autoregress."""
    PREFILL_RANGE = MAX_PREFILL_PREDICT_LENGTH

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    train_model = models.Transformer(
        config=self.cfg,
        mesh=mesh,
        quant=quant,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=self.rng, aqt=self.rng, dropout=self.rng),
    )
    prefill_model = models.Transformer(
        config=self.cfg,
        mesh=mesh,
        quant=quant,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=self.rng, aqt=self.rng, dropout=self.rng),
    )
    nnx.update(prefill_model, nnx.state(train_model, nnx.Param))

    ids, decoder_segment_ids, decoder_positions = self.get_data()

    @nnx.jit(static_argnames=("model_mode",))
    def call_model(model, ids_in, pos_in, seg_in, model_mode):
      return model(
          ids_in,
          pos_in,
          decoder_segment_ids=seg_in,
          model_mode=model_mode,
          enable_dropout=False,
      )

    full_train_logits = call_model(train_model, ids, decoder_positions, decoder_segment_ids, MODEL_MODE_TRAIN)
    partial_prefill_logits = call_model(
        prefill_model,
        ids[:, :PREFILL_RANGE],
        decoder_positions[:, :PREFILL_RANGE],
        decoder_segment_ids[:, :PREFILL_RANGE],
        MODEL_MODE_PREFILL,
    )

    np.testing.assert_allclose(
        full_train_logits[:, :PREFILL_RANGE, :],
        partial_prefill_logits,
        rtol=1e-01,
        atol=1e-01,
        equal_nan=False,
    )

    for idx in range(PREFILL_RANGE, min(PREFILL_RANGE + 3, self.cfg.max_target_length)):
      ids_idx = ids[:, idx : idx + 1]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      ar_logits = call_model(prefill_model, ids_idx, decoder_positions_idx, None, MODEL_MODE_AUTOREGRESSIVE)

      full_train_logits_idx = full_train_logits[:, idx : idx + 1, :]
      self.assertTrue(full_train_logits_idx.shape == ar_logits.shape)
      np.testing.assert_allclose(
          full_train_logits_idx,
          ar_logits,
          rtol=1e-01,
          atol=1e-01,
          equal_nan=False,
      )


if __name__ == "__main__":
  unittest.main()
