"""Unit tests for DeepSeek V4 MTP execution logic."""

import unittest
import os
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.layers import multi_token_prediction
from maxtext.models import deepseek4
from maxtext.common.common_types import MODEL_MODE_TRAIN


class DeepSeek4MTPTest(unittest.TestCase):
  """Unit tests for DeepSeek4 MTP layer and block execution."""

  def setUp(self):
    os.environ["LIBTPU_INIT_ARGS"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

  def test_mtp_initialization(self):
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/base.yml",
            "model_name=deepseek4-284b",
            "base_num_decoder_layers=5",
            "num_experts=4",
            "num_experts_per_tok=2",
            "compress_ratios=[0, 0, 4, 128, 4]",
            "mtp_num_layers=1",
            "override_model_config=True",
            "skip_jax_distributed_system=True",
        ]
    )
    mesh = Mesh(jax.devices(), ("data",))

    layer = multi_token_prediction.MultiTokenPredictionLayer(
        config=config,
        mesh=mesh,
        layer_number=1,
        transformer_layer_module=deepseek4.DeepSeek4DecoderLayer,
        rngs=nnx.Rngs(0),
    )

    self.assertIsInstance(layer.transformer_layer, deepseek4.DeepSeek4DecoderLayer)
    self.assertEqual(layer.transformer_layer.config.mhc_expansion_rate, config.mhc_expansion_rate)

  def test_mtp_forward_execution(self):
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/base.yml",
            "model_name=deepseek4-284b",
            "base_num_decoder_layers=5",
            "num_experts=4",
            "num_experts_per_tok=2",
            "compress_ratios=[0, 0, 4, 128, 4]",
            "mtp_num_layers=1",
            "override_model_config=True",
            "skip_jax_distributed_system=True",
        ]
    )
    mesh = Mesh(jax.devices(), ("data",))
    rngs = nnx.Rngs(0)

    class MockDecoder(nnx.Module):
      """Mock decoder for test forward execution."""

      def __init__(self, config):
        self.config = config
        self.model_mode = MODEL_MODE_TRAIN

      def _apply_embedding(self, _shared_embedding, input_ids, _position_ids, _deterministic, model_mode):
        batch_size, seq_len = input_ids.shape
        return jnp.zeros((batch_size, seq_len, self.config.base_emb_dim), dtype=self.config.dtype)

      def apply_output_head(
          self, _shared_embedding, hidden_state, _deterministic, model_mode, reduce_mhc=True, decoder_norm=None
      ):
        batch_size, seq_len = hidden_state.shape[:2]
        return jnp.zeros((batch_size, seq_len, self.config.vocab_size), dtype=self.config.dtype)

    decoder_mock = MockDecoder(config)

    mtp_block = multi_token_prediction.MultiTokenPredictionBlock(
        config=config,
        mesh=mesh,
        transformer_layer_module=deepseek4.DeepSeek4DecoderLayer,
        decoder=decoder_mock,
        rngs=rngs,
    )

    batch_size = 2
    seq_len = 8
    main_hidden_state = jnp.zeros(
        (batch_size, seq_len, config.mhc_expansion_rate, config.emb_dim),
        dtype=config.dtype,
    )
    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    with jax.set_mesh(mesh):
      mtp_block(
          shared_embedding=None,
          main_hidden_state=main_hidden_state,
          input_ids=input_ids,
          target_ids=input_ids,
          target_mask=jnp.ones((batch_size, seq_len), dtype=jnp.int32),
          position_ids=jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0),
          decoder_segment_ids=None,
          model_mode=MODEL_MODE_TRAIN,
          deterministic=True,
      )

    self.assertTrue(hasattr(mtp_block, "losses"))
    self.assertEqual(len(mtp_block.losses.value), 1)


if __name__ == "__main__":
  unittest.main()
