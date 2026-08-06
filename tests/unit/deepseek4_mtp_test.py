"""Unit tests for DeepSeek V4 MTP execution logic."""

import unittest
import jax
from maxtext.configs import pyconfig
from maxtext.layers import multi_token_prediction
from maxtext.models import deepseek4
from jax.sharding import Mesh
import os
from flax import nnx


class DeepSeek4MTPTest(unittest.TestCase):

  def setUp(self):
    os.environ["LIBTPU_INIT_ARGS"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

  def test_mtp_initialization(self):
    config = pyconfig.initialize(
        ["", "src/maxtext/configs/base.yml", "base_config=deepseek4-tiny", "skip_jax_distributed_system=True"]
    )
    mesh = Mesh(jax.devices(), ("data",))

    layer = multi_token_prediction.MultiTokenPredictionLayer(
        config=config,
        mesh=mesh,
        layer_number=1,
        transformer_layer_module=deepseek4.DeepSeek4LayerToLinen,
        rngs=nnx.Rngs(0),
    )

    self.assertIsInstance(layer.transformer_layer, deepseek4.DeepSeek4LayerToLinen)
    self.assertEqual(layer.transformer_layer.kwargs.get("compress_ratio", 0), 0)
    self.assertEqual(layer.transformer_layer.kwargs.get("is_hash_routing", False), False)

    # Test that we pass the unabridged config unmodified, so it keeps its HC logic
    self.assertEqual(layer.transformer_layer.kwargs["config"].mhc_expansion_rate, config.mhc_expansion_rate)


if __name__ == "__main__":
  unittest.main()
